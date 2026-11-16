"""Check in-pitch restart and whistle-gated stoppage cues with frozen LightGBM."""

import argparse
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.ball_recovery_experiment import DEV_GAMES, contest_tracking
from dead_ball import read_cues
from evaluation import possession_metrics as metrics
from external_possession import (
    IDSSE_FPS, IDSSE_MATCHES, decode_costs, load_idsse, load_model, mask_ball,
    probabilities, tracking_features,
)
from games import GAME, half_video_blocks
from starter.baseline import write_submission
from starter.load_smart_tagging import load as load_tags
from tactical_baseline import fill_short_gaps


POLICIES = {
    "none": (0, 0.0, False),
    "tag-short": (0.24, 0.35, False),
    "tag-long": (0.60, 0.35, False),
    "paired-short": (0.24, 0.35, True),
    "paired-long": (0.60, 0.35, True),
}


def free_kicks(root, match_id):
    path, = root.glob(f"DFL_03_02_events_raw_*-{match_id}.xml")
    for event in ET.parse(path).iter("Event"):
        restart = event.find("FreeKick")
        if restart is not None and restart.get("DecisionTimestamp"):
            yield (datetime.fromisoformat(event.get("EventTime")),
                   datetime.fromisoformat(restart.get("DecisionTimestamp")))


def moving_ball(positions, pitch, fps):
    ball, confidence = fill_short_gaps(positions[:, -1], fps)
    speed = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps]
    tracked = np.r_[False, np.minimum(confidence[1:], confidence[:-1]) >= 0.5]
    inside = ((ball >= 0.5) & (ball <= pitch - 0.5)).all(axis=1)
    return tracked & inside & (speed < 1.5), tracked & (speed > 2.5) & (speed < 35), inside


def restart_cues(motion, windows, whistle, fps, seconds, paired):
    still, moving, inside = motion
    width = max(2, round(seconds * fps))
    n = len(still)
    prior = np.r_[0, np.cumsum(still)]
    after = np.r_[0, np.cumsum(moving)]
    for start, end in windows:
        frames = np.arange(max(width, start), min(n - width, end))
        before_count = prior[frames] - prior[frames - width]
        after_count = after[frames + width] - after[frames]
        quality = np.minimum(before_count, after_count)
        okay = (quality >= 0.67 * width) & inside[frames - 1]
        if not np.any(okay):
            continue
        candidates = frames[okay & (quality == quality[okay].max())]
        restart = int(candidates[np.argmin(np.abs(candidates - (start + end) / 2))])
        begin = max(0, restart - round(3 * fps))
        if paired:
            possible = whistle[(whistle >= max(start, restart - 20 * fps))
                               & (whistle <= restart - fps)]
            if len(possible) != 1:
                continue
            begin = int(possible[0])
        else:
            last_moving = np.flatnonzero(moving[begin:restart - width])
            if len(last_moving):
                begin += last_moving[-1] + 1
        yield begin, restart


def classify(emission, motion, windows, whistle, fps, costs, policy):
    seconds, weight, paired = POLICIES[policy]
    cues = list(restart_cues(motion, windows, whistle, fps, seconds, paired)) if weight else []
    if cues:
        emission = emission.copy()
        add = np.zeros(len(emission), dtype=bool)
        for start, end in cues:
            add[start:end] = True
        emission[add, metrics.D] += weight
    return decode_costs(emission, fps, *costs), cues


def external_match(root, match, checkpoint):
    model, costs, lags = load_model(checkpoint)
    rng = np.random.default_rng(int(match[1:], 36))
    fps = IDSSE_FPS
    events = list(free_kicks(root, match))
    scores, anchors = [], np.zeros((len(POLICIES), 3), dtype=int)
    for x, weights, teams, pitch, labels, clock in load_idsse(root, match, with_clock=True):
        origin = datetime.fromisoformat(clock)
        frames = [(round((restart - origin).total_seconds() * fps),
                   round((decision - origin).total_seconds() * fps)) for restart, decision in events]
        frames = [(r, d) for r, d in frames if 0 <= d < r < len(x) and r - d <= 50 * fps]
        jitter = rng.integers(-8 * fps, 8 * fps + 1, size=len(frames))
        widths = round(17.5 * fps)
        windows = [(r + j - widths, r + j + widths) for (r, _), j in zip(frames, jitter)]
        # Proxy whistles come from external decision times, with dropouts, jitter and false cues.
        whistle = [d + round(rng.normal(0, 0.5) * fps) for _, d in frames if rng.random() < 0.65]
        whistle.extend(rng.integers(0, len(x), size=round(len(x) / fps / 250)))
        whistle = np.sort(whistle)
        masked = mask_ball(x, fps, rng, mean_gap=12, coverage=0.60)
        motion = moving_ball(masked, pitch, fps)
        p = probabilities(model, *tracking_features(masked, weights, teams, pitch, fps, lags))
        emission = np.log(np.maximum(p, 1e-6))
        runs = metrics.valid_runs(labels)
        half_scores = []
        for index, policy in enumerate(POLICIES):
            pred, cues = classify(emission, motion, windows, whistle, fps, costs, policy)
            codes = metrics.label_codes(pred)
            spell = metrics.spell_scores(labels, codes, runs)["mean_f1"]
            confusion = metrics.scored_confusion(metrics.confusion(labels, codes))
            half_scores.append((spell, metrics.per_class_prf(confusion, metrics.D)[2]))
            hits = sum(any(abs(r - expected) <= fps for expected, _ in frames) for _, r in cues)
            anchors[index] += len(frames), len(cues), hits
        scores.append(half_scores)
    return np.mean(scores, axis=0), anchors


def external(root, checkpoint, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {match: pool.submit(external_match, root, match, checkpoint) for match in IDSSE_MATCHES}
        for name, matches in (("development", DEV_GAMES),
                              ("held-out", [m for m in IDSSE_MATCHES if m not in DEV_GAMES])):
            results = [jobs[m].result() for m in matches]
            scores = np.mean([row[0] for row in results], axis=0)
            anchors = np.sum([row[1] for row in results], axis=0)
            print(f"{name}: external matches {matches}")
            for policy, (spell, dead), (windows, cues, hits) in zip(POLICIES, scores, anchors):
                print(f"{policy:12} spellF1={spell:.5f} D-F1={dead:.5f} "
                      f"restart hits={hits}/{cues} from {windows} windows")
            if name == "development":
                winner = max(range(len(POLICIES)), key=lambda i: tuple(scores[i]))
                print(f"Selected: {list(POLICIES)[winner]}", flush=True)


def contest(model_path, data_dir, whistle_path, policy, output):
    model, costs, lags = load_model(model_path)
    x, teams, pitch, fps = contest_tracking(data_dir)
    whistles = read_cues(whistle_path, len(x))[:, 0]
    tags = [tag for tag in load_tags(data_dir=data_dir) if (tag["code"] or "").casefold() == "set pieces"
            and (tag["tags"].get("Type") or "").casefold() == "free kick"]
    labels, count = np.full(len(x), "X"), 0
    for _, start, last in half_video_blocks(GAME):
        half = x[start:last + 1]
        windows = [(ev["start_frame"] - start, ev["end_frame"] - start)
                   for ev in tags if ev["start_frame"] <= last and ev["end_frame"] > start]
        p = probabilities(model, *tracking_features(half, None, teams, pitch, fps, lags))
        emission = np.log(np.maximum(p, 1e-6))
        pred, cues = classify(emission, moving_ball(half, pitch, fps), windows,
                              np.flatnonzero(whistles[start:last + 1]), fps, costs, policy)
        labels[start:last + 1] = pred
        count += len(cues)
    write_submission(output, labels)
    print(f"Applied {count} stoppage intervals across {len(tags)} free-kick tags")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("external")
    check.add_argument("--idsse", type=Path, required=True)
    check.add_argument("--model", type=Path, required=True)
    check.add_argument("--workers", type=int, default=4)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--data-dir", type=Path)
    predict.add_argument("--whistles", type=Path, required=True)
    predict.add_argument("--policy", choices=POLICIES, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "external":
        external(args.idsse, args.model, args.workers)
    else:
        contest(args.model, args.data_dir, args.whistles, args.policy, args.output)
