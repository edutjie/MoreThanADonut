"""Train a Bundesliga live/dead specialist and test it next to the frozen GRU."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from lightgbm import LGBMClassifier

from ball_recovery_experiment import contest_tracking
from bundesliga_dead_ball import observed_motion
from evaluation import possession_metrics as metrics
from external_possession import (
    DEV_MATCHES, IDSSE_FPS, LAG_SECONDS, SKILLCORNER_FPS, TEST_MATCHES,
    decode_costs, load_idsse, load_skillcorner, mask_ball, tracking_features,
)
from games import GAME, half_video_blocks
from idsse_training import IDSSE_DEV, IDSSE_TEST, IDSSE_TRAIN
from player_track_quality import repair_player_tracks
from possession_sources import load_legacy_skillcorner
from sequence_possession import DECODER_COSTS, restore, sequence_probabilities, visibility_features
from starter.baseline import write_submission


POLICIES = [("reference", 0.0)] + [
    (name, strength) for name in ("uniform", "hard-negative") for strength in (0.15, 0.35, 0.60)
]
# The three halves whose team-specific StatsBomb offsets agreed in the pass-anchor audit.
CLASICO_HALVES = {2841: (303596, {2: -0.85}), 3442: (303470, {1: -0.85, 2: -0.45})}


def observed(positions, weights, pitch, fps):
    ball = positions[:, -1].copy()
    if weights is not None:
        ball[weights < 1] = np.nan
    return observed_motion(ball, pitch, fps)


def train(root):
    xs, ys, hard = [], [], []
    for match in IDSSE_TRAIN:
        for positions, weights, teams, pitch, labels in load_idsse(root, match):
            inside, slow = observed(positions, weights, pitch, IDSSE_FPS)
            selected = np.flatnonzero((inside & (labels != metrics.X))[::5]) * 5
            x, flipped = tracking_features(positions, weights, teams, pitch, IDSSE_FPS, LAG_SECONDS)
            xs.extend((x[selected], flipped[selected]))
            target = labels[selected] == metrics.D
            ys.append(np.tile(target, 2))
            hard.append(np.tile(slow[selected] & ~target, 2))
    x, y, negative = map(np.concatenate, (xs, ys, hard))
    print(f"Training rows with mirrored copies: {len(y):,}; dead {y.sum():,}; "
          f"slow live {negative.sum():,}", flush=True)
    models = {}
    for name, weight in (("uniform", None), ("hard-negative", np.where(negative, 4.0, 1.0))):
        model = LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1, n_jobs=4,
                               deterministic=True, force_col_wise=True, verbosity=-1)
        models[name] = model.fit(x, y, sample_weight=weight)
    return models


def log_odds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blended_emissions(p, x, flipped, inside, specialists, policies):
    base = np.log(np.maximum(p, 1e-6))
    delta = {}
    for name in {name for name, strength in policies if strength}:
        model = specialists[name]
        q = (model.predict_proba(x[inside])[:, 1] +
             model.predict_proba(flipped[inside])[:, 1]) / 2 if inside.any() else np.empty(0)
        delta[name] = (log_odds(q) - log_odds(p[inside, metrics.D])).clip(-3, 3)
    for name, strength in policies:
        emission = base.copy()
        if strength:
            emission[inside, metrics.D] += strength * delta[name]
        yield emission


def predictions(reference, specialists, block, fps, device, policies):
    positions, weights, teams, pitch, _ = block
    if reference.repair_players:
        positions = repair_player_tracks(positions, fps)
    inside, _ = observed(positions, weights, pitch, fps)
    x, flipped = tracking_features(positions, weights, teams, pitch, fps, LAG_SECONDS)
    motion = tracking_features(positions, weights, teams, pitch, fps, LAG_SECONDS,
                               continuous_motion=True) if reference.continuous_motion else (x, flipped)
    visibility = visibility_features(positions[:, -1], fps)
    normal, mirrored = (np.column_stack((features, visibility)) for features in motion)
    p = sequence_probabilities(reference, normal, mirrored, fps, device)
    for emission in blended_emissions(p, x, flipped, inside, specialists, policies):
        yield decode_costs(emission, fps, *DECODER_COSTS)


def score(root, matches, reference, specialists, device, policies, masked=False):
    idsse = isinstance(matches[0], str)
    loader, fps = (load_idsse, IDSSE_FPS) if idsse else (load_skillcorner, SKILLCORNER_FPS)
    games = []
    for match in matches:
        rng = np.random.default_rng(int(match[1:], 36) if idsse else match)
        halves = []
        for positions, weights, teams, pitch, labels in loader(root, match):
            inside, slow = observed(positions, weights, pitch, fps)
            slow_live = inside & slow & np.isin(labels, (metrics.H, metrics.A))
            tracks = mask_ball(positions, fps, rng, mean_gap=12, coverage=0.60) if masked else positions
            runs = metrics.valid_runs(labels)
            rows = []
            for prediction in predictions(reference, specialists, (tracks, weights, teams, pitch, labels),
                                          fps, device, policies):
                pred = metrics.label_codes(prediction)
                cm = metrics.scored_confusion(metrics.confusion(labels, pred))
                in_pitch = metrics.scored_confusion(metrics.confusion(labels[inside], pred[inside]))
                rows.append((metrics.spell_scores(labels, pred, runs)["mean_f1"],
                             metrics.macro_f1(cm), metrics.per_class_prf(cm, metrics.D)[2],
                             metrics.per_class_prf(in_pitch, metrics.D)[2],
                             int((pred[slow_live] == metrics.D).sum()), int(slow_live.sum())))
            halves.append(rows)
        halves = np.asarray(halves)
        games.append(np.column_stack((halves[:, :, :4].mean(axis=0),
                                      halves[:, :, 4:].sum(axis=0))))
    values = np.asarray(games)
    return np.column_stack((values[:, :, :4].mean(axis=0), values[:, :, 4:].sum(axis=0)))


def event_seconds(event):
    h, m, s = event["timestamp"].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def foul_intervals(events, offset, n_frames):
    for idx, event in enumerate(events):
        if event["type"]["name"] != "Foul Won" or event.get("foul_won", {}).get("advantage"):
            continue
        start = event_seconds(event)
        for restart in events[idx + 1:]:
            right = event_seconds(restart)
            kind = restart["type"]["name"]
            if right - start > 90 or kind == "Foul Won":
                break
            if right <= start or kind not in ("Pass", "Shot", "Carry", "Dribble"):
                continue
            if (restart["team"]["id"] == event["team"]["id"] and
                    restart.get(kind.lower(), {}).get("type", {}).get("name") == "Free Kick"):
                left = max(0, round((start + offset + 1) * SKILLCORNER_FPS))
                stop = min(n_frames, round((right + offset - 1) * SKILLCORNER_FPS))
                if left < stop:
                    yield slice(left, stop)
            break


def broadcast_check(args, reference, specialists, policies, device):
    rows = np.zeros((len(policies), 4), dtype=int)
    for match, (event_id, offsets) in CLASICO_HALVES.items():
        events = json.loads((args.statsbomb / f"{event_id}.json").read_text())
        for period, block in enumerate(load_legacy_skillcorner(args.legacy, match), 1):
            if period not in offsets:
                continue
            positions, weights, _, pitch, _ = block
            inside, _ = observed(positions, weights, pitch, SKILLCORNER_FPS)
            select = list(foul_intervals([e for e in events if e["period"] == period],
                                        offsets[period], len(inside)))
            for idx, pred in enumerate(predictions(reference, specialists, block, SKILLCORNER_FPS,
                                                    device, policies)):
                for window in select:
                    total = int(inside[window].sum())
                    if total:
                        found = int((pred[window][inside[window]] == "D").sum())
                        rows[idx] += (found, total, 2 * found >= total, 1)
    return rows.tolist()


def held_out_checks(args, reference, specialists, device, selection):
    policies = [POLICIES[0]]
    for name in specialists:
        indices = [i for i, policy in enumerate(POLICIES) if policy[0] == name]
        policies.append(POLICIES[max(indices, key=lambda i: selection[i])])
    sets = (("A-League", args.skillcorner, sorted(TEST_MATCHES), False),
            ("A-League masked", args.skillcorner, sorted(TEST_MATCHES), True),
            ("Bundesliga", args.idsse, IDSSE_TEST, False),
            ("Bundesliga masked", args.idsse, IDSSE_TEST, True))
    return {
        "held_out_policies": policies,
        "held_out": {name: score(root, matches, reference, specialists, device, policies, masked).tolist()
                     for name, root, matches, masked in sets},
        "clasico_metrics": ["dead_frames_found", "weak_dead_frames", "half_covered_intervals", "weak_intervals"],
        "clasico": broadcast_check(args, reference, specialists, policies, device),
    }


def calibrate(args):
    device = torch.device(args.device)
    reference, _ = restore(args.reference, device)
    specialists = train(args.idsse)
    dev = {
        "A-League": score(args.skillcorner, sorted(DEV_MATCHES), reference, specialists, device, POLICIES),
        "Bundesliga": score(args.idsse, IDSSE_DEV, reference, specialists, device, POLICIES),
    }
    selection = np.mean([values[:, (0, 3)].mean(axis=1) for values in dev.values()], axis=0)
    best = int(selection.argmax())
    enabled = int(selection[1:].argmax()) + 1
    print(f"Frozen selection {POLICIES[best]}; best enabled {POLICIES[enabled]}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump((specialists, POLICIES[best], POLICIES[enabled]), args.output / "specialists.joblib")
    report = {
        "metrics": ["spell_f1", "macro_f1", "dead_f1", "in_pitch_dead_f1", "slow_live_as_dead", "slow_live"],
        "policies": POLICIES, "development": {name: result.tolist() for name, result in dev.items()},
        "selection_scores": selection.tolist(), "selected": POLICIES[best],
    }
    report.update(held_out_checks(args, reference, specialists, device, selection))
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def predict(args):
    device = torch.device(args.device)
    reference, _ = restore(args.reference, device)
    specialists, selected, enabled = joblib.load(args.specialists)
    positions, teams, pitch, fps = contest_tracking(args.data_dir)
    labels = np.full(len(positions), "X")
    for _, start, last in half_video_blocks(GAME):
        half = slice(start, last + 1)
        block = (positions[half], None, teams, pitch, None)
        labels[half], = predictions(reference, specialists, block, fps, device,
                                    [enabled if args.best_enabled else selected])
    write_submission(args.output, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("calibrate")
    for name in ("idsse", "skillcorner", "legacy", "statsbomb"):
        training.add_argument(f"--{name}", type=Path, required=True)
    inference = commands.add_parser("predict")
    inference.add_argument("--data-dir", type=Path)
    inference.add_argument("--specialists", type=Path, required=True)
    inference.add_argument("--best-enabled", action="store_true")
    for command in (training, inference):
        command.add_argument("--device", default="cpu")
        command.add_argument("--reference", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (calibrate if args.command == "calibrate" else predict)(args)
