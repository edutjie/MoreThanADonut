"""Evaluate how to use frozen YOLO ball recoveries with the possession model."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as metrics
from external_possession import (
    IDSSE_FPS, IDSSE_MATCHES, decode_costs, load_idsse, load_model, mask_ball,
    probabilities, tracking_features,
)
from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track, write_submission
from tactical_baseline import read_tracking, write_ball_positions


RECOVERY_WEIGHT = 0.55
DEV_GAMES = ("J03WMX", "J03WOH", "J03WPY", "J03WQQ")
POLICIES = {
    "none": ("joint", 0.0),
    "replace": ("joint", 1.0),
    "joint-25": ("joint", 0.25),
    "joint-50": ("joint", 0.5),
    "side-25": ("side", 0.25),
    "side-50": ("side", 0.5),
    "side-100": ("side", 1.0),
    "boundary-50": ("boundary", 0.5),
}


def fusion(original, repaired, outside, policy):
    """Fuse probabilities while optionally preserving the original stoppage signal."""
    kind, weight = POLICIES[policy]
    if kind == "joint":
        return (1 - weight) * original + weight * repaired
    sides = []
    for distribution in (original, repaired):
        live = distribution[:, :2]
        total = live.sum(axis=1, keepdims=True)
        sides.append(np.divide(live, total, out=np.full_like(live, 0.5), where=total > 0))
    side = (1 - weight) * sides[0] + weight * sides[1]
    dead = original[:, 2].copy()
    if kind == "boundary":
        dead[outside] += weight * np.maximum(0, repaired[outside, 2] - dead[outside])
    combined = np.column_stack((side * (1 - dead[:, None]), dead))
    unchanged = np.all(original == repaired, axis=1)
    combined[unchanged] = original[unchanged]
    return combined


def evidence(model, lags, positions, recoveries, weights, teams, pitch, fps):
    original = probabilities(model, *tracking_features(positions, weights, teams, pitch, fps, lags))
    hidden = ~np.isfinite(positions[:, -1]).all(axis=1)
    recovered = np.isfinite(recoveries).all(axis=1)
    if np.any(recovered & ~hidden):
        raise ValueError("recoveries must only fill missing source positions")
    outside = (np.maximum(-recoveries, recoveries - pitch) > 1.0).any(axis=1) & recovered
    if not recovered.any():
        return original, original, outside
    repaired = positions.copy()
    repaired[recovered, -1] = recoveries[recovered]
    weights = np.ones(len(positions)) if weights is None else weights.copy()
    weights[recovered] = RECOVERY_WEIGHT
    improved = probabilities(model, *tracking_features(repaired, weights, teams, pitch, fps, lags))
    return original, improved, outside


def predict_labels(model_inputs, fps, costs, policy):
    p = fusion(*model_inputs, policy)
    return decode_costs(np.log(np.maximum(p, 1e-6)), fps, *costs)


def synthetic_recovery(truth, masked, pitch, rate, rng):
    """Restore short tracks with projection error, using only simulated gaps."""
    eligible = np.isfinite(truth).all(axis=1) & ~np.isfinite(masked).all(axis=1)
    recovered = np.full_like(truth, np.nan)
    # A typical track lasts only a few consecutive 25 fps frames.
    indices = np.flatnonzero(eligible)
    for index in rng.choice(indices, round(rate * len(indices) / 4), replace=False):
        end = index + rng.integers(2, 7)
        batch = slice(index, end)
        offset = rng.normal(0, 2.0 if rng.random() < 0.10 else 0.5, size=2)
        points = truth[batch] + offset
        within_margin = ((-3 <= points) & (points <= pitch + 3)).all(axis=1)
        use = eligible[batch] & within_margin
        recovered[batch][use] = points[use]
    return recovered


def calibrate_match(root, model_path, match_id, informative_gaps):
    # The classifier and its Viterbi penalties stay frozen in this comparison.
    model, costs, lags = load_model(model_path)
    rng = np.random.default_rng(int(match_id[1:], 36))
    fps = IDSSE_FPS
    rows = []
    for positions, weights, teams, pitch, labels in load_idsse(root, match_id):
        masked = mask_ball(positions, fps, rng, mean_gap=12,
                           coverage=0.80 if informative_gaps else 0.60)
        if informative_gaps:
            # Use external labels only to simulate greater camera loss during stoppages.
            extra = mask_ball(positions, fps, rng, mean_gap=12, coverage=0.47)
            hidden_in_stoppages = (labels == metrics.D) & ~np.isfinite(extra[:, -1]).all(axis=1)
            masked[hidden_in_stoppages, -1] = np.nan
        runs = metrics.valid_runs(labels)
        for rate in (0.05, 0.30):
            recovery = synthetic_recovery(positions[:, -1], masked[:, -1], pitch, rate, rng)
            inputs = evidence(model, lags, masked, recovery, weights, teams, pitch, fps)
            result = []
            for policy in POLICIES:
                pred = metrics.label_codes(predict_labels(inputs, fps, costs, policy))
                spell = metrics.spell_scores(labels, pred, runs)["mean_f1"]
                confusion = metrics.scored_confusion(metrics.confusion(labels, pred))
                _, _, dead_f1 = metrics.per_class_prf(confusion, metrics.D)
                result.append((spell, dead_f1))
            rows.append(result)
    return np.mean(rows, axis=0)


def calibrate(root, model_path, workers, informative_gaps):
    held_out = tuple(match for match in IDSSE_MATCHES if match not in DEV_GAMES)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {match: pool.submit(calibrate_match, root, model_path, match, informative_gaps)
                for match in IDSSE_MATCHES}
        development = np.mean([jobs[match].result() for match in DEV_GAMES], axis=0)
        policies = list(POLICIES)
        winner = max(range(len(policies)), key=lambda index: tuple(development[index]))
        print(f"Selected on external development: {policies[winner]}", flush=True)
        testing = np.mean([jobs[match].result() for match in held_out], axis=0)
        for policy, dev, test in zip(policies, development, testing):
            print(f"{policy:<12} dev spellF1={dev[0]:.4f} D-F1={dev[1]:.4f} "
                  f"test spellF1={test[0]:.4f} D-F1={test[1]:.4f}", flush=True)


def contest_tracking(data_dir):
    meta, raw = require(["atd_meta", "atd_raw"], data_dir=data_dir)
    teams = team_of_each_track(meta)
    pitch = np.array(GAMES[GAME]["pitch"], dtype=np.float32)
    return read_tracking(raw, source_frames(), len(teams), pitch), teams, pitch, GAMES[GAME]["fps"]


def extract(data_dir, checkpoint, device, output, frame_range, sparse):
    from video_ball import recover_ball

    tracking, _, pitch, fps = contest_tracking(data_dir)
    ball = tracking[:, -1].copy()
    recovered = recover_ball(ball, pitch, fps, data_dir, checkpoint, device, frame_range, sparse=sparse)
    write_ball_positions(output, np.flatnonzero(recovered), ball[recovered], pitch)
    print(f"Saved {recovered.sum():,} accepted observations to {output}")


def read_recoveries(path, n_frames, pitch):
    recoveries = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["frame", "x", "y"]:
            raise ValueError(f"{path}: expected frame,x,y header")
        for row in rows:
            frame = int(row["frame"])
            point = np.array([float(row["x"]), float(row["y"])]) * pitch
            if (not 0 <= frame < n_frames or np.isfinite(recoveries[frame]).any()
                    or not ((-3.01 <= point) & (point <= pitch + 3.01)).all()):
                raise ValueError(f"{path}: invalid or duplicate recovery on frame {frame}")
            recoveries[frame] = point
    return recoveries


def predict(data_dir, model_path, candidates, policy, output):
    model, costs, lags = load_model(model_path)
    positions, teams, pitch, fps = contest_tracking(data_dir)
    recoveries = read_recoveries(candidates, len(positions), pitch)
    labels = np.full(len(positions), "X")
    for _, start, stop in half_video_blocks(GAME):
        half = slice(start, stop + 1)
        inputs = evidence(model, lags, positions[half], recoveries[half], None, teams, pitch, fps)
        labels[half] = predict_labels(inputs, fps, costs, policy)
    write_submission(output, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extracting = commands.add_parser("extract")
    extracting.add_argument("--data-dir", type=Path)
    extracting.add_argument("--checkpoint", type=Path, required=True)
    extracting.add_argument("--device", default="cpu")
    extracting.add_argument("--output", type=Path, required=True)
    extracting.add_argument("--frame-range", type=int, nargs=2, metavar=("START", "STOP"))
    extracting.add_argument("--sparse", action="store_true",
                            help="scan gap edges and one adjacent frame pair per second")
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--idsse", type=Path, required=True)
    calibration.add_argument("--model", type=Path, required=True)
    calibration.add_argument("--workers", type=int, default=4)
    calibration.add_argument("--informative-gaps", action="store_true",
                             help="simulate higher missingness during externally labeled stoppages")
    inference = commands.add_parser("predict")
    inference.add_argument("--data-dir", type=Path)
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--recoveries", type=Path, required=True)
    inference.add_argument("--policy", choices=POLICIES, required=True)
    inference.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        if args.frame_range and not 0 <= args.frame_range[0] < args.frame_range[1] <= source_frames():
            parser.error("--frame-range must be a valid half-open frame range")
        extract(args.data_dir, args.checkpoint, args.device, args.output, args.frame_range, args.sparse)
    elif args.command == "calibrate":
        calibrate(args.idsse, args.model, args.workers, args.informative_gaps)
    else:
        predict(args.data_dir, args.model, args.recoveries, args.policy, args.output)


if __name__ == "__main__":
    main()
