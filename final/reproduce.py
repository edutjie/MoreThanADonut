"""Recreate Absolute Donut's frozen submission from the official tracking bundle."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import joblib
import numpy as np
import torch

from evaluation import possession_metrics as metrics
from experiments.ball_recovery_experiment import contest_tracking
from experiments.possession_hybrid_refinement import LiveDecoder, restore, team_p
from experiments.possession_physics import turnover_adjustment
from experiments.possession_retrain import load_model, predict
from experiments.possession_retrain_data import canonical
from experiments.possession_retrain_score import native_probabilities
from experiments.possession_spell_selection import build_case, select_spells
from experiments.possession_study import decode_sampled, dense_scores
from experiments.possession_tcn import PossessionTCN
from games import GAME, half_video_blocks
from kickoff import decode_after_kickoff, kickoff_offset, match_kickoffs
from player_track_quality import repair_player_tracks
from sequence_possession import sequence_features
from starter.baseline import write_submission


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def feature_transform(x):
    """Exact exported numeric AutoGluon transform; other columns retain NaNs."""
    config = json.loads((ROOT / "models/deadball/transform.json").read_text())
    if x.ndim != 2 or x.shape[1] != config["feature_count"]:
        raise ValueError("unexpected baseline feature width")
    result = x.astype(np.float32, copy=True)
    for column, value in config["boolean_columns"].items():
        result[:, int(column)] = x[:, int(column)] == value
    return result


def deadball_baseline(positions, teams, pitch, fps, device, data_dir):
    state = torch.load(ROOT / "models/deadball/model.pt", map_location="cpu", weights_only=True)["state"]
    model = PossessionTCN(state["mean"], state["scale"])
    model.load_state_dict(state)
    model.to(device).eval()
    kickoffs = match_kickoffs(data_dir, ball=positions[:, -1])
    labels = np.full(len(positions), "X")
    for half, first, last in half_video_blocks(GAME):
        tracks = repair_player_tracks(positions[first:last + 1], fps)
        sample = np.rint(np.arange(0, len(tracks), fps / 10)).astype(int)
        normal, mirror = sequence_features(
            tracks, None, teams, pitch, fps, True, observed_ball_motion=True)
        normal, mirror = (feature_transform(x[sample]) for x in (normal, mirror))
        p = dense_scores(model, normal, mirror, device)
        physics = 2 * turnover_adjustment(tracks, teams, fps, observed_only=True)
        offset = kickoff_offset(kickoffs, half, first, last)
        labels[first:last + 1] = decode_after_kickoff(
            lambda k: decode_sampled(p, sample, len(tracks), fps, start=k,
                                     turnover_adjustment=physics), len(tracks), offset)
        print(f"Dead-ball baseline: half {half} complete", flush=True)
    return labels


def main(args):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for relative, expected in manifest["artifacts"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"frozen artifact differs: {relative}")
    if args.output.resolve() == (ROOT / "Absolute Donut.csv").resolve():
        raise ValueError("choose a separate output path to preserve the frozen submission")
    positions, teams, pitch, fps = contest_tracking(args.data_dir)
    fixed = deadball_baseline(positions, teams, pitch, fps, args.device, args.data_dir)
    model = load_model(ROOT / "models/spatial/final-seed413.pt", args.device)
    judge = joblib.load(ROOT / "models/spells/judge.joblib")
    config = json.loads((ROOT / "models/spells/selection.json").read_text())
    labels = fixed.copy()
    with tempfile.TemporaryDirectory(prefix="absolute-donut-") as cache:
        decoder = LiveDecoder(Path(cache))
        for half, first, last in half_video_blocks(GAME):
            tracks = canonical(positions[first:last + 1:round(fps / 5)], teams, pitch)
            p = team_p(predict(model, tracks, teams, pitch, 0))
            native = native_probabilities(p, last - first + 1, fps)
            physics = turnover_adjustment(repair_player_tracks(tracks, 5), teams, 5, observed_only=True)
            sample = np.arange(len(tracks)) * fps / 5
            physics = restore(physics, sample, len(native))
            visible = np.interp(np.arange(len(native)), sample,
                                np.isfinite(tracks[:, -1]).all(axis=-1).astype(float))
            case = build_case(native, fixed[first:last + 1] == "D", physics, visible, fps, decoder)
            quality = (np.clip(judge.predict(case["features"]), 0, 1)
                       if len(case["features"]) else np.empty(0))
            codes = select_spells(case, quality, config["modes"]["learned"]["penalty"])
            labels[first:last + 1] = np.array(["H", "A", "D"])[codes]
            print(f"Learned spell scoring: half {half} complete", flush=True)
    for code in ("D", "X"):
        if not np.array_equal(labels == code, fixed == code):
            raise ValueError(f"spell scoring changed the baseline {code} mask")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(args.output, labels)
    reason = metrics.validate_submission(metrics.read_label_csv(args.output), len(positions))
    if reason:
        raise ValueError(reason)
    actual = sha256(args.output)
    if actual != manifest["submission_sha256"]:
        raise ValueError(f"format valid, but reproduction differs: SHA-256 {actual}")
    print(f"PASS: {len(labels)} rows; byte-identical frozen submission\nSHA-256 {actual}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="official bundle root containing tactical/")
    parser.add_argument("--output", type=Path, default=ROOT / "recreated.csv")
    parser.add_argument("--device", default="cuda", help="cuda (verified exact); cpu checks but can fail the exact hash")
    ARGS = parser.parse_args()
    main(ARGS)
