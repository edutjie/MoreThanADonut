"""Run the frozen possession GRU on aligned broadcast and tactical tracking."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from broadcast_alignment import load_feed, load_frame_pairs
from external_possession import decode_costs
from games import GAME, GAMES, half_video_blocks
from sequence_possession import DECODER_COSTS, restore, tracking_probabilities
from starter.baseline import write_submission


POLICIES = ("tactical", "broadcast-first", "broadcast-ball-gap", "tactical-clipped")


def replace_probabilities(tactical, broadcast, indices, gaps):
    """Replace only verified tactical rows and preserve the tactical fallback."""
    hybrid = tactical.copy()
    hybrid[indices] = broadcast
    gap_only = tactical.copy()
    gap_only[indices[gaps]] = broadcast[gaps]
    return tactical, hybrid, gap_only


def aligned_probabilities(model, tactical, broadcast, tactical_teams, broadcast_teams, pitch, fps,
                          device, broadcast_indices, tactical_indices):
    """Predict independently on each contiguous, verified section of the broadcast."""
    cuts = np.flatnonzero((np.diff(broadcast_indices) != 1) | (np.diff(tactical_indices) != 1)) + 1
    broadcast_spans = np.split(broadcast_indices, cuts) if len(broadcast_indices) else []
    tactical_spans = np.split(tactical_indices, cuts) if len(tactical_indices) else []
    # Use the same frame cuts on tactical as a control for the shorter sequence context.
    estimates = []
    for feed, teams, spans in ((broadcast, broadcast_teams, broadcast_spans),
                               (tactical, tactical_teams, tactical_spans)):
        estimates.append(np.concatenate([
            tracking_probabilities(model, feed[span[0]:span[-1] + 1], teams, pitch, fps, device)
            for span in spans
        ]) if spans else np.empty((0, 3)))
    predicted, clipped = estimates
    base = tracking_probabilities(model, tactical, tactical_teams, pitch, fps, device)
    tactical_ball = np.isfinite(tactical[tactical_indices, -1]).all(axis=1)
    broadcast_ball = np.isfinite(broadcast[broadcast_indices, -1]).all(axis=1)
    gap = broadcast_ball & ~tactical_ball
    coverage = {
        "tactical_half_frames": len(tactical),
        "verified_frames": len(tactical_indices),
        "verified_spans": len(broadcast_spans),
        "both_observe_ball": int((broadcast_ball & tactical_ball).sum()),
        "broadcast_only_ball": int(gap.sum()),
        "tactical_only_ball": int((tactical_ball & ~broadcast_ball).sum()),
        "neither_observes_ball": int((~broadcast_ball & ~tactical_ball).sum()),
    }
    clipped_base = base.copy()
    clipped_base[tactical_indices] = clipped
    return (*replace_probabilities(base, predicted, tactical_indices, gap), clipped_base), coverage


def run(args):
    game = GAMES[GAME]
    pitch, fps = np.asarray(game["pitch"], dtype=np.float32), game["fps"]
    tactical, tactical_teams, names = load_feed("tactical", args.data_dir, pitch)
    broadcast, broadcast_teams, broadcast_names = load_feed("broadcast", args.data_dir, pitch)
    if names != broadcast_names:
        raise ValueError("Broadcast and tactical player team order differs")
    broadcast_idx, tactical_idx = load_frame_pairs(args.broadcast_map, len(tactical), len(broadcast) - 1)
    device = torch.device(args.device)
    model, _ = restore(args.model, device)
    outputs = {policy: np.full(len(tactical), "X") for policy in POLICIES}
    coverage = []
    for half, start, last in half_video_blocks(GAME):
        select = (tactical_idx >= start) & (tactical_idx <= last)
        estimates, counts = aligned_probabilities(
            model, tactical[start:last + 1], broadcast, tactical_teams, broadcast_teams, pitch, fps,
            device, broadcast_idx[select], tactical_idx[select] - start,
        )
        for policy, p in zip(POLICIES, estimates):
            outputs[policy][start:last + 1] = decode_costs(
                np.log(np.maximum(p, 1e-6)), fps, *DECODER_COSTS)
        coverage.append({"half": half, **counts})

    args.output.mkdir(parents=True, exist_ok=True)
    for policy, labels in outputs.items():
        path = args.output / policy / "possession.csv"
        path.parent.mkdir(exist_ok=True)
        write_submission(path, labels)
    summary = {"coverage": coverage,
               "changed_from_tactical": {policy: int((labels != outputs["tactical"]).sum())
                                         for policy, labels in outputs.items()}}
    (args.output / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "broadcast-map", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())
