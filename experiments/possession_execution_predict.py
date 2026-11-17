"""Run the externally selected provider-head GRU recipe on the official frame grid."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.ball_recovery_experiment import contest_tracking
from experiments.possession_execution import dump, fingerprint
from experiments.possession_provider_heads import Inference, ProviderGRU, boundary_probabilities, decode_boundary
from games import GAME, half_video_blocks
from player_track_quality import repair_player_tracks
from sequence_possession import DECODER_COSTS, sequence_features, sequence_probabilities
from starter.baseline import write_submission


def main(args):
    device = torch.device(args.device)
    selection = json.loads(args.selection.read_text())
    if selection["selected"] != "pooled-1":
        raise ValueError("expected the external development selection of equal pooling and boundary strength 1")
    checkpoint_sha256 = fingerprint(args.checkpoint)
    if checkpoint_sha256 != selection["checkpoint_sha256"]["provider"]:
        raise ValueError("checkpoint differs from the external development selection")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = payload["state"]
    model = ProviderGRU(state["mean"], state["scale"])
    model.load_state_dict(state)
    model.to(device).eval()
    inference = Inference(model, None).eval()
    source, teams, pitch, fps = contest_tracking(args.data_dir)
    labels = np.full(len(source), "X")
    for _, start, end in half_video_blocks(GAME):
        tracks = repair_player_tracks(source[start:end + 1], fps)
        x, mx = sequence_features(tracks, None, teams, pitch, fps, True)
        p = sequence_probabilities(inference, x, mx, fps, device)
        boundary = boundary_probabilities(model, x, mx, fps, device)
        labels[start:end + 1] = np.array(["H", "A", "D"])[decode_boundary(p, fps, boundary, 1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(args.output, labels)
    dump(args.output.parent / "frozen-recipe.json", {
        "architecture": "provider-head GRU", "provider_pooling": "equal",
        "boundary_strength": 1, "decoder_costs": DECODER_COSTS, "fps": fps,
        "frames": len(labels), "epoch": payload["epoch"], "checkpoint_sha256": checkpoint_sha256,
        "external_selection_sha256": fingerprint(args.selection), "prediction_sha256": fingerprint(args.output),
        "input": "audited tactical tracking", "preprocessing": "repair + continuous speed, feature set 113",
    })
    print(f"Frozen {len(labels):,}-frame prediction: {fingerprint(args.output)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("checkpoint", "selection", "data-dir", "output"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
