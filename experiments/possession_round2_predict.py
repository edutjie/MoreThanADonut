"""Regenerate the one externally confirmed round-two competition submission."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.ball_recovery_experiment import contest_tracking
from experiments.possession_execution import fingerprint
from experiments.possession_round2 import adapt_match, decoded, source_from, variants_emissions
from experiments.possession_round2_adapt import prediction
from experiments.possession_round2_decoder import Decoder
from experiments.possession_round2_segments import likelihood, touch_features, touch_inputs
from experiments.possession_round2_train import load, train_data
from games import GAME, half_video_blocks
from kickoff import decode_after_kickoff, kickoff_offset, match_kickoffs
from player_track_quality import repair_player_tracks
from sequence_possession import sequence_features
from starter.baseline import write_submission


def main(args):
    decision = json.loads(args.selection.read_text())
    if fingerprint(args.selection.parent / "development-freeze.json") != decision["development_sha256"]:
        raise ValueError("development manifest differs from the frozen decision")
    source = source_from(decision["source"])
    recipe = decision["decoder"]
    decoder = Decoder(args.selection.parent)
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    models = {path: load(path, device) for path in source.paths}
    tracking, teams, pitch, fps = contest_tracking(args.data_dir)
    blocks = half_video_blocks(GAME)
    halves, physical = [], {}
    for i, (_, start, end) in enumerate(blocks):
        repaired = repair_player_tracks(tracking[start:end + 1], fps)
        x, mx = sequence_features(repaired, None, teams, pitch, fps, True)
        halves.append((fps, x, mx))
        if "snap" in recipe:
            physical[i] = touch_inputs(repaired, fps)
    if "adaptation" in recipe:
        mode, setting = recipe["adaptation"]
        external = {}
        if mode == "self":
            if args.observations is None:
                raise ValueError("self-training needs the frozen external training features")
            members = json.loads((args.selection.parent / "run-index.json").read_text())["split"]["train"]
            for path in source.paths:
                kind = path.stem.rsplit("-", 1)[0]
                matches = [v for v in members if isinstance(v, int) or kind not in ("gru", "tcn-aleague")]
                external[kind] = [row for row in train_data(args.observations, matches, ("natural",))
                                  if len(row[2]) >= 512]
        emissions = adapt_match(source, models, halves, mode, setting, decoder, external)
    else:
        bank = {path: dict(enumerate(prediction(model, halves))) for path, model in models.items()}
        emissions = variants_emissions(source, range(len(halves)), bank)
    touch = {}
    if physical:
        reference = source_from(decision["touch_reference"])
        model = load(reference.paths[0], device)
        for i, p in enumerate(prediction(model, halves)):
            touch[i] = likelihood(touch_features(physical[i], p, fps), decision["touch"])
    labels = np.full(len(tracking), "X")
    states = np.array(["H", "A", "D"])
    kickoffs = match_kickoffs(args.data_dir, ball=tracking[:, -1]) if args.prekickoff else {}
    for i, (half, start, end) in enumerate(blocks):
        offset = kickoff_offset(kickoffs, half, start, end)

        def decode_from(k, i=i):
            cue = touch.get(i)
            return states[decoded(decoder, emissions[i][k:], fps, recipe, decision["duration"],
                                  None if cue is None else cue[k:])]

        labels[start:end + 1] = decode_after_kickoff(decode_from, end + 1 - start, offset)
        if args.prekickoff:
            print(f"half {half}: kickoff {kickoffs.get(half)}, {offset} graded frames set to D")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(args.output, labels)
    print(f"{len(labels):,} frames, SHA-256 {fingerprint(args.output)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("selection", "data-dir", "output", "observations"):
        parser.add_argument(f"--{field}", type=Path, required=field != "observations")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prekickoff", action="store_true",
                        help="label graded frames before each detected kickoff D and decode from it")
    main(parser.parse_args())
