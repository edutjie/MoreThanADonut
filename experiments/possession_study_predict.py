"""Run a frozen possession study candidate on the tactical submission grid."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from experiments.ball_recovery_experiment import contest_tracking
from experiments.possession_study import (
    MIRROR, columns, decode_sampled, dense_scores, mirror_objects, rolling_features,
)
from experiments.possession_study_data import track_channels
from experiments.possession_tcn import PlayerTCN, PossessionTCN
from external_possession import SKILLCORNER_FPS
from games import GAME, half_video_blocks
from kickoff import decode_after_kickoff, kickoff_offset, match_kickoffs
from player_track_quality import repair_player_tracks
from sequence_possession import ARCHITECTURES, sequence_features
from starter.baseline import write_submission


def frozen_summary(kind, directory, device):
    if kind == "lgbm":
        model = joblib.load(directory / "model.joblib")
        prob = model.predict_proba
    elif kind == "automl":
        import pandas as pd
        from autogluon.tabular import TabularPredictor
        model = TabularPredictor.load(str(directory / "autogluon"))

        def prob(x):
            return model.predict_proba(pd.DataFrame(x, columns=columns(x.shape[1])))[[0, 1, 2]].to_numpy()

    else:
        factory = PossessionTCN if kind == "tcn" else ARCHITECTURES[kind]
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)["state"]
        model = factory(state["mean"], state["scale"])
        model.load_state_dict(state)
        model.to(device).eval()

        def predict(x, mirror):
            return dense_scores(model, x, mirror, device)

        return predict

    def predict(x, mirror):
        return (prob(x) + prob(mirror)[:, MIRROR[:3]]) / 2

    return predict


def frozen_candidate(directory, device):
    config = json.loads((directory / "benchmark.json").read_text())
    kind = config.get("model", directory.name)
    if kind == "player-tcn":
        model = PlayerTCN()
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)["state"]
        model.load_state_dict(state)
        model.to(device).eval()

        def predict(tracks, teams, pitch, fps, sample):
            objects = track_channels(tracks, teams, pitch, fps, None)[sample]
            return dense_scores(model, objects, mirror_objects(objects), device)

        return predict

    if kind in {"moment", "mantis"}:
        from experiments.possession_foundation import frozen_probabilities

        predict_summary = frozen_probabilities(kind, directory, device)
    else:
        predict_summary = frozen_summary(kind, directory, device)
    generator = (joblib.load(directory / "features.joblib")
                 if config.get("features", "original") != "original" else None)
    if generator is not None:
        import pandas as pd

    def predict(tracks, teams, pitch, fps, sample):
        normal, mirror = (x[sample] for x in sequence_features(tracks, None, teams, pitch, fps, True))
        if config.get("features") == "rolling-auto":
            normal, mirror = rolling_features(normal), rolling_features(mirror)
        if generator is not None:
            normal, mirror = (
                generator.transform(pd.DataFrame(a, columns=columns(a.shape[1]))).to_numpy(dtype=np.float32)
                for a in (normal, mirror)
            )
        return predict_summary(normal, mirror)

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prekickoff", action="store_true",
                        help="label graded frames before each detected kickoff D and decode from it")
    args = parser.parse_args()
    predict = frozen_candidate(args.candidate, args.device)
    positions, teams, pitch, fps = contest_tracking(args.data_dir)
    kickoffs = match_kickoffs(args.data_dir, ball=positions[:, -1]) if args.prekickoff else {}
    labels = np.full(len(positions), "X")
    for half, first, last in half_video_blocks(GAME):
        tracks = repair_player_tracks(positions[first:last + 1], fps)
        sample = np.rint(np.arange(0, len(tracks), fps / SKILLCORNER_FPS)).astype(int)
        p = predict(tracks, teams, pitch, fps, sample)
        offset = kickoff_offset(kickoffs, half, first, last)
        labels[first:last + 1] = decode_after_kickoff(
            lambda k: decode_sampled(p, sample, len(tracks), fps, start=k), len(tracks), offset)
        if args.prekickoff:
            print(f"half {half}: kickoff {kickoffs.get(half)}, {offset} graded frames set to D")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(args.output, labels)


if __name__ == "__main__":
    main()
