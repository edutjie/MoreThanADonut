"""Compare raw, repaired and retrained player inputs for GRU possession."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from ball_recovery_experiment import contest_tracking
from external_possession import (
    DEV_MATCHES, IDSSE_FPS, LAG_SECONDS, SKILLCORNER_FPS, TEST_MATCHES,
    decode_costs, load_idsse, load_skillcorner, tracking_features,
)
from games import GAME, half_video_blocks
from gru_dead_ball import blended_emissions, observed
from idsse_training import IDSSE_TEST
from player_track_quality import corrupt_player_tracks, repair_player_tracks
from sequence_possession import (
    DECODER_COSTS, restore, score_prediction, sequence_probabilities,
    visibility_features,
)
from starter.baseline import write_submission


# Each candidate has the same decoder and, where used, the same frozen specialist policy.
CANDIDATES = {
    "GRU": ("original", False, False),
    "GRU + Specialist": ("original", False, True),
    "GRU repaired": ("original", True, False),
    "GRU + Specialist repaired": ("original", True, True),
    "GRU retrained": ("retrained", True, False),
    "GRU retrained + Specialist": ("retrained", True, True),
    "GRU continuous motion": ("continuous", True, False),
    "GRU continuous motion + Specialist": ("continuous", True, True),
}


def predict_half(positions, weights, teams, pitch, fps, models, specialist, device, candidates):
    inside, _ = observed(positions, weights, pitch, fps)
    visible = visibility_features(positions[:, -1], fps)
    predictions = {}
    for repair in dict.fromkeys(CANDIDATES[name][1] for name in candidates):
        tracks = repair_player_tracks(positions, fps) if repair else positions
        x, mirror = tracking_features(tracks, weights, teams, pitch, fps, LAG_SECONDS)
        same_inputs = [name for name in candidates if CANDIDATES[name][1] == repair]
        for checkpoint in dict.fromkeys(CANDIDATES[name][0] for name in same_inputs):
            model = models[checkpoint]
            features = (tracking_features(tracks, weights, teams, pitch, fps, LAG_SECONDS,
                                          continuous_motion=True)
                        if model.continuous_motion else (x, mirror))
            normal, flipped = (np.column_stack((side, visible)) for side in features)
            p = sequence_probabilities(model, normal, flipped, fps, device)
            names = [name for name in same_inputs if CANDIDATES[name][0] == checkpoint]
            specialists, enabled = specialist
            policies = [enabled if CANDIDATES[name][2] else ("reference", 0.0) for name in names]
            emissions = blended_emissions(p, x, mirror, inside, specialists, policies)
            for name, emission in zip(names, emissions):
                predictions[name] = decode_costs(emission, fps, *DECODER_COSTS)
    return predictions


def score_split(root, matches, models, specialist, device, candidates, inject):
    idsse = isinstance(matches[0], str)
    loader, fps = (load_idsse, IDSSE_FPS) if idsse else (load_skillcorner, SKILLCORNER_FPS)
    scores = {name: [] for name in candidates}
    for match in matches:
        rng = np.random.default_rng(int(match[1:], 36) if idsse else match)
        halves = {name: [] for name in candidates}
        for positions, weights, teams, pitch, labels in loader(root, match):
            if inject:
                positions = corrupt_player_tracks(positions, pitch, fps, rng)
            missing = ~np.isfinite(positions[:, -1]).all(axis=1)
            predictions = predict_half(positions, weights, teams, pitch, fps, models, specialist,
                                       device, candidates)
            for name, prediction in predictions.items():
                halves[name].append(score_prediction(prediction, labels, missing, fps)[:3])
        for name, values in halves.items():
            scores[name].append(np.mean(values, axis=0))
    return {name: np.mean(values, axis=0).tolist() for name, values in scores.items()}


def run(args):
    device = torch.device(args.device)
    models = {name: restore(path, device)[0] for name, path in (
        ("original", args.original), ("retrained", args.retrained), ("continuous", args.continuous)
    ) if path is not None}
    candidates = [name for name, (checkpoint, _, _) in CANDIDATES.items() if checkpoint in models]
    specialists, policy, _ = joblib.load(args.specialists)
    specialist = specialists, policy
    development = {}
    for inject in (False, True):
        name = "injected" if inject else "original"
        development[name] = score_split(args.skillcorner, sorted(DEV_MATCHES), models, specialist,
                                        device, candidates, inject)
        print(f"Development {name}: {development[name]}", flush=True)
    means = {name: np.mean([group[name][0] for group in development.values()]) for name in candidates}
    chosen = max(means, key=means.get)
    cleaned = max((name for name in means if CANDIDATES[name][1]), key=means.get)
    print(f"Frozen overall choice {chosen}; best repaired {cleaned}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    sets = (("A-League", args.skillcorner, sorted(TEST_MATCHES), False),
            ("A-League injected", args.skillcorner, sorted(TEST_MATCHES), True),
            ("Bundesliga", args.idsse, IDSSE_TEST, False),
            ("Bundesliga injected", args.idsse, IDSSE_TEST, True))
    result = {}
    for name, root, matches, inject in sets:
        result[name] = score_split(root, matches, models, specialist, device, candidates, inject)
        print(f"Held-out {name}: {result[name]}", flush=True)
    report = {"metrics": ["spell_f1", "macro_f1", "dead_f1"], "selected": chosen,
              "best_repaired": cleaned, "development": development, "held_out": result}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    tracks, teams, pitch, fps = contest_tracking(args.data_dir)
    labels = np.full(len(tracks), "X")
    changed = count = 0
    for _, start, stop in half_video_blocks(GAME):
        half = slice(start, stop + 1)
        players = tracks[half, :-1]
        repaired = repair_player_tracks(tracks[half], fps)[:, :-1]
        finite = np.isfinite(players).all(axis=-1)
        changed += int((finite & np.any(players != repaired, axis=-1)).sum())
        count += int(finite.sum())
        labs = predict_half(tracks[half], None, teams, pitch, fps, models, specialist,
                            device, [cleaned])
        labels[half] = labs[cleaned]
    candidate = args.output / "candidate" / "possession.csv"
    candidate.parent.mkdir(exist_ok=True)
    write_submission(candidate, labels)
    print(f"Player cleanup changed {changed:,} of {count:,} observed tactical player-frame "
          f"positions in the match halves. Wrote {candidate}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("skillcorner", "idsse", "original", "retrained", "specialists", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--continuous", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path)
    run(parser.parse_args())
