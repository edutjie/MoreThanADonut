"""Verify the original half runner and score a separate Metrica event reference."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from evaluation import possession_metrics as m
from experiments.player_quality_experiment import predict_half
from experiments.possession_execution import (
    SEED, dataset, dead_intervals, dump, fixed, measure, restart_cues, restart_decode,
    restart_probability, rows, split_members,
)
from experiments.possession_provider_heads import Inference, ProviderGRU, boundary_probabilities, decode_boundary
from games import GAME, half_video_blocks, id_to_video, video_to_id
from player_track_quality import repair_player_tracks
from possession_sources import METRICA_FPS, METRICA_MATCHES, load_metrica
from sequence_possession import restore, sequence_features, sequence_probabilities
from starter.baseline import write_submission


def synthetic_grid(output):
    samples = np.r_[np.tile([0.98, 0.01, 0.01], (25, 1)),
                    np.tile([0.01, 0.01, 0.98], (25, 1)),
                    np.tile([0.98, 0.01, 0.01], (25, 1))]
    neutral = np.full_like(samples[:, :2], 0.5)
    labels = restart_decode(samples, 25, np.ones(len(samples)), neutral, 4)
    assert np.array_equal(labels, fixed(samples, 25))
    assert len(m.possession_spells(labels, m.valid_runs(labels))) == 2
    _, a, b = half_video_blocks(GAME)[0]
    _, c, d = half_video_blocks(GAME)[1]
    assert b < c
    assert all(id_to_video(GAME, video_to_id(GAME, f)) == f for f in (a, b, c, d))
    assert video_to_id(GAME, a + 1) == video_to_id(GAME, a) + 1
    fixture = output / "synthetic-grid.csv"
    write_submission(fixture, np.array(["H", "D", "H", "X", "H", "A"]))
    assert m.validate_submission(m.read_label_csv(fixture), 6) is None
    assert m.validate_submission(m.read_label_csv(fixture), 5) is not None


def compare(args, members, model, device):
    checked = {}
    candidate = "GRU continuous motion"
    for match, half, stored in rows(args.output, members["test"]):
        if half == 1:
            root, loader, fps = dataset(match, args.skillcorner, args.idsse)
            halves = list(loader(root, match))
        positions, weights, teams, pitch, _ = halves[half - 1]
        expected = predict_half(positions, weights, teams, pitch, fps, {"continuous": model},
                                (None, ("reference", 0.0)), device, [candidate])[candidate]
        assert np.array_equal(m.label_codes(expected), stored["baseline"]), (match, half)
        checked.setdefault(str(match), []).append(len(positions))
    return checked


def metrica(args, models, device):
    output = {}
    for match in METRICA_MATCHES:
        for tracks, weights, teams, pitch, ground in load_metrica(args.metrica, match):
            fps = METRICA_FPS
            edge = restart_cues(tracks, teams, pitch, fps)[-1]
            repaired = repair_player_tracks(tracks, fps)
            x, mx = sequence_features(repaired, weights, teams, pitch, fps, True)
            missing = ~np.isfinite(tracks[:, -1]).all(axis=1)
            for name, model in models.items():
                p = sequence_probabilities(model, x, mx, fps, device)
                pred = fixed(p, fps)
                if isinstance(model, Inference):
                    boundary = boundary_probabilities(model.model, x, mx, fps, device)
                    pred = decode_boundary(p, fps, boundary, 1)
                value = measure(ground, pred, missing, edge, fps)
                value["annotated_frames"] = int((ground != m.X).sum())
                output.setdefault(name, {}).setdefault(str(match), []).append(value)
    return output


def synthetic_tags(args, members, head):
    # A deliberately noisy oracle proxy checks sensitivity only. No tag model is selected from it.
    report = {}
    rng = np.random.default_rng(SEED)
    for match, _, row in rows(args.output, members["dev"]):
        p, timing, ground = row["p"], row["timing"], row["ground"]
        fps = int(row["fps"])
        cues = timing.copy()
        side = restart_probability(head, row)
        hinted = side.copy()
        intervals = list(dead_intervals(ground))
        proposals = [(stop, int(ground[stop])) for _, stop in intervals if rng.random() > 0.3]
        proposals.extend((int(frame), int(rng.integers(2))) for frame in
                         rng.integers(len(p), size=len(intervals) // 4))
        for stop, team in proposals:
            center = stop + rng.integers(-fps, fps + 1)
            region = slice(max(0, center - fps // 2), min(len(p), center + fps // 2 + 1))
            team = team if rng.random() > 0.3 else 1 - team
            soft = np.array([0.35, 0.65]) if team else np.array([0.65, 0.35])
            hinted[region] = 0.75 * hinted[region] + 0.25 * soft
            cues[region] = np.minimum(1, cues[region] + 0.2)
        comparisons = {"tag_free": (timing, side), "noisy_proxy": (cues, hinted)}
        for name, (kick, team) in comparisons.items():
            pred = restart_decode(p, fps, kick, team, 4)
            value = measure(ground, pred, row["missing"], row["edge"], fps)
            report.setdefault(name, {}).setdefault(str(match), []).append(value)
    dump(args.output / "restart-noisy-tag-proxy.json", report)


def main(args):
    members = split_members(args.skillcorner)
    device = torch.device(args.device)
    frozen, _ = restore(args.gru, device)
    synthetic_grid(args.output)
    dump(args.output / "baseline-replay.json", compare(args, members, frozen, device))
    head = joblib.load(args.output / "restart-team.joblib")
    report = {}
    for role in ("dev", "test"):
        report[role] = {}
        for match, half, item in rows(args.output, members[role]):
            indices = [stop for _, stop in dead_intervals(item["ground"])]
            side = restart_probability(head, item | {
                key: item[key][indices] for key in ("normal", "flipped")
            }).argmax(axis=1)
            correct = int((side == item["ground"][indices]).sum())
            report[role].setdefault(str(match), []).append({"known": len(indices), "correct": correct})
    dump(args.output / "restart-known-team.json", report)
    synthetic_tags(args, members, head)
    models = {"frozen": frozen, "random": restore(args.random, device)[0]}
    if args.provider:
        weights = torch.load(args.provider, weights_only=True, map_location="cpu")["state"]
        provider = ProviderGRU(weights["mean"], weights["scale"])
        provider.load_state_dict(weights)
        models["pooled-1"] = Inference(provider.to(device).eval(), None).eval()
    dump(args.output / "metrica-event-reference.json", metrica(args, models, device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("skillcorner", "idsse", "metrica", "gru", "random", "output"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    parser.add_argument("--provider", type=Path)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
