"""Test Bundesliga LightGBM as a gated dead-ball specialist."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from experiments.ball_recovery_experiment import contest_tracking
from evaluation import possession_metrics as metrics
from external_possession import (
    DEV_MATCHES, IDSSE_FPS, LAG_SECONDS, TEST_MATCHES, decode_costs, load_idsse, load_model,
    probabilities, tracking_features,
)
from games import GAME, half_video_blocks
from experiments.idsse_training import IDSSE_DEV, IDSSE_TEST, IDSSE_TRAIN, evaluation_inputs
from starter.baseline import write_submission


POLICIES = [("none", 0)] + [
    (kind, strength) for kind in ("global", "in-pitch", "in-pitch-slow", "in-pitch-uncertain")
    for strength in (0.15, 0.35, 0.60)
]


def observed_motion(ball, pitch, fps):
    inside = ((ball > 0.5) & (ball < pitch - 0.5)).all(axis=1)
    # Use only observed frames for the gate. A gap is never treated as a stationary ball.
    width = max(1, round(0.5 * fps))
    slow = np.zeros(len(ball), dtype=bool)
    if len(ball) > width:
        step = np.linalg.norm(np.diff(ball, axis=0), axis=1)
        still = np.isfinite(step) & (step <= 1.5 / fps)
        prefix = np.r_[0, np.cumsum(still)]
        slow[width:] = (prefix[width:] - prefix[:-width]) == width
    return inside, slow


def prepare(reference, specialist, positions, weights, teams, pitch, fps):
    features = tracking_features(positions, weights, teams, pitch, fps, LAG_SECONDS)
    base, expert = (probabilities(model, *features) for model in (reference, specialist))
    base_d, expert_d = (np.clip(p[:, metrics.D], 1e-6, 1 - 1e-6) for p in (base, expert))
    diff = np.log(expert_d / (1 - expert_d)) - np.log(base_d / (1 - base_d))
    inside, slow = observed_motion(positions[:, -1], pitch, fps)
    return np.log(np.maximum(base, 1e-6)), np.clip(diff, -3, 3), base_d, inside, slow


def blend(prepared, policy):
    base, difference, base_d, inside, slow = prepared
    gate, strength = policy
    if gate == "none":
        return base
    result = base.copy()
    if gate == "global":
        result[:, metrics.D] += strength * difference
    elif gate == "in-pitch":
        result[inside, metrics.D] += strength * difference[inside]
    else:
        selected = inside & slow
        if gate == "in-pitch-uncertain":
            selected &= (base_d >= 0.15) & (base_d <= 0.85)
        result[selected, metrics.D] += strength * np.maximum(0, difference[selected])
    return result


def learned_durations(root):
    durations = []
    for match in IDSSE_TRAIN:
        for positions, _, _, pitch, labels in load_idsse(root, match):
            inside = ((positions[:, -1] > 0.5) & (positions[:, -1] < pitch - 0.5)).all(axis=1)
            edges = np.flatnonzero(np.diff(np.r_[False, labels == metrics.D, False]))
            durations.extend((stop - start) / IDSSE_FPS
                             for start, stop in zip(edges[::2], edges[1::2])
                             if start > 0 and stop < len(labels) and inside[start:stop].mean() >= 0.8)
    if not durations:
        raise ValueError("IDSSE training matches contained no in-pitch stoppages")
    limits = np.unique(np.quantile(durations, (0.10, 0.25, 0.50)))
    print(f"Duration limits from {len(durations)} IDSSE training stoppages: {limits.tolist()}", flush=True)
    return [POLICIES[0], *[("duration", float(value)) for value in limits]]


def bridge_short_stoppages(labels, inside, fps, limit):
    edges = np.flatnonzero(np.diff(np.r_[False, labels == "D", False]))
    for start, stop in zip(edges[::2], edges[1::2]):
        if (start > 0 and stop < len(labels) and labels[start - 1] == labels[stop]
                and (stop - start) / fps < limit and inside[start:stop].all()):
            labels[start:stop] = labels[start - 1]
    return labels


def decode_policy(prepared, fps, costs, policy):
    if policy[0] == "duration":
        labels = decode_costs(prepared[0], fps, *costs)
        return bridge_short_stoppages(labels, prepared[3], fps, policy[1])
    return decode_costs(blend(prepared, policy), fps, *costs)


def checkpoints(reference_path, specialist_path):
    reference, costs, lags = load_model(reference_path)
    specialist, _, expert_lags = load_model(specialist_path)
    if lags != LAG_SECONDS or expert_lags != LAG_SECONDS:
        raise ValueError("both experts must use the frozen lag feature layout")
    return reference, specialist, costs


def benchmark_match(root, league, match, reference_path, specialist_path, policies):
    reference, specialist, costs = checkpoints(reference_path, specialist_path)
    total = []
    for positions, weights, teams, pitch, labels, inside, fps in evaluation_inputs(root, league, match):
        prepared = prepare(reference, specialist, positions, weights, teams, pitch, fps)
        runs = metrics.valid_runs(labels)
        scores = []
        for policy in policies:
            pred = metrics.label_codes(decode_policy(prepared, fps, costs, policy))
            spell = metrics.spell_scores(labels, pred, runs)["mean_f1"]
            cm = metrics.scored_confusion(metrics.confusion(labels, pred))
            inside_cm = metrics.scored_confusion(metrics.confusion(labels[inside], pred[inside]))
            scores.append([spell, metrics.macro_f1(cm), metrics.per_class_prf(cm, metrics.D)[2],
                           metrics.per_class_prf(inside_cm, metrics.D)[2]])
        total.append(scores)
    return np.mean(total, axis=0)


def evaluate(pool, idsse, skillcorner, reference, specialist, german, australian, policies):
    splits = {"Bundesliga": (idsse, german), "A-League": (skillcorner, sorted(australian))}
    jobs = {league: [pool.submit(benchmark_match, root, league, match, reference, specialist, policies)
                     for match in matches] for league, (root, matches) in splits.items()}
    return {league: np.mean([task.result() for task in tasks], axis=0) for league, tasks in jobs.items()}


def calibrate(args):
    policies = learned_durations(args.idsse) if args.duration_only else POLICIES
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        common = (pool, args.idsse, args.skillcorner, args.reference, args.specialist)
        dev = evaluate(*common, IDSSE_DEV, DEV_MATCHES, policies)
        averages = np.mean(list(dev.values()), axis=0)
        selected = max(range(len(policies)), key=lambda i: (averages[i, 0], -i))
        for index, policy in enumerate(policies):
            print(f"Development {policy}: balanced spellF1={averages[index, 0]:.6f}; "
                  f"A-League={dev['A-League'][index, 0]:.6f}; "
                  f"Bundesliga={dev['Bundesliga'][index, 0]:.6f}", flush=True)
        print(f"Frozen external selection: {policies[selected]}", flush=True)
        test = evaluate(*common, IDSSE_TEST, TEST_MATCHES, [policies[0], policies[selected]])
    report = {
        "metrics": ["spell_f1", "macro_f1", "dead_f1", "in_pitch_dead_f1"],
        "policies": policies, "selected": policies[selected],
        "development": {league: scores.tolist() for league, scores in dev.items()},
        "held_out_reference_and_selected": {league: scores.tolist() for league, scores in test.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Held-out reference and selected: {report['held_out_reference_and_selected']}")


def predict(args):
    reference, specialist, costs = checkpoints(args.reference, args.specialist)
    policy = tuple(json.loads(args.calibration.read_text())["selected"])
    if policy not in POLICIES and not (policy[0] == "duration" and 0 < policy[1] < float("inf")):
        raise ValueError("unknown calibrated policy")
    x, teams, pitch, fps = contest_tracking(args.data_dir)
    labels = np.full(len(x), "X")
    for _, start, last in half_video_blocks(GAME):
        prepared = prepare(reference, specialist, x[start:last + 1], None, teams, pitch, fps)
        labels[start:last + 1] = decode_policy(prepared, fps, costs, policy)
    write_submission(args.output, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    external = commands.add_parser("calibrate")
    external.add_argument("--idsse", type=Path, required=True)
    external.add_argument("--skillcorner", type=Path, required=True)
    external.add_argument("--workers", type=int, default=4)
    external.add_argument("--duration-only", action="store_true",
                          help="test duration limits learned from Bundesliga training stops")
    contest = commands.add_parser("predict")
    contest.add_argument("--data-dir", type=Path)
    contest.add_argument("--calibration", type=Path, required=True)
    for command in (external, contest):
        command.add_argument("--reference", type=Path, required=True)
        command.add_argument("--specialist", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args)
    else:
        predict(args)
