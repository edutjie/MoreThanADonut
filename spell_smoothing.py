"""Calibrate possession spell smoothing on external matches."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ball_recovery_experiment import contest_tracking
from evaluation import possession_metrics as metrics
from external_possession import DEV_MATCHES, TEST_MATCHES, load_model, probabilities, tracking_features
from games import GAME, half_video_blocks
from idsse_training import IDSSE_DEV, IDSSE_TEST, evaluation_inputs
from starter.baseline import write_submission
from tactical_baseline import LABELS, decode


# Turnover cost, D-entry cost, D-exit cost, maximum low-margin live flicker seconds.
POLICIES = (
    (5, 10, 10, 0),  # frozen LightGBM decoder
    (3.5, 10, 10, 0), (7.5, 10, 10, 0), (10, 10, 10, 0), (15, 10, 10, 0),
    (5, 7.5, 10, 0), (5, 12.5, 10, 0), (5, 15, 10, 0),
    (5, 10, 7.5, 0), (5, 10, 15, 0), (7.5, 12.5, 10, 0), (7.5, 10, 7.5, 0),
    (5, 10, 10, 0.75), (5, 10, 10, 1.5), (5, 10, 10, 3.0),
)


def bridge_live_flickers(labels, scaled_emission, fps, seconds, evidence_limit):
    edges = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    result = labels.copy()
    for start, stop in zip(edges[1:-2], edges[2:-1]):
        before, middle, after = labels[start - 1], labels[start], labels[stop]
        if before != after or "D" in (before, middle) or (stop - start) / fps > seconds:
            continue
        keep = int(np.flatnonzero(LABELS == before)[0])
        change = int(np.flatnonzero(LABELS == middle)[0])
        margin = (scaled_emission[start:stop, change] - scaled_emission[start:stop, keep]).sum()
        if margin <= evidence_limit:
            result[start:stop] = before
    return result


def decode_policy(emission, fps, policy):
    turnover, enter, leave, flicker = policy
    scaled = emission * (10 / fps)
    transitions = np.array([[0, turnover, enter], [turnover, 0, enter], [leave, leave, 0]])
    labels = decode(scaled, transitions=transitions)
    # A genuine short turnover can survive if its evidence outweighs the two switches by a margin.
    return bridge_live_flickers(labels, scaled, fps, flicker, 2 * turnover + 5) if flicker else labels


def score_match(root, league, match, checkpoint, policies):
    model, _, lags = load_model(checkpoint)
    halves = []
    for positions, weights, teams, pitch, gt, _, fps in evaluation_inputs(root, league, match):
        p = probabilities(model, *tracking_features(positions, weights, teams, pitch, fps, lags))
        emission = np.log(np.maximum(p, 1e-6))
        runs = metrics.valid_runs(gt)
        scores = []
        for policy in policies:
            pred = metrics.label_codes(decode_policy(emission, fps, policy))
            spells = metrics.spell_scores(gt, pred, runs)
            cm = metrics.scored_confusion(metrics.confusion(gt, pred))
            scores.append([spells["mean_f1"], metrics.macro_f1(cm), spells["n_pred"], spells["n_gt"]])
        halves.append(scores)
    return np.mean(halves, axis=0)


def evaluate(pool, args, german, australian, policies):
    data = {"Bundesliga": (args.idsse, german), "A-League": (args.skillcorner, sorted(australian))}
    jobs = {league: [pool.submit(score_match, root, league, match, args.model, policies)
                     for match in matches] for league, (root, matches) in data.items()}
    return {league: np.mean([job.result() for job in tasks], axis=0) for league, tasks in jobs.items()}


def calibrate(args):
    _, costs, _ = load_model(args.model)
    if tuple(costs) != (5, 10):
        raise ValueError("expected the frozen lagged LightGBM decoder costs (5, 10)")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        development = evaluate(pool, args, IDSSE_DEV, DEV_MATCHES, POLICIES)
        balanced = np.mean(list(development.values()), axis=0)
        selected = max(range(len(POLICIES)), key=lambda n: (balanced[n, 0], -n))
        for policy, values in zip(POLICIES, balanced):
            print(f"Development {policy}: spellF1={values[0]:.6f}, macroF1={values[1]:.6f}", flush=True)
        print(f"Frozen external selection {POLICIES[selected]}", flush=True)
        test = evaluate(pool, args, IDSSE_TEST, TEST_MATCHES, [POLICIES[0], POLICIES[selected]])
    report = {"metrics": ["spell_f1", "macro_f1", "predicted_spells", "true_spells"],
              "policies": POLICIES, "selected": POLICIES[selected],
              "development": {league: values.tolist() for league, values in development.items()},
              "held_out_reference_and_selected": {league: values.tolist() for league, values in test.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Held-out reference and selected: {report['held_out_reference_and_selected']}")


def predict(args):
    model, _, lags = load_model(args.model)
    policy = tuple(json.loads(args.calibration.read_text())["selected"])
    if policy not in POLICIES:
        raise ValueError("unknown calibrated smoothing policy")
    x, teams, pitch, fps = contest_tracking(args.data_dir)
    labels = np.full(len(x), "X")
    for _, start, last in half_video_blocks(GAME):
        features = tracking_features(x[start:last + 1], None, teams, pitch, fps, lags)
        p = probabilities(model, *features)
        labels[start:last + 1] = decode_policy(np.log(np.maximum(p, 1e-6)), fps, policy)
    write_submission(args.output, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    external = commands.add_parser("calibrate")
    external.add_argument("--idsse", type=Path, required=True)
    external.add_argument("--skillcorner", type=Path, required=True)
    external.add_argument("--workers", type=int, default=4)
    contest = commands.add_parser("predict")
    contest.add_argument("--data-dir", type=Path)
    contest.add_argument("--calibration", type=Path, required=True)
    for command in (external, contest):
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args)
    else:
        predict(args)
