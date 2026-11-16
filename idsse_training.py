"""Compare Bundesliga and pooled training with a frozen A-League reference."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from evaluation import possession_metrics as metrics
from external_possession import (
    DEV_MATCHES, IDSSE_FPS, IDSSE_MATCHES, LAG_SECONDS, SKILLCORNER_FPS, TEST_MATCHES,
    classifier, decode_costs, load_idsse, load_model, load_skillcorner, mask_ball,
    prepare_match, probabilities, tracking_features,
)

IDSSE_TRAIN = ("J03WMX", "J03WOH", "J03WPY", "J03WQQ")
IDSSE_DEV = ("J03WN1",)
IDSSE_TEST = ("J03WOY", "J03WR9")
DECODER_COSTS = tuple((turnover, stoppage) for turnover in (0, 2, 5, 10)
                      for stoppage in (0, 2, 5, 10))


def idsse_training_rows(root, match):
    fps = IDSSE_FPS
    rng = np.random.default_rng(int(match[1:], 36))
    result = []
    for positions, weights, teams, pitch, labels in load_idsse(root, match):
        selection = np.flatnonzero(labels != metrics.X)[::5]  # 5 Hz, as in the A-League fit.
        masked = mask_ball(positions, fps, rng, mean_gap=12, coverage=0.60)
        all_rows = []
        for tracks in (positions, masked):
            normal, mirrored = tracking_features(tracks, weights, teams, pitch, fps, LAG_SECONDS)
            all_rows.extend((normal[selection], mirrored[selection]))
        normal_labels = labels[selection]
        mirrored_labels = np.array([metrics.A, metrics.H, metrics.D])[normal_labels]
        result.append((np.concatenate(all_rows),
                       np.tile(np.r_[normal_labels, mirrored_labels], 2)))
    return result


def evaluation_inputs(root, league, match):
    if league == "Bundesliga":
        halves = load_idsse(root, match)
        fps = IDSSE_FPS
        rng = np.random.default_rng(int(match[1:], 36))
    else:
        halves = load_skillcorner(root, match)
        fps = SKILLCORNER_FPS
    for positions, weights, teams, pitch, labels in halves:
        ball = positions[:, -1]
        inside = ((ball > 0.5) & (ball < pitch - 0.5)).all(axis=1)
        if league == "Bundesliga":
            positions = mask_ball(positions, fps, rng, mean_gap=12, coverage=0.60)
        yield positions, weights, teams, pitch, labels, inside, fps


def prepare_evaluation(root, league, match):
    result = []
    for positions, weights, teams, pitch, labels, inside, fps in evaluation_inputs(root, league, match):
        normal, mirrored = tracking_features(positions, weights, teams, pitch, fps, LAG_SECONDS)
        result.append((normal, mirrored, labels, inside, fps))
    return result


def score_match(model, halves, costs):
    result = []
    for normal, mirrored, labels, inside, fps in halves:
        p = probabilities(model, normal, mirrored)
        emission = np.log(np.maximum(p, 1e-6))
        runs = metrics.valid_runs(labels)
        scores = []
        for pair in costs:
            pred = metrics.label_codes(decode_costs(emission, fps, *pair))
            spell = metrics.spell_scores(labels, pred, runs)["mean_f1"]
            cm = metrics.scored_confusion(metrics.confusion(labels, pred))
            inside_cm = metrics.scored_confusion(metrics.confusion(labels[inside], pred[inside]))
            scores.append((spell, metrics.macro_f1(cm), metrics.per_class_prf(cm, metrics.D)[2],
                           metrics.per_class_prf(inside_cm, metrics.D)[2]))
        result.append(scores)
    return np.mean(result, axis=0)


def evaluate(pool, model, datasets, costs):
    jobs = {league: [pool.submit(score_match, model, halves, costs) for halves in matches]
            for league, matches in datasets.items()}
    return {league: np.stack([task.result() for task in tasks]) for league, tasks in jobs.items()}


def fit(x, y, workers, weights=None):
    model = classifier("lightgbm", workers)
    with threadpool_limits(limits=workers):
        model.fit(x, y, sample_weight=weights)
    return model


def load_split(pool, idsse, skillcorner, bundesliga_ids, a_league_ids):
    jobs = {
        "Bundesliga": [pool.submit(prepare_evaluation, idsse, "Bundesliga", match)
                       for match in bundesliga_ids],
        "A-League": [pool.submit(prepare_evaluation, skillcorner, "A-League", match)
                     for match in sorted(a_league_ids)],
    }
    return {league: [task.result() for task in tasks] for league, tasks in jobs.items()}


def compare(idsse, skillcorner, reference_path, output, workers):
    idsse_split = IDSSE_TRAIN + IDSSE_DEV + IDSSE_TEST
    if sorted(idsse_split) != sorted(IDSSE_MATCHES):
        raise ValueError("Bundesliga split must include each external match exactly once")
    reference, reference_costs, lags = load_model(reference_path)
    if lags != LAG_SECONDS:
        raise ValueError("reference model must use the frozen lag settings")
    match_ids = {int(r["id"]) for r in json.loads((skillcorner / "matches.json").read_text())}
    if not (DEV_MATCHES | TEST_MATCHES).issubset(match_ids):
        raise ValueError("pinned SkillCorner development or test matches are missing")
    train_ids = sorted(match_ids - DEV_MATCHES - TEST_MATCHES)
    output.mkdir(parents=True, exist_ok=True)
    # The reference is loaded before workers start. Spawn to avoid copying the
    # initialized LightGBM/OpenMP state into scoring workers.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        german_jobs = [pool.submit(idsse_training_rows, idsse, match) for match in IDSSE_TRAIN]
        australian_jobs = [pool.submit(prepare_match, skillcorner, match, True, LAG_SECONDS)
                           for match in train_ids]
        german = [half for task in german_jobs for half in task.result()]
        x_de = np.concatenate([x for x, _ in german])
        y_de = np.concatenate([y for _, y in german])
        del german, german_jobs
        print(f"Bundesliga fit: {len(y_de):,} rows, H/A/D counts {np.bincount(y_de).tolist()}", flush=True)
        german_model = fit(x_de, y_de, workers)
        australian = [half for task in australian_jobs for half in task.result()]
        x = np.concatenate([x_de, *[x for x, _ in australian]])
        y = np.concatenate([y_de, *[y for _, y in australian]])
        n_de = len(y_de)
        del x_de, y_de, australian, australian_jobs
        print(f"Combined fit: {len(y):,} rows, H/A/D counts {np.bincount(y).tolist()}", flush=True)
        # Equal total sample weight per source keeps the larger set from dominating.
        weights = np.full(len(y), len(y) / (2 * (len(y) - n_de)), dtype=np.float32)
        weights[:n_de] = len(y) / (2 * n_de)
        combined_model = fit(x, y, workers, weights)
        del x, y, weights

        development = load_split(pool, idsse, skillcorner, IDSSE_DEV, DEV_MATCHES)
        trained = {"bundesliga": german_model, "combined": combined_model}
        checkpoints = {"a-league": (reference, reference_costs, reference_path)}
        report = {}
        for name, model in {"a-league": reference, **trained}.items():
            costs = (reference_costs,) if name == "a-league" else DECODER_COSTS
            scores = evaluate(pool, model, development, costs)
            league_means = {league: numbers.mean(axis=0) for league, numbers in scores.items()}
            average = np.mean(list(league_means.values()), axis=0)
            index = max(range(len(costs)), key=lambda i: average[i, 0])
            selected = costs[index]
            report[name] = {"costs": selected, "development_balance": float(average[index, 0]),
                            "development": {league: values[index].tolist()
                                            for league, values in league_means.items()}}
            if name in trained:
                path = output / f"{name}.joblib"
                joblib.dump((model, selected, LAG_SECONDS), path)
                checkpoints[name] = (model, selected, path)
            print(f"Development {name}: {report[name]}", flush=True)
        winner = max(report, key=lambda name: report[name]["development_balance"])
        print(f"Frozen development selection: {winner} ({checkpoints[winner][2]})", flush=True)
        del development

        test = load_split(pool, idsse, skillcorner, IDSSE_TEST, TEST_MATCHES)
        for name, (model, costs, _) in checkpoints.items():
            results = evaluate(pool, model, test, [costs])
            report[name]["test"] = {league: scores[:, 0].mean(axis=0).tolist()
                                     for league, scores in results.items()}
            print(f"Held-out {name}: {report[name]['test']}", flush=True)
    result = {"metrics": ["spell_f1", "macro_f1", "dead_f1", "in_pitch_dead_f1"],
              "selected": winner, "models": report}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idsse", type=Path, required=True)
    parser.add_argument("--skillcorner", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    compare(args.idsse, args.skillcorner, args.reference, args.output, args.workers)
