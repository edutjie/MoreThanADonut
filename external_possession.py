"""Train and evaluate a tracking-only possession model on SkillCorner A-League games."""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from evaluation import possession_metrics as metrics
from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track, write_submission
from tactical_baseline import LABELS, decode, fill_short_gaps, read_tracking, tracking_evidence


SKILLCORNER_FPS = 10
DEV_MATCHES = {1886347, 1959846, 2006363, 2013725}
TEST_MATCHES = {1925299, 1996435, 2007448, 2017461}
IDSSE_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
IDSSE_FPS = 25


def tracking_features(positions, weights, teams, pitch, fps):
    """Return shared features and the same team features in home/away order."""
    ball, confidence = fill_short_gaps(positions[:, -1], fps, weights)
    players = positions[:, :-1]
    relative = players - ball[:, None]
    distance = np.linalg.norm(relative, axis=-1)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    outside = np.maximum(-ball, ball - pitch).max(axis=1)
    visible = np.isfinite(players).all(axis=-1)
    frames = np.arange(len(ball))

    shared = [confidence, outside, distance.min(axis=1)]
    team_features = [[], []]
    for side, features in enumerate(team_features):
        side_distance = np.sort(distance[:, teams == side], axis=1)
        features.extend([side_distance[:, 0], side_distance[:, 1],
                         (side_distance < 5).sum(axis=1), visible[:, teams == side].sum(axis=1)])

    for seconds in (0.2, 0.6, 1.5, 4.0):
        window = max(1, round(fps * seconds))
        shared.append(uniform_filter1d(confidence, size=window, mode="nearest"))
        for other in (np.maximum(0, frames - window), np.minimum(len(ball) - 1, frames + window)):
            elapsed = np.maximum(np.abs(other - frames), 1) / fps
            ball_speed = np.linalg.norm(ball[other] - ball, axis=-1) / elapsed
            shared.append(ball_speed)
            player_speed = np.linalg.norm(players[other] - players, axis=-1) / elapsed[:, None]
            for side, features in enumerate(team_features):
                near = features[0]
                side_speed = player_speed[:, teams == side]
                count = np.isfinite(side_speed).sum(axis=1)
                mean_speed = np.nansum(np.minimum(side_speed, 12), axis=1) / np.maximum(count, 1)
                features.extend([near[other], mean_speed])

    shared = np.column_stack(shared).astype(np.float32)
    home, away = (np.column_stack(features).astype(np.float32) for features in team_features)
    normal = np.column_stack((shared, home, away))
    mirrored = np.column_stack((shared, away, home))
    normal[~np.isfinite(normal)] = np.nan
    mirrored[~np.isfinite(mirrored)] = np.nan
    return normal, mirrored


def load_skillcorner(root, match_id):
    meta = json.loads((root / f"{match_id}_match.json").read_text())
    pitch = np.array([meta["pitch_length"], meta["pitch_width"]], dtype=np.float32)
    side_by_id = {meta[f"{side}_team"]["id"]: i for i, side in enumerate(("home", "away"))}
    players = [player for player in meta["players"] if player["team_id"] in side_by_id]
    player_slot = {player["id"]: i for i, player in enumerate(players)}
    teams = np.array([side_by_id[player["team_id"]] for player in players])
    periods = {entry["period"]: (entry["start_frame"], entry["end_frame"])
               for entry in meta["match_periods"] if entry["period"] in (1, 2)}
    if set(periods) != {1, 2}:
        raise ValueError(f"{match_id}: expected two regulation periods")
    positions = {period: np.full((end - start, len(teams) + 1, 2), np.nan, dtype=np.float32)
                 for period, (start, end) in periods.items()}
    present = {period: np.zeros(len(values), dtype=bool) for period, values in positions.items()}
    weights = {period: np.ones(len(values), dtype=np.float32) for period, values in positions.items()}
    with (root / f"{match_id}_tracking_extrapolated.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            period, frame = row["period"], row["frame"]
            if period not in periods:
                continue
            start, end = periods[period]
            if not start <= frame < end:
                continue
            index = frame - start
            present[period][index] = True
            values = positions[period][index]
            for player in row["player_data"]:
                slot = player_slot.get(player["player_id"])
                if slot is not None:
                    values[slot] = (player["x"], player["y"])
            ball = row["ball_data"]
            values[-1] = (ball["x"], ball["y"])
            weights[period][index] = 1.0 if ball["is_detected"] else 0.55

    phases = {period: [] for period in periods}
    with (root / f"{match_id}_phases_of_play.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            period = int(row["period"])
            if period in periods:
                phases[period].append((int(row["frame_start"]), int(row["frame_end"]),
                                       side_by_id[int(row["team_in_possession_id"])]))

    for period, (start, end) in sorted(periods.items()):
        intervals = sorted((max(left, start), min(right, end), side)
                           for left, right, side in phases[period]
                           if min(right, end) > max(left, start))
        if not intervals or not present[period].any():
            raise ValueError(f"{match_id}: missing phases or tracking in period {period}")
        labels = np.full(end - start, metrics.X, dtype=np.int8)
        # Only gaps bracketed by live phases are labeled as stoppages.
        labels[intervals[0][0] - start:intervals[-1][1] - start] = metrics.D
        previous_end = start
        for left, right, side in intervals:
            if left < previous_end:
                raise ValueError(f"{match_id}: overlapping phase at {left} in period {period}")
            labels[left - start:right - start] = side
            previous_end = right
        labels[~present[period]] = metrics.X
        yield positions[period] + pitch / 2, weights[period], teams, pitch, labels


def load_idsse(root, match_id):
    meta_path, = root.glob(f"DFL_02_01_matchinformation_*-{match_id}.xml")
    tracking_path, = root.glob(f"DFL_04_03_positions_raw_observed_*-{match_id}.xml")
    meta = ET.parse(meta_path)
    env = meta.find(".//Environment")
    pitch = np.array([env.get("PitchX"), env.get("PitchY")], dtype=np.float32)
    general = meta.find(".//General")
    sides = {general.get("HomeTeamId"): 0, general.get("GuestTeamId"): 1}
    slot_by_id = {}
    teams = []
    for team in meta.findall(".//Team"):
        if team.get("TeamId") in sides:
            for player in team.findall(".//Player"):
                slot_by_id[player.get("PersonId")] = len(teams)
                teams.append(sides[team.get("TeamId")])
    tracks = {"firstHalf": {}, "secondHalf": {}}
    # FrameSets are player-major. Read one at a time to bound XML memory use.
    for _, node in ET.iterparse(tracking_path):
        if node.tag != "FrameSet":
            continue
        section = tracks.get(node.get("GameSection"))
        slot = slot_by_id.get(node.get("PersonId"))
        if section is not None and len(node):
            is_ball = node[0].get("BallStatus") is not None
            if is_ball or slot is not None:
                key = len(teams) if is_ball else slot
                rows = section.setdefault(key, [])
                for frame in node:
                    label = metrics.X
                    if is_ball:
                        if frame.get("BallStatus") == "0":
                            label = metrics.D
                        elif frame.get("BallStatus") == "1":
                            label = {"1": metrics.H, "2": metrics.A}.get(
                                frame.get("BallPossession"), metrics.X)
                    rows.append((int(frame.get("N")), float(frame.get("X")), float(frame.get("Y")), label))
        node.clear()
    teams = np.array(teams)
    for name, section in tracks.items():
        ball = np.array(section[len(teams)])
        start, end = int(ball[:, 0].min()), int(ball[:, 0].max()) + 1
        positions = np.full((end - start, len(teams) + 1, 2), np.nan, dtype=np.float32)
        labels = np.full(end - start, metrics.X, dtype=np.int8)
        for slot, frames in section.items():
            rows = np.array(frames)
            index = rows[:, 0].astype(int) - start
            inside = (index >= 0) & (index < len(labels))
            positions[index[inside], slot] = rows[inside, 1:3]
        labels[ball[:, 0].astype(int) - start] = ball[:, 3]
        if not {metrics.H, metrics.A, metrics.D}.issubset(np.unique(labels)):
            raise ValueError(f"{match_id}: missing possession or stoppage labels in {name}")
        yield positions + pitch / 2, np.ones(len(labels)), teams, pitch, labels


def mask_ball(positions, fps, rng, mean_gap, coverage):
    """Simulate bursty camera dropouts without using labels."""
    masked = positions.copy()
    mean_visible = mean_gap * coverage / (1 - coverage)
    start = round(rng.exponential(mean_visible) * fps)
    while start < len(masked):
        stop = start + max(1, round(rng.exponential(mean_gap) * fps))
        masked[start:stop, -1] = np.nan
        start = stop + max(1, round(rng.exponential(mean_visible) * fps))
    return masked


def prepare_match(root, match_id, fit):
    rng = np.random.default_rng(match_id)
    fps = SKILLCORNER_FPS
    halves = []
    for positions, weights, teams, pitch, labels in load_skillcorner(root, match_id):
        normal, mirrored = tracking_features(positions, weights, teams, pitch, fps)
        if fit:
            selected = np.flatnonzero(labels != metrics.X)[::2]
            # Broadcast tracking covers more of the ball trajectory than the target.
            # Synthetic, label-independent dropouts expose the model to blind stretches.
            dropout = positions.copy()
            starts = rng.choice(len(positions), size=len(positions) // 80, replace=False)
            lengths = rng.integers(fps // 2, fps * 5 + 1, size=len(starts))
            for start, length in zip(starts, lengths):
                dropout[start:start + length, -1] = np.nan
            hidden, hidden_mirror = tracking_features(dropout, weights, teams, pitch, fps)
            both_sides = np.array([1, 0, 2, 3], dtype=np.int8)[labels[selected]]
            halves.append((np.concatenate((normal[selected], mirrored[selected],
                                           hidden[selected], hidden_mirror[selected])),
                           np.concatenate((labels[selected], both_sides,
                                           labels[selected], both_sides))))
        else:
            ball, confidence = fill_short_gaps(positions[:, -1], fps, weights)
            evidence = tracking_evidence(positions, ball, confidence, teams, pitch, fps)[0]
            halves.append((normal, mirrored, labels, evidence, fps))
    return halves


def probabilities(model, normal, mirrored):
    return (model.predict_proba(normal) + model.predict_proba(mirrored)[:, [1, 0, 2]]) / 2


def decode_costs(evidence, fps, turnover, stoppage):
    transitions = np.array([[0, turnover, stoppage], [turnover, 0, stoppage],
                            [stoppage, stoppage, 0]])
    return decode(evidence * (10 / fps), transitions=transitions)


def spell_f1(labels, pred):
    runs = metrics.valid_runs(labels)
    return metrics.spell_scores(labels, metrics.label_codes(pred), runs)["mean_f1"]


def measure(model, matches, costs):
    for match_id, halves in matches:
        baseline, learned = [], []
        for normal, mirrored, labels, evidence, fps in halves:
            baseline.append(spell_f1(labels, decode(evidence * (25 / fps))))
            emission = np.log(np.maximum(probabilities(model, normal, mirrored), 1e-6))
            learned.append([spell_f1(labels, decode_costs(emission, fps, *pair))
                            for pair in costs])
        yield match_id, np.mean(baseline), np.mean(learned, axis=0)


def classifier(backend, workers):
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(objective="multiclass", n_estimators=100, num_leaves=31,
                              learning_rate=0.1, random_state=0, n_jobs=workers,
                              deterministic=True, force_col_wise=True, verbosity=-1)
    if backend == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(objective="multi:softprob", n_estimators=100, max_depth=0,
                             max_leaves=31, grow_policy="lossguide", learning_rate=0.1,
                             tree_method="hist", random_state=0, n_jobs=workers)
    if backend == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(loss_function="MultiClass", iterations=100, depth=5,
                                  learning_rate=0.1, random_seed=0, thread_count=workers,
                                  verbose=False, allow_writing_files=False)
    return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31,
                                          early_stopping=False, random_state=0)


def train(root, output, workers, backend="hist"):
    match_ids = sorted(int(match["id"]) for match in json.loads((root / "matches.json").read_text()))
    if not (TEST_MATCHES | DEV_MATCHES).issubset(match_ids):
        raise ValueError("pinned SkillCorner development or test matches are missing")
    train_ids = sorted(set(match_ids) - DEV_MATCHES - TEST_MATCHES)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare_match, root, match_id, True) for match_id in train_ids]
        fitted = [half for future in futures for half in future.result()]
        features = np.concatenate([x for x, _ in fitted])
        labels = np.concatenate([y for _, y in fitted])
        print(f"Training {backend} on {len(features):,} rows from matches {train_ids}", flush=True)
        model = classifier(backend, workers)
        with threadpool_limits(limits=workers):
            model.fit(features, labels)
        del fitted, features, labels, futures
        costs = [(turnover, stoppage) for turnover in (0, 2, 5, 10)
                 for stoppage in (0, 2, 5, 10)]
        dev_jobs = [(match_id, pool.submit(prepare_match, root, match_id, False))
                    for match_id in sorted(DEV_MATCHES)]
        with threadpool_limits(limits=workers):
            dev = list(measure(model, ((i, job.result()) for i, job in dev_jobs), costs))
            means = np.mean([row[2] for row in dev], axis=0)
            best = int(means.argmax())
            selected = costs[best]
            print(f"Development costs and spellF1: {list(zip(costs, means.round(4).tolist()))}")
            print(f"Frozen costs {selected}; dev v5 {np.mean([row[1] for row in dev]):.4f}; "
                  f"dev learned {means[best]:.4f}", flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump((model, selected), output)
            test_jobs = [(match_id, pool.submit(prepare_match, root, match_id, False))
                         for match_id in sorted(TEST_MATCHES)]
            test = list(measure(model, ((i, job.result()) for i, job in test_jobs), [selected]))
            for match_id, baseline, learned in test:
                print(f"Test match {match_id}: v5 {baseline:.4f}, learned {learned[0]:.4f}")
            print(f"Test mean: v5 {np.mean([row[1] for row in test]):.4f}, "
                  f"learned {np.mean([row[2][0] for row in test]):.4f}")


def check_transfer(root, model_path):
    model, costs = joblib.load(model_path)
    fps = IDSSE_FPS
    scores = []
    for match_id in IDSSE_MATCHES:
        rng = np.random.default_rng(int(match_id[1:], 36))
        halves = []
        for positions, weights, teams, pitch, labels in load_idsse(root, match_id):
            masked = mask_ball(positions, fps, rng, mean_gap=12, coverage=0.60)
            normal, mirrored = tracking_features(masked, weights, teams, pitch, fps)
            ball, confidence = fill_short_gaps(masked[:, -1], fps, weights)
            evidence = tracking_evidence(masked, ball, confidence, teams, pitch, fps)[0]
            halves.append((normal, mirrored, labels, evidence, fps))
        _, baseline, learned = next(measure(model, [(match_id, halves)], [costs]))
        print(f"Transfer match {match_id}: v5 {baseline:.4f}, learned {learned[0]:.4f}", flush=True)
        scores.append((baseline, learned[0]))
    baseline, learned = np.mean(scores, axis=0)
    print(f"Transfer mean: v5 {baseline:.4f}, learned {learned:.4f}")


def predict(model_path, data_dir, output):
    model, costs = joblib.load(model_path)
    meta, raw = require(["atd_meta", "atd_raw"], data_dir=data_dir)
    teams = np.asarray(team_of_each_track(meta))
    fps = GAMES[GAME]["fps"]
    pitch = np.asarray(GAMES[GAME]["pitch"], dtype=np.float32)
    n_frames = source_frames()
    positions = read_tracking(raw, n_frames, len(teams), pitch)
    labels = np.full(n_frames, "X")
    for _, start, last in half_video_blocks(GAME):
        half = positions[start:last + 1]
        x, mirrored = tracking_features(half, None, teams, pitch, fps)
        emission = np.log(np.maximum(probabilities(model, x, mirrored), 1e-6))
        labels[start:last + 1] = decode_costs(emission, fps, *costs)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(output, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    training = subcommands.add_parser("train")
    training.add_argument("--skillcorner", type=Path, required=True)
    training.add_argument("--model", type=Path, required=True)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--backend", choices=["hist", "lightgbm", "xgboost", "catboost"], default="hist")
    inference = subcommands.add_parser("predict")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--data-dir", type=Path)
    inference.add_argument("--output", type=Path, required=True)
    transfer = subcommands.add_parser("check-transfer")
    transfer.add_argument("--idsse", type=Path, required=True)
    transfer.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.skillcorner, args.model, args.workers, args.backend)
    elif args.command == "predict":
        predict(args.model, args.data_dir, args.output)
    else:
        check_transfer(args.idsse, args.model)


if __name__ == "__main__":
    main()
