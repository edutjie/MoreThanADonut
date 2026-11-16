"""Evaluate player-motion evidence learned from external stoppage intervals."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier

from evaluation import possession_metrics as metrics
from external_possession import (
    DEV_MATCHES, IDSSE_FPS, SKILLCORNER_FPS, TEST_MATCHES, decode_costs,
    load_idsse, load_model, load_skillcorner, probabilities, tracking_features,
)
from idsse_training import IDSSE_DEV, IDSSE_TEST
from possession_sources import METRICA_FPS, METRICA_MATCHES, load_metrica


SOCCERTRACK_MATCHES = (117092, 118575)
SOCCERTRACK_SECONDS = 600
BLENDS = (0.0, 0.25, 0.5, 1.0)


def motion_features(players, teams, fps):
    """Summarize player motion symmetrically without inspecting the ball."""
    frames = np.arange(len(players))
    columns = []
    for seconds in (0.2, 0.6, 1.5, 4.0):
        width = round(fps * seconds)
        for offset in (-width, width):
            other = np.clip(frames + offset, 0, len(players) - 1)
            elapsed = np.maximum(abs(other - frames), 1) / fps
            speed = np.linalg.norm(players[other] - players, axis=-1) / elapsed[:, None]
            summaries = []
            for team in (0, 1):
                v = speed[:, teams == team]
                count = np.isfinite(v).sum(axis=1).astype(float)
                count[count == 0] = np.nan
                summaries.append(np.column_stack((np.nansum(np.minimum(v, 12), axis=1) / count,
                                                  (v < 1).sum(axis=1) / count,
                                                  (v > 3).sum(axis=1) / count)))
            columns.extend((np.minimum(*summaries), np.maximum(*summaries)))
    return np.column_stack(columns).astype(np.float32)


def soccertrack_labels(events, seconds, fps, stride):
    labels = np.full(seconds * fps, metrics.X, dtype=np.int8)
    for first, second in zip(events, events[1:]):
        left, right = first["frame"] - 1, second["frame"] - 1
        if not (0 <= left < right < len(labels)):
            continue
        interval = (right - left) / fps
        if first["label"] == "OUT" and second["label"] == "THROW IN" and 1 <= interval <= 60:
            labels[left + fps // 5:right - fps // 5] = metrics.D
        elif first["label"] == "PASS" and second["label"] in ("PASS", "DRIVE"):
            if first["team"] == second["team"] and interval <= 2:
                labels[left:right] = metrics.H  # motion model uses only live versus dead
    return labels[::stride]


def soccertrack_halves(root, match):
    import ijson

    events = json.loads((root / "bas" / str(match) / f"{match}_12_class_events.json").read_text())
    fps = int(events["fps"])
    stride = fps // 5
    for period, suffix in ((1, "1st"), (2, "2nd")):
        labels = soccertrack_labels([e for e in events["actions"]
                                     if e["gameTime"].startswith(f"{period} -")],
                                    SOCCERTRACK_SECONDS, fps, stride)
        filename = root / "gsr" / str(match) / f"{match}_{suffix}.json"
        if not filename.exists():
            filename = filename.with_suffix(".json.part")
        tracks, teams = {}, {}
        with filename.open("rb") as handle:
            for row in ijson.items(handle, "annotations.item", use_float=True):
                frame = int(row["image_id"][-6:]) - 1
                if frame >= SOCCERTRACK_SECONDS * fps:
                    break
                if frame < 0 or frame % stride or row["supercategory"] != "object":
                    continue
                actor = row["attributes"]
                if actor["role"] not in ("player", "goalkeeper") or not row.get("bbox_pitch"):
                    continue
                track = row["track_id"]
                if track not in tracks:
                    tracks[track] = np.full((len(labels), 2), np.nan, dtype=np.float32)
                    teams[track] = {"left": 0, "right": 1}[actor["team"]]
                xy = row["bbox_pitch"]
                tracks[track][frame // stride] = xy["x_bottom_middle"], xy["y_bottom_middle"]
            else:
                raise ValueError(f"Missing tracking coverage for first {SOCCERTRACK_SECONDS}s: {filename}")
        positions = np.stack(list(tracks.values()), axis=1)
        teams = np.array(list(teams.values()))
        visible = np.isfinite(positions).all(axis=-1)
        covered = (visible[:, teams == 0].sum(axis=1) >= 5) & (visible[:, teams == 1].sum(axis=1) >= 5)
        if covered.mean() < 0.9:
            raise ValueError(f"Only {covered.mean():.1%} SoccerTrack player coverage in {filename}")
        labels[~covered] = metrics.X
        print(f"SoccerTrack {match} half {period}: {covered.sum()}/{len(covered)} covered frames",
              flush=True)
        yield motion_features(positions, teams, fps / stride), labels


def metrica_training(root):
    for match in METRICA_MATCHES:
        for positions, observed, teams, pitch, labels in load_metrica(root, match):
            print(f"Metrica {match}: {np.bincount(labels, minlength=4).tolist()}, "
                  f"observed D ball frames {int(observed[labels == metrics.D].sum())}", flush=True)
            yield motion_features(positions[:, :-1], teams, METRICA_FPS)[::5], labels[::5]


def fit_expert(rows):
    x = np.concatenate([x[y != metrics.X] for x, y in rows])
    target = np.concatenate([y[y != metrics.X] == metrics.D for _, y in rows])
    model = LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.05, n_jobs=4,
                           deterministic=True, force_col_wise=True, verbosity=-1)
    model.fit(x, target)
    return model, float(np.log(target.mean() / (1 - target.mean())))


def expert_evidence(expert, motions):
    model, prior = expert
    p = model.predict_proba(motions)[:, 1].clip(1e-5, 1 - 1e-5)
    return (np.log(p / (1 - p)) - prior).clip(-3, 3)


def score_grid(root, matches, model, costs, lags, experts):
    idsse = isinstance(matches[0], str)
    load = load_idsse if idsse else load_skillcorner
    fps = IDSSE_FPS if idsse else SKILLCORNER_FPS
    rows = []
    for match in matches:
        halves = []
        for positions, weights, teams, pitch, labels in load(root, match):
            ball = positions[:, -1]
            inside = ((ball > 0.5) & (ball < pitch - 0.5)).all(axis=1) & (weights == 1)
            logp = np.log(probabilities(model, *tracking_features(
                positions, weights, teams, pitch, fps, lags)).clip(1e-6))
            motions = motion_features(positions[:, :-1], teams, fps)
            runs = metrics.valid_runs(labels)
            scores = {}
            for name, expert in experts.items():
                adjustment = expert_evidence(expert, motions)
                values = []
                for strength in BLENDS:
                    emission = logp.copy()
                    emission[:, metrics.D] += strength * adjustment
                    pred = metrics.label_codes(decode_costs(emission, fps, *costs))
                    cm = metrics.scored_confusion(metrics.confusion(labels, pred))
                    inside_cm = metrics.scored_confusion(metrics.confusion(labels[inside], pred[inside]))
                    values.append((metrics.spell_scores(labels, pred, runs)["mean_f1"],
                                   metrics.macro_f1(cm), metrics.per_class_prf(cm, metrics.D)[2],
                                   metrics.per_class_prf(inside_cm, metrics.D)[2]))
                scores[name] = values
            halves.append(scores)
        rows.append({name: np.mean([half[name] for half in halves], axis=0) for name in experts})
    return {name: np.mean([row[name] for row in rows], axis=0) for name in experts}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    metrica = list(metrica_training(args.metrica))
    experts = {"Metrica": fit_expert(metrica)}
    if args.soccertrack:
        soccertrack = [half for match in SOCCERTRACK_MATCHES for half in
                       soccertrack_halves(args.soccertrack, match)]
        print("SoccerTrack 5 Hz rows", [np.bincount(y, minlength=4).tolist() for _, y in soccertrack],
              flush=True)
        experts["Metrica + SoccerTrack"] = fit_expert(metrica + soccertrack)
    ref, costs, lags = load_model(args.reference)
    dev = [score_grid(args.skillcorner, sorted(DEV_MATCHES), ref, costs, lags, experts),
           score_grid(args.idsse, IDSSE_DEV, ref, costs, lags, experts)]
    selected = {name: int(np.mean([d[name][:, 3] for d in dev], axis=0).argmax())
                for name in experts}
    test = [score_grid(args.skillcorner, sorted(TEST_MATCHES), ref, costs, lags, experts),
            score_grid(args.idsse, IDSSE_TEST, ref, costs, lags, experts)]
    report = {
        name: {"strength": BLENDS[index],
               "development_in_pitch_dead_f1": [d[name][:, 3].tolist() for d in dev],
               "test_baseline": [d[name][0].tolist() for d in test],
               "test_candidate": [d[name][index].tolist() for d in test]}
        for name, index in selected.items()
    }
    joblib.dump((experts, {name: BLENDS[index] for name, index in selected.items()}),
                args.output / "experts.joblib")
    text = json.dumps(report, indent=2)
    (args.output / "results.json").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("skillcorner", "idsse", "reference", "metrica", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--soccertrack", type=Path)
    run(parser.parse_args())
