"""Run the external-only possession execution study."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import possession_metrics as m
from experiments.idsse_training import IDSSE_DEV, IDSSE_TEST, IDSSE_TRAIN
from external_possession import (
    DEV_MATCHES, IDSSE_FPS, SKILLCORNER_FPS, TEST_MATCHES, decode_costs,
    load_idsse, load_skillcorner,
)
from player_track_quality import repair_player_tracks
from sequence_possession import (
    DECODER_COSTS, restore, sequence_features, sequence_probabilities, visibility_features,
)

BASE = "7070119746e3a6b2b10b29cfde2401385f419263"
MIN_GAIN = 0.005
SEED = 413
STRENGTHS = (1.0, 2.0, 4.0)
ARMS = {"timing": (True, False), "team": (False, True), "both": (True, True)}


def fingerprint(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def dump(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def split_members(skillcorner):
    matches = {int(row["id"]) for row in json.loads((skillcorner / "matches.json").read_text())}
    train = sorted(matches - DEV_MATCHES - TEST_MATCHES)
    assert len(train) == 12 and (DEV_MATCHES | TEST_MATCHES) <= matches
    return {
        "train": train + list(IDSSE_TRAIN),
        "dev": sorted(DEV_MATCHES) + list(IDSSE_DEV),
        "test": sorted(TEST_MATCHES) + list(IDSSE_TEST),
    }


def dataset(match, skillcorner, idsse):
    if isinstance(match, str):
        return idsse, load_idsse, IDSSE_FPS
    return skillcorner, load_skillcorner, SKILLCORNER_FPS


def restart_cues(tracks, teams, pitch, fps):
    ball = tracks[:, -1]
    frames = np.arange(len(ball))
    delta = max(1, round(fps * 0.3))
    before = np.maximum(frames - delta, 0)
    after = np.minimum(frames + delta, len(ball) - 1)
    distance = np.linalg.norm(tracks[:, :-1] - ball[:, None], axis=-1)
    relative = ball / pitch
    geometry = np.minimum(relative, 1 - relative)
    centered = np.abs(relative - 0.5)
    pre_speed = np.linalg.norm(ball - ball[before], axis=-1) / 0.3
    post_speed = np.linalg.norm(ball[after] - ball, axis=-1) / 0.3
    pre_players = np.linalg.norm(tracks[:, :-1] - tracks[before, :-1], axis=-1) / 0.3
    post_players = np.linalg.norm(tracks[after, :-1] - tracks[:, :-1], axis=-1) / 0.3
    visible = np.isfinite(ball).all(axis=1)
    raw = visibility_features(ball, fps)
    edge = np.minimum(geometry[:, 0] * pitch[0], geometry[:, 1] * pitch[1])
    release = np.clip((post_speed - pre_speed - 1.0) / 6, 0, 1)
    team_features = []
    accelerations = []
    for team in (0, 1):
        indices = teams == team
        d = np.where(np.isfinite(distance[:, indices]), distance[:, indices], np.inf).min(axis=1)
        d[~np.isfinite(d)] = np.nan
        pre, post = pre_players[:, indices], post_players[:, indices]
        mean_pre = np.nansum(np.minimum(pre, 12), axis=1) / np.maximum(np.isfinite(pre).sum(axis=1), 1)
        mean_post = np.nansum(np.minimum(post, 12), axis=1) / np.maximum(np.isfinite(post).sum(axis=1), 1)
        accelerations.append(mean_post - mean_pre)
        team_features.append(np.column_stack((d[before], d, d[after], mean_pre, mean_post,
                                              np.isfinite(post).mean(axis=1))))
    collective = np.clip((np.mean(accelerations, axis=0) + 0.15) / 0.8, 0, 1)
    geometry_weight = np.where((edge < 3) | (centered.max(axis=1) < 0.08), 1.0, 0.55)
    timing = np.nan_to_num(release * geometry_weight * (0.4 + 0.6 * collective))
    common = np.column_stack((centered, geometry, raw, pre_speed, post_speed,
                              visible[before], visible[after]))
    a, b = team_features
    normal = np.column_stack((common, a, b)).astype(np.float32)
    flipped = np.column_stack((common, b, a)).astype(np.float32)
    return normal, flipped, timing, edge


def fixed(p, fps):
    return m.label_codes(decode_costs(np.log(np.maximum(p, 1e-6)), fps, *DECODER_COSTS))


def cache(args, members, roles):
    model, epoch = restore(args.gru, args.device)
    assert model.repair_players and model.continuous_motion
    print(f"Frozen continuous GRU epoch {epoch}", flush=True)
    weights_hash = fingerprint(args.gru)
    for role in roles:
        for match in members[role]:
            root, loader, fps = dataset(match, args.skillcorner, args.idsse)
            for half, (tracks, weights, teams, pitch, labels) in enumerate(loader(root, match), 1):
                destination = args.output / f"emissions-{match}-{half}.npz"
                track_hash = hashlib.sha256(memoryview(np.ascontiguousarray(tracks))).hexdigest()
                if destination.exists():
                    with np.load(destination) as prior:
                        if not (prior["weights_hash"] == weights_hash
                                and prior["track_hash"] == track_hash
                                and np.array_equal(prior["ground"], labels)
                                and np.array_equal(prior["grid"], [0, len(tracks), fps])):
                            raise ValueError(f"stale emission cache {destination}")
                    continue
                repaired = repair_player_tracks(tracks, fps)
                x, mirror = sequence_features(repaired, weights, teams, pitch, fps, True)
                p = sequence_probabilities(model, x, mirror, fps, args.device)
                normal, flipped, timing, edge = restart_cues(repaired, teams, pitch, fps)
                seen = np.isfinite(tracks[:, -1]).all(axis=1)
                temporary = destination.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, p=p, ground=labels, fps=fps, normal=normal,
                                    flipped=flipped, timing=timing, edge=edge, missing=~seen,
                                    track_hash=track_hash, weights_hash=weights_hash,
                                    grid=np.array([0, len(p), fps]), baseline=fixed(p, fps))
                temporary.replace(destination)
            print(f"Cached {role} match {match}", flush=True)


def rows(output, matches):
    for match in matches:
        for half in (1, 2):
            with np.load(output / f"emissions-{match}-{half}.npz") as item:
                yield match, half, dict(item)


def dead_intervals(ground):
    for start, stop, label in m.segments(ground, m.valid_runs(ground)):
        if label == m.D and stop < len(ground) and ground[stop] < m.D and start > 0:
            yield start, stop


def fit_restart(training):
    x, y, weights = [], [], []
    for _, _, row in training:
        for start, stop in dead_intervals(row["ground"]):
            samples = np.unique(np.linspace(max(start, stop - 2 * int(row["fps"])), stop,
                                            7, dtype=int))
            target = int(row["ground"][stop])
            x.extend((row["normal"][samples], row["flipped"][samples]))
            y.extend((np.full(len(samples), target), np.full(len(samples), 1 - target)))
            weights.extend(np.full(2 * len(samples), 1 / (2 * len(samples))))
    head = make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=250))
    head.fit(np.concatenate(x), np.concatenate(y), logisticregression__sample_weight=weights)
    return head, len(x) // 2


def restart_probability(head, row):
    return (head.predict_proba(row["normal"]) + head.predict_proba(row["flipped"])[:, ::-1]) / 2


def restart_decode(p, fps, timing, side, strength):
    """Decode H/A and two dead states whose exits correspond to the next team."""
    turnover, stoppage = DECODER_COSTS
    log_dead = np.log(np.clip(p[:, 2], 1e-6, 1))
    log_live = np.log(np.clip(p[:, :2].sum(axis=1), 1e-6, 1))
    conditional = p[:, :2] / np.maximum(p[:, :2].sum(axis=1, keepdims=True), 1e-6)
    evidence = np.column_stack((log_live[:, None] + np.log(np.clip(conditional, 1e-6, 1)),
                                log_dead, log_dead)) * (10 / fps)
    cost = np.array([[0, turnover, stoppage, stoppage], [turnover, 0, stoppage, stoppage],
                     [stoppage, np.inf, 0, np.inf], [np.inf, stoppage, np.inf, 0]])
    score = evidence[0].copy()
    previous = np.zeros(evidence.shape, dtype=np.int8)
    states = np.arange(4)
    # The team evidence is charged once, at the candidate restart. Uncertainty is neutral.
    bonus = strength * (timing[:, None] + np.log(np.clip(side * 2, 0.5, 1.5)))
    for i in range(1, len(p)):
        options = score[:, None] - cost
        options[2, 0] += bonus[i, 0]
        options[3, 1] += bonus[i, 1]
        origin = options.argmax(axis=0)
        best = options[origin, states]
        previous[i] = np.where(np.diag(options) == best, states, origin)
        score = best + evidence[i]
    result = np.empty(len(p), dtype=np.int8)
    result[-1] = score.argmax()
    for i in range(len(p) - 1, 0, -1):
        result[i - 1] = previous[i, result[i]]
    return np.minimum(result, m.D)


def measure(ground, pred, missing, edge, fps):
    runs = m.valid_runs(ground)
    cm = m.scored_confusion(m.confusion(ground, pred))
    inside = edge > 0
    cm_inside = m.scored_confusion(m.confusion(ground[inside], pred[inside]))
    targets, found = (m.possession_spells(values, runs) for values in (ground, pred))
    overlaps = m.spell_overlaps(targets, found)
    matched = m.matched_gt_flags(overlaps, len(targets), 0.5)
    # Count distinct same-team overlaps, including fragments below the matching threshold.
    fragments = np.bincount(np.asarray([pair[1] for pair in overlaps], dtype=int), minlength=len(targets))
    short = np.array([2 * fps <= b - a <= 4 * fps for a, b, _ in targets], dtype=bool)
    gap = np.array([missing[a:b].any() for a, b, _ in targets], dtype=bool)
    transitions = [m.transitions(values, runs) for values in (ground, pred)]
    tolerance = round(fps * 0.5)
    typed = m.transition_scores(*transitions, tolerance, typed="pair")
    restarts = [[row for row in group if row[1] == m.D] for group in transitions]
    pairs, missed, spurious = m.match_transitions(*restarts, tolerance, typed="none")
    offsets = np.array([(b[0] - a[0]) / fps for a, b in pairs])
    return {
        "spell_f1": m.spell_scores(ground, pred, runs)["mean_f1"],
        "macro_f1": m.macro_f1(cm), "dead_f1": m.per_class_prf(cm, m.D)[2],
        "in_pitch_dead_f1": m.per_class_prf(cm_inside, m.D)[2],
        "transition_f1": typed["f1"], "transition_mae_frames": typed["mean_abs_latency"],
        "spells": len(targets), "predicted_spells": len(found),
        "false_splits": int((fragments > 1).sum()),
        "short_spells": int(short.sum()), "short_missed": int((short & ~matched).sum()),
        "gap_spells": int(gap.sum()), "gap_hits": int((gap & matched).sum()),
        "restart_matched": len(pairs), "restart_missed": len(missed),
        "restart_spurious": len(spurious), "restart_team_correct": sum(a[2] == b[2] for a, b in pairs),
        "restart_early": int((offsets < 0).sum()), "restart_late": int((offsets > 0).sum()),
        "restart_absolute_seconds_sum": float(np.abs(offsets).sum()),
        "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
    }


def evaluate_restart(output, match_rows, head, configurations, name):
    scores = {key: {} for key in configurations}
    for match, half, row in match_rows:
        fps = int(row["fps"])
        p = row["p"]
        baseline = fixed(p, fps)
        assert np.array_equal(row["baseline"], baseline), (match, half)
        assert np.array_equal(restart_decode(p, fps, np.zeros(len(p)), np.full_like(p[:, :2], 0.5), 0),
                              baseline), (match, half, "neutral decoder must reproduce baseline")
        side = restart_probability(head, row)
        for key, (timing, team, strength) in configurations.items():
            pred = baseline if not strength else restart_decode(
                p, fps, row["timing"] if timing else np.zeros(len(p)),
                side if team else np.full_like(side, 0.5), strength,
            )
            score = measure(row["ground"], pred, row["missing"], row["edge"], fps)
            scores[key].setdefault(str(match), []).append(score)
        print(f"{name}: measured {match} half {half}", flush=True)
    dump(output / f"restart-{name}.json", scores)
    return scores


def averages(scores):
    return {key: float(np.mean([np.mean([h["spell_f1"] for h in halves])
                                for halves in matches.values()]))
            for key, matches in scores.items()}


def restart(args, members):
    head, n = fit_restart(rows(args.output, members["train"]))
    head_path = args.output / "restart-team.joblib"
    joblib.dump(head, head_path)
    configs = {"fixed": (False, False, 0)}
    configs.update({f"{arm}-{s:g}": (*flags, s) for arm, flags in ARMS.items() for s in STRENGTHS})
    dev = averages(evaluate_restart(args.output, rows(args.output, members["dev"]), head, configs, "dev"))
    winners = {arm: max((key for key in dev if key.startswith(arm + "-")), key=dev.get) for arm in ARMS}
    winner = max(winners.values(), key=dev.get)
    adopted = winner if dev[winner] - dev["fixed"] >= MIN_GAIN else "fixed"
    dump(args.output / "restart-selection.json",
         {"training_stoppages": n, "restart_team_sha256": fingerprint(head_path),
          "development": dev, "arm_winners": winners, "selected": adopted,
          "strongest_challenger": winner, "selected_config": configs[adopted]})
    print(f"Restart development {dev}; selected {adopted}", flush=True)
    keys = ["fixed", *winners.values()]
    cache(args, members, ("test",))
    test = averages(evaluate_restart(args.output, rows(args.output, members["test"]), head,
                                     {key: configs[key] for key in keys}, "test"))
    print(f"Frozen restart test {test}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cache", "restart"))
    for name in ("skillcorner", "idsse", "gru", "tcn", "detector", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    members = split_members(args.skillcorner)
    manifest = {
        "base": BASE, "splits": members, "minimum_spell_f1_gain": MIN_GAIN,
        "seed": SEED, "skillcorner_revision": "4340d274572876239c154c90bc507a9b3250a656",
        "idsse_release": "IDSSE seven-match release",
        "metric": {"implementation": "evaluation/possession_metrics.py",
                   "sha256": fingerprint("evaluation/possession_metrics.py"),
                   "typed_transition_tolerance_seconds": 0.5, "official_spell_ious": list(m.SPELL_IOUS)},
        "source_sha256": fingerprint(__file__), "decoder_costs": DECODER_COSTS,
        "checkpoint_sha256": {key: fingerprint(getattr(args, key)) for key in ("gru", "tcn", "detector")},
        "restart": {"strengths": STRENGTHS, "tags": "disabled; comparable external tags unavailable"},
        "command": sys.argv, "training_inputs": ["SkillCorner", "IDSSE"],
    }
    if args.stage == "cache":
        dump(args.output / "manifest.json", manifest)
        cache(args, members, ("train", "dev"))
    else:
        restart(args, members)


if __name__ == "__main__":
    main()
