"""Select possession recipes on external development and confirm them in one holdout pass."""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from evaluation import possession_metrics as m
from experiments.possession_execution import dataset, dump, fingerprint, measure
from experiments.possession_round2_adapt import normalized_model, prediction, self_train, target_stats
from experiments.possession_round2_decoder import Decoder
from experiments.possession_round2_segments import (
    duration_prior, fit_duration, fit_touch, likelihood, snap, touch_features, touch_inputs,
)
from experiments.possession_round2_train import (
    ARMS, BOOTSTRAP, CONDITIONS, COSTS, MAX_A_DROP, MAX_DEAD_DROP, MIN_GAIN, SEEDS,
    STOPPAGE_GRID, TURNOVER_GRID, arguments, cached_emissions, halves, load, pin, score, train_data,
)
from player_track_quality import repair_player_tracks
from sequence_possession import sequence_features, sequence_probabilities


@dataclass(frozen=True)
class Source:
    paths: tuple
    weights: tuple
    space: str = "prob"

    def spec(self):
        return {"paths": [str(p) for p in self.paths], "weights": self.weights, "space": self.space,
                "sha256": [fingerprint(p) for p in self.paths]}


def mix(arrays, weights, space):
    if len(arrays) == 1:
        return arrays[0]
    if space == "log":
        result = np.exp(np.average([np.log(np.maximum(p, 1e-6)) for p in arrays], weights=weights, axis=0))
    else:
        result = np.average(arrays, weights=weights, axis=0)
    return result / result.sum(axis=1, keepdims=True)


def combined(bank, source, key):
    return mix([bank[path][key] for path in source.paths], source.weights, source.space)


def plain(path):
    return Source((path,), (1,))


def dev_mean(rows):
    by_match = {}
    for (match, _), value in rows.items():
        by_match.setdefault(match, []).append(value)
    return float(np.mean([np.mean(halves) for halves in by_match.values()]))


def stats(results, control=None):
    # Retain pairing by match, and use the same bootstrap resample for every arm.
    by_match = {}
    for results_by_seed in results:
        games = {}
        for (match, _), value in results_by_seed.items():
            games.setdefault(match, []).append(value)
        for match, halves_ in games.items():
            by_match.setdefault(match, []).append(np.mean(halves_))
    keys = list(by_match)
    values = np.array(list(by_match.values()))
    seeds = values.mean(axis=0)
    deltas = values.copy()
    if control:
        deltas -= np.array([control["seed_results_by_match"][str(match)] for match in keys])
    per_match = deltas.mean(axis=1)
    rng = np.random.default_rng(20261117)
    resampled = rng.integers(len(per_match), size=(BOOTSTRAP, len(per_match)))
    interval = np.percentile(per_match[resampled].mean(axis=1), [2.5, 97.5])
    a_league = np.array([isinstance(match, int) for match in keys])
    return {
        "mean": float(seeds.mean()), "seed_sd": float(seeds.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "seed_results_by_match": {str(k): v for k, v in zip(keys, values.tolist())},
        "aleague_mean": float(values[a_league].mean()), "idsse_mean": float(values[~a_league].mean()),
        "comparison_mean": float(per_match.mean()), "comparison_ci95": interval.tolist(),
        "paired_seed_sd": float(deltas.mean(axis=0).std(ddof=1)) if deltas.shape[1] > 1 else 0.0,
        "aleague_gain": float(per_match[a_league].mean()),
        "idsse_gain": float(per_match[~a_league].mean()), "improved_matches": int((per_match > 0).sum()),
        "comparison_by_match": dict(zip(map(str, keys), per_match.tolist())),
    }


def dev_gate(item):
    return item["comparison_mean"] >= MIN_GAIN and item["comparison_mean"] > item["paired_seed_sd"]


def test_gate(item, dead):
    return (item["comparison_mean"] >= MIN_GAIN and item["improved_matches"] >= 4
            and item["aleague_gain"] >= -MAX_A_DROP and dead >= -MAX_DEAD_DROP)


def decoded(decoder, p, fps, recipe, duration_model, touch):
    duration = (duration_prior(duration_model, fps, recipe["duration"])
                if "duration" in recipe else None)
    pred = decoder(p, fps, recipe.get("costs", COSTS), duration)
    return snap(pred, touch, recipe["snap"]) if "snap" in recipe else pred


def extract_touch(args, matches, root):
    features = {}
    for match in matches:
        source, loader, fps = dataset(match, args.skillcorner, args.idsse)
        for half, (tracks, _, _, _, _) in enumerate(loader(source, match), 1):
            key = match, half
            path = root / f"touch-{match}-{half}.npy"
            if path.exists():
                features[key] = np.load(path)
                continue
            physical = touch_inputs(repair_player_tracks(tracks, fps), fps)
            np.save(path, physical)
            features[key] = physical
    return features


def round_one(args, matches):
    for match in matches:
        for half in (1, 2):
            path = args.round_one / f"emissions-{match}-{half}.npz"
            with np.load(path) as row:
                if row["weights_hash"] != fingerprint(args.gru):
                    raise ValueError(f"wrong GRU reference in {path}")
                yield match, half, dict(row)


def prepare_bank(args, models, matches, conditions):
    folder = args.output / "emissions"
    folder.mkdir(exist_ok=True)
    results = {}
    for condition in conditions:
        info, bank = {}, {path: {} for path in models}
        for match, half, row in halves(args.observations, matches, condition):
            key = match, half
            info[key] = {k: row[k] for k in ("y", "fps", "missing", "edge")}
            for path, model in models.items():
                bank[path][key] = cached_emissions(folder, path, model, match, half, condition, row, args.device)
        results[condition] = info, bank
    return results


def touch_bank(args, matches, folder, model, reference, info):
    physical = extract_touch(args, matches, folder)
    return {key: likelihood(touch_features(value, reference[key], int(info[key]["fps"])), model)
            for key, value in physical.items()}


def score_rows(info, emissions, recipe, decoder, duration, touch, full=False):
    values = {}
    for key, row in info.items():
        fps = int(row["fps"])
        pred = decoded(decoder, emissions[key], fps, recipe, duration, touch.get(key))
        if full:
            values[key] = measure(row["y"], pred, row["missing"], row["edge"], fps)
            timing = values[key]["transition_mae_frames"]
            values[key]["transition_mae_seconds"] = None if timing is None else timing / fps
        else:
            values[key] = score(row, pred)
    return values


def metric(results, key):
    return [{match: row[key] for match, row in values.items()} for values in results]


def packed(results):
    return {f"{key[0]}/{key[1]}": row for key, row in results.items()}


def adaptation(args, source, models, matches, mode, setting, decoder, external):
    if mode == "self":
        if external is None:
            external = {}
        for path in source.paths:
            kind = path.stem.rsplit("-", 1)[0]
            if kind not in external:
                train_matches = [v for v in args.members["train"] if isinstance(v, int) or path != args.gru]
                if "aleague" in path.stem:
                    train_matches = [v for v in train_matches if isinstance(v, int)]
                external[kind] = [row for row in train_data(args.observations, train_matches, ("natural",))
                                  if len(row[2]) >= 512]
    values = {}
    match_halves = {}
    for match, half, row in halves(args.observations, matches, "natural"):
        match_halves.setdefault(match, []).append((int(row["fps"]), row["x"], row["mx"]))
    for match, halves_ in match_halves.items():
        arrays = adapt_match(source, models, halves_, mode, setting, decoder, external)
        for i, p in enumerate(arrays, 1):
            values[match, i] = p
    return values


def adapt_match(source, models, halves_, mode, setting, decoder, external):
    target = target_stats(halves_) if mode == "norm" else None
    outputs = []
    for path in source.paths:
        model = models[path]
        if mode == "norm":
            adapted = normalized_model(model, target, setting)
        else:
            seed = int(path.stem.rsplit("-", 1)[-1]) if path.stem.rsplit("-", 1)[-1].isdigit() else 413
            kind = path.stem.rsplit("-", 1)[0]
            per_seed = prediction(model, halves_)
            adapted = self_train(model, halves_, per_seed, decoder, setting, external[kind], seed)
        outputs.append(prediction(adapted, halves_))
    return [mix(list(arrays), source.weights, source.space) for arrays in zip(*outputs)]


def fit_side_models(args, folder):
    train = list(round_one(args, args.members["train"]))
    physical = extract_touch(args, args.members["train"], folder)
    touch = fit_touch((row["ground"], int(row["fps"]), touch_features(physical[match, half],
                      row["p"], int(row["fps"]))) for match, half, row in train)
    duration = fit_duration(train)
    return touch, duration


def candidates(family, paths, best_seed):
    options = {"best-seed": plain(paths[family][best_seed])}
    for space in ("prob", "log"):
        for group in ("gru-mixed", family):
            group_paths = paths[group]
            options[f"{group}-{space}"] = Source(group_paths, (0.2,) * len(group_paths), space)
        tcn = family if family.startswith("tcn") else "tcn-mixed"
        if f"{tcn}-{space}" not in options:
            options[f"{tcn}-{space}"] = Source(paths[tcn], (0.2,) * 5, space)
        for weight in (0.25, 0.5, 0.75):
            options[f"mixed-{weight:g}-{space}"] = Source(
                paths["gru-mixed"] + paths[tcn], ((1-weight)/5,) * 5 + (weight/5,) * 5, space)
    return options


def score_variants(info, bank, source, decoder, duration, touch, reference_seed, control,
                   include_adaptation, args, models, adapted_cache):
    actual = {key: combined(bank, source, key) for key in info}
    seeded_sources = reference_seed
    seed_emissions = [{key: combined(bank, s, key) for key in info} for s in seeded_sources]
    per_seed_control = [score_rows(info, e, control, decoder, duration, touch) for e in seed_emissions]
    control_stats = stats(per_seed_control)
    actual_control = score_rows(info, actual, control, decoder, duration, touch)

    def search(configs):
        scores = [(dev_mean(score_rows(info, actual, config, decoder, duration, touch)), config)
                  for config in configs]
        return max(scores, key=lambda row: row[0]), scores

    results = {"fixed": {"recipe": control, "actual_dev": dev_mean(actual_control)}}
    (value, config), all_costs = search(
        [{"costs": (a, b)} for a in TURNOVER_GRID for b in STOPPAGE_GRID])
    for name, config, value in (
        ("cost", config, value),
        ("hsmm", *reversed(search([{"duration": d} for d in (0.5, 1.0)])[0])),
        ("snap", *reversed(search([{"snap": n} for n in (5, 10, 15)])[0])),
    ):
        scores = [score_rows(info, e, config, decoder, duration, touch) for e in seed_emissions]
        gate = stats(scores, control_stats)
        actual_gain = value - dev_mean(actual_control)
        results[name] = {"recipe": config, "actual_dev": value, "actual_gain": actual_gain,
                         "seed_dev": gate,
                         "dev_pass": dev_gate(gate) and actual_gain >= MIN_GAIN}
    results["cost"]["grid"] = [{"score": value, "recipe": spec} for value, spec in all_costs]
    edge = (results["cost"]["recipe"]["costs"][0] in (TURNOVER_GRID[0], TURNOVER_GRID[-1])
            or results["cost"]["recipe"]["costs"][1] in (STOPPAGE_GRID[0], STOPPAGE_GRID[-1]))
    results["cost"]["grid_edge"] = edge
    results["cost"]["dev_pass"] &= not edge
    if results["hsmm"]["dev_pass"] and results["snap"]["dev_pass"]:
        config = results["hsmm"]["recipe"] | results["snap"]["recipe"]
        value = dev_mean(score_rows(info, actual, config, decoder, duration, touch))
        gate = stats([score_rows(info, e, config, decoder, duration, touch)
                      for e in seed_emissions], control_stats)
        results["hsmm-snap"] = {"recipe": config, "actual_dev": value, "seed_dev": gate,
                                "dev_pass": dev_gate(gate) and value - dev_mean(actual_control) >= MIN_GAIN}
    for mode, choices in (("norm", (0.25, 0.5, 1.0)), ("self", (0.8, 0.9, 0.95))):
        if not include_adaptation or (mode == "self" and results["norm"]["dev_pass"]):
            continue
        values = []
        for setting in choices:
            all_probs = {}
            for path in set(source.paths).union(*(s.paths for s in seeded_sources)):
                cache_key = path, mode, setting
                if cache_key not in adapted_cache:
                    adapted_cache[cache_key] = adaptation(
                        args, plain(path), models, args.members["dev"], mode, setting, decoder, None)
                all_probs[path] = adapted_cache[cache_key]
            adapted = {key: combined(all_probs, source, key) for key in info}
            value = dev_mean(score_rows(info, adapted, control, decoder, duration, touch))
            values.append((value, setting, all_probs))
        value, setting, probs = max(values, key=lambda item: item[0])
        gate = stats([score_rows(info, {k: combined(probs, s, k) for k in info},
                                 control, decoder, duration, touch) for s in seeded_sources], control_stats)
        results[mode] = {"recipe": {"adaptation": [mode, setting]}, "actual_dev": value,
                         "grid": [{"setting": s, "actual_dev": v} for v, s, _ in values],
                         "seed_dev": gate, "dev_pass": dev_gate(gate) and value - dev_mean(actual_control) >= MIN_GAIN}
    return results


def variants_emissions(source, information, bank):
    return {key: combined(bank, source, key) for key in information}


def assemble_checkpoints(args):
    return {arm: tuple(args.output / f"{arm}-{seed}.pt" for seed in SEEDS) for arm, _ in ARMS}


def compare_emissions(info, bank, source, peers, decoder):
    values = score_rows(info, variants_emissions(source, info, bank), {}, decoder, {}, {})
    per_seed = [score_rows(info, variants_emissions(s, info, bank), {}, decoder, {}, {}) for s in peers]
    # A single deployable ensemble has one score. Compare it to each control seed to report
    # how much the calculated improvement depends on the choice of reference checkpoint.
    return stats([values] * len(peers), stats(per_seed))


def dev_stage(args):
    root = args.output
    paths = assemble_checkpoints(args)
    models = {path: load(path, args.device) for path in [args.gru, *sum(paths.values(), ())]}
    decoder = Decoder(root)
    dev = prepare_bank(args, models, args.members["dev"], CONDITIONS)
    info, bank = dev["natural"]
    frozen = plain(args.gru)
    fixed = [{key: score(row, decoder(bank[args.gru][key], int(row["fps"]))) for key, row in info.items()}]
    control = stats(fixed * len(SEEDS))
    family_rows, seed_scores = {}, {}
    for arm, weights in paths.items():
        seed_scores[arm] = [score_rows(info, bank[path], {}, decoder, {}, {}) for path in weights]
        family_rows[arm] = stats(seed_scores[arm], control)
    lead = max(family_rows, key=lambda arm: (dev_gate(family_rows[arm]), family_rows[arm]["mean"]))
    best_seed = int(np.argmax([dev_mean(rows) for rows in seed_scores[lead]]))
    sources = candidates(lead, paths, best_seed)
    source_rows = {name: compare_emissions(info, bank, source, [plain(p) for p in paths[lead]], decoder)
                   for name, source in sources.items()}
    best_ensemble = max((name for name in sources if name != "best-seed"),
                        key=lambda name: source_rows[name]["mean"])
    # Retain the family-selected individual seed as an independent fallback.
    selected = best_ensemble if dev_gate(source_rows[best_ensemble]) else "best-seed"
    sources["frozen"] = frozen
    folder = root / "touch"
    folder.mkdir(exist_ok=True)
    touch_model, duration = fit_side_models(args, folder)
    touch = touch_bank(args, args.members["dev"], folder, touch_model, bank[args.gru], info)
    decoders = {}
    adapted_cache = {}
    evaluation_sources = dict.fromkeys((selected, "best-seed", "frozen"))
    for name in evaluation_sources:
        peers = [frozen] if name == "frozen" else [plain(p) for p in paths[lead]]
        decoders[name] = score_variants(info, bank, sources[name], decoder, duration, touch, peers,
                                        {}, True, args, models, adapted_cache)
    epoch_condition_scores = {}
    for condition, (rows, observations) in dev.items():
        epoch_condition_scores[condition] = {
            arm: stats([score_rows(rows, observations[path], {}, decoder, {}, {}) for path in checkpoints])
            for arm, checkpoints in paths.items()
        }
    manifest = {
        "lead_family": lead, "family_dev": family_rows, "frozen_dev": control,
        "development_seed_condition_results": epoch_condition_scores,
        "best_seed": SEEDS[best_seed], "ensemble_dev": source_rows,
        "selected_source": selected, "tested_ensemble": best_ensemble,
        "sources": {name: source.spec() for name, source in sources.items()},
        "decoder_dev": decoders, "touch": touch_model, "duration": duration,
        "matched_control_dev": {
            name: stats(seed_scores["tcn-mixed"], stats(seed_scores[name]))
            for name in ("tcn-aleague", "gru-mixed")
        },
    }
    dump(root / "development-freeze.json", manifest)
    print(f"Development frozen: {lead}, {selected}; "
          f"family means { {key: round(v['mean'], 4) for key, v in family_rows.items()} }", flush=True)


def source_from(spec):
    paths = tuple(Path(path) for path in spec["paths"])
    if [fingerprint(p) for p in paths] != spec["sha256"]:
        raise ValueError("checkpoint differs from frozen source")
    return Source(paths, tuple(spec["weights"]), spec["space"])


def against(results, control):
    baseline = stats(metric(control, "spell_f1"))
    comparison = stats(metric(results, "spell_f1"), baseline)
    dead = stats(metric(results, "in_pitch_dead_f1"), stats(metric(control, "in_pitch_dead_f1")))
    comparison["in_pitch_dead_gain"] = dead["comparison_mean"]
    comparison["in_pitch_dead_f1"] = dead["mean"]
    comparison["passed"] = test_gate(comparison, dead["comparison_mean"])
    return comparison


def aggregate_metrics(results):
    numeric = ("macro_f1", "dead_f1", "in_pitch_dead_f1", "transition_f1",
               "transition_mae_seconds", "short_missed")
    return {key: (float(np.mean([row[key] for row in results.values() if row[key] is not None]))
                  if key != "short_missed" else sum(row[key] for row in results.values()))
            for key in numeric}


def heldout_stage(args):
    root = args.output
    frozen_path = root / "development-freeze.json"
    dev = json.loads(frozen_path.read_text())
    source = {name: source_from(spec) for name, spec in dev["sources"].items()}
    paths = assemble_checkpoints(args)
    models = {path: load(path, args.device) for path in [args.gru, *sum(paths.values(), ())]}
    decoder = Decoder(root)
    heldout = prepare_bank(args, models, args.members["test"], CONDITIONS)
    info, bank = heldout["natural"]
    reference = score_rows(info, bank[args.gru], {}, decoder, {}, {}, full=True)
    assert round(dev_mean(metric([reference], "spell_f1")[0]), 4) == 0.6220
    recovery = list(round_one(args, args.members["test"]))
    for match, half, row in recovery:
        assert np.array_equal(row["baseline"], decoder(bank[args.gru][match, half], int(row["fps"])))
    family_data, family_full = {}, {}
    for condition, (rows, observations) in heldout.items():
        controls = [score_rows(rows, observations[args.gru], {}, decoder, {}, {}, full=True)] * len(SEEDS)
        all_families = {arm: [score_rows(rows, observations[path], {}, decoder, {}, {}, full=True)
                             for path in weights] for arm, weights in paths.items()}
        family_full[condition] = all_families
        family_data[condition] = {arm: against(values, controls) for arm, values in all_families.items()}
    lead = dev["lead_family"]
    lead_passes = dev_gate(dev["family_dev"][lead]) and family_data["natural"][lead]["passed"]
    touch_model, duration = dev["touch"], dev["duration"]
    touch = touch_bank(args, args.members["test"], root / "touch", touch_model, bank[args.gru], info)
    test_results = {"family": family_data, "matched_controls": {
        name: against(family_full["natural"]["tcn-mixed"], family_full["natural"][name])
        for name in ("tcn-aleague", "gru-mixed")
    }, "ensemble": {}, "decoder_and_adaptation": {}, "halves": {
        "frozen": packed(reference),
        "family": {condition: {arm: {str(seed): packed(result) for seed, result in zip(SEEDS, values)}
                               for arm, values in families.items()}
                   for condition, families in family_full.items()},
        "ensemble": {}, "decoder_and_adaptation": {},
    }}
    lead_reference = family_full["natural"][lead]
    for name in ("best-seed", dev["tested_ensemble"]):
        values = score_rows(info, variants_emissions(source[name], info, bank), {},
                            decoder, duration, touch, full=True)
        test_results["ensemble"][name] = against([values] * len(SEEDS), lead_reference)
        test_results["halves"]["ensemble"][name] = packed(values)
    adapted_cache = {}
    for name, experiments in dev["decoder_dev"].items():
        chosen = source[name]
        emissions = variants_emissions(chosen, info, bank)
        before = score_rows(info, emissions, {}, decoder, duration, touch, full=True)
        replicas = [source["frozen"]] if name == "frozen" else [plain(p) for p in paths[lead]]
        seed_base = [score_rows(info, variants_emissions(replica, info, bank), {}, decoder,
                                duration, touch, full=True) for replica in replicas]
        test_results["decoder_and_adaptation"][name] = alternatives = {}
        test_results["halves"]["decoder_and_adaptation"][name] = by_half = {"fixed": packed(before)}
        alternatives["fixed"] = against([before], [before]) | aggregate_metrics(before)
        for stage, arm in experiments.items():
            if stage == "fixed":
                continue
            recipe = arm["recipe"]
            adapted = "adaptation" in recipe
            after_emissions = emissions
            if adapted:
                mode, setting = recipe["adaptation"]
                all_probs = {}
                for path in set(chosen.paths).union(*(replica.paths for replica in replicas)):
                    cache_key = path, mode, setting
                    if cache_key not in adapted_cache:
                        adapted_cache[cache_key] = adaptation(
                            args, plain(path), models, args.members["test"], mode, setting, decoder, None)
                    all_probs[path] = adapted_cache[cache_key]
                after_emissions = variants_emissions(chosen, info, all_probs)
            after = score_rows(info, after_emissions, recipe, decoder, duration, touch, full=True)
            pairs = [score_rows(info, variants_emissions(replica, info, all_probs if adapted else bank),
                                recipe, decoder, duration, touch, full=True) for replica in replicas]
            paired = against(pairs, seed_base)
            actual = against([after], [before])
            alternatives[stage] = actual | aggregate_metrics(after) | {
                "paired_seed_test": paired, "passed": actual["passed"] and paired["passed"],
            }
            by_half[stage] = {"deployed": packed(after),
                              "seeds": [packed(result) for result in pairs]}
    choice = "frozen"
    if lead_passes:
        best_ensemble = dev["selected_source"]
        choice = best_ensemble if (
            best_ensemble != "best-seed" and test_results["ensemble"][best_ensemble]["passed"]) else "best-seed"
        # A dev-selected individual must also preserve the matched family-level gain
        # against the frozen deployed control on the holdouts.
        deployed = test_results["halves"]["ensemble"][choice]
        keys = list(info)
        actual = {key: deployed[f"{key[0]}/{key[1]}"] for key in keys}
        if not against([actual], [reference])["passed"]:
            choice = "frozen"
    best_stage = "fixed"
    candidates_ = dev["decoder_dev"][choice]
    for stage in sorted(candidates_, key=lambda key: -candidates_[key]["actual_dev"]):
        if stage == "fixed":
            continue
        if stage == "hsmm-snap" and not all(
                test_results["decoder_and_adaptation"][choice][part]["passed"] for part in ("hsmm", "snap")):
            continue
        if candidates_[stage]["dev_pass"] and test_results["decoder_and_adaptation"][choice][stage]["passed"]:
            best_stage = stage
            break
    selection = {
        "source_name": choice, "family": lead if choice != "frozen" else "original continuous GRU",
        "source": source[choice].spec(), "stage": best_stage,
        "decoder": candidates_[best_stage]["recipe"], "touch": touch_model, "duration": duration,
        "touch_reference": source["frozen"].spec(),
        "development_sha256": fingerprint(frozen_path),
        "external_natural": test_results["decoder_and_adaptation"][choice][best_stage],
        "external_frozen": dev_mean(metric([reference], "spell_f1")[0]),
        "public_use": "single automated correctness check after full-grid deterministic replay",
    }
    dump(root / "holdout-confirmation.json", test_results)
    dump(root / "decision-before-public.json", selection)
    print(f"Holdout confirmed: {choice} / {best_stage}; external spellF1 "
          f"{selection['external_natural']['mean']:.5f}", flush=True)


if __name__ == "__main__":
    parser = arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("stage", choices=("dev", "test"))
    parser.add_argument("--idsse", type=Path, required=True)
    parser.add_argument("--round-one", type=Path, required=True)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    args.members = pin(args)
    torch.use_deterministic_algorithms(True)
    if args.stage == "dev":
        dev_stage(args)
    else:
        heldout_stage(args)
