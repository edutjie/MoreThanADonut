"""Compare matched random and informative ball dropouts on external matches."""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.possession_execution import (
    MIN_GAIN, SEED, averages, dataset, dump, fingerprint, fixed, measure, restart_cues, split_members,
)
from evaluation import possession_metrics as m
from external_possession import short_ball_dropouts
from player_track_quality import corrupt_player_tracks, repair_player_tracks
from sequence_possession import (
    EPOCHS, PossessionGRU, batches, normalize, restore, sequence_features, sequence_probabilities,
)

REGIMES = {"weak": 0.6, "strong": 3.0}


def dropout_masks(tracks, pitch, labels, fps, seed):
    # An everywhere-observed probe exposes the simulated mask before natural gaps.
    probe = np.zeros((len(tracks), 1, 2), dtype=np.float32)
    random = np.isnan(short_ball_dropouts(probe, fps, np.random.default_rng(seed))[:, 0, 0])
    bounds = np.flatnonzero(np.diff(np.r_[False, random, False])).reshape(-1, 2)
    lengths = bounds[:, 1] - bounds[:, 0]
    ball = tracks[:, -1]
    speed = np.r_[0, np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps]
    speed = np.clip(np.nan_to_num(speed) / 15, 0, 1)
    crowding = (np.linalg.norm(tracks[:, :-1] - ball[:, None], axis=-1) < 5).sum(axis=1)
    edge = np.nan_to_num(np.minimum(ball, pitch - ball).min(axis=1), nan=10)
    base = 1 + speed + 0.2 * crowding + (edge < 4) * 0.75
    rng = np.random.default_rng(seed + 29)
    # Jitter each dead interval independently, avoiding exact label boundaries.
    events = (labels == m.D)
    spans = np.flatnonzero(np.diff(np.r_[False, events, False])).reshape(-1, 2)
    dead_halo = np.zeros(len(ball), dtype=bool)
    for a, b in spans:
        left, right = rng.integers(-fps, fps + 1, size=2)
        start, end = max(0, a + left), min(len(ball), b + right)
        if end > start:
            dead_halo[start:end] = True
    masks = {"random": random}
    lengths_descending = sorted(lengths, reverse=True)
    for name, strength in REGIMES.items():
        weights = base + strength * dead_halo
        candidates = np.arange(len(ball))
        allowed = np.ones(len(ball), dtype=bool)
        hidden = np.zeros(len(ball), dtype=bool)
        for length in lengths_descending:
            # One clear frame between intervals preserves the control gap-length histogram.
            prefix = np.r_[0, np.cumsum(~allowed)]
            starts = candidates[:len(ball) - length + 1]
            viable = starts[(prefix[starts + length] - prefix[starts]) == 0]
            if not len(viable):
                raise ValueError("cannot place matched dropout lengths")
            probability = weights[viable]
            offset = rng.choice(viable, p=probability / probability.sum())
            hidden[offset:offset + length] = True
            allowed[max(0, offset - 1):offset + length + 1] = False
        assert hidden.sum() == random.sum()
        masks[name] = hidden
    return masks


def seed_for(match, half):
    return (int(match[1:], 36) if isinstance(match, str) else match) + SEED + half * 100


def features(tracks, weights, teams, pitch, fps):
    return sequence_features(repair_player_tracks(tracks, fps), weights, teams, pitch, fps, True)


def prepare(args, members, role, conditions):
    for match in members[role]:
        root, loader, fps = dataset(match, args.skillcorner, args.idsse)
        for half, (tracks, weights, teams, pitch, labels) in enumerate(loader(root, match), 1):
            seed = seed_for(match, half)
            masks = dropout_masks(tracks, pitch, labels, fps, seed)
            edge = restart_cues(tracks, teams, pitch, fps)[-1]
            for condition in conditions:
                path = args.output / f"obs-{match}-{half}-{condition}.npz"
                if path.exists():
                    continue
                observed = tracks.copy()
                if condition != "natural":
                    observed[masks[condition], -1] = np.nan
                    if role == "train":
                        observed = corrupt_player_tracks(observed, pitch, fps, np.random.default_rng(seed + 1))
                arrays = features(observed, weights, teams, pitch, fps)
                missing = ~np.isfinite(observed[:, -1]).all(axis=1)
                np.savez(path, x=arrays[0], mx=arrays[1], y=labels, missing=missing, edge=edge, fps=fps,
                         simulated=int(masks[condition].sum()) if condition != "natural" else 0)
        print(f"Prepared {role} {match} {conditions}", flush=True)


def records(args, matches, condition):
    for match in matches:
        for half in (1, 2):
            with np.load(args.output / f"obs-{match}-{half}-{condition}.npz") as r:
                yield match, dict(r)


def evaluate(args, models, matches, conditions):
    scores = {}
    for condition in conditions:
        scores[condition] = {name: {} for name in models}
        for match, r in records(args, matches, condition):
            fps = int(r["fps"])
            for name, model in models.items():
                p = sequence_probabilities(model, r["x"], r["mx"], fps, args.device)
                score = measure(r["y"], fixed(p, fps), r["missing"], r["edge"], fps)
                score["missing_rate"] = float(r["missing"].mean())
                scores[condition][name].setdefault(str(match), []).append(score)
    return scores


def fit(args, members, condition):
    matches = [match for match in members["train"] if isinstance(match, int)]
    train = [(r["x"], r["mx"], r["y"]) for arm in ("natural", condition)
             for _, r in records(args, matches, arm)]
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = PossessionGRU(*normalize(train)).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=m.X)
    history = []
    best = -1.0
    path = args.output / f"observation-{condition}.pt"
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, y, _, _ in batches(train, rng, len(train)):
            opt.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(x).to(args.device))
            target = torch.from_numpy(y).to(args.device, dtype=torch.long)
            loss = criterion(logits.flatten(0, 1), target.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step()
            losses.append(loss.detach())
        model.eval()
        dev = [match for match in members["dev"] if isinstance(match, int)]
        scores = evaluate(args, {condition: model}, dev, ("natural", "weak", "strong"))
        results = {arm: averages(values)[condition] for arm, values in scores.items()}
        mean = float(np.mean(list(results.values())))
        print(f"{condition} epoch {epoch}: loss={torch.stack(losses).mean().item():.4f} {results}", flush=True)
        history.append({"epoch": epoch, "mean_spell_f1": mean, **results})
        if mean > best:
            best = mean
            torch.save({"state": model.state_dict(), "repair_players": True, "continuous_motion": True,
                        "architecture": "gru", "epoch": epoch}, path)
    dump(args.output / f"observation-{condition}-history.json", history)
    return path


def run(args):
    members = split_members(args.skillcorner)
    conditions = ("natural", "random", "weak", "strong")
    settings = {"splits": members, "train_provider": "A-League", "train_arms": ["random", "strong"],
                "dev_conditions": ["natural", "weak", "strong"], "epochs": EPOCHS,
                "minimum_spell_f1_gain": MIN_GAIN, "seed": SEED, "missingness_weights": REGIMES,
                "device": str(args.device), "code_sha256": fingerprint(__file__)}
    dump(args.output / "observation-manifest.json", settings)
    prepare(args, {"train": [m for m in members["train"] if isinstance(m, int)]},
            "train", ("natural", "random", "strong"))
    prepare(args, members, "dev", conditions)
    paths = {arm: fit(args, members, arm) for arm in ("random", "strong")}
    paths["frozen"] = args.gru
    models = {arm: restore(path, args.device)[0] for arm, path in paths.items()}
    dev = evaluate(args, models, members["dev"], conditions)
    dump(args.output / "observation-dev.json", dev)
    # Compare natural and matched corruptions with equal condition weights.
    values = {arm: float(np.mean([averages(dev[c])[arm] for c in ("natural", "weak", "strong")]))
              for arm in models}
    best = max(values, key=values.get)
    selected = best if values[best] - values["frozen"] >= MIN_GAIN else "frozen"
    dump(args.output / "observation-selection.json", {"development": values, "selected": selected,
         "checkpoint_sha256": {arm: fingerprint(path) for arm, path in paths.items()}})
    print(f"Observation development {values}; selected {selected}", flush=True)
    prepare(args, members, "test", conditions)
    test = evaluate(args, models, members["test"], conditions)
    dump(args.output / "observation-test.json", test)
    print(f"Observation held out { {c: averages(result) for c, result in test.items()} }", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("skillcorner", "idsse", "gru", "output"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)
