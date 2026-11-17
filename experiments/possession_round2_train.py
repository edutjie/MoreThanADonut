"""Repeat the round-one shared classifier recipe, and cache native-grid emissions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluation import possession_metrics as m
from experiments.possession_execution import dump, fingerprint, split_members
from experiments.possession_round2_decoder import Decoder
from experiments.possession_tcn import PossessionTCN
from sequence_possession import EPOCHS, PossessionGRU, batches, normalize, sequence_probabilities


SEEDS = (413, 733, 1049, 1493, 2027)
ARMS = (("tcn-mixed", True), ("tcn-aleague", False), ("gru-mixed", True))
CONDITIONS = ("natural", "weak", "strong")
CLASSES = {"tcn": PossessionTCN, "gru": PossessionGRU}
COSTS = (5, 10)
TURNOVER_GRID = (1, 2.5, 4, 5, 7.5, 10, 15, 25)
STOPPAGE_GRID = (2, 5, 7.5, 10, 12.5, 15, 22.5, 35)
BOOTSTRAP = 10000
MIN_GAIN = 0.005
MAX_A_DROP = 0.003
MAX_DEAD_DROP = 0.005


def halves(root, matches, condition):
    for match in matches:
        for half in (1, 2):
            path = root / f"obs-{match}-{half}-{condition}.npz"
            with np.load(path) as row:
                yield match, half, dict(row)


def score(row, pred):
    ground = row["y"]
    return m.spell_scores(ground, pred, m.valid_runs(ground))["mean_f1"]


def train_data(root, matches, conditions=("natural", "random")):
    result = []
    for condition in conditions:
        for _, _, row in halves(root, matches, condition):
            length = len(row["y"])
            frames = np.minimum(np.rint(np.arange(0, length, int(row["fps"]) / 10)).astype(int), length - 1)
            result.append(tuple(row[key][frames] for key in ("x", "mx", "y")))
    return result


def fit(root, arm, seed, train, dev, decoder, device):
    path = root / f"{arm}-{seed}.pt"
    if path.exists():
        return path
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    architecture = arm.split("-")[0]
    model = CLASSES[architecture](*normalize(train)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=m.X)
    best, history = -1, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y, _, _ in batches(train, rng, len(train)):
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = criterion(output.flatten(0, 1), y.flatten())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
        model.eval()
        scores = {}
        for condition, match, _, row in dev:
            fps = int(row["fps"])
            p = sequence_probabilities(model, row["x"], row["mx"], fps, device)
            scores.setdefault((condition, match), []).append(score(row, decoder(p, fps)))
        peak = float(np.mean([np.mean(values) for values in scores.values()]))
        print(f"{arm} seed {seed} epoch {epoch}: {peak:.6f}", flush=True)
        history.append(peak)
        if peak > best:
            best = peak
            temporary = path.with_suffix(".tmp")
            torch.save({"state": model.state_dict(), "epoch": epoch, "seed": seed,
                        "architecture": architecture, "continuous_motion": True,
                        "repair_players": True}, temporary)
    temporary.replace(path)
    dump(path.with_suffix(".json"), history)
    return path


def load(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state"]
    model = CLASSES[payload.get("architecture", "gru")](state["mean"], state["scale"])
    model.load_state_dict(state)
    return model.to(device).eval()


def array_hash(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def cached_emissions(folder, checkpoint, model, match, half, condition, row, device):
    path = folder / f"{checkpoint.stem}-{match}-{half}-{condition}.npz"
    checkpoint_hash = fingerprint(checkpoint)
    input_hash = [array_hash(row[key]) for key in ("x", "mx", "y")]
    fps = int(row["fps"])
    grid = [0, len(row["y"]), fps]
    if path.exists():
        with np.load(path) as cache:
            if (cache["weights_hash"] != checkpoint_hash or cache["input_hash"].tolist() != input_hash
                    or cache["grid"].tolist() != grid or array_hash(cache["p"]) != cache["probability_hash"]):
                raise ValueError(f"unverified emission cache {path}")
            return cache["p"]
    p = sequence_probabilities(model, row["x"], row["mx"], fps, device)
    np.savez(path, p=p, grid=grid, weights_hash=checkpoint_hash, input_hash=input_hash,
             probability_hash=array_hash(p))
    return p


def pin(args):
    members = split_members(args.skillcorner)
    plan = {
        "previous_run": "345dff2181254fe08c0c744541c50e5aee4e8412",
        "split": members, "seeds": SEEDS, "epochs": EPOCHS, "decoder_control": COSTS,
        "turnover_grid": TURNOVER_GRID, "stoppage_grid": STOPPAGE_GRID,
        "snap_radius_native_frames": [5, 10, 15], "duration_cap_seconds": 30,
        "duration_strengths": [0.5, 1.0], "norm_blend": [0.25, 0.5, 1.0],
        "self_training_confidence": [0.8, 0.9, 0.95], "self_training_steps": 16,
        "ensembles": {"space": ["prob", "log"], "tcn_weights": [0.25, 0.5, 0.75]},
        "paired_bootstrap": {"draws": BOOTSTRAP, "seed": 20261117},
        "gate": {"dev_min": MIN_GAIN, "dev_gt_paired_seed_sd": True, "test_min": MIN_GAIN,
                 "test_matches_improved": 4, "maximum_aleague_drop": MAX_A_DROP,
                 "maximum_in_pitch_dead_drop": MAX_DEAD_DROP},
        "checkpoint_sha256": fingerprint(args.gru),
        "selection": "dev configuration only; one batched external holdout confirmation",
        "training": "round-one matching: natural and fixed random corruption; seed-specific epochs "
                    "selected by equal-match natural, weak and strong development spellF1",
    }
    path = args.output / "run-index.json"
    normalized = json.loads(json.dumps(plan))
    if path.exists():
        stored = json.loads(path.read_text())
        if any(stored.get(key) != value for key, value in normalized.items()):
            raise ValueError(f"manifest mismatch {path}")
    else:
        dump(path, plan)
    return members


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    members = pin(args)
    decoder = Decoder(args.output)
    device = torch.device(args.device)
    dev = [(condition, match, half, row) for condition in CONDITIONS
           for match, half, row in halves(args.observations, members["dev"], condition)]
    data = {}
    for arm, use_idsse in ARMS:
        if use_idsse not in data:
            matches = [match for match in members["train"] if use_idsse or isinstance(match, int)]
            data[use_idsse] = train_data(args.observations, [m for m in matches if isinstance(m, int)])
            if use_idsse:
                data[use_idsse] += train_data(args.observations, [m for m in matches if isinstance(m, str)])
        for seed in SEEDS:
            fit(args.output, arm, seed, data[use_idsse], dev, decoder, device)


def arguments(parser):
    for arg in ("skillcorner", "gru", "observations", "output"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    main(arguments(argparse.ArgumentParser(description=__doc__)).parse_args())
