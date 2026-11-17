"""Compare shared and provider-specific labels under the matched random dropout recipe."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluation import possession_metrics as m
from experiments.possession_execution import MIN_GAIN, SEED, dump, fingerprint, measure, split_members
from experiments.possession_observation import prepare, records
from sequence_possession import (
    DECODER_COSTS, EPOCHS, PossessionGRU, batches, normalize, restore, sequence_probabilities,
)
from tactical_baseline import decode
from experiments.possession_tcn import PossessionTCN


class ProviderGRU(PossessionGRU):
    def __init__(self, mean, scale):
        super().__init__(mean, scale)
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(96, 6))
        self.boundary = nn.Linear(96, 1)

    def forward(self, features):
        states, _ = self.gru(self.input_projection(self.standardized_inputs(features)))
        return self.head(states).unflatten(-1, (2, 3)), self.boundary(states).squeeze(-1)


class ProviderTCN(PossessionTCN):
    def __init__(self, mean, scale):
        super().__init__(mean, scale)
        self.head = nn.Linear(48, 6)
        self.boundary = nn.Linear(48, 1)

    def forward(self, features):
        states = self.project(self.standardized_inputs(features))
        states = self.blocks(states.transpose(1, 2)).transpose(1, 2)
        return self.head(states).unflatten(-1, (2, 3)), self.boundary(states).squeeze(-1)


def factor(logits):
    team = logits[..., :2].log_softmax(-1)
    dead = logits[..., 2:]
    return torch.cat((team + nn.functional.logsigmoid(-dead), nn.functional.logsigmoid(dead)), dim=-1)


class Inference(nn.Module):
    def __init__(self, model, provider):
        super().__init__()
        self.model = model
        self.provider = provider

    def forward(self, features):
        outputs, _ = self.model(features)
        logs = factor(outputs)
        if self.provider is not None:
            return logs[:, :, self.provider]
        return torch.logsumexp(logs, dim=2) - np.log(2)


@torch.inference_mode()
def boundary_probabilities(model, x, mirror, fps, device):
    from sequence_possession import CONTEXT, WINDOW

    sample = np.minimum(np.rint(np.arange(0, len(x), fps / 10)).astype(int), len(x) - 1)
    prediction = np.empty(len(sample), dtype=np.float32)
    central = WINDOW - 2 * CONTEXT
    for start in range(0, len(sample), central):
        stop = min(len(sample), start + central)
        left, right = max(0, start - CONTEXT), min(len(sample), stop + CONTEXT)
        idx = sample[left:right]
        window = torch.from_numpy(np.stack((x[idx], mirror[idx]))).to(device)
        _, b = model(window)
        prediction[start:stop] = b.sigmoid().mean(0)[start - left:stop - left].cpu().numpy()
    return np.interp(np.arange(len(x)), sample, prediction)


def decode_boundary(p, fps, boundary=None, strength=0):
    from external_possession import decode_costs

    logged = np.log(np.maximum(p, 1e-6))
    if not strength:
        return m.label_codes(decode_costs(logged, fps, *DECODER_COSTS))
    t, d = DECODER_COSTS
    cost = np.array([[0, t, d], [t, 0, d], [d, d, 0]], dtype=float)
    bonus = strength * np.clip((boundary - 0.1) / 0.5, 0, 1)
    return m.label_codes(decode(logged * (10 / fps), transitions=cost, turnover_bonus=bonus))


def average(splits, key):
    return float(np.mean([np.mean([h["spell_f1"] for h in halves])
                          for group in splits.values() for halves in group[key].values()]))


def evaluate(args, models, matches, settings, conditions):
    scores = {}
    for condition in conditions:
        result = {name: {} for name in settings}
        for match, r in records(args, matches, condition):
            fps = int(r["fps"])
            emitted = {}
            boundary = None
            for name, (kind, provider, strength) in settings.items():
                key = kind, provider
                if key not in emitted:
                    model = models[kind] if kind != "provider" else Inference(models[kind], provider)
                    emitted[key] = sequence_probabilities(model, r["x"], r["mx"], fps, args.device)
                if strength and boundary is None:
                    boundary = boundary_probabilities(models[kind], r["x"], r["mx"], fps, args.device)
                pred = decode_boundary(emitted[key], fps, boundary, strength)
                score = measure(r["y"], pred, r["missing"], r["edge"], fps)
                result[name].setdefault(str(match), []).append(score)
        scores[condition] = result
    return scores


def training_data(args, matches):
    def load(match_group):
        halves = []
        for condition in ("natural", "random"):
            for _, r in records(args, match_group, condition):
                fps = int(r["fps"])
                frames = np.minimum(np.rint(np.arange(0, len(r["y"]), fps / 10)).astype(int),
                                    len(r["y"]) - 1)
                halves.append((r["x"][frames], r["mx"][frames], r["y"][frames]))
        return halves

    a = load([match for match in matches if isinstance(match, int)])
    b = load([match for match in matches if isinstance(match, str)])
    return a + b, len(a)


def fit(args, data, partial_start, dev, kind, architecture="gru"):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    provider = kind == "provider"
    classes = {"gru": (PossessionGRU, ProviderGRU), "tcn": (PossessionTCN, ProviderTCN)}
    model = classes[architecture][int(provider)](*normalize(data)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=m.X)
    choices = ({label: (kind, index, 0) for label, index in
                (("pooled", None), ("a-league", 0), ("idsse", 1))}
               if provider else {kind: (kind, None, 0)})
    if architecture == "tcn" and provider:
        choices = {"pooled-1": (kind, None, 1)}
    best = -1
    history = []
    file_id = kind if architecture == "gru" else f"{architecture}-{kind}"
    path = args.output / f"provider-{file_id}.pt"
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y, _, partial in batches(data, rng, partial_start):
            x = torch.from_numpy(x).to(args.device)
            y = torch.from_numpy(y).to(args.device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            if provider:
                logits, boundary = output
                batch_index = torch.arange(len(x), device=args.device)
                p_index = torch.from_numpy(partial.astype(np.int64)).to(args.device)
                pred = factor(logits[batch_index, :, p_index])
                loss = nn.functional.nll_loss(pred.flatten(0, 1), y.flatten(), ignore_index=m.X)
                valid_team = (y[:, :-1] < m.D) & (y[:, 1:] < m.D)
                changed = (y[:, :-1] != y[:, 1:]).to(pred.dtype)
                errors = nn.functional.binary_cross_entropy_with_logits(
                    boundary[:, 1:], changed, reduction="none", pos_weight=changed.new_tensor(12))
                loss = loss + 0.20 * (errors * valid_team).sum() / valid_team.sum().clamp(min=1)
            else:
                loss = criterion(output.flatten(0, 1), y.flatten())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
        model.eval()
        scores = evaluate(args, {kind: model}, dev, choices, ("natural", "weak", "strong"))
        means = {name: average(scores, name) for name in choices}
        peak = max(means.values())
        history.append({"epoch": epoch, **means})
        print(f"Provider {architecture} {kind} epoch {epoch} {means}", flush=True)
        if peak > best:
            best = peak
            torch.save({"state": model.state_dict(), "epoch": epoch, "architecture": architecture,
                        "continuous_motion": True, "repair_players": True}, path)
    dump(args.output / f"provider-{file_id}-history.json", history)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=args.device)["state"])
    return model, path


def main(args):
    members = split_members(args.skillcorner)
    # Freeze the selected observation recipe before fitting either provider architecture.
    settings = {"seed": SEED, "splits": members, "epochs": EPOCHS, "training": "natural + random",
                "provider_choices": ["pooled", "a-league", "idsse"], "boundary_strengths": [1, 2],
                "code_sha256": fingerprint(__file__), "frozen_gru": fingerprint(args.gru)}
    if args.replicate_tcn:
        settings.update({"architecture": "tcn", "fixed_provider_choice": "pooled", "boundary_strength": 1})
        dump(args.output / "provider-tcn-manifest.json", settings)
    else:
        dump(args.output / "provider-manifest.json", settings)
    prepare(args, {"train": [match for match in members["train"] if isinstance(match, str)]},
            "train", ("natural", "random"))
    data, partial_start = training_data(args, members["train"])
    models = {}
    if args.replicate_tcn:
        paths = {}
        for kind in ("shared", "provider"):
            models[kind], paths[kind] = fit(args, data, partial_start, members["dev"], kind, "tcn")
        specs = {"shared": ("shared", None, 0), "pooled-1": ("provider", None, 1)}
        dev = evaluate(args, models, members["dev"], specs, ("natural", "weak", "strong"))
        dump(args.output / "provider-tcn-freeze.json", {
            "dev": {name: average(dev, name) for name in specs},
            "checkpoint_sha256": {key: fingerprint(value) for key, value in paths.items()},
        })
        test = evaluate(args, models, members["test"], specs, ("natural", "weak", "strong"))
        dump(args.output / "provider-tcn-test.json", test)
        print(f"TCN held out { {name: average(test, name) for name in specs} }", flush=True)
        return
    paths = {"frozen": args.gru}
    for kind in ("shared", "provider"):
        models[kind], paths[kind] = fit(args, data, partial_start, members["dev"], kind)
    models["frozen"], _ = restore(args.gru, args.device)
    specs = {key: (key, None, 0) for key in ("shared", "frozen")}
    for name, index in zip(settings["provider_choices"], (None, 0, 1)):
        for strength in (0, *settings["boundary_strengths"]):
            specs[f"{name}-{strength}"] = ("provider", index, strength)
    conditions = ("natural", "weak", "strong")
    dev = evaluate(args, models, members["dev"], specs, conditions)
    dump(args.output / "provider-dev.json", dev)
    values = {name: average(dev, name) for name in specs}
    variants = list(specs)[2:]
    winners = [max([name for name in variants if bool(specs[name][-1]) == boundary], key=values.get)
               for boundary in (False, True)]
    winner = max(("shared", *winners), key=values.get)
    selected = winner if values[winner] - values["frozen"] >= MIN_GAIN else "frozen"
    dump(args.output / "provider-selection.json", {"development": values, "selected": selected,
         "winners": winners, "checkpoint_sha256": {k: fingerprint(v) for k, v in paths.items()}})
    print(f"Provider development {values}; selected {selected}", flush=True)
    test = evaluate(args, models, members["test"],
                    {name: specs[name] for name in ("frozen", "shared", *winners)}, conditions)
    dump(args.output / "provider-test.json", test)
    print(f"Provider held out { {name: average(test, name) for name in test['natural']} }", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("skillcorner", "idsse", "gru", "output"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replicate-tcn", action="store_true")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
