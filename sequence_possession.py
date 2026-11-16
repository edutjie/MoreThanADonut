"""Evaluate bidirectional sequence models on the frozen external possession splits."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ball_recovery_experiment import contest_tracking
from evaluation import possession_metrics as metrics
from external_possession import (
    DEV_MATCHES, IDSSE_FPS, LAG_SECONDS, SKILLCORNER_FPS, TEST_MATCHES, decode_costs,
    load_idsse, load_model, load_skillcorner, mask_ball, probabilities,
    short_ball_dropouts, tracking_features,
)
from games import GAME, half_video_blocks
from idsse_training import IDSSE_TEST
from possession_sources import LEGACY_MATCHES, load_legacy_skillcorner
from starter.baseline import write_submission


SEED = 42
WINDOW = 512
CONTEXT = 80
BATCH_SIZE = 24
EPOCHS = 8
BASE_FEATURES = 111
DECODER_COSTS = (5, 10)


class SequenceModel(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def standardized_inputs(self, features):
        missing = ~torch.isfinite(features)
        x = ((features - self.mean) / self.scale).masked_fill(missing, 0).clamp(-8, 8)
        return torch.cat((x, missing.to(x.dtype)), dim=-1)


class PossessionGRU(SequenceModel):
    architecture = "gru"

    def __init__(self, mean, scale):
        super().__init__(mean, scale)
        width = len(mean)
        self.input_projection = nn.Sequential(nn.Linear(2 * width, 48), nn.LayerNorm(48),
                                              nn.GELU(), nn.Dropout(0.15))
        self.gru = nn.GRU(48, 48, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(96, 3))

    def forward(self, features, padding=None):
        x = self.standardized_inputs(features)
        states, _ = self.gru(self.input_projection(x))
        return self.head(states)


class PossessionTransformer(SequenceModel):
    architecture = "transformer"

    def __init__(self, mean, scale):
        super().__init__(mean, scale)
        width = 64
        self.input_projection = nn.Sequential(nn.Linear(2 * len(mean), width), nn.LayerNorm(width),
                                              nn.GELU(), nn.Dropout(0.15))
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=128,
                                           dropout=0.15, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=4, norm=nn.LayerNorm(width),
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(width, 3))
        frequencies = 10000 ** (-torch.arange(0, width, 2, dtype=torch.float32) / width)
        phases = torch.arange(WINDOW)[:, None] * frequencies
        positions = torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(1)
        self.register_buffer("positions", positions, persistent=False)

    def forward(self, features, padding=None):
        x = self.input_projection(self.standardized_inputs(features))
        x = x + self.positions[:x.shape[1]].to(x.dtype)
        return self.head(self.encoder(x, src_key_padding_mask=padding))


ARCHITECTURES = {"gru": PossessionGRU, "transformer": PossessionTransformer}


def visibility_features(ball, fps):
    visible = np.isfinite(ball).all(axis=1)
    frames = np.arange(len(ball))
    last = np.maximum.accumulate(np.where(visible, frames, -1))
    age = np.where(last >= 0, np.log1p((frames - last) / fps), np.nan)
    return np.column_stack((visible, age)).astype(np.float32)


def sequence_features(positions, weights, teams, pitch, fps):
    base, mirrored = tracking_features(positions, weights, teams, pitch, fps, LAG_SECONDS)
    visibility = visibility_features(positions[:, -1], fps)
    return np.column_stack((base, visibility)), np.column_stack((mirrored, visibility))


def training_halves(root, match, loader=load_skillcorner):
    rng = np.random.default_rng(match)
    halves = []
    fps = SKILLCORNER_FPS
    for positions, weights, teams, pitch, labels in loader(root, match):
        augmented = short_ball_dropouts(positions, fps, rng)
        for tracks in (positions, augmented):
            x, mirrored = sequence_features(tracks, weights, teams, pitch, fps)
            halves.append((x, mirrored, labels))
    return halves


def evaluation_halves(root, match, stress):
    idsse = isinstance(match, str)
    fps = IDSSE_FPS if idsse else SKILLCORNER_FPS
    load = load_idsse if idsse else load_skillcorner
    rng = np.random.default_rng(int(match[1:], 36) if idsse else match)
    halves = []
    for positions, weights, teams, pitch, labels in load(root, match):
        if stress:
            positions = mask_ball(positions, fps, rng, mean_gap=12, coverage=0.60)
        missing = ~np.isfinite(positions[:, -1]).all(axis=1)
        halves.append((*sequence_features(positions, weights, teams, pitch, fps), labels, missing, fps))
    return halves


def normalize(half_data):
    sample = np.concatenate([side[::16] for x, mirrored, _ in half_data for side in (x, mirrored)])
    mean = np.nanmean(sample, axis=0)
    scale = np.nanstd(sample, axis=0)
    return mean, np.maximum(scale, 1e-3)


def batches(data, rng, partial_start):
    windows = [(index, start) for index, (_, _, labels) in enumerate(data)
               for start in range(0, len(labels), WINDOW)
               if np.any(labels[start:start + WINDOW] != metrics.X)]
    rng.shuffle(windows)
    mirror_labels = np.array([metrics.A, metrics.H, metrics.D, metrics.X], dtype=np.int8)
    width = data[0][0].shape[1]
    for start in range(0, len(windows), BATCH_SIZE):
        selection = windows[start:start + BATCH_SIZE]
        x = np.full((len(selection), WINDOW, width), np.nan, dtype=np.float32)
        y = np.full((len(selection), WINDOW), metrics.X, dtype=np.int8)
        padding = np.ones((len(selection), WINDOW), dtype=bool)
        for row, (index, offset) in enumerate(selection):
            normal, mirrored, labels = data[index]
            segment = slice(offset, offset + WINDOW)
            targets = labels[segment]
            use_mirror = rng.random() < 0.5
            x[row, :len(targets)] = (mirrored if use_mirror else normal)[segment]
            y[row, :len(targets)] = mirror_labels[targets] if use_mirror else targets
            padding[row, :len(targets)] = False
        partial = np.array([index >= partial_start for index, _ in selection])
        yield x, y, padding, partial


def supervised_loss(logits, labels, partial):
    # For partially labeled sources, condition on live play and learn only H vs A.
    joint = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                                        ignore_index=metrics.X, reduction="none").view_as(labels)
    conditional = nn.functional.cross_entropy(
        logits[:, :, :2].reshape(-1, 2), labels.flatten().masked_fill(
            labels.flatten() == metrics.D, metrics.X),
        ignore_index=metrics.X, reduction="none").view_as(labels)
    return torch.where(partial[:, None], conditional, joint).sum() / (labels != metrics.X).sum()


@torch.inference_mode()
def sequence_probabilities(model, normal, mirrored, fps, device):
    # Sequence inference runs at 10 Hz. Restore the official frame grid before decoding.
    frames = np.arange(0, len(normal), fps / SKILLCORNER_FPS)
    sample = np.minimum(np.rint(frames).astype(int), len(normal) - 1)
    central = WINDOW - 2 * CONTEXT
    p = np.empty((len(sample), 3), dtype=np.float32)
    for start in range(0, len(sample), central):
        stop = min(len(sample), start + central)
        left, right = max(0, start - CONTEXT), min(len(sample), stop + CONTEXT)
        index = sample[left:right]
        window = torch.from_numpy(np.stack((normal[index], mirrored[index]))).to(device)
        scores = model(window).softmax(-1).cpu().numpy()
        merged = (scores[0] + scores[1][:, [1, 0, 2]]) / 2
        p[start:stop] = merged[start - left:stop - left]
    if fps == SKILLCORNER_FPS:
        return p
    return np.column_stack([np.interp(np.arange(len(normal)), sample, p[:, k]) for k in range(3)])


def score_half(p, labels, missing, fps):
    pred = metrics.label_codes(decode_costs(np.log(np.maximum(p, 1e-6)), fps, *DECODER_COSTS))
    runs = metrics.valid_runs(labels)
    spells = metrics.spell_scores(labels, pred, runs)
    target = metrics.possession_spells(labels, runs)
    found = metrics.possession_spells(pred, runs)
    matches = metrics.matched_gt_flags(metrics.spell_overlaps(target, found), len(target), 0.5)
    gap_bounds = np.flatnonzero(np.diff(np.r_[False, missing, False])).reshape(-1, 2)
    gap_spells = [idx for idx, (start, stop, _) in enumerate(target)
                  if np.any(np.minimum(gap_bounds[:, 1], stop) -
                            np.maximum(gap_bounds[:, 0], start) >= fps)]
    cm = metrics.scored_confusion(metrics.confusion(labels, pred))
    return [spells["mean_f1"], metrics.macro_f1(cm), metrics.per_class_prf(cm, metrics.D)[2],
            int(matches[gap_spells].sum()), len(gap_spells)]


def evaluate(model, records, device=None):
    by_match, total = [], []
    for halves in records:
        scores = []
        for normal, mirrored, labels, missing, fps in halves:
            if device is None:
                p = probabilities(model, normal[:, :BASE_FEATURES], mirrored[:, :BASE_FEATURES])
            else:
                p = sequence_probabilities(model, normal, mirrored, fps, device)
            scores.append(score_half(p, labels, missing, fps))
        by_match.append(np.mean(scores, axis=0))
        total.extend(scores)
    average = np.mean(by_match, axis=0)
    count = np.sum(total, axis=0)
    return dict(spell_f1=float(average[0]), macro_f1=float(average[1]), dead_f1=float(average[2]),
                gap_hits=int(count[3]), gap_spells=int(count[4]))


def fit(args):
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    match_ids = {int(item["id"]) for item in json.loads((args.skillcorner / "matches.json").read_text())}
    if not (DEV_MATCHES | TEST_MATCHES).issubset(match_ids):
        raise ValueError("pinned A-League development or test matches are missing")
    train_ids = sorted(match_ids - DEV_MATCHES - TEST_MATCHES)
    reference, costs, lags = load_model(args.reference)
    if tuple(costs) != DECODER_COSTS or lags != LAG_SECONDS:
        raise ValueError("expected the frozen lagged LightGBM reference")
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        jobs = [pool.submit(training_halves, args.skillcorner, match) for match in train_ids]
        if args.legacy_skillcorner:
            jobs.extend(pool.submit(training_halves, args.legacy_skillcorner, match,
                                    load_legacy_skillcorner) for match in LEGACY_MATCHES)
        development_jobs = [pool.submit(evaluation_halves, args.skillcorner, match, False)
                            for match in sorted(DEV_MATCHES)]
        primary = [half for job in jobs[:len(train_ids)] for half in job.result()]
        partial_start = len(primary)
        training = primary + [half for job in jobs[len(train_ids):] for half in job.result()]
        del primary
        if not args.conditional_legacy:
            partial_start = len(training)
        development = [job.result() for job in development_jobs]
        mean, scale = normalize(training)
        model = ARCHITECTURES[args.architecture](mean, scale).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=metrics.X)
        best = -1.0
        checkpoint = args.output / f"{args.architecture}.pt"
        ref_dev = evaluate(reference, development)
        print(f"Training {args.architecture} on {sum(len(y) for _, _, y in training):,} sequence frames "
              f"across {len(jobs)} matches; reference dev {ref_dev}", flush=True)
        history = []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []
            for features, labels, padding, partial in batches(training, rng, partial_start):
                x = torch.from_numpy(features).to(device)
                y = torch.from_numpy(labels).to(device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                pad = torch.from_numpy(padding).to(device) if args.architecture == "transformer" else None
                amp = args.architecture == "transformer" and device.type == "cuda"
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    # Padding has no tracking frame. Missing-ball and X-labeled frames still attend.
                    output = model(x, pad)
                    if args.conditional_legacy:
                        loss = supervised_loss(output, y, torch.from_numpy(partial).to(device))
                    else:
                        loss = criterion(output.flatten(0, 1), y.flatten())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                losses.append(loss.detach())
            model.eval()
            dev = evaluate(model, development, device)
            dev["epoch"] = epoch
            history.append(dev)
            print(f"Epoch {epoch}: loss={torch.stack(losses).mean().item():.4f}, dev={dev}", flush=True)
            if dev["spell_f1"] > best:
                best = dev["spell_f1"]
                torch.save({"state": model.state_dict(), "epoch": epoch,
                            "architecture": args.architecture}, checkpoint)
        del training, development, jobs, development_jobs
        model, epoch = restore(checkpoint, device)
        print(f"Frozen {args.architecture} epoch {epoch}, development spellF1 {best:.4f}", flush=True)
        sets = [("A-League", args.skillcorner, sorted(TEST_MATCHES), False),
                ("A-League masked", args.skillcorner, sorted(TEST_MATCHES), True),
                ("Bundesliga masked", args.idsse, IDSSE_TEST, True)]
        result = {}
        for name, root, split, stress in sets:
            records = list(pool.map(evaluation_halves, [root] * len(split), split, [stress] * len(split)))
            result[name] = {"lightgbm": evaluate(reference, records),
                            args.architecture: evaluate(model, records, device)}
            print(f"Held-out {name}: {result[name]}", flush=True)
    (args.output / "benchmark.json").write_text(json.dumps(
        {"architecture": args.architecture, "selected_epoch": epoch,
         "legacy_skillcorner_matches": list(LEGACY_MATCHES) if args.legacy_skillcorner else [],
         "conditional_legacy": args.conditional_legacy,
         "reference_development": ref_dev, "history": history, "held_out": result},
        indent=2) + "\n")


def restore(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state"]
    model = ARCHITECTURES[payload.get("architecture", "gru")](state["mean"], state["scale"])
    model.load_state_dict(state)
    return model.to(device).eval(), payload["epoch"]


def predict(args):
    device = torch.device(args.device)
    model, epoch = restore(args.model, device)
    positions, teams, pitch, fps = contest_tracking(args.data_dir)
    labels = np.full(len(positions), "X")
    for _, start, last in half_video_blocks(GAME):
        normal, mirrored = sequence_features(positions[start:last + 1], None, teams, pitch, fps)
        p = sequence_probabilities(model, normal, mirrored, fps, device)
        labels[start:last + 1] = decode_costs(np.log(np.maximum(p, 1e-6)), fps, *DECODER_COSTS)
    write_submission(args.output, labels)
    print(f"Wrote {len(labels):,} contest frames using {model.architecture} frozen at epoch {epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--skillcorner", type=Path, required=True)
    training.add_argument("--legacy-skillcorner", type=Path)
    training.add_argument("--conditional-legacy", action="store_true",
                          help="use legacy matches for home vs away loss only")
    training.add_argument("--idsse", type=Path, required=True)
    training.add_argument("--reference", type=Path, required=True)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--architecture", choices=ARCHITECTURES, default="gru")
    inference = commands.add_parser("predict")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--data-dir", type=Path)
    for command in (training, inference):
        command.add_argument("--device", default="cpu")
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        if args.conditional_legacy and not args.legacy_skillcorner:
            parser.error("--conditional-legacy requires --legacy-skillcorner")
        fit(args)
    else:
        predict(args)
