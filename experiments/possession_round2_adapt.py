"""Track-only target adaptation with deterministic, isolated match updates."""
import copy

import numpy as np
import torch
from torch import nn

from evaluation import possession_metrics as m
from sequence_possession import WINDOW, normalize, sequence_probabilities


def target_stats(halves):
    data = [(x, mx, None) for _, x, mx in halves]
    return normalize(data)


def normalized_model(model, stats, blend):
    result = copy.deepcopy(model)
    mean, scale = [torch.as_tensor(v, device=model.mean.device) for v in stats]
    present = mean.isfinite() & scale.isfinite()
    result.mean.copy_(model.mean.lerp(torch.where(present, mean, model.mean), blend))
    result.scale.copy_(model.scale.lerp(torch.where(present, scale, model.scale), blend))
    return result


def prediction(model, halves):
    return [sequence_probabilities(model, x, mx, fps, model.mean.device) for fps, x, mx in halves]


def self_train(model, halves, reference, decode, threshold, external, seed):
    pseudo = []
    for (fps, x, mx), probabilities in zip(halves, reference):
        labels = decode(probabilities, fps)
        selected = np.full(len(labels), m.X, dtype=np.int8)
        for start, stop, label in m.segments(labels, [(0, len(labels))]):
            spell = probabilities[start:stop]
            confidences = spell[:, label]
            alternatives = spell[:, [s for s in range(3) if s != label]].max(axis=1)
            if confidences.mean() >= threshold and (confidences - alternatives).mean() >= 0.4:
                selected[start:stop] = label
        frames = np.minimum(np.rint(np.arange(0, len(labels), fps / 10)).astype(int), len(labels) - 1)
        pseudo.append((x[frames], mx[frames], selected[frames]))
    candidates = [(index, start) for index, row in enumerate(pseudo)
                  for start in range(0, len(row[2]) - WINDOW + 1, WINDOW // 2)
                  if (row[2][start:start + WINDOW] != m.X).any()]
    if not candidates:
        return model
    rng = np.random.default_rng(seed)
    result = copy.deepcopy(model)
    result.requires_grad_(False)
    result.head.requires_grad_(True)
    optimizer = torch.optim.AdamW(result.head.parameters(), lr=0.0002)
    mirrored = np.array([m.A, m.H, m.D, m.X])
    for _ in range(16):
        features, targets = [], []
        for _ in range(4):
            index, start = candidates[rng.integers(len(candidates))]
            data, flipped, labels = pseudo[index]
            segment = slice(start, start + WINDOW)
            use_flip = rng.random() < 0.5
            features.append((flipped if use_flip else data)[segment])
            targets.append(mirrored[labels[segment]] if use_flip else labels[segment])
        for _ in range(4):
            data, flipped, labels = external[rng.integers(len(external))]
            start = rng.integers(len(labels) - WINDOW + 1)
            segment = slice(start, start + WINDOW)
            use_flip = rng.random() < 0.5
            features.append((flipped if use_flip else data)[segment])
            targets.append(mirrored[labels[segment]] if use_flip else labels[segment])
        x = torch.from_numpy(np.stack(features)).to(result.mean.device)
        y = torch.from_numpy(np.stack(targets)).long().to(result.mean.device)
        logits = result(x)
        pseudo_loss, external_loss = (
            nn.functional.cross_entropy(a.flatten(0, 1), b.flatten(), ignore_index=m.X, reduction="sum")
            / (b != m.X).sum().clamp(min=1)
            for a, b in ((logits[:4], y[:4]), (logits[4:], y[4:])))
        optimizer.zero_grad(set_to_none=True)
        (pseudo_loss * 0.5 + external_loss * 0.5).backward()
        optimizer.step()
    return result
