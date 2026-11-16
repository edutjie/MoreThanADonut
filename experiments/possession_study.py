"""Compare models on the same repaired tracking, splits, and spell decoder."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluation import possession_metrics as metrics
from experiments.possession_tcn import PlayerTCN, PossessionTCN
from external_possession import decode_costs
from sequence_possession import (
    ARCHITECTURES, CONTEXT, DECODER_COSTS, SEED, WINDOW, batches, normalize,
    restore, score_prediction,
)


MIRROR = np.array([1, 0, 2, 3])
AUTOML_FAMILIES = ("GBM", "CAT", "XGB", "XT", "NN_TORCH")


def array(folder, field):
    return np.load(folder / f"{field}.npy", mmap_mode="r")


def sampled_labels(folder):
    return array(folder, "labels")[array(folder, "sample")]


def paths(root, split):
    return sorted((root / split).glob("*"))


def mirror_objects(x):
    mirrored = x.copy()
    mirrored[..., -1] *= -1
    return mirrored


def model_rows(folder, objects=False):
    if objects:
        x = array(folder, "objects")
        return x, mirror_objects(x)
    return array(folder, "x"), array(folder, "mirror")


@torch.inference_mode()
def dense_scores(model, normal, mirror, device):
    central = WINDOW - 2 * CONTEXT
    p = np.empty((len(normal), 3), dtype=np.float32)
    for start in range(0, len(p), central):
        stop = min(len(p), start + central)
        left, right = max(0, start - CONTEXT), min(len(p), stop + CONTEXT)
        x = torch.from_numpy(np.stack((normal[left:right], mirror[left:right])))
        probs = model(x.to(device, dtype=torch.float32)).softmax(-1).cpu().numpy()
        merged = (probs[0] + probs[1][:, MIRROR[:3]]) / 2
        p[start:stop] = merged[start - left:stop - left]
    return p


def decode_sampled(p, frames, length, fps):
    restored = np.column_stack([np.interp(np.arange(length), frames, p[:, k]) for k in range(3)])
    return decode_costs(np.log(np.maximum(restored, 1e-6)), fps, *DECODER_COSTS)


def evaluate(folders, predict):
    groups = defaultdict(lambda: defaultdict(list))
    for folder in folders:
        match, _, version = folder.name.split("-")
        group = ("Bundesliga" if match.startswith("J") else "A-League") + " " + version
        labels, inside, missing, frames = (array(folder, f) for f in ("labels", "inside", "missing", "sample"))
        fps = int((folder / "fps").read_text())
        pred = decode_sampled(predict(folder), frames, len(labels), fps)
        score = score_prediction(pred, labels, missing, fps)
        valid = inside & (labels != metrics.X)
        dead = labels[valid] == metrics.D
        predicted_dead = metrics.label_codes(pred)[valid] == metrics.D
        score.extend((np.sum(dead & predicted_dead), np.sum(predicted_dead & ~dead),
                      np.sum(dead & ~predicted_dead)))
        groups[group][match].append(score)
    report = {}
    for group, matches in groups.items():
        scores = [score for match in matches.values() for score in match]
        spell, macro, dead = np.mean([np.mean(match, axis=0)[:3]
                                      for match in matches.values()], axis=0)
        hits, gaps, tp, fp, fn = np.sum(scores, axis=0)[3:]
        report[group] = dict(spell_f1=float(spell), macro_f1=float(macro), dead_f1=float(dead),
                             in_pitch_dead_f1=float(2 * tp / max(2 * tp + fp + fn, 1)),
                             gap_hits=int(hits), gap_spells=int(gaps))
    return report


def player_batches(training, rng):
    window = 256
    offsets = [(i, start) for i, (_, _, y) in enumerate(training)
               for start in range(0, len(y), window) if (y[start:start + window] != metrics.X).any()]
    rng.shuffle(offsets)
    for start in range(0, len(offsets), 12):
        selected = offsets[start:start + 12]
        players = max(training[i][0].shape[1] - 1 for i, _ in selected)
        x = np.zeros((len(selected), window, players + 1, 9), dtype=np.float16)
        y = np.full((len(selected), window), metrics.X, dtype=np.int64)
        for row, (i, offset) in enumerate(selected):
            objects, _, labels = training[i]
            segment = slice(offset, offset + window)
            tracks = objects[segment]
            target = labels[segment]
            if rng.random() < 0.5:
                tracks, target = mirror_objects(tracks), MIRROR[target]
            x[row, :len(tracks), :tracks.shape[1] - 1] = tracks[:, :-1]
            x[row, :len(tracks), -1] = tracks[:, -1]
            y[row, :len(target)] = target
        yield x, y, None, None


def columns(size):
    return [f"f{i:03d}" for i in range(size)]


def rolling_features(x):
    """Generate fixed window summaries within a single 10 Hz half."""
    from scipy.ndimage import uniform_filter1d

    selected = x[:, [0, 1, 4, 15, 20, 35, 40, 111]]
    visible = np.isfinite(selected).astype(np.float32)
    filled = np.nan_to_num(selected, nan=0.0)
    output = [x]
    for width in (5, 15, 41):
        fraction = uniform_filter1d(visible, width, axis=0, mode="constant")
        valid = fraction > 1e-4
        mean = np.divide(uniform_filter1d(filled, width, axis=0, mode="constant"),
                         fraction, out=np.full_like(filled, np.nan), where=valid)
        square = np.divide(uniform_filter1d(filled ** 2, width, axis=0, mode="constant"),
                           fraction, out=np.full_like(filled, np.nan), where=valid)
        output.extend((mean, np.sqrt(np.maximum(square - mean ** 2, 0)), fraction))
    return np.column_stack(output)


def feature_transform(root, output, mode):
    import joblib
    import pandas as pd
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator

    generator = AutoMLPipelineFeatureGenerator()
    precursor = rolling_features if mode == "rolling-auto" else None
    x, _ = rows(paths(root, "train"), precursor)
    names = columns(x.shape[1])
    generator.fit(pd.DataFrame(x, columns=names))
    joblib.dump(generator, output / "features.joblib")

    def transform(x):
        if precursor:
            x = precursor(x)
        return generator.transform(pd.DataFrame(x, columns=names)).to_numpy(dtype=np.float32)

    source = array(paths(root, "train")[0], "x")
    sample = source[:10000]
    generated = transform(sample)
    info = dict(input=sample.shape[1], generator_input=len(names), output=list(generator.features_out),
                equal_on_sample=bool(np.array_equal(generated, sample, equal_nan=True)))
    (output / "features.json").write_text(json.dumps(info, indent=2) + "\n")
    print(info, flush=True)
    return transform


def rows(folders, transform, step=2):
    blocks, labels = [], []
    for folder in folders:
        y = sampled_labels(folder)
        i = np.flatnonzero(y != metrics.X)[::step]
        for j, x in enumerate(model_rows(folder)):
            if transform:
                x = transform(x)
            blocks.append(x[i])
            labels.append(y[i] if j == 0 else MIRROR[y[i]])
    return np.concatenate(blocks), np.concatenate(labels)


def temporal(args, transform):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    player = args.model == "player-tcn"
    training = []
    for folder in paths(args.cache, "train"):
        x = array(folder, "objects" if player else "x")
        other = None if player else array(folder, "mirror")
        if transform:
            x, other = transform(x), transform(other)
        training.append((x, other, sampled_labels(folder)))
    if player:
        model = PlayerTCN().to(args.device)
    else:
        factory = PossessionTCN if args.model == "tcn" else ARCHITECTURES[args.model]
        model = factory(*normalize(training)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)

    def predict(folder):
        x, mirror = model_rows(folder, player)
        if transform:
            x, mirror = transform(x), transform(mirror)
        return dense_scores(model, x, mirror, args.device)

    best = -1
    checkpoint = args.output / "model.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        stream = player_batches(training, rng) if player else batches(training, rng, len(training))
        total, count = 0.0, 0
        for x, y, padding, _ in stream:
            features = torch.as_tensor(x, device=args.device, dtype=torch.float32)
            target = torch.as_tensor(y, device=args.device, dtype=torch.long)
            pad = torch.as_tensor(padding, device=args.device) if args.model == "transformer" else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                enabled=args.device.startswith("cuda") and args.model != "gru"):
                logits = model(features, pad)
                loss = nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(),
                                                   ignore_index=metrics.X)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += loss.detach()
            count += 1
        model.eval()
        dev = evaluate(paths(args.cache, "dev"), predict)["A-League natural"]
        history.append(dict(epoch=epoch, loss=float(total / count), **dev))
        print(history[-1], flush=True)
        if dev["spell_f1"] > best:
            best = dev["spell_f1"]
            torch.save(dict(state=model.state_dict(), epoch=epoch), checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True)["state"])
    return predict, history


def tabular(args, transform):
    import joblib

    x, y = rows(paths(args.cache, "train"), transform, step=args.row_stride)
    if args.model == "lgbm":
        from external_possession import classifier

        model = classifier("lightgbm", args.workers).fit(x, y)
        joblib.dump(model, args.output / "model.joblib")

        def prob(x):
            return model.predict_proba(x)

    else:
        import pandas as pd
        from autogluon.tabular import TabularPredictor

        def frame(x, y=None):
            table = pd.DataFrame(x, columns=columns(x.shape[1]))
            if y is not None:
                table["label"] = y
            return table

        dev_x, dev_y = rows(paths(args.cache, "dev"), transform, step=4)
        model = TabularPredictor(label="label", problem_type="multiclass",
                                 eval_metric="log_loss", path=str(args.output / "autogluon"))
        fit_args = ({"ag_args_fit": {"max_time_limit": args.family_time_limit}}
                    if args.family_time_limit is not None else {})
        model.fit(frame(x, y), tuning_data=frame(dev_x, dev_y), time_limit=args.time_limit,
                  num_cpus=args.workers, num_gpus=int(args.device.startswith("cuda")),
                  num_stack_levels=0, num_bag_folds=0,
                  hyperparameters={name: fit_args for name in args.automl_families})
        model.leaderboard(silent=True).to_csv(args.output / "leaderboard.csv", index=False)

        def prob(x):
            return model.predict_proba(frame(x))[[0, 1, 2]].to_numpy()

    def predict(folder):
        x, mirror = model_rows(folder)
        if transform:
            x, mirror = transform(x), transform(mirror)
        return (prob(x) + prob(mirror)[:, MIRROR[:3]]) / 2

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("baseline", "gru", "transformer", "tcn", "player-tcn", "lgbm", "automl"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", choices=("original", "auto", "rolling-auto"), default="original")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--row-stride", type=int, default=2,
                        help="keep every nth graded 10 Hz row for tabular fits")
    parser.add_argument("--time-limit", type=int, default=180)
    parser.add_argument("--family-time-limit", type=int)
    parser.add_argument("--automl-families", nargs="+", choices=AUTOML_FAMILIES, default=AUTOML_FAMILIES)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", 48), ("dev", 8), ("test", 24)):
        if len(paths(args.cache, split)) != count:
            parser.error(f"expected {count} prepared arrays in {split}")
    if args.features != "original" and args.model in {"baseline", "player-tcn"}:
        parser.error("AutoGluon features apply to the frame summaries only")
    if args.model == "baseline" and args.reference is None:
        parser.error("baseline requires a frozen GRU --reference")
    if args.row_stride < 1:
        parser.error("--row-stride must be positive")
    transform = (feature_transform(args.cache, args.output, args.features)
                 if args.features != "original" else None)
    history = []
    if args.model == "baseline":
        model, _ = restore(args.reference, args.device)

        def predict(folder):
            return dense_scores(model, *model_rows(folder), args.device)

    elif args.model in {"lgbm", "automl"}:
        predict = tabular(args, transform)
    else:
        predict, history = temporal(args, transform)
    result = dict(model=args.model, features=args.features, row_stride=args.row_stride, history=history,
                  development=evaluate(paths(args.cache, "dev"), predict),
                  held_out=evaluate(paths(args.cache, "test"), predict))
    (args.output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
