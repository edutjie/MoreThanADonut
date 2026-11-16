"""Probe frozen MOMENT and Mantis embeddings on local tracking windows."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from evaluation import possession_metrics as metrics
from experiments.possession_study import MIRROR, array, evaluate, paths, sampled_labels


# Tracking feature indexes are defined in external_possession.tracking_features.
CHANNELS = [0, 1, 4, 15, 35, 111]  # confidence, pitch exit, speed, H/A proximity, visibility
SWAP = [0, 1, 2, 4, 3, 5]
OFFSETS = np.arange(-48, 16)
MODEL_IDS = {"moment": "AutonLab/MOMENT-1-small", "mantis": "paris-noah/Mantis-8M"}
REVISIONS = {"moment": "411e288267f82cce86296dbe4d6c8bc533cc162f",
             "mantis": "bc7d5ab40c02133386a28e2c127f35c17c86901d"}


def encoder(name, device):
    if name == "moment":
        from momentfm import MOMENTPipeline

        model = MOMENTPipeline.from_pretrained(MODEL_IDS[name], revision=REVISIONS[name],
                                               model_kwargs={"task_name": "embedding"})
        model.init()
        model.to(device).eval()

        def encode(x):
            return model.embed(x_enc=x, reduction="none").embeddings.mean(dim=2)

    else:
        from mantis.architecture import MantisV1

        model = MantisV1(device=device, return_transf_layer=2, output_token="combined")
        model = model.from_pretrained(MODEL_IDS[name], revision=REVISIONS[name]).to(device).eval()

        def encode(x):
            series = x.flatten(0, 1).unsqueeze(1)
            series = F.interpolate(series, size=512, mode="linear", align_corners=False)
            return model(series).reshape(*x.shape[:2], -1)

    return encode


def anchor_rows(x, centers):
    window = centers[:, None] + OFFSETS
    inside = (window >= 0) & (window < len(x))
    return np.where(inside[..., None], x[np.clip(window, 0, len(x) - 1)][:, :, CHANNELS], np.nan)


@torch.inference_mode()
def embeddings(x, centers, encode, device):
    reduced = []
    projection = None
    for start in range(0, len(centers), 128):
        values = anchor_rows(x, centers[start:start + 128])
        tensor = torch.from_numpy(np.nan_to_num(values, nan=0.0)).transpose(1, 2).to(device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            representation = encode(tensor)
        if projection is None:
            width = representation.shape[-1]
            rng = np.random.default_rng(42)
            projection = torch.as_tensor(rng.standard_normal((width, 32)) / np.sqrt(32),
                                         device=device, dtype=torch.float32)
        reduced.append((representation.float() @ projection).cpu().numpy())
    features = np.concatenate(reduced)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite foundation encoding")
    return features


def head_probabilities(classifier, normal, mirror):
    return (classifier.predict_proba(normal) + classifier.predict_proba(mirror)[:, MIRROR[:3]]) / 2


def restore_grid(probabilities, centers, length):
    return np.column_stack([np.interp(np.arange(length), centers, p) for p in probabilities.T])


def frozen_probabilities(name, directory, device):
    encode = encoder(name, device)
    _, classifier = joblib.load(directory / "heads.joblib")

    def predict(x, mirror):
        centers = np.arange(0, len(x), 5)
        z = embeddings(x, centers, encode, device)
        normal, swapped = (np.column_stack((side[centers], features.reshape(len(centers), -1)))
                           for side, features in ((x, z), (mirror, z[:, SWAP])))
        return restore_grid(head_probabilities(classifier, normal, swapped), centers, len(x))

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(MODEL_IDS))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    encode = encoder(args.model, args.device)

    def feature_rows(folder, stride):
        x, mirrored = array(folder, "x"), array(folder, "mirror")
        centers = np.arange(0, len(x), stride)
        cache = args.output / f"{folder.name}-{stride}.npy"
        if cache.exists():
            z = np.load(cache)
        else:
            z = embeddings(x, centers, encode, args.device)
            np.save(cache, z, allow_pickle=False)
            print(f"Embedded {folder.name} ({len(centers)} windows)", flush=True)
        raw = (x[centers], mirrored[centers])
        with_embeddings = tuple(np.column_stack((side, codes.reshape(len(centers), -1)))
                                for side, codes in zip(raw, (z, z[:, SWAP])))
        return centers, raw, with_embeddings

    training = paths(args.cache, "train")
    train_x, train_y = [[], []], []
    for folder in training:
        centers, raw, fused = feature_rows(folder, 10)
        y = sampled_labels(folder)[centers]
        keep = y != metrics.X
        for features, bucket in zip((raw, fused), train_x):
            bucket.extend(x[keep] for x in features)
        train_y.extend((y[keep], MIRROR[y[keep]]))
    labels = np.concatenate(train_y)
    heads = []
    for x in train_x:
        predictor = make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(),
                                  LogisticRegression(max_iter=250, C=0.2))
        heads.append(predictor.fit(np.concatenate(x), labels))
    joblib.dump(heads, args.output / "heads.joblib")

    def predict(folder, head):
        centers, raw, fused = feature_rows(folder, 5)
        normal, mirror = (raw, fused)[head]
        classifier = heads[head]
        probs = head_probabilities(classifier, normal, mirror)
        length = len(array(folder, "x"))
        return restore_grid(probs, centers, length)

    result = {}
    for head, name in enumerate(("linear-control", args.model)):
        def run(folder):
            return predict(folder, head)

        result[name] = dict(development=evaluate(paths(args.cache, "dev"), run),
                            held_out=evaluate(paths(args.cache, "test"), run))
    result["model"] = args.model
    (args.output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
