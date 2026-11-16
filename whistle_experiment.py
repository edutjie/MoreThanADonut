"""Train a whistle cue detector on public RoboCup audio and run it locally."""

import argparse
import csv
import io
import json
import zipfile
from math import gcd
from pathlib import Path

import joblib
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.ensemble import HistGradientBoostingClassifier

from dead_ball import AUDIO_HOP, AUDIO_WINDOW, SAMPLE_RATE, read_audio, whistle_frames
from games import GAME, GAMES, half_video_blocks, require, source_frames


TRAIN = ("WhistleTest_tuhhnao", "Lab_")
DEV = ("HTWK_Dutch", "SRC_Dutch", "TestBHuman_")
TEST = ("LUnited", "HTWK_HULKs")


def features(samples):
    windows = np.lib.stride_tricks.sliding_window_view(samples, AUDIO_WINDOW)[::AUDIO_HOP]
    taper = np.hanning(AUDIO_WINDOW)
    parts = []
    for offset in range(0, len(windows), 4096):
        power = np.abs(np.fft.rfft(windows[offset:offset + 4096] * taper)) ** 2
        audible = power[:, 16:220]
        whistle = power[:, 64:161]
        triple = whistle[:, :-2] + whistle[:, 1:-1] + whistle[:, 2:]
        frequency = triple.argmax(axis=1)
        concentration = triple.max(axis=1) / np.maximum(audible.sum(axis=1), 1e-12)
        bands = 10 * np.log10(np.maximum(audible.reshape(-1, 51, 4).mean(axis=2), 1e-12))
        relative = bands - np.median(bands, axis=1, keepdims=True)
        parts.append(np.column_stack((relative, frequency, concentration)).astype(np.float32))
    return np.concatenate(parts)


def grouped_candidates(scores, threshold):
    hits = np.flatnonzero(scores >= threshold)
    if not len(hits):
        return np.empty(0)
    segments = np.split(hits, np.flatnonzero(np.diff(hits) > 10) + 1)
    # Use the earliest supported peak in each short cluster.
    onsets = [frames[0] for frames in segments if len(frames) >= 4]
    return (np.array(onsets) * AUDIO_HOP + AUDIO_WINDOW / 2) / SAMPLE_RATE


def recordings(path):
    with zipfile.ZipFile(path) as archive:
        entries = json.loads(archive.read("whistle/whistledb.json"))["audioFiles"]
        for entry in entries:
            name = entry["path"]
            category = next(group for group, prefixes in enumerate((TRAIN, DEV, TEST))
                            if name.startswith(prefixes))
            channel = next(i for i, ch in enumerate(entry["channels"]) if ch["completelyLabeled"])
            rate, audio = wavfile.read(io.BytesIO(archive.read(f"whistle/{name}")))
            mono = audio[:, channel].astype(np.float32)
            if audio.dtype == np.uint8:
                mono = (mono - 128) / 128
            else:
                mono /= np.iinfo(audio.dtype).max
            factor = gcd(rate, SAMPLE_RATE)
            samples = resample_poly(mono, SAMPLE_RATE // factor, rate // factor)
            spans = entry["channels"][channel]["whistleLabels"]
            times = np.array([(s["start"] / rate, s["end"] / rate) for s in spans])
            yield category, name, samples, times


def counts(prediction, reference):
    """Match each detection to at most one labeled event, with 250 ms slack."""
    matched = set()
    for time in prediction:
        for index, (start, end) in enumerate(reference):
            if index not in matched and start - 0.25 <= time <= end + 0.25:
                matched.add(index)
                break
    hits = len(matched)
    return np.array([hits, len(prediction) - hits, len(reference) - hits])


def f1(totals):
    hit, false, missed = totals
    return 2 * hit / max(1, 2 * hit + false + missed)


def train(archive, output):
    sets = [[], [], []]
    for group, name, samples, times in recordings(archive):
        x = features(samples)
        baseline = whistle_frames(samples, 100) / 100
        if group == 0:
            center = (np.arange(len(x)) * AUDIO_HOP + AUDIO_WINDOW / 2) / SAMPLE_RATE
            y = np.zeros(len(x), dtype=bool)
            for start, end in times:
                y |= (center >= start) & (center < end)
            stride = np.arange(len(y)) % 5 == 0
            sets[group].append((x[y | stride], y[y | stride]))
        else:
            sets[group].append((name, x, times, baseline))

    x = np.concatenate([a for a, _ in sets[0]])
    y = np.concatenate([b for _, b in sets[0]])
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=0,
                                          early_stopping=False, class_weight="balanced").fit(x, y)
    print(f"External training windows: {len(y):,}, positive: {y.sum():,}", flush=True)
    scored = [[(name, model.predict_proba(x)[:, 1], y, baseline) for name, x, y, baseline in rows]
              for rows in sets[1:]]
    grid = (0.50, 0.70, 0.85, 0.95, 0.98, 0.995)
    totals = [sum(counts(grouped_candidates(p, t), y) for _, p, y, _ in scored[0]) for t in grid]
    best = max(range(len(grid)), key=lambda index: (f1(totals[index]), -totals[index][1], grid[index]))
    threshold = grid[best]
    for name, rows in zip(("development", "held-out"), scored):
        tonal = sum(counts(baseline, y) for _, _, y, baseline in rows)
        learned = sum(counts(grouped_candidates(p, threshold), y) for _, p, y, _ in rows)
        print(f"{name} tonal F1={f1(tonal):.4f} counts={tonal.tolist()}; "
              f"learned F1={f1(learned):.4f} counts={learned.tolist()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, threshold), output)
    print(f"Frozen threshold {threshold}")


def detect(checkpoint, data_dir, output, broadcast_map=None):
    model, threshold = joblib.load(checkpoint)
    source = "broadcast" if broadcast_map is not None else "tactical"
    video, = require(["video"], data_dir=data_dir, source=source)
    fps = GAMES[GAME]["fps"]
    blocks = ([(0, source_frames("broadcast") - 1)] if broadcast_map is not None
              else [(start, end + 1) for _, start, end in half_video_blocks(GAME)])
    frames = []
    for start, stop in blocks:
        samples = read_audio(video, start, stop, fps)
        scores = model.predict_proba(features(samples))[:, 1]
        onsets = np.rint(grouped_candidates(scores, threshold) * fps).astype(int) + start
        frames.extend(onsets[onsets < stop])
    print(f"Learned {source} whistle candidates: {len(frames)}")
    if broadcast_map is not None:
        with np.load(broadcast_map, allow_pickle=False) as archive:
            mapping = archive["tactical_frame"]
        if (mapping.shape != (source_frames("broadcast"),) or mapping.dtype.kind != "i"
                or np.any((mapping < -1) | (mapping >= source_frames()))):
            raise ValueError(f"{broadcast_map}: invalid broadcast-to-tactical map")
        indices = mapping[np.array(frames, dtype=int)]
        frames = np.unique(indices[indices >= 0])
        print(f"Mapped onto verified tactical frames: {len(frames)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "cue"])
        writer.writerows((frame, "whistle") for frame in frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--archive", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    scan = commands.add_parser("detect")
    scan.add_argument("--model", type=Path, required=True)
    scan.add_argument("--data-dir", type=Path)
    scan.add_argument("--broadcast-map", type=Path,
                      help="use broadcast audio and map cues to verified tactical frames")
    scan.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.archive, args.output)
    else:
        detect(args.model, args.data_dir, args.output, args.broadcast_map)
