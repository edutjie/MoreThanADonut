"""Prepare match-disjoint arrays for the possession architecture study."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.idsse_training import IDSSE_TEST
from external_possession import (
    DEV_MATCHES, IDSSE_FPS, SKILLCORNER_FPS, TEST_MATCHES, load_idsse,
    load_skillcorner, mask_ball, short_ball_dropouts,
)
from player_track_quality import corrupt_player_tracks, repair_player_tracks
from sequence_possession import sequence_features


def track_channels(tracks, teams, pitch, fps, weights):
    xy = tracks / pitch * 2 - 1
    visible = np.isfinite(xy).all(axis=-1)
    step = np.diff(tracks, axis=0, prepend=tracks[:1])
    speed = np.linalg.norm(step, axis=-1)
    step[(speed > max(1.0, 15 / fps)) | ~np.isfinite(speed)] = np.nan
    step[0] = np.nan
    distance = np.linalg.norm(tracks - tracks[:, -1:], axis=-1) / 20
    roles = np.broadcast_to(np.r_[1 - 2 * teams, 0], visible.shape)
    confidence = visible.astype(np.float32)
    if weights is not None:
        confidence[:, -1] *= weights
    return np.concatenate((
        xy, step * (fps / 10), np.minimum(distance[..., None], 5),
        visible[..., None], np.isfinite(step[..., :1]), confidence[..., None],
        roles[..., None],
    ), axis=-1).astype(np.float16)


def prepare(output, root, match, split):
    idsse = isinstance(match, str)
    fps = IDSSE_FPS if idsse else SKILLCORNER_FPS
    loader = load_idsse if idsse else load_skillcorner
    rng = np.random.default_rng(int(match[1:], 36) if idsse else match)
    for half, (tracks, weights, teams, pitch, labels) in enumerate(loader(root, match), 1):
        versions = {"natural": tracks}
        if split == "train":
            versions["augmented"] = corrupt_player_tracks(
                short_ball_dropouts(tracks, fps, rng), pitch, fps, rng)
        if split == "test":
            versions["masked"] = mask_ball(tracks, fps, rng, mean_gap=12, coverage=0.60)
        for version, raw in versions.items():
            folder = output / split / f"{match}-{half}-{version}"
            folder.mkdir(parents=True, exist_ok=True)
            repaired = repair_player_tracks(raw, fps)
            x, mirror = sequence_features(repaired, weights, teams, pitch, fps, True)
            ball = tracks[:, -1]
            inside = ((ball > 0.5) & (ball < pitch - 0.5)).all(axis=1)
            missing = ~np.isfinite(raw[:, -1]).all(axis=1)
            sample = np.rint(np.arange(0, len(x), fps / SKILLCORNER_FPS)).astype(int)
            sample = np.minimum(sample, len(x) - 1)
            for name, array in (
                ("x", x[sample]), ("mirror", mirror[sample]),
                ("objects", track_channels(repaired, teams, pitch, fps, weights)[sample]),
                ("labels", labels), ("inside", inside), ("missing", missing), ("sample", sample),
            ):
                np.save(folder / f"{name}.npy", array, allow_pickle=False)
            (folder / "fps").write_text(str(fps))
    return match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skillcorner", type=Path, required=True)
    parser.add_argument("--idsse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    matches = {int(r["id"]) for r in json.loads((args.skillcorner / "matches.json").read_text())}
    if not (DEV_MATCHES | TEST_MATCHES) <= matches:
        raise ValueError("expected the pinned A-League matches")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(prepare, args.output, args.skillcorner, match, split)
                for split, ids in (("train", matches - DEV_MATCHES - TEST_MATCHES),
                                   ("dev", DEV_MATCHES), ("test", TEST_MATCHES))
                for match in sorted(ids)]
        jobs += [pool.submit(prepare, args.output, args.idsse, match, "test")
                 for match in IDSSE_TEST]
        for job in jobs:
            print(f"Prepared {job.result()}", flush=True)


if __name__ == "__main__":
    main()
