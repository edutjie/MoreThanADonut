"""Label graded frames before each half's kickoff as dead ball.

A half can be graded from before its kickoff. The ball is not in play until the kickoff strike,
so every graded frame before it is a stoppage (`D`). A model trained on halves that start at the
kickoff has never seen pre-kickoff play, and it tends to predict live possessions on players who
are only warming up.

The kickoff frame is found from the provided inputs only:

1. The Smart Tagging `Kick Off (Start)` instance for the half bounds the search window.
2. Inside that window, the kickoff is the first frame where the tracked ball leaves a still
   position near the centre spot and keeps moving away from it.
3. If the tracking never shows that, the midpoint of the tagged window is used instead.

The thresholds below are physical and set before looking at any match: a ball placed on the spot
is still to within tracking noise, and a kicked ball moves well clear of it within half a second.

    python kickoff.py --input possession.csv --output possession_kickoff.csv

The rule only ever rewrites frames to `D`, only inside a graded half, and only before that
half's detected kickoff. It never writes `X`, because the scorer forward-fills `X` with the
previous label, which would carry live play into the stoppage.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.load_smart_tagging import KICKOFF_CODE, KICKOFF_TYPE  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402

# Params, fixed a priori in metres and seconds.
SPOT_RADIUS_M = 2.0       # the ball must rest this close to the centre spot
STILL_TOLERANCE_M = 0.3   # tracking jitter allowed while the ball rests
STILL_MIN_FRAMES = 3      # observed frames at rest before the strike
STILL_LOOKBACK_S = 0.5    # those frames must fall inside this lookback
MOVE_M = 0.3              # first displacement that counts as the strike
CONFIRM_M = 1.5           # the ball must get this far from rest ...
CONFIRM_S = 0.6           # ... within this time of the strike
NO_RETURN_S = 1.0         # and not come back to rest within this time
WINDOW_PAD_S = 2.0        # search this much either side of the tagged window
HALF_NAMES = {1: "1st half", 2: "2nd half"}

DEAD = "D"
LIVE = ("H", "A")


def kickoff_windows(tags, fps):
    """{half: (first_frame, stop_frame)} from `Kick Off (Start)` tags, padded on both sides."""
    pad = int(round(WINDOW_PAD_S * fps))
    windows = {}
    for tag in tags:
        if (tag["code"] or "").strip().lower() != KICKOFF_CODE:
            continue
        values = {(value or "").strip().lower() for value in tag["tags"].values()}
        if KICKOFF_TYPE not in values:
            continue
        for half, name in HALF_NAMES.items():
            if name in values:
                if half in windows:
                    raise ValueError(f"more than one kickoff tag for half {half}")
                windows[half] = (max(0, tag["start_frame"] - pad), tag["end_frame"] + pad,
                                 (tag["start_frame"] + tag["end_frame"]) // 2)
    return windows


def detect_kickoff(ball, window, centre, fps):
    """(frame, method) of the kickoff strike inside window = (first, stop, fallback).

    ball holds pitch coordinates in metres, NaN where the ball is not tracked, indexed by the same
    frames as the window. method is "tracking" when the strike was seen and "tag" otherwise.
    """
    first, stop, fallback = window
    lookback = int(round(STILL_LOOKBACK_S * fps))
    confirm = int(round(CONFIRM_S * fps))
    no_return = int(round(NO_RETURN_S * fps))
    centre = np.asarray(centre, dtype=float)
    stop = min(stop, len(ball))
    # Candidates must strike before `stop`, but the checks after a strike may look past it.
    horizon = min(len(ball), stop + no_return + 1)
    observed = np.flatnonzero(np.isfinite(ball[:horizon]).all(axis=1))
    observed = observed[observed > first - lookback]

    for frame in observed:
        if frame < first:
            continue
        if frame >= stop:
            break
        # A rest position is the median of a short, fully still stretch of observations.
        recent = observed[(observed > frame - lookback) & (observed <= frame)]
        if len(recent) < STILL_MIN_FRAMES:
            continue
        anchor = np.median(ball[recent], axis=0)
        if np.linalg.norm(anchor - centre) > SPOT_RADIUS_M:
            continue
        if np.any(np.linalg.norm(ball[recent] - anchor, axis=1) > STILL_TOLERANCE_M):
            continue
        # Keep the anchor fixed and find the first observation that leaves it.
        later = observed[observed > frame]
        distance = np.linalg.norm(ball[later] - anchor, axis=1)
        leaving = np.flatnonzero(distance > MOVE_M)
        if not len(leaving):
            break
        # Err early: the strike follows the last observation still well inside the jitter band,
        # so a ball that creeps off the spot is not counted as resting.
        settled = later[:leaving[0]][distance[:leaving[0]] <= STILL_TOLERANCE_M / 2]
        strike = (settled[-1] if len(settled) else frame) + 1
        if strike >= stop:
            break
        # Time the checks from the first observation off the spot, which a tracking gap can delay.
        after = later[leaving[0]:]
        after_distance = distance[leaving[0]:]
        reached = np.any(after_distance[after <= after[0] + confirm] >= CONFIRM_M)
        returned = np.any(after_distance[after <= after[0] + no_return] <= STILL_TOLERANCE_M)
        if reached and not returned:
            # The strike happened after the last frame the ball was seen at rest.
            return int(strike), "tracking"
    return int(fallback), "tag"


def prekickoff_dead(labels, blocks, kickoffs):
    """Copy of labels with every graded frame before its half's kickoff set to D.

    blocks are (half, first, last) graded frame ranges; kickoffs maps half -> kickoff frame.
    Returns (labels, {half: frames changed}).
    """
    out = np.asarray(labels).astype("<U1").copy()
    changed = {}
    for half, first, last in blocks:
        kickoff = kickoffs.get(half)
        if kickoff is None or kickoff <= first:
            changed[half] = 0
            continue
        stop = min(kickoff, last + 1)
        span = out[first:stop]
        changed[half] = int(np.sum(span != DEAD))
        out[first:stop] = DEAD
    return out, changed


def read_ball(raw_path, n_frames, pitch):
    """Ball positions in metres per tracking row, NaN where missing. Row i is video frame i."""
    ball = np.full((n_frames, 2), np.nan)
    with open(raw_path, encoding="utf-8-sig") as handle:
        for row, line in enumerate(handle):
            if row >= n_frames:
                break
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3 or int(parts[0]) != row + 1:
                raise ValueError(f"{raw_path}: unexpected tracking row {row + 1}")
            x, y = parts[2].split(",")[:2]
            if x != "NaN" and y != "NaN":
                ball[row] = float(x) * pitch[0], float(y) * pitch[1]
    return ball


def match_kickoffs(data_dir=None, game=GAME):
    """{half: (frame, method)} for the provided match."""
    fps = GAMES[game]["fps"]
    pitch = GAMES[game]["pitch"]
    (raw_path,) = require(["atd_raw"], game, data_dir)
    ball = read_ball(raw_path, source_frames(game=game), pitch)
    windows = kickoff_windows(load_tags(game, data_dir), fps)
    centre = (pitch[0] / 2.0, pitch[1] / 2.0)
    return {half: detect_kickoff(ball, window, centre, fps) for half, window in windows.items()}


def read_labels(path):
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if rows[0] != ["frame", "label"]:
        raise ValueError(f"{path}: expected a frame,label header")
    frames = [int(row[0]) for row in rows[1:]]
    if frames != list(range(len(frames))):
        raise ValueError(f"{path}: frames must run 0..n-1 in order")
    # Same normalisation as the official scorer, and never a silent truncation.
    labels = [row[1].strip().upper() for row in rows[1:]]
    unknown = sorted(set(labels) - {"H", "A", "D", "X"})
    if unknown:
        raise ValueError(f"{path}: unexpected labels {unknown[:5]}")
    return np.array(labels, dtype="<U1")


def write_labels(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))


def main():
    parser = argparse.ArgumentParser(description="Label graded pre-kickoff frames as dead ball.")
    parser.add_argument("--input", type=Path, required=True, help="frame,label submission CSV")
    parser.add_argument("--output", type=Path, required=True, help="where to write the result")
    parser.add_argument("--data-dir", help="bundle root (defaults to MTAH_DATA_DIR or ./data)")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")

    labels = read_labels(args.input)
    if len(labels) != source_frames():
        raise ValueError(f"{args.input}: expected {source_frames()} rows, found {len(labels)}")
    kickoffs = match_kickoffs(args.data_dir)
    blocks = half_video_blocks(GAME)
    for half, first, _ in blocks:
        if half not in kickoffs:
            print(f"half {half}: no kickoff tag, left unchanged")
            continue
        frame, method = kickoffs[half]
        print(f"half {half}: graded from {first}, kickoff {frame} ({method})")
    out, changed = prekickoff_dead(labels, blocks, {h: f for h, (f, _) in kickoffs.items()})
    for half, count in changed.items():
        print(f"half {half}: {count} frames set to D")
    write_labels(args.output, out)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
