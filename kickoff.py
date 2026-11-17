"""Label graded frames before each half's kickoff as dead ball.

A half can be graded from before its kickoff. The ball is not in play until the kickoff strike,
so every graded frame before it is a stoppage (`D`). A model trained on halves that start at the
kickoff has never seen pre-kickoff play, and it tends to predict live possessions on players who
are only warming up.

The kickoff frame is found from the provided inputs only:

1. The Smart Tagging `Kick Off (Start)` instance for the half bounds the search window.
2. Inside that window, the kickoff is the first frame where the tracked ball, seen near the
   centre spot, moves clearly away from it and does not come back.
3. If the tracking never shows that, the midpoint of the tagged window is used instead.

The thresholds are physical, except the spot radius, which is chosen on external kickoffs. None
of them is fitted to the provided match.

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

# Params in metres and seconds. SPOT_RADIUS_M is chosen on external kickoffs (IDSSE) from a grid
# declared before the run; see experiments/kickoff_validation.py. The rest are physical a priori.
SPOT_RADIUS_M = 3.0       # the ball must be seen this close to the centre spot
STILL_TOLERANCE_M = 0.3   # tracking jitter around a stationary ball
CONFIRM_M = 1.5           # a kicked ball gets this far from the spot ...
CONFIRM_S = 0.6           # ... within this time
NO_RETURN_S = 1.0         # and does not come back within this time
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


def detect_kickoff(ball, window, centre, fps, radius=None):
    """(frame, method) of the kickoff strike inside window = (first, stop, fallback).

    ball holds pitch coordinates in metres, NaN where the ball is not tracked, indexed by the same
    frames as the window. method is "tracking" when the strike was seen and "tag" otherwise.

    A candidate is an observation within `radius` of the centre spot from which the ball travels
    CONFIRM_M within CONFIRM_S of its next observation and, once there, comes back to neither the
    candidate position nor the spot within NO_RETURN_S. Trackers often hide or freeze the ball
    while it waits on the spot, so the ball need not be seen at rest first. The strike is the
    frame after the last observation still at the rest position. When the ball was last seen near
    the spot shortly before a gap, the strike is placed right after that sighting, so a kick
    hidden by the gap is not detected late.
    """
    first, stop, fallback = window
    radius = SPOT_RADIUS_M if radius is None else radius
    confirm = int(round(CONFIRM_S * fps))
    no_return = int(round(NO_RETURN_S * fps))
    centre = np.asarray(centre, dtype=float)
    stop = min(stop, len(ball))
    # Candidates must lie before `stop`; the checks after a candidate may look past it.
    observed = np.flatnonzero(np.isfinite(ball).all(axis=1))
    near = np.linalg.norm(ball[observed] - centre, axis=1) <= radius

    for position in range(np.searchsorted(observed, first), len(observed)):
        frame = observed[position]
        if frame >= stop:
            break
        if not near[position]:
            continue
        later = observed[position + 1:]
        if not len(later):
            break
        distance = np.linalg.norm(ball[later] - ball[frame], axis=1)
        reach = np.flatnonzero((distance >= CONFIRM_M) & (later <= later[0] + confirm))
        if not len(reach):
            continue
        reached = later[reach[0]]
        settle = later[reach[0]:][later[reach[0]:] <= reached + no_return]
        back = np.minimum(np.linalg.norm(ball[settle] - ball[frame], axis=1),
                          np.linalg.norm(ball[settle] - centre, axis=1))
        if np.any(back <= STILL_TOLERANCE_M):
            continue  # the ball came back: a nudge or a placement, not a kick
        # Rest position: median of the candidate and the observations that stay within jitter of
        # it. The strike follows the last observation before the reach point that is still at
        # that position, so an isolated noisy sighting during the rest is not read as the kick.
        track = np.r_[frame, later[:reach[0] + 1]]
        stay = np.flatnonzero(distance > STILL_TOLERANCE_M)
        anchor = np.median(ball[np.r_[frame, later[:stay[0]]]], axis=0)
        still = np.flatnonzero(np.linalg.norm(ball[track[:-1]] - anchor, axis=1)
                               <= STILL_TOLERANCE_M / 2)
        last_still = still[-1] if len(still) else 0
        strike = track[last_still] + 1
        previous = observed[position - 1] if position else None
        if (last_still == 0 and previous is not None and near[position - 1]
                and frame - previous <= no_return
                and np.linalg.norm(ball[frame] - ball[previous]) > STILL_TOLERANCE_M):
            strike = previous + 1  # the kick happened during the gap before this sighting
        if strike >= stop:
            break
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


def match_kickoffs(data_dir=None, game=GAME, ball=None):
    """{half: (frame, method)} for the provided match.

    ball, when given, is the match's ball track in metres on the video frame grid; otherwise it is
    read from the raw tracking.
    """
    fps = GAMES[game]["fps"]
    pitch = GAMES[game]["pitch"]
    if ball is None:
        (raw_path,) = require(["atd_raw"], game, data_dir)
        ball = read_ball(raw_path, source_frames(game=game), pitch)
    windows = kickoff_windows(load_tags(game, data_dir), fps)
    centre = (pitch[0] / 2.0, pitch[1] / 2.0)
    return {half: detect_kickoff(np.asarray(ball, dtype=float), window, centre, fps)
            for half, window in windows.items()}


def kickoff_offset(kickoffs, half, first, last):
    """Graded frames of a half that precede its detected kickoff, clamped to the half."""
    if half not in kickoffs:
        return 0
    return int(np.clip(kickoffs[half][0], first, last + 1)) - first


def decode_after_kickoff(decode, length, offset):
    """Labels for one half: D before offset, then decode(offset) for the frames from offset on.

    decode receives the offset and must return labels for frames offset..length-1, so a sequence
    decoder starts fresh at the kickoff instead of carrying warm-up evidence across it.
    """
    out = np.full(length, DEAD, dtype="<U1")
    if offset < length:
        decoded = np.asarray(decode(offset))
        if len(decoded) != length - offset:
            raise ValueError(f"decoder returned {len(decoded)} labels for {length - offset} frames")
        out[offset:] = decoded
    return out


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
