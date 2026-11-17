"""Generate a possession submission from tactical tracking and optional Smart Tags."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from games import GAME, GAMES, asset_paths, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track, write_submission
from starter.load_smart_tagging import load as load_tags

LABELS = np.array(["H", "A", "D"])
ATTACK_TAGS = {"BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "CREATING CHANCES"}
TEAM_NAMES = {"barcelona": 0, "atletic club": 1, "athletic club": 1}


def read_tracking(path, n_frames, n_slots, pitch):
    """Load positions in metres and verify the tactical frame and slot alignment."""
    positions = np.empty((n_frames, n_slots + 1, 2), dtype=np.float32)
    for index, (players, ball) in enumerate_tracking(path, n_frames):
        values = np.fromstring(players.replace(";", ",") + "," + ball, sep=",", dtype=np.float32)
        if values.size != (n_slots + 1) * 3:
            raise ValueError(f"{path}: unexpected tracking field count at frame {index + 1}")
        positions[index] = values.reshape(-1, 3)[:, :2]
    return positions * np.asarray(pitch, dtype=np.float32)


def enumerate_tracking(path, n_frames):
    count = 0
    with open(path, encoding="utf-8-sig") as handle:
        for count, row in enumerate(handle, start=1):
            frame, players, ball = row.strip().split(":")
            if count > n_frames or int(frame) != count:
                raise ValueError(f"{path}: unexpected tracking frame {frame} at row {count}")
            yield count - 1, (players, ball)
    if count != n_frames:
        raise ValueError(f"{path}: expected {n_frames} frames, found {count}")


def fill_short_gaps(ball, fps, observation_confidence=None):
    """Interpolate bounded gaps of up to 0.5 s within a half."""
    filled = ball.copy()
    observed = np.isfinite(ball).all(axis=1)
    anchors = np.flatnonzero(observed)
    for left, right in zip(anchors[:-1], anchors[1:]):
        gap = right - left - 1
        if 0 < gap <= int(0.5 * fps):
            # A jump between anchors is more likely a tracking error than a useful interpolation.
            if np.linalg.norm(ball[right] - ball[left]) <= 32 * (right - left) / fps:
                weight = np.arange(1, gap + 1)[:, None] / (gap + 1)
                filled[left + 1:right] = ball[left] + weight * (ball[right] - ball[left])
    weights = np.ones(len(ball)) if observation_confidence is None else observation_confidence
    confidence = np.where(observed, weights, np.where(np.isfinite(filled).all(axis=1), 0.45, 0.0))
    return filled, confidence


def tracking_evidence(positions, ball, confidence, teams, pitch, fps):
    """Score each live team by proximity and score D from observed pitch exits."""
    distance = np.linalg.norm(positions[:, :-1] - ball[:, None], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    nearest = np.column_stack([distance[:, teams == side].min(axis=1) for side in (0, 1)])
    control = confidence[:, None] * np.exp(-0.5 * (nearest / 2.0) ** 2)

    outside = np.maximum(-ball, ball - pitch).max(axis=1)
    outside = np.where(np.isfinite(outside), outside, -np.inf)
    in_play = outside < -0.5
    near_line = outside > -3.0
    evidence = np.zeros((len(ball), 3))
    # A close player is weak turnover evidence when the opposing team is equally close.
    margin = control[:, 0] - control[:, 1]
    live = 0.035 * confidence * in_play + 0.045 * control.max(axis=1) * (outside <= 0.5)
    live = np.where(confidence > 0, live, 0.0)
    evidence[:, 0] = live + 0.16 * np.maximum(margin, 0) * (outside <= 0.5)
    evidence[:, 1] = live + 0.16 * np.maximum(-margin, 0) * (outside <= 0.5)
    evidence[:, 2] = 0.24 * confidence * (outside > 0.5)

    speed = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps]
    pair_confidence = np.r_[0.0, np.minimum(confidence[:-1], confidence[1:])]
    return evidence, speed, pair_confidence, near_line


def add_tag_evidence(evidence, speed, confidence, near_line, tags, offset, fps):
    """Use attacking phases as soft team cues and movement to anchor a restart."""
    phase_support = np.zeros((len(evidence), 2))
    restart_support = np.zeros_like(evidence)
    # A restart looks like a change in sustained speed over two short windows.
    width = max(2, round(0.24 * fps))
    fast = np.convolve((speed > 2.5) & (speed < 35) & (confidence > 0), np.ones(width), "full")
    slow = np.convolve((speed < 1.5) & (confidence > 0), np.ones(width), "full")
    for event in tags:
        start = max(0, event["start_frame"] - offset)
        stop = min(len(evidence), event["end_frame"] - offset)
        if start >= stop:
            continue
        side = TEAM_NAMES.get((event["tags"].get("Team") or "").casefold())
        code = (event["code"] or "").upper()
        if code in ATTACK_TAGS and side is not None:
            phase_support[start:stop, side] = 0.045
        if code != "SET PIECES":
            continue

        candidates = np.arange(max(width, start), min(stop, len(evidence) - width + 1))
        if not len(candidates):
            continue
        # Require tracked stationary time before the moving ball. An occlusion by itself
        # says little about whether the preceding play was dead.
        before = slow[candidates - 1]
        after = fast[candidates + width - 1]
        bounds = near_line[candidates - 1] | near_line[candidates]
        restart_type = event["tags"].get("Type") or ""
        if restart_type in {"Throw-in", "Corner Kick"}:
            after = after * bounds
        quality = np.where((before >= width * 0.67) & (after >= width * 0.67),
                           np.minimum(before, after), 0)
        if not np.any(quality):
            continue
        peak = quality.max()
        frame = candidates[np.flatnonzero(quality == peak)[0]]
        # Extend back only through the observed setup, plus a brief tracking dropout.
        setup = slice(max(start, frame - 4 * fps), frame)
        moving = np.flatnonzero(speed[setup] > 2.5)
        dead_start = setup.start + moving[-1] + 1 if len(moving) else setup.start
        # A tracked setup can have a player standing next to the ball throughout.
        # The combined tag and motion evidence must outweigh that proximity cue.
        restart_support[dead_start:frame, 2] = 0.32
        live_stop = min(len(evidence), frame + fps)
        restart_support[frame:live_stop, :2] = np.maximum(
            restart_support[frame:live_stop, :2], 0.07)
        if side is not None:
            restart_support[frame:live_stop, side] = 0.16

    # Multiple overlapping tags may refer to the same action.
    evidence[:, :2] += phase_support
    evidence += restart_support


def decode(evidence, dead_entry=None, transitions=None, turnover_bonus=None):
    """Find the best half-wide sequence with explicit turnover and stoppage costs."""
    # Uniform cues can support the current state without creating a frame-level vote.
    if transitions is None:
        transitions = np.array([[0.0, 1.55, 0.85], [1.55, 0.0, 0.85], [0.85, 0.85, 0.0]])
    state_ids = np.arange(len(LABELS))
    previous = np.empty(evidence.shape, dtype=np.int8)
    # During ties, prefer staying in the current state to avoid an unsupported change.
    score = evidence[0].copy()
    for frame in range(1, len(evidence)):
        options = score[:, None] - transitions
        if turnover_bonus is not None:
            options[0, 1] += turnover_bonus[frame]
            options[1, 0] += turnover_bonus[frame]
        if dead_entry is not None:
            options[:2, 2] += dead_entry[frame]
        origin = options.argmax(axis=0)
        best = options[origin, state_ids]
        origin = np.where(np.diag(options) == best, state_ids, origin)
        previous[frame] = origin
        score = best + evidence[frame]

    states = np.empty(len(evidence), dtype=np.int8)
    states[-1] = score.argmax()
    for frame in range(len(evidence) - 1, 0, -1):
        states[frame - 1] = previous[frame, states[frame]]
    return LABELS[states]


def write_ball_positions(path, frames, ball, pitch):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as output:
        temporary = Path(output.name)
        try:
            with output:
                np.savetxt(output, np.column_stack((frames, ball / pitch)),
                           fmt=("%d", "%.9g", "%.9g"), delimiter=",", header="frame,x,y", comments="")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def build(data_dir=None, use_tags=True, ball_detector=None, device="cpu", detector_range=None,
          recovery_output=None, interpolation_output=None, stoppage_cues=None):
    if recovery_output is not None and ball_detector is None:
        raise ValueError("recovery output requires a ball detector")
    meta, raw = require(["atd_meta", "atd_raw"], data_dir=data_dir)
    tags_path = Path(asset_paths(data_dir=data_dir)["smart_tagging"])
    tags = load_tags(data_dir=data_dir) if use_tags and tags_path.is_file() else []
    teams = team_of_each_track(meta)
    if set(teams) != {0, 1}:
        raise ValueError("tracking metadata must assign slots to both teams")

    fps = GAMES[GAME]["fps"]
    pitch = np.asarray(GAMES[GAME]["pitch"], dtype=np.float32)
    n_frames = source_frames()
    if stoppage_cues is not None:
        from dead_ball import read_cues, stoppage_entry_bonus

        cues = read_cues(stoppage_cues, n_frames)
    print(f"Reading tactical tracking and {len(tags)} Smart Tags", flush=True)
    tracking = read_tracking(raw, n_frames, len(teams), pitch)
    observation_confidence = np.ones(n_frames)
    if ball_detector is not None:
        from video_ball import recover_ball

        recovered = recover_ball(tracking[:, -1], pitch, fps, data_dir, ball_detector,
                                 device, detector_range)
        observation_confidence[recovered] = 0.55
    labels = np.full(n_frames, "X")
    interpolated = []
    for _, start, last in half_video_blocks(GAME):
        half = slice(start, last + 1)
        original = tracking[half, -1]
        ball, weights = fill_short_gaps(original, fps, observation_confidence[half])
        if interpolation_output is not None:
            filled = np.flatnonzero((weights > 0) & ~np.isfinite(original).all(axis=1))
            interpolated.append((filled + start, ball[filled]))
        evidence, speed, confidence, near_line = tracking_evidence(
            tracking[half], ball, weights, teams, pitch, fps)
        add_tag_evidence(evidence, speed, confidence, near_line, tags, start, fps)
        if np.any(evidence):
            entry = (stoppage_entry_bonus(cues[half], speed, confidence, fps)
                     if stoppage_cues is not None else None)
            labels[half] = decode(evidence, entry)
    if recovery_output is not None:
        frames = np.flatnonzero(recovered)
        write_ball_positions(recovery_output, frames, tracking[frames, -1], pitch)
    if interpolation_output is not None:
        frames, positions = zip(*interpolated)
        write_ball_positions(interpolation_output, np.concatenate(frames), np.concatenate(positions), pitch)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="bundle root (defaults to MTAH_DATA_DIR or ./data)")
    parser.add_argument("--tracking-only", action="store_true", help="skip Smart Tags")
    parser.add_argument("--ball-detector", type=Path, help="local SoccerNet-v3D checkpoint")
    parser.add_argument("--device", default="cpu", help="inference device (cpu by default)")
    parser.add_argument("--detector-range", type=int, nargs=2, metavar=("START", "STOP"),
                        help="limit video inference to a half-open frame range for diagnostics")
    parser.add_argument("--recovery-output", type=Path,
                        help="write accepted YOLO positions as frame,x,y with normalized coordinates")
    parser.add_argument("--interpolation-output", type=Path,
                        help="write interpolated ball positions separately as frame,x,y")
    parser.add_argument("--stoppage-cues", type=Path, help="optional whistle and raised-arm CSV from dead_ball.py")
    parser.add_argument("--output", type=Path,
                        default=Path("evaluation/results/tactical_decoder.csv"))
    args = parser.parse_args()
    if args.recovery_output is not None and args.ball_detector is None:
        parser.error("--recovery-output requires --ball-detector")
    outputs = [path.resolve() for path in (args.output, args.recovery_output, args.interpolation_output)
               if path is not None]
    if len(outputs) != len(set(outputs)):
        parser.error("each output needs a separate path")
    if args.stoppage_cues is not None and args.stoppage_cues.resolve() in outputs:
        parser.error("--stoppage-cues must be separate from the output paths")
    if args.detector_range and (
        args.ball_detector is None
        or not 0 <= args.detector_range[0] < args.detector_range[1] <= source_frames()
    ):
        parser.error("--detector-range needs a checkpoint and valid tactical frame bounds")
    try:
        labels = build(args.data_dir, use_tags=not args.tracking_only,
                       ball_detector=args.ball_detector, device=args.device,
                       detector_range=args.detector_range, recovery_output=args.recovery_output,
                       interpolation_output=args.interpolation_output, stoppage_cues=args.stoppage_cues)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=args.output.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            write_submission(temporary, labels)
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Wrote {len(labels):,} frames to {args.output}")


if __name__ == "__main__":
    main()
