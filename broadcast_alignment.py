"""Synchronize the official broadcast and tactical feeds using their automatic tracking."""

import argparse
import json
from pathlib import Path

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import team_names, team_of_each_track
from starter.load_homography import check_alignment, load
from tactical_baseline import read_tracking

MATCH_DISTANCE = 2.0  # metres, the decoder's player-control scale


def load_feed(source, data_dir, pitch):
    meta, raw = require(["atd_meta", "atd_raw"], data_dir=data_dir, source=source)
    teams = team_of_each_track(meta)
    positions = read_tracking(raw, source_frames(source), len(teams), pitch)
    return positions, teams, team_names(meta)


def ball_offsets(tactical, broadcast, valid, fps):
    """Match moving, two-second ball paths anywhere on the tactical clock."""
    taps = np.rint(np.linspace(-fps, fps, 5)).astype(int)
    stride = 5 * fps
    offsets_by_half = {half: [] for half, _, _ in half_video_blocks(GAME)}
    regions = np.zeros(len(tactical), dtype=int)
    for half, start, last in half_video_blocks(GAME):
        regions[start + fps:last - fps + 1] = half
    searchable = np.flatnonzero((regions > 0) & np.isfinite(tactical).all(axis=1))
    tactical_points = tactical[searchable]

    for center in range(stride, len(broadcast) - fps, stride):
        query = broadcast[center + taps]
        seen = np.isfinite(query).all(axis=1) & valid[center + taps]
        if not seen[2] or seen.sum() < 4 or np.linalg.norm(np.ptp(query[seen], axis=0)) < 3:
            continue
        dx, dy = (tactical_points - query[2]).T
        candidates = searchable[dx * dx + dy * dy <= 9]
        if not len(candidates):
            continue
        delta = tactical[candidates[:, None] + taps[seen]] - query[seen]
        residual = np.linalg.norm(delta, axis=2)
        scores = np.where(np.isfinite(residual), residual, 20).mean(axis=1)
        best = scores.argmin()
        if scores[best] <= MATCH_DISTANCE:
            match = candidates[best]
            offsets_by_half[regions[match]].append((center // stride, match - center))
    return offsets_by_half


def consensus_offset(anchors, fps):
    # Interleave discovery and independent verification samples.
    records = np.asarray(anchors, dtype=int).reshape(-1, 2)
    discovery = np.sort(records[records[:, 0] % 2 == 0, 1])
    verification = records[records[:, 0] % 2 == 1, 1]
    if min(len(discovery), len(verification)) < 10:
        raise ValueError("Too few independent ball-path matches for synchronization")
    tolerance = round(0.2 * fps)
    stop = np.searchsorted(discovery, discovery + 2 * tolerance, side="right")
    start = np.argmax(stop - np.arange(len(discovery)))
    offset = int(np.rint(np.median(discovery[start:stop[start]])))
    agreeing = int(np.count_nonzero(np.abs(verification - offset) <= tolerance))
    if agreeing * 2 <= len(verification):
        raise ValueError("Ball-path matches do not support a stable offset")
    return offset, agreeing, len(verification)


def same_live_view(tactical, broadcast, tactical_teams, broadcast_teams, indices, offset):
    """Check same-team ground positions without relying on shared track IDs."""
    result = []
    same_team = broadcast_teams[:, None] == tactical_teams
    for start in range(0, len(indices), 512):
        frames = indices[start:start + 512]
        a, b = tactical[frames + offset], broadcast[frames]
        dx = b[:, :, None, 0] - a[:, None, :, 0]
        dy = b[:, :, None, 1] - a[:, None, :, 1]
        distance = np.hypot(dx, dy)
        close = (distance <= MATCH_DISTANCE) & same_team
        seen = np.isfinite(b).all(axis=2).sum(axis=1)
        matched = close.any(axis=2).sum(axis=1)
        result.extend((matched >= 4) & (matched * 4 >= seen * 3))
    return np.asarray(result, dtype=bool)


def synchronize(tactical, broadcast, tactical_teams, broadcast_teams, valid, fps):
    mapping = np.full(len(broadcast), -1, dtype=np.int32)
    anchors = ball_offsets(tactical[:, -1], broadcast[:, -1], valid, fps)
    summaries = []
    live_frames = []
    for half, first, last in half_video_blocks(GAME):
        offset, agreeing, tested = consensus_offset(anchors[half], fps)
        start, stop = max(0, first - offset), min(len(broadcast), last + 1 - offset)
        if stop <= start:
            raise ValueError("Empty broadcast half after clock synchronization")
        indices = np.arange(start, stop)
        probes = indices[::max(1, fps // 5)]
        agreement = valid[probes] & same_live_view(
            tactical[:, :-1], broadcast[:, :-1], tactical_teams, broadcast_teams, probes, offset,
        )
        # Require agreement on both sides of every short interval. Keep each
        # frame's native calibration validity check as well.
        accepted = np.zeros(len(indices), dtype=bool)
        for left, right in zip(probes[:-1][agreement[:-1] & agreement[1:]],
                               probes[1:][agreement[:-1] & agreement[1:]]):
            accepted[left - start:right - start + 1] = True
        accepted &= valid[indices]
        live_frames.append(indices[accepted])
        summaries.append(dict(half=half, offset_frames=offset, broadcast_start=start,
                              broadcast_stop=stop, verified_anchors=agreeing, tested_anchors=tested,
                              possible_frames=len(indices)))

    first, second = summaries
    if first["broadcast_start"] >= second["broadcast_start"]:
        raise ValueError("Broadcast half order disagrees with the tactical clock")
    if first["broadcast_stop"] > second["broadcast_start"]:
        # The nominal ranges can overlap when TV edits the first half or
        # intermission. Retain the most position-verified frames in play order.
        cuts = np.arange(second["broadcast_start"], first["broadcast_stop"] + 1)
        left = np.searchsorted(live_frames[0], cuts)
        right = len(live_frames[1]) - np.searchsorted(live_frames[1], cuts)
        cut = cuts[(left + right).argmax()]
        live_frames[0] = live_frames[0][live_frames[0] < cut]
        live_frames[1] = live_frames[1][live_frames[1] >= cut]
    for summary, frames in zip(summaries, live_frames):
        mapping[frames] = frames + summary["offset_frames"]
        summary["live_frames"] = len(frames)
    return mapping, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Private .npz frame map")
    args = parser.parse_args()
    game = GAMES[GAME]
    pitch = np.asarray(game["pitch"])
    tactical, tactical_teams, tactical_names = load_feed("tactical", args.data_dir, pitch)
    broadcast, broadcast_teams, broadcast_names = load_feed("broadcast", args.data_dir, pitch)
    if tactical_names != broadcast_names:
        raise ValueError("Broadcast and tactical player team order differs")
    if not check_alignment(data_dir=args.data_dir, source="broadcast")["agrees"]:
        raise ValueError("Broadcast homographies do not agree with its tracking frame axis")
    _, _, valid = load(data_dir=args.data_dir, source="broadcast")
    mapping, summaries = synchronize(
        tactical, broadcast, tactical_teams, broadcast_teams, valid, game["fps"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, tactical_frame=mapping)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
