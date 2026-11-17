'Conservative repairs for player tracklets used by the possession models.'


import numpy as np


def repair_player_tracks(positions, fps):
    """Drop long exact holds and repair short out-and-back position spikes."""
    cleaned = positions.copy()
    min_hold = round(2 * fps)
    max_spike = max(1, int(0.2 * fps))
    for track in cleaned[:, :-1].transpose(1, 0, 2):
        same = np.all(track[:-1] == track[1:], axis=1)
        boundaries = np.flatnonzero(np.diff(np.r_[False, same, False]))
        for start, end in zip(boundaries[::2], boundaries[1::2]):
            if end + 1 - start >= min_hold:
                track[start:end + 1] = np.nan
        for width in range(1, max_spike + 1):
            jumps = np.linalg.norm(np.diff(track, axis=0), axis=1) > 2.0
            starts = np.flatnonzero(jumps[:-width] & jumps[width:])
            ends = starts + width + 1
            displacement = track[ends] - track[starts]
            plausible = np.linalg.norm(displacement, axis=1) <= 12 * (width + 1) / fps
            starts, displacement = starts[plausible], displacement[plausible]
            if starts.size:
                origin = track[starts].copy()
                for offset in range(1, width + 1):
                    track[starts + offset] = origin + displacement * (offset / (width + 1))
    return cleaned
