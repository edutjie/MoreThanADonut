'Directional turnover costs from observed ball and player motion.'


import numpy as np


from scipy.ndimage import maximum_filter1d, uniform_filter1d


from tactical_baseline import fill_short_gaps


def turnover_adjustment(tracks, teams, fps, *, observed_only):
    """Return signed costs for entering each live team at a proposed change."""
    raw = tracks[:, -1]
    ball, _ = fill_short_gaps(raw, fps)
    players = tracks[:, :-1]
    frames = np.arange(len(tracks))
    lag = max(1, round(.16 * fps))
    left = np.maximum(0, frames - lag)
    right = np.minimum(len(frames) - 1, frames + lag)
    before = (ball - ball[left]) * (fps / np.maximum(1, frames - left))[:, None]
    after = (ball[right] - ball) * (fps / np.maximum(1, right - frames))[:, None]
    valid = (frames >= lag) & (frames + lag < len(frames))
    if observed_only:
        missing = ~np.isfinite(raw).all(axis=1)
        gap = np.cumsum(missing)
        valid &= ~missing[left] & (gap[left] == gap[right])
    valid &= np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
    dv = np.linalg.norm(after - before, axis=1)
    speed = np.minimum(np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1))
    distance = np.linalg.norm(players - ball[:, None], axis=-1)
    distance[~np.isfinite(distance)] = np.inf
    player_velocity = (players[right] - players[left]) * (fps / np.maximum(1, right - left))[:, None, None]
    ball_velocity = (before + after) / 2
    relative = np.linalg.norm(player_velocity - ball_velocity[:, None], axis=-1)
    moving_control = ((distance <= 1.4) & (relative <= 2.0)
                      & (np.linalg.norm(player_velocity, axis=-1) <= 12) & valid[:, None])
    dwell = max(1, round(.36 * fps))
    moving_control = uniform_filter1d(moving_control.astype(np.float32), dwell, axis=0,
                                     mode="constant") >= .75
    masks = [teams == side for side in (0, 1)]
    nearest = [(np.where(mask, distance, np.inf) if teams.ndim == 2 else distance[:, mask]).min(axis=1)
               for mask in masks]
    horizon = max(1, round(.24 * fps))
    out = np.zeros((len(frames), 2))
    for side in (0, 1):
        near = nearest[side]
        uncontested = near + .35 < nearest[1 - side]
        impulse = (near <= 1.5) & (dv >= 4.0) & uncontested & valid
        carry = (moving_control & masks[side]).any(axis=1) & uncontested & valid
        flyby = (near <= 2.0) & (speed >= 8) & (dv <= 2.5) & valid
        support = maximum_filter1d((impulse | carry).astype(np.int8), horizon, mode="constant") > 0
        penalty = maximum_filter1d(flyby.astype(np.int8), horizon, mode="constant") > 0
        out[:, side] = np.where(valid, np.where(support, -1.0, penalty), 0.0)
    return out
