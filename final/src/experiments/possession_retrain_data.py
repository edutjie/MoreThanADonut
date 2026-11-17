'External match preparation and spatial features for full possession training.'


import numpy as np


from scipy.ndimage import uniform_filter1d


FPS = 5


def touch_history(tracks, teams, fps):
    """Signed evidence with explicit confidence and decaying gap support."""
    distance = np.linalg.norm(tracks[:, :-1] - tracks[:, -1:], axis=-1)
    near = np.stack([np.min(np.where(np.isfinite(distance[:, teams == side]),
                                    distance[:, teams == side], np.inf), axis=1) for side in (0, 1)], axis=1)
    observed = np.isfinite(tracks[:, -1]).all(axis=1)
    closest = near.argmin(axis=1)
    minimum = near.min(axis=1)
    both_teams = np.isfinite(near).all(axis=1)
    finite_near = np.where(np.isfinite(near), near, 0)
    margin = np.abs(finite_near[:, 0] - finite_near[:, 1])
    confidence = np.where(observed & both_teams,
                          np.clip(margin / 1.5, 0, 1) * np.exp(-minimum / 1.5), 0)
    valid = (minimum < 2) & (confidence > .1)
    signed = np.where(valid, 1 - 2 * closest, 0)
    frames = np.arange(len(tracks))
    last = np.maximum.accumulate(np.where(valid, frames, -1))
    nxt = np.minimum.accumulate(np.where(valid, frames, len(tracks))[::-1])[::-1]
    li, ni = last.clip(0), nxt.clip(max=len(tracks) - 1)
    past = np.where(last >= 0, confidence[li] * np.exp(-(frames - last) / (3 * fps)), 0)
    future = np.where(nxt < len(tracks), confidence[ni] * np.exp(-(nxt - frames) / (3 * fps)), 0)
    disagree = (last >= 0) & (nxt < len(tracks)) & (signed[li] != signed[ni])
    support = np.minimum(past, future)
    return np.column_stack((signed * confidence, signed[li] * past, signed[ni] * future,
                            past, future, observed, support * ~disagree, support * disagree)).astype(np.float32)


def canonical(tracks, teams, pitch):
    centers = [np.nanmedian(tracks[:, :-1, 0][:, teams == side]) for side in (0, 1)]
    return (pitch - tracks if centers[0] > centers[1] else tracks).astype(np.float32)


def velocity(positions):
    difference = np.diff(positions, axis=0, prepend=positions[:1]) * FPS
    valid = np.isfinite(difference).all(axis=-1) & (np.linalg.norm(difference, axis=-1) < 45)
    return np.where(valid[..., None], difference, np.nan)


def average(values, axis=1):
    valid = np.isfinite(values)
    return np.where(valid, values, 0).sum(axis=axis) / np.maximum(valid.sum(axis=axis), 1)


def frame_features(tracks, teams, pitch, source):
    ball, players = tracks[:, -1], tracks[:, :-1]
    visible = np.isfinite(tracks).all(axis=-1)
    motion = velocity(tracks)
    speed = np.linalg.norm(motion, axis=-1)
    n = len(tracks)
    frames = np.arange(n)
    last = np.maximum.accumulate(np.where(visible[:, -1], frames, -1))
    nxt = np.minimum.accumulate(np.where(visible[:, -1], frames, n)[::-1])[::-1]
    since = np.where(last >= 0, np.minimum((frames - last) / FPS, 30), 30)
    until = np.where(nxt < n, np.minimum((nxt - frames) / FPS, 30), 30)
    position = ball / pitch
    outside = np.maximum(-ball, ball - pitch).max(axis=1)
    features = [position, motion[:, -1] / 30, speed[:, -1:] / 30,
                np.column_stack((visible[:, -1], since / 10, until / 10, outside / 5))]
    distance = np.linalg.norm(players - ball[:, None], axis=-1)
    for side in (0, 1):
        use = teams == side
        p, v, d, s = players[:, use], visible[:, :-1][:, use], distance[:, use], speed[:, :-1][:, use]
        minimum = np.min(np.where(np.isfinite(d), d, np.inf), axis=1)
        sorted_d = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
        centroid = average(p, axis=1)
        spread = np.sqrt(average((p - centroid[:, None]) ** 2, axis=1))
        quantiles = []
        # Fixed pitch bins remain defined when a whole team is missing.
        for axis in (0, 1):
            sorted_p = np.sort(np.where(v, p[..., axis], np.inf), axis=1)
            count = v.sum(axis=1)
            for q in (.1, .5, .9):
                index = np.minimum((q * np.maximum(count - 1, 0)).astype(int), p.shape[1] - 1)
                quantiles.append(np.where(count > 0, sorted_p[np.arange(n), index] / pitch[axis], np.nan))
        walking = average(np.where(np.isfinite(s), s < 1.5, np.nan))
        mean_speed = average(s)
        features.extend((centroid / pitch, spread / pitch, np.column_stack(quantiles),
                         np.column_stack((v.sum(axis=1) / 11, mean_speed / 5, walking,
                                          minimum / 10, sorted_d[:, 1] / 10,
                                          ((d < 2) & v).sum(axis=1) / 3))))
    touch = touch_history(tracks, teams, FPS)
    features.append(touch)
    raw = np.column_stack(features).astype(np.float32)
    raw[~np.isfinite(raw)] = np.nan
    windows = []
    for seconds in (1, 3):
        length = round(seconds * FPS)
        valid = np.isfinite(raw)
        total = uniform_filter1d(np.where(valid, raw, 0), length, axis=0, mode="nearest")
        count = uniform_filter1d(valid.astype(np.float32), length, axis=0, mode="nearest")
        windows.append(np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0))
    return np.column_stack((raw, *windows, np.full(n, source))).astype(np.float32)


def mirrored(tracks, teams, pitch):
    return pitch - tracks, 1 - teams
