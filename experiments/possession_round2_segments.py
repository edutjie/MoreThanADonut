"""Training-only duration and touch models for native-frame possession decoding."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression

from evaluation import possession_metrics as m


def fit_duration(training):
    counts = np.full((2, 60), 0.5)
    tails = [[], []]
    for _, _, row in training:
        ground = row["ground"]
        fps = int(row["fps"])
        for left, right in m.valid_runs(ground):
            for start, stop, state in m.segments(ground, [(left, right)]):
                if start == left or stop == right:
                    continue
                group = int(state == m.D)
                duration = (stop - start) / fps
                if duration <= 30:
                    counts[group, min(59, int(duration * 2))] += 1
                else:
                    tails[group].append(duration - 30)
    counts = gaussian_filter1d(counts, 1, axis=1)
    logs = np.log(counts / counts.max(axis=1, keepdims=True))
    slope = [-1 / max(np.mean(tail), 0.5) if tail else -2 for tail in tails]
    return {"log_duration": logs.tolist(), "tail_per_second": slope}


def duration_prior(settings, fps, strength):
    points = np.arange(1, 30 * fps + 1) / fps
    grid = np.arange(60) * 0.5 + 0.25
    values = np.column_stack([np.interp(points, grid, row) for row in settings["log_duration"]])
    return (np.ascontiguousarray(values[:, [0, 0, 1]] * strength),
            (np.asarray(settings["tail_per_second"]) * (strength / fps))[[0, 0, 1]])


def offsets(length, lag):
    frames = np.arange(length)
    return np.maximum(0, frames - lag), np.minimum(length - 1, frames + lag)


def touch_inputs(tracks, fps):
    ball = tracks[:, -1]
    lag = max(1, round(fps * 0.2))
    before, after = offsets(len(ball), lag)
    pre, post = ball - ball[before], ball[after] - ball
    pre_norm, post_norm = (np.linalg.norm(v, axis=1) for v in (pre, post))
    direction = (pre * post).sum(axis=1) / np.maximum(pre_norm * post_norm, 0.04)
    acceleration = np.linalg.norm(post - pre, axis=1)
    distances = np.linalg.norm(tracks[:, :-1] - ball[:, None], axis=2)
    nearest = np.where(np.isfinite(distances), distances, np.inf).min(axis=1)
    nearest[~np.isfinite(nearest)] = np.nan
    return np.column_stack((acceleration, direction, nearest, nearest[before] - nearest,
                            nearest[after] - nearest, pre_norm, post_norm))


def touch_features(physical, p, fps):
    before, after = offsets(len(p), max(1, round(fps * 0.2)))
    # Team exchange and restart clues from the frozen reference, without labels.
    switch = np.abs((p[:, 0] - p[:, 1])[after] - (p[:, 0] - p[:, 1])[before])
    restart = p[before, 2] - p[after, 2]
    return np.column_stack((physical, switch, restart))


def touch_matrix(x, median, scale):
    missing = ~np.isfinite(x)
    z = np.clip((np.where(missing, median, x) - median) / scale, -12, 12)
    return np.column_stack((z, missing))


def fit_touch(training):
    x, y = [], []
    for ground, fps, features in training:
        for start, stop in m.valid_runs(ground):
            changes = np.flatnonzero(ground[start:stop-1] != ground[start+1:stop]) + start + 1
            choices = changes[ground[changes] < m.D]
            for frame in choices:
                negative = frame + np.rint(np.array([-1, -.6, -.3, .3, .6, 1]) * fps).astype(int)
                negative = negative[(negative >= start) & (negative < stop)]
                distance = np.abs(negative[:, None] - changes).min(axis=1)
                negative = negative[distance > round(fps * .15)]
                x.extend(features[np.r_[frame, negative]])
                y.extend([1, *([0] * len(negative))])
    matrix = np.asarray(x)
    median = np.nanmedian(matrix, axis=0)
    scale = np.maximum(np.nanstd(matrix, axis=0), 1e-3)
    model = LogisticRegression(C=0.3, max_iter=250, class_weight="balanced")
    model.fit(touch_matrix(matrix, median, scale), y)
    return {"median": median.tolist(), "scale": scale.tolist(),
            "coeff": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
            "positive_frames": int(sum(y)), "negative_frames": len(y) - int(sum(y))}


def likelihood(features, model):
    matrix = touch_matrix(features, np.array(model["median"]), np.array(model["scale"]))
    return matrix @ model["coeff"] + model["intercept"]


def snap(pred, touch, radius):
    boundaries = np.flatnonzero(pred[1:] != pred[:-1]) + 1
    moved = np.r_[0, boundaries, len(pred)]
    original = moved.copy()
    for i, frame in enumerate(boundaries, 1):
        if pred[frame] == m.D:
            continue
        # Adjacent searches never overlap, and every original state retains frames.
        start = max(frame - radius, (original[i-1] + frame) // 2 + 1)
        stop = min(frame + radius, (frame + original[i+1]) // 2, len(pred)-1)
        if start <= stop:
            window = touch[start:stop + 1]
            candidates = start + np.flatnonzero(window == window.max())
            moved[i] = candidates[np.argmin(np.abs(candidates - frame))]
    result = np.empty_like(pred)
    for start, stop, original_start in zip(moved, moved[1:], original):
        result[start:stop] = pred[original_start]
    return result
