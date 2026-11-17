"""Tracking loading, bounded gap interpolation and Viterbi utilities."""


import numpy as np


LABELS = np.array(["H", "A", "D"])


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


def decode(evidence, dead_entry=None, transitions=None, turnover_bonus=None):
    """Find the best half-wide sequence with explicit turnover and stoppage costs."""
    # Uniform cues can support the current state without creating a frame-level vote.
    if transitions is None:
        transitions = np.array([[0.0, 1.55, 0.85], [1.55, 0.0, 0.85], [0.85, 0.85, 0.0]])
    state_ids = np.arange(len(LABELS))
    previous = np.empty(evidence.shape, dtype=np.int8)
    if turnover_bonus is not None:
        bonus = np.asarray(turnover_bonus)
        if bonus.ndim == 1:
            bonus = np.broadcast_to(bonus[:, None], (len(evidence), 2))
        if bonus.shape != (len(evidence), 2):
            raise ValueError("turnover bonus must match the evidence frame grid and live teams")
    # During ties, prefer staying in the current state to avoid an unsupported change.
    score = evidence[0].copy()
    for frame in range(1, len(evidence)):
        options = score[:, None] - transitions
        if turnover_bonus is not None:
            options[0, 1] += bonus[frame, 1]
            options[1, 0] += bonus[frame, 0]
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
