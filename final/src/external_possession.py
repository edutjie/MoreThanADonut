'Train and evaluate a tracking-only possession model on SkillCorner A-League games.'


import numpy as np


from scipy.ndimage import uniform_filter1d


from tactical_baseline import decode, fill_short_gaps


SKILLCORNER_FPS = 10


LAG_SECONDS = (0.5, 1.0, 2.0, 4.0)


def tracking_features(positions, weights, teams, pitch, fps, lags=(), *,
                      continuous_motion=False, observed_ball_motion=False):
    """Return shared features and the same team features in home/away order."""
    ball, confidence = fill_short_gaps(positions[:, -1], fps, weights)
    players = positions[:, :-1]
    relative = players - ball[:, None]
    distance = np.linalg.norm(relative, axis=-1)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    outside = np.maximum(-ball, ball - pitch).max(axis=1)
    visible = np.isfinite(players).all(axis=-1)
    frames = np.arange(len(ball))
    if observed_ball_motion:
        missing = ~np.isfinite(positions[:, -1]).all(axis=1)
        gap_count = np.cumsum(missing, dtype=np.int32)
    if continuous_motion:
        step = np.linalg.norm(np.diff(players, axis=0), axis=-1)
        breaks = ~np.isfinite(step) | (step > max(1.0, 15 / fps))
        continuity = np.zeros(visible.shape, dtype=np.int32)
        np.cumsum(breaks, axis=0, out=continuity[1:])

    shared = [confidence, outside, distance.min(axis=1)]
    team_features = [[], []]
    for side, features in enumerate(team_features):
        side_distance = np.sort(distance[:, teams == side], axis=1)
        features.extend([side_distance[:, 0], side_distance[:, 1],
                         (side_distance < 5).sum(axis=1), visible[:, teams == side].sum(axis=1)])
    lag_shared = shared.copy()
    lag_teams = [features.copy() for features in team_features]

    for seconds in (0.2, 0.6, 1.5, 4.0):
        window = max(1, round(fps * seconds))
        shared.append(uniform_filter1d(confidence, size=window, mode="nearest"))
        neighbors = (np.maximum(0, frames - window), np.minimum(len(ball) - 1, frames + window))
        for direction, other in enumerate(neighbors):
            elapsed = np.maximum(np.abs(other - frames), 1) / fps
            ball_speed = np.linalg.norm(ball[other] - ball, axis=-1) / elapsed
            if observed_ball_motion:
                ball_speed[missing | missing[other] | (gap_count[other] != gap_count)] = np.nan
            shared.append(ball_speed)
            short_backward = seconds == 0.2 and direction == 0
            if short_backward:
                lag_shared.append(ball_speed)
            player_speed = np.linalg.norm(players[other] - players, axis=-1) / elapsed[:, None]
            if continuous_motion:
                player_speed[continuity[other] != continuity] = np.nan
            for side, features in enumerate(team_features):
                near = features[0]
                side_speed = player_speed[:, teams == side]
                count = np.isfinite(side_speed).sum(axis=1)
                mean_speed = np.nansum(np.minimum(side_speed, 12), axis=1) / np.maximum(count, 1)
                if continuous_motion:
                    mean_speed[count == 0] = np.nan
                features.extend([near[other], mean_speed])
                if short_backward:
                    lag_teams[side].append(mean_speed)

    shared = np.column_stack(shared).astype(np.float32)
    home, away = (np.column_stack(features).astype(np.float32) for features in team_features)
    normal = np.column_stack((shared, home, away))
    mirrored = np.column_stack((shared, away, home))
    if lags:
        normal_lags = lagged_columns((*lag_shared, *lag_teams[0], *lag_teams[1]), lags, fps)
        mirrored_lags = lagged_columns((*lag_shared, *lag_teams[1], *lag_teams[0]), lags, fps)
        normal = np.column_stack((normal, normal_lags))
        mirrored = np.column_stack((mirrored, mirrored_lags))
    normal[~np.isfinite(normal)] = np.nan
    mirrored[~np.isfinite(mirrored)] = np.nan
    return normal, mirrored


def lagged_columns(columns, lags, fps):
    """Stack earlier snapshots, leaving the start of each half unpadded."""
    values = np.column_stack(columns).astype(np.float32)
    width = len(columns)
    result = np.full((len(values), width * len(lags)), np.nan, dtype=np.float32)
    for index, seconds in enumerate(lags):
        offset = max(1, round(seconds * fps))
        if offset < len(values):
            result[offset:, index * width:(index + 1) * width] = values[:-offset]
    return result


def decode_costs(evidence, fps, turnover, stoppage, turnover_adjustment=None):
    transitions = np.array([[0, turnover, stoppage], [turnover, 0, stoppage],
                            [stoppage, stoppage, 0]])
    bonus = None if turnover_adjustment is None else -np.asarray(turnover_adjustment)
    return decode(evidence * (10 / fps), transitions=transitions, turnover_bonus=bonus)
