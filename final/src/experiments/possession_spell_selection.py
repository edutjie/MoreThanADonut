'Rank complete H/A spell proposals while preserving fixed baseline stoppages.'


from collections import Counter


import numpy as np


from evaluation import possession_metrics as m


from experiments.possession_hybrid_refinement import COSTS, STRENGTHS


def intervals(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def build_case(p, dead, physics, visible, fps, decoder):
    paths = [decoder(p, dead, fps, cost, strength, physics) for cost in COSTS for strength in STRENGTHS]
    reference = decoder(p, dead, fps, 2.5)
    reference_spells = m.possession_spells(reference, [(0, len(p))])
    votes = Counter(spell for path in paths for spell in m.possession_spells(path, [(0, len(p))]))
    spans = sorted(votes, key=lambda row: (row[1], row[0], row[2]))
    features = []
    window = round(2 * fps)
    odds = np.log(np.maximum(p[:, 0], 1e-6)) - np.log(np.maximum(p[:, 1], 1e-6))
    by_side = (odds, -odds)
    for a, b, side in spans:
        own = p[:, side]
        log_odds = by_side[side]
        inside = own[a:b]
        margin = log_odds[a:b]
        pre = own[max(0, a - window):a]
        post = own[b:min(len(p), b + window)]
        overlaps = [max(0, min(b, stop) - max(a, start)) / (max(b, stop) - min(a, start))
                    for start, stop, team in reference_spells if team == side]
        features.append([
            np.log1p((b - a) / fps), votes[a, b, side] / len(paths),
            float((a, b, side) in reference_spells), max(overlaps, default=0),
            inside.mean(), inside.std(), inside.min(), np.quantile(inside, .1),
            margin.mean(), margin.min(), np.mean(inside > .5),
            float(pre.mean()) if len(pre) else .5, float(post.mean()) if len(post) else .5,
            log_odds[a] - log_odds[max(0, a - 1)],
            log_odds[b - 1] - log_odds[min(len(p) - 1, b)],
            visible[a:b].mean(),
            np.mean(physics[a:b, side] < 0), np.mean(physics[a:b, 1 - side] < 0),
            float(a == 0 or dead[a - 1]), float(b == len(p) or dead[b]),
            np.mean(-(inside * np.log(np.maximum(inside, 1e-6)) +
                      (1 - inside) * np.log(np.maximum(1 - inside, 1e-6)))),
            np.mean(np.abs(margin) < .5),
        ])
    return {"spans": np.asarray(spans, dtype=np.int32).reshape(-1, 3),
            "features": np.asarray(features, dtype=np.float32).reshape(-1, 22),
            "dead": dead, "fps": fps, "reference": reference}


def select_spells(case, quality, penalty):
    n = len(case["dead"])
    output = np.full(n, m.D, dtype=np.int8)
    if not len(case["spans"]):
        if not case["dead"].all():
            raise ValueError("live frames require at least one spell proposal")
        return output
    value = np.full((n + 1, 2), -np.inf)
    trace = np.full((n + 1, 2), -1, dtype=np.int32)
    for index, (a, b, side) in enumerate(case["spans"]):
        previous = 0. if a == 0 or case["dead"][a - 1] else value[a, 1 - side]
        score = previous + float(quality[index]) - penalty
        if score > value[b, side]:
            value[b, side] = score
            trace[b, side] = index
    for start, stop in intervals(~case["dead"]):
        side = int(value[stop].argmax())
        if not np.isfinite(value[stop, side]):
            raise ValueError("spell proposals do not cover the live interval")
        cursor = stop
        while cursor > start:
            index = trace[cursor, side]
            if index < 0:
                raise ValueError("invalid spell path")
            a, b, team = case["spans"][index]
            output[a:b] = team
            cursor, side = a, 1 - team
    if np.any((output == m.D) != case["dead"]):
        raise ValueError("spell selection changed the fixed dead-ball mask")
    return output
