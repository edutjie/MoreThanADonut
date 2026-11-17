"""Check the kickoff detector in kickoff.py against external kickoffs with known frames.

References:
- IDSSE (DFL): a frame where BallStatus switches from dead to alive while the ball is inside
  the centre circle (REFERENCE_RADIUS_M). This is the convention the ground truth follows.
- Metrica sample games 1 and 2: `SET PIECE` / `KICK OFF` event start frames.

Kickoffs with less than MIN_HISTORY_S of tracking before them are skipped, because the detector
needs to see the ball at rest. Each reference gets a simulated tag window of the same length as the
provided `Kick Off (Start)` tags, placed at a random offset, so the fallback is exercised the way
it would be on a new match. Ball dropout conditions use the published 41% missing rate.
Only the spot radius is chosen here, on IDSSE, from RADII_M; Metrica is the held-out test.

    python -m experiments.kickoff_validation --idsse external_data/idsse \
        --metrica external_data/metrica-sample --output <dir>
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as metrics
from external_possession import IDSSE_FPS, IDSSE_MATCHES, load_idsse, mask_ball
from kickoff import WINDOW_PAD_S, detect_kickoff

TAG_FRAMES = 125            # length of the provided Kick Off (Start) tags at 25 fps
REFERENCE_RADIUS_M = 9.15   # centre circle; well beyond every radius in the grid
MIN_HISTORY_S = 2.0
METRICA_FPS = 25
METRICA_PITCH = (105.0, 68.0)
METRICA_GAMES = (1, 2)
COVERAGE = 0.59             # published share of frames with a tracked ball
CONDITIONS = {"natural": None, "missing-2s": 2.0, "missing-12s": 12.0}
MASK_SEEDS = (0, 1, 2)
TOLERANCE_FRAMES = 2        # a one- or two-frame offset is close to free in spellF1
RADII_M = (1.0, 2.0, 3.0)   # declared before the run; chosen on IDSSE, tested on Metrica


def idsse_halves(root, match):
    out = []
    for half, (positions, _, _, pitch, labels) in enumerate(load_idsse(root, match), 1):
        ball = positions[:, -1].astype(float)
        centre = np.asarray(pitch, dtype=float) / 2
        alive = np.isin(labels, [metrics.H, metrics.A])
        switches = np.flatnonzero((labels[:-1] == metrics.D) & alive[1:]) + 1
        refs = [int(k) for k in switches
                if np.isfinite(ball[k]).all() and np.linalg.norm(ball[k] - centre) <= REFERENCE_RADIUS_M]
        out.append({"source": "idsse", "half": f"{match}-{half}", "ball": ball,
                    "centre": centre, "fps": IDSSE_FPS, "refs": refs})
    return out


def metrica_halves(root):
    out = []
    for game in METRICA_GAMES:
        folder = Path(root) / "data" / f"Sample_Game_{game}"
        tracking = folder / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv"
        with open(tracking, newline="") as handle:
            rows = csv.reader(handle)
            for _ in range(2):
                next(rows)
            header = next(rows)
            x_col = header.index("Ball")
            frames, ball = [], []
            for row in rows:
                frames.append(int(row[1]))
                ball.append((float(row[x_col]), float(row[x_col + 1])))
        frames = np.array(frames)
        if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
            raise ValueError(f"{tracking}: frames are not contiguous from 1")
        ball = np.array(ball) * np.array(METRICA_PITCH)
        refs = set()
        with open(folder / f"Sample_Game_{game}_RawEventsData.csv", newline="") as handle:
            for event in csv.DictReader(handle):
                if event["Type"] == "SET PIECE" and event["Subtype"] == "KICK OFF":
                    refs.add(int(event["Start Frame"]) - 1)
        out.append({"source": "metrica", "half": f"game{game}", "ball": ball,
                    "centre": np.array(METRICA_PITCH) / 2, "fps": METRICA_FPS, "refs": sorted(refs)})
    return out


def evaluate(half, condition, seed, radius):
    fps = half["fps"]
    ball = half["ball"]
    if CONDITIONS[condition] is not None:
        rng = np.random.default_rng(seed)
        ball = mask_ball(ball[:, None, :], fps, rng, CONDITIONS[condition], COVERAGE)[:, -1]
    tag = int(round(TAG_FRAMES * fps / 25))
    pad = int(round(WINDOW_PAD_S * fps))
    rows = []
    for k in half["refs"]:
        if k < MIN_HISTORY_S * fps:
            continue
        rng = np.random.default_rng([seed, k])
        start = k - int(rng.integers(0, tag))
        window = (max(0, start - pad), start + tag + pad, start + tag // 2)
        frame, method = detect_kickoff(ball, window, half["centre"], fps, radius)
        rows.append({"source": half["source"], "half": half["half"], "condition": condition,
                     "seed": seed, "radius": radius, "reference": k, "detected": frame, "method": method,
                     "error": frame - k, "fallback_error": window[2] - k,
                     "reference_distance_m": float(np.linalg.norm(half["ball"][k] - half["centre"])),
                     "ball_at_reference": bool(np.isfinite(ball[k]).all())})
    return rows


def summary(rows, key):
    if not rows:
        return {}
    error = np.array([r[key] for r in rows])
    return {
        "n": len(rows),
        "within_2_frames": float(np.mean(np.abs(error) <= TOLERANCE_FRAMES)),
        "late_over_2_frames": float(np.mean(error > TOLERANCE_FRAMES)),
        "early_over_1s": float(np.mean(error < -25)),
        "median_abs_frames": float(np.median(np.abs(error))),
        "p90_abs_frames": float(np.percentile(np.abs(error), 90)),
    }


def run(args):
    with ProcessPoolExecutor(min(len(IDSSE_MATCHES), args.workers)) as pool:
        halves = [h for match in pool.map(idsse_halves, [args.idsse] * len(IDSSE_MATCHES), IDSSE_MATCHES)
                  for h in match]
    halves += metrica_halves(args.metrica)
    rows = []
    for radius in RADII_M:
        for half in halves:
            for condition, gap in CONDITIONS.items():
                for seed in (MASK_SEEDS if gap is not None else (0,)):
                    rows.extend(evaluate(half, condition, seed, radius))
    fit = {radius: summary([r for r in rows if r["radius"] == radius and r["source"] == "idsse"],
                           "error")["within_2_frames"] for radius in RADII_M}
    chosen = max(RADII_M, key=lambda radius: (fit[radius], -radius))
    report = {"settings": {"tag_frames": TAG_FRAMES, "reference_radius_m": REFERENCE_RADIUS_M,
                           "min_history_s": MIN_HISTORY_S, "coverage": COVERAGE,
                           "conditions": CONDITIONS, "mask_seeds": MASK_SEEDS, "radii_m": RADII_M},
              "idsse_within_2_frames_by_radius": fit, "chosen_radius_m": chosen,
              "references": {h["half"]: h["refs"] for h in halves}, "results": {}}
    print(f"IDSSE share within 2 frames by radius: {fit}; chosen {chosen} m")
    natural = [r for r in rows if r["condition"] == "natural" and r["radius"] == chosen]
    bands = {"<=1": (0, 1), "1-2": (1, 2), "2-3": (2, 3), ">3": (3, np.inf)}
    report["reference_distance_bands_m"] = {
        name: sum(lo < r["reference_distance_m"] <= hi or (lo == 0 and r["reference_distance_m"] == 0)
                  for r in natural) for name, (lo, hi) in bands.items()}
    print(f"reference distance to spot at the kickoff frame: {report['reference_distance_bands_m']}")
    for radius, source, condition in (
            (radius, source, condition) for radius in RADII_M
            for source in ("idsse", "metrica", "all") for condition in CONDITIONS):
        subset = [r for r in rows if r["condition"] == condition and r["radius"] == radius
                  and (source == "all" or r["source"] == source)]
        hits = [r for r in subset if r["method"] == "tracking"]
        report["results"][f"r{radius:g}/{source}/{condition}"] = {
            "tracking_rate": len(hits) / len(subset) if subset else None,
            "tracking_hits": summary(hits, "error"),
            "detector": summary(subset, "error"),
            "tag_midpoint_only": summary(subset, "fallback_error"),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "kickoff-validation.json").write_text(json.dumps(report, indent=1))
    with open(args.output / "kickoff-validation.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'radius/source/condition':<26} {'n':>4} {'track':>6} {'<=2f':>6} {'late':>6} {'medAbs':>7}"
          f" | {'tag <=2f':>8} {'tag late':>8} {'tag med':>8}")
    for name, item in report["results"].items():
        d, t = item["detector"], item["tag_midpoint_only"]
        if d:
            print(f"{name:<26} {d['n']:>4} {item['tracking_rate']:>6.2f} {d['within_2_frames']:>6.2f}"
                  f" {d['late_over_2_frames']:>6.2f} {d['median_abs_frames']:>7.1f}"
                  f" | {t['within_2_frames']:>8.2f} {t['late_over_2_frames']:>8.2f} {t['median_abs_frames']:>8.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the kickoff detector on external kickoffs.")
    parser.add_argument("--idsse", type=Path, required=True)
    parser.add_argument("--metrica", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=7)
    run(parser.parse_args())
