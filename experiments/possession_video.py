"""Measure tracking-only coordinate fusion and extract local video ball candidates."""
import argparse
import time
from pathlib import Path

import numpy as np

from experiments.possession_execution import (
    dump, fingerprint, fixed, measure, restart_cues, split_members, dataset,
)
from player_track_quality import repair_player_tracks
from sequence_possession import restore, sequence_features, sequence_probabilities, visibility_features


def support(points, confidence, fps, speed_limit, slack):
    """Require a candidate on both sides and plausible velocity change."""
    n, k = confidence.shape
    result = np.full((n, 2), np.nan, dtype=np.float32)
    quality = np.zeros(n, dtype=np.float32)
    limit = speed_limit / fps + slack
    for i in range(1, n - 1):
        if not confidence[i].any():
            continue
        p = points[i]
        neighbors = []
        for j in (i - 1, i + 1):
            delta = np.linalg.norm(p[:, None] - points[j], axis=2)
            delta[~np.isfinite(delta)] = np.inf
            near = delta.argmin(axis=1)
            neighbors.append((points[j, near], delta[np.arange(k), near], confidence[j, near]))
        left, ls, lc = neighbors[0]
        right, rs, rc = neighbors[1]
        curve = np.linalg.norm(left + right - 2 * p, axis=1)
        score = np.minimum(np.minimum(lc, rc), confidence[i])
        score[~((ls <= limit) & (rs <= limit) & (curve <= limit))] = 0
        best = score.argmax()
        if score[best] > 0:
            result[i], quality[i] = p[best], score[best]
    return result, quality


def suspect(ball, fps):
    step = np.linalg.norm(np.diff(ball, axis=0), axis=1)
    chord = np.linalg.norm(ball[:-2] - ball[2:], axis=1)
    jump = 35 / fps + 0.75
    spike = np.r_[False, (step[:-1] > jump) & (step[1:] > jump) & (chord < jump), False]
    holds = np.flatnonzero(np.diff(np.r_[False, step < 0.025, False])).reshape(-1, 2)
    still = np.zeros(len(ball), dtype=bool)
    for a, b in holds:
        if b - a >= fps:
            still[a + 1:b + 1] = True
    return spike, still


def fuse(raw, candidates, quality, pitch, fps, threshold, fill, replace):
    """Gate pitch-space coordinates with support independent of the raw observation at that frame."""
    observed = np.isfinite(raw).all(axis=1)
    spike, still = suspect(raw, fps)
    eligible = ((quality >= threshold) & np.isfinite(candidates).all(axis=1)
                & ((candidates >= -3) & (candidates <= pitch + 3)).all(axis=1))
    # Keep original points outside suspect spans as potential anchors.
    anchors = np.flatnonzero(observed & ~(spike | still))
    index = np.searchsorted(anchors, np.arange(len(raw)))
    for i in np.flatnonzero(eligible):
        adjacent = anchors[max(0, index[i] - 1):index[i] + 2]
        nearby = adjacent[(np.abs(adjacent - i) <= max(1, round(0.2 * fps))) & (adjacent != i)]
        if len(nearby):
            distance = np.linalg.norm(raw[nearby] - candidates[i], axis=1)
            eligible[i] &= np.all(distance <= 35 * np.abs(nearby - i) / fps + 1.5)
    disagreement = np.linalg.norm(raw - candidates, axis=1)
    accepted = eligible & (((fill & ~observed)
                           | (replace & observed & ((spike & (disagreement > 2))
                                                     | (still & (disagreement > 3))))))
    track = raw.copy()
    track[accepted] = candidates[accepted]
    provenance = np.where(observed, "observed", "missing")
    provenance[accepted & ~observed] = "filled"
    provenance[accepted & observed] = "replaced"
    reason = np.full(len(raw), "", dtype="<U20")
    reason[accepted & ~observed] = "supported_video_gap"
    reason[accepted & spike] = "isolated_jump"
    reason[accepted & still] = "implausible_hold"
    error_proxy = np.full(len(raw), np.nan, dtype=np.float32)
    speed = np.r_[0, np.linalg.norm(np.diff(candidates, axis=0), axis=1) * fps]
    error_proxy[accepted] = (0.8 + 1.5 * (1 - quality[accepted])
                             + np.where(speed[accepted] > 15, 2.0, 0.0))
    return track, provenance, reason, error_proxy


def corrupt(raw, fps, seed):
    """Generate observation faults and an independent, imperfect sensor from a reference copy."""
    rng = np.random.default_rng(seed)
    observed = raw.copy()
    starts = rng.choice(np.arange(2, len(raw) - 4 * fps), len(raw) // (10 * fps), replace=False)
    for index, start in enumerate(starts):
        if index % 3 == 0:
            observed[start:start + rng.integers(fps // 5, 3 * fps)] = np.nan
        elif index % 3 == 1:
            observed[start] += rng.choice((-1, 1), 2) * rng.uniform(4, 15, 2)
        elif np.isfinite(observed[start]).all():
            observed[start:start + rng.integers(fps, 4 * fps)] = observed[start]
    candidates = np.full((len(raw), 2, 2), np.nan, dtype=np.float32)
    candidates[:, 0] = raw + rng.normal(0, 0.5, raw.shape)
    candidates[:, 1] = raw + rng.normal(0, 10, raw.shape)
    confidence = rng.uniform(0.25, 0.95, (len(raw), 2)).astype(np.float32)
    confidence[:, 1] *= 0.65
    dropout = np.convolve(rng.random(len(raw)) < 0.035, np.ones(7), mode="same") > 0
    candidates[dropout, 0] = np.nan
    confidence[~np.isfinite(candidates).all(axis=2)] = 0
    return observed, candidates, confidence


ARMS = {"baseline": (False, False), "fill": (True, False),
        "replace": (False, True), "fill_replace": (True, True)}


def ball_metrics(reference, raw, result, provenance, fps):
    prior = np.linalg.norm(raw - reference, axis=1)
    after = np.linalg.norm(result - reference, axis=1)
    replaced = provenance == "replaced"
    filled = provenance == "filled"
    reference_visible = np.isfinite(reference).all(axis=1)
    good = prior < 1
    speed = np.r_[0, np.linalg.norm(np.diff(reference, axis=0), axis=1) * fps]
    valid = np.isfinite(after)
    return {"filled": int(filled.sum()), "replaced": int(replaced.sum()),
            "eligible_gap_points": int((reference_visible & ~np.isfinite(prior)).sum()),
            "replacement_improved": int((replaced & (after < prior)).sum()),
            "good_points_made_worse": int((good & (after > prior + 0.5)).sum()),
            "initially_good": int(good.sum()), "evaluated": int(valid.sum()),
            "fast_evaluated": int((valid & (speed > 15)).sum()),
            "fast_error_sum": float(after[valid & (speed > 15)].sum()),
            "error_sum": float(after[valid].sum()), "filled_error_sum": float(after[filled].sum())}


def stress(args):
    import torch
    device = torch.device(args.device)
    model, _ = restore(args.gru, device)
    members = split_members(args.skillcorner)
    settings = {"grus": fingerprint(args.gru), "seed": 626, "thresholds": [0.35, 0.50, 0.65],
                "arms": ARMS, "provider": "IDSSE", "report": "synthetic two-sensor stress only"}
    dump(args.output / "fusion-manifest.json", settings)
    frozen_threshold = None
    raw_mask_diagnostic = False
    for role in ("dev", "test"):
        results = {}
        for match in (match for match in members[role] if isinstance(match, str)):
            root, loader, fps = dataset(match, args.skillcorner, args.idsse)
            for half, (tracks, weights, teams, pitch, target) in enumerate(loader(root, match), 1):
                raw, images, confidence = corrupt(tracks[:, -1], fps, 626 + int(match[1:], 36) + half)
                candidate, quality = support(images, confidence, fps, 35, 0.75)
                edge = restart_cues(tracks, teams, pitch, fps)[-1]
                raw_visible = visibility_features(raw, fps)
                repaired = repair_player_tracks(tracks, fps)
                for threshold in settings["thresholds"] if role == "dev" else [frozen_threshold]:
                    for name, (fill, replace) in ARMS.items():
                        ball, provenance, reason, uncertainty = fuse(
                            raw, candidate, quality, pitch, fps, threshold, fill, replace)
                        positions = repaired.copy()
                        positions[:, -1] = ball
                        point_weights = np.ones(len(ball)) if weights is None else weights.copy()
                        accepted = (provenance == "filled") | (provenance == "replaced")
                        point_weights[accepted] = quality[accepted]
                        x, mx = sequence_features(positions, point_weights, teams, pitch, fps, True)
                        modes = ("fused-mask",) if role == "test" and not raw_mask_diagnostic else (
                            "fused-mask", "raw-mask")
                        for mode in modes:
                            if mode == "raw-mask":
                                x[:, -2:] = mx[:, -2:] = raw_visible
                            p = sequence_probabilities(model, x, mx, fps, device)
                            pred = fixed(p, fps)
                            score = measure(target, pred, ~np.isfinite(raw).all(axis=1), edge, fps)
                            score.update(ball_metrics(tracks[:, -1], raw, ball, provenance, fps))
                            results.setdefault(f"{threshold:g}/{name}/{mode}", {}).setdefault(match, []).append(score)
                        if role == "test":
                            np.savez_compressed(args.output / f"fusion-{match}-{half}-{name}.npz",
                                                positions=ball, provenance=provenance, reason=reason,
                                                confidence=quality, uncertainty_metres=uncertainty,
                                                raw_visibility=raw_visible)
                print(f"Measured fusion {role} {match} half {half}", flush=True)
        dump(args.output / f"fusion-{role}.json", results)
        if role == "dev":
            def mean(threshold, name, mode, score):
                values = results[f"{threshold:g}/{name}/{mode}"]
                return np.mean([r[score] for halves in values.values() for r in halves])
            frozen_threshold = max(settings["thresholds"], key=lambda t: mean(t, "fill_replace", "fused-mask", "spell_f1"))
            raw_mask_diagnostic = (mean(frozen_threshold, "fill_replace", "fused-mask", "in_pitch_dead_f1")
                                   < mean(frozen_threshold, "baseline", "fused-mask", "in_pitch_dead_f1"))
            dump(args.output / "fusion-selection.json", {"threshold": frozen_threshold,
                 "raw_mask_diagnostic": bool(raw_mask_diagnostic),
                 "promotion": "unresolved; synthetic stress only"})
            print(f"Frozen fusion stress threshold {frozen_threshold}; raw mask {raw_mask_diagnostic}",
                  flush=True)


def video_candidates(video, checkpoint, frame_numbers, imgsz, device, *, tiled=False):
    import cv2
    from ultralytics import YOLO
    from video_ball import ball_class_ids

    model = YOLO(str(checkpoint), task="detect")
    ids = ball_class_ids(model.names)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    points = np.full((len(frame_numbers), 8, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((len(frame_numbers), 8), dtype=np.float32)
    start = time.monotonic()
    next_frame = -1
    for i in range(0, len(frame_numbers), 8):
        batch = []
        for frame in frame_numbers[i:i + 8]:
            if frame != next_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, im = cap.read()
            if not ok:
                raise ValueError(f"cannot read frame {frame}")
            next_frame = frame + 1
            batch.append(im)
        height, width = batch[0].shape[:2]
        regions = [(0, 0, width, height)]
        if tiled:
            regions = [(x0, y0, x1, y1)
                       for x0, x1 in ((0, 3 * width // 5), (2 * width // 5, width))
                       for y0, y1 in ((0, 3 * height // 5), (2 * height // 5, height))]
        detections = model.predict([im[y0:y1, x0:x1] for im in batch
                                    for x0, y0, x1, y1 in regions],
                                   imgsz=imgsz, conf=0.05, classes=ids, device=device,
                                   verbose=False, max_det=8)
        for frame_idx in range(len(batch)):
            candidates = []
            for roi, (x0, y0, _, _) in enumerate(regions):
                box = detections[frame_idx * len(regions) + roi].boxes
                xy = box.xywh[:, :2].cpu().numpy() + [x0, y0]
                for center, conf in zip(xy, box.conf.cpu().numpy(), strict=True):
                    candidates.append((float(conf), center))
            count = 0
            for conf, center in sorted(candidates, key=lambda item: -item[0]):
                if np.any(np.linalg.norm(points[i + frame_idx, :count] - center, axis=1) < 10):
                    continue
                points[i + frame_idx, count] = center
                confidence[i + frame_idx, count] = conf
                count += 1
                if count == points.shape[1]:
                    break
    cap.release()
    return points, confidence, fps, time.monotonic() - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("skillcorner", "idsse", "gru", "output"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stress(args)
