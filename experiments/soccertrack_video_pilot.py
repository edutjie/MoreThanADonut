"""Run local detector comparisons with independent SoccerTrack image-space ball labels."""
import argparse
import ast
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from experiments.possession_execution import dump, fingerprint
from experiments.possession_video import support, video_candidates

REVISION = "325bd4c57ee8e135ab652f3921d274526ada496b"
MATCHES = {"dev": 117092, "confirmation": 128057}
RESTARTS = {"dev": 6575, "confirmation": 386}  # Provider 1-based action frames minus one.
TOLERANCE_PIXELS = 14
THRESHOLDS = (0.05, 0.1, 0.25, 0.5)
IMAGE_SIZE = 1280


def paths(root, match):
    video = root / "videos" / str(match) / f"{match}_panorama_1st_half.mp4"
    if not video.is_file():
        video = video.with_name(f"{match}-pilot.mp4")
    raw = root / "raw" / str(match) / str(match)
    return video, raw


def check_source(root, match):
    video, raw = paths(root, match)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    actions = root / "bas" / str(match) / f"{match}_12_class_events.json"
    provider = json.loads(actions.read_text())
    first = [x for x in provider["actions"] if x["gameTime"].startswith("1 -")]
    max_clock_error = max(abs((x["frame"] - 1) / fps - int(x["position"]) / 1000) for x in first)
    assert fps == provider["fps"] == 25 and int(first[-1]["frame"]) <= frames
    assert max_clock_error <= 1 / fps + 0.001
    keypoints = json.loads(Path(f"{raw}_keypoints.json").read_text())
    xy = np.array(list(keypoints.values()), dtype=np.float64)
    pitch = np.array([ast.literal_eval(key) for key in keypoints], dtype=np.float64)
    matrix = np.load(f"{raw}_homography.npy")
    projected = cv2.perspectiveTransform(pitch[:, None], matrix)[:, 0]
    errors = np.linalg.norm(projected - xy, axis=1)
    remap = {}
    map_paths = [Path(f"{raw}_map{axis}.npy") for axis in ("x", "y")]
    if all(path.is_file() for path in map_paths):
        maps = cv2.convertMaps(*(np.load(path) for path in map_paths), cv2.CV_32FC1)
        restored = np.column_stack([map_coordinates(
            axis, projected[:, ::-1].T, order=1, mode="constant", cval=np.nan) for axis in maps])
        valid = np.isfinite(restored).all(axis=1)
        remap = {"map_sha256": [fingerprint(path) for path in map_paths],
                 "calibrated_keypoints_checked": int(valid.sum()),
                 "remapped_keypoint_median_pixels": float(np.median(np.linalg.norm(
                     restored[valid] - xy[valid], axis=1)))}
    return {"video_dimensions": [w, h], "fps": fps, "half_frames": frames,
            "provider_actions_sha256": fingerprint(actions), "frame_clock_max_error_seconds": max_clock_error,
            "provider_homography_sha256": fingerprint(Path(f"{raw}_homography.npy")),
            "provided_homography_keypoint_median_pixels": float(np.median(errors)),
            "pitch_projection_verified": False, **remap}


def score(frames, points, confidence, annotations, threshold):
    errors, top_errors = [], []
    missed = duplicate_or_wrong = evaluated = 0
    for row in annotations:
        if row["status"] != "visible":
            continue
        i = int(frames.searchsorted(int(row["frame"])))
        keep = confidence[i] >= threshold
        center = np.array([float(row["x"]), float(row["y"])])
        distances = np.linalg.norm(points[i, keep] - center, axis=1)
        hit = len(distances) and distances.min() <= TOLERANCE_PIXELS
        duplicate_or_wrong += len(distances) - int(bool(hit))
        evaluated += 1
        missed += int(not hit)
        if hit:
            errors.append(float(distances.min()))
        if len(distances) and distances[0] <= TOLERANCE_PIXELS:
            top_errors.append(float(distances[0]))
    return {"visible": evaluated, "recall": (evaluated - missed) / evaluated,
            "missed": missed, "extra_detections_on_visible_frames": duplicate_or_wrong,
            "top_hit_count": len(top_errors),
            "mean_hit_pixels": float(np.mean(errors)) if errors else None}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with args.annotations.open() as file:
        annotations = list(csv.DictReader(file))
    reference = {role: [r for r in annotations if int(r["match"]) == match] for role, match in MATCHES.items()}
    manifest = {
        "matches": MATCHES, "revision": REVISION, "restart_anchors_zero_based": RESTARTS,
        "sample": "union of ±6 source frames around each independently reviewed frame",
        "detector_sha256": fingerprint(args.detector), "annotations_sha256": fingerprint(args.annotations),
        "tiled": args.tiled, "backend": "local Ultralytics PyTorch",
        "thresholds": list(THRESHOLDS), "imgsz": IMAGE_SIZE, "tolerance_pixels": TOLERANCE_PIXELS,
        "source": {role: check_source(args.soccertrack, match) for role, match in MATCHES.items()},
    }
    dump(args.output / "pilot-manifest.json", manifest)
    best = None
    for role, match in MATCHES.items():
        frames = np.unique([max(0, f + delta) for r in reference[role]
                            for f in (int(r["frame"]),) for delta in range(-6, 7)])
        video, _ = paths(args.soccertrack, match)
        source = args.output / f"{role}-raw.npz"
        if source.is_file():
            with np.load(source) as array:
                assert np.array_equal(frames, array["frames"])
                assert array["detector_sha256"] == manifest["detector_sha256"]
                points, confidence, runtime = array["points"], array["confidence"], float(array["runtime"])
        else:
            points, confidence, fps, runtime = video_candidates(
                video, args.detector, frames, IMAGE_SIZE, args.device, tiled=args.tiled)
            np.savez_compressed(source, frames=frames, points=points, confidence=confidence,
                                runtime=runtime, detector_sha256=manifest["detector_sha256"])
        values = {t: score(frames, points, confidence, reference[role], t)
                  for t in (THRESHOLDS if role == "dev" else (best,))}
        if role == "dev":
            feasible = [t for t, result in values.items()
                        if result["extra_detections_on_visible_frames"] <= 0.5 * result["visible"]]
            best = max(feasible or list(values), key=lambda t: (
                values[t]["recall"] if feasible else -values[t]["extra_detections_on_visible_frames"], t))
            dump(args.output / "pilot-selection.json", {"confidence": best})
        linked = np.full((len(frames), 2), np.nan, dtype=np.float32)
        strengths = np.zeros(len(frames), dtype=np.float32)
        for idx in np.split(np.arange(len(frames)), np.flatnonzero(np.diff(frames) != 1) + 1):
            cs = np.where(confidence[idx] >= best, confidence[idx], 0)
            pts = np.where(cs[:, :, None] > 0, points[idx], np.nan)
            linked[idx], strengths[idx] = support(pts, cs, 25, 1500, 6)
        np.savez_compressed(args.output / f"{role}-linked.npz", frames=frames, points=linked,
                            confidence=strengths)
        steps = np.linalg.norm(np.diff(linked, axis=0), axis=1)
        restart = RESTARTS[role]
        near = (np.abs(frames[1:] - restart) <= 6) & (np.diff(frames) == 1) & np.isfinite(steps)
        transition = int(frames[1:][near][steps[near].argmax()]) if near.any() else None
        result = {"raw_scores": {str(k): v for k, v in values.items()},
                  "linked": score(frames, linked[:, None], strengths[:, None], reference[role], best),
                  "selected_confidence": best, "n_inference_frames": len(frames), "runtime_seconds": runtime,
                  "linked_frames": int((strengths > 0).sum()),
                  "unscored_annotation_frames": sum(r["status"] != "visible" for r in reference[role]),
                  "restart_peak_motion_frame": transition,
                  "restart_error_seconds": None if transition is None else (transition - restart) / 25}
        dump(args.output / f"pilot-{role}.json", result)
        print(role, result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("soccertrack", "annotations", "detector", "output"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tiled", action="store_true")
    run(parser.parse_args())
