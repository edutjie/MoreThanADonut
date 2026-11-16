"""Extract whistle and referee-gesture cues, and use them during possession decoding."""

import argparse
import csv
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames

SAMPLE_RATE = 16000
AUDIO_WINDOW = 512
AUDIO_HOP = 160
CUE_NAMES = ("whistle", "referee_arm")


def whistle_frames(samples, fps):
    """Find short, tonal onsets over the local noise floor in bands of 2–5 kHz."""
    if len(samples) < AUDIO_WINDOW:
        return np.empty(0, dtype=int)
    windows = np.lib.stride_tricks.sliding_window_view(samples, AUDIO_WINDOW)[::AUDIO_HOP]
    frequencies = np.fft.rfftfreq(AUDIO_WINDOW, 1 / SAMPLE_RATE)
    band = (frequencies >= 2000) & (frequencies <= 5000)
    audible = (frequencies >= 400) & (frequencies <= 6500)
    peaks, concentrations = [], []
    taper = np.hanning(AUDIO_WINDOW)
    for offset in range(0, len(windows), 4096):
        spectrum = np.abs(np.fft.rfft(windows[offset:offset + 4096] * taper)) ** 2
        tonal = spectrum[:, band]
        peak = (
            tonal[:, :-2] + tonal[:, 1:-1] + tonal[:, 2:]
        ).max(axis=1)
        peaks.append(peak)
        concentrations.append(peak / np.maximum(spectrum[:, audible].sum(axis=1), 1e-14))
    db = 10 * np.log10(np.maximum(np.concatenate(peaks), 1e-14))
    tonal = np.concatenate(concentrations) >= 0.35
    # Compare with the immediately preceding 0.5 seconds. Sustained crowd tones
    # only get an onset if they also pass the short-burst duration test.
    history = np.lib.stride_tricks.sliding_window_view(np.pad(db, (50, 0), mode="edge"), 50)[:-1]
    noise = np.concatenate([np.median(history[offset:offset + 4096], axis=1)
                            for offset in range(0, len(history), 4096)])
    above = tonal & (db >= noise + 10)
    edges = np.flatnonzero(np.diff(np.r_[False, above, False]))
    keep = [(start, stop) for start, stop in zip(edges[::2], edges[1::2], strict=True)
            if 4 <= stop - start <= 60 and (stop + 15 >= len(tonal) or not tonal[stop:stop + 15].all())]
    frames = np.array([round((start * AUDIO_HOP + AUDIO_WINDOW / 2) * fps / SAMPLE_RATE)
                       for start, _ in keep], dtype=int)
    # Merge brief breaks in the same whistle.
    return frames[np.r_[True, np.diff(frames) >= round(0.25 * fps)]] if len(frames) else frames


def read_audio(video, first, stop, fps):
    command = ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(first / fps), "-i", str(video),
               "-t", str((stop - first) / fps), "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    return np.frombuffer(result.stdout, dtype="<f4")


def extract_whistles(video, first, stop, fps):
    frames = first + whistle_frames(read_audio(video, first, stop, fps), fps)
    return frames[frames < stop]


def raised_arm(keypoints):
    """COCO arm with confident joints and a wrist visibly above the shoulder."""
    for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
        joints = keypoints[[shoulder, elbow, wrist]]
        if joints[:, 2].min() < 0.45:
            continue
        upper, lower = joints[1, :2] - joints[0, :2], joints[2, :2] - joints[1, :2]
        arm_length = np.linalg.norm(upper) + np.linalg.norm(lower)
        if arm_length > 0 and upper[1] < 0 and (upper[1] + lower[1]) < -0.55 * arm_length:
            return True
    return False


def projected_official(foot, inverse, players, pitch):
    projected = inverse @ [*foot, 1.0]
    if abs(projected[2]) < 1e-9:
        return None
    point = projected[:2] / projected[2] * pitch
    # Relaxed: allow anywhere near pitch (within 5m of boundaries)
    if not np.all((point >= -5.0) & (point <= pitch + 5.0)):
        return None
    # Relaxed: allow closer to players (1.0m instead of 1.25m)
    if len(players) and np.any(np.linalg.norm(players - point, axis=1) < 1.0):
        return None
    return point


def official_candidate(image, box, joints, inverse, players, pitch):
    """Find a yellow-shirted on-pitch person apart from the tracked players."""
    import cv2

    left, top, right, bottom = box
    # Relaxed: minimum height 15 pixels, shoulder confidence 0.25
    if bottom - top < 15 or joints[[5, 6], 2].min() < 0.25:
        return None
    ankles = joints[[15, 16]]
    reliable = ankles[:, 2] >= 0.3  # Relaxed from 0.4
    foot = ankles[reliable, :2].mean(axis=0) if reliable.any() else [(left + right) / 2, bottom]
    point = projected_official(foot, inverse, players, pitch)
    if point is None:
        return None

    shoulder = joints[[5, 6], :2].mean(axis=0)
    width = right - left
    height = bottom - top
    x0, x1 = np.clip([shoulder[0] - 0.16 * width, shoulder[0] + 0.16 * width + 1], 0, image.shape[1]).astype(int)
    y0, y1 = np.clip([shoulder[1], shoulder[1] + 0.20 * height + 1], 0, image.shape[0]).astype(int)
    crop = image[y0:y1, x0:x1]
    if not crop.size:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Relaxed: accept any bright color (not just yellow), threshold 0.15
    yellow = (cv2.inRange(hsv, (10, 60, 80), (45, 255, 255)) > 0).mean()
    return (point, yellow, raised_arm(joints)) if yellow >= 0.15 else None


def referee_crops(images, results, indices, matrices, players, pitch):
    crops, assignments = [], []
    for index, (image, result, frame) in enumerate(zip(images, results, indices, strict=True)):
        matrix = matrices[frame]
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) <= 1e-12:
            continue
        inverse = np.linalg.inv(matrix)
        for left, top, right, bottom, confidence, _ in result.boxes.data.cpu().numpy():
            height, width = bottom - top, right - left
            if height < 18 or width <= 0:
                continue
            point = projected_official(((left + right) / 2, bottom), inverse, players[frame], pitch)
            if point is None:
                continue
            x0, y0 = np.maximum(np.floor([left - .5 * height, top - .5 * height]), 0).astype(int)
            x1, y1 = np.minimum(np.ceil([right + .5 * height, bottom + .15 * height]),
                                [image.shape[1], image.shape[0]]).astype(int)
            crop = image[y0:y1, x0:x1]
            if crop.size:
                crops.append(crop)
                assignments.append((index, point, confidence, (left + right) / 2 - x0, (top + bottom) / 2 - y0))
    return crops, assignments


def open_vocab_candidates(locator, pose_model, images, indices, matrices, players, pitch, device, counts):
    results = locator.predict(images, imgsz=1280, conf=0.25, max_det=8, device=device, verbose=False)
    counts["boxes"] += sum(len(result.boxes) for result in results)
    crops, assignments = referee_crops(images, results, indices, matrices, players, pitch)
    counts["crops"] += len(crops)
    candidates = [[] for _ in images]
    for start in range(0, len(crops), 8):
        predictions = pose_model.predict(crops[start:start + 8], imgsz=256, conf=0.25, device=device, verbose=False)
        for result, (index, point, confidence, x, y) in zip(predictions, assignments[start:start + 8], strict=True):
            if not len(result.boxes) or result.keypoints is None:
                continue
            centers = result.boxes.xywh.cpu().numpy()[:, :2]
            nearest = np.linalg.norm(centers - [x, y], axis=1).argmin()
            joints = result.keypoints.data[nearest].cpu().numpy()
            if joints.shape != (17, 3):
                raise ValueError("expected 17 COCO keypoints with confidence")
            candidates[index].append((point, confidence, raised_arm(joints)))
    return candidates


def extract_referee_arms(video, checkpoint, device, blocks, fps, data_dir, referee_model=None):
    import cv2
    from ultralytics import YOLO

    from starter.baseline import team_of_each_track
    from starter.load_homography import check_alignment, load
    from tactical_baseline import read_tracking

    if not checkpoint.is_file():
        raise ValueError(f"pose checkpoint not found: {checkpoint}")
    model = YOLO(str(checkpoint), task="pose")
    if model.task != "pose":
        raise ValueError("referee detection requires a COCO pose checkpoint")
    locator = None
    if referee_model is not None:
        from ultralytics import YOLOE
        from ultralytics.utils.files import WorkingDirectory

        if not referee_model.is_file():
            raise ValueError(f"YOLOE checkpoint not found: {referee_model}")
        locator = YOLOE(str(referee_model.resolve()))
        with WorkingDirectory(referee_model.resolve().parent):
            if not Path("mobileclip2_b.ts").is_file():
                raise ValueError("put mobileclip2_b.ts next to the YOLOE-26 checkpoint")
            locator.set_classes(["soccer referee"])
    if not check_alignment(data_dir=data_dir)["agrees"]:
        raise ValueError("referee detection requires frame-aligned homographies")
    matrices, _, valid = load(data_dir=data_dir)
    pitch = np.asarray(GAMES[GAME]["pitch"], dtype=np.float32)
    meta, raw = require(["atd_meta", "atd_raw"], data_dir=data_dir)
    players = read_tracking(raw, source_frames(), len(team_of_each_track(meta)), pitch)[:, :-1]
    stride = max(1, round(fps / 5))
    events = []
    candidate_frames = 0
    counts = Counter()
    cap = cv2.VideoCapture(str(video))
    try:
        if (not cap.isOpened() or round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(matrices)
                or not np.isclose(cap.get(cv2.CAP_PROP_FPS), fps)):
            raise ValueError("tactical video and pose timeline disagree")
        for first, stop in blocks:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
                raise ValueError(f"could not seek to frame {first}")
            last_point = None
            streak = 0
            last_event = first - 2 * fps
            for start in range(first, stop, stride * 8):
                indices, images = [], []
                for frame in range(start, min(start + stride * 8, stop)):
                    if frame % stride != first % stride or not valid[frame]:
                        if not cap.grab():
                            raise ValueError(f"could not decode frame {frame}")
                        continue
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f"could not read frame {frame}")
                    indices.append(frame)
                    images.append(image)
                if not images:
                    continue
                if locator is not None:
                    candidates_by_frame = open_vocab_candidates(
                        locator, model, images, indices, matrices, players, pitch, device, counts)
                else:
                    predictions = model.predict(images, imgsz=1280, conf=0.25, device=device, verbose=False)
                    candidates_by_frame = [
                        pose_candidates(image, result, matrices[frame], players[frame], pitch)
                        for frame, image, result in zip(indices, images, predictions, strict=True)
                    ]
                for frame, candidates in zip(indices, candidates_by_frame, strict=True):
                    adjacent = last_point is not None and frame - last_frame == stride
                    nearby = [c for c in candidates if adjacent and np.linalg.norm(c[0] - last_point) < 3]
                    if not candidates:
                        last_point, streak = None, 0
                        continue
                    candidate_frames += 1
                    point, _, arm = max(nearby or candidates, key=lambda c: c[1])
                    streak = (streak + 1 if nearby else 1) if arm else 0
                    last_point, last_frame = point, frame
                    onset = frame - stride
                    if streak == 2 and onset - last_event >= 2 * fps:
                        events.append(onset)
                        last_event = onset
    finally:
        cap.release()
    if locator is not None:
        print(f"YOLOE boxes: {counts['boxes']}; eligible pose crops: {counts['crops']}", flush=True)
    print(f"Referee candidates with pose: {candidate_frames} sampled frames", flush=True)
    return events


def pose_candidates(image, result, matrix, players, pitch):
    if result.keypoints is None or result.keypoints.data.shape[1:] != (17, 3):
        raise ValueError("expected 17 COCO keypoints with confidence")
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) <= 1e-12:
        return []
    inverse = np.linalg.inv(matrix)
    boxes = result.boxes.xyxy.cpu().numpy()
    poses = result.keypoints.data.cpu().numpy()
    return [candidate for box, joints in zip(boxes, poses, strict=True)
            if (candidate := official_candidate(image, box, joints, inverse, players, pitch)) is not None]


def read_cues(path, n_frames):
    cues = np.zeros((n_frames, len(CUE_NAMES)), dtype=bool)
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != ["frame", "cue"]:
            raise ValueError(f"{path}: expected frame,cue header")
        for number, row in enumerate(rows, start=2):
            token = row["frame"] or ""
            if row["cue"] not in CUE_NAMES or not token.isascii() or not token.isdecimal() or None in row:
                raise ValueError(f"{path}: invalid cue on row {number}")
            frame = int(token)
            if not 0 <= frame < n_frames:
                raise ValueError(f"{path}: frame out of range on row {number}")
            cues[frame, CUE_NAMES.index(row["cue"])] = True
    return cues


def stoppage_entry_bonus(cues, speed, confidence, fps):
    """Lower the cost of live→dead when an onset agrees with other evidence."""
    whistle, arm = cues.T
    motion = np.isfinite(speed) & (confidence >= 0.5) & (speed > 2.5)
    slow = np.isfinite(speed) & (confidence >= 0.5) & (speed <= 2.5)
    bonus = np.zeros(len(speed))
    for frame in np.flatnonzero(whistle | arm):
        left, right = max(0, frame - fps), min(len(speed), frame + fps)
        followed_by_slowdown = motion[left:frame].sum() >= fps / 4 and slow[frame:right].sum() >= fps / 4
        multimodal = whistle[left:right].any() and arm[left:right].any()
        # No cue forces a label, and a whistle may also signal a restart.
        if followed_by_slowdown or multimodal:
            bonus[frame] = 0.65 if multimodal and followed_by_slowdown else 0.45
    return bonus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--pose-model", type=Path, help="optional local COCO pose checkpoint")
    parser.add_argument("--referee-model", type=Path, help="optional local YOLOE-26 checkpoint for referee crops")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--range", type=int, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.referee_model is not None and args.pose_model is None:
        parser.error("--referee-model requires --pose-model")
    first, stop = args.range or (0, source_frames())
    if not 0 <= first < stop <= source_frames():
        parser.error("--range must be a valid half-open frame interval")
    blocks = [(max(first, a), min(stop, b + 1)) for _, a, b in half_video_blocks(GAME)
              if max(first, a) < min(stop, b + 1)]
    fps = GAMES[GAME]["fps"]
    try:
        video, = require(["video"], data_dir=args.data_dir)
        events = [(int(i), "whistle") for a, b in blocks for i in extract_whistles(video, a, b, fps)]
        print(f"Whistle candidates: {len(events)}", flush=True)
        if args.pose_model is not None:
            arms = extract_referee_arms(video, args.pose_model, args.device, blocks, fps, args.data_dir,
                                        args.referee_model)
            events.extend((i, "referee_arm") for i in arms)
            print(f"Raised-arm candidates: {len(arms)}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", dir=args.output.parent, newline="", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                with handle:
                    rows = csv.writer(handle)
                    rows.writerow(("frame", "cue"))
                    rows.writerows(sorted(events))
                temporary.replace(args.output)
            finally:
                temporary.unlink(missing_ok=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
