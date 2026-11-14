"""Recover missing tactical ball observations with a frozen, local detector."""

import cv2
import numpy as np
from ultralytics import YOLO

from games import GAME, half_video_blocks, require
from starter.load_homography import check_alignment, load, unproject

MAX_SPEED = 35.0  # m/s, with slack for projection error when checking displacement
POSITION_SLACK = 0.75  # metres
TOUCHLINE_MARGIN = 3.0  # metres
BATCH_SIZE = 8


def plausible_pairs(left, right, elapsed, fps):
    return np.linalg.norm(left - right, axis=-1) <= MAX_SPEED * elapsed / fps + POSITION_SLACK


def accept_candidates(ball, candidates, fps):
    """Require immediate corroboration and consistency with nearby ATD anchors."""
    combined = np.where(np.isfinite(candidates), candidates, ball)
    adjacent = plausible_pairs(combined[:-1], combined[1:], 1, fps)
    supported = np.r_[False, adjacent] | np.r_[adjacent, False]
    observed = np.flatnonzero(np.isfinite(ball).all(axis=1))
    for frame in np.flatnonzero(supported & np.isfinite(candidates).all(axis=1)):
        insertion = observed.searchsorted(frame)
        neighbors = observed[max(0, insertion - 1):insertion + 1]
        nearby = neighbors[np.abs(neighbors - frame) <= round(0.2 * fps)]
        supported[frame] = plausible_pairs(ball[nearby], candidates[frame],
                                           np.abs(nearby - frame), fps).all()
    return supported & np.isfinite(candidates).all(axis=1)


def detect_candidates(cap, model, frames, matrices, pitch, device, start, stop):
    """Decode sequentially and run the detector on eligible frames in batches."""
    result = np.full((stop - start, 2), np.nan, dtype=np.float32)
    if not len(frames):
        return result
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0])):
        raise ValueError("could not seek to the requested tactical video frame")
    next_frame = int(frames[0])
    for batch_start in range(0, len(frames), BATCH_SIZE):
        indices = frames[batch_start:batch_start + BATCH_SIZE]
        images = []
        for target in indices:
            while next_frame < target:
                if not cap.grab():
                    raise ValueError(f"could not decode tactical video frame {next_frame}")
                next_frame += 1
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"could not read tactical video frame {target}")
            next_frame += 1
            images.append(image)
        predictions = model.predict(images, imgsz=1280, conf=0.25, device=device, verbose=False)
        for frame, prediction in zip(indices, predictions, strict=True):
            boxes = prediction.boxes
            if not len(boxes):
                continue
            pixel_x, pixel_y = boxes.xywh[boxes.conf.argmax(), :2].cpu().numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                point = np.asarray(unproject(matrices[frame], pixel_x, pixel_y)) * pitch
            if np.all((-TOUCHLINE_MARGIN <= point) & (point <= pitch + TOUCHLINE_MARGIN)):
                result[frame - start] = point
        if batch_start and batch_start % 512 == 0:
            print(f"Detector frames processed: {batch_start:,}/{len(frames):,} in this half",
                  flush=True)
    return result


def recover_ball(ball, pitch, fps, data_dir, checkpoint, device, frame_range=None):
    """Fill missing ball positions in place and return a mask of video observations."""
    if not checkpoint.is_file():
        raise ValueError(f"detector checkpoint not found: {checkpoint}")
    video = require(["video"], data_dir=data_dir)[0]
    if not check_alignment(data_dir=data_dir)["agrees"]:
        raise ValueError("homography rows do not match tactical frame indices")
    matrices, _, valid = load(data_dir=data_dir)
    valid &= np.isfinite(matrices).all(axis=(1, 2))
    valid[valid] = np.abs(np.linalg.det(matrices[valid])) > 1e-12
    model = YOLO(str(checkpoint), task="detect")
    if model.names != {0: "ball"}:
        raise ValueError("expected a detector with one class named ball")
    missing = ~np.isfinite(ball).all(axis=1)
    recovered = np.zeros(len(ball), dtype=bool)
    cap = cv2.VideoCapture(video)
    try:
        if not cap.isOpened() or round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(ball):
            raise ValueError("video frame count does not match tactical tracking")
        if not np.isclose(cap.get(cv2.CAP_PROP_FPS), fps):
            raise ValueError("video frame rate does not match tactical tracking")
        selected_start, selected_stop = frame_range or (0, len(ball))
        for _, first, last in half_video_blocks(GAME):
            start, stop = max(first, selected_start), min(last + 1, selected_stop)
            if start >= stop:
                continue
            region = slice(start, stop)
            frames = np.flatnonzero(missing[region] & valid[region]) + start
            print(f"Running {checkpoint.name} on {len(frames):,} eligible missing frames",
                  flush=True)
            candidates = detect_candidates(cap, model, frames, matrices, pitch, device, start, stop)
            accepted = accept_candidates(ball[region], candidates, fps)
            ball[region][accepted] = candidates[accepted]
            recovered[region] = accepted
            print(f"Projected {np.isfinite(candidates).all(axis=1).sum():,}; "
                  f"accepted {accepted.sum():,} of {missing[region].sum():,} missing frames",
                  flush=True)
    finally:
        cap.release()
    return recovered
