"""Serve a read-only tactical timeline on localhost."""
import argparse
import csv
import json
import re
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import groupby
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from broadcast_alignment import load_frame_pairs
from games import DATA_DIR, GAME, GAMES, asset_paths, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track
from starter.load_homography import check_alignment, load, project
from tactical_baseline import read_tracking

ROOT = Path(__file__).resolve().parent.parent
CHUNK = 256
RECOVERY_COLORS = ("#8fe4ab", "#6faeff", "#df9df2", "#f6c46e")
BROADCAST_MAP = ROOT / "submissions/frozen-broadcast/frame-map.npz"
FROZEN_PREDICTION = ROOT / "data/external_possession/skillcorner-frozen/possession.csv"
CURRENT_BASELINE = ROOT / "data/external_possession/player-quality/final/candidate/possession.csv"
CURRENT_BASELINE_NAME = "Current baseline · GRU with continuous player speeds · 0.6897 public spellF1"
TCN_PREDICTION = ROOT / "data/external_possession/tcn-study/public/autotcn.csv"
TCN_PREDICTION_NAME = "TCN with the frozen GRU’s features · 0.6815 public spellF1"
SKILLCORNER_SAMPLE = ROOT / "data/external_possession/skillcorner-sample.mp4"


def bundle_path():
    root = Path(DATA_DIR)
    if (root / "tactical").is_dir() or not root.is_dir():
        return root
    candidates = sorted(folder for folder in root.iterdir() if (folder / "tactical").is_dir())
    return candidates[0] if len(candidates) == 1 else root


def load_labels(path, n_frames):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != ["frame", "label"]:
            raise ValueError(f"{path}: expected frame,label header")
        labels = []
        for frame, row in enumerate(rows):
            if row["frame"] != str(frame) or row["label"] not in {"H", "A", "D", "X"} or None in row:
                raise ValueError(f"{path}: invalid label or index on row {frame + 2}")
            labels.append(row["label"])
    if len(labels) != n_frames:
        raise ValueError(f"{path}: expected {n_frames} label rows, found {len(labels)}")
    return labels


def runs(values, omitted):
    start = 0
    result = []
    for value, group in groupby(values):
        stop = start + sum(1 for _ in group)
        if value != omitted:
            result.append([start, stop, value])
        start = stop
    return result


def recovery_pixels(path, observed_ball, in_play, matrices, valid):
    pixels = np.full((len(observed_ball), 2), np.nan, dtype="<f4")
    recovered = np.zeros(len(observed_ball), dtype=bool)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != ["frame", "x", "y"]:
            raise ValueError(f"{path}: expected frame,x,y header")
        for number, row in enumerate(rows, start=2):
            try:
                frame = int(row["frame"])
                x, y = float(row["x"]), float(row["y"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}: invalid recovery on row {number}") from None
            if (not 0 <= frame < len(pixels) or observed_ball[frame] or not in_play[frame]
                    or not valid[frame] or recovered[frame] or not np.isfinite([x, y]).all()):
                raise ValueError(f"{path}: invalid or duplicate recovered frame {frame}")
            with np.errstate(divide="ignore", invalid="ignore"):
                point = project(matrices[frame], x, y)
            if not np.isfinite(point).all():
                raise ValueError(f"{path}: cannot project recovered frame {frame}")
            pixels[frame] = point
            recovered[frame] = True
    return pixels, recovered


def clear_held_player_edges(positions, pixels, fps):
    """Hide long, exact coordinate holds at the edges of the player feed."""
    if len(positions) < 5 * fps:
        return
    for slot in range(positions.shape[1] - 1):  # Ball is the final slot.
        track, overlay = positions[:, slot], pixels[:, slot]
        for points, dots in ((track, overlay), (track[::-1], overlay[::-1])):
            if not np.isfinite(points[0]).all():
                continue
            changes = np.flatnonzero(np.any(points != points[0], axis=1))
            held = changes[0] if changes.size else len(points)
            if held >= 5 * fps:
                dots[:held] = np.nan


def tracking_pixels(data_dir, source):
    meta, raw = require(["atd_meta", "atd_raw"], data_dir=data_dir, source=source)
    n_frames = source_frames(source)
    teams = team_of_each_track(meta)
    pitch = np.asarray(GAMES[GAME]["pitch"], dtype=np.float32)
    normalized = read_tracking(raw, n_frames, len(teams), pitch) / pitch
    observed = np.isfinite(normalized).all(axis=-1)
    if not check_alignment(data_dir=data_dir, source=source)["agrees"]:
        raise ValueError(f"Homography rows do not match {source} tracking frames")
    matrices, _, valid = load(data_dir=data_dir, source=source)
    pixels = np.full(normalized.shape, np.nan, dtype="<f4")
    for start in range(0, n_frames, 4096):
        span = slice(start, start + 4096)
        points = normalized[span]
        with np.errstate(invalid="ignore", divide="ignore"):
            x, y = project(matrices[span].transpose(1, 2, 0)[..., None],
                           points[..., 0], points[..., 1])
        pixels[span] = np.stack((x, y), axis=-1)
    pixels[~(valid[:, None] & observed)] = np.nan
    clear_held_player_edges(normalized, pixels, GAMES[GAME]["fps"])
    return teams, observed, matrices, valid, pixels


def match_data(data_dir, predictions, recoveries=()):
    video, = require(["video"], data_dir=data_dir)
    n_frames = source_frames()
    teams, observed, matrices, valid, pixels = tracking_pixels(data_dir, "tactical")
    in_play = np.zeros(n_frames, dtype=bool)
    halves = [[first, last + 1] for _, first, last in half_video_blocks(GAME)]
    for first, stop in halves:
        in_play[first:stop] = True

    gt = ROOT / "evaluation" / "gt" / f"{GAME}_possession_gt_public.csv"
    tracks = [
        {"name": "Ball gaps", "kind": "gap", "spans": runs(np.where(in_play & ~observed[:, -1], 1, 0).tolist(), 0)},
    ]
    for team, name in enumerate(("Barcelona gaps", "Athletic gaps")):
        deficit = np.maximum(0, 11 - observed[:, :-1][:, teams == team].sum(axis=1))
        tracks.append({"name": name, "kind": "gap", "spans": runs(np.where(in_play, deficit, 0).tolist(), 0)})
    recovery_tracks = []
    extra_positions = []
    for index, (name, path) in enumerate(recoveries):
        positions, recovered = recovery_pixels(path, observed[:, -1], in_play, matrices, valid)
        extra_positions.append(positions[:, None, :])
        recovery_tracks.append({"name": name, "kind": "recovery",
                                "color": RECOVERY_COLORS[index % len(RECOVERY_COLORS)],
                                "spans": runs(recovered.tolist(), False)})
    if recovery_tracks:
        pixels = np.concatenate((pixels, *extra_positions), axis=1)
        tracks[1:1] = recovery_tracks
    tracks.append({"name": "Public ground truth", "kind": "gt", "spans": runs(load_labels(gt, n_frames), "X")})
    tracks.extend({"name": name, "kind": "prediction", "spans": runs(labels, "X")} for name, labels in predictions)
    info = {"frames": n_frames, "fps": GAMES[GAME]["fps"], "chunk": CHUNK,
            "teams": teams.tolist(), "halves": halves, "tracks": tracks}
    return info, pixels, Path(video)


def broadcast_data(path, info):
    # The supplied MP4 ends one frame before the broadcast tracking feed.
    frames = source_frames("broadcast") - 1
    broadcast, tactical = load_frame_pairs(path, info["frames"], frames)
    cuts = np.flatnonzero((np.diff(broadcast) != 1) | (np.diff(tactical) != 1)) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts - 1, len(tactical) - 1]
    # Runs pair equally long, consecutive spans on each native frame axis.
    pairs = np.stack((tactical[starts], tactical[ends] + 1, broadcast[starts]), axis=1).tolist()
    tracks = []
    for track in info["tracks"]:
        if track["kind"] not in {"gt", "prediction"}:
            continue
        labels = np.full(info["frames"], "X", dtype="U1")
        for start, stop, label in track["spans"]:
            labels[start:stop] = label
        mapped = np.full(frames, "X", dtype="U1")
        mapped[broadcast] = labels[tactical]
        tracks.append({**track, "spans": runs(mapped.tolist(), "X")})
    return {"frames": frames, "fps": info["fps"], "tracks": tracks, "pairs": pairs,
            "mappedFrames": len(tactical)}


def handler_for(info, pixels, video, broadcast=None, broadcast_video=None, broadcast_pixels=None,
                skillcorner_video=None):
    manifest = json.dumps(info, separators=(",", ":")).encode()
    broadcast_manifest = json.dumps(broadcast, separators=(",", ":")).encode()
    static = {
        "/": ("text/html; charset=utf-8", "index.html"),
        "/broadcast": ("text/html; charset=utf-8", "index.html"),
        "/aligned": ("text/html; charset=utf-8", "index.html"),
        "/skillcorner": ("text/html; charset=utf-8", "skillcorner.html"),
        "/style.css": ("text/css; charset=utf-8", "style.css"),
        "/app.js": ("text/javascript; charset=utf-8", "app.js"),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

        def respond(self, send_body):
            if self.headers.get("Host") not in {f"127.0.0.1:{self.server.server_port}",
                                                f"localhost:{self.server.server_port}"}:
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            try:
                if path == "/video.mp4":
                    self.stream_video(video, send_body)
                    return
                if path == "/broadcast.mp4" and broadcast_video:
                    self.stream_video(broadcast_video, send_body)
                    return
                if path == "/skillcorner.mp4" and skillcorner_video and skillcorner_video.is_file():
                    self.stream_video(skillcorner_video, send_body)
                    return
                if path == "/match.json":
                    mime, body = "application/json", manifest
                elif path == "/broadcast.json" and broadcast is not None:
                    mime, body = "application/json", broadcast_manifest
                elif path in static:
                    mime, filename = static[path]
                    body = (Path(__file__).parent / filename).read_bytes()
                elif match := re.fullmatch(r"/(positions|broadcast-positions)/([0-9]{1,8})", path):
                    positions = pixels if match[1] == "positions" else broadcast_pixels
                    start = int(match[2]) * CHUNK
                    if positions is None or start >= len(positions):
                        self.send_error(404)
                        return
                    mime = "application/octet-stream"
                    body = positions[start:start + CHUNK].tobytes()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if send_body:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Scrubbing often cancels an in-progress video range.

        def stream_video(self, path, send_body):
            with path.open("rb") as source:
                size = path.stat().st_size
                start, stop = 0, size
                request_range = self.headers.get("Range")
                if request_range:
                    match = re.fullmatch(r"bytes=([0-9]{0,20})-([0-9]{0,20})", request_range)
                    if match and match[1]:
                        start = int(match[1])
                        stop = min(size, int(match[2]) + 1) if match[2] else size
                    elif match and match[2]:
                        start = max(0, size - int(match[2]))
                    else:
                        start = stop
                    if start >= stop:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return

                self.send_response(206 if request_range else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(stop - start))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                if request_range:
                    self.send_header("Content-Range", f"bytes {start}-{stop - 1}/{size}")
                self.end_headers()
                if not send_body:
                    return
                if not request_range:
                    shutil.copyfileobj(source, self.wfile, 256 * 1024)
                    return
                source.seek(start)
                remaining = stop - start
                while remaining:
                    block = source.read(min(remaining, 256 * 1024))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=bundle_path(), help="root of tactical data bundle")
    parser.add_argument("--prediction", action="append", nargs=2, metavar=("NAME", "CSV"),
                        help="prediction track to display, repeat for multiple CSVs")
    parser.add_argument("--recovery", type=Path, help="accepted YOLO positions from --recovery-output")
    parser.add_argument("--recovery-track", action="append", nargs=2, metavar=("NAME", "CSV"), default=[],
                        help="named ball-recovery track, repeat to compare checkpoints")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--broadcast-map", type=Path, default=BROADCAST_MAP,
                        help="broadcast-to-tactical NPZ from broadcast_alignment.py")
    parser.add_argument("--skillcorner-sample", type=Path, default=SKILLCORNER_SAMPLE,
                        help="public tracking replay from dashboard.render_skillcorner")
    args = parser.parse_args()
    if args.prediction:
        files = args.prediction
    else:
        files = []
        if CURRENT_BASELINE.is_file():
            files.append((CURRENT_BASELINE_NAME, CURRENT_BASELINE))
        elif FROZEN_PREDICTION.is_file():
            files.append(("SkillCorner model · 0.6250 public spellF1", FROZEN_PREDICTION))
        if TCN_PREDICTION.is_file():
            files.append((TCN_PREDICTION_NAME, TCN_PREDICTION))
        if not files:
            files.append(("Tactical baseline", ROOT / "evaluation/results/tactical_decoder.csv"))
    try:
        predictions = [(name, load_labels(Path(path), source_frames())) for name, path in files]
        print("Loading tracking and timeline…", flush=True)
        recoveries = [(name, Path(path)) for name, path in args.recovery_track]
        if args.recovery is not None:
            recoveries.insert(0, ("YOLO ball recovery", args.recovery))
        info, pixels, video = match_data(args.data_dir, predictions, recoveries)
        broadcast_video = Path(asset_paths(data_dir=args.data_dir, source="broadcast")["video"])
        broadcast = None
        broadcast_pixels = None
        if broadcast_video.is_file() and args.broadcast_map.is_file():
            try:
                broadcast = broadcast_data(args.broadcast_map, info)
            except (OSError, KeyError, ValueError) as error:
                print(f"Broadcast views unavailable: {error}", flush=True)
        if broadcast is not None:
            try:
                print("Loading broadcast tracking…", flush=True)
                teams, observed, _, _, positions = tracking_pixels(args.data_dir, "broadcast")
                broadcast_pixels = positions[:broadcast["frames"]]
                broadcast.update(teams=teams.tolist(), chunk=CHUNK)
                ball = np.where(observed[:broadcast["frames"], -1], "Present", "Missing")
                ball_track = {"name": "Broadcast ball", "kind": "ball-availability"}
                broadcast["tracks"].insert(0, {**ball_track, "spans": runs(ball.tolist(), "")})
                aligned_ball = np.full(info["frames"], "", dtype=ball.dtype)
                for start, stop, native_start in broadcast["pairs"]:
                    aligned_ball[start:stop] = ball[native_start:native_start + stop - start]
                broadcast["alignedBall"] = {**ball_track, "spans": runs(aligned_ball.tolist(), "")}
            except (OSError, ValueError) as error:
                print(f"Broadcast overlays unavailable: {error}", flush=True)
        with ThreadingHTTPServer(("127.0.0.1", args.port),
                                 handler_for(info, pixels, video, broadcast, broadcast_video,
                                             broadcast_pixels, args.skillcorner_sample)) as server:
            print(f"Tactical dashboard → http://127.0.0.1:{server.server_port}", flush=True)
            server.serve_forever()
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\nGenerate a baseline CSV first or use --prediction NAME CSV.\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
