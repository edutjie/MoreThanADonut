"""Render one held-out SkillCorner tracking sample with source and model labels."""

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from external_possession import (
    SKILLCORNER_FPS,
    TEST_MATCHES,
    decode_costs,
    load_model,
    load_skillcorner,
    probabilities,
    tracking_features,
)


SIZE = (1280, 720)
COLORS = {"H": "#6dd0eb", "A": "#ef8499", "D": "#b7a4e8", "X": "#a7a7a1"}
BALL_COLOR = "#f6e78e"
SAMPLE_MATCH = 1996435
SAMPLE_START = 6380


def choose_start(positions, labels, frames):
    visible = np.isfinite(positions[:, -1]).all(axis=1)
    for start in range(0, len(labels) - frames + 1, SKILLCORNER_FPS):
        clip = slice(start, start + frames)
        if set(labels[clip]) == {0, 1, 2} and visible[clip].mean() >= 0.85:
            return start
    raise ValueError("no sample covering all three possession labels with at least 85% ball coverage")


def round_two_prediction(selection_path, positions, weights, teams, pitch, fps):
    import torch

    from experiments.possession_round2 import decoded, mix, source_from
    from experiments.possession_round2_decoder import Decoder
    from experiments.possession_round2_train import load
    from player_track_quality import repair_player_tracks
    from sequence_possession import sequence_features, sequence_probabilities

    decision = json.loads(selection_path.read_text())
    if decision["decoder"]:
        raise ValueError("the round-two replay requires the frozen fixed-decoder recipe")
    source = source_from(decision["source"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    repaired = repair_player_tracks(positions, fps)
    normal, mirrored = sequence_features(repaired, weights, teams, pitch, fps, True)
    probabilities_by_model = [sequence_probabilities(load(path, device), normal, mirrored, fps, device)
                              for path in source.paths]
    emissions = mix(probabilities_by_model, source.weights, source.space)
    decoder = Decoder(selection_path.parent)
    return np.array(list("HAD"))[decoded(decoder, emissions, fps, {}, decision["duration"], None)]


def render(root, model_path, output, match_id, start_frame=None, round_two_selection=None):
    match = json.loads((root / f"{match_id}_match.json").read_text())
    home = match["home_team"]["name"]
    away = match["away_team"]["name"]
    positions, weights, teams, pitch, labels = next(load_skillcorner(root, match_id))
    fps = SKILLCORNER_FPS
    frames = 20 * fps
    if start_frame is None:
        start_frame = SAMPLE_START if match_id == SAMPLE_MATCH else choose_start(positions, labels, frames)
    start = start_frame
    if not 0 <= start <= len(labels) - frames:
        raise ValueError("the 20-second clip must fit within the first half")
    if round_two_selection is None:
        model, costs, lags = load_model(model_path)
        x, mirrored = tracking_features(positions, weights, teams, pitch, fps, lags)
        emission = np.log(np.maximum(probabilities(model, x, mirrored), 1e-6))
        prediction = decode_costs(emission, fps, *costs)
    else:
        prediction = round_two_prediction(round_two_selection, positions, weights, teams, pitch, fps)
    source_labels = "HADX"
    names = {"H": home, "A": away, "D": "Stopped play", "X": "Unlabeled"}
    font = ImageFont.truetype("DejaVuSans.ttf", 21)
    title = ImageFont.truetype("DejaVuSans-Bold.ttf", 29)

    scale = min((SIZE[0] - 140) / pitch[0], (SIZE[1] - 190) / pitch[1])
    left, top = (SIZE[0] - pitch[0] * scale) / 2, 102

    def project(x, y):
        return left + x * scale, top + (pitch[1] - y) * scale

    def rectangle(draw, x0, y0, x1, y1, **style):
        draw.rectangle((*project(x0, y1), *project(x1, y0)), **style)

    canvas = Image.new("RGB", SIZE, "#111315")
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 20), f"{home} vs {away}", font=title, fill="#f3f0e7")
    draw.text((32, 63), "SkillCorner tracking replay / A-League", font=font, fill="#b0bbb6")
    length, width = pitch
    for strip in range(10):
        rectangle(draw, length * strip / 10, 0, length * (strip + 1) / 10, width,
                  fill=("#1c4737", "#21513e")[strip % 2])
    line = "#acc4b1"
    rectangle(draw, 0, 0, length, width, outline=line, width=2)
    draw.line((*project(length / 2, 0), *project(length / 2, width)), fill=line, width=2)
    draw.ellipse((*project(length / 2 - 9.15, width / 2 + 9.15),
                  *project(length / 2 + 9.15, width / 2 - 9.15)), outline=line, width=2)
    for x0, x1 in ((0, 16.5), (length - 16.5, length)):
        rectangle(draw, x0, width / 2 - 20.16, x1, width / 2 + 20.16, outline=line, width=2)
    for x0, x1 in ((0, 5.5), (length - 5.5, length)):
        rectangle(draw, x0, width / 2 - 9.16, x1, width / 2 + 9.16, outline=line, width=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        temporary = Path(scratch) / "clip.mp4"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                   "-pixel_format", "rgb24", "-video_size", f"{SIZE[0]}x{SIZE[1]}",
                   "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            for frame in range(start, start + frames):
                image = canvas.copy()
                draw = ImageDraw.Draw(image)
                for side, point in zip(teams, positions[frame, :-1]):
                    if np.isfinite(point).all():
                        px, py = project(*point)
                        draw.ellipse((px-7, py-7, px+7, py+7), fill=COLORS[source_labels[side]],
                                     outline="#182c24", width=2)
                ball = positions[frame, -1]
                ball_status = "Ball missing"
                if np.isfinite(ball).all():
                    px, py = project(*ball)
                    detected = weights[frame] == 1
                    draw.ellipse((px-9, py-9, px+9, py+9), fill=BALL_COLOR if detected else None,
                                 outline=BALL_COLOR, width=3)
                    location = "Outside pitch" if ((ball < 0) | (ball > pitch)).any() else "Ball"
                    ball_status = f"{location} · {'detected' if detected else 'extrapolated'}"
                reference, predicted = source_labels[labels[frame]], prediction[frame]
                if reference == "D":
                    draw.rounded_rectangle((1060, 56, 1248, 96), radius=6, fill=COLORS["D"])
                    draw.text((1154, 76), "STOPPED PLAY", anchor="mm", font=font, fill="#111315")
                draw.text((32, 652), f"Source label  {names[reference]}", font=font, fill=COLORS[reference])
                draw.text((32, 682), f"Model guess  {names[predicted]}", font=font, fill=COLORS[predicted])
                minutes, seconds = divmod(frame / fps, 60)
                draw.text((1248, 652), f"1st half   {int(minutes):02d}:{seconds:04.1f}", anchor="ra",
                          font=font, fill="#f3f0e7")
                draw.text((1248, 682), ball_status, anchor="ra", font=font, fill=BALL_COLOR)
                process.stdin.write(image.tobytes())
            process.stdin.close()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        temporary.replace(output)
    print(f"Match {match_id}, first-half frames {start}–{start + frames - 1} at {fps} fps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skillcorner", type=Path, required=True)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", type=Path)
    model.add_argument("--round-two-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match", type=int, choices=sorted(TEST_MATCHES), default=SAMPLE_MATCH)
    parser.add_argument("--start-frame", type=int, help="zero-based first-half start of the 20-second clip")
    args = parser.parse_args()
    render(args.skillcorner, args.model, args.output, args.match, args.start_frame, args.round_two_selection)
