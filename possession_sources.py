"""External tracking sources with partial frame supervision."""

import csv
import json
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from evaluation import possession_metrics as metrics


LEGACY_REVISION = "1b99d401ddc82e366c5c20d6860878668fd5858f"
LEGACY_MATCHES = (2068, 2269, 2417, 2440, 2841, 3442, 3518, 3749, 4039)
METRICA_MATCHES = (1, 2, 3)
METRICA_FPS = 25


def load_legacy_skillcorner(root, match_id):
    def read(filename):
        source = f"{LEGACY_REVISION}:data/matches/{match_id}/{filename}.json"
        return json.loads(subprocess.check_output(["git", "-C", str(root), "show", source]))

    meta = read("match_data")
    rows = read("structured_data")
    pitch = np.array([meta["pitch_length"], meta["pitch_width"]], dtype=np.float32)
    side_by_id = {meta[f"{side}_team"]["id"]: team for side, team in (("home", 0), ("away", 1))}
    players = [p for p in meta["players"] if p["team_id"] in side_by_id]
    teams = np.array([side_by_id[p["team_id"]] for p in players])
    slots = {p["trackable_object"]: index for index, p in enumerate(players)}
    slots[meta["ball"]["trackable_object"]] = len(teams)
    groups = {"home team": metrics.H, "away team": metrics.A, None: metrics.X}
    for period in (1, 2):
        half = [row for row in rows if row["period"] == period]
        start, stop = min(row["frame"] for row in half), max(row["frame"] for row in half) + 1
        positions = np.full((stop - start, len(teams) + 1, 2), np.nan, dtype=np.float32)
        labels = np.full(stop - start, metrics.X, dtype=np.int8)
        for row in half:
            frame = row["frame"] - start
            labels[frame] = groups[row["possession"]["group"]]
            for obj in row["data"]:
                slot = slots.get(obj.get("trackable_object"))
                if slot is not None:
                    positions[frame, slot] = (obj["x"], obj["y"])
        weights = np.isfinite(positions[:, -1]).all(axis=1).astype(np.float32)
        yield positions + pitch / 2, weights, teams, pitch, labels


def metrica_events(folder, stem, fifa):
    if not fifa:
        with (folder / f"{stem}_RawEventsData.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                yield (int(row["Period"]), int(row["Start Frame"]), int(row["End Frame"]),
                       {"Home": metrics.H, "Away": metrics.A}[row["Team"]],
                       row["Type"], row["Subtype"])
        return
    path = folder / f"{stem}_events.json"
    for event in json.loads(path.read_text())["data"]:
        subtypes = event.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        yield (event["period"], event["start"]["frame"], event["end"]["frame"],
               {"FIFATMA": metrics.H, "FIFATMB": metrics.A}[event["team"]["id"]],
               event["type"]["name"], " ".join(s["name"] for s in subtypes))


def metrica_fifa_tracks(folder, stem):
    metadata = ET.parse(folder / f"{stem}_metadata.xml").getroot()
    pitch_element = metadata.find(".//FieldSize")
    pitch = np.array([float(pitch_element.findtext(axis)) for axis in ("Width", "Height")],
                     dtype=np.float32)
    channels = {c.attrib["id"]: c.attrib["playerId"]
                for c in metadata.findall(".//PlayerChannel")}
    player_teams = {p.attrib["id"]: {"FIFATMA": 0, "FIFATMB": 1}[p.attrib["teamId"]]
                    for p in metadata.findall(".//Players/Player")}
    slots = {player: idx for idx, player in enumerate(player_teams)}
    formats = metadata.findall(".//DataFormatSpecification")
    teams = np.array(list(player_teams.values()))
    final_frame = int(formats[-1].attrib["endFrame"])
    positions = np.full((final_frame, len(teams) + 1, 2), np.nan, dtype=np.float32)
    with (folder / f"{stem}_tracking.txt").open() as handle:
        for spec in formats:
            first, last = (int(spec.attrib[name]) for name in ("startFrame", "endFrame"))
            player_slots = [slots[channels[xy[0].attrib["playerChannelId"]]] for xy in spec[1]]
            for expected in range(first, last + 1):
                frame, coords, ball = next(handle).strip().split(":")
                if int(frame) != expected:
                    raise ValueError(f"Unexpected Metrica frame {frame}, expected {expected}")
                xy = np.fromstring(coords.replace(";", ","), dtype=np.float32, sep=",")
                positions[expected - 1, player_slots] = xy.reshape(-1, 2)
                positions[expected - 1, -1] = np.fromstring(ball, dtype=np.float32, sep=",")
    half_end = int(formats[0].attrib["endFrame"])
    periods = np.repeat([1, 2], [half_end, final_frame - half_end])
    return np.arange(1, final_frame + 1), periods, positions, teams, pitch


def metrica_csv_tracks(folder, stem):
    home, away = (np.genfromtxt(folder / f"{stem}_RawTrackingData_{side}_Team.csv",
                               delimiter=",", skip_header=3, dtype=np.float32)
                  for side in ("Home", "Away"))
    if not np.array_equal(home[:, :2], away[:, :2]):
        raise ValueError("Metrica home and away frame grids differ")
    teams = np.repeat([0, 1], [(len(track[0]) - 5) // 2 for track in (home, away)])
    positions = np.concatenate((home[:, 3:-2], away[:, 3:]), axis=1)
    return (home[:, 1].astype(int), home[:, 0].astype(int),
            positions.reshape(len(home), -1, 2), teams, np.array([105, 68], dtype=np.float32))


def metrica_labels(start, stop, events):
    labels = np.full(stop - start, metrics.X, dtype=np.int8)
    events = sorted(events, key=lambda event: event[1])
    for _, left, right, team, kind, subtype in events:
        if kind == "PASS" and not subtype and start <= left < right < stop:
            labels[left - start:right - start] = team
    for index, (_, frame, _, team, kind, subtype) in enumerate(events):
        foul = kind == "FAULT RECEIVED"
        offside = "OFFSIDE" in subtype
        if not (foul or offside):
            continue
        for _, restart, _, side, event, detail in events[index + 1:]:
            if restart <= frame:
                continue
            if restart - frame > 90 * METRICA_FPS or event in ("FAULT RECEIVED", "BALL OUT"):
                break
            if event == "SET PIECE":
                recipient = team if foul else 1 - team
                if "FREE KICK" in detail and side == recipient and start <= frame < restart < stop:
                    labels[frame + 5 - start:restart - 5 - start] = metrics.D
                break
            if event in ("PASS", "CARRY", "RECOVERY", "SHOT") and restart > frame + METRICA_FPS // 5:
                break
    return labels


def load_metrica(root, match_id):
    stem = f"Sample_Game_{match_id}"
    folder = root / "data" / stem
    fifa = match_id == 3
    load = metrica_fifa_tracks if fifa else metrica_csv_tracks
    frames, periods, normalized, teams, pitch = load(folder, stem)
    events = list(metrica_events(folder, stem, fifa))
    for period in (1, 2):
        select = periods == period
        frame = frames[select]
        start, stop = frame.min(), frame.max() + 1
        if not np.array_equal(frame, np.arange(start, stop)):
            raise ValueError(f"Non-contiguous Metrica grid in period {period}")
        positions = normalized[select] * pitch
        labels = metrica_labels(start, stop, [event for event in events if event[0] == period])
        weights = np.isfinite(positions[:, -1]).all(axis=1).astype(np.float32)
        yield positions, weights, teams, pitch, labels
