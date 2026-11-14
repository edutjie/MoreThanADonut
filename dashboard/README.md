# Tactical timeline

From the repository root, install the [baseline dependencies](../TACTICAL_BASELINE.md)
and generate a prediction if needed. Replace the bundle path with your local folder and run:

```bash
.venv/bin/python tactical_baseline.py \
  --data-dir "data/More Than A Hack 2027 - Initial Challenge - Dataset" \
  --output submissions/atd_tags.csv
.venv/bin/python -m dashboard.serve \
  --prediction "Tracking + Smart Tags" submissions/atd_tags.csv
```

Open [the dashboard](http://127.0.0.1:8765) in a browser with H.264 support.
Use `--data-dir /path/to/bundle` if the bundle is stored elsewhere. Repeat
`--prediction "Track name" /path/to/predictions.csv` to add other runs.
To compare accepted recovery frames and their projected coordinates, add one
`--recovery-track "Name" path.csv` per checkpoint:

```bash
.venv/bin/python -m dashboard.serve \
  --prediction "Tracking + Smart Tags" submissions/atd_tags.csv \
  --prediction "SoccerNet possession" submissions/atd_tags_video.csv \
  --prediction "YOLO26x possession" submissions/yolo26x_mps_public.csv \
  --recovery-track "yolo-sn-ball-opt" submissions/atd_tags_video_recovery.csv \
  --recovery-track "YOLO26x" submissions/yolo26x_mps_recovery.csv
```

Generate the recovery CSVs with `--recovery-output` when running the
[video-assisted baseline](../TACTICAL_BASELINE.md). Each recovery track
uses its own color and overlay toggle. Orange marks still show where the
original tracking was missing.
The server binds to localhost and uses the original MP4 with byte-range
seeking. Stop it with Ctrl+C.

- Drag on the timeline to scrub. Use **Space** to play or pause, **←** and
  **→** to step one frame, or type a zero-based video frame number.
- **Public 5 min** jumps to the released ground-truth window. Ground-truth
  regions outside that window are blank. The full view includes pre-match,
  halftime and post-match footage, shaded outside the match halves.
- The dots are raw ATD positions projected into video pixels using each
  frame's tactical homography. A track slot is a tracklet, so a dot does
  not represent a persistent player ID.
- Ball gaps mean the source has no finite ball position. Player-gap tracks
  use the shortfall from 11 observed positions per team during the match
  halves. This marks missing observations, which can also reflect a player
  being off the pitch.
- Use the viewer for data and playback alignment checks. Competition
  predictions remain algorithmic and the public labels are for pipeline
  checks only.
