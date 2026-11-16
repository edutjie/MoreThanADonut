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

## SkillCorner model and tracking replay

The frozen histogram boosting model's 0.6578 spellF1 is the mean over four
held-out A-League matches. Its contest timeline uses the separate 0.6250
public-window check. When the frozen contest CSV is at
`data/external_possession/skillcorner-frozen/possession.csv`, the dashboard
loads it by default if no `--prediction` options are provided.

See [External Possession](../EXTERNAL_POSSESSION.md) to build the model and
generate the CSV at that path. Install Pillow and FFmpeg with H.264 encoding
support to render the **SkillCorner** page's 20-second tracking animation.
It shows the ball, both teams, the source labels, and the frozen model's
predictions for one held-out Sydney FC–Adelaide United sample. The ball crosses
the touchline near 0:05 and the source marks play stopped for 11.4 seconds,
before the ball returns to play.

```bash
.venv/bin/python -m dashboard.render_skillcorner \
  --skillcorner data/skillcorner-4340d27 \
  --model data/external_possession/skillcorner-4340d27.joblib \
  --output data/external_possession/skillcorner-sample.mp4

.venv/bin/python -m dashboard.serve --port 8767 \
  --prediction "v5 reference · 0.4914 public spellF1" evaluation/results/improved_dead_ball_v5.csv \
  --prediction "SkillCorner model · 0.6250 public spellF1" data/external_possession/skillcorner-frozen/possession.csv
```

The server picks up the generated MP4 at that path. Pass `--skillcorner-sample`
to use a different local copy. The SkillCorner Open Data grant covers tracking
and labels. The sample page uses an animation of those coordinates.

## Playback and overlay options

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

Use **Broadcast** to scrub the original broadcast on its own frame clock.
Its H/A/D lanes show the public tactical ground truth and loaded predictions
projected onto verified broadcast frames. Use **Aligned** to play both feeds
from the tactical timeline. The live alignment lane marks where the second
feed is available. Unmapped intervals have blank broadcast labels, and the
aligned player shows a gap message for those intervals.

The broadcast overlay uses its own ATD text, team metadata XML, and per-frame
homography CSV. The ball and team toggles apply to both feeds in Aligned view.
Dots appear only where the corresponding feed has a tracked position and a
valid homography. On the standalone Broadcast page that also includes
calibrated frames outside the alignment map.
The **Broadcast ball** lane marks raw ATD availability in yellow (present)
and orange (missing). In Aligned, it is blank wherever there is no verified
broadcast frame. A tracked position can appear in the lane even when there is
no valid homography to draw a dot.

Both pages use the official local broadcast MP4 and default to
`submissions/frozen-broadcast/frame-map.npz`. Use `--broadcast-map PATH` to
select another map. To generate a map from the supplied automatic tracking:

```bash
.venv/bin/python broadcast_alignment.py \
  --data-dir "data/More Than A Hack 2027 - Initial Challenge - Dataset" \
  --output submissions/frozen-broadcast/frame-map.npz
```

- Drag on the timeline to scrub. Use **Space** to play or pause, **←** and
  **→** to step one frame, or type a zero-based video frame number.
- **Public 5 min** jumps to the released ground-truth window. Ground-truth
  regions outside that window are blank. The full view includes pre-match,
  halftime and post-match footage, shaded outside the match halves.
- The dots use ATD positions projected through the per-frame homography.
  The overlay hides player coordinates that stay exactly fixed for at least
  five seconds from the start or end of a recording. Gap lanes still show
  raw tracking availability. A track slot can change player identity.
- Ball gaps mean the source has no finite ball position. Player-gap tracks
  use the shortfall from 11 observed positions per team during the match
  halves. This marks missing observations, which can also reflect a player
  being off the pitch.
- Use the viewer for data and playback alignment checks. Competition
  predictions remain algorithmic and the public labels are for pipeline
  checks only.
