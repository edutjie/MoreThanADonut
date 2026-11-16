# Tactical possession baseline

## Current experiment baseline

Use the frozen [GRU with continuous player speeds](docs/experiments/PLAYER_TRACK_QUALITY.md)
for new possession comparisons. It scored 0.6897 spellF1 and 0.9002 macroF1
in the automated public window check after selection on external matches.
The checkpoint stores the same player-coordinate and speed handling used
during training.

```bash
work=data/external_possession/player-quality
.venv/bin/python sequence_possession.py predict \
  --model "$work/continuous/gru.pt" --device cuda \
  --output "$work/final/candidate/possession.csv"
.venv/bin/python evaluation/validate_submission.py "$work/final/candidate/possession.csv"
```

## Tactical tracking and Smart Tags reference

The baseline reads the tactical automatic tracking export and, if present, the
automatic Smart Tagging XML. Unpack the official data bundle under `data/` or
pass its root with `--data-dir`. The required files are
`tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt`.
The optional tagging file is `tactical/BAR-ATH_tactical_smart_tagging.xml`.

From the repo root with Python 3.11:

```bash
uv venv --python 3.11
uv pip install -r requirements-baseline.txt
.venv/bin/python tactical_baseline.py --output submissions/TEAM_NAME.csv
.venv/bin/python evaluation/validate_submission.py submissions/TEAM_NAME.csv
```

Use `--tracking-only` to run without Smart Tags. To check the prediction
pipeline with the public five-minute example, generate the CSV at the
default path and run `.venv/bin/python evaluation/run_evaluation.py`.
Keep generated CSVs in `submissions/` or at the default path, both ignored by Git.

## Method

Tracking coordinates are converted to metres. Ball gaps up to half a second
are interpolated between plausible endpoints and receive weaker weight than
observed positions. Relative player proximity supplies evidence of team
control. If players from both teams are close, the decoder has less reason
to switch. Ball observations outside the pitch supply dead-ball evidence.

Attacking Smart Tags are weak team cues. For set pieces, a sustained change
from a slow ball to a moving ball within the tagged interval provides a
possible restart and preceding stoppage. The decoder scores the full
sequence within each graded half using Viterbi, with penalties for team
turnovers and changes between live and dead states. Long occlusions supply
no tracking evidence, allowing the state to persist. Frames outside the
published graded ranges are written as `X`.

The transition costs and evidence weights are heuristics. The generator
reads no labels and uses no training samples. Long occlusions can conceal a
turnover or stoppage. The restart heuristic can miss restarts with poor
tracking and may miss fouls or other stoppages in the middle of the pitch.
The model uses the tactical tracking and tag intervals directly and does
not need video decoding, homography, or broadcast synchronization.

## Optional ball recovery from video

Download `yolo-sn-ball-opt.pt` from the authors' [SoccerNet-v3D weight list](https://github.com/mguti97/SoccerNet-v3D#weights)
and put it under `models/`. This mode also requires the tactical MP4 and
homography CSV in the same bundle as the tracking export.

```bash
uv pip install --python .venv/bin/python -r requirements-video.txt
.venv/bin/python tactical_baseline.py \
  --data-dir "data/More Than A Hack 2027 - Initial Challenge - Dataset" \
  --ball-detector models/yolo-sn-ball-opt.pt \
  --output submissions/TEAM_NAME.csv
.venv/bin/python evaluation/validate_submission.py submissions/TEAM_NAME.csv
```

Use `--device` to select an accelerator supported by the local installation.
On Apple Silicon, use `--device mps` from a regular Terminal. Check access with
`.venv/bin/python -c 'import torch; print(torch.backends.mps.is_available())'`.
The Codex command sandbox on this laptop blocks Metal. With the same Python
environment outside that sandbox, PyTorch can use the GPU.
For a quick public-window pipeline check, use `--detector-range 11185 18685`.
That option only enables video recovery in that window. Omit it for
full-match recovery.

To inspect the accepted detections in the [local dashboard](dashboard/README.md),
add `--recovery-output submissions/atd_tags_video_recovery.csv` to the
detector run. The CSV stores zero-based frame indices and normalized
pitch coordinates for detections that pass the recovery filter.
`--interpolation-output submissions/interpolated.csv` writes the additional
ball positions estimated between neighboring observations. The interpolation
already runs before possession decoding. It bridges up to 0.5 seconds when
the endpoints imply a plausible speed, and gives these estimates less weight
than direct detections. Long, unbounded, and implausible gaps stay unresolved.

The frozen detector runs locally on frames with missing ball coordinates.
The inverse homography projects the highest-confidence box center onto the
pitch plane. Candidates must fall within three metres of the pitch bounds,
have a plausible neighboring observation, and agree with nearby ATD anchors
when available. Accepted video locations contribute at 55% of the weight of
an original tracking observation. The existing interpolation, Smart Tags and
Viterbi decoder then run as usual. Projection can be inaccurate for aerial
balls, and a persistent false detection can still pass the temporal checks.
The detector and filter settings are fixed before the automatic public check.

For a frozen general-purpose comparison, download `yolo26s.pt` or `yolo26x.pt` from the
[official Ultralytics release](https://github.com/ultralytics/assets/releases/tag/v8.4.0)
and pass the local checkpoint with `--ball-detector`. The same recovery
pipeline filters these checkpoints to the COCO `sports ball` class (32) and
keeps the original confidence, geometry, and motion checks.

## Optional referee stoppage cues

Use FFmpeg and the video dependencies. Download `yolo26n-pose.pt` from the
[official Ultralytics release](https://github.com/ultralytics/assets/releases/tag/v8.4.0)
and put it under `models/`. Extract cue onsets locally, then pass the CSV to
the possession decoder:

```bash
.venv/bin/python dead_ball.py --data-dir "$BUNDLE" \
  --pose-model models/yolo26n-pose.pt --device mps \
  --output submissions/stoppage_cues.csv

.venv/bin/python tactical_baseline.py --data-dir "$BUNDLE" \
  --ball-detector models/yolo-sn-ball-opt.pt --device mps \
  --stoppage-cues submissions/stoppage_cues.csv \
  --output submissions/with_stoppage_cues.csv
```

Set `BUNDLE` to the local data folder and use `--device cpu` on machines
without MPS. Omit `--pose-model` for audio-only extraction. Use
`--range 11185 18685` on the extractor for the public pipeline check.

To localize the official before pose estimation, install the
[Ultralytics CLIP tokenizer](https://github.com/ultralytics/CLIP):

```bash
uv pip install --python .venv/bin/python \
  "git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40"
```

Download `yoloe-26s-seg.pt` and `mobileclip2_b.ts` from the
[Ultralytics model assets](https://github.com/ultralytics/assets/releases/tag/v8.4.0)
into `models/`. Add `--referee-model models/yoloe-26s-seg.pt` alongside
`--pose-model`. This mode uses the fixed text prompt `soccer referee`,
filters boxes by pitch location and player proximity, and runs pose on
padded crops around the remaining candidates.

- **Referee arm.** At 5 fps, frozen COCO pose estimates shoulders, elbows,
  wrists, and ankles. The shirt-color filter looks for yellow in the torso.
  Foot positions must project inside the pitch and away from tracked players.
  A raised arm must persist on a nearby candidate across two samples.
- **Whistle.** A short, narrow-band burst at 2–5 kHz must rise above the
  preceding local noise floor. This can reject sustained tones. Brief crowd
  whistles can still look like a referee whistle.
- **Decoder.** A confirmed cue reduces the cost of entering `D` when
  both modalities agree within a second, or when observed ball motion
  changes from moving to slow. Existing restart and pitch-exit evidence
  still determine whether the surrounding sequence supports a stoppage.

These are fixed prototype heuristics. A raised arm can also signal advantage
or a restart. Wind, crowd noise, and distant or occluded officials can hide
events. Use external match data for gesture and whistle accuracy validation.
