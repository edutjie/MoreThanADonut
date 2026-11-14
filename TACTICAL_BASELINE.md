# Tactical possession baseline

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
