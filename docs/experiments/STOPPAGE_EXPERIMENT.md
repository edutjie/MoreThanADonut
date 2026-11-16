# In-pitch free-kick stoppage experiment

The frozen lagged LightGBM is the possession reference. We tested a soft boost
to D just before an inferred free-kick restart. One variant also required a
unique whistle candidate between one and twenty seconds before the restart.
The whistle gates the start of the interval and the change from stationary to
moving ball supplies its end. A whistle close to the inferred restart cannot
open an interval by itself. See the
[IFAB guidance on whistle use](https://www.theifab.com/laws/latest/guidelines/body-language-communication-and-whistle/).

## External benchmark

Use the seven [IDSSE Bundesliga matches](https://doi.org/10.1038/s41597-025-04505-y)
with their tracking and event XML. Ball-frame timestamps align the clocks.
The test simulates 35-second free-kick windows with up to 8 seconds of center
jitter and includes masked ball tracking. Decision timestamps supply proxy
whistles with simulated misses, jitter and background candidates. Those are
time proxies rather than recorded acoustic cues.

| Policy | Development spellF1 | Held-out spellF1 | Held-out D F1 | Held-out restart hits / cues |
| --- | ---: | ---: | ---: | ---: |
| Frozen LightGBM | 0.45718 | 0.39655 | 0.93425 | — |
| Tag, 0.24 s motion windows | 0.45718 | 0.39655 | 0.93429 | 11 / 51 |
| Tag, 0.60 s motion windows | 0.45718 | 0.39655 | 0.93428 | 12 / 44 |
| Tag + whistle proxy, 0.24 s | 0.45718 | 0.39655 | 0.93427 | 0 / 9 |
| Tag + whistle proxy, 0.60 s | 0.45718 | 0.39655 | 0.93427 | 1 / 6 |

The same four development and three held-out matches as the
[recovery experiment](BALL_RECOVERY_EXPERIMENT.md) were used. The short paired
rule won the development tie-break on D-class F1. A restart hit falls within
one second of any official free-kick execution timestamp in the half. Those
timestamps are themselves imperfect proxies for a D-to-live transition.
Across 155 usable events, the median distance to the nearest labeled D-to-live
transition was 1.36 seconds.

## Contest check

The supplied Smart Tags contain five free-kick windows of 35 seconds and no
foul or offside onset. The paired rule applied zero intervals using either the
existing tonal cues, learned tactical-audio cues or the learned cues mapped
from broadcast. All three generated predictions passed validation and matched
the frozen CSV byte for byte. An additional public score would therefore
repeat the frozen reference.

## Reproduce

Download the seven `DFL_03_02_events_raw_*.xml` files from the
[pinned IDSSE mirror](https://huggingface.co/datasets/pysport/idsse-data/tree/a715a38dfbaf5f58e431727c2b78d174101a703c)
alongside the metadata and tracking XML. Prepare the LightGBM model using
[External Possession](EXTERNAL_POSSESSION.md).

```bash
model=data/external_possession/lightgbm-lagged-4340d27.joblib
work=data/external_possession/stoppage
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m experiments.stoppage_experiment external \
  --idsse data/idsse --model "$model" --workers 4
python dead_ball.py --output "$work/tonal-whistles.csv"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m experiments.stoppage_experiment predict \
  --model "$model" --whistles "$work/tonal-whistles.csv" --policy paired-short \
  --output "$work/paired-tonal/possession.csv"
python evaluation/validate_submission.py "$work/paired-tonal/possession.csv"
```
