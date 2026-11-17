# Possession from external tracking and phase labels

This experiment learns team possession and stoppages from SkillCorner A-League
tracking. The competition run reads the tactical ATD and its slot-to-team
metadata. It uses neither Smart Tags nor video. The competition labels enter only
the final automated public-window check.

## Method

- Build features from player-to-ball distances, ball coverage and velocity,
  player motion and pitch boundaries. Use time windows of 0.2, 0.6, 1.5 and 4
  seconds before and after each frame. Interpolate only short, bounded ball gaps
  with the tactical baseline's existing interpolation.
- Use half-open SkillCorner phase intervals for H/A. Mark gaps between phases in
  the same half as D and exclude the unbracketed ends and absent tracking rows.
  These are provider-derived labels, so their timing and treatment of stoppages
  can differ from the contest annotations.
- Fit 100 iterations of histogram gradient boosting using 12 whole matches.
  Include every other labeled source frame, swap home and away during training,
  and add a copy with simulated ball dropouts of 0.5–5 seconds. At inference,
  average the ordinary and team-swapped predictions.
- Decode the log probabilities with Viterbi. Search the Cartesian product of
  turnover and stoppage costs `{0, 2, 5, 10}` on four A-League development
  matches using mean spellF1. The selected costs were `(10, 10)` in units of
  10 Hz log evidence. Scale the emissions by `10 / fps` when applying those
  costs at other frame rates. Run the decoder independently per half.

## Results

The external comparisons use the organizer's spell metric per half, averaged
over both halves and then over matches. The historical v5 comparison has been
retired because its settings were chosen on the public window. It supplied no
features, labels, or parameters to the learned candidate.

| A-League split | Matches | Learned spellF1 |
| --- | --- | ---: |
| Development | 1886347, 1959846, 2006363, 2013725 | 0.6902 |
| Held-out test | 1925299, 1996435, 2007448, 2017461 | 0.6578 |

On the same development split, the learned classifier with zero transition costs scored 0.2845,
showing the effect of decoding entire spells.

### Boosting backends

The follow-up reused the same split, feature matrix, augmentation and decoder
cost grid. Every backend ran 100 boosting rounds at learning rate 0.1. The
three alternatives were evaluated on the external games. The contest dashboard
uses the original frozen model.

| Backend | Turnover, stoppage | Development spellF1 | Held-out spellF1 |
| --- | --- | ---: | ---: |
| Histogram gradient boosting | 10, 10 | 0.6902 | 0.6578 |
| LightGBM 4.7.0 | 5, 10 | 0.6865 | 0.6563 |
| XGBoost 3.4.1 | 5, 10 | 0.6835 | 0.6489 |
| CatBoost 1.2.10 | 10, 10 | 0.6516 | 0.6333 |

### Explicit lag features

The original 55 features already include nearest-player distances in both time
directions, ball and team speeds over multiple time offsets, and windowed ball
availability. The lag experiment keeps those features and adds past snapshots
from 0.5, 1, 2 and 4 seconds earlier. Each snapshot contains ball confidence,
distance outside the pitch, distance to the nearest player, short-window ball
speed, and, per team, the two nearest-player distances, count of players near
the ball, count of tracked players and short-window player speed. This adds 56
columns. Lags reset at each half and unavailable history stays missing.

The experiment uses the original four backends, match split, augmentation,
training parameters and decoder grid. Each lagged model selects its decoder on
the external development split. The existing public contest score still belongs
to the original frozen histogram boosting model.

| Backend | Lagged turnover, stoppage | Lagged dev spellF1 | Original held-out spellF1 | Lagged held-out spellF1 |
| --- | --- | ---: | ---: | ---: |
| Histogram gradient boosting | 10, 10 | 0.6923 | 0.6578 | 0.6596 |
| LightGBM | 5, 10 | 0.6938 | 0.6563 | 0.6616 |
| XGBoost | 5, 10 | 0.6919 | 0.6489 | 0.6560 |
| CatBoost | 10, 5 | 0.6594 | 0.6333 | 0.6276 |

LightGBM had the highest lagged A-League mean. XGBoost gained the most over
its own original fit. The scores are based on four matches, and the gains
were small.

Both variants of each booster also ran on the same seven masked IDSSE matches
described below. These models and decoder costs were carried over from the
A-League experiment.

| Backend | Original IDSSE spellF1 | Lagged IDSSE spellF1 |
| --- | ---: | ---: |
| Histogram gradient boosting | 0.4078 | 0.4090 |
| LightGBM | 0.4126 | 0.4188 |
| XGBoost | 0.4115 | 0.4142 |
| CatBoost | 0.3669 | 0.3602 |

Three backends improved on both external checks. All four lagged models also
generated valid 199,319-frame contest CSVs. The lagged outputs were checked for
format only.

The frozen A-League model was also tested on seven IDSSE Bundesliga matches
with official per-frame possession and ball-status labels. Synthetic ball
dropouts used a mean gap of 12 seconds and expected visibility of 60%.
The learned model averaged 0.4078 spellF1 on the masked matches. This measures
cross-provider transfer under simulated, label-independent tracking loss.

| Contest public window | spellF1 | macroF1 |
| --- | ---: | ---: |
| Frozen external model | 0.6250 | 0.8162 |
| Frozen lagged LightGBM | 0.6452 | 0.8084 |

The candidate was fixed before the public check. Format validation passed on all
199,319 output frames. The scores here cover only the 7,500 labeled public
frames and 17 possession spells. The private full-match score is unknown.
Lagged LightGBM is the working reference for subsequent stoppage experiments.
See [the YOLO recovery experiment](BALL_RECOVERY_EXPERIMENT.md) for the
recovery comparisons.

## Reproduce

Source data is the [SkillCorner Open Data release at
`4340d274`](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656)
under its MIT license. Get `data/matches.json` and the 20 sets of
`<id>_match.json`, `<id>_phases_of_play.csv` and
`<id>_tracking_extrapolated.jsonl` under `data/matches/<id>/`. Put the files in one
ignored local directory. GitHub serves the large tracking files through its
media endpoint. Confirm that downloads contain JSON lines rather than Git LFS
pointer files.

```bash
rev=4340d274572876239c154c90bc507a9b3250a656
dataset=data/skillcorner-4340d27
raw="https://raw.githubusercontent.com/SkillCorner/opendata/$rev"
media="https://media.githubusercontent.com/media/SkillCorner/opendata/$rev"
mkdir -p "$dataset"
curl -fsSL --retry 3 "$raw/data/matches.json" -o "$dataset/matches.json"
curl -fsSL --retry 3 "$raw/LICENSE" -o "$dataset/LICENSE"
python -c 'import json,sys; print(*(m["id"] for m in json.load(open(sys.argv[1]))))' "$dataset/matches.json" |
  tr ' ' '\n' |
  xargs -P 6 -I '{}' bash -c '
    set -e
    m=$1
    for suffix in match.json phases_of_play.csv tracking_extrapolated.jsonl; do
      file="${m}_$suffix"
      source=$3
      if [ "$suffix" = tracking_extrapolated.jsonl ]; then source=$4; fi
      curl -fsSL --retry 3 "$source/data/matches/$m/$file" -o "$2/$file.part"
      mv "$2/$file.part" "$2/$file"
    done
  ' fetch '{}' "$dataset" "$raw" "$media"
```

The run used Python 3.12.3 with NumPy 2.4.6, pandas 3.1.1, SciPy 1.18.1,
scikit-learn 1.9.1, joblib 1.6.0 and threadpoolctl 3.8.0.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 python external_possession.py train \
  --skillcorner data/skillcorner-4340d27 \
  --model data/external_possession/skillcorner-4340d27.joblib --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python external_possession.py predict \
  --model data/external_possession/skillcorner-4340d27.joblib \
  --output data/external_possession/skillcorner-frozen/possession.csv
python evaluation/validate_submission.py data/external_possession/skillcorner-frozen/possession.csv
python evaluation/run_evaluation.py --gt public --no-detail --bootstrap 0 \
  --results data/external_possession/skillcorner-frozen \
  --out data/external_possession/public-check
```

To repeat the backend comparison, install the three versions listed above and run:

```bash
for backend in lightgbm xgboost catboost; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=5 python external_possession.py train \
    --skillcorner data/skillcorner-4340d27 --workers 5 --backend "$backend" \
    --model "data/external_possession/$backend-4340d27.joblib"
done
```

To repeat the lag experiment, include `hist` in the loop, pass
`--lag-features`, and use a distinct model path such as
`data/external_possession/$backend-lagged-4340d27.joblib`. The saved model
carries its lag settings for prediction, replay and the IDSSE transfer check.

The dataset, model and outputs are local artifacts. Load only a model you
have built locally with the training command.

For the Bundesliga transfer check, get the seven `DFL_02_01_matchinformation_*.xml`
and seven `DFL_04_03_positions_raw_observed_*.xml` files from the
[pinned IDSSE mirror](https://huggingface.co/datasets/pysport/idsse-data/tree/a715a38dfbaf5f58e431727c2b78d174101a703c).
The dataset is distributed under CC BY 4.0. Cite
[Bassek et al., Scientific Data (2025)](https://doi.org/10.1038/s41597-025-04505-y).
Place the XML files in `data/idsse` and run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python external_possession.py check-transfer \
  --idsse data/idsse --model data/external_possession/skillcorner-4340d27.joblib
```
