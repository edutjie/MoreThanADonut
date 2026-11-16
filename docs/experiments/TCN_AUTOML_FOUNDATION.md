# TCN, AutoGluon, and foundation-model comparisons

The [continuous-track GRU](PLAYER_TRACK_QUALITY.md) is the reference. These
experiments use 12 A-League matches for training, four separate A-League matches
for checkpoint selection and AutoML validation, four held-out A-League matches
for testing, and two held-out IDSSE Bundesliga matches for transfer testing.
All fitted feature transformations, normalization, and classifier heads use
external data. The competition match supplies frozen inference inputs and an
automated public score check. It supplies no training or validation labels.

## Methods

- **Shared inputs.** The aggregate models use the current repaired player
  tracking, continuous speed calculation, lag features, and ball visibility and
  observation age. Training includes the natural tracks and a second copy with
  synthetic ball and player artifacts. For evaluation, natural tracking and
  60% synthetic ball masking are separate conditions. The fixed Viterbi decoder
  uses the same costs for all contenders. Predictions are interpolated from
  the model grid to the native frame grid before decoding.
- **TCNs.** The aggregate TCN swaps the GRU encoder for six dilated convolutions
  over the same summaries. The [Borghesi et al. adaptation](https://doi.org/10.1016/j.eswa.2023.119780)
  embeds each track with one shared, approximately four-second symmetric TCN.
  The ball token queries all visible player trajectories through separate
  attention pools for dead/live and conditional home/away. It retains the ball
  token when its coordinates are missing and uses explicit observation,
  velocity-validity and confidence channels. Pitch coordinates, movement
  and player-to-ball distance are inputs. Referee trajectories are omitted.
- **AutoGluon feature generator.** The default pipeline was fitted on training
  rows only and fed to LightGBM, GRU, Transformer, and aggregate TCN. The rolling
  variant adds centered 0.5 s, 1.5 s, and 4.1 s means, standard deviations,
  and valid fractions of eight visibility and movement signals. Each window
  stays within its own match half. The feature generator then removes redundant
  columns. The convolutional and recurrent fits use eight epochs and freeze
  their checkpoint by development spellF1.
- **AutoGluon modeling.** The TabularPredictor runs used development log loss
  with no row-random bagging or stacked models. The 240-second run used roughly
  5 Hz training rows and spent its budget on LightGBM. The 360-second run used
  roughly 1.67 Hz training rows, allowing up to 60 seconds per family. A
  dedicated 230-second GPU run retried the neural model that timed out.
- **Frozen pretrained encoders.** MOMENT-1-small and the original Mantis-8M
  encode six tracking signals across 6.4 s, with about 4.8 s of history.
  The model families are channel independent. A fixed random projection
  compresses each channel embedding to 32 dimensions. A regularized logistic
  head uses the embeddings and current tracking features. The matched control
  uses those current features alone. Fit anchors are 1 Hz and evaluation
  anchors 2 Hz, so this is a feature probe with coarser temporal resolution.
  The two pretrained encoders remain frozen.
  [MOMENT model card](https://huggingface.co/AutonLab/MOMENT-1-small)
  [Mantis reference implementation](https://github.com/vfeofanov/mantis)

## Results

The default feature generator preserved all 113 columns and their values on
1,871,380 orientation-specific external development and test rows. LightGBM
produced identical metrics with and without it. The rolling comparison supplied
185 columns to the generator and retained 175.

### External holdouts

SpellF1 averages the two halves, then matches. In-pitch D-F1 pools frames on
which the original ball was observed inside the pitch. Natural gap hits count
A-League target spells with at least one second of missing ball that were matched
at IoU 0.5.

| Model | A-League dev spellF1 | A-League spellF1 | A-League masked | Bundesliga spellF1 | Bundesliga masked | Bundesliga in-pitch D-F1 natural / masked | A-League natural gap hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen GRU | 0.7110 | 0.6748 | 0.5855 | 0.5163 | 0.4341 | 0.9310 / 0.8984 | 15/30 |
| LightGBM + default AutoGluon features | 0.6890 | 0.6601 | 0.5481 | 0.5393 | 0.4459 | 0.9545 / 0.9079 | 11/30 |
| GRU + default AutoGluon features | 0.7088 | 0.6753 | 0.5958 | 0.5084 | 0.4429 | 0.9221 / 0.9103 | 13/30 |
| Transformer + default AutoGluon features | 0.6729 | 0.6451 | 0.5651 | 0.4834 | 0.4158 | 0.9158 / 0.8775 | 12/30 |
| Aggregate TCN + default AutoGluon features | 0.7067 | 0.6704 | 0.5945 | 0.5132 | 0.4470 | 0.9115 / 0.8960 | 12/30 |
| Per-player two-head TCN | 0.6251 | 0.6054 | 0.5488 | 0.4735 | 0.4182 | 0.8869 / 0.8385 | 10/30 |
| LightGBM + rolling features | 0.6915 | 0.6568 | 0.5418 | 0.5368 | 0.4421 | 0.9494 / 0.9060 | 12/30 |
| GRU + rolling features | 0.7196 | 0.6722 | 0.5741 | 0.5195 | 0.4338 | 0.9340 / 0.8071 | 15/30 |
| Transformer + rolling features | 0.6885 | 0.6369 | 0.5524 | 0.4883 | 0.4193 | 0.9219 / 0.8361 | 12/30 |
| Aggregate TCN + rolling features | 0.7107 | 0.6679 | 0.5909 | 0.5281 | 0.4576 | 0.9258 / 0.8634 | 11/30 |
| AutoGluon dense LightGBM | 0.6999 | 0.6808 | 0.5665 | 0.5625 | 0.4532 | 0.9559 / 0.9134 | 12/30 |
| AutoGluon budgeted ensemble | 0.6923 | 0.6759 | 0.5642 | 0.5368 | 0.4445 | 0.8979 / 0.8716 | 11/30 |
| AutoGluon GPU neural net | 0.7022 | 0.6751 | 0.5374 | 0.5447 | 0.4419 | 0.9490 / 0.8952 | 13/30 |
| Foundation probe linear control | 0.6123 | 0.5957 | 0.4767 | 0.4540 | 0.3817 | 0.9028 / 0.8056 | 9/30 |
| MOMENT-1-small + linear head | 0.6303 | 0.6198 | 0.4927 | 0.4632 | 0.3945 | 0.8925 / 0.7897 | 8/30 |
| Mantis-8M + linear head | 0.6519 | 0.6315 | 0.4963 | 0.4806 | 0.3931 | 0.8944 / 0.7784 | 12/30 |

### Automated public window

All 15 frozen CSVs passed the 199,319-row submission validator. The
public window covers 7,500 graded frames and 17 possession spells. Rows are
shown for diagnosis. The public window does not select checkpoints or tune
models. Private scoring is unavailable.

| Model | Public spellF1 | Public macroF1 | Public dead-ball F1 |
| --- | ---: | ---: | ---: |
| Frozen GRU | 0.6897 | 0.9002 | 0.8882 |
| Aggregate TCN + default AutoGluon features | 0.6815 | 0.9160 | 0.8926 |
| Aggregate TCN + rolling features | 0.6759 | 0.8907 | 0.8204 |
| GRU + default AutoGluon features | 0.6667 | 0.8080 | 0.6640 |
| AutoGluon GPU neural net | 0.6625 | 0.8593 | 0.7871 |
| GRU + rolling features | 0.6533 | 0.8221 | 0.6960 |
| MOMENT-1-small + linear head | 0.6483 | 0.8318 | 0.7078 |
| LightGBM + rolling features | 0.6452 | 0.8468 | 0.7828 |
| AutoGluon budgeted ensemble | 0.6323 | 0.8368 | 0.7505 |
| LightGBM + default AutoGluon features | 0.6250 | 0.8027 | 0.6699 |
| AutoGluon dense LightGBM | 0.5939 | 0.8420 | 0.7749 |
| Mantis-8M + linear head | 0.5750 | 0.7225 | 0.4926 |
| Per-player two-head TCN | 0.5290 | 0.5326 | 0.0000 |
| Transformer + rolling features | 0.4769 | 0.8768 | 0.8423 |
| Transformer + default AutoGluon features | 0.4640 | 0.8478 | 0.7907 |

The aggregate TCN was close on held-out A-League spellF1 and improved masked
Bundesliga spellF1. Its public macroF1 was the strongest, while it placed
10 public spells against 17 labeled. The per-player version did worse on
spellF1 and in-pitch stoppages across the tested conditions and predicted
live play throughout the public window. The dense
AutoML fit improved natural Bundesliga spellF1 and in-pitch dead-ball F1,
while it lost ground under A-League ball masking and in the public check.
The budgeted ensemble combined XGBoost (73.9%) and LightGBM (26.1%).
The frozen foundation features improved on the matched linear control for
spellF1 across the external conditions. Their masked in-pitch dead-ball F1
was lower than that control. The existing frozen GRU remains the baseline.

The experiment uses one seed per neural architecture and two Bundesliga
holdout games. Timing budgets can change AutoGluon iterations and weights.

## Reproduce

The commands assume the local datasets and GRU checkpoint described in
[Player track quality](PLAYER_TRACK_QUALITY.md). Use the project's PyTorch
environment and install the optional dependencies in separate directories.
The foundation runner pins both checkpoint revisions.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
work=data/external_possession/tcn-study
uv pip install --target "$work/automl-deps" 'autogluon.tabular[lightgbm,catboost,xgboost]==1.5.0'
uv pip install --target "$work/foundation-deps" 'transformers==4.57.6' \
  safetensors 'huggingface-hub==0.36.2' einops
uv pip install --target "$work/foundation-deps" --no-deps 'momentfm==0.1.4' 'mantis-tsfm==1.1.0'
python -m experiments.possession_study_data \
  --skillcorner external_data/skillcorner-4340d27 --idsse external_data/idsse \
  --output "$work/cache"
python -m experiments.possession_study baseline --cache "$work/cache" --output "$work/baseline" \
  --reference data/external_possession/player-quality/continuous/gru.pt
export PYTHONPATH="$work/automl-deps"
python -m experiments.possession_study lgbm --cache "$work/cache" --output "$work/lgbm"
for model in lgbm gru transformer tcn; do
  python -m experiments.possession_study "$model" --cache "$work/cache" \
    --features auto --output "$work/auto$model"
  python -m experiments.possession_study "$model" --cache "$work/cache" \
    --features rolling-auto --output "$work/rolling-$model"
done
python -m experiments.possession_study player-tcn --cache "$work/cache" --output "$work/player-tcn"
python -m experiments.possession_study automl --cache "$work/cache" --output "$work/automl" \
  --device cpu --time-limit 240
python -m experiments.possession_study automl --cache "$work/cache" --output "$work/automl-budgeted" \
  --device cpu --workers 8 --row-stride 6 --time-limit 360 --family-time-limit 60
python -m experiments.possession_study automl --cache "$work/cache" --output "$work/automl-torch" \
  --device cuda --row-stride 6 --time-limit 230 --automl-families NN_TORCH
export PYTHONPATH="$work/foundation-deps" HF_HUB_CACHE="$work/hf-cache"
for model in moment mantis; do
  python -m experiments.possession_foundation "$model" --cache "$work/cache" --output "$work/$model"
done
```

Use the matching dependency directory for each candidate. The foundation
checkpoints can be used offline after the first download.

```bash
candidate=autotcn
export PYTHONPATH="$work/automl-deps"
python -m experiments.possession_study_predict --candidate "$work/$candidate" \
  --data-dir data --output "$work/public/$candidate.csv"
python evaluation/validate_submission.py "$work/public/$candidate.csv"
python evaluation/run_evaluation.py --gt public --results "$work/public" \
  --out "$work/public-check" --no-detail --bootstrap 0
```
