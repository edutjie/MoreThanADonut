# Bundesliga and combined H/A/D training

This experiment tests training the lagged possession LightGBM on the seven
[IDSSE Bundesliga matches](https://doi.org/10.1038/s41597-025-04505-y).
At 25 fps, inactive ball status gives D, and active ball status paired
with team possession gives H or A. Unknown labels are excluded from fitting.
These labels include stoppages with the ball still on the pitch.

## Training and evaluation

| Source | Fit | Development | Held-out |
| --- | --- | --- | --- |
| IDSSE Bundesliga | J03WMX, J03WOH, J03WPY, J03WQQ | J03WN1 | J03WOY, J03WR9 |
| SkillCorner A-League | Original 12 whole matches | Original four | Original four |

IDSSE training has 457,400 rows after sampling each half at 5 Hz, mirroring
the teams and adding a masked copy of each sequence. The combined fit
has 1,876,032 rows and gives each dataset equal total sample weight.
The feature set, number of boosting rounds and team-symmetric prediction
are the same as the frozen A-League LightGBM. The original A-League training
and augmentation are reused as documented in
[External Possession](EXTERNAL_POSSESSION.md).

IDSSE masking simulates label-independent gaps with a 12-second mean
duration and 60% expected ball visibility. Development and held-out IDSSE
receive this mask. SkillCorner evaluation uses the original observations.
SpellF1 is computed per half, then averaged over halves and matches. For
in-pitch D F1, only frames with a ball coordinate at least 0.5 m inside
the pitch in the unmasked reference are included. That coordinate is
available to the metric even when hidden from the model.

For each new model, pick its Viterbi costs on the development set using
the mean of the two league spellF1 scores. The original A-League model
keeps its frozen decoder settings. This same development metric selects
among the three checkpoints before the held-out evaluation is read. The
IDSSE games have been used in earlier transfer experiments. The held-out
designation here refers to the current retraining comparison.

## External results

| LightGBM training source | Development spellF1, equal-league mean | Held-out Bundesliga spellF1 | Held-out Bundesliga in-pitch D F1 | Held-out A-League spellF1 |
| --- | ---: | ---: | ---: | ---: |
| A-League | 0.5105 | 0.4369 | 0.9100 | **0.6616** |
| Bundesliga | 0.4268 | **0.4859** | **0.9405** | 0.4487 |
| Combined | **0.5276** | 0.4609 | 0.9264 | 0.6490 |

Both new fits selected the same decoder costs as the reference, 5 for
turnovers and 10 for stoppages. The combined fit won the development
comparison and was frozen before the contest check. Bundesliga-only
training scored highest on masked Bundesliga tracking and dropped
sharply when applied to A-League broadcast tracking.

## Contest pipeline check

Both new checkpoints generated valid 199,319-frame predictions. The
externally selected combined model was scored on the 7,500 public frames
once. The Bundesliga-only prediction was format-checked.

| Training source | Public spellF1 | Public macroF1 | Public D F1 |
| --- | ---: | ---: | ---: |
| Frozen A-League reference | 0.6452 | 0.8084 | 0.6861 |
| Frozen combined | 0.5294 | 0.7635 | 0.5983 |

The two held-out Bundesliga matches showed better in-pitch D detection
with additional Bundesliga training. That gain did not carry over to the
public contest check. The A-League checkpoint remains the working reference.
The private score is unknown.

## Reproduce

Download the metadata and observed-position files from the
[pinned IDSSE mirror](https://huggingface.co/datasets/pysport/idsse-data/tree/a715a38dfbaf5f58e431727c2b78d174101a703c)
under CC BY 4.0. The event XML is unnecessary for this experiment.

```bash
work=data/external_possession/idsse-training
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m experiments.idsse_training \
  --idsse data/idsse --skillcorner data/skillcorner-4340d27 \
  --reference data/external_possession/lightgbm-lagged-4340d27.joblib \
  --output "$work" --workers 4
```

The output includes both retrained checkpoints and a JSON report of the
development selection and held-out scores.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python external_possession.py predict \
  --model "$work/combined.joblib" --output "$work/combined-candidate/possession.csv"
python evaluation/validate_submission.py "$work/combined-candidate/possession.csv"
python evaluation/run_evaluation.py --gt public --results "$work/combined-candidate" \
  --out "$work/combined-public" --no-detail --bootstrap 0
```
