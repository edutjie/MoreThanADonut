# Possession spell smoothing and Bundesliga YOLO check

The [official scorer](evaluation/possession_metrics.py) treats every contiguous
H or A interval as a possession spell. D separates spells. Same-team intervals
are matched one to one by temporal intersection over union. The ranking
metric averages F1 at thresholds 0.3, 0.4, 0.5, 0.6 and 0.7. A short
mistaken label can split one long possession into two predicted spells.

## Bundesliga-only LightGBM with YOLO recovery

The Bundesliga-only checkpoint used the same match-wide sparse recovery cache
and fixed direct-replacement policy as the earlier [YOLO experiment](BALL_RECOVERY_EXPERIMENT.md).
The policy was carried over as a diagnostic. It was fixed before any public
scoring of this combination.

| Frozen Bundesliga-only prediction | Public spellF1 | Public macroF1 | Public D F1 |
| --- | ---: | ---: | ---: |
| Tracking | 0.4800 | 0.7035 | 0.4496 |
| Tracking + sparse YOLO | 0.4800 | 0.6901 | 0.4145 |

All 199,319 frames passed submission validation. Public scores use the same
7,500 public labels and 17 true possession spells.

## SpellF1-based decoder calibration

The frozen A-League LightGBM emits the same per-frame H/A/D probabilities
as before. The search compares 15 settings, including the original decoder.
Viterbi variants change the penalty for switching teams, entering D or
leaving D. Three other settings merge short, weakly supported live pulses
between the same team, such as a brief A between two H runs. That rule
preserves D intervals and strong short turnovers.

Selection uses the average of the two league spellF1 scores on the external
development split in [IDSSE Training](IDSSE_TRAINING.md). Held-out tests are
scored after selection.

Development selected the original Viterbi costs plus a 1.5-second limit
for weak live flickers. Its balanced spellF1 was 0.511384 versus 0.510493
for the reference. The held-out tests did not reproduce that small gain.

| Model outputs | Held-out masked Bundesliga spellF1 | Held-out A-League spellF1 |
| --- | ---: | ---: |
| Frozen LightGBM | **0.4369** | **0.6616** |
| Frozen LightGBM + selected smoothing | 0.4350 | 0.6597 |

The frozen smoothing candidate passed full CSV validation. It changed 81
labels outside the 7,500 public frames. Public spellF1 and macroF1 remained
0.6452 and 0.8084. The frozen A-League decoder remains the working reference.

## Reproduce

```bash
data_root=data/external_possession
base="$data_root/lightgbm-lagged-4340d27.joblib"
work="$data_root/spell-smoothing"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python spell_smoothing.py calibrate \
  --idsse data/idsse --skillcorner data/skillcorner-4340d27 --model "$base" \
  --output "$work/calibration.json" --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python spell_smoothing.py predict \
  --model "$base" --calibration "$work/calibration.json" --output "$work/candidate/possession.csv"
python evaluation/validate_submission.py "$work/candidate/possession.csv"
python evaluation/run_evaluation.py --gt public --results "$work/candidate" \
  --out "$work/public-check" --no-detail --bootstrap 0
```

Use `ball_recovery_experiment.py predict` with
`--model "$data_root/idsse-training/bundesliga.joblib"`,
`--recoveries "$data_root/yolo-recovery/sparse-atd-ball.csv"` and
`--policy replace` for the Bundesliga-only YOLO diagnostic. Keep the
predictions, model files and recovered coordinates under ignored local paths.
