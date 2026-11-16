# Ball recovery with lagged LightGBM

The frozen lagged LightGBM and its Viterbi decoder are the reference. A second
inference pass adds accepted YOLO observations at the same confidence weight
(0.55) as the prior tactical recovery experiment. No model weights or decoder
costs change. Both passes use the same players, and only missing ball positions
are eligible for recovery.

## Fusion policies

- `none` keeps the original probabilities.
- `replace` uses all three probabilities from the recovered stream.
- `joint-25` and `joint-50` blend all three probabilities, with 25% or 50%
  weight on the recovered stream.
- The `side-*` policies blend conditional H/A probabilities while preserving
  the original D probability. This preserves the gap signal before decoding.
  Decoded D intervals can still change when stronger live-team evidence changes
  the best path through the half.
- `boundary-50` uses 50% side fusion. It can also increase D probability if a
  directly recovered observation projects more than one metre outside the pitch
  and the recovered stream gives D more support.

## External calibration

Across the 12 SkillCorner training matches, ball coordinates are missing for
6,089 of 379,288 live frames (1.6%) and 177,926 of 330,016 stopped frames
(53.9%). This motivates a second simulation where gaps are more common during
stoppages.

The independent IDSSE simulation uses 60% expected visibility. The informative
simulation uses 80% expected visibility and overlays a second mask within
externally labeled stoppages, for roughly 38% expected visibility there. The
external labels are used for this simulation and evaluation. Gap durations
average 12 seconds before the masks are combined. Simulated accepted recoveries
are short bursts of two to six frames with spatially coherent offsets. Most offsets have 0.5 m standard
deviation, with 10% using 2 m. Test two nominal coverage levels, 5% and 30% of
masked frames before overlap and pitch filtering. This tests how the fusion
responds to sparse and denser observations. The simulated recovery positions
come from IDSSE tracking, so this experiment measures fusion rather than
detector precision. Development games are J03WMX, J03WOH, J03WPY and J03WQQ.
J03WN1, J03WOY and J03WR9 are the held-out games. Average over both halves and
both coverage settings, then equally over matches. Choose by development
spellF1, with D-class F1 as the tie-breaker.

| Policy | Independent dev spellF1 | Independent test spellF1 | Informative dev spellF1 | Informative test spellF1 | Informative test D F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.4290 | 0.4064 | 0.4600 | 0.4244 | 0.9195 |
| replace | **0.4737** | **0.4615** | **0.4950** | **0.4620** | **0.9465** |
| joint-25 | 0.4488 | 0.4350 | 0.4730 | 0.4403 | 0.9323 |
| joint-50 | 0.4628 | 0.4513 | 0.4836 | 0.4509 | 0.9392 |
| side-25 | 0.4422 | 0.4263 | 0.4648 | 0.4325 | 0.9206 |
| side-50 | 0.4500 | 0.4396 | 0.4694 | 0.4390 | 0.9203 |
| side-100 | 0.4569 | 0.4455 | 0.4728 | 0.4425 | 0.9188 |
| boundary-50 | 0.4514 | 0.4400 | 0.4702 | 0.4402 | 0.9221 |

Both development simulations selected `replace`. The classifier can score
stopped play from recovered ball geometry and motion, and unrecovered positions
still contribute missingness. The synthetic recoveries are deliberately derived
from the actual ball trace. Their coverage and localization may exceed the
observations available from the checkpoint.

## Contest checks

The match-wide sparse scan processed 6,451 eligible frames and accepted 555
recoveries, including 28 in the public window. It produced a valid 199,319-frame
CSV. The public check covers 7,500 graded frames and 17 possession spells.

| Pipeline | Public spellF1 | Public macroF1 | Public D F1 |
| --- | ---: | ---: | ---: |
| Frozen lagged LightGBM | 0.6452 | 0.8084 | 0.6861 |
| LightGBM + sparse YOLO recovery | 0.6452 | 0.8098 | 0.6866 |
| LightGBM + dense YOLO, window diagnostic | 0.6194 | 0.8092 | 0.6861 |

The dense diagnostic used the same frozen fusion policy on frames 10985–18884,
including 8 seconds of context on each side of the public window. It accepted
273 of 885 missing observations. The two detector checks gave no public
spellF1 gain. They are pipeline diagnostics, and the original frozen LightGBM
remains the reference. The private score is unknown.

## Reproduce

Use the lagged LightGBM checkpoint from [External Possession](EXTERNAL_POSSESSION.md)
and get `yolo-sn-ball-opt.pt` from the authors'
[SoccerNet-v3D weight list](https://github.com/mguti97/SoccerNet-v3D#weights).
The detector checkpoint used in this run has SHA-256
`a3082fb435a8501ae17cfe4ac78e66ca7041205e115feda344d34d1693064f36`.
The detector uses the tactical video locally. The projection, confidence
threshold and temporal acceptance filter are the same as in
[Tactical Baseline](../../TACTICAL_BASELINE.md). The sparse pass scans gap edges
within 0.2 seconds of original tracking and one consecutive frame pair per
second inside long gaps, throughout both halves. Omit `--sparse` to run on
every eligible missing frame.

```bash
model=data/external_possession/lightgbm-lagged-4340d27.joblib
work=data/external_possession/yolo-recovery
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m experiments.ball_recovery_experiment calibrate \
  --idsse data/idsse --model "$model" --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m experiments.ball_recovery_experiment calibrate \
  --idsse data/idsse --model "$model" --workers 4 --informative-gaps
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m experiments.ball_recovery_experiment extract \
  --device cuda --checkpoint models/yolo-sn-ball-opt.pt --sparse \
  --output "$work/sparse-atd-ball.csv"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m experiments.ball_recovery_experiment predict \
  --model "$model" --recoveries "$work/sparse-atd-ball.csv" --policy replace \
  --output "$work/candidate/possession.csv"
python evaluation/validate_submission.py "$work/candidate/possession.csv"
python evaluation/run_evaluation.py --gt public --results "$work/candidate" \
  --out "$work/public-check" --no-detail --bootstrap 0
```

For the dense diagnostic, replace `--sparse` with
`--frame-range 10985 18885` and use a separate cache, prediction and report
directory.
