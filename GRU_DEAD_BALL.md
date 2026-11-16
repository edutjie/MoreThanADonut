# Bundesliga live/dead specialist with a frozen GRU

This experiment keeps the [original GRU](SEQUENCE_POSSESSION.md) for
possession and tests whether a dedicated classifier improves its dead-ball
decisions when the ball is visible inside the pitch.

## Method

- Fit two binary LightGBM models on the four IDSSE training matches.
  Sample at 5 Hz and mirror the teams. A ball coordinate must be observed
  and at least 0.5 m inside the pitch. Use the official ball status and
  remove unknown targets. Reuse the GRU's tracking feature inputs without
  the sequence-only visibility features.
- One fit uses uniform weights. The other gives slow live balls four
  times normal weight. A 0.5-second observed history with speed at most
  1.5 m/s throughout defines a slow ball. Missing steps break that history.
- Average the specialist probability across both team orientations. For
  observed in-pitch frames, adjust only the GRU's dead-ball log emission
  by the specialist-minus-GRU dead-ball log odds, clipped to ±3 and
  scaled by the blend weight. Keep the GRU's decoder costs at (5, 10).
- Compare weights 0.15, 0.35 and 0.60 for both fits alongside the GRU alone.
  Select by the mean of spellF1 and in-pitch dead-ball F1, giving A-League
  and Bundesliga development equal weight. Freeze the overall winner
  and each classifier's best weight on development. Run the public check
  with the winning policy.

The masked test adds label-independent dropouts with 12-second mean gap
duration and 60% expected visibility. Its in-pitch evaluation uses the
original observed ball coordinates. The models and blend gate see the
masked tracks.

The separate broadcast check pairs StatsBomb non-advantage foul markers
with a same-team free kick in three Clásico halves with consistent clock
offsets. It excludes the first and last second of each interval and
counts only observed in-pitch frames. These are weak positive labels,
so the check measures coverage, and false positives are measured on
slow live balls in the fully labeled tests.

## Results

Training used 161,920 rows including mirrored copies, with 34,184 dead-ball
rows and 3,504 slow-live rows. Development selected weight 0.15 for both
fits. The selection score was 0.72977 for the GRU, 0.73421 with uniform
weights, and 0.73422 with slow-live upweighting.

| Held-out tracking | GRU spellF1 | GRU + weighted specialist spellF1 | GRU in-pitch D F1 | GRU + weighted specialist in-pitch D F1 |
| --- | ---: | ---: | ---: | ---: |
| A-League | 0.6728 | 0.6693 | 0.8237 | 0.8101 |
| A-League with simulated gaps | 0.5909 | 0.5890 | 0.7072 | 0.7049 |
| Bundesliga | 0.5100 | 0.5136 | 0.9369 | 0.9486 |
| Bundesliga with simulated gaps | 0.4376 | 0.4406 | 0.8891 | 0.8984 |

At weight 0.15, uniform and slow-live-weighted fits differed by less than
0.0008 across the held-out metrics. They made the same number of false
dead-ball calls on slow live examples and matched the same number of weak
dead-ball frames in the Clásico check.

The combined model made fewer false dead-ball calls on tracked slow live
examples. On clean A-League it mislabeled 166 of 3,655 such frames, down
from 207. On clean Bundesliga it mislabeled 30 of 5,847, down from 44.
In the Clásico check, weak dead-ball frame coverage went from 1,392 to
1,324 of 1,736. Both covered at least half of 25 of the 30 eligible
intervals.

| Automated public check | SpellF1 | MacroF1 | Dead-ball F1 |
| --- | ---: | ---: | ---: |
| Frozen GRU | 0.6621 | 0.8894 | 0.8946 |
| GRU + weighted specialist | 0.6621 | 0.8871 | 0.8893 |

The 199,319-frame CSV passed validation and reproduced byte for byte.
The public window contains 7,500 graded frames and 17 possession spells.
The original GRU was retained after this experiment. See the
[current baseline](TACTICAL_BASELINE.md#current-experiment-baseline) for
the subsequent player-tracking experiment.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
work=data/external_possession/gru-dead-ball
gru=data/external_possession/sequence-gru/gru.pt
python gru_dead_ball.py calibrate --device cuda --reference "$gru" \
  --idsse external_data/idsse --skillcorner external_data/skillcorner-4340d27 \
  --legacy external_data/skillcorner-history --statsbomb external_data/statsbomb-clasicos \
  --output "$work"
python gru_dead_ball.py predict --device cuda --reference "$gru" \
  --specialists "$work/specialists.joblib" --best-enabled \
  --output "$work/best-enabled/possession.csv"
python evaluation/validate_submission.py "$work/best-enabled/possession.csv"
python evaluation/run_evaluation.py --gt public --results "$work/best-enabled" \
  --out "$work/best-enabled/public-check" --no-detail --bootstrap 0
```

Omit `--best-enabled` to reproduce the overall development selection.
The aggregate metrics file is `results.json`. Checkpoints and prediction
CSVs stay in the ignored data directory.
