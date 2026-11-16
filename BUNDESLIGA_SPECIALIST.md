# Targeted transfer from Bundesliga to A-League LightGBM

The [pooled fit](IDSSE_TRAINING.md) improved several external metrics and scored
0.5294 public spellF1, versus 0.6452 for the frozen A-League checkpoint. This
follow-up keeps the A-League model and its decoder fixed, and tests two narrow
uses of the Bundesliga training data.

## Dead-ball specialist blend

The Bundesliga classifier supplies only a correction to the D log probability
based on the difference in dead-ball log odds. The H/A log probabilities and
decoder costs come from the frozen A-League model. The log-odds difference is
clipped to ±3, with blend strengths of 0.15, 0.35 and 0.60.

- Global and in-pitch blends can move D evidence in either direction. The
  second requires an observed ball at least 0.5 m inside every boundary.
- Slow and uncertain variants can only raise D evidence. Both require the
  in-pitch condition and half a second of observed ball movement below
  1.5 m/s. The uncertain variant also requires the A-League D probability to
  be between 0.15 and 0.85.

No blend beat the reference on equal-league development spellF1. The best
nonzero global blend scored 0.5087 versus 0.5105 for the reference. The
bidirectional in-pitch blend peaked at 0.5090. The conservative slow and
uncertain variants tied the reference on spellF1. The no-blend reference
won ties.

## Duration transfer

Fit candidate limits from the 10th, 25th and 50th percentiles of labeled
stoppage lengths in the four IDSSE training matches. Retain only stoppages
fully inside the half with the ball inside the pitch on at least 80% of frames.
After decoding, apply a candidate cutoff only to a shorter D run bracketed
by the same team on both sides and with observed in-pitch ball throughout.
The rule leaves uncertain missing-ball gaps and changes of possession alone.

The development objective is the equal-weight mean of masked Bundesliga
spellF1 and A-League spellF1. The source splits, artificial tracking loss and
held-out scoring match the prior [IDSSE experiment](IDSSE_TRAINING.md).

| Duration limit | Development spellF1 |
| --- | ---: |
| Reference | **0.510493** |
| 7.98 s | 0.509917 |
| 10.88 s | 0.510245 |
| 22.80 s | 0.508483 |

The cutoffs came from 106 eligible labeled Bundesliga stoppages. No cutoff
improved the development mean. Both experiments selected the original model
before the held-out checks. It reproduced the existing 199,319-frame contest
CSV byte for byte, so the existing public score of 0.6452 still applies.

## Reproduce

```bash
work=data/external_possession/idsse-training
base=data/external_possession/lightgbm-lagged-4340d27.joblib
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python bundesliga_dead_ball.py calibrate \
  --idsse data/idsse --skillcorner data/skillcorner-4340d27 \
  --reference "$base" --specialist "$work/bundesliga.joblib" \
  --output "$work/dead-ball-specialist.json" --workers 4
```

For the duration experiment add `--duration-only` and use a different report
filename. Model and JSON artifacts stay under ignored `data/`.
