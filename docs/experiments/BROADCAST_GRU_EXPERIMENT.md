# Aligned broadcast tracking with the frozen GRU

The [current GRU baseline](../../TACTICAL_BASELINE.md#current-experiment-baseline)
was trained on SkillCorner A-League broadcast-derived tracking. This experiment
tests using the contest broadcast tracking wherever the existing
[frame map](BROADCAST_INTEGRATION_EXPERIMENT.md) verifies alignment.

## Method

Run the same frozen GRU and its saved tracking preprocessing on the tactical
half and on each contiguous verified broadcast span. A span stops if either
frame index skips, so the model does not see a replay or other unverified
footage as temporal context. Project the broadcast probabilities onto the
tactical frame grid and apply the original decoder once per tactical half.

- **Broadcast first** uses broadcast probabilities on every verified frame
  and tactical probabilities elsewhere.
- **Broadcast for ball gaps** uses broadcast probabilities where the map
  verifies alignment, tactical has no observed ball, and broadcast has one.
- **Clipped tactical control** uses tactical tracking restricted to the
  same verified spans to separate the effect of shorter context from input
  feed differences. It was run as a diagnostic after the first comparison.

The rules and checkpoint were fixed for each run before scoring its output.
The decoder can propagate a change beyond the exact replacement frames.

## Results

The frame map covers 92,629 of 158,857 frames in the tactical match halves
(58.3%), spanning 199 verified intervals. Within those intervals the
broadcast feed supplied a ball coordinate absent from tactical on 5,632
frames. Tactical alone supplied a coordinate on 4,244 frames. Alignment
covered 5,575 of the 7,500 public graded frames, with 157 broadcast-only ball
observations in that window.

| Frozen input policy | Public spellF1 | Public macroF1 | Public dead-ball F1 | Public predicted spells |
| --- | ---: | ---: | ---: | ---: |
| Tactical whole halves | **0.6897** | **0.9002** | **0.8882** | 12 |
| Broadcast first | 0.5697 | 0.8813 | 0.8685 | 16 |
| Broadcast for ball gaps | 0.6897 | 0.9002 | 0.8882 | 12 |
| Clipped tactical control | 0.6483 | 0.8775 | 0.8591 | 12 |

The clipped tactical control lost spellF1, so shortening the GRU context
contributed to the decline. Broadcast first still produced more spells and
lower spellF1 than the clipped tactical control. The ball-gap policy changed
1,618 decoded labels across the full match but produced identical public
labels. The public check covers 17 true possession spells and cannot
establish performance on ungraded frames. Tactical whole halves remain the
current baseline.

All four CSVs passed validation for 199,319 tactical frames. The tactical
output matched the frozen baseline byte for byte, and the first three
policies reproduced byte for byte on a second run.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
work=data/external_possession/broadcast-gru
python -m experiments.broadcast_gru_experiment \
  --model data/external_possession/player-quality/continuous/gru.pt \
  --broadcast-map submissions/frozen-broadcast/frame-map.npz \
  --data-dir data --device cuda --output "$work"
for policy in tactical broadcast-first broadcast-ball-gap tactical-clipped; do
  python evaluation/validate_submission.py "$work/$policy/possession.csv"
  python evaluation/run_evaluation.py --gt public --results "$work/$policy" \
    --out "$work/$policy/public-check" --no-detail --bootstrap 0
done
```

The script writes aggregate coverage to `coverage.json`. The frame map,
checkpoints, predictions and scorer outputs remain local.
