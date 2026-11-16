# Player tracking quality in the possession models

Player coordinates influence ball-to-player distances, team visibility and
player speed features. This experiment changes the tracking inputs used by
the [GRU](SEQUENCE_POSSESSION.md) and the
[dead-ball specialist](GRU_DEAD_BALL.md).

## Model inputs

- Treat exact coordinate holds of two seconds or more as missing player
  observations. Leave the affected interval missing because there is no
  reliable position to recover from the tracklet alone.
- Repair short out-and-back jumps up to 0.2 seconds long when each edge
  jumps more than 2 m, and the surrounding observed positions imply
  at most 12 m/s. Interpolate only between those surrounding positions.
- The continuity variant also excludes player-speed samples crossing a
  missing frame or a jump larger than the greater of 1 m and 15 m/s per
  frame. It leaves a team's speed feature missing if no valid track
  contributes. It retains the positions for distance calculations.
- Ball coordinates and timestamps carry through the same input pipeline.
  The retrained GRU checkpoints record their preprocessing settings. The
  frozen specialist continues to use its original speed convention on
  the repaired coordinates.

The frozen GRU comparisons isolate inference-time coordinate repair.
The retrained runs use the same 12 A-League training matches, two copies,
eight epochs, feature widths, decoder, and seed as the original GRU.
Both training copies use repaired inputs. The existing dropout copy
also receives label-independent player holds and short spikes before repair.
The two retrained runs differ in how they calculate player speeds.

## Selection and validation

The checkpoint and model comparison use equal weight on original tracking
and injected player faults in the four A-League development matches.
The stress test attempts one fault per four seconds across all players on
average. Most faults are 2.5–5 second exact holds. The other events move one
track for 0.04–0.16 seconds. Both held-out leagues use the same corruption
procedure across candidates. These synthetic tests target specific faults
and do not represent every assignment error in the contest tracker.

On tactical match halves, coordinate repair changed 6,698 of 3,254,570
observed player-frame positions, all due to exact held coordinates.

## Results

The final development selection was the retrained GRU with continuous
player speeds. The table reports spellF1 and uses the same fault draws
for every row within an injected test.

| Model | A-League dev mean | A-League test | A-League injected | Bundesliga test | Bundesliga injected |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original GRU | 0.7006 | 0.6728 | 0.6651 | 0.5100 | 0.5140 |
| Original GRU + Specialist | 0.7012 | 0.6693 | 0.6649 | 0.5136 | 0.5176 |
| Original GRU, repaired inputs | 0.7010 | 0.6728 | 0.6549 | 0.5099 | 0.5088 |
| Original GRU + Specialist, repaired inputs | 0.7017 | 0.6693 | 0.6550 | 0.5129 | 0.5124 |
| Retrained GRU, repaired inputs | 0.7023 | 0.6744 | 0.6623 | 0.5164 | 0.5128 |
| Retrained GRU + Specialist, repaired inputs | 0.7023 | 0.6737 | 0.6595 | 0.5194 | 0.5159 |
| Retrained GRU, continuous speeds | **0.7078** | **0.6748** | 0.6620 | 0.5163 | 0.5170 |
| Retrained GRU + Specialist, continuous speeds | 0.7068 | 0.6718 | 0.6623 | 0.5170 | 0.5194 |

Coordinate repair alone was weaker on the injected tests. Retraining
with continuous speeds improved three of four held-out conditions versus
the original GRU and trailed on injected A-League. In the older
ball-dropout test, its spellF1 was 0.5855 on masked A-League and 0.4341 on
masked Bundesliga, versus 0.5909 and 0.4376 for the original GRU.

| Frozen model | Public spellF1 | Public macroF1 | Public dead-ball F1 |
| --- | ---: | ---: | ---: |
| Original GRU | 0.6621 | 0.8894 | 0.8946 |
| Original GRU + Specialist | 0.6621 | 0.8871 | 0.8893 |
| Retrained GRU + Specialist, repaired inputs | 0.6897 | 0.8959 | 0.8820 |
| Retrained GRU, continuous speeds | 0.6897 | 0.9002 | 0.8882 |

The two new rows are the development selections from the comparison
without continuous speeds and the final comparison, respectively.
Both predictions passed the 199,319-frame validator and reproduced
byte for byte through the normal inference entry points. The public
check covers 7,500 frames and 17 possession spells.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
work=data/external_possession/player-quality
aleague=external_data/skillcorner-4340d27
idsse=external_data/idsse
lightgbm=data/external_possession/lightgbm-lagged-4340d27.joblib
python sequence_possession.py train --skillcorner "$aleague" --idsse "$idsse" \
  --reference "$lightgbm" --device cuda --repair-players --output "$work/retrained"
python sequence_possession.py train --skillcorner "$aleague" --idsse "$idsse" \
  --reference "$lightgbm" --device cuda --repair-players --continuous-motion \
  --output "$work/continuous"
python player_quality_experiment.py --skillcorner "$aleague" --idsse "$idsse" \
  --original data/external_possession/sequence-gru/gru.pt --retrained "$work/retrained/gru.pt" \
  --continuous "$work/continuous/gru.pt" \
  --specialists data/external_possession/gru-dead-ball/specialists.joblib \
  --device cuda --output "$work/final"
python evaluation/validate_submission.py "$work/final/candidate/possession.csv"
python evaluation/run_evaluation.py --gt public --results "$work/final/candidate" \
  --out "$work/final/public-check" --no-detail --bootstrap 0
```

Omit `--continuous` in the comparison to reproduce the coordinate repair
ablation on its own. The run saves the aggregate results and selected
prediction CSV below its output path in the ignored data directory.
