# Pre-kickoff dead ball

FAQ.md:38-40 says the second half is graded from frame 110,808 but does
not kick off until 113,742. Every one of those 2,934 graded frames is `D`.
No external training half contains pre-kickoff play, and both TCN
candidates predicted live possessions while the players warmed up:

| Candidate | H / A / D on 110,808–113,741 | False possessions |
| --- | --- | ---: |
| Original 113-feature TCN | 1,685 / 253 / 996 | 7 |
| Round-two mixed TCN, seed 733 | 1,801 / 411 / 722 | 8 |

## Method

[`kickoff.py`](../../kickoff.py) finds each half's kickoff from the provided
inputs and labels every graded frame before it `D`. With `--prekickoff`,
the round-two and study predict scripts also restart the Viterbi decoder
state at the kickoff. The sequence model and any interpolation still use
the whole half as context, so pre-kickoff tracking can still influence the
probabilities after the kickoff. The same step is available as a
post-processing CLI over any `frame,label` CSV.

1. The Smart Tag `Kick Off (Start)` instance for the half, padded by 2 s,
   bounds the search.
2. A candidate is a ball observation within the spot radius of the centre
   spot from which the ball travels 1.5 m within 0.6 s of its next
   observation. Within 1 s of getting there, it must not come back to
   either the candidate position or the spot.
3. The strike is the frame after the last observation, before that point,
   that is still at the median rest position. If the ball was last seen
   near the spot shortly before a tracking gap, the strike is placed right
   after that sighting.
4. Without such a candidate, the midpoint of the tag is used.

The rule writes `D` only, and only inside a graded half before its
kickoff. It never writes `X`, because the scorer forward-fills `X` with
the previous label. The published kickoff frames are correctness
references; they are not inputs.

## External validation

`experiments/kickoff_validation.py` scores the detector on restarts with
known frames.
- **IDSSE (7 games):** a BallStatus switch from dead to alive while the
  ball is inside the centre circle (9.15 m). This is the ground truth's own
  convention. The circle is far wider than any detector radius, so the
  reference does not depend on the radius being tested. It can include
  restarts other than kickoffs.
- **Metrica sample (games 1–2):** `KICK OFF` event frames.

Restarts with less than 2 s of earlier tracking are skipped, leaving 38:
26 from IDSSE and 12 from Metrica. At the restart frame, 27 have the ball
within 1 m of the spot, 2 within 1–2 m, 1 within 2–3 m, and 8 farther out.

Each restart gets a simulated 125-frame tag window at a random offset. Two
dropout conditions apply the published 41% missing-ball rate, with mean
gaps of 2 s and 12 s, over three seeds each.

A first design required the ball to rest on the spot before the strike.
It fired on only 4–8% of external kickoffs: IDSSE freezes the ball's last
live position during the stoppage, and Metrica leaves it untracked until
the kick. The departure rule above replaced it.

The spot radius was then chosen from a grid declared before the run. The
choice used IDSSE only, maximising the share of restarts within 2 frames,
with ties going to the smaller radius. The scores were 0.379, 0.396 and
0.418 for 1, 2 and 3 m, so 3 m was chosen. That sits at the edge of the
grid, and the score still rises with the radius.

The grid was not widened afterwards. The second-half ball on the provided
match was already known to stay 3.34 m from the spot, so any wider choice
would be contaminated. On Metrica, the held-out set, every radius scores
the same.

| Radius 3 m | Restarts | Seen by tracking | Within 2 frames | More than 2 frames late | Tag midpoint within 2 frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDSSE natural | 26 | 69% | 62% | 23% | 0% |
| Metrica natural | 12 | 100% | 100% | 0% | 0% |
| All natural | 38 | 79% | 74% | 16% | 0% |
| All, 2 s dropouts | 114 | 53% | 44% | 33% | 3% |
| All, 12 s dropouts | 114 | 46% | 42% | 28% | 3% |

For restarts with the ball within 3 m of the spot, 93% of natural-track
detections are within 2 frames, and about 52% under dropouts.

Tracking detections were never more than 2 frames early. Late tracking
detections were mostly 1 s or slightly more, when the first sighting failed
the no-return check and a later one passed. The tag midpoint's errors depend
on where the simulated window falls; no external source shows where real
tags place the kickoff.

## Provided match

| Half | Graded from | Detected kickoff | Method | Published reference | Graded frames set to `D` |
| --- | ---: | ---: | --- | ---: | ---: |
| 1 | 11,185 | 11,188 | tracking | 11,185 | 3 |
| 2 | 110,808 | 113,722 | tag | 113,742 | 2,914 |

The second-half tracked ball never comes within 3.34 m of the spot inside
the search window, so that half uses the tag midpoint, 20 frames before
the reference. The first-half detection is 3 frames late, which is
consistent with the external late tail. No threshold was changed in
response to either result.

| Candidate | Full-match possessions before → after | Public spellF1 | Public macro F1 before → after |
| --- | ---: | ---: | ---: |
| Original 113-feature TCN | 215 → 209 | 0.6815 (unchanged) | 0.9160 → 0.9155 |
| Round-two TCN, seed 733 | 231 → 224 | 0.6800 (unchanged) | 0.8576 → 0.8570 |

The public window covers only the first 3 changed frames. The second-half
change lies outside every published label, so its effect on the private
score is unmeasured.

Reproduction checks:
- With the flag off, both predict scripts reproduced the frozen CSVs byte
  for byte (`839a01f1…` and `ee7bd4f3…`).
- With the flag on, two runs matched each other and matched the
  post-processing CLI: `737cc552…` for the original TCN and `8cdce512…` for
  round two. Restarting the decoder therefore changed nothing on this match
  beyond the forced `D` frames.
- Both outputs passed `evaluation/validate_submission.py`.
- An independent review of the detector led to four fixes before these
  runs: rest jitter read as the strike, the no-return window timing, a
  truncated lookahead, and a validation reference tied to the radius grid.
  None of them changed the provided-match outputs.

The model's own short A spell from 113,722 to about 113,753 remains. It
straddles the reference kickoff, because the tag midpoint falls 20 frames
early.

## Reproduce

Point the commands at the match bundle and the frozen round-two and TCN
artifacts described in [TCN_AUTOML_FOUNDATION.md](TCN_AUTOML_FOUNDATION.md)
and the round-two report.

```bash
python -m unittest discover -s tests
python -m experiments.kickoff_validation --idsse external_data/idsse \
  --metrica external_data/metrica-sample --output "$work/validation"
python -m experiments.possession_round2_predict --selection "$round2/decision-before-public.json" \
  --data-dir data --observations "$execution/observation" --prekickoff --output "$work/round2/possession.csv"
PYTHONPATH="$tcn/automl-deps" python -m experiments.possession_study_predict \
  --candidate "$tcn/autotcn" --data-dir data --prekickoff --output "$work/origtcn/possession.csv"
python evaluation/validate_submission.py "$work/round2/possession.csv"
```

`MTAH_DATA_DIR=data python kickoff.py --input <csv> --output <csv>` applies
the same rule to an existing prediction.
