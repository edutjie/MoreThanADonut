# Possession improvement, round two

This round repeats the strongest [round-one](POSSESSION_EXECUTION.md) TCN
with five seeds and tests whether averaging or changing the decoder recovers
short spells. The released competition labels are reserved for one automated
correctness check after external selection and full-match replay.

## Protocol

- Fixed whole-match splits from round one. [SkillCorner](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656)
  supplies 12/4/4 A-League matches under MIT.
  [IDSSE](IDSSE_TRAINING.md) supplies 4/1/2 Bundesliga matches under CC BY 4.0.
  The six holdouts are the same matches studied in round one.
- Retrained TCN on A-League + IDSSE, TCN on A-League, and matched GRU on
  A-League + IDSSE with seeds 413, 733, 1049, 1493 and 2027. All use 113
  continuous-track features, repaired player tracks, and the fixed round-one
  natural plus random-corruption training mix. Each seed's epoch was selected
  on the equally weighted natural, weak and strong development conditions.
- Adoption requires a natural development gain of at least 0.005 spellF1
  above one paired seed SD, then an equally weighted held-out gain of at
  least 0.005, improvement on at least four of six matches, no A-League
  drop over 0.003 and no in-pitch dead-ball F1 drop over 0.005. Match-paired
  percentile intervals use 10,000 resamples. Each match contributes the mean
  of its halves. The original frozen GRU is kept in the family comparison.
- The best development ensemble was locked before holdout scoring. The
  ensemble itself has a single deployed score. Its development gate
  compares that score with each of the five matched control seeds. For
  decoder changes we also measured the paired effect on the five individual
  checkpoint streams, preserving the same native frame grid.
- The duration model uses training-only spell lengths and an exponential
  tail beyond 30 seconds. The touch head uses local ball acceleration and
  direction, nearest player distance and its local change, and the fixed
  original GRU's change in team or dead-ball evidence. The fit uses only
  external training transitions. Snapping preserves each decoded segment.
- Target normalization blends checkpoint feature statistics with the target
  match's unlabeled statistics. Per-example LayerNorm has no running
  population statistics. If normalization misses the dev gate, head-only
  self-training uses 16 steps of 50/50, independently normalized loss on
  model pseudo-labels and labeled external training data. Each target
  resets to the original seed checkpoint. Target labels are reserved for scoring.

## External results

Natural-input spellF1, with weak and strong corruption shown separately.
The frozen GRU replay matched round one's predictions on all 12 held-out
halves. The adoption threshold uses natural observations.

| Shared classifier | Dev mean ± seed SD | Held-out mean ± seed SD | A-League | Bundesliga | Weak | Strong | In-pitch D F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen continuous GRU | 0.6547 | 0.6220 | 0.6748 | 0.5163 | 0.5610 | 0.5719 | 0.8978 |
| TCN, A-League | 0.6573 ± 0.0036 | 0.6221 ± 0.0047 | 0.6753 | 0.5157 | 0.5679 | 0.5806 | 0.8945 |
| GRU, A-League + IDSSE | 0.6655 ± 0.0064 | 0.6309 ± 0.0063 | 0.6742 | 0.5444 | 0.5741 | 0.5770 | 0.9095 |
| **TCN, A-League + IDSSE** | **0.6659 ± 0.0038** | **0.6366 ± 0.0028** | **0.6803** | **0.5491** | **0.5800** | **0.5903** | 0.9080 |

The mixed-data TCN passed both gates against the frozen GRU. Its held-out
gain was 0.0146 (95% paired match bootstrap interval 0.0037 to 0.0270), with
five matches improving. A-League gained 0.0055, and in-pitch D F1 gained
0.0102. Most of the gain against the frozen GRU came from the two
Bundesliga matches. The matched TCN comparison isolates the training mix on a fixed
backbone and gained 0.0145 held out, with 0.0050 on A-League and 0.0334 on
Bundesliga. The TCN-versus-GRU comparison on the mixed data gained 0.0056
held out, while its development advantage was only 0.0004. The evidence
supports the mixed-data TCN recipe as a package. It gives a weaker case
for attributing the gain to architecture alone.

| Held-out match | Frozen GRU | Mixed TCN mean | TCN seed 733 | GRU–TCN ensemble |
| --- | ---: | ---: | ---: | ---: |
| A-League 1925299 | 0.6554 | 0.6698 | 0.6622 | 0.6679 |
| A-League 1996435 | 0.7119 | 0.7086 | 0.7043 | 0.7234 |
| A-League 2007448 | 0.6949 | 0.6989 | 0.6926 | 0.6987 |
| A-League 2017461 | 0.6371 | 0.6439 | 0.6423 | 0.6491 |
| Bundesliga J03WOY | 0.5120 | 0.5355 | 0.5367 | 0.5361 |
| Bundesliga J03WR9 | 0.5205 | 0.5628 | 0.5695 | 0.5626 |

The dev-selected ensemble averaged log probabilities from five mixed-data
GRUs and five mixed-data TCNs, with equal weight per family. It reached
0.6758 on dev and 0.6396 held out. Its 0.0031 held-out gain over the mixed
TCN seed mean fell short of 0.005, and three of six matches improved. The
dev-selected individual checkpoint, TCN seed 733, reached 0.6705 on dev
and 0.6346 held out. Against the frozen GRU it gained 0.0126 spellF1 on four
of six matches, with 0.0005 on A-League and 0.0111 in in-pitch D F1. This is
the single checkpoint used for full-match inference.

### Decoder and adaptation

The table follows the dev-leading ensemble through each dev-selected arm.
The same comparisons were also run on seed 733 and on the frozen GRU.
The local tables preserve every half and five-checkpoint decoder comparison.

| Arm | Setting | Dev spellF1 | Held-out spellF1 | Missed 2–4 s spells | Matched typed-boundary MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed Viterbi | Turnover 5, stoppage 10 | 0.6758 | 0.6396 | 252 | 0.177 s |
| Re-selected costs | Turnover 1, stoppage 2 | 0.6786 | 0.6452 | 230 | 0.179 s |
| Explicit durations | Strength 0.5 | 0.6771 | 0.6391 | 252 | 0.177 s |
| Boundary snapping | ±5 native frames | 0.6762 | 0.6374 | 252 | 0.134 s |
| Target normalization | Blend 0.25 | 0.6724 | 0.6394 | 253 | 0.176 s |
| Target self-training | Confidence 0.95 | 0.6758 | 0.6382 | 254 | 0.178 s |

Re-selected costs recovered 22 short spells on the ensemble, while the
selected pair sat on the grid edge and missed the minimum development
gain. On seed 733 the interior pair (2.5, 5) recovered 18 and gained 0.0083
on the holdouts, but gained only 0.0044 on dev. Snapping reduced timing
error on matched transitions and reduced spellF1. No decoder or adaptation
arm cleared both gates. The duration and snapping arms therefore stayed
separate. Target adaptation was tested across the repeated lead checkpoints;
the original frozen GRU's self-training fallback was an additional
one-checkpoint diagnostic. The optional observation-matching stage was
skipped after confirmation of the mixed-data TCN. Given round one's
natural-input regression from synthetic missingness, this round prioritized
the confirmed model and segmentation evidence.

## Competition pipeline check

The frozen recipe uses mixed-data TCN seed 733, repaired tactical player
tracks, the original 113 features and the fixed (5, 10) Viterbi decoder.
It uses tournament tracking for inference and adds no target adaptation.
Two independent runs produced byte-identical CSVs. The official validator
accepted all 199,319 rows, frames 0 through 199,318.

| Scope | SpellF1 | Macro F1 | Dead-ball F1 |
| --- | ---: | ---: | ---: |
| External holdouts, frozen seed 733 | 0.6346 | 0.9095 | 0.9605 |
| Official public check, 7,500 frames | 0.6800 | 0.8576 | 0.7697 |
| Official private score | Unknown | Unknown | Unknown |

The only round-two public scorer run followed the external decision and
full-grid replay. The historical frozen GRU public spellF1 from [round one](POSSESSION_EXECUTION.md)
was 0.6897. The original GRU stays available as the reference, and the
round-two TCN remains the externally selected candidate.

- Checkpoint SHA-256 `ed33dc6b97faf911b18a09875ff873957eccca14553c15018c38bb609ec80ca3`
- CSV SHA-256 `839a01f16ae51416c484e6b80dcd3d04664ccbf7444075b682c1e6523d6e6938`

## Reproduce

Use the pinned round-one local assets and observation caches described in
the [round-one handoff](../../handoffs/20261117T072920Z-edutjie-codex-possession-execution.md).
The decoder builds a local shared library using `cc`. The model runtime
uses PyTorch with CUDA, NumPy, SciPy and scikit-learn. The output directory
is ignored by Git.

```bash
src=/home/xuheping/codes/MoreThanADonut
py="$src/.venv/bin/python"
previous="$src/data/external_possession/execution-20261117"
work="$src/data/external_possession/round2-20261117"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
shared=(--output "$work" --skillcorner "$src/external_data/skillcorner-4340d27"
        --gru "$src/data/external_possession/player-quality/continuous/gru.pt"
        --observations "$previous/observation")
"$py" -m experiments.possession_round2_train "${shared[@]}"
"$py" -m experiments.possession_round2_verify "$work"
for stage in dev test; do
    "$py" -m experiments.possession_round2 "$stage" "${shared[@]}" \
        --idsse "$src/external_data/idsse" --round-one "$previous"
done
for destination in frozen replay; do
    "$py" -m experiments.possession_round2_predict \
        --selection "$work/decision-before-public.json" --data-dir "$src/data" \
        --observations "$previous/observation" --output "$work/$destination/possession.csv"
done
cmp "$work/frozen/possession.csv" "$work/replay/possession.csv"
"$py" evaluation/validate_submission.py "$work/frozen/possession.csv"
"$py" evaluation/run_evaluation.py --gt public --results "$work/frozen" \
    --out "$work/public-check" --no-detail --bootstrap 0
```

`run-index.json` records settings and hashes. `development-freeze.json` records
the dev-only choices and freezes the model fingerprints before holdout
confirmation. `holdout-confirmation.json` records each natural and stressed
family result by seed and half, plus the preselected ensemble, decoder and
adaptation comparisons. `natural-by-match.csv` is the flat per-match export.
`decision-before-public.json` identifies the one
recipe used to generate the full official CSV.
