# Bidirectional GRU with explicit tracking gaps

This experiment compares a small bidirectional GRU with the
[frozen lagged A-League LightGBM](EXTERNAL_POSSESSION.md). Both use the
same A-League development and held-out match splits and the same fixed
Viterbi turnover and stoppage costs, 5 and 10. No YOLO recoveries, Smart
Tags or contest labels enter training or checkpoint selection.

## Method

- Use the 111 existing tracking features and add a raw ball-visibility
  flag and log seconds since the last observed ball. Carry missing feature
  flags alongside standardized values. The training split provides the
  normalization statistics.
- Fit on contiguous 51.2-second windows from the same 12 A-League matches.
  Include the same short simulated ball dropouts as LightGBM and randomly
  mirror teams. A single bidirectional GRU with 48 units per direction
  outputs H/A/D probabilities. Skip unknown-label frames in the loss.
- Train eight epochs with AdamW and select the checkpoint by development
  spellF1. At inference, average normal and team-mirrored outputs. Overlap
  windows with up to eight seconds of context on either side. The recurrent
  model runs at 10 Hz. For 25 Hz tracking, interpolate probabilities back
  to the original frame grid before Viterbi decoding.

## Evaluation

The primary test is the original four held-out A-League matches. Additional
tests add label-independent gaps to those matches and to the two masked
Bundesliga test games used in [IDSSE Training](IDSSE_TRAINING.md). A gap
stress test adds missing runs averaging 12 seconds at 60% expected visibility
before any naturally missing observations.

Gap-spell recall counts true H/A intervals with at least one continuous
second of missing ball tracking. An interval counts as recovered if the
official same-team one-to-one matcher finds it at temporal IoU 0.5.

## Results

Eight epochs ran on 1,426,780 sequence frames, including the augmented
copies. Epoch 8 was frozen on A-League development, where spellF1 was
0.7064 versus 0.6938 for LightGBM.

| Held-out test | LightGBM spellF1 | GRU spellF1 | LightGBM gap spells matched | GRU gap spells matched |
| --- | ---: | ---: | ---: | ---: |
| A-League, original tracking | 0.6616 | 0.6728 | 11 / 30 | 14 / 30 |
| A-League, added long gaps | 0.5540 | 0.5909 | 337 / 590 | 317 / 590 |
| Bundesliga, masked tracking | 0.4369 | 0.4376 | 164 / 379 | 147 / 379 |

The GRU's overall spellF1 improved under the synthetic gap stress tests,
while its recall on the gap-overlapping subset declined. This is evidence
for sequence modeling overall, with recovery of sustained-gap possessions
still a separate weakness.

| Public contest check | SpellF1 | MacroF1 | Dead-ball F1 |
| --- | ---: | ---: | ---: |
| Frozen LightGBM | 0.6452 | 0.8084 | 0.6861 |
| Frozen GRU | 0.6621 | 0.8894 | 0.8946 |

The GRU produced 12 possession spells in the public window, versus 14 for
LightGBM and 17 labeled spells. Its CSV passed validation for all 199,319
frames, and a second inference from the checkpoint reproduced it exactly.
The private full-match score is unknown.

## Reproduce

```bash
work=data/external_possession/sequence-gru
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python sequence_possession.py train \
  --skillcorner data/skillcorner-4340d27 --idsse data/idsse \
  --reference data/external_possession/lightgbm-lagged-4340d27.joblib \
  --device cuda --workers 4 --output "$work"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python sequence_possession.py predict \
  --model "$work/gru.pt" --device cuda --output "$work/candidate/possession.csv"
python evaluation/validate_submission.py "$work/candidate/possession.csv"
python evaluation/run_evaluation.py --gt public --results "$work/candidate" \
  --out "$work/public-check" --no-detail --bootstrap 0
```

The experiment uses PyTorch 2.15.0+cu132. The checkpoint, predictions and
full report are local artifacts under ignored `data/`.

## Bidirectional Transformer comparison

The Transformer uses the same 113 tracking inputs, missing-value flags, 10 Hz
sequence grid, 512-frame windows, 80-frame inference context, training
copies, splits and decoder. It uses four encoder layers with four attention
heads each, 64 hidden units, a 128-unit feed-forward block, dropout of 0.15,
and sinusoidal position encoding. Attention sees both directions within the
window. Only padded frames are masked. Real frames with missing ball
coordinates or unknown training labels remain visible to attention.

Train with the same eight epochs and choose the checkpoint on A-League
development spellF1. The CUDA training run uses bfloat16 autocast; inference
uses float32.

The Transformer froze at epoch 8 with 0.6813 development spellF1, versus
0.7064 for the GRU and 0.6938 for LightGBM.

| Held-out test | LightGBM spellF1 | GRU spellF1 | Transformer spellF1 | Matched gap spells (LightGBM / GRU / Transformer) |
| --- | ---: | ---: | ---: | ---: |
| A-League, original tracking | 0.6616 | **0.6728** | 0.6482 | 11 / 14 / 13 of 30 |
| A-League, added long gaps | 0.5540 | **0.5909** | 0.5805 | 337 / 317 / 321 of 590 |
| Bundesliga, masked tracking | 0.4369 | **0.4376** | 0.4215 | 164 / 147 / 145 of 379 |

The Transformer matched four additional gap spells relative to the GRU
on stressed A-League tracking. Its total spellF1 was lower than the GRU
on all three held-out tests.

| Public contest check | SpellF1 | MacroF1 | Dead-ball F1 |
| --- | ---: | ---: | ---: |
| Frozen LightGBM | 0.6452 | 0.8084 | 0.6861 |
| Frozen GRU | **0.6621** | **0.8894** | **0.8946** |
| Frozen Transformer | 0.5000 | 0.8887 | 0.8408 |

At temporal IoU 0.5, the Transformer matched 7 of the 17 public possessions
and the GRU matched 10. Both froze before the contest check. The Transformer
CSV was valid over all 199,319 frames and repeated inference reproduced it
byte for byte. The GRU checkpoint also reproduced its original CSV through
the shared sequence entry point.

```bash
work=data/external_possession/sequence-transformer
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python sequence_possession.py train \
  --architecture transformer --skillcorner data/skillcorner-4340d27 --idsse data/idsse \
  --reference data/external_possession/lightgbm-lagged-4340d27.joblib \
  --device cuda --workers 4 --output "$work"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python sequence_possession.py predict \
  --model "$work/transformer.pt" --device cuda --output "$work/candidate/possession.csv"
python evaluation/validate_submission.py "$work/candidate/possession.csv"
python evaluation/run_evaluation.py --gt public --results "$work/candidate" \
  --out "$work/public-check" --no-detail --bootstrap 0
```
