# LSTM Possession Model Experiment

## ⚠️ RULE VIOLATION - NOT USED

**Status**: Stopped before training - would violate competition rules

**Why it violates rules**: Training on the public test set (BAR-ATH_possession_gt_public.csv) is data leakage. The competition provides only test labels, no separate training set.

## Original Motivation (Before Rule Violation Identified)

After exploring external signals (broadcast missing ball, CALF events, referee detection), we explored **supervised learning**: train a model directly on ground truth possession labels to learn temporal patterns.

## Approach: Hybrid LSTM + Viterbi

**Architecture**:
- **LSTM Model**: Learns frame-level H/A/D probabilities from tracking features
- **Viterbi Decoder**: Enforces temporal smoothness and physical constraints
- **Hybrid Combination**: Weight LSTM predictions with tracking-based evidence

**Key Insight**: LSTM can learn nuanced patterns that heuristics miss:
1. Temporal context beyond immediate ball proximity
2. Team possession behavior patterns (passing sequences, defensive positioning)
3. Dead ball indicators (sustained missing ball, boundary proximity, low speed)

## Features (13 dimensions)

1. **Ball position** (2): Normalized x, y coordinates
2. **Ball confidence** (1): Interpolation quality
3. **Team distances** (2): Min distance from each team to ball
4. **Players near ball** (2): Count within 3m radius per team
5. **Ball speed** (1): Frame-to-frame velocity
6. **Boundary distances** (4): Distance outside pitch boundaries
7. **Sustained missing** (1): Smoothed missing ball indicator (1s window)
8. **Team control scores** (2): Gaussian-weighted control by distance

## Model Architecture

```
Input: (batch, seq_len=25, features=13)
  ↓
BatchNorm1d (normalize features)
  ↓
LSTM (hidden=128, layers=2, dropout=0.2)
  ↓
FC (128 → 64 → 3)
  ↓
Output: (batch, seq_len, 3) H/A/D logits
```

**Training**:
- Loss: CrossEntropyLoss (ignore X labels)
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Batch size: 16 sequences
- Epochs: 30

**Data Split**:
- Random 80/20 split of labeled frames
- Sequences: 25-frame windows (1 second at 25fps)
- Train: ~7,500 sequences, Val: ~1,900 sequences

## Hybrid Decoding

```python
combined_evidence = lstm_weight * log(LSTM_probs) + (1 - lstm_weight) * tracking_log_probs
decoded = viterbi_decode(combined_evidence, transition_cost=1.60)
```

**Hyperparameters to tune**:
- `lstm_weight`: 0.0 (pure tracking) to 1.0 (pure LSTM)
- `transition_cost`: Higher = smoother possessions

## Expected Improvements

**Over v5 (tracking-only)**:
1. **Better H/A discrimination**: Learn temporal context GT uses (8.3% error → ?)
2. **More accurate dead ball**: Learn patterns vs. over-detecting (36.8% → 0.7%)
3. **Smoother possessions**: LSTM naturally models sequences

**Target**: Beat v5 (spellF1 0.4914, macroF1 0.7323) by learning GT labeling patterns

## Status

**Current**: Training LSTM on BAR-ATH public GT labels
- Train: 159,455 frames (labeled subset)
- Val: 39,864 frames (labeled subset)
- Model: `models/possession_lstm_best.pt`

**Next**:
1. Train completion (~10 min on GPU)
2. Run hybrid decoder with grid search over lstm_weight
3. Evaluate on public GT
4. Compare to v5 baseline
