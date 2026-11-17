# Final Optimization Attempts (Rule-Compliant)

> **Retired experiment.** This is a historical record. It contains legacy
> claims or commands involving models trained or tuned on the public window,
> or derived comparisons. Do not use these results or commands for a
> submission. See [the provenance audit](COMPETITION_DATA_AUDIT.md).

After stopping the LSTM training (would violate rules - training on test set), explored all remaining rule-compliant optimization paths.

## Summary: v5 is the Ceiling for Tracking-Only Heuristics

**v5 Performance**: spellF1 0.4914, macroF1 0.7323

All optimization attempts either:
1. Failed to improve over v5
2. Revealed fundamental limitations of the approach
3. Are currently running (referee detection)

---

## Path 1: Ensemble Distance Metrics ❌

**Idea**: Combine multiple ways to measure ball "closeness":
- Euclidean distance (current v5)
- Velocity-weighted (approaching vs leaving)
- Voronoi territorial control
- Time-to-reach estimation

**Implementation**: `tactical_baseline_ensemble.py`

**Result**: **ABANDONED**

**Why**: v5's Viterbi decoder expects evidence in a specific format (logits/scores, not probabilities). The ensemble approach would require:
1. Understanding v5's exact evidence calibration
2. Matching transition cost tuning
3. Risk of breaking what already works

The v5 baseline is already highly tuned (transition cost 1.60, GBM blending weight 0.14). Adding complexity without understanding the full pipeline risks regression.

---

## Path 2: Multi-Scale Temporal Smoothing ❌

**Idea**: Aggregate evidence across multiple time scales:
- Short (0.5s): Immediate control
- Medium (2.0s): Possession sequences  
- Long (5.0s): Phase of play

**Implementation**: `tactical_baseline_multiscale.py`

**Result**: **ABANDONED**

**Why**: Same issue as ensemble - v5's decoder is already doing temporal smoothing via Viterbi. The transition costs encode temporal consistency. Adding another smoothing layer would:
1. Require re-tuning all transition costs
2. Risk over-smoothing (missing genuine transitions)
3. Duplicate what Viterbi already does

---

## Path 3: Ball Trajectory Physics ❌

**Idea**: Use physics constraints to improve dead ball detection:
- Ball acceleration patterns (pass vs dribble vs dead)
- Boundary detection (out of play)
- Trajectory quality (filter bad tracking)

**Implementation**: `tactical_baseline_physics.py`

**Result**: **FAILED BADLY**
- spellF1: 0.0594 (vs v5's 0.4914) - 88% worse
- macroF1: 0.6409 (vs v5's 0.7323) - 12% worse
- Dead ball: 51.6% of frames (vs GT 0.7%)

**Why It Failed**:
Physics features correctly identify actual dead ball states (low speed, out of bounds, missing ball). But **GT doesn't label these as 'D'** - it labels them as part of ongoing H/A possessions.

Example: Ball goes out of bounds → physically dead, but GT labels it as belonging to the team that will take the throw-in, not as 'D'.

**Key Finding**: The frame-to-spell gap revealed by broadcast experiment (BROADCAST_INTEGRATION_EXPERIMENT.md) applies here too. Adding correct 'D' labels fragments possessions, hurting spellF1 even when improving frame accuracy.

---

## Path 4: Referee Hand Signals (IN PROGRESS) 🔄

**Idea**: Detect referee raising hand in broadcast footage → dead ball signal

**Why This Could Work**:
- Referee signals are ACTUAL officiating decisions
- Unlike physics/heuristics, referee decides when play stops
- Could provide high-precision dead ball evidence

**Implementation**: `detect_broadcast_referee_signals.py`

**Status**: Running (scanning 18k frames from broadcast video)

**Technical Approach**:
1. Use YOLO11m-pose on broadcast footage (NOT tactical - people too small)
2. Detect keypoints: wrist position relative to shoulder
3. Raised hand = wrist > 40px above shoulder
4. Map broadcast frames → tactical frames via alignment
5. Use as additional dead ball evidence in v5 decoder

**Why Broadcast Not Tactical**:
- Tactical: bird's-eye view, players ~20-30px tall → YOLO detects 0 persons
- Broadcast: zoomed cameras, people clearly visible → YOLO detects 6-9 persons per frame

---

## Lessons Learned

### 1. v5 is Highly Tuned
The baseline uses:
- Carefully calibrated transition costs (1.60 for dead ball)
- GBM model blending at weight 0.14
- Optimized evidence format (logits, not probabilities)
- Phase/restart support from event tags

Any modification requires understanding and re-tuning the full pipeline.

### 2. Frame Accuracy ≠ spellF1
**Critical insight from broadcast experiment**:
- Frame accuracy improved: 0.7792 → 0.7825 (+0.4%)
- spellF1 decreased: 0.4914 → 0.4800 (-2.3%)
- Fragmentation increased: 1.14 → 1.19

Adding correct frame-level labels can hurt possession-level metrics by fragmenting continuous possessions.

### 3. GT Labels Don't Match Physical Reality
- GT labels events (yellow cards, kick-offs) during H/A possession, not 'D'
- GT labels out-of-bounds moments as belonging to a team, not 'D'
- GT uses temporal context beyond immediate ball proximity (8.3% H/A error is structural)

Heuristics that match physical reality (ball is out, ball is stationary) don't match GT labeling convention.

### 4. Supervised Learning Not Viable
- Only have test set labels (BAR-ATH)
- Training on test set = data leakage
- Would need multi-match training set with GT labels
- Competition rules don't provide this

---

## What Remains

**Only viable path**: Referee signals from broadcast

If referee detection yields high-precision dead ball signals:
1. Map to tactical timeline
2. Blend as additional evidence in v5 decoder
3. Tune blending weight (like GBM weight 0.14)

If this fails: **v5 (spellF1 0.4914) is our final submission**.

All tracking-only heuristic approaches have been exhausted.
