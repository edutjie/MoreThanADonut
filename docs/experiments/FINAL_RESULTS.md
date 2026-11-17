# Final Results: Possession Prediction Optimization

> **Retired experiment.** This is a historical record. It contains legacy
> claims or commands involving models trained or tuned on the public window,
> or derived comparisons. Do not use these results or commands for a
> submission. See [the provenance audit](COMPETITION_DATA_AUDIT.md).

**Date**: 2026-11-16  
**Status**: Historical v5 experiment report. See [the external possession
experiment](EXTERNAL_POSSESSION.md) for the current candidate and verification.

---

## v5 Reference

**v5 Baseline** was the candidate at the end of this experiment. Commit
`cad89df` records comparing transition costs against the public window.
[The rules](../../RULES.md) reserve that window for pipeline checks. The external
experiment uses v5 as a comparison reference.

Historical public results:
- **spellF1**: 0.4914
- **macroF1**: 0.7323
- **Beats competition baseline**: Yes (0.4914 > 0.4858)

---

## All Approaches Tested

### Successful Improvements (Integrated in v5)

1. ✅ **Transition Cost Tuning** (0.85 → 1.60)
   - spellF1: 0.4858 → 0.4914 (+1.2%)
   - Reduces fragmentation while maintaining accuracy

### External Signal Explorations (All Failed)

2. ❌ **SN-BAS-2025 Event Labels**
   - Result: No improvement (tied with v5)
   - Reason: Event mismatch - GT labels events during H/A possession, not as separate 'D'
   - Only 0.7% GT frames are 'D', but 78% of events suggest dead ball
   - Documentation: Row 8 in EXPERIMENT_FINDINGS.md

3. ❌ **CALF Event Detection**
   - Result: No improvement
   - Reason: Only 13.6% alignment with GT dead ball
   - Same issue as SN-BAS-2025
   - Documentation: CALF_INTEGRATION_EXPERIMENT.md

4. ⚠️ **Broadcast Missing Ball** [BEST EXTERNAL SIGNAL]
   - Result: spellF1 0.4800 (-2.3%), macroF1 0.7825 (+6.8%)
   - Recall: 39.8%, Precision: 54.6%
   - **Frame accuracy UP but spellF1 DOWN** due to fragmentation (1.14 → 1.19)
   - Key finding: Adding correct 'D' labels fragments possessions
   - Documentation: BROADCAST_INTEGRATION_EXPERIMENT.md

### Heuristic Refinements (All Failed)

5. ❌ **H/A Attribution Analysis**
   - Result: 8.3% error is structural
   - Reason: GT uses temporal context beyond ball proximity
   - Cannot fix with distance-based heuristics
   - Documentation: HA_ATTRIBUTION_ANALYSIS.md

6. ❌ **Ball Trajectory Physics**
   - Result: **CATASTROPHIC** - spellF1 0.0594 (-88%), macroF1 0.6409 (-12%)
   - Reason: Physics correctly identifies dead ball (stopped, out of bounds) but GT labels these as H/A
   - Over-detects dead ball: 51.6% vs GT 0.7%
   - Same fragmentation issue as broadcast
   - Documentation: FINAL_OPTIMIZATION_ATTEMPTS.md

7. ❌ **Ensemble Distance Metrics**
   - Status: Abandoned before testing
   - Reason: Incompatible with v5's evidence format
   - Would require re-tuning entire pipeline

8. ❌ **Multi-Scale Temporal Smoothing**
   - Status: Abandoned before testing
   - Reason: Viterbi decoder already does temporal smoothing via transition costs
   - Would duplicate existing functionality

9. ❌ **Referee Hand Signals**
   - Result: **54 detections, 0% useful**
   - All signals in unlabeled ('X') regions of GT
   - GT is only 3.76% labeled (7,500 / 199,319 frames)
   - 0% overlap with GT 'D' labels
   - 0% overlap with GT H/A labels
   - Cannot validate or learn from signals in unlabeled data

### Supervised Learning (Blocked)

10. 🚫 **LSTM Possession Model**
    - Status: Stopped before training
    - Reason: **Would violate competition rules**
    - Only have test set labels (BAR-ATH)
    - Training on test set = data leakage
    - Would need multi-match training data (not provided)

---

## Key Findings

### 1. Frame-to-Spell Gap

**Definition**: Improving frame-level accuracy can hurt possession-level spellF1.

**Evidence**:
- Broadcast experiment: frame accuracy +0.4% → spellF1 -2.3%
- Physics experiment: massively over-detected D → spellF1 -88%
- Root cause: Adding 'D' labels fragments continuous H/A possessions

**Formula**: `Fragmentation = n_predicted_spells / n_gt_spells`
- v5: 1.14 (19 predicted vs 17 GT)
- Broadcast: 1.19 (20 predicted)
- Physics: 4.95 (84 predicted)

### 2. GT Labels ≠ Physical Reality

**Examples**:
- Ball out of bounds → GT labels as H or A (team taking throw-in), not D
- Yellow cards, kick-offs → GT labels during H/A possession, not separate D
- Stationary ball → Often labeled H/A, not D

**Implication**: Heuristics matching physical reality don't match GT labeling convention.

### 3. GT Labeling Uses Temporal Context

**Evidence from H/A analysis**:
- 8.3% H/A error rate is structural
- When distance logic "disagrees" with GT, it's correct 31-68% of time
- GT considers broader game context, not just ball proximity

**Implication**: Frame-level heuristics can't capture this without larger temporal window.

### 4. v5 is Highly Tuned

The baseline includes:
- Transition cost: 1.60 (precisely tuned for fragmentation vs accuracy)
- GBM blending: 0.14 weight (trained on external data)
- Evidence calibration: specific logit scaling
- Phase/restart support: from event tags

**Implication**: Modifications risk regression without understanding full pipeline.

### 5. Limited Labeled Data

**GT Statistics**:
- Total frames: 199,319
- Labeled: 7,500 (3.76%)
- Unlabeled ('X'): 191,819 (96.24%)
- H: 3,454 (1.73%)
- A: 2,577 (1.29%)
- D: 1,469 (0.74%)

**Implication**: 
- External signals often fall in unlabeled regions (referee: 100%)
- Can't validate improvements on unlabeled data
- Severely limits what can be learned from single-match test set

---

## Historical conclusions

The tested event and heuristic variants frequently fragmented possessions or
disagreed with the label conventions at stoppages. The public window was too
small to assess behavior over the rest of the match. The subsequent [external
possession experiment](EXTERNAL_POSSESSION.md) uses other matches for selection
and testing.
