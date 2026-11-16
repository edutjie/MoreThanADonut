# Broadcast Missing Ball Integration Experiment

## Objective
Use extended broadcast ball missing periods (likely replays/highlights during dead ball) as additional dead ball signal to improve spellF1 beyond v5 (0.4914).

## Hypothesis
When broadcast ball is missing for extended periods (>2s), it indicates replays/highlights being shown during actual game dead ball moments. This signal should align better with GT dead ball labels than external event detectors like CALF.

## Implementation

### Step 1: Extract Broadcast Missing Periods
Created `extract_broadcast_missing.py`:
- Loads broadcast tracking (164,488 frames)
- Identifies ball missing runs: 49.5% of frames missing
- Filters to long runs (>2s): 132 runs

### Step 2: Align Broadcast to Tactical Timeline
Used `broadcast_alignment.py`:
- Synchronizes broadcast and tactical feeds via ball path matching
- Maps broadcast frames to tactical timeline
- Result: 118 tactical runs from 132 broadcast missing periods

### Step 3: Generate Dead Ball Evidence
- Creates frame-level evidence with ±1s window and exponential decay
- Evidence strength: 0.25 within missing period, decays at edges
- Applied to 52,717 tactical frames

### Step 4: Integrate into Viterbi Decoder
Created `tactical_baseline_broadcast.py`:
- Adds broadcast evidence to dead ball channel: `evidence[:, 2] += weight * broadcast_evidence`
- Tested weights: 0.10, 0.12, 0.15, 0.18, 0.20

## Results

### Alignment with GT
**Broadcast missing periods vs GT dead ball:**
- GT dead ball: 1,469 frames (19.6% of labeled window)
- Broadcast covers: 1,069 frames
- Overlap: 584 frames
- **Recall: 39.8%** (584 / 1,469) — captures 40% of GT dead ball
- **Precision: 54.6%** (584 / 1,069) — when broadcast missing, 55% chance it's GT dead

Much better than CALF (13.6% overlap) or SN-BAS-2025 event patterns!

### Performance by Weight

| Weight | Frame Acc | Changes | H→D correct | A→D wrong | SpellF1 |
|--------|-----------|---------|-------------|-----------|---------|
| 0.10   | 0.7704    | 70      | 2           | 68        | ?       |
| 0.12   | 0.7817    | 157     | 88          | 69        | ?       |
| 0.15   | 0.7823    | 161     | 92          | 69        | ?       |
| 0.18   | **0.7825** | 165    | **95**      | 70        | **0.4800** |
| 0.20   | 0.7825    | 167     | 96          | 71        | ?       |

**Best: weight 0.18**
- Frame accuracy: 0.7792 → 0.7825 (+0.33%, +25 net correct frames)
- Fixed 95 H→D errors, created 70 A→D errors

### Ranking (evaluation/run_evaluation.py)

```
rank  submission                macroF1  spellF1   poss     dur  frag
----  ------------------------  -------  -------  -----  ------  ----
   1  improved_dead_ball_v5      0.7323   0.4914     18    14.2  1.14
   3  improved_dead_ball_v7_b~   0.7421   0.4800     18    13.8  1.19
```

**v7 ranks #3 but with lower spellF1 than v5:**
- SpellF1: 0.4914 → 0.4800 (-2.3%)
- Frame accuracy: 0.7792 → 0.7823 (+0.4%)
- Fragmentation: 1.14 → 1.19 (+4.4%)
- Spell duration: 14.2s → 13.8s (-2.8%)

## Why Frame Accuracy Doesn't Improve SpellF1

**The fragmentation problem**: Adding D labels breaks up H/A possessions into shorter segments. SpellF1 penalizes this because:
1. More segment boundaries = more opportunities for boundary errors
2. Shorter segments = harder to achieve high IoU thresholds (0.3-0.7)
3. SpellF1 counts each possession once, not frames — breaking one long spell into two short ones doesn't help

**Example**:
- v5: [HHHHHHHHHHH] → one 11-frame spell, IoU with GT can be high
- v7: [HHHHDDHHHH] → two 4-frame spells, both harder to match GT boundaries

Even though v7 correctly labels the D frames (improving frame accuracy), the fragmentation hurts spell-level metrics.

### Conservative Approach (v8)
Tried limiting broadcast evidence to frames where tracking already suggests dead ball (`threshold` parameter):

```python
mask = base_evidence[:, 2] >= threshold
evidence[:, 2] += weight * broadcast_evidence * mask
```

**Result**: All thresholds (0.10-0.25) produced identical spellF1 = 0.4914 (same as v5). The conservative approach prevents fragmentation but also prevents any improvement.

## Error Analysis

### Coverage of v5 Errors
**H→D errors** (v5=H, GT=D): 612 frames
- Broadcast covers: 272 (44.4%)
- Not covered: 340 (55.6%)

**A→D errors** (v5=A, GT=D): 135 frames
- Broadcast covers: 12 (8.9%)
- Not covered: 123 (91.1%)

Broadcast signal is biased toward H frames, explaining why v7 fixes H→D but creates A→D errors.

### Accuracy at v5's D Predictions
When v5 predicts D:
- Broadcast missing: 60.8% correct (393/646)
- Broadcast NOT missing: 68.3% correct (329/482)

**Surprising**: Broadcast missing doesn't help distinguish true from false D labels. In fact, v5's D predictions are MORE accurate when broadcast is NOT missing.

## Root Cause

Same structural issue as CALF and SN-BAS-2025:

**Broadcast philosophy**: Missing ball = replay/highlight being shown, game is paused
**GT philosophy**: Replay timing doesn't perfectly align with GT's dead ball definition

Broadcast replays often start during possession (showing the play that led to the stoppage) or extend past the restart. GT labels the actual ball state, not what TV is showing.

## Conclusion

**Broadcast missing ball signal is the best external signal we've tested** (39.8% recall vs CALF's 13.6%), but it still:
1. ✗ Reduces spellF1 despite improving frame accuracy
2. ✗ Causes possession fragmentation (1.14 → 1.19)
3. ✗ Creates A→D errors while fixing H→D errors

**Frame accuracy and spellF1 measure different things**:
- Frame accuracy: per-frame correctness
- SpellF1: possession-level boundary matching

Improving one can hurt the other.

## Files Created
- `extract_broadcast_missing.py` — Extract and analyze broadcast missing periods
- `tactical_baseline_broadcast.py` — Integrate broadcast signal (tested)
- `tactical_baseline_broadcast_conservative.py` — Conservative masking (no improvement)
- `BROADCAST_INTEGRATION_EXPERIMENT.md` — This document

## Recommendation

**v5 (spellF1 0.4914, macroF1 0.7323) remains the best submission.**

Broadcast integration is the most promising external signal (better alignment than CALF or SN-BAS-2025), but the frame-to-spell gap means improving frame accuracy doesn't translate to better spellF1.

All external signal approaches (SN-BAS-2025, CALF, broadcast) face the same fundamental issue: **they detect real phenomena that don't align with GT's possession labeling philosophy**.

The only remaining path to higher spellF1: supervised learning on continuous possession labels that use the same definition as GT.
