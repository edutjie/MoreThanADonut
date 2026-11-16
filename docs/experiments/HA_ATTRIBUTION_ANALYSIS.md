# H/A Attribution Error Analysis

## Summary
Analyzed 503 H/A attribution errors (8.3% of public window frames) to identify patterns and potential improvements.

**Bottom line**: The 8.3% error rate is **not fixable with better distance logic**. GT uses temporal context and game state beyond instantaneous ball proximity.

## Error Distribution

### H→A Errors (GT=H, pred=A): 314 frames
- **68% have clear Away advantage** (ball >2m closer to Away) — distance logic says Away
- **32% ambiguous** (both teams within 2m)
- **Mean distance advantage: -3.28m** (ball closer to Away)

### A→H Errors (GT=A, pred=H): 189 frames  
- **31% have clear Home advantage** (ball >2m closer to Home) — distance logic says Home
- **68% ambiguous** (both teams within 2m)
- **Mean distance advantage: +1.25m** (ball closer to Home)

**Key insight**: When our distance-based logic "disagrees" with GT, we're correct by distance 31-68% of the time. GT is using additional context.

## Temporal Patterns

### Error Clustering
- **12 error runs** (not 503 scattered frames)
- **Mean run length: 41.9 frames** (1.7 seconds)
- **Longest run: 115 frames** (4.6 seconds)
- **38.8% of errors near GT transitions** (±1 second)

### Error Types

#### Type 1: Transition Lag (at spell START)
Example: Frames 11351-11465 (115 frames, 4.6s)
- GT transitions to H immediately
- We predict A for 115 frames before switching
- **Cause**: Viterbi transition cost (1.60) creates inertia
- **Distance evidence**: Ball actually closer to A during this period

#### Type 2: Sustained Disagreement (MIDDLE/END of spell)  
Example: Frames 11704-11790 (87 frames, 3.5s)
- GT labels entire period as H
- We predict A throughout
- **Distance evidence**: Ball 0.16-2.01m closer to A
- **Cause**: GT assigns possession earlier than distance logic would

## Distance Evidence at Error Frames

Sampled 100 H→A and 100 A→H errors with valid tracking:

| Error Type | Mean Min H Dist | Mean Min A Dist | Mean Advantage |
|-----------|----------------|----------------|----------------|
| H→A (GT=H, pred=A) | 6.15m | 2.87m | -3.28m (A closer) |
| A→H (GT=A, pred=H) | 1.94m | 3.19m | +1.25m (H closer) |

**Our distance logic matches the ball proximity** — we're not making computational errors.

## Root Cause

GT possession labeling uses:
1. ✓ Ball proximity (we have this)
2. ✓ Temporal context (game flow, recent actions)  
3. ✓ Game state (who last touched, intended direction)
4. ✓ Event timing (passes, interceptions)

Our method only uses (1). The 8.3% gap reflects (2-4).

## Why Standard Fixes Won't Work

### Reduce transition cost?
- Would fix Type 1 errors (transition lag)
- But would increase flickering (currently at 11.4 transitions/min)
- SpellF1 penalizes short spells — more transitions = lower score
- Current 1.60 cost already optimized for spellF1

### Use mean of closest 3 players?
- Tested: doesn't change which team is "closer"
- Same errors, just with smoothed distances

### Add directional bias?
- Tracking data doesn't include team attack direction
- Would require manual annotation or inference

### Add ball velocity?
- Ball velocity in tracking is noisy (low frame rate)
- Doesn't capture "intended" direction vs actual movement

## Conclusion

The 8.3% H/A error rate is **structural, not fixable with heuristics**.

To push beyond v5 (spellF1 0.4914), we need:

1. **Supervised learning on continuous possession labels**
   - Train on tracking + GT labels (not just events)
   - Learn the temporal context patterns
   - Requires dataset with frame-level GT (not SN-BAS-2025 events)

2. **Accept current performance**
   - 0.4914 beats baseline by 8.8%
   - Fully rule-compliant
   - Further improvement requires different approach

3. **Optimize other aspects**
   - Focus on spell boundary quality, not frame accuracy
   - Ensemble with other tracking-based models

## Files Created
- `analyze_ha_errors.py` — Error pattern analysis (this analysis)
- `HA_ATTRIBUTION_ANALYSIS.md` — This document

## Recommendation

**Stop optimizing H/A attribution with heuristics.** The path to higher spellF1 requires supervised learning on continuous possession labels, which we don't have access to under competition rules.

Current v5 (0.4914 spellF1, 0.7323 macroF1) is strong performance given the constraints.
