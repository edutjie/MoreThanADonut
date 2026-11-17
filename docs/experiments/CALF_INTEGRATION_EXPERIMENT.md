# CALF Integration Experiment

> **Retired experiment.** This is a historical record. It contains legacy
> claims or commands involving models trained or tuned on the public window,
> or derived comparisons. Do not use these results or commands for a
> submission. See [the provenance audit](COMPETITION_DATA_AUDIT.md).

## Objective
Integrate CALF event detection to improve dead ball detection and push spellF1 beyond v5 (0.4914).

## Approach
CALF (Context-Aware Late Fusion) detects 17 event types including:
- Dead ball: Throw-in, Goal kick, Corner, Offside, Foul, Penalty, Yellow card, Kick-off, etc.
- Live ball: Shots, Clearance, Goal

Add CALF event evidence as an additional signal to the Viterbi decoder's dead ball channel.

## Implementation
Created `tactical_baseline_calf.py`:
- Loads CALF predictions from `submissions/frozen-soccernet/calf-first-half.json`
- Generates frame-level dead ball evidence from events (±25 frame window with exponential decay)
- Adds evidence to tracking-based dead ball channel before Viterbi decoding
- Tested weights: 0.20 (too strong), 0.10 (harmful), 0.05 (no effect)

## Results

### Weight 0.20
- **Disaster**: Converted 30K X→D labels
- Frame accuracy: 23.2% → 2.9% (-20.4%)
- Dead ball: 36.8% → 51.8% (+30K frames)
- Root cause: Overwhelming tracking evidence, affecting unlabeled frames

### Weight 0.10
- 97 frames changed (all H/A→D)
- Frame accuracy: 77.92% → 76.71% (-1.2%)
- **0 improvements, 97 regressions**

### Weight 0.05
- Only 2 frames changed (both A→D, incorrect)
- Essentially no effect

## Root Cause Analysis

Checked GT labels at CALF dead ball event frames:

**CALF dead ball events in labeled window: 22**
- GT=H: 12 (54.5%)
- GT=A: 7 (31.8%)
- GT=D: 3 (13.6%) ← **Only 13.6% align!**

**Examples**:
- Frame 11773: Yellow card → GT=H
- Frame 12335: Kick-off → GT=H
- Frame 14373: Penalty → GT=H
- Frame 12935: Yellow card → GT=D ✓

**GT's labeling philosophy**: Events like yellow cards, kick-offs, and penalties occur *during* possession, not as separate dead ball moments. The ball is still "in possession of" the team that will restart play.

CALF detects real events correctly, but **GT intentionally labels those moments as H/A possession**, not D.

## Comparison to SN-BAS-2025

Same structural issue:
- SN-BAS-2025: External event labels didn't help because GT treats tracking anomalies as continued possession
- CALF: External event detection doesn't help because GT treats events as occurring during possession

Both external signal sources are **correct** in detecting events/anomalies, but **incompatible** with GT's possession philosophy.

## Conclusion

**CALF integration cannot improve v5** because:
1. CALF events don't align with GT dead ball labels (13.6% overlap)
2. Adding CALF evidence pushes predictions toward D at frames GT labels H/A
3. Every weight tested either has no effect (too low) or causes regressions (too high)

**v5 (spellF1 0.4914, macroF1 0.7323) remains optimal.**

## Files Created
- `tactical_baseline_calf.py` — CALF integration (tested, doesn't improve v5)
- `CALF_INTEGRATION_EXPERIMENT.md` — This document

## Recommendation

**Stop pursuing external event/signal integration.** Both SN-BAS-2025 and CALF experiments confirm that external signals—even when detecting real events correctly—cannot improve heuristic-based possession prediction because GT uses a different labeling philosophy.

The only remaining path to higher performance: **supervised learning on continuous possession labels** (frame-level GT from a compatible dataset), which we cannot access under competition rules.

v5 is the ceiling for tracking-only heuristics.
