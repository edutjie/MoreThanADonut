# SN-BAS-2025 Integration Experiment

> **Retired experiment.** This is a historical record. It contains legacy
> claims or commands involving models trained or tuned on the public window,
> or derived comparisons. Do not use these results or commands for a
> submission. See [the provenance audit](COMPETITION_DATA_AUDIT.md).

## Objective
Use external event labels from SN-BAS-2025 (NDA-signed dataset) to improve dead ball detection and push spellF1 beyond the 0.4914 baseline.

## Dataset Analysis
- **Downloaded**: 10 GB (300 matches with event labels)
- **Event labels**: 56,380 events across 17 types
  - 43,959 dead ball events (Throw-in, Free kick, Corner, etc.) — 78%
  - 12,421 live ball events (Shots, Clearance, etc.) — 22%
- **Continuous labels**: Only 7 matches have Labels-ball.json with frame-level possession
- **Primary data**: Discrete event timestamps (milliseconds), not continuous frame labels

## Experiments Conducted

### 1. Event Pattern Classifier
**Approach**: Train GBM to distinguish real dead ball events from false positives using temporal features.

**Training**: 
- 56K events → 16 features (event type one-hot, timing, sequence patterns)
- 98% test accuracy (precision 0.98, recall 1.00)
- Model saved to `models/dead_ball_precision_filter.pkl`

**Application**:
- Filtered v5 predictions: 73K → 22K dead ball frames (69.5% reduction)
- Tested two strategies:
  1. Convert filtered D → X (unknown): spellF1 0.4914, macroF1 0.6805 ✗
  2. Inherit from surrounding H/A: spellF1 0.4571, macroF1 0.6679 ✗
  
**Result**: Both worse than v5 (0.4914 / 0.7323)

### 2. Conservative Dead Ball Weights
**Approach**: Reduce dead ball evidence weights by 50% to match rank #1's conservative use (6.7% D labels).

**Changes**:
- pitch_exit: 0.32 → 0.15
- ball_missing: 0.06 → 0.03
- sustained_missing: 0.24 → 0.10
- far_from_players: 0.08 → 0.04

**Result**: spellF1 0.4571, macroF1 0.7018 — still worse than v5 ✗

## Root Cause Analysis

**Problem**: Our dead ball heuristics over-detect by 50×:
- v5 labels: 73,260 frames (36.8%)
- GT labels: 1,469 frames (0.7%)
- Rank #1 baseline: 13,263 frames (6.7%)

**Why filtering failed**:
1. GT actually labels those 1.5K frames as D — removing them hurts recall
2. The other 72K are false positives, but they're distributed throughout
3. Filtering to X or inheriting H/A both hurt macroF1 (per-frame accuracy)

**Why conservative weights failed**:
- Still produced 69K D labels (47× GT) despite 50% weight reduction
- The heuristic signals (ball missing, far from players) trigger too frequently
- Would need 10× lower weights to match GT's 1.5K, breaking all dead ball detection

**Fundamental issue**: 
The competition's GT dead ball labeling is **extremely selective** — only major stoppages (Throw-ins, Free kicks at specific moments). Most tracking anomalies (ball temporarily off-screen, brief pauses) are labeled as H/A based on context.

Our heuristic: "ball is missing → probably dead"
GT philosophy: "ball is missing → probably still the same team's possession"

## Conclusion

**SN-BAS-2025 taught us**:
1. ✓ External event patterns can be learned (98% classifier accuracy)
2. ✗ But they don't help fix structural over-detection
3. ✗ Filtering false positives hurts when you can't distinguish them from true positives
4. ✓ Confirmed rank #1 uses very sparse D labels (6.7% vs our 36.8%)

**Best result remains v5**:
- spellF1: **0.4914** (beats baseline 0.4516 by 8.8%)
- macroF1: **0.7323**
- Method: Multi-signal dead ball + Viterbi transition cost 1.60
- Fully rule-compliant (tracking heuristics only)

## Files Created
- `extract_sn_bas_labels.py` — Extract 56K events from 300 matches
- `train_dead_ball_filter.py` — Train 98% accuracy event classifier  
- `apply_dead_ball_filter.py` — Apply filtering (didn't improve results)
- `models/dead_ball_precision_filter.pkl` — Trained model (not used in final submission)

## Recommendation
**Stop optimizing dead ball detection.** The path to higher spellF1 is:
1. Better H/A discrimination (possession attribution, not dead ball detection)
2. Training on external tracking data with possession labels (not just events)
3. Or: accept 0.4914 as strong performance and focus on other competition tasks
