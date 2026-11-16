# v5 Documentation - What Was Missing from Handoffs

## Critical Information Not Documented

The v5 baseline (spellF1 0.4914, macroF1 0.7323) was generated with:

```bash
python tactical_baseline.py --tracking-only --output evaluation/results/improved_dead_ball_v5.csv
```

**The `--tracking-only` flag was never documented in handoffs**, causing confusion about:
- Whether v5 used ball recovery (it doesn't)
- Whether v5 used GBM (it doesn't - GBM violates rules)
- Why v5 performs better than default (Smart Tags hurt performance)

## v5 Configuration

**What v5 uses:**
- Transition cost: 1.60 (dead ball penalty in Viterbi decoder)
- `--tracking-only` flag: Skips Smart Tags event annotations
- Pure tracking heuristics: Distance-based possession assignment
- NO ball recovery (273 recovered frames would only affect 0.14% of data)
- NO GBM model (would violate rules - trained on public test set)
- NO external data

**What v5 does NOT use:**
- Smart Tags (event annotations) - these HURT performance
- Ball detector / recovery - not used
- GBM model - rule violation
- Any training on GT labels

## Performance Comparison - Final Results

| Configuration | spellF1 | macroF1 | Notes |
|--------------|---------|---------|-------|
| Default (with Smart Tags) | 0.3529 | 0.7092 | Uses event annotations |
| **v5 (--tracking-only)** | **0.4914** | **0.7323** | **BEST - Final submission** |
| v5 + Ball Recovery | 0.4556 | 0.7039 | Worse (-7.3%) - recovery hurts |

**Ball recovery tested and rejected**: Recovers 273 frames but increases fragmentation and reduces performance.

## Why Smart Tags Hurt Performance

Smart Tags are event annotations (Kick Off, Free Kick, Throw In, Corner, etc.) extracted from video. They might hurt because:

1. **Timing mismatch**: Events might not align perfectly with GT possession labels
2. **Labeling convention**: GT labels events as occurring during H/A possession, not as separate dead ball moments
3. **Over-constraining**: Adding event constraints reduces decoder flexibility

The `add_tag_evidence()` function in tactical_baseline.py adds evidence from these tags, but empirically they degrade spellF1 by 39% (0.4914 → 0.3529).

## Ball Recovery Status - TESTED AND REJECTED

**Result**: v5 + ball recovery = spellF1 0.4556 (-7.3% worse than v5)

Ball recovery uses YOLO11m to detect ball in video frames where tracking fails. It recovers:
- 273 frames out of 85,831 missing (0.32%)
- 100% of recovered frames are in originally-missing regions

**Why it hurts performance**:
1. Recovered positions may have timing errors (video frame vs tracking frame sync)
2. Increases fragmentation: 1.14 → 1.24 (more possession switches)
3. Noise in recovered positions propagates through Viterbi decoder

**Conclusion**: Pure tracking (v5) is better than tracking + recovery.

## Lesson Learned

**Document generation commands in handoffs**, not just results. The v5 handoff should have included:

```markdown
## v5 Generation Command

```bash
python tactical_baseline.py --tracking-only --output evaluation/results/improved_dead_ball_v5.csv
```

**Critical flag**: `--tracking-only` skips Smart Tags, which improves spellF1 by 39%
**No ball recovery used**: Recovery would only affect 273 / 85,831 frames (0.32%)
```

This would have saved hours of investigation.
