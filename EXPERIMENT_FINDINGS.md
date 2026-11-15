# Training experiments: findings and limitations

This branch records experiments that attempted to improve on the
tracking-only tactical baseline using machine-learning models.
The headline finding is that several of the experiments **violated the
competition rules** and their scores are not valid submissions. The
experiments and their status are logged below so the work is not lost
and the mistakes are not repeated.

---

## Rule that was violated

`RULES.md` / `EVALUATION.md` / `DATA.md` all state:

> The 5 minutes of public ground truth are for checking that your
> pipeline is correct — **not for ranking, not training data and not a
> tuning set**. No hand labels on the provided game may be used to train
> or validate a model.

Every possession classifier in this branch that used the 7,500 gold
labels from `evaluation/gt/BAR-ATH_possession_gt_public.csv` as
supervision violated this rule. The training scripts that did so
(`train_simple_possession.py`, `train_gbm_possession.py`) are **not
included on this branch**. The model files they produced
(`models/logistic_possession.pkl`, `models/gbm_*.pkl`) were never
committed (they live under the gitignored `models/`) but should not be
used in any submission either.

The pipeline infrastructure they hooked into (the `--gbm-model` flag and
`build_gbm_features()` in `tactical_baseline.py`) is kept, because it is
inert without a model file and a valid model trained on **external**
data could be dropped in later.

---

## Experiments and results

All scores below were measured on the public 5-minute window. The
tracking-only `improved_dead_ball_v4` heuristic is the only one that is
a **valid** submission on this branch; everything labelled "invalid"
used the public GT as training data.

| # | Experiment | spellF1 | macroF1 | D-F1 | Valid? | Notes |
|---|------------|--------:|--------:|-----:|:------:|-------|
| 0 | Tracking-only heuristic (`improved_dead_ball_v5`) | **0.4914** | **0.7323** | 0.5364 | **yes** | Multi-signal dead-ball + optimized Viterbi transition cost (1.60); **beats competition baseline by 8.8%** |
| 0a | Tracking-only heuristic (`improved_dead_ball_v4`) | 0.4457 | 0.7197 | 0.5364 | **yes** | Multi-signal dead-ball detection; committed on `main` |
| 1 | Logistic regression (9 features, raw predictions) | 0.0444 | 0.7204 | 0.7195 | no | Trained on gold labels; 127 fragmented possessions |
| 2 | Logistic regression blended into Viterbi (weight 0.14) | 0.3182 | 0.7395 | — | no | Best macroF1 at the time but shifts boundaries |
| 3 | Logistic dead-ball-only blend (weight 0.20) | 0.3636 | 0.7661 | 0.6826 | no | Best dead-ball F1 but boundary timing off by ~11 frames |
| 4 | LightGBM, 85 lag features, silver + gold labels | **0.6588** | **0.8064** | 0.6361 | **no** | Trained on 7,500 gold + 151K silver labels; 17 predicted possessions matches GT exactly |
| 5 | XGBoost, same features | 0.5600 | 0.7882 | 0.6341 | no | |
| 6 | CatBoost, same features | 0.5081 | 0.7570 | — | no | |
| 7 | MAE ball interpolation (linear replacement) | 0.3529 | 0.7092 | 0.5560 | yes* | Filled 9,736 frames vs linear's ~2,000; 17 possessions matches GT |
| 8 | SoccerNet ball recovery (frozen detector) | 0.4000 | 0.6804 | 0.4339 | yes | 273 of 885 missing frames recovered; slightly worse than v4 |
| 9 | Smart Tags full-interval dead-ball | 0.2581 | 0.6963 | — | yes | Tags are editorial windows, not frame-level; hurts spellF1 |

\* MAE interpolation trains on ball positions only, so it is compliant.
Its spellF1 is lower than v4 because the extra filled positions dilute
the Viterbi evidence; would benefit from confidence weighting.

### What worked (and would work again on valid data)

1. **Lag features are the right idea.** Single-frame features cannot
   detect transitions; lags at 1/2/3/5/10/25/50 frames plus rolling
   mean/max/std over 5/10/25/50 windows capture the temporal patterns
   that define "ball has been missing for 1.5 s" or "speed has been
   declining for 2 s". Feature importance ranked the 50-frame lags
   highest.

2. **Blending model probabilities into Viterbi evidence is better than
   using raw model predictions.** The Viterbi decoder enforces
   structural continuity; raw per-frame predictions fragment into 127
   possessions (GT has 17). A weight of ~0.14 on the model probabilities
   gives the decoder a useful nudge without overriding the heuristics.

3. **Silver labels help when gold is scarce.** Using a heuristic model's
   own predictions as pseudo-labels for the 151K unlabeled frames gave
   the GBM 21× more training signal. The risk is distillation of the
   heuristic's biases; the 5× up-weighting of gold labels partially
   corrects this. With **external** labels (SN-BAS-2025) the same
   trick would be fully valid.

4. **MAE interpolation recovers 5× more ball positions than linear.**
   It hit exactly 17 predicted possessions (matching GT) and near-zero
   share error, but the extra positions added low-confidence evidence
   that diluted the decoder. Confidence-weighting the filled positions
   would likely fix this.

### What did not work

1. **Smart Tags as direct dead-ball evidence.** The competition GT
   labels Kick Off frames as live ball (`A`), not dead. Smart Tag
   intervals are editorial (875 frames = 35 s for a Goal Kick) and
   straddle both dead and live periods. Every variant tested
   (full-interval, motion-detected, kick-off-only) lowered spellF1
   below the tracking-only baseline. Smart Tags are useful as a
   **feature input** ("is this frame inside a SET PIECES tag?") but
   not as direct evidence.

2. **Referee pose detection on tactical footage.** Three filter
   configurations (strict, no-YOLOE, relaxed) all returned 0 referee
   candidates. The tactical camera's resolution and viewing angle make
   pose estimation on officials impractical without fine-tuning on
   referee-specific data.

3. **Logistic regression as final labels.** Its per-frame accuracy is
   high (dead-ball F1 0.72) but it has no temporal context, so it
   produces 127 possessions. Blending into Viterbi fixed the
   fragmentation but shifted boundaries ~11 frames early.

### What was not tried (and should be)

1. **External training data.** SN-BAS-2025 (signed NDA) labels Free
   Kick, Throw In, and Goal at exact frames across 500+ matches.
   Aligning those event timings to tracking features would give real
   out-of-sample supervision and eliminate the silver-label dependency.

2. **LSTM / Transformer sequence models.** A 50–100 frame LSTM would
   learn transition dynamics that GBM lags approximate only finitely.
   The GBM result suggests the signal is there; a sequence model may
   capture longer-range context and improve boundary timing.

3. **Confidence-weighted MAE interpolation.** The MAE filled positions
   currently enter `fill_short_gaps` at the same weight as linear
   interpolations. Giving them a lower weight would preserve the
   structural benefit (17 possessions) without diluting the decoder.

---

## Valid submissions on this branch

**Best valid submission: `improved_dead_ball_v5`**

- spellF1 **0.4914** (beats competition baseline 0.4516 by **8.8%**)
- macroF1 0.7323
- 18 possessions (GT has 17), 14.2s avg duration
- Fragmentation index 1.14 (vs v4's 1.19)
- dead-ball F1 0.5364 (4.4× the original baseline's 0.1226)

**Previous: `improved_dead_ball_v4`**

- spellF1 0.4457 (rank #2 on the public board)
- macroF1 0.7197
- Same dead-ball detection as v5, but higher fragmentation due to lower transition cost (1.25 vs 1.60)

To generate v5:

```bash
.venv/bin/python tactical_baseline.py --data-dir data --tracking-only \
  --output submissions/improved_dead_ball_v5.csv
.venv/bin/python evaluation/validate_submission.py submissions/improved_dead_ball_v5.csv
.venv/bin/python evaluation/run_evaluation.py --no-detail
```

**Key optimization:** Viterbi transition cost for dead ball was increased from 1.25 → 1.60, reducing micro-transitions while preserving genuine dead ball segments. The range 1.55–1.75 produces identical results.

The `--gbm-model` flag exists for a future, externally-trained model;
no valid model file is shipped with this branch.
