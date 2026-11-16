# Session handoff: experiments/training-approach branch

- Session: 20261115T053640Z-xuheping-opus55-branch
- User: xuheping
- Agent: Claude Opus 5.5
- Repository: MoreThanADonut
- Branch / worktree: experiments/training-approach
- Base commit: bf506b1 (on main)
- Started: 2026-11-15T05:36Z
- Status: completed, branch ready to push
- Objective: Remove rule-violating training code, log findings, push clean branch
- Owned paths: this handoff
- Recipient: xuheping

## Outcome

Created `experiments/training-approach` from `bf506b1` (the last clean
commit on main) and rebuilt it with only competition-compliant code:

1. `fix(dead-ball): relax referee candidate geometry filters` — dead_ball.py
2. `feat(baseline): add GBM feature pipeline and evidence blend hook` — tactical_baseline.py
3. `feat(ball): add MAE ball interpolation training scripts` — ball_mae.py, train_mae_simple.py, prepare_ball_training_data.py, evaluate_when_ready.sh
4. `docs: log training experiment findings and rule limitations` — EXPERIMENT_FINDINGS.md

## What was removed vs main

- `train_simple_possession.py` — trained on public GT (rule violation)
- `train_gbm_possession.py` — trained on public GT (rule violation)
- `logistic_features()` and `--logistic-model` flag in tactical_baseline.py — bridge to the GT-trained logistic model
- Verbose session summaries (COMPLETED_WORK.md, FINAL_STATUS.md,
  PROGRESS_REPORT.md, SESSION_SUMMARY.md, TRAINING_PLAN.md)
- Invalid submissions from evaluation/results/ (gbm_*, logistic_*, lr_*,
  mae_interpolation, smart_tags_nobounds, full_tags*)

## What was kept

- `build_gbm_features()` and `--gbm-model` flag — inert without a valid model file
- `ball_mae.py`, `train_mae_simple.py`, `prepare_ball_training_data.py` —
  train on ball positions only, no possession labels
- `dead_ball.py` relaxed referee filters
- `add_tag_evidence` np.maximum guards and goal-kick/free-kick comment
- `EXPERIMENT_FINDINGS.md` documenting what was tried and what violated rules

## Valid submission on this branch

`improved_dead_ball_v4` (also on main) is the only valid submission:
- spellF1 0.4457 (rank #2), macroF1 0.7197, dead-ball F1 0.5364

## Outstanding state

- Uncommitted owned files: none
- Blockers: none
- Next actions: push branch, then optionally train GBM on SN-BAS-2025

## Resume here

```bash
git push -u origin experiments/training-approach
```

Then to pursue external-data training:
1. Download SN-BAS-2025 via HuggingFace (NDA signed)
2. Align event labels (Free Kick, Throw In, Goal) to tracking features
3. Train GBM on those labels instead of silver/gold
4. Use `--gbm-model models/gbm_external.pkl` in tactical_baseline.py
