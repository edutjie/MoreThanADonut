# Session handoff: Bundesliga H/A/D training

- Session: 20261116T160156Z-xuheping-codex-idsse-training
- User: xuheping
- Agent: Codex (GPT-6), implementation and evaluation
- Branch: experiments/training-approach
- Base commit: 46f27d997ce2b0a42dba946c624c21d2fa103d18
- Started: 2026-11-16T16:01:56Z
- Updated: 2026-11-16T16:08:46Z
- Status: completed
- Delivery: local branch and ignored artifacts
- Integration owner: xuheping
- Owned paths: Bundesliga training experiment, results documentation and this handoff

## Outcome

- Trained Bundesliga-only and equal-source-weight pooled LightGBM variants on
  whole external matches. Both picked decoder costs (5, 10). Combined won the
  equal-league development comparison with 0.5276 spellF1.
- On two held-out masked Bundesliga games, spellF1 rose from 0.4369
  (A-League reference) to 0.4859 (Bundesliga-only) and 0.4609 (combined).
  The in-pitch D F1 scores were 0.9100, 0.9405 and 0.9264 respectively.
- The combined model dropped to 0.5294 public contest spellF1 against
  0.6452 for the existing reference. Both new predictions passed validation
  for 199,319 frames. The Bundesliga-only prediction received a format check.
- The initial scoring workers stalled with inherited OpenMP state. Spawned
  workers completed a checkpoint smoke check and the full rerun.

## Changes by commit

### 20261116T160156Z-xuheping-codex-idsse-training-morethanadonut-01

- Changes: add a reproducible external training and held-out benchmark.
- Validation: full external benchmark, contest prediction validation,
  one frozen public check, worker smoke check, Python compilation and Git diff.
- Evidence: [results and reproduction commands](../IDSSE_TRAINING.md).
- Artifacts: `data/external_possession/idsse-training/` is ignored.
- Publication: local branch. The original lagged A-League LightGBM remains the
  working reference.
