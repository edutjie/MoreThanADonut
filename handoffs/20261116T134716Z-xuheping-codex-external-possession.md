# Session handoff: External possession training

- Session: 20261116T134716Z-xuheping-codex-external-possession
- User: xuheping
- Agent: Codex (GPT-6), implementation and evaluation
- Repository: MoreThanADonut
- Branch / worktree: experiments/training-approach
- Base commit: a52b939342a089d511de25087eec9d8039227fe9
- Started: 2026-11-16T13:47:16Z
- Updated: 2026-11-16T13:56:15Z
- Status: completed
- Delivery: local branch
- Objective: Improve possession spell F1 by developing and selecting methods on permitted external data.
- Owned paths: `external_possession.py`, decoder argument in `tactical_baseline.py`,
  `EXTERNAL_POSSESSION.md`, status notes in `FINAL_RESULTS.md` and
  `V5_DOCUMENTATION.md`, and this handoff
- Related sessions: [Branch pickup](20261116T134503Z-xuheping-codex-training-pickup.md)

## Outcome

- Externally trained possession model and Viterbi costs selected on four external
  development matches. The held-out A-League mean spellF1 rose from 0.4667 to
  0.6578 over four matches. The independent IDSSE test with simulated tracking
  gaps rose from 0.2213 to 0.4078 over seven matches.
- The public pipeline check of the frozen candidate scored 0.6250 spellF1 and
  0.8162 macroF1 over 7,500 labeled frames. The v5 reference scores are 0.4914
  and 0.7323. Private scoring is unavailable in this session.
- The candidate passed `evaluation/validate_submission.py` for all 199,319
  frames. A second training and inference run reproduced the exact candidate
  CSV. The default decoder still regenerates v5 byte for byte.
- Team-swap invariance, Python compilation and Git whitespace checks passed.

## Changes by commit

### 20261116T134716Z-xuheping-codex-external-possession-morethanadonut-01

- Changes: Add a reproducible external training, prediction and transfer check.
  Reuse the baseline decoder with externally calibrated costs and update the
  candidate documentation.
- Validation: Run the commands in [EXTERNAL_POSSESSION.md](../EXTERNAL_POSSESSION.md).

## Integration handoff

- Branch: `experiments/training-approach`
- Remote starting tip: `a52b939342a089d511de25087eec9d8039227fe9`
- Integration owner: xuheping
- Local artifacts: candidate at `data/external_possession/skillcorner-frozen/possession.csv`,
  model at `data/external_possession/skillcorner-4340d27.joblib`. Both are ignored.
- Other work observed: pre-existing untracked artifacts and the pickup handoff
  were preserved.
- Publication: local work only. No submission upload.
