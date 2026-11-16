# Session handoff: Bundesliga YOLO and possession smoothing

- Session: 20261116T162307Z-xuheping-codex-spell-smoothing
- User: xuheping
- Agent: Codex (GPT-6), implementation and evaluation
- Branch: experiments/training-approach
- Base commit: 14ec37725754110363f708920ce198a428eea00f
- Started: 2026-11-16T16:23:07Z
- Updated: 2026-11-16T16:25:40Z
- Status: completed
- Delivery: local branch and ignored diagnostic artifacts
- Integration owner: xuheping
- Owned paths: spell smoothing experiment, results documentation and this handoff

## Outcome

- The frozen Bundesliga-only model with sparse YOLO recovery scored 0.4800
  public spellF1, equal to Bundesliga tracking alone. MacroF1 fell from
  0.7035 to 0.6901 and D F1 from 0.4496 to 0.4145.
- Tested 15 settings against the official spellF1 implementation. External
  development selected merging weak live pulses of up to 1.5 seconds.
  Balanced development spellF1 rose from 0.510493 to 0.511384, while it
  declined slightly in both held-out leagues.
- The smoothed contest CSV changed 81 frames outside the public window
  and kept 0.6452 spellF1 and 0.8084 macroF1 in the official check.
  Both predictions passed 199,319-frame validation.

## Changes by commit

### 20261116T162307Z-xuheping-codex-spell-smoothing-morethanadonut-01

- Changes: add decoder calibration, weak live-turnover smoothing and
  documentation of the Bundesliga YOLO diagnostic.
- Validation: external development and holdout, frozen contest scoring,
  synthetic smoothing and decoder-equivalence checks, compilation and diff.
- Evidence: [results and reproduction commands](../SPELL_SMOOTHING.md).
- Publication: local branch. Original A-League LightGBM remains the
  working reference.
