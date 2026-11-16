# Session handoff: Missing-ball recovery experiment

- Session: 20261116T150834Z-xuheping-codex-ball-recovery
- User: xuheping
- Agent: Codex (GPT-6), implementation and evaluation
- Branch: experiments/training-approach
- Base commit: 2e084f1fd1bb6a0ea41d2aa0d90f51eeb231dbe8
- Started: 2026-11-16T15:08:34Z
- Status: completed
- Delivery: local branch and local experiment artifacts
- Owned paths: ball recovery experiment, optional sparse detector sampling, experiment documentation and this handoff

## Outcome

- Retained the frozen lagged LightGBM and decoder. Compared eight fusion
  policies on external development and held-out IDSSE matches under
  independent and stoppage-associated missingness.
- Both external development scenarios selected direct replacement. On the
  stoppage-associated held-out simulation, spellF1 rose from 0.4244 to 0.4620
  and D F1 rose from 0.9195 to 0.9465.
- Ran the local YOLO checkpoint across both halves with a sparse frame scan.
  Accepted 555 detections over both halves and 28 in the public window.
  Public spellF1 stayed at 0.6452. A dense public-window diagnostic used the
  same fixed policy, recovered 273 frames and scored 0.6194. Both CSVs passed
  full format validation. The lagged LightGBM reference stays in use.
- Verified fusion probabilities, sparse frame selection and byte-identical
  no-recovery output. Code and reproduction commands are in
  [BALL_RECOVERY_EXPERIMENT.md](../BALL_RECOVERY_EXPERIMENT.md).

## Changes by commit

### 20261116T150834Z-xuheping-codex-ball-recovery-morethanadonut-01

- Changes: add the fusion experiments and optional sparse YOLO scan.
- Validation: both external simulations, two real detector runs, format checks,
  Python compilation and Git whitespace checks.
- Integration owner: xuheping. Generated coordinates and predictions remain in
  the ignored `data/external_possession/yolo-recovery/` directory.
