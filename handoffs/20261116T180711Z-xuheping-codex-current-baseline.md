# Session handoff for the current GRU baseline

- Session: 20261116T180711Z-xuheping-codex-current-baseline
- User: xuheping
- Agent: Codex, GPT-6, implementation and verification
- Repository: MoreThanADonut
- Branch / worktree: `experiments/training-approach`
- Base commit: `bde47690a89d60a3db76132290053b290f1554d0`
- Started: 2026-11-16T18:07:11Z
- Updated: 2026-11-16T18:08:00Z
- Status: completed
- Delivery: local commit and running dashboard
- Objective: adopt the externally selected GRU with continuous player speeds as the current comparison baseline
- Owned paths: `TACTICAL_BASELINE.md`, `PLAYER_TRACK_QUALITY.md`, `GRU_DEAD_BALL.md`, `dashboard/serve.py`, `dashboard/README.md`, this handoff
- Related sessions: [player tracking experiment](20261116T175552Z-xuheping-codex-player-quality.md)
- Recipient: xuheping and the next modeling session

## Current outcome

The current baseline is documented in
[Tactical Baseline](../TACTICAL_BASELINE.md#current-experiment-baseline).
The checkpoint is `data/external_possession/player-quality/continuous/gru.pt`.
The dashboard shows its frozen prediction first, with previous lanes
available for comparison.

## Changes by commit

### 20261116T180711Z-xuheping-codex-current-baseline-morethanadonut-01

- State: committed
- Changes: designate the checkpoint in the baseline guide and default the dashboard to its local prediction
- Why: use the development-selected player-tracking model for subsequent comparisons
- Validation: validated the 199,319-frame CSV, verified default dashboard selection, checked tactical lane contents against the CSV and confirmed the broadcast lane
- Risks / follow-up: local checkpoint and prediction are ignored artifacts. Held-out tracking-fault results were mixed.

## Findings and decisions

- Adopted the continuous-speed GRU checkpoint from the external development selection. No model was retrained during this change.

## Communication log

- 2026-11-16T18:07Z | xuheping to Codex | accepted the continuous-speed GRU as the current baseline

## Integration handoff

- Source branch / head SHA: `experiments/training-approach`, commit located by the change ID above
- Target branch / tested target SHA: `experiments/training-approach` at the base commit above
- Integration owner: xuheping
- Shared branch / PR: local-only
- Dependencies / merge order: player track quality experiment
- Conflicts / resolutions: none
- Combined validation: dashboard and submission checks above
- Integration result: local on the target branch and serving port 8767
- Cleanup: keep the local checkpoints and predictions

## Resume here

Use the current baseline checkpoint and the documented external splits for
the next possession experiment.

## Outstanding state

- Uncommitted owned files: ignored dashboard server log
- Other work observed: pre-existing untracked files
- Blockers / access needed: none
- Failed / unrun checks: private competition scores are unavailable
- Cross-repository dependencies: none
- Publication / recipient acknowledgement: localhost only
