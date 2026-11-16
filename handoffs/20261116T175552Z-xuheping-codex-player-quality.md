# Session handoff for player tracking in possession inference

- Session: 20261116T175552Z-xuheping-codex-player-quality
- User: xuheping
- Agent: Codex, GPT-6, implementation and evaluation
- Repository: MoreThanADonut
- Branch / worktree: `experiments/training-approach`
- Base commit: `4bb8f8de9ea61871115f6922f3f27d954679133a`
- Started: 2026-11-16T17:55:52Z
- Updated: 2026-11-16T18:03:37Z
- Status: completed
- Delivery: local commit and ignored experiment artifacts
- Objective: address stale and inconsistent player coordinates in model inputs and evaluate performance
- Owned paths: `player_track_quality.py`, `player_quality_experiment.py`, `external_possession.py`, `sequence_possession.py`, `gru_dead_ball.py`, `PLAYER_TRACK_QUALITY.md`, this handoff
- Related sessions: [GRU specialist](20261116T173425Z-xuheping-codex-gru-dead-ball.md), [dashboard held dots](20261116T175158Z-xuheping-codex-dashboard-held-dots.md)
- Recipient: xuheping and the next possession modeling session

## Current outcome

Coordinate repair is now usable during training and inference. The two
retrained checkpoints store their input handling. External development
selected the GRU with continuous speeds. It scored 0.6897 public spellF1
and 0.9002 macroF1, with mixed consistency under injected faults. The
[experiment report](../PLAYER_TRACK_QUALITY.md) compares frozen inference
repair and the two retrained runs under natural and injected faults.

## Changes by commit

### 20261116T175552Z-xuheping-codex-player-quality-morethanadonut-01

- State: committed
- Changes: repair long held positions and short out-and-back spikes, optionally exclude track breaks from speed features, and evaluate the frozen and retrained GRUs with and without the specialist
- Why: coordinate artifacts enter model features and affect scoring as well as visualization
- Validation: tested 10/25 Hz repairs and continuity, external development and four held-out conditions, submission validation and automated public scoring. The old GRU and both new predictions reproduced byte for byte.
- Risks / follow-up: the injected faults are simplified. Persistent identity errors can still affect distance features.

## Findings and decisions

- Raw tactical coordinate repair affected 6,698 of 3,254,570 observed player-frame positions inside the halves. All changes were exact holds.
- The eight-epoch runs reuse the original A-League training split. The extra corruption is applied to the existing augmented copy.
- Contest labels were used only in the automated public scorer, after settings and checkpoints were chosen externally.

## Communication log

- 2026-11-16T17:55Z | xuheping to Codex | clarified that the tracking defect must be addressed in model inputs

## Integration handoff

- Source branch / head SHA: `experiments/training-approach`, commit located by the change ID above
- Target branch / tested target SHA: `experiments/training-approach` at the base commit above
- Integration owner: xuheping
- Shared branch / PR: local-only
- Dependencies / merge order: prior GRU and specialist checkpoints plus pinned external sources
- Conflicts / resolutions: none
- Combined validation: external experiments and public pipeline checks
- Integration result: local on target branch
- Cleanup: retain experiment artifacts under ignored data directory

## Resume here

Use the [report](../PLAYER_TRACK_QUALITY.md) to compare candidate outputs.
The coordinate repair is still exposed explicitly and the saved GRU
parameters control feature handling for existing inference entry points.

## Outstanding state

- Uncommitted owned files: ignored results and prediction files in `data/external_possession/player-quality/`
- Other work observed: pre-existing untracked files
- Blockers / access needed: none for local reproduction
- Failed / unrun checks: private competition scoring is unavailable
- Cross-repository dependencies: pinned external sources from the prior audit
- Publication / recipient acknowledgement: local commit only
