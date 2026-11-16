# Session handoff for external possession experiments

- Session: 20261116T171814Z-xuheping-codex-external-data-study
- User: xuheping
- Agent: Codex, GPT-6, implementation and evaluation
- Repository: MoreThanADonut
- Branch / worktree: `experiments/training-approach`
- Base commit: `54efb3348dce2b494d9a9bbe706c50c15cb567e5`
- Started: 2026-11-16T17:18:14Z
- Updated: 2026-11-16T17:28:02Z
- Status: completed
- Delivery: local commit
- Objective: test the additional sources already accessible for possession and in-pitch stoppages
- Owned paths: `possession_sources.py`, `external_stoppage_experiment.py`, `external_possession.py`, `sequence_possession.py`, `EXTERNAL_DATA_EXPERIMENTS.md`, this handoff
- Related sessions: [external data audit](20261116T164836Z-xuheping-codex-external-data.md)
- Recipient: xuheping and the next possession modeling session

## Current outcome

Ran lagged LightGBM, GRU and Transformer on A-League plus legacy SkillCorner.
Both sequence architectures also ran with conditional H/A loss for the legacy
data. The original GRU leads the three aggregate held-out spellF1 tests and
the automated public contest check. A Metrica player-motion specialist and a
version augmented with SoccerTrack both selected zero weight on external
development. The [experiment report](../EXTERNAL_DATA_EXPERIMENTS.md) has
the numbers and reproduction commands.

## Changes by commit

### 20261116T171814Z-xuheping-codex-external-data-study-morethanadonut-01

- State: committed
- Changes: add legacy and Metrica loaders, conditional sequence supervision, and a player-motion stoppage transfer experiment
- Why: test whether additional accessible tracking improves possession with a missing ball or identifies stoppages with the ball still in play's area
- Validation: ran five legacy fits, checked both stoppage variants on development and held-out games, validated the three 199,319-frame public candidates, and checked conditional gradients, mask handling, symmetric motion features, and tag-pairing boundaries
- Risks / follow-up: legacy training adds updates per epoch; Metrica weak stoppage tracks rarely include a ball; SoccerTrack pilot uses 40 tracking minutes and touchline stoppages

## Findings and decisions

- The conditional Transformer improved natural-gap matches from 13/30 to 18/30. Its long-gap and Bundesliga spellF1 declined.
- Metrica provides 53 candidate foul-to-restart intervals with tracked ball XY in only 16 labeled dead-ball frames.
- The StatsBomb pass-anchor audit highlighted a clock-offset disagreement for the first half of legacy match 2841. No derived StatsBomb targets entered training.
- Gradient files were not available in this workspace during the run.

## Communication log

- 2026-11-16T17:18Z | xuheping to Codex | requested experiments with data currently accessible and indicated Gradient access can be obtained

## Integration handoff

- Source branch / head SHA: `experiments/training-approach`, commit located by the change ID above
- Target branch / tested target SHA: `experiments/training-approach` at the base commit above
- Integration owner: xuheping
- Shared branch / PR: local-only
- Dependencies / merge order: follows the external data audit and sequence benchmarks
- Conflicts / resolutions: none
- Combined validation: experiment runs and checks above
- Integration result: local on target branch
- Cleanup: keep the worktree and local source files

## Resume here

1. Locate the Gradient full-match files once available and audit duplicate games,
   synchronized foul or offside markers, restart times, and observed ball
   status before training.
2. Use the current original GRU and external held-out splits as references.

## Outstanding state

- Uncommitted owned files: downloaded SoccerTrack inputs and experiment outputs are local data
- Other work observed: pre-existing untracked scripts, models, results and handoff
- Blockers / access needed: local Gradient files for a World Cup experiment
- Failed / unrun checks: the first SoccerTrack extraction rejected one untracked starting frame; the rerun kept the frame unknown and passed. No private contest scoring was run.
- Cross-repository dependencies: pinned public sources in the audit and SoccerTrack v1.2
- Publication / recipient acknowledgement: local commit only
