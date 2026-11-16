# Session handoff for aligned broadcast GRU inference

- Session: 20261116T182455Z-xuheping-codex-aligned-gru
- User: xuheping
- Agent: Codex, GPT-6, implementation and evaluation
- Repository: MoreThanADonut
- Branch / worktree: `experiments/training-approach`
- Base commit: `8b4e7067a53566d76b1d9d0b176cf438743f0377`
- Started: 2026-11-16T18:24:55Z
- Updated: 2026-11-16T18:28:15Z
- Status: completed
- Delivery: local commit and ignored results
- Objective: test the frozen GRU on verified broadcast tracking with tactical fallback
- Owned paths: `broadcast_gru_experiment.py`, `broadcast_alignment.py`, `dashboard/serve.py`, `sequence_possession.py`, `BROADCAST_GRU_EXPERIMENT.md`, this handoff
- Related sessions: [current baseline](20261116T180711Z-xuheping-codex-current-baseline.md)
- Recipient: xuheping

## Current outcome

Broadcast-first public spellF1 was 0.5697 versus 0.6897 for tactical. The
broadcast-ball-gap policy had identical public predictions to tactical.
Clipping tactical context to the same spans produced 0.6483. See the
[experiment report](../BROADCAST_GRU_EXPERIMENT.md) for coverage and
reproduction commands. The current baseline remains tactical.

## Changes by commit

### 20261116T182455Z-xuheping-codex-aligned-gru-morethanadonut-01

- State: committed
- Changes: share alignment validation and saved GRU preprocessing, add mapped probability substitution and a control for lost temporal context
- Why: test whether the contest broadcast feed improves possession predictions from a broadcast-trained checkpoint
- Validation: validated four 199,319-frame CSVs, repeated the first three byte for byte, checked malformed maps and fallback, verified dashboard predictions and frame pairs, and ran Python compilation and Git whitespace checks
- Risks / follow-up: public checks have 17 possession spells and no result for ungraded frames

## Findings and decisions

- Use the frame map only on verified native indices. Each broadcast span gets its own context and all policies decode on the tactical halves.
- The context control was a diagnostic for the broadcast-first performance drop.

## Communication log

- 2026-11-16T18:24Z | xuheping to Codex | requested broadcast-first possession inference with tactical fallback

## Integration handoff

- Source branch / head SHA: `experiments/training-approach`, commit located by the change ID above
- Target branch / tested target SHA: `experiments/training-approach` at the base commit above
- Integration owner: xuheping
- Shared branch / PR: local commit related to PR #1
- Dependencies / merge order: frozen GRU checkpoint and aligned broadcast frame map
- Conflicts / resolutions: none
- Combined validation: model and dashboard checks above
- Integration result: local on the target branch
- Cleanup: retain the current artifacts below the ignored data directory

## Resume here

Use the documented tactical checkpoint for new comparisons. The aligned
experiment artifacts remain under `data/external_possession/broadcast-gru/`.

## Outstanding state

- Uncommitted owned files: ignored outputs
- Other work observed: pre-existing untracked assets and scripts
- Blockers / access needed: none for local reproduction
- Failed / unrun checks: the scorer was rerun for each policy directory after an initial invocation at their common parent. Private scoring was unavailable.
- Cross-repository dependencies: none
- Publication / recipient acknowledgement: local commit only
