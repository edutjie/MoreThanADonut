# Session handoff for the GRU dead-ball specialist

- Session: 20261116T173425Z-xuheping-codex-gru-dead-ball
- User: xuheping
- Agent: Codex, GPT-6, implementation and evaluation
- Repository: MoreThanADonut
- Branch / worktree: `experiments/training-approach`
- Base commit: `4b19a08bd754620877a73cd15a085edfa775e908`
- Started: 2026-11-16T17:34:25Z
- Updated: 2026-11-16T17:40:00Z
- Status: completed
- Delivery: local commit and ignored result artifacts
- Objective: fit an observed in-pitch live/dead specialist and test it with the frozen GRU
- Owned paths: `gru_dead_ball.py`, `GRU_DEAD_BALL.md`, this handoff
- Related sessions: [external data experiments](20261116T171814Z-xuheping-codex-external-data-study.md)
- Recipient: xuheping and the next possession modeling session

## Current outcome

The hard-negative fit won on external development at a blend weight of 0.15.
The candidate and reference both scored 0.6621 spellF1 on the public window.
Candidate dead-ball F1 was 0.8893, compared with 0.8946 for the reference.
Keep the original GRU as the default. The [report](../GRU_DEAD_BALL.md)
records the external breakdown, comparison limits, and commands.

## Changes by commit

### 20261116T173425Z-xuheping-codex-gru-dead-ball-morethanadonut-01

- State: committed
- Changes: binary training, development-only blend calibration, observed in-pitch evaluation, and the separate Clásico check
- Why: learn dead-ball evidence from official tracking while testing for the slow-live false-positive failure mode
- Validation: verified observed-only gating, finite logits and foul interval rules. Completed external checks and validated 199,319 contest rows. Repeated inference and verified that the zero-weight policy reproduced the prior GRU CSV byte for byte.
- Risks / follow-up: the Clásico check uses weak positive intervals with approximate clocks. Official dead-ball training uses four Bundesliga matches.

## Findings and decisions

- Added six candidate blend settings to the untouched GRU reference and froze the policy on external development.
- Uniform and hard-negative training stayed within 0.0008 on the held-out metrics, with identical slow-live error counts.
- No contest ground truth entered fitting or calibration.
- Keep contest artifacts in the ignored data directory.

## Communication log

- 2026-11-16T17:34Z | xuheping to Codex | requested the Bundesliga specialist and broadcast-transfer experiments

## Integration handoff

- Source branch / head SHA: `experiments/training-approach`, commit located by the change ID above
- Target branch / tested target SHA: `experiments/training-approach` at the base commit above
- Integration owner: xuheping
- Shared branch / PR: local-only
- Dependencies / merge order: requires the existing original GRU checkpoint, IDSSE files and the prior external source setup
- Conflicts / resolutions: none
- Combined validation: external calibration and held-out checks plus contest format and deterministic inference
- Integration result: local on target branch
- Cleanup: keep the worktree and local artifacts

## Resume here

Use `data/external_possession/gru-dead-ball/results.json` for aggregate results
and the original GRU from `data/external_possession/sequence-gru/gru.pt` as
the working reference.

## Outstanding state

- Uncommitted owned files: ignored specialist checkpoints and predictions in `data/external_possession/gru-dead-ball/`
- Other work observed: pre-existing untracked scripts, source files and results
- Blockers / access needed: none for reproducing the local benchmark
- Failed / unrun checks: private contest scoring was unavailable
- Cross-repository dependencies: source pins in the previous data audit
- Publication / recipient acknowledgement: local commit only
