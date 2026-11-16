# Session handoff for the external training-data audit

- Session: 20261116T164836Z-xuheping-codex-external-data
- User: xuheping
- Agent: Codex, GPT-6, research and validation
- Repository: MoreThanADonut
- Branch / worktree: `experiments/training-approach`
- Base commit: `4dc7ab666c8c03ebb940b7a3b4d185d0a87a9731`
- Started: 2026-11-16T16:48:36Z
- Updated: 2026-11-16T16:54:00Z
- Status: completed
- Delivery: local commit and local untracked source data
- Objective: find additional external training sources for the possession models
- Owned paths: this handoff, `EXTERNAL_TRAINING_DATA.md`, and the three newly downloaded directories listed in the audit
- Related sessions: [Transformer benchmark](20261116T164230Z-xuheping-codex-sequence-transformer.md)
- Recipient: xuheping and the next possession modeling session

## Current outcome

Downloaded and audited nine historical SkillCorner games and three Metrica
games. Matched and downloaded complementary StatsBomb event files for two
SkillCorner games. Verified provider documentation for the two access-gated
leads. The current benchmark artifacts predate this audit.

## Changes by commit

### 20261116T164836Z-xuheping-codex-external-data-morethanadonut-01

- State: committed
- Changes: record pinned source revisions, checked coverage, and label constraints
- Why: prioritize extra H/A supervision and collect in-pitch stoppage examples
- Validation: parsed all nine legacy tracking files, checked Metrica event frame IDs against tracking, and joined two StatsBomb match IDs by date and opposition
- Risks / follow-up: legacy null labels remain unknown; event-derived D labels and StatsBomb alignment need validation before training

## Findings and decisions

- Legacy SkillCorner provides 343,250 H/A frames. For 76,119 of those frames the ball XY is missing.
- Metrica provides three aligned match sources, including explicit foul and offside event markers. These markers provide candidate intervals when paired with compatible restarts.
- Keep the current external development and test matches fixed when measuring gains.

## Communication log

- 2026-11-16T16:48:36Z | xuheping to Codex | requested additional external training data

## Integration handoff

- Source branch / head SHA: `experiments/training-approach`, audit commit identified by the change ID above
- Target branch / tested target SHA: `experiments/training-approach` at the base commit above
- Integration owner: xuheping
- Shared branch / PR: local-only
- Dependencies / merge order: documentation follows the sequence model benchmark
- Conflicts / resolutions: none
- Combined validation: source and data checks above
- Integration result: local on target branch
- Cleanup: keep the worktree and local downloads

## Resume here

1. Adapt the legacy match loader and compare adding those matches as masked H/A supervision.
2. Build and audit Metrica foul-to-restart labels as a separate experiment. Verify StatsBomb clocks by half before using those logs for any frame targets.

## Outstanding state

- Uncommitted owned files: downloaded datasets are deliberately untracked under `external_data/`
- Other work observed: pre-existing untracked scripts, results, model files and a handoff
- Blockers / access needed: Gradient request form and SoccerTrack Hugging Face acceptance for those leads
- Failed / unrun checks: Game 1 Metrica opening kickoff has untracked end frame 0. New training and contest scoring were outside the audit.
- Cross-repository dependencies: public release pins in the audit
- Publication / recipient acknowledgement: local commit only
