# Session handoff for possession improvement, round two

- Session: 20261117T081314Z-edutjie-codex-possession-improvement
- User: edutjie
- Agent: Codex, GPT-6, experiment implementation and evaluation
- Repository: MoreThanADonut
- Branch / worktree: `codex/edutjie/possession-improvement-081314`, `/tmp/morethanadonut-round2-081314`
- Base commit: `345dff2181254fe08c0c744541c50e5aee4e8412`
- Started: 2026-11-17T08:13:14Z
- Updated: 2026-11-17
- Status: completed
- Delivery: committed locally on the topic branch
- Objective: execute `docs/plans/POSSESSION_IMPROVEMENT_PLAN.md`
- Owned paths: round-two experiment runners, results report and this handoff
- Related sessions: [round one](20261117T072920Z-edutjie-codex-possession-execution.md)
- Recipient: edutjie

## Current outcome

[Round-two report](../docs/experiments/POSSESSION_IMPROVEMENT.md) documents
the confirmed mixed-data TCN (0.6366 ± 0.0028 held-out spellF1 over five seeds,
against 0.6220 for the frozen GRU). The selected model is dev seed 733 with
fixed Viterbi. It scored 0.6346 on the external holdouts and passed the
full-match validator and single public correctness check. Both independent
CSV runs matched byte for byte. The ensemble, decoder and adaptation
proposals missed their adoption gates. The optional observation matching
stage was skipped after confirmation of the mixed-data TCN.

## Changes by commit

### 20261117T081314Z-edutjie-codex-possession-improvement-morethanadonut-01

- State: included in the round-two topic branch commit
- Changes: five-seed training, native emissions, frozen ensemble and decoder comparisons, track-only adaptation, deterministic competition inference and audit report
- Why: replicate the strongest round-one TCN across seeds and isolate gains from training mix, ensembling and decoding
- Validation: matched 12 frozen-GRU halves, verified native Viterbi and exhaustive small HSMM cases, passed same-team-stoppage and halftime fixtures, and validated the 199,319-frame competition CSV after a byte-identical replay
- Risks / follow-up: two Bundesliga holdouts account for most of the gain. The lower-cost decoder missed its development gate, and the official private score remains unknown.

## Findings and decisions

- Dev fold selected seed 733 and the 50/50 log-probability ensemble. Holdout confirmation admitted the mixed-data TCN family and the individual fallback, while the ensemble's incremental gain fell short.
- On seed 733, reducing turnover/stoppage settings to (2.5, 5) improved held-out spellF1 by 0.0083 and recovered 18 short spells. Its dev gain was 0.0044 versus the 0.005 gate. The frozen CSV keeps (5, 10).
- A smoke run exposed an overlapping native Viterbi traceback buffer. It was corrected, and its first partial seed artifact was isolated before retraining and dev selection. The final first seed reproduced round one's training trajectory to four decimals.
- The single public check used the frozen individual TCN and recorded 0.6800 spellF1, 0.8576 macro F1 and 0.7697 D F1. No public metric re-entered selection.

## Communication log

- 2026-11-17 | edutjie to Codex | requested execution of `docs/plans/POSSESSION_IMPROVEMENT_PLAN.md`
- 2026-11-17 | Codex to edutjie | executed the experiments and returned this report and local topic branch

## Integration handoff

- Source branch / head SHA: `codex/edutjie/possession-improvement-081314`; use the branch tip
- Target branch / tested target SHA: round-one `codex/edutjie/possession-execution-072920` at `345dff2181254fe08c0c744541c50e5aee4e8412`
- Integration owner: edutjie
- Shared branch / PR: local
- Dependencies / merge order: round one's experiment utilities and pinned local caches
- Conflicts / resolutions: none during implementation
- Combined validation: commands in the report cover the round-two branch and its round-one base
- Integration result: pending
- Cleanup: preserve both worktrees and the host-local ignored artifacts

## Resume here

1. Open the report for the per-match summary and exact CLI commands.
2. Get the seed-733 checkpoint and the complete evidence index from the experiment host under `data/external_possession/round2-20261117/`.

## Outstanding state

- Uncommitted owned files: none after the topic commit
- Other work observed: the source checkout's untracked plans and pre-existing handoff were preserved
- Blockers / access needed: the private scorer is organizer-only
- Failed / unrun checks: the optional observation-matching stage was skipped. Previously parked video and restart experiments were left parked.
- Publication / recipient acknowledgement: local branch and this conversation
