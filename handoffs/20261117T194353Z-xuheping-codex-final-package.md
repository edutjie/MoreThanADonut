# Session handoff: Absolute Donut final package

- Session: 20261117T194353Z-xuheping-codex-final-package
- User: xuheping
- Agent: Codex; model identifier unavailable; implementation and verification
- Repository: MoreThanADonut
- Branch: codex/xuheping/final-submission-194353
- Base commit: 6a6a8750ad577b506fd3ebe41de234adbd5382aa
- Status: ready-for-handoff
- Delivery: commit and PR publication pending
- Objective: Package frozen learned spell scoring, reproduce CSV, document method, and open PR to main.
- Owned paths: final/, this handoff
- Recipient / integration owner: xuheping

## Findings and decisions

- User explicitly authorizes final model weights and generated submission in Git, overriding the general project exclusion for those artifacts only. Match inputs remain excluded.
- Preserve the four supplied member entries, including repeated names.

## Changes by commit

### 20261117T194353Z-xuheping-codex-final-package-morethanadonut-01
- State: ready to commit
- Changes: standalone final package, frozen weights, submission, methodology PDF and reproduction checks.
- Validation: two complete CUDA reconstructions, including isolated final-only copy, exactly matched frozen SHA-256. Official validator passed 199319 rows; public scorer returned 0.720000 spellF1 / 0.918671 macroF1 on 7500 frames. All 29 existing tests passed. Artifact-tamper rejection passed. Exported transform exactly matched the original fitted transform on both orientations of both halves. PDF reader and visual render verified all three pages. Whitespace check passed.
- Limits: CPU reconstruction differed on two H/A frames (D/X identical); command correctly failed checksum. Exact reproducibility verified on NVIDIA GB10, torch 2.15.0+cu132. Fresh dependency installation and other GPUs were not tested.

## Integration handoff

- Target: main
- Integration owner: xuheping
- Publication: PR requested, pending
- Other work: existing untracked docs/plans and old training handoff preserved.

## Current outcome

Standalone final/ includes the frozen submission, both TCN checkpoints, numeric transform, learned segment scorer, inference definitions, dependencies, checksums, README, and three-page methodology PDF with two schematic diagrams and their generator. No competition tracking, video, homography, recovered coordinates or private labels are included. The user-authorized model and submission artifacts are ordinary Git files (all below 2 MB each).

- Source branch: codex/xuheping/final-submission-194353
- Tested base/target: 6a6a8750ad577b506fd3ebe41de234adbd5382aa
- Known conflicts: none; existing experiment worktrees and untracked work preserved.
- Integration owner: xuheping; merge remains pending.
