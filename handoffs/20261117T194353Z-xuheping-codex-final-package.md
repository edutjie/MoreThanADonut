# Session handoff: Absolute Donut final package

- Session: 20261117T194353Z-xuheping-codex-final-package
- User: xuheping
- Agent: Codex; model identifier unavailable; implementation and verification
- Repository: MoreThanADonut
- Branch: codex/xuheping/final-submission-194353
- Base commit: 6a6a8750ad577b506fd3ebe41de234adbd5382aa
- Status: completed
- Delivery: shared; PR #6 to main
- Objective: Package frozen learned spell scoring, reproduce CSV, document method, and open PR to main.
- Owned paths: final/, this handoff
- Recipient / integration owner: xuheping

## Findings and decisions

- User explicitly authorizes final model weights and generated submission in Git, overriding the general project exclusion for those artifacts only. Match inputs remain excluded.
- Preserve the four supplied member entries, including repeated names.

## Changes by commit

### 20261117T194353Z-xuheping-codex-final-package-morethanadonut-01
- State: committed; dbcfa83032891bd3e12da173a915233f6914fca8
- Changes: standalone final package, frozen weights, submission, methodology PDF and reproduction checks.
- Validation: two complete CUDA reconstructions, including isolated final-only copy, exactly matched frozen SHA-256. Official validator passed 199319 rows; public scorer returned 0.720000 spellF1 / 0.918671 macroF1 on 7500 frames. All 29 existing tests passed. Artifact-tamper rejection passed. Exported transform exactly matched the original fitted transform on both orientations of both halves. PDF reader and visual render verified all three pages. Whitespace check passed.
- Limits: CPU reconstruction differed on two H/A frames (D/X identical); command correctly failed checksum. Exact reproducibility verified on NVIDIA GB10, torch 2.15.0+cu132. Fresh dependency installation and other GPUs were not tested.

## Integration handoff

- Target: main
- Integration owner: xuheping
- Publication: https://github.com/edutjie/MoreThanADonut/pull/6 (open)
- Other work: existing untracked docs/plans and old training handoff preserved.

## Current outcome

Standalone final/ includes the frozen submission, both TCN checkpoints, numeric transform, learned segment scorer, inference definitions, dependencies, checksums, README, and three-page methodology PDF with two schematic diagrams and their generator. No competition tracking, video, homography, recovered coordinates or private labels are included. The user-authorized model and submission artifacts are ordinary Git files (all below 2 MB each).

- Source branch: codex/xuheping/final-submission-194353; implementation head dbcfa83032891bd3e12da173a915233f6914fca8
- Tested base/target: 6a6a8750ad577b506fd3ebe41de234adbd5382aa
- Known conflicts: none; existing experiment worktrees and untracked work preserved.
- Integration owner: xuheping; merge remains pending.

### 20261117T194353Z-xuheping-codex-final-package-morethanadonut-02
- State: committed; b6017012cadf87d15ea74f36b754fef10de79df7
- Changes: record verified publication and integration instructions.
- Why: final package is shared in PR #6; merge remains the human integration owner's action.
- Validation: GitHub returned the open PR with base main at 6a6a8750ad577b506fd3ebe41de234adbd5382aa and implementation head dbcfa83032891bd3e12da173a915233f6914fca8.
- Risks / follow-up: merge pending; keep worktree and existing experiments.

## Outstanding state

- Uncommitted owned files: none after publication-record commit.
- Other work: untracked docs/plans and prior training handoff remain untouched.
- Delivery: final package committed and pushed; PR created, not merged; form submission not performed.
- Continue: xuheping can review/merge PR #6 and submit final/Absolute Donut.csv plus final/Absolute Donut.pdf.

### 20261117T194353Z-xuheping-codex-final-package-morethanadonut-03
- State: committed; cc1585f
- Changes: explain nested external spellF1, scored data sources and source-balanced aggregation in README and methodology PDF; use first-person team wording at README opening. Refresh package checksums.
- Why: distinguish external test games (seven DFL/IDSSE Bundesliga and three Metrica) from training-only SkillCorner A-League data, and explain inner parameter selection versus outer evaluation.
- Validation: checked aggregation against experiment measure/source_mean functions and official spell metric; rebuilt and read three-page PDF; inspected rendered final page; all package checksums pass and submission CSV SHA-256 is unchanged.
- Risks / follow-up: documentation only; no inference or evaluation rerun needed. Update existing PR #6.

### 20261117T194353Z-xuheping-codex-final-package-morethanadonut-04
- State: ready to commit
- Changes: rewrite the technical document as a self-contained research methodology; derive 113 aggregate and 148 spatial features; add training, provider/match nested evaluation, A-League component and public-baseline tables. Correct all four authors in README, manifest and PDF metadata. Add aggregate evaluation evidence and refresh checksums.
- Why: readers should understand the submitted algorithm without development context; distinguish training-only A-League data from primary validation and component results from complete-pipeline results.
- Validation: re-ran official public evaluator on frozen final CSV and published baseline_smooth50.csv: 0.720000 / 0.918671 versus 0.325581 / 0.620733 spellF1 / macroF1. Verified provider averaging and external frame-count totals against saved reports; checked feature-count arithmetic against implementation. Read and visually reviewed all six PDF pages, including seven tables; confirmed four corrected author names and PDF metadata. All local documentation links and package checksums pass; submission SHA-256 unchanged.
- Limits: no independent A-League holdout for the complete submission (all twenty auxiliary games used in spatial training); available aggregate-TCN component tests are labeled explicitly. Inference code/weights unchanged; no model retraining or inference rerun needed.
- Accepted correction: user replaced repeated author entries with Eduardus Tjitrahardja, Louis Jefferson Zhang, Stuart Davis Tioniwar and Joel Van Doli Marpaung.
- Delivery: update PR #6; merge remains pending. Current methodology is six pages, superseding the initial three-page document.
