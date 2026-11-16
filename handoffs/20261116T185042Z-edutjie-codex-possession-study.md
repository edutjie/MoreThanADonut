# Session handoff: possession architecture and AutoML experiments

- Session: 20261116T185042Z-edutjie-codex-possession-study
- User: edutjie
- Agent: Codex, GPT-6, experiment implementation and evaluation
- Repository: MoreThanADonut
- Branch / worktree: codex/edutjie/possession-study-185042, primary checkout
- Base commit: 5b467ca6e9c331468d5e26a1a7a7244ca3d740cf
- Started: 2026-11-16T18:50:42Z
- Updated: 2026-11-16T19:10:04Z
- Status: completed
- Delivery: local topic branch
- Objective: compare two TCNs, AutoGluon feature engineering and AutoML,
  and frozen MOMENT and Mantis representations with the current GRU.
- Owned paths: experiments/possession_study*.py, experiments/possession_tcn.py,
  experiments/possession_foundation.py, docs/experiments/README.md,
  docs/experiments/TCN_AUTOML_FOUNDATION.md, and this handoff.
- Related sessions: 20261116T183623Z folder reorganization
- Recipient: edutjie

## Current outcome

The [report](../docs/experiments/TCN_AUTOML_FOUNDATION.md) records the
comparison under external-only fitting and checkpoint selection. The frozen
continuous-track GRU remains the designated baseline. Checkpoints, caches,
full-match predictions, and scores are local under
`data/external_possession/tcn-study/`.

## Changes by commit

### 20261116T185042Z-edutjie-codex-possession-study-morethanadonut-01

- State: included in this local commit.
- Changes: reproducible preparation, training, and frozen inference for the
  TCN, AutoGluon and foundation-model studies.
- Why: assess transfer, ball gaps and in-pitch stoppages with the same
  official frame grid and shared spell decoder.
- Validation: external match-disjoint benchmarks, submission validation and
  automated public scoring of all 15 frozen CSVs, exact numeric pass-through of all 1,871,380
  dev/test rows (both orientations), syntax checks, and synthetic player
  permutation, missing track, and window alignment checks.
- Risks / follow-up: one seed per neural architecture, two held-out
  Bundesliga matches, and no private score. Foundation results are frozen
  2 Hz probes. AutoML wall-clock stopping can change selected iterations.

## Findings and decisions

- Preserve the currently designated baseline. The experiment also supplies
  separate observations for frame classification, masked transfer, and
  in-pitch dead balls.
- The default AutoGluon transformation kept all 113 numeric columns. The
  separate rolling comparison supplied 185 columns, of which 175 survived.
- The 240 s dense AutoML run only trained LightGBM. The budgeted run
  trained LightGBM, CatBoost, ExtraTrees, and XGBoost. The Torch model
  timed out before producing a candidate in the ensemble run, then completed
  with a separate 230-second GPU allocation. The budgeted ensemble used
  approximately 74% XGBoost and 26% LightGBM.
- Foundation weights were downloaded from the public repositories.
  Tactical inference ran with the checkpoint loader offline.

## Communication log

- 2026-11-16 | edutjie -> Codex
  - State: resolved
  - Message / requested action: run both TCN comparisons, AutoGluon feature
    generation and modeling, and frozen foundation-model experiments.
    Use edutjie as the human identity.
  - Evidence / reply: session conversation and experiment report.

## Integration handoff

- Source branch / head SHA: codex/edutjie/possession-study-185042, this commit
- Target branch / tested target SHA: main at 94db68a, with organization commit
  5b467ca applied
- Integration owner: edutjie
- Shared branch / PR: local experiment branch
- Dependencies / merge order: 5b467ca folder reorganization
- Conflicts / resolutions: none observed on the tested base
- Combined validation: see the comparison report
- Integration result: local commit, integration pending
- Cleanup: retain the prior worktrees and experiment outputs

## Resume here

1. Review the report to choose any follow-up evaluations.
2. For a full-match prediction, load one frozen candidate with the matching
   dependency directory and the documented inference command.

## Outstanding state

- Uncommitted owned files: none after the session commit.
- Other work observed: the pre-existing training pickup handoff remains local.
- Blockers / access needed: none.
- Failed / unrun checks: private scoring is unavailable. Larger or fully
  fine-tuned foundation encoders were outside this run.
- Cross-repository dependencies: none.
- Publication / recipient acknowledgement: local experiment delivery only.
