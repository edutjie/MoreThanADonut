# Session handoff: frozen TCN dashboard lane

- Session: 20261117T054734Z-edutjie-codex-dashboard-tcn
- User: edutjie
- Agent: Codex, GPT-6, dashboard implementation and verification
- Repository: MoreThanADonut
- Branch / worktree: codex/edutjie/possession-study-185042, primary checkout
- Base commit: 35ab1da, the local experiment commit before rebasing
- Started: 2026-11-17T05:47:34Z
- Updated: 2026-11-17T05:49:00Z
- Status: completed
- Delivery: dashboard running locally, branch prepared for publication
- Objective: show the TCN trained on the frozen GRU features and push the current work
- Owned paths: dashboard/serve.py, dashboard/README.md, and this handoff
- Related sessions: 20261116T185042Z-edutjie-codex-possession-study
- Recipient: edutjie

## Current outcome

The dashboard at `http://127.0.0.1:8767/` shows the frozen TCN prediction
next to the GRU and retains the five previous prediction lanes. A default
launch adds the TCN automatically when its local CSV exists. Explicit
`--prediction` flags still choose the requested lane set.

## Changes by commit

### 20261117T054734Z-edutjie-codex-dashboard-tcn-morethanadonut-01

- State: included in this commit.
- Changes: add the optional default TCN comparison lane and document how
  to load its frozen prediction.
- Why: make the 0.6815 public spellF1 run directly comparable in the dashboard.
- Validation: the CSV passed `evaluation/validate_submission.py`.
  Independently reconstructed all 199,319 labels from the default and live
  dashboard manifests and matched the CSV exactly. The live broadcast
  manifest retained its previous predictions plus the TCN. Python syntax
  and Git whitespace checks passed.
- Risks / follow-up: the TCN CSV remains a local artifact. Generate it with
  the [frozen inference command](../docs/experiments/TCN_AUTOML_FOUNDATION.md#reproduce)
  on another checkout.

## Findings and decisions

- The default AutoGluon transform passed the original 113 valid feature
  values through. The lane uses the label “TCN with the frozen GRU’s features.”
- Main incorporated the folder reorganization in f29ee31. Its tree matched
  the prior base 5b467ca, allowing the unpublished experiment commits to
  be placed on current main before publication.

## Communication log

- 2026-11-17 | edutjie -> Codex
  - State: resolved
  - Message / requested action: display the TCN prediction, commit and push.
  - Evidence / reply: live dashboard, commit, and remote branch verification.

## Integration handoff

- Source branch / head SHA: codex/edutjie/possession-study-185042, this commit
- Target branch / tested target SHA: origin/main at f29ee31
- Integration owner: edutjie
- Shared branch / PR: origin/codex/edutjie/possession-study-185042 is the push target
- Dependencies / merge order: experiment commit followed by this dashboard commit
- Conflicts / resolutions: none expected. The old and new bases have identical trees.
- Combined validation: live and default dashboard manifest checks above
- Integration result: branch prepared for push
- Cleanup: retain the pre-existing worktrees and ignored local artifacts

## Resume here

1. Refresh the local dashboard or start it with the documented command.

## Outstanding state

- Uncommitted owned files: none after this commit.
- Other work observed: pre-existing untracked training-pickup handoff.
- Blockers / access needed: none.
- Failed / unrun checks: browser rendering was checked through the manifest and
  existing lane renderer. No video review was performed.
- Cross-repository dependencies: none.
- Publication / recipient acknowledgement: remote branch verification follows the commit.
