# Session handoff: competition rules and data provenance

- Session: 20261117T064940Z-edutjie-codex-rule-audit
- User: edutjie
- Agent: Codex, GPT-6, audit and cleanup
- Repository: MoreThanADonut
- Branch / worktree: codex/edutjie/possession-study-185042, primary checkout
- Base commit: e2f2722
- Started: 2026-11-17T06:49:40Z, audit checkout timestamp
- Updated: 2026-11-17T06:49:40Z
- Status: completed
- Delivery: local commit and live dashboard
- Objective: verify training provenance and remove prohibited paths from active use
- Owned paths: tactical and referee defaults, legacy experiment entry points,
  affected experiment documentation, dashboard documentation, and this handoff
- Related sessions: 20261116T185042Z-edutjie-codex-possession-study,
  20261117T054734Z-edutjie-codex-dashboard-tcn
- Recipient: edutjie

## Current outcome

The [provenance audit](../docs/experiments/COMPETITION_DATA_AUDIT.md) records
the external-only fit and selection paths for the GRU, TCN, and lagged LightGBM.
The two current frozen sequence candidates reproduce byte for byte. The local
dashboard retains the GRU and TCN and omits the retired v5 lane.

## Changes by commit

### 20261117T064940Z-edutjie-codex-rule-audit-morethanadonut-01

- State: included in this commit.
- Changes: restore pre-tuning heuristic and referee settings, remove the
  legacy public-trained model hook and retired runners, remove the tactical
  source from the ball preparation script, and label historical reports.
- Why: keep public-label fitting and public-window setting selection out of
  the active submission workflow.
- Validation: reproduced and validated both frozen 199,319-frame CSVs with
  public spellF1 0.6897 and 0.6815. Restored tracking-only output matched
  the earlier reference exactly. Checked baseline and TCN dashboard manifests,
  Python syntax, docs links, and staged whitespace. The external-only
  preparer accepted synthetic valid sequences and rejected the short-clip
  local export without writing a cache.
- Risks / follow-up: exact frozen checkpoint access still needs to be
  arranged for organizer verification before submission.

## Findings and decisions

- The old logistic and GBM checkpoints used released competition labels.
  The v5 transition cost was selected on the public window. Neither path
  supplies labels, features or constants to the external GRU and TCN candidates.
- Reversibly retired 130 files into the ignored local audit quarantine.
  The manifest is under `data/retired-rule-audit-20261117/`. The approval
  reviewer rejected a broad destructive cleanup. The scoped quarantine and
  reviewed edits removed the affected files from active use.
- The prior qualitative TCN preference is recorded separately from selection.

## Communication log

- 2026-11-17 | edutjie -> Codex
  - State: resolved
  - Message / requested action: audit the best methods for rule violations
    and remove any violating paths.
  - Evidence / reply: provenance report and verified frozen inference.

## Integration handoff

- Source branch / head SHA: codex/edutjie/possession-study-185042, this commit
- Target branch / tested target SHA: published experiment branch at e2f2722
- Integration owner: edutjie
- Shared branch / PR: cleanup local to the checkout
- Dependencies / merge order: experiment and TCN dashboard history
- Conflicts / resolutions: none
- Combined validation: see the audit report
- Integration result: checked locally
- Cleanup: retain the isolated quarantine and active external checkpoints

## Resume here

1. Give the organizers access to the exact chosen checkpoint for reproduction
   when preparing a submission.

## Outstanding state

- Uncommitted owned files: none after the audit commit.
- Other work observed: the pre-existing training-pickup handoff remains local.
- Blockers / access needed: organizer checkpoint delivery pending.
- Failed / unrun checks: no private score or organizer adjudication.
- Cross-repository dependencies: none.
- Publication / recipient acknowledgement: cleanup is local.
