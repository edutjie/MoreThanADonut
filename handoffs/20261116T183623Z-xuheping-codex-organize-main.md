# Session handoff: sync main and organize experiments

- Session: 20261116T183623Z-xuheping-codex-organize-main
- User: xuheping
- Agent: Codex, GPT-6, implementation and verification
- Repository: MoreThanADonut
- Branch / worktree: codex/xuheping/organize-experiments-183623, /home/xuheping/codes/MoreThanADonut
- Base commit: 94db68afb288237d2bd614f469e2f292e2ff30fc
- Started: 2026-11-16T18:36:23Z
- Updated: 2026-11-16T18:39:00Z
- Status: completed
- Delivery: organization commit prepared for push to origin on the topic branch
- Objective: update main from origin and organize the experiment files
- Owned paths: relocated reports under docs/experiments, relocated runners
  under experiments, updated links and imports, .gitignore, this handoff,
  and the relocation of loose local experiment files
- Related sessions: 20261116T182455Z-xuheping-codex-aligned-gru
- Recipient: xuheping or the next repository maintainer

## Current outcome

Local main tracks origin/main at 94db68a following a successful
`git pull --ff-only`. The previous local main had four divergent, unpublished
commits ending at 8871540, now preserved at `backup/main-before-sync-8871540`.
The experiment branch and its worktree are intact.

## Changes by commit

### 20261116T183623Z-xuheping-codex-organize-main-morethanadonut-01

- State: included in this commit
- Changes: Move 22 reports under docs/experiments and 18 Python runners plus
  a shell runner under experiments. Update current report links, module imports
  and documented commands. Move loose untracked scratch scripts into
  experiments/local and loose artifacts into data/local-experiments. Ignore
  established model, dataset and generated-output locations while preserving
  the tracked reference CSVs.
- Why: Make experiment source and results discoverable and keep generated
  and sensitive files out of the source tree.
- Validation: GRU inference and all four aligned-broadcast policy CSVs
  matched prior outputs byte for byte and all five passed full submission
  validation. Automated public GRU check returned 0.6897 spellF1 and
  0.9002 macroF1. Checked relative documentation links, module and dashboard
  help, and shell syntax. Legacy scratch experiments were not rerun.
- Risks / follow-up: A first import smoke check executed the legacy
  train_mae_simple.py top level, trained on the legacy tactical trajectory,
  and overwrote the pre-existing local models/ball_mae_trained.pt. No exact
  backup was found under the workspace or temporary files. Moved the newly
  produced checkpoint to ignored data/local-experiments/ball_mae_accidental_import.pt
  and guarded the executable script body. The replacement should not be used
  in experiments. No current GRU checkpoint was written by this check.

## Findings and decisions

- 2026-11-16T18:36:23Z | Codex | observed
  - A fast-forward pull on old main failed due to divergence. Renamed that local
    branch and created a fresh main tracking the remote before pulling again.
- Existing baseline commands retain their root paths. The relocated experiment
  runners use `python -m experiments.<module>` from the repository root.

## Communication log

- 2026-11-16 | xuheping -> Codex
  - State: resolved
  - Message / requested action: check out main, pull latest, and organize files.
  - Evidence / reply: session conversation, branch refs, and working-tree changes.

## Integration handoff

- Source branch / head SHA: codex/xuheping/organize-experiments-183623, this commit
- Target branch / tested target SHA: origin/main at 94db68a
- Integration owner: xuheping
- Shared branch / PR: origin/codex/xuheping/organize-experiments-183623 is the push target
- Dependencies / merge order: none
- Conflicts / resolutions: none
- Combined validation: see checks above
- Integration result: main synced, organization committed on the topic branch
- Cleanup: backup branch and pre-existing worktrees retained

## Resume here

1. Review the topic branch before integrating into main.
2. Restore the old MAE checkpoint from a separate machine backup if available.

## Outstanding state

- Uncommitted owned files: none after this commit.
- Other work observed: previous local handoff and ignored experiment inputs
  and outputs remain in the checkout.
- Blockers / access needed: separate backup required for the overwritten
  legacy MAE checkpoint.
- Failed / unrun checks: first import check exposed the MAE side effect.
  Legacy training and scratch scripts were not rerun after that check.
- Cross-repository dependencies: none.
- Publication / recipient acknowledgement: push requested in the session, remote ref verification follows commit creation.
