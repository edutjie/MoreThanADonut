# AGENTS.md

Behavioral guidelines for coding and collaboration. This is the canonical policy;
`CLAUDE.md` points here. Follow applicable directory instructions and the user's scope.

**Tradeoff:** Favor care over speed for consequential changes. Use judgment for
trivial tasks; keep planning, checks, and records proportionate to the work.

## 1. Think Before Coding

**State assumptions. Surface uncertainty and tradeoffs.**

- Read the relevant code and project instructions before proposing changes.
- State assumptions that affect the result. If plausible interpretations would
  lead to materially different work, explain them and ask before that work.
- If a simpler approach solves the problem, say so. Push back on unnecessary complexity.
- If essential information is missing, name the gap; continue independent work
  while resolving it. Do not silently invent requirements or project decisions.

## 2. Simplicity First

**Make the smallest change that solves the requested problem.**

- No speculative features, single-use abstractions, or unrequested configurability.
- Handle realistic failure cases; do not add machinery for impossible scenarios.
- Prefer existing patterns and dependencies over new infrastructure.
- If the solution is much longer than necessary, simplify it before delivering.
- Ask: "Would a senior engineer consider this overcomplicated?"

## 3. Surgical Changes

**Touch only what the task requires. Clean up your own changes.**

- Match the existing style. Do not refactor or reformat adjacent code without need.
- Remove imports, variables, or functions made unused by your changes.
- Mention unrelated defects or dead code; do not silently fix or delete them.
- Preserve other users' and agents' edits. Do not reset, stash, overwrite, stage,
  or commit unrelated work.
- Every changed line should trace to the request or a necessary consequence.

## 4. Goal-Driven Execution

**Define success, then verify it.**

- For multi-step work, state a brief plan with a check for each step.
- For a bug fix, reproduce the failure and add a regression test when practical.
- For new validation, check relevant valid and invalid inputs.
- For a refactor, check behavior before and after the change.
- Run checks appropriate to the risk. Documentation-only edits need consistency
  and link checks, not unrelated application tests.
- Inspect the final diff. Report what changed, what passed, what failed or was
  not run, and remaining limitations. Never present an unrun check as passed.

## 5. Multi-User, Multi-Agent Collaboration

**Make ownership clear and leave enough context for the next agent.**

### Start and coordinate

- Inspect the branch, working tree, and staged changes. Read recent full commit
  messages and relevant handoffs; follow older history when needed. An unborn
  repository has no commit history.
- For each substantive session, create `handoffs/<session-id>.md` from
  [the template](handoffs/TEMPLATE.md). Use
  `YYYYMMDDTHHMMSSZ-<user>-<agent>-<short-id>`; record the human, agent,
  starting commit, objective, and owned paths. Continue the same file when resuming.
- Prefer separate branches/worktrees for concurrent sessions. In a shared
  worktree, use non-overlapping ownership and serialize Git/index operations.
  Resolve ownership conflicts before editing overlapping paths; a handoff is not a lock.
- Verify handoff claims against current code. Local records do not reveal work
  on other machines or branches until shared.

### Branches, worktrees, and integration

- Use one topic branch per concurrent task, based on the latest fetched target
  branch (normally `main`). Name Codex branches `codex/<user>/<task>-<short-id>`;
  other agents should use an equally distinct owner/task name. Record the base SHA.
- Check existing worktrees before creating or switching branches. Do not reuse
  another session's active branch/worktree. Worktrees isolate working files;
  collaborators exchange committed branches, not worktree folders.
- Before handoff, commit owned work and share the branch/PR within authorization.
  Record source branch/head SHA, target branch, dependencies/merge order, checks,
  known conflicts, and one named integration owner. Uncommitted work is not mergeable.
- The integrator fetches current source and target refs and uses a clean,
  dedicated checkout or integration branch. Integrate one change at a time;
  follow required reviews and repository merge rules.
- Default to merging shared branches so contributor history survives. Rebase
  only unpublished, exclusively owned work unless affected collaborators agree.
  Do not force-push shared branches. If squash is required, retain all contributor
  attribution, change IDs, and handoff pointers in the integration record.
- For conflicts, inspect the common base and both changes and read both handoffs.
  Resolve the intended combined behavior; never blanket-select `ours` or `theirs`.
  If requirements disagree, coordinate with the owners before choosing behavior.
  Preserve separate session logs; reconcile shared policy/specification text deliberately.
- Check for unresolved conflicts, review the combined diff, and rerun affected
  tests plus relevant integration checks. A clean Git merge can still break
  schemas, imports, timing semantics, or metrics. If the target advances,
  integrate the new tip and revalidate before publishing; never force past a rejection.
- Record source/target SHAs tested, conflict decisions, checks, and resulting
  integration commit/PR when known in the integrator's handoff. Mark a branch
  integrated only after verifying its changes reached the intended target.
- Remove a branch/worktree only with its owner's agreement, after verifying
  integration and preserving any uncommitted, untracked, ignored, or unpushed
  work. Do not force-delete another session's worktree.

### Every commit

- Use [.gitmessage](.gitmessage): explain **Changes**, **Why**, **Validation**,
  and **Risks / follow-up** for that specific commit.
- Include `Human-Owner`, `Agent-Identity`, `Agent-Session`, `Change-Id`,
  and `Handoff` trailers. Use known identities; mark unavailable model names
  honestly and resolve an unknown human owner before committing.
- Use `<session-id>-<repo-slug>-<sequence>` for each logical change ID.
  Include the matching handoff entry in the same commit; do not put the
  containing commit's own hash inside its handoff.
- Review all staged changes. Preserve contributor attribution and handoff
  pointers through merges and squashes; do not rewrite shared history to add them.
- Attribution is not cryptographic signing. Preserve legitimate Git identity;
  do not fabricate signatures or use `Signed-off-by` merely to label an agent.
  For multiple contributors, repeat their identity/session/handoff records.

### Record and hand off

- Keep one file per session, not a shared global log. Update it at meaningful
  checkpoints with findings, decisions, validation, blockers, and next actions.
  Log concise evidence and outcomes, not every tool call or private reasoning.
- Record consequential communications with time, sender, recipient, and state:
  `requested`, `acknowledged`, or `resolved`. A written request is not proof
  of delivery. Recipients record acknowledgement in their own session file.
- Edit only your own handoff unless ownership is explicitly transferred.
  Put lasting accepted decisions in the owning specification and link them.
- Before ending, record status, uncommitted files, exact checks, remaining work,
  and who should continue. Distinguish completed work from committed, shared,
  merged, or deployed work; say when delivery or acknowledgement is pending.
- For cross-repository changes, keep a handoff in each repository with the same
  session ID, repository-specific change IDs, and counterpart links/dependencies.
- Keep secrets and sensitive participant data out of these records.
  Commit and share within the user's authorization. These files do not install
  enforcement hooks or send notifications.

## Repository context

MoreThanADonut generates frame-level possession predictions for More Than a
Hack 2027. [TACTICAL_BASELINE.md](TACTICAL_BASELINE.md) describes the team
pipeline. [DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), and
[SUBMISSION.md](SUBMISSION.md) document the organizer's data, scorer, and CSV
contract. Organizer clarifications sent to participants take precedence over
older repository guidance.

- Treat tactical video frame numbers as authoritative. Preserve the zero-based
  submission frame index and the one-based ATD conversion.
- Use the provided match for inference and the public labels for automated
  pipeline checks. Train and tune on external data with suitable usage rights.
  Keep competition assets out of hosted AI services and manual output review
  against the match video out of the workflow.
- Run `evaluation/validate_submission.py` on generated CSVs. Report the public
  window separately from full-match processing and private scoring.
- Keep contest media, tracking, homographies, restricted links, model files,
  recovered coordinates, generated predictions, and credentials out of Git.
  Record public model sources, settings, and aggregate verification results.
