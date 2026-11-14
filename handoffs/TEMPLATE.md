# Session handoff: <short task title>

- Session: <YYYYMMDDTHHMMSSZ-user-agent-short-id>
- User: <known name/handle; mark unconfirmed if needed>
- Agent: <tool; model identifier or unavailable; role>
- Repository: <name>
- Branch / worktree: <branch; local path if useful>
- Base commit: <hash or unborn>
- Started: <ISO-8601 UTC>
- Updated: <ISO-8601 UTC>
- Status: <active | blocked | ready-for-handoff | completed>
- Delivery: <uncommitted | committed locally | shared; branch/PR/commit if available>
- Objective: <authorized task and completion condition>
- Owned paths: <explicit files/directories>
- Related sessions: <predecessors, collaborators, counterpart repo, or none>
- Recipient: <agent/session or next agent on task>

## Current outcome

<What is implemented, verified, pending, or blocked. State how current this is.>

## Changes by commit

### <session-id>-<repo-slug>-01
- State: <planned | uncommitted | committed; known hash only from a later update>
- Changes: <concrete changes for this commit>
- Why: <reason>
- Validation: <commands/checks, results, limits; or not run and why>
- Risks / follow-up: <items or none>

<!-- Add an entry for each commit. The containing commit uses the same Change-Id
in its message. Do not insert that commit's own hash into this file. -->

## Findings and decisions

- <time> | <user / agent / session> | <observed / hypothesis / proposed / accepted>
  - Finding or decision: <concise statement>
  - Evidence: <path, prior commit, reproducible check, or authorized source>
  - Implication / limits: <what follows and what remains unknown>

## Communication log

- <time> | <sender user / agent / session> -> <recipient>
  - State: <requested | acknowledged | resolved>
  - Message / requested action: <concise content>
  - Evidence / reply: <link to actual communication or responding session;
    say local-only if it has not been delivered>

## Integration handoff

- Source branch / head SHA: <branch and committed tip; pending if uncommitted>
- Target branch / tested target SHA: <usually main; exact tip used for checks>
- Integration owner: <human/agent responsible for integration>
- Shared branch / PR: <remote ref or URL; local-only if not shared>
- Dependencies / merge order: <branches, changes, or counterpart repo; none if independent>
- Conflicts / resolutions: <files, intended behavior, rationale, owner agreement; or none>
- Combined validation: <checks on source + target together, results, or not run and why>
- Integration result: <pending | integrated; resulting commit/PR when known>
- Cleanup: <owner agreement and confirmation all work is preserved; or keep worktree>

## Resume here

1. <First concrete action; relevant file and expected result>
2. <Next action or none>

## Outstanding state

- Uncommitted owned files: <paths or none; do not include others' work as yours>
- Other work observed: <paths/ownership if known; otherwise unknown>
- Blockers / access needed: <items or none>
- Failed / unrun checks: <items and reasons or none>
- Cross-repository dependencies: <repo + branch/hash/handoff; or none>
- Publication / recipient acknowledgement: <evidence or pending>
