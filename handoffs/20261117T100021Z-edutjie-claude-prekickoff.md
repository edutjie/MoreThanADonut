# Session handoff: pre-kickoff dead-ball rule

- Session: 20261117T100021Z-edutjie-claude-prekickoff
- User: edutjie
- Agent: Claude Code; claude-opus-5-5; implementation and verification
- Repository: MoreThanADonut
- Branch / worktree: `claude/edutjie/prekickoff-100021`; main checkout
- Base commit: `f1a113673865233a76c033cd77fa2bf9ef8a98f2` (`main` after the round-two merge; work began on `7070119`)
- Started: 2026-11-17T10:00:21Z
- Updated: 2026-11-17
- Status: ready-for-handoff
- Delivery: committed locally
- Objective: round-3 plan §1. Label graded frames before each half's detected kickoff as `D`, using provided inputs only.
- Owned paths: `kickoff.py`, `tests/test_kickoff.py`, this handoff
- Related sessions: round-two possession improvement (`codex/edutjie/possession-improvement-081314`)
- Recipient: edutjie, or the next execution agent

## Current outcome

`kickoff.py` is a standalone post-processing step for any `frame,label`
submission CSV. FAQ.md:38-40 says the second half is graded from frame
110,808 but kicks off at 113,742, and those frames are all `D`. Both TCN
candidates predicted live play there.

How the kickoff is found:
1. The Smart Tag `Kick Off (Start)` window for each half bounds the search.
2. Inside it, a tracking detector looks for the ball resting near the
   centre spot and then being struck away. Its thresholds were fixed a
   priori in metres and seconds.
3. Otherwise the detector falls back to the tag midpoint.

The published kickoff frame is a correctness reference only and is not an
input.

On the provided match, both halves fell back to the tag midpoint. In the
second half the ball is tracked on every frame of the search window but
never comes within 2 m of the centre spot (closest 3.34 m). In the first
half the ball is tracked on 85 of 225 window frames and passes within 2 m
on only 2 of them, too few for the stillness check. That half is a no-op
regardless, because its kickoff precedes the graded start. The thresholds
were deliberately not loosened to fit this match.

| Half | Graded from | Detected kickoff | Method | Published reference | Effect |
| --- | ---: | ---: | --- | ---: | --- |
| 1 | 11,185 | 11,159 | tag | 11,185 | no-op (before the graded start) |
| 2 | 110,808 | 113,722 | tag | 113,742 | `D` on 110,808–113,721 |

| Input CSV (local, ignored) | Frames set to `D` | Full-match spells before → after |
| --- | ---: | ---: |
| Original 113-feature TCN | 1,918 | 215 → 209 |
| Round-two TCN seed 733 | 2,192 | 231 → 224 |

Outputs are under `data/external_possession/prekickoff-20261117/<name>/possession.csv`
(SHA-256 `25a358d5…f391e` for the original TCN and `37986b3a…3afb3736` for
round two). Both pass `evaluation/validate_submission.py`, and a rerun was
byte-identical. The public window (11,185–18,684) cannot change, so no
public check was run.

Reproduce, with `MTAH_DATA_DIR` pointing at the match bundle:

```bash
python -m unittest discover -s tests
python kickoff.py --input <candidate>/possession.csv --output <out>/possession.csv
python evaluation/validate_submission.py <out>/possession.csv
```

An independent review (correctness, compliance and scoring lenses, each
finding re-checked by a refuter) confirmed three defects, all fixed before
commit:
- the rest anchor moved with the ball, so kicks below 7.5 m/s were missed;
- the confirm and no-return checks were cut off at the window end;
- label parsing truncated values silently.

Refuted: midpoint-fallback overshoot, out-of-range kickoffs, and `X`
forward-fill after the kickoff. None of them can occur on the provided
match's inputs.

## Changes by commit

### 20261117T100021Z-edutjie-claude-prekickoff-morethanadonut-01
- State: committed
- Changes: add `kickoff.py` (tag window parser, kickoff detector, pre-kickoff `D` rule, CSV CLI) and `tests/test_kickoff.py` (17 synthetic cases)
- Why: graded pre-kickoff frames were predicted as live play, which created 6–7 false possessions in each candidate
- Validation: 17 unit tests pass; both candidate CSVs rewritten, validated, and reproduced byte for byte; detected kickoff within 20 frames of the published reference
- Risks / follow-up: the tracking detector is untested on external kickoffs (no external data on this machine). A short model spell still straddles the true kickoff (tag midpoint 20 frames early). The rule is a CSV post-step, not yet inside the decoder.

## Findings and decisions

- 2026-11-17 | agent | accepted
  - Finding or decision: keep the a-priori detector thresholds and accept the tag fallback on the provided match.
  - Evidence: the second-half tracked ball sits at least 3.34 m from the centre spot throughout the window.
  - Implication / limits: changing the radius after seeing this match would be match-specific tuning. Validate any change on external kickoffs first.
- 2026-11-17 | agent | proposed
  - Finding or decision: write `D`, never `X`, on pre-kickoff frames.
  - Evidence: `evaluation/possession_metrics.py` forward-fills `X` with the previous label inside graded runs.
  - Implication / limits: none.

## Communication log

- 2026-11-17 | edutjie -> agent
  - State: resolved
  - Message / requested action: implement round-3 §1
  - Evidence / reply: this session

## Integration handoff

- Source branch / head SHA: `claude/edutjie/prekickoff-100021`; local branch tip
- Target branch / tested target SHA: `main` at `f1a1136`
- Integration owner: edutjie
- Shared branch / PR: local-only
- Dependencies / merge order: none; based on `main` after the round-two merge, new files only
- Conflicts / resolutions: none
- Combined validation: not run against the round-two tree
- Integration result: pending
- Cleanup: keep branch

## Resume here

1. On the host with the checkpoints, call `prekickoff_dead` inside `experiments/possession_round2_predict.py` (and `experiments/possession_study_predict.py` for the original TCN). Restart the Viterbi decode at each detected kickoff instead of rewriting after decoding. Check that the result equals the two-step output above, or explain the difference. Until then, the submission path is predict, then `kickoff.py`.
2. Validate the detector on external kickoffs with known frames (IDSSE, Metrica events) before considering any threshold change.

## Outstanding state

- Uncommitted owned files: none
- Other work observed: untracked plan files and older handoffs in the checkout, not owned here
- Blockers / access needed: external kickoff data for detector validation is not on this machine
- Failed / unrun checks: external detector validation; integration with round-two inference (checkpoints are not on this machine)
- Cross-repository dependencies: none
- Publication / recipient acknowledgement: pending
