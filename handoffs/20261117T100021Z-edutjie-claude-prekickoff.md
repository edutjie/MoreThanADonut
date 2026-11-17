# Session handoff: pre-kickoff dead-ball rule

- Session: 20261117T100021Z-edutjie-claude-prekickoff
- User: edutjie
- Agent: Claude Code; claude-opus-5-5; implementation and verification
- Repository: MoreThanADonut
- Branch / worktree: `claude/edutjie/prekickoff-100021`. Local main checkout; GPU runs in `/tmp/morethanadonut-prekickoff-100021` on hci-dgx-xhp-campus.
- Base commit: `f1a113673865233a76c033cd77fa2bf9ef8a98f2` (`main` after the round-two merge; work began on `7070119`)
- Started: 2026-11-17T10:00:21Z
- Updated: 2026-11-17
- Status: ready-for-handoff
- Delivery: committed locally
- Objective: round-3 plan §1. Label graded frames before each half's detected kickoff as `D`, using provided inputs only. Wire the rule into both TCN predict scripts and validate the detector on external kickoffs.
- Owned paths: `kickoff.py`, `tests/test_kickoff.py`, `experiments/kickoff_validation.py`, the `--prekickoff` additions in `experiments/possession_round2_predict.py` and `experiments/possession_study_predict.py`, the `start` argument of `decode_sampled` in `experiments/possession_study.py`, `docs/experiments/PREKICKOFF.md`, and this handoff
- Related sessions: round-two possession improvement (merged as `f1a1136`)
- Recipient: edutjie, or the next execution agent

## Current outcome

The full results are in [the report](../docs/experiments/PREKICKOFF.md).

- **Rule.** `kickoff.py` finds each half's kickoff and sets every graded
  frame before it to `D`.
  - The search is bounded by the Smart Tag `Kick Off (Start)` window.
  - A ball observation near the centre spot that then moves clearly away
    and does not return marks the kickoff.
  - Otherwise the tag midpoint is used.
- **Integration.** With `--prekickoff`, both predict scripts also restart
  the Viterbi decoder state at the kickoff. The model still sees the whole
  half as context. With the flag off, they reproduce the frozen CSVs byte
  for byte.
- **External validation.** On 38 IDSSE and Metrica restarts, using
  natural tracks and a 3 m spot radius chosen on IDSSE:
  - 74% of detections fall within 2 frames, and 93% when the ball is
    within 3 m of the spot;
  - with the published 41% ball dropout, about 43% do;
  - detections are never more than 2 frames early;
  - late ones are mostly about 1 s.
- **Provided match.**
  - The second half falls back to the tag midpoint, 113,722, which is 20
    frames before the reference. 2,914 graded frames start as `D`.
  - The first half detects 11,188 by tracking, 3 frames late, so 3 graded
    frames become `D`.
  - Public spellF1 is unchanged (0.6815 and 0.6800). Macro F1 drops by
    0.0005 from those 3 frames.
  - Full-match possessions: 215 → 209 for the original TCN and 231 → 224
    for round two.

## Changes by commit

### 20261117T100021Z-edutjie-claude-prekickoff-morethanadonut-01
- State: committed
- Changes: add `kickoff.py` (tag window parser, kickoff detector, pre-kickoff `D` rule, CSV CLI) and `tests/test_kickoff.py`
- Why: graded pre-kickoff frames were predicted as live play, which created 6–7 false possessions in each candidate
- Validation: unit tests passed; both candidate CSVs rewritten, validated, and reproduced byte for byte
- Risks / follow-up: superseded in part by the detector redesign in -02

### 20261117T100021Z-edutjie-claude-prekickoff-morethanadonut-02
- State: committed
- Changes:
  - replace the rest-then-strike detector with the departure detector;
  - add an external validation runner with a declared radius grid;
  - set the spot radius to 3 m from IDSSE;
  - add `--prekickoff` to both predict scripts, restarting the decode at the kickoff;
  - add `decode_after_kickoff` and `kickoff_offset`;
  - add the report.
- Why:
  - the first detector fired on only 4–8% of external kickoffs, because external trackers freeze or hide the ball while it waits on the spot;
  - the submission must regenerate from the predict scripts.
- Validation:
  - 25 unit tests pass;
  - on DGX, with the flag off, both frozen CSVs reproduced byte for byte;
  - with the flag on, the outputs are deterministic and equal the post-processing CLI (`737cc552…` and `8cdce512…`);
  - both outputs passed the validator, then one public correctness check each;
  - an independent review of the final diff found four real defects, all fixed before the final DGX run: rest jitter read as the strike, the no-return window timing, a truncated lookahead, and a validation reference radius equal to the grid maximum. Outputs were unchanged by the fixes.
- Risks / follow-up:
  - the radius is on the edge of the declared grid;
  - the dropout conditions are synthetic;
  - the second half still relies on the tag midpoint;
  - the short A spell straddling the reference kickoff remains.

## Findings and decisions

- 2026-11-17 | agent | accepted
  - Finding or decision: replace the rest-then-strike detector with a departure detector.
  - Evidence: IDSSE holds the ball's last live position about 53 m away until the kickoff frame, and Metrica has no ball before the kick. The first design fired on 4–8% of external kickoffs.
  - Implication / limits: the change is motivated by external data only.
- 2026-11-17 | agent | accepted
  - Finding or decision: spot radius 3 m, chosen on IDSSE from the grid {1, 2, 3} m declared before the run.
  - Evidence: IDSSE within-2-frame share 0.379 / 0.418 for 1 / 3 m (0.396 at 2 m), with references taken inside the centre circle; Metrica is identical across radii.
  - Implication / limits: the choice sits on the grid edge and the score still rises with the radius. The grid was not widened, because the provided match's second-half ball distance (3.34 m) was already known. That half still uses the tag fallback.
- 2026-11-17 | agent | accepted
  - Finding or decision: keep the first-half detection at 11,188 (3 frames after the reference) and do not adjust for it.
  - Evidence: it matches the external late tail. Any correction derived from this match would be tuning on the test match.
  - Implication / limits: macro F1 on the public window drops by 0.0005; spellF1 is unchanged.
- 2026-11-17 | agent | accepted
  - Finding or decision: write `D`, never `X`, on pre-kickoff frames.
  - Evidence: `evaluation/possession_metrics.py` forward-fills `X` with the previous label inside graded runs.
  - Implication / limits: none.

## Communication log

- 2026-11-17 | edutjie -> agent
  - State: resolved
  - Message / requested action: implement round-3 §1; then wire it into the predict scripts and validate the detector externally on DGX, and report the evaluation
  - Evidence / reply: this session; report in `docs/experiments/PREKICKOFF.md`

## Integration handoff

- Source branch / head SHA: `claude/edutjie/prekickoff-100021`; local branch tip
- Target branch / tested target SHA: `main` at `f1a1136`
- Integration owner: edutjie
- Shared branch / PR: local-only. The DGX worktree has the same commits, fetched from a bundle.
- Dependencies / merge order: none
- Conflicts / resolutions: none
- Combined validation: run on DGX against `f1a1136` plus this branch
- Integration result: pending
- Cleanup: keep the DGX worktree `/tmp/morethanadonut-prekickoff-100021` until this branch merges. Its untracked `external_data` link is local only.

## Resume here

1. Use `--prekickoff` in whichever predict command produces the submission. The outputs are under `data/external_possession/prekickoff-20261117/v3/` on DGX, and the validation is in `validation-v3/`.
2. Next plan item: §2, the goal-to-restart window. It can reuse `detect_kickoff` on `Kick Off (After Goal)` tags.

## Outstanding state

- Uncommitted owned files: none
- Other work observed: untracked plan files and older handoffs in the local checkout; others' worktrees and untracked files on DGX, all untouched
- Blockers / access needed: none
- Failed / unrun checks: none outstanding. The private score is unknown.
- Cross-repository dependencies: none
- Publication / recipient acknowledgement: pending
