# Session handoff: In-pitch stoppage signals

- Session: 20261116T154326Z-xuheping-codex-stoppage-signals
- User: xuheping
- Agent: Codex (GPT-6), publication and experiments
- Branch: experiments/training-approach
- Base commit: d5f3e12f57ace6b60564caa698d812062e294220
- Started: 2026-11-16T15:43:26Z
- Status: active
- Integration owner: xuheping
- Owned paths: external possession results, stoppage experiments and this handoff

## Checkpoint before cue experiments

- Frozen lagged LightGBM is the working reference. It was selected externally,
  validated over 199,319 frames and scored 0.6452 spellF1 on the public window.
- Match-wide sparse YOLO recovery tied 0.6452. A dense public-window diagnostic
  scored 0.6194. See [the recovery report](../BALL_RECOVERY_EXPERIMENT.md).
- Aggregate Smart Tag audit found five 35-second free-kick windows. The
  existing slow-to-fast tracking heuristic found a plausible in-pitch restart
  in one window. Its cue is outside the public scoring window. No public
  labels were read for this audit.
- The branch contains modular commits for external training, model
  comparisons and the dashboard, explicit lags, and YOLO recovery.

## Resume here

1. Verify the checkpoint is on the remote experiment branch.
2. Calibrate in-pitch stoppage and restart cues on external football tracking
   and event data.
3. Benchmark a whistle detector on openly licensed, labeled external audio.
   Check audio availability and cue alignment on the local contest feeds.

## Changes by commit

### 20261116T154326Z-xuheping-codex-stoppage-signals-morethanadonut-01

- Changes: log the frozen reference and pre-experiment handoff.
- Validation: existing official public scoreboard and CSV validator output,
  aggregate Smart Tag audit, and Git whitespace and scope review.
