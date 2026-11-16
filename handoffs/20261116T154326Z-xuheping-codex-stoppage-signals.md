# Session handoff: In-pitch stoppage signals

- Session: 20261116T154326Z-xuheping-codex-stoppage-signals
- User: xuheping
- Agent: Codex (GPT-6), publication and experiments
- Branch: experiments/training-approach
- Base commit: d5f3e12f57ace6b60564caa698d812062e294220
- Started: 2026-11-16T15:43:26Z
- Updated: 2026-11-16T15:54:32Z
- Status: completed
- Delivery: experiment branch
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

## Experiment scope

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

### 20261116T154326Z-xuheping-codex-stoppage-signals-morethanadonut-02

- Changes: add an external event-aligned benchmark and a local free-kick
  experiment around frozen LightGBM.
- Validation: four external development games and three held-out games showed
  unchanged spellF1 across the tested cue policies. The selected paired rule
  produced no contest intervals. All three contest outputs passed 199,319-frame
  validation and matched the frozen CSV exactly.
- Findings: official free-kick execution times were a median 1.36 seconds from
  the nearest labeled D-to-live transition across 155 usable IDSSE events.
- Evidence: [stoppage experiment report](../STOPPAGE_EXPERIMENT.md).

### 20261116T154326Z-xuheping-codex-stoppage-signals-morethanadonut-03

- Changes: train and test a spectral whistle detector on public B-Human
  RoboCup audio, and add local tactical and verified broadcast cue extraction.
- Validation: held-out event F1 was 0.7097 learned versus 0.1967 tonal,
  over 12 annotated whistles. The tactical detectors emitted 1 and 25
  candidates respectively. The learned broadcast detector emitted 2, with 1
  mapped to a tactical frame. No paired intervals were applied.
- Evidence: [whistle experiment report](../WHISTLE_EXPERIMENT.md).

## Publication checkpoint

- The pre-experiment checkpoint was verified on origin at
  `527dc0e294caf0fb2085675f660de9a0fa3d44df`.
- The frozen lagged LightGBM remains the reference at 0.6452 public spellF1.
  The experimental runs did not change the CSV. Private scoring is unknown.
- Local audio data, weights and cue CSVs are excluded from publication. The
  2024 NaoDevils dataset and Metrica sample matches were identified for future
  validation. They were outside the completed training run.
