# Possession experiments

The [baseline guide](../../TACTICAL_BASELINE.md) gives the current inference command.
Run experiment commands from the repository root with
`python -m experiments.<module>` and the arguments in the relevant report.

If your checkout has legacy scratch scripts in `experiments/local/`, run them
from the repository root with `PYTHONPATH=. python experiments/local/<script>.py`.
That folder is local to your checkout.

## Training and temporal models

- [A-League possession and lagged boosting](EXTERNAL_POSSESSION.md)
- [IDSSE Bundesliga training](IDSSE_TRAINING.md)
- [GRU and Transformer](SEQUENCE_POSSESSION.md)
- [TCN, AutoGluon, MOMENT, and Mantis](TCN_AUTOML_FOUNDATION.md)
- [Player track quality](PLAYER_TRACK_QUALITY.md)
- [External training data audit](EXTERNAL_TRAINING_DATA.md)
- [Additional external data experiments](EXTERNAL_DATA_EXPERIMENTS.md)
- [LSTM](LSTM_EXPERIMENT.md)

## Ball recovery and stoppages

- [YOLO ball recovery](BALL_RECOVERY_EXPERIMENT.md)
- [Stoppage cues](STOPPAGE_EXPERIMENT.md)
- [Whistle detection](WHISTLE_EXPERIMENT.md)
- [Bundesliga dead-ball specialist](BUNDESLIGA_SPECIALIST.md)
- [GRU with a dead-ball specialist](GRU_DEAD_BALL.md)
- [Spell smoothing](SPELL_SMOOTHING.md)
- [SoccerNet ball action spotting](SN_BAS_2025_EXPERIMENT.md)
- [CALF action spotting](CALF_INTEGRATION_EXPERIMENT.md)

## Broadcast and historical comparisons

- [Broadcast alignment and tracking](BROADCAST_INTEGRATION_EXPERIMENT.md)
- [Aligned broadcast with GRU](BROADCAST_GRU_EXPERIMENT.md)
- [Home/away attribution](HA_ATTRIBUTION_ANALYSIS.md)
- [Historical findings](EXPERIMENT_FINDINGS.md)
- [v5 baseline](V5_DOCUMENTATION.md)
- [v5 optimization attempts](FINAL_OPTIMIZATION_ATTEMPTS.md)
- [v5 results](FINAL_RESULTS.md)
