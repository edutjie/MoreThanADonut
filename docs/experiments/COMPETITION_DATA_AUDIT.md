# Competition data provenance

The [competition rules](../../RULES.md) reserve the released five-minute
ground truth for pipeline checks and restrict human-produced data about the
supplied match. [The FAQ](../../FAQ.md) explains the local handling of match
assets and the use of external matches.

| Candidate | Fit and settings | Supplied match |
| --- | --- | --- |
| [Frozen continuous-track GRU](PLAYER_TRACK_QUALITY.md) | 12 MIT-licensed SkillCorner A-League matches. Four separate A-League development matches select the checkpoint. Ball and player corruptions are algorithmic. The decoder passes explicit costs calibrated on external A-League matches. | Tactical ATD for inference. Official public labels read by the scorer after the run is frozen. |
| [Aggregate TCN with 113 features](TCN_AUTOML_FOUNDATION.md) | The same external train and development split. AutoGluon fits on external training rows and passes the 113 inputs through. The TCN selects its checkpoint by external development spellF1 and uses the same explicit decoder costs. | Tactical ATD for inference. Official public labels read by the scorer after the run is frozen. |
| [External lagged LightGBM](EXTERNAL_POSSESSION.md) | SkillCorner A-League phase-derived labels and external development calibration. | Tactical ATD for inference and an automated public-window check after freezing. |

The restricted SoccerNet event experiment is outside these candidates.
The StatsBomb event files used in a separate specialist diagnostic concern
2019 and 2020 Barcelona–Real Madrid matches. The supplied competition
match is 2024–25 Barcelona–Athletic Club.
[External source inventory](EXTERNAL_TRAINING_DATA.md)

## Retired paths

- Early local logistic regression and GBM prototypes fitted released
  competition labels. The old LSTM training entry point also accepted
  labels for the supplied match. Their entry points, checkpoints and known
  outputs are outside the active workflow.
- The old v4/v5 tracking heuristics and follow-on blends used results from
  the public window to set parameters. The tactical reference uses the
  earlier heuristic and referee filters again. The shared Viterbi
  implementation still accepts explicit externally calibrated costs.
- The legacy MAE shortcut used the supplied tactical trajectory as a
  training input. This was coordinate-only self-supervision, which the
  organizer rules distinguish from training on labels. The active project
  workflow uses external training data, so the preparation script now
  accepts SoccerNet-v3D only. The mixed trajectory cache and model are retired.
  The local SoccerNet export contains no sequences long enough for this
  preparer. It exits without writing a training cache.

Legacy files moved out of their active paths are in a local, Git-ignored
quarantine. The official public labels and reference examples remain
available for automated checks. The historical record is preserved.

## Verification

After the cleanup, both frozen predictions matched their previous CSVs
byte for byte and passed validation for all 199,319 rows. The automated
7,500-frame public check reproduced 0.6897 spellF1 for the GRU and
0.6815 for the TCN. The dashboard no longer loads the retired v5 lane.

This verifies reproduction from the existing checkpoints. Submission
verification still requires giving the organizers access to the exact
checkpoint used. The repository policy keeps model weights out of Git.
