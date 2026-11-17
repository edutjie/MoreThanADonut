# Absolute Donut: Learned Spell Scoring for Team Possession

**Team:** Absolute Donut

**Authors:** Eduardus Tjitrahardja, Louis Jefferson Zhang, Stuart Davis Tioniwar, Joel Van Doli Marpaung

We predict home possession (H), away possession (A), dead ball (D), or ungraded (X) for every tactical-video frame. Our method combines a temporal stoppage model, a spatial team-control model, and learned selection of complete possession segments. It produces 199,319 zero-based frame labels at 25 Hz.

## 1. Method

The inputs are automatic player/ball tracking, XML slot-to-team metadata, and automatic kickoff Smart Tags. Normalized coordinates are converted to metres using the supplied 105.3 by 68.0 m pitch. Track slots are pooled by team; persistent player identities are not required. Features and decoding are computed independently within each half.

![Inference architecture](figures/architecture.png)

### Tracking features and stoppage estimation

The stoppage model uses 113 scalar features at 10 Hz. Distances are computed from player and ball coordinates; speeds are displacement divided by elapsed time. The features comprise 15 shared ball summaries, 40 team summaries, 56 lagged snapshots and two observation-history signals:

- Shared summaries describe ball confidence, signed distance beyond the pitch boundary, nearest-player distance, local confidence averages, and past/future ball speeds over 0.2, 0.6, 1.5 and 4 s.
- Each team's summaries describe its two nearest ball distances, count within 5 m, observed-player count, nearby-player distances at temporal offsets, and mean player speeds.
- Lagged snapshots retain 14 proximity, motion and confidence signals from 0.5, 1, 2 and 4 s earlier. Observation history adds ball visibility and log time since the last observation.

Player tracks held exactly constant for at least 2 s are marked missing; short plausible out-and-back spikes are interpolated. Bounded ball gaps up to 0.5 s are interpolated when implied speed is at most 32 m/s, with confidence 0.45 versus 1 for observations. Ball-speed measurements crossing missing raw observations are excluded. Player-speed measurements crossing track discontinuities are also excluded.

Stored training statistics standardize each feature, and missingness indicators double the input width. A width-48 temporal convolutional network (TCN) with six residual convolutions, dilations 1 through 32, predicts H/A/D probabilities. Predictions are averaged under exchanged team inputs and interpolated to 25 Hz. Viterbi decoding uses turnover cost 5, live/dead cost 10 and log evidence scaled by 10/fps. Directional motion cues, weighted by 2, lower turnover cost near credible control/deflection and raise it for a fast ball continuing past a player.

Kickoff tags bound a search for the ball leaving the centre spot; the tag midpoint is used when tracking is insufficient. Graded frames before kickoff are D. Published half boundaries define X. These D/X masks remain fixed.

### Spatial team-control model

The spatial model uses 148 features at 5 Hz. Its 49 instantaneous features comprise nine ball position/motion/visibility measures, 32 team formation/proximity measures, and eight touch-history measures. Team measures summarize centroids, spread, position quantiles, visible counts, walking share and nearest-player distances. Touch history combines distance-margin-weighted control evidence with decaying past/future support. One- and three-second finite-value means add 98 features; a source indicator completes the vector. Missingness indicators accompany standardized values.

Pitch orientation is normalized automatically from team median positions. A width-96 TCN with seven residual convolutions, dilations 1 through 64 and layer normalization provides about 25.4 s of convolutional context per side. We average ordinary and rotated, team-swapped predictions, renormalize H/A probabilities, interpolate to 25 Hz and round to four decimals. One frozen network, trained with seed 413, supplies live-team evidence.

### Learned possession selection

For each live interval, two-state Viterbi decoding uses five turnover costs {0.5, 1, 2.5, 5, 10} and three directional motion strengths {0, 1, 2}. The 15 paths supply candidate possession segments. Each candidate has 22 team-symmetric descriptors of duration, proposal agreement, confidence, uncertainty, boundary changes, ball visibility, control support and adjacency to dead ball.

A histogram gradient-boosted regressor scores each candidate. Its target is the fraction of overlap thresholds {0.3, 0.4, 0.5, 0.6, 0.7} cleared by the candidate's best same-team target possession in external training data. Temporal intersection-over-union (IoU) is overlap duration divided by union duration. The regressor uses 150 iterations, learning rate 0.05, at most 15 leaves, minimum leaf size 40 and L2 regularization 10.

Dynamic programming joins candidates only at identical endpoints with alternating teams, maximizing the sum of predicted quality minus 0.4 per segment. The resulting path covers every live frame exactly once and preserves all D/X frames. Temporal evidence therefore carries possession across passes and loose balls without forcing a switch at every proximity change.

## 2. Training and validation

| Training source | Games | Supervision and use |
| --- | --- | --- |
| DFL/IDSSE Bundesliga | 7 | Released frame-level H/A/D labels; primary training and cross-validation. |
| Metrica Sports sample data | 3 | H/A/D targets derived mechanically from synchronized events; primary training and cross-validation. |
| SkillCorner A-League | 20 | Provider possession phases; auxiliary H/A training for the spatial model. |

The stoppage TCN uses the A-League collection with 12 training, four development and four test games; its checkpoint is selected after eight training epochs. The spatial TCN trains on all 30 external games for 12 epochs of 50 batches, batch size 12, using AdamW at peak learning rate 0.0005. Training uses missing-ball/player corruption, smooth position errors and moving dead-ball examples. Unknown targets are masked. Live-frame weights scale inversely with square-root possession duration, clipped to [1/3, 3]. Auxiliary sampling weight is one quarter; transition and within-state smoothing losses have weights 0.15 and 0.05.

Nested validation holds out two of the ten primary games in each of five outer folds. Inner neural fits and leave-one-training-match-out regressor validation fit the scorer and choose its penalty from {0.2, 0.4, 0.6} without the outer games. Normalization and neural fitting also exclude the outer games. The 20 A-League games remain training-only in this benchmark. Final fitting uses all external games and out-of-fold primary predictions for scorer training.

## 3. Results

SpellF1 averages one-to-one, same-team possession-matching F1 over IoU thresholds 0.3 through 0.7. External scores average halves and natural/corrupted conditions within each held-out match, then games within each provider. The nested headline weights the DFL and Metrica provider means equally. It is a segment score, not frame accuracy.

| Evaluation | Games / frames | spellF1 | macroF1 |
| --- | --- | --- | --- |
| Nested DFL/IDSSE | 7 games | 0.612391 | — |
| Nested Metrica | 3 games | 0.701545 | — |
| Nested equal-provider mean | 10 games | 0.656968 | — |
| Public competition baseline | 7,500 frames | 0.325581 | 0.620733 |
| Public Absolute Donut | 7,500 frames | 0.720000 | 0.918671 |

The public baseline is the competition's published `baseline_smooth50.csv`. Both submissions were scored with the same official evaluator; our output contains 18 public possessions versus 17 targets. Public labels are used only for automated checks after freezing the method. The aggregate TCN component scored 0.670428 spellF1 on four natural A-League test games; the complete submission has no independent A-League holdout because the spatial model uses all 20 games for training. Detailed component and per-match tables accompany the code.

## 4. Reproduction and limitations

Run `python final/reproduce.py --data-dir BUNDLE --device cuda --output OUT.csv` with the included weights and pinned dependencies. The documented NVIDIA GB10 / PyTorch 2.15.0+cu132 environment reproduces the full CSV byte for byte. The script checks model fingerprints, CSV validity and the output SHA-256. CPU inference differed at two H/A frames and fails this exactness check.

The selector cannot repair fixed stoppage errors or choose boundaries absent from its candidates. Tracking gaps, team-code errors and differences in provider label definitions remain limitations. Benchmark reuse during development limits interpretation as a fresh blind test. Future context makes the system offline; private full-match performance is unknown.

## Data attribution

DFL/IDSSE: Bassek et al., Scientific Data (2025), CC BY 4.0; credit to Deutsche Fußball Liga and the dataset authors. https://doi.org/10.1038/s41597-025-04505-y

SkillCorner Open Data, MIT; revision `4340d274572876239c154c90bc507a9b3250a656`. https://github.com/SkillCorner/opendata

Metrica Sports Sample Data; credit to Metrica Sports for tracking and events; revision `e706dd506b360d69d9d123d5b8026e7294b13996`. https://github.com/metrica-sports/sample-data
