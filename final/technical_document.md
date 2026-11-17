# Absolute Donut: Learned spell scoring

**Team:** Absolute Donut

**Members:** Eduardus Tjitrahardja, Louis Jefferson Zhang, Louis Jefferson Zhang, Louis Jefferson Zhang

## 1. Task and inputs

We assign home possession (H), away possession (A), dead ball (D), or ungraded
(X) to every tactical-video frame. The submitted file contains 199,319 rows
at 25 Hz. All indices are zero-based; one-based ATD rows are converted at load time.
The provided pitch dimensions are 105.3 × 68.0 m.

Inference uses the official tactical automatic tracking, XML slot-to-team
metadata, and automatic kickoff Smart Tags. Tracking is converted to metres.
The method uses no video, audio, homography, or manual correction. Team codes
come from the tracking metadata. No kit-color or referee model is used.

The pipeline separates stoppage detection from live-team assignment. An
aggregate TCN establishes fixed dead-ball intervals. A spatial TCN supplies
live H/A probabilities. Several decoders propose complete possessions, then
a learned scorer selects a continuous path through those proposals.

## 2. Fixed stoppages and live-team evidence

**Dead-ball branch.** The original TCN receives 113 aggregate features and
their missing-value indicators at 10 Hz: nearest-player distances, ball and
player movement, pitch-boundary geometry, temporal lags, and observation
history. Standardization statistics are stored in the model. Six residual
convolutions use width 48 and dilations 1, 2, 4, 8, 16, 32. Normal and
team-swapped outputs are averaged. A fitted numeric feature transform preserves
column order and converts one visibility column to Boolean; the package
exports that transform exactly.

Player trajectories receive the existing automatic gap/spike repair. Bounded
ball gaps up to 0.5 s can be interpolated with reduced confidence. Ball speeds
used for the gap-safe features and physics costs require observed windows,
preventing inferred motion across missing observations. Probabilities are
interpolated to 25 Hz. A half-wise Viterbi decoder uses turnover cost 5,
stoppage cost 10 and directional physics adjustment strength 2; log evidence
is scaled by 10/fps. Nearby co-moving control or credible ball deflection can
reduce turnover cost, while a continuing fast pass raises it.

Automatic kickoff tags bound a search for the tracked ball leaving the centre
spot. Graded frames before that inferred kickoff are D, and decoding starts
fresh at kickoff. A tag midpoint is the fallback when tracking is insufficient.
Published half metadata determines X outside the graded blocks. The final
spell selector preserves every baseline D/X frame.

**Spatial branch.** A second TCN receives 148 features at 5 Hz, with missingness
channels: ball position and velocity, time before/after observations, team
centroids and spread, formation quantiles, visible counts, walking share,
nearest-player distances and decaying touch evidence. One- and three-second
averages supply local context. Pitch orientation is normalized automatically.
Seven residual width-96 convolutions have dilations 1 through 64, layer
normalization and about 25.4 s of context per side. Inference averages the
ordinary and rotated, team-swapped outputs. Only the frozen seed-413 network
is used. Its H/A outputs are renormalized, interpolated to 25 Hz and rounded
to four decimals. Its dead-ball output and auxiliary boundary output do not
enter final live decoding.

## 3. Learned possession selection

Within each live interval, a two-state Viterbi decoder runs over switch costs
{0.5, 1, 2.5, 5, 10} and directional physics strengths {0, 1, 2}. The 15
paths supply unique candidate H/A segments. The cost-2.5, zero-physics
reference path guarantees complete coverage.

Each candidate has 22 team-symmetric features: log duration; proposal vote
fraction; reference membership and overlap; mean, standard deviation, minimum
and lower-decile own-team confidence; mean/minimum log-odds and majority share;
confidence before and after the segment; log-odds changes at both boundaries;
ball-observation share; directional contact support for either side; adjacency
to dead ball at each end; mean entropy; and uncertain-margin share.

A histogram gradient-boosted regressor predicts segment quality, clipped to
[0,1]. It has 150 iterations, learning rate 0.05, up to 15 leaves, minimum
leaf size 40 and L2 regularization 10. Its external target is the fraction
of overlap thresholds {0.3, 0.4, 0.5, 0.6, 0.7} cleared by the candidate's
best same-team target possession. This is a regression score, not a calibrated
probability that the segment is correct.

A directed acyclic graph joins candidate segments only when their endpoints
meet exactly and their teams alternate. Dynamic programming maximizes
sum(segment quality − 0.4). This penalty was selected externally. Every live
frame is covered once; dead-ball intervals separate paths. There is no
per-frame abstention within live play and no post-hoc hand editing.

Operationally, passes and loose balls retain team control unless temporal
evidence supports an actual switch. A challenge alone does not force a new
possession. Restarts use the fixed stoppage boundaries; the next live team
is chosen by the spatial probabilities and segment scores.

## 4. External training and selection

The original stoppage TCN trained on 12 SkillCorner A-League matches with
four other matches for checkpoint selection; four more were held out for
testing. Labels are derived from the provider's possession phases: team
phases map to H/A, bracketed within-half gaps map to D, and unknown endpoints
are excluded. It trained for eight epochs with synthetic tracking corruption.

The spatial network trained from scratch on 30 external matches: seven
DFL/IDSSE games with released frame-level possession and ball-status labels,
three Metrica sample games with mechanically event-derived targets, and 20
SkillCorner games supplying auxiliary H/A supervision only. Unknown targets
are masked. No team member annotated the provided game or these datasets.
Metrica restarts/control events establish H/A; compatible stoppage events
establish D; ambiguous offside intervals and unexplained gaps remain unknown.

Spatial training uses 12 epochs × 50 batches, batch size 12, 768-frame crops
with 480 central supervised frames. AdamW uses peak learning rate 0.0005,
5% warmup, cosine decay, weight decay 0.01 and gradient clipping at 1.
Inverse-square-root spell-duration weights emphasize short live possessions,
clipped to [1/3,3]. Auxiliary source sampling weight is 1/4. State loss includes
an auxiliary transition loss (0.15) and within-state smoothing loss (0.05).
Corruption introduces missing ball/player observations, smooth ball error and
occasional moving dead-ball examples.

Spell-scorer validation uses five outer folds, each holding out two primary
matches. Four inner neural fits per outer fold exclude both the outer pair
and an inner pair. They generate scorer-training features for the eight
remaining primary games. Leave-one-training-match-out scorer fits choose
the path penalty from {0.2,0.4,0.6}; provider and match weights balance fitting.
Outer matches are excluded from neural normalization, fitting, scorer fitting
and penalty selection in this experiment. The final scorer fits all ten
original out-of-fold predictions and pairs with the frozen final spatial TCN.
These external games have prior experimental exposure, so this is not a fresh
blind estimate. Inner, outer and final training sizes can change confidence.

Sources and attribution:

- SkillCorner Open Data, 20 A-League matches, MIT; revision
  `4340d274572876239c154c90bc507a9b3250a656`:
  https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656
- DFL/IDSSE, Bassek et al., Scientific Data (2025), CC BY 4.0;
  credit to Deutsche Fußball Liga and the dataset authors:
  https://doi.org/10.1038/s41597-025-04505-y
  Mirror revision: `a715a38dfbaf5f58e431727c2b78d174101a703c`,
  https://huggingface.co/datasets/pysport/idsse-data
- Metrica Sports Sample Data, three games; credit to Metrica Sports for tracking
  and events; revision `e706dd506b360d69d9d123d5b8026e7294b13996`:
  https://github.com/metrica-sports/sample-data/tree/e706dd506b360d69d9d123d5b8026e7294b13996

## 5. Validation, reproduction and limits

The frozen learned method scored **0.656968 nested external spellF1**. On the
official public window of 7,500 frames, it scored **0.720000 spellF1** and
**0.918671 macroF1**, predicting 18 possessions against 17 targets. The public
labels were used for automated checks after freezing the experiment, not
training or hyperparameter selection. The public window is small and full-match
private performance is unknown.

Run `python final/reproduce.py --data-dir BUNDLE --device cuda --output OUT.csv`
with the frozen dependencies. The package includes both TCN checkpoints,
normalization, feature transform, scorer and decoder code. No external data
download or training is required for inference. The documented NVIDIA GB10
CUDA environment reproduces the entire submission byte for byte and validates
all 199,319 rows. The CPU check differed at two H/A frames, so exact reproduction
is verified for the documented CUDA environment; every run enforces SHA-256.

Main limits are inherited dead-ball mistakes that the scorer cannot repair,
tracking slot/team errors, ambiguous loose-ball control, long ball occlusions,
event-derived label noise and cross-provider domain shift. The model is offline:
future context is available. Segment quality is a surrogate for the one-to-one
spell metric, and the proposal graph cannot select boundaries absent from its
candidate paths. No claim is made about unseen private labels.
