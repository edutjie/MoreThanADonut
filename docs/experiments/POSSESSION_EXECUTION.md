# Possession execution results

The continuous-track GRU remains the baseline. The pooled provider-head GRU is a
frozen challenger. It improved native-label spellF1 by **0.0108** across the six
external holdouts, with **+0.0002 on A-League** and **+0.0319 on Bundesliga**.
Natural A-League in-pitch dead-ball F1 fell from 0.8808 to 0.8773. The two
Bundesliga matches account for nearly all of the natural-input gain, and the
provider-head improvement did not replicate in the matched TCN comparison.
This decision was written before the public correctness check.

## Protocol

The study uses base `7070119746e3a6b2b10b29cfde2401385f419263`, the 12/4/4
SkillCorner A-League split at revision `4340d274572876239c154c90bc507a9b3250a656`,
and the 4/1/2 IDSSE Bundesliga split. Every half and corruption of a match
stays in its original split. Development settings and thresholds were chosen
using external data. A gain of 0.005 spellF1 was the minimum set before
comparison. A-League and Bundesliga holdout results are previously used
benchmarks. The two Bundesliga games are a transfer check. Metrica provides
a separate partial event reference, and SoccerTrack match 128057 is a fresh
video confirmation for this detector comparison.

Scores use the official mean over possession IoU thresholds 0.3–0.7. Tables
average halves within each match and then weight matches equally. Per-half
JSON includes macro F1, in-pitch D-F1 where the reference ball is in bounds,
typed transition F1 at ±0.5 s, timing error, spell counts, same-team overlaps,
missed 2–4 s spells, natural ball-gap cases, and prediction fingerprints.
The six-match bank is small, so retain the paired match results.

## 1. Restart-aware decoding

Cached the frozen GRU pre-decoder probabilities, native grid, input track
fingerprints and checkpoint hash. The fixed decoder and the neutral
four-state decoder replayed the same labels on every evaluated half. The
original player-quality runner matched all 12 held-out halves exactly. A
synthetic case verified that a true stoppage splits same-team possessions
and that each half uses its own frame mapping.

| Frozen decoder | Development spellF1 | Held-out spellF1 |
| --- | ---: | ---: |
| Fixed costs | 0.6547 | **0.6220** |
| Timing only | 0.6555 | 0.6215 |
| Restart team only | 0.6551 | 0.6212 |
| Timing and team | **0.6563** | 0.6212 |

The restart head scored 516/584 at known test restart frames. With unknown
restart times, the combined decoder got the team right on 435/454 matched
restarts versus 434/452 for the control. Mean absolute delay among those
matches fell from 0.164 s to 0.160 s. Missed short possessions stayed at 276.
The combined decoder missed the development adoption gate and stayed out of
competition inference. The noisy-tag development proxy was built from
external restart labels with omissions, jitter, false regions and wrong
team hints. It scored 0.6561 versus 0.6563 tag-free. It measures sensitivity
to artificial tag noise. No comparable automatic external tag stream was
available, so competition Smart Tags were excluded.

## 2. Video and coordinate fusion

SoccerTrack v2 v1.2 revision `325bd4c57ee8e135ab652f3921d274526ada496b`.
Development used match 117092 and confirmation used 128057. The first-half
footage, player tracks, calibration, and action annotations were fetched
locally. The 25 Hz video grid and 1-based action frame convention agree to
within one frame on both matches. The provider's calibrated plane mapping
and distortion maps yielded median reprojection errors of approximately
70.5 and 7.2 source pixels on their 65 annotated landmarks. Development
calibration and a separate pitch-space ball reference did not support
verified position replacement on these clips.
[SoccerTrack release](https://huggingface.co/datasets/atomscott/soccertrack-v2)
[Provider calibration](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/calibration.md)

Ball centers were marked from the source images before looking at detector
outputs. The sample included visible play, small/fast balls, a free kick,
and a throw-in. Nine development and twelve confirmation points were
confidently visible. Ten uncertain images were kept out of the recall
denominator. No definitive invisible-ball subset was available for a
separate false-alarm rate. All runs inferred the same 99 development and
98 confirmation frames. A hit is within 14 source pixels. The confidence
setting was chosen on development under a common extra-detection ceiling
of 0.5 per annotated visible frame. Tiled SoccerNet could not meet that
ceiling, so the lowest extra-detection setting is shown.

| Local detector | Confidence | Dev hits / 9 | Confirm hits / 12 | Confirm extra candidates | Confirm linked hits | Confirm runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SoccerNet-v3D, whole image | 0.10 | 2 | 0 | 1 | 0 | 3.1 s |
| SoccerNet-v3D, tiled | 0.50 | 1 | 1 | 0 | 0 | 41.5 s |
| Roboflow Sports example, whole image | 0.25 | 3 | 0 | 1 | 0 | 4.7 s |
| Roboflow Sports example, tiled | 0.50 | 2 | 2 | 1 | 0 | 41.9 s |

The two tiled Roboflow example hits averaged 1.07 pixels of error. Tiled
SoccerNet's one confirmation hit was 0.84 pixels away. The whole-image
SoccerNet linker produced a peak-motion proxy 0.24 s after the development
free-kick action. No variant confirmed a valid linked restart track.
The downloadable Roboflow Sports example is a separately fingerprinted
checkpoint. The exact Universe `football-ball-detection-rejhg/4` checkpoint
was unavailable through the local setup. Publisher documentation calls
for an API key for that version, and the published question about artifact
equivalence remains unanswered. All footage stayed local for detection.
[Roboflow detector method](https://roboflow.com/blog/tracking-ball-sports-computer-vision)
[Checkpoint identity question](https://github.com/roboflow/sports/issues/19)

The independent tracking stress test injected gaps, spikes, and holds into
IDSSE tracks and used a simulated noisy second sensor. Its observations were
derived separately from the uncorrupted reference and run through the same
support and coordinate gates. These numbers measure resilience to simulated
faults.

| Stress-test arm | SpellF1 | In-pitch D-F1 | Filled | Replaced |
| --- | ---: | ---: | ---: | ---: |
| Original corrupted track | 0.4974 | 0.9145 | 0 | 0 |
| Fill only | 0.5025 | 0.9165 | 5,153 | 0 |
| Replace only | 0.5081 | 0.9254 | 0 | 2,990 |
| Fill and replace | **0.5097** | **0.9268** | 5,153 | 2,990 |

The synthetic sensor filled 35.9% of eligible gaps at 0.54 m mean error.
All 2,990 replacements improved over the simulated faulty coordinates, and
zero initially good points worsened by more than 0.5 m. The combined policy
beat filling alone by 0.0072 spellF1. The raw-mask diagnostic was run on
development. The selected fused-mask policy improved its in-pitch D-F1,
so the alternative mask did not advance. The video pilot lacked verified
linked detections and synchronized possession ground truth. Full-rate
competition-video inference and adoption remain unresolved.

## 3. Missingness, provider heads and TCN replication

Both GRU retrains used eight epochs and the same natural training tracks,
player cleanup, fixed decoder, and optimizer. Random and informative masks
matched the exact number and length of simulated gaps before overlap with
natural gaps. The informative sampler conditioned live losses on speed,
crowding and position and jittered stoppage windows. The provider experiment
used the better random-burst recipe and added the four designated IDSSE
training matches. It compared a shared classifier against provider-specific
dead/live and conditional team logits with separately supervised switches.
The pooled model's decoder was evaluated both with and without using that
switch score. Both variants share the auxiliary training loss.

| Model or decode arm | Natural spellF1 | Weak dropout | Strong dropout |
| --- | ---: | ---: | ---: |
| Frozen continuous-track GRU | 0.6220 | 0.5610 | 0.5719 |
| GRU retrained with random bursts | 0.6170 | 0.5737 | 0.5772 |
| GRU retrained with informative bursts | 0.6099 | 0.5653 | 0.5772 |
| GRU with shared classifier, A-League + IDSSE | 0.6241 | 0.5675 | 0.5715 |
| GRU with pooled provider heads | 0.6329 | 0.5755 | 0.5873 |
| GRU with pooled heads and switch cue | **0.6327** | **0.5758** | **0.5877** |
| TCN with shared classifier | 0.6409 | 0.5878 | 0.5959 |
| TCN replication with pooled heads and switch cue | 0.6408 | 0.5884 | 0.5964 |

The dev average selected the pooled GRU with switch strength 1 at epoch 7
(0.6228 versus 0.6159 frozen). The cue increased natural typed transition
F1 from 0.4737 to 0.4771. Its spellF1 change versus the same pooled model
without the cue was -0.0002. The TCN experiment tested replication of the
provider effect. Its paired spellF1 gain averaged only +0.0003 across
conditions, with a natural A-League regression for the provider-head arm
versus the shared TCN. It produced no new competition recipe.

| Natural-input match | Frozen GRU | Selected GRU | Delta |
| --- | ---: | ---: | ---: |
| A-League 1925299 | 0.6554 | 0.6561 | +0.0007 |
| A-League 1996435 | 0.7119 | 0.7169 | +0.0050 |
| A-League 2007448 | 0.6949 | 0.6856 | -0.0093 |
| A-League 2017461 | 0.6371 | 0.6412 | +0.0041 |
| Bundesliga J03WOY | 0.5120 | 0.5377 | +0.0257 |
| Bundesliga J03WR9 | 0.5205 | 0.5588 | +0.0383 |

The separate Metrica pass and paired-stoppage proxy scored 0.9271 frozen,
0.9283 random retrain and 0.9293 pooled GRU. Unlabeled intervals were masked
and the proxy was excluded from native-label comparisons. Its high scores
reflect truncated event-labeled intervals.

## Frozen inference and reproduction

Only the pooled GRU challenger was carried through to the competition
correctness check. It used whole-half local tactical tracking, existing
player repairs, the original 113 features, equal provider pooling, and
switch strength 1. The format validator passed all 199,319 zero-based rows.
The public window covers 7,500 graded frames and 17 reference possessions.
The automated check reported 0.6667 spellF1, 0.8663 macro F1, and 0.8267 D-F1.
The reference's previously verified scores are 0.6897, 0.9002, and 0.8882
respectively. The private score is unknown.

| Artifact | SHA-256 |
| --- | --- |
| Frozen continuous-track GRU | `2fab357a393edf6747bb7d9847bcd633da3ec7706a2be2d480f614d36923604c` |
| Original aggregate TCN | `607d421ae5619a5684fda3bfe6ff2d34c4da6dcb9a4cefea12891d450ccb2d2b` |
| Pooled GRU checkpoint | `bef048ffc7b618f4c3eca460abcc27c1981c0c2a6bf640bc252b4dd7a2480125` |
| Pooled GRU competition CSV | `c20124f73979a2034614ec50a5c8bdede47b9c33842f13e6608a0ce47afe2600` |
| Local SoccerNet-v3D | `a3082fb435a8501ae17cfe4ac78e66ca7041205e115feda344d34d1693064f36` |
| Local Roboflow Sports example | `678fbad05134f19c5094cb8d273812ec9c6691228180d46832551ecf99ed2912` |

Ignored evidence is under `data/external_possession/execution-20261117` in
the source checkout. It includes the full manifests, model selection
records, per-match metrics, hashes, annotation CSV, detector caches, format
check, and public scoreboard. The execution runners and report are tracked.
Reproduction commands from the study branch are in the
[session handoff](../../handoffs/20261117T072920Z-edutjie-codex-possession-execution.md).
