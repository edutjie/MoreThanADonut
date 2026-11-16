# Additional external possession training data

Audited on 2026-11-17 in the project timezone. This extends the
[A-League and IDSSE experiments](SEQUENCE_POSSESSION.md) with independent matches.

## Downloaded and checked

| Release | Local source | Coverage | Training use |
| --- | --- | --- | --- |
| [Legacy SkillCorner](https://github.com/SkillCorner/opendata/tree/1b99d401ddc82e366c5c20d6860878668fd5858f) | `external_data/skillcorner-history` | Nine European league matches with 10 Hz broadcast tracking and per-frame team possession | Supervise H/A on labeled frames, including frames where the ball is missing. Mask unknown targets. |
| [Metrica Sports Sample Data](https://github.com/metrica-sports/sample-data/tree/e706dd506b360d69d9d123d5b8026e7294b13996) | `external_data/metrica-sample` | Three matches with 25 Hz tracking and synchronized events | Derive weak dead-ball intervals from tagged stoppages and compatible restarts. |
| [StatsBomb Open Data](https://github.com/hudl/open-data/tree/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb) | `external_data/statsbomb-clasicos` | Event logs for two of the nine SkillCorner matches | Align to tracking and check clock offsets before using events to derive dead-ball targets. |

The legacy SkillCorner release is MIT-licensed. Its original tracking JSON is
available in the local clone at the pinned revision under
`data/matches/<id>/structured_data.json`, alongside `match_data.json`.
Match IDs are 2068, 2269, 2417, 2440, 2841, 3442, 3518, 3749 and 4039.
The audit counted frames by `possession.group` and checked for an object matching
the metadata ball ID with non-null XY.

| Possession group | Ball XY present | Ball XY missing | Total |
| --- | ---: | ---: | ---: |
| Home | 136,537 | 37,687 | 174,224 |
| Away | 130,594 | 38,432 | 169,026 |
| Unspecified | 15,289 | 245,565 | 260,854 |

The 76,119 H/A frames with no ball XY are useful real examples of the
missing-ball problem. The documentation defines team possession and leaves
the null group unspecified. Keep those frames available as sequence context
and mask their labels in the training loss.

Across the three Metrica games, 7,299 of 7,300 event records have both endpoint
frame IDs in the tracking files, including 54 foul markers and six events
tagged with offside. The exception is Game 1's opening kickoff with end
frame 0. Games 1–2 use CSV tracking and events. Game 3 uses EPTS tracking and
JSON events. Its repeated set-piece subtypes can be an object, list or null.
Metrica asks for acknowledgment in public use.

The StatsBomb match list confirms the following joins by home team, away team
and match date. Both event files contain foul and offside signals as well as
free-kick restarts. Frame-level clock alignment is pending. Follow the StatsBomb
research and attribution terms before using or publishing any derived outputs.
[Official match list](https://github.com/hudl/open-data/blob/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/data/matches/11/42.json).

| Date | Home vs away | SkillCorner ID | StatsBomb ID |
| --- | --- | ---: | ---: |
| 2019-12-18 | Barcelona vs Real Madrid | 2841 | 303596 |
| 2020-03-01 | Real Madrid vs Barcelona | 3442 | 303470 |

## Additional leads

- **Gradient Sports 2022 World Cup** offers broadcast tracking and per-match
  events for all 64 games through its request form. Confirm the event schema,
  stoppage coverage and training terms after obtaining access.
  [Provider release](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset).
- **SoccerTrack v2 (release v1.2)** has 10 Japanese university matches and 21,428
  synchronized ball-action labels. Its auxiliary ball trajectories interpolate
  between annotated event anchors, so the trajectories encode target timing.
  Consider the player tracks and action labels for auxiliary training. The
  CC BY 4.0 Hugging Face release requires sign-in and acceptance.
  [Provider dataset card](https://huggingface.co/datasets/atomscott/soccertrack-v2).

## First experiment

Use the nine legacy matches as H/A supervision and keep the current A-League
development and held-out matches fixed. For the GRU and Transformer, retain
unknown frames as temporal context and give them zero loss. For LightGBM,
use the labeled frames while building lag features from the surrounding tracks.
Test the Metrica stoppage labels as a separate data addition and check spellF1
and dead-ball F1 on held-out external matches. Pair only plausible stoppages
and restarts within the same half and leave ambiguous intervals unlabeled.
This audit produced no new model scores.
