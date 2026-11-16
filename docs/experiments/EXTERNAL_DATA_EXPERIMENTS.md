# Legacy tracking and stoppage experiments

## Legacy SkillCorner

Training adds the nine historical matches from the
[data audit](EXTERNAL_TRAINING_DATA.md) to the same 12 A-League matches.
Legacy team labels supervise home and away and unspecified labels are masked.
Tracks use identified players and the metadata ball ID. Legacy frames outside
the two regulation halves are excluded. Within a half, absent frames retain
their positions on the time grid.

The GRU and Transformer each train for eight epochs with the original features,
augmentation, inference windows and decoder. The larger dataset increases the
number of optimizer updates per epoch. The checkpoint comes from the four
original A-League development matches. The second run uses a conditional loss
on legacy frames that compares only H versus A. A-League frames retain the
three-way loss. LightGBM uses joint training and its original decoder grid.

| Training | A-League spellF1 | A-League long-gap spellF1 | Masked Bundesliga spellF1 | Natural A-League gap matches |
| --- | ---: | ---: | ---: | ---: |
| Original lagged LightGBM | 0.6616 | 0.5540 | 0.4369 | 11/30 |
| LightGBM + legacy, original decoder costs | 0.6602 | 0.5481 | 0.4183 | 12/30 |
| Original GRU | **0.6728** | **0.5909** | **0.4376** | 14/30 |
| GRU + legacy | 0.6584 | 0.5477 | 0.4028 | 15/30 |
| GRU + legacy, conditional | 0.6673 | 0.5456 | 0.4038 | 15/30 |
| Original Transformer | 0.6482 | 0.5805 | 0.4215 | 13/30 |
| Transformer + legacy | 0.6571 | 0.5623 | 0.4032 | 13/30 |
| Transformer + legacy, conditional | 0.6547 | 0.5707 | 0.4015 | **18/30** |

LightGBM chose costs (5, 5) on external development, where it scored 0.6954.
These costs gave 0.6549 on clean A-League test. The fixed-cost row uses (5, 10)
to isolate the training change. The original GRU remains the strongest overall
candidate. The conditional Transformer gained five natural-gap matches over its
own baseline while losing 18 simulated A-League gap matches (321 to 303).

### Automated public contest check

The three legacy candidates in this table were chosen within their model
families using external development spellF1. All predictions passed the
format validator for 199,319 frames. The scores cover the 7,500 public
labeled frames and 17 possession spells.

| Frozen candidate | SpellF1 | MacroF1 | Dead-ball F1 |
| --- | ---: | ---: | ---: |
| Original GRU | **0.6621** | **0.8894** | **0.8946** |
| LightGBM + legacy | 0.5375 | 0.7552 | 0.5673 |
| GRU + legacy, conditional | 0.5677 | 0.8545 | 0.8486 |
| Transformer + legacy, conditional | 0.5333 | 0.7781 | 0.6246 |

## In-pitch stoppage evidence

Across the three Metrica games, conservatively pairing a foul or offside
marker with the next compatible free-kick restart produced 53 intervals
and 30,544 weak dead-ball frames. Only 16 of these frames had observed ball
coordinates. Plain passes provided weak live examples. The specialist learned
from team-symmetric player-speed summaries only, sampled at 5 Hz during fit.

The SoccerTrack pilot added the first ten minutes of each half from two
v1.2 training matches, 117092 and 118575. Adjacent OUT and THROW IN events
give weak stoppage labels. Adjacent same-team PASS and PASS/DRIVE events
within two seconds give weak live labels. Of 12,000 sampled frames, 11,999
met the player visibility requirement. Training used 833 weak dead and 1,932
weak live samples. It used the annotated player positions and event times.

The specialist's clipped, prior-corrected dead-ball log odds were added to
the original LightGBM emissions. The decoder stayed fixed. Selection used the
mean in-pitch dead-ball F1 over the original A-League and Bundesliga
development sets, with candidate weights 0, 0.25, 0.5 and 1. In-pitch scoring
requires an observed ball more than 0.5 m inside the pitch. Bundesliga here
uses its original observed tracking.

| Player-motion training | A-League dev in-pitch F1 | Bundesliga dev in-pitch F1 | Selected weight |
| --- | ---: | ---: | ---: |
| Original LightGBM | 0.7840 | 0.9612 | 0 |
| Metrica only, weight 0.25 | 0.7755 | 0.9662 | 0 |
| Metrica + SoccerTrack, weight 0.25 | 0.7748 | 0.9661 | 0 |

Both specialists selected zero weight across the two development sources.
The frozen in-pitch F1 on held-out games therefore stayed at 0.7644 for
A-League and 0.9493 for Bundesliga. SoccerTrack adds touchline stoppages
in this pilot. Its ball-action labels do not identify the start of a foul
or offside stoppage.

The StatsBomb clock audit held out every third pass anchor when estimating a
constant offset and orientation per team and half. Seven team-halves had
offsets between -0.9 and -0.4 seconds. Real Madrid in the first match's first
half hit the +5.0-second search bound while Barcelona in the same half selected
-0.4 seconds. Held-out median position errors were 2.5–5.2 m. These event files
need a validated clock mapping before supplying stoppage targets.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
work=data/external_possession/legacy-study
aleague=external_data/skillcorner-4340d27
legacy=external_data/skillcorner-history
idsse=external_data/idsse
reference=data/external_possession/lightgbm-lagged-4340d27.joblib
python external_possession.py train --skillcorner "$aleague" --legacy-skillcorner "$legacy" \
  --backend lightgbm --lag-features --model "$work/lightgbm.joblib"
python sequence_possession.py train --skillcorner "$aleague" --legacy-skillcorner "$legacy" \
  --idsse "$idsse" --reference "$reference" --architecture gru --device cuda \
  --conditional-legacy --output "$work/gru-conditional"
python -m experiments.external_stoppage_experiment --skillcorner "$aleague" --idsse "$idsse" \
  --reference "$reference" --metrica external_data/metrica-sample \
  --soccertrack external_data/soccertrack-v1.2 --output "$work/stoppage-soccertrack-10min"
```

For the sequence comparisons, repeat with `--architecture transformer` and
with the conditional loss flag omitted. Use a distinct output path for each
run. For the stoppage ablation, omit the SoccerTrack argument or compare the
two rows emitted by the combined run. The SoccerTrack input uses the v1.2
`bas/<id>/<id>_12_class_events.json` and `gsr/<id>/<id>_{1st,2nd}.json`
paths. A GSR prefix ending after the requested interval is sufficient, with a
`.json.part` suffix. Parsing uses `ijson==3.4.0.post0` when SoccerTrack is
enabled. Downloads, model files, scores and prediction CSVs are local.
