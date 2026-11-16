# Whistle detection on external audio

Trained a small histogram gradient booster on the public
[B-Human RoboCup whistle recordings](https://www.b-human.de/datasets-en.html)
and compared it with the existing narrow-band tonal heuristic. Training uses
51 relative spectral bands, peak frequency and tonal concentration on 32 ms
windows. The development recordings set the probability threshold to 0.995.
Four or more high-scoring windows support a grouped event. The recordings are
partitioned by session. Nominally related recordings, including both halves of
LUnited–UPenn and the Dutch recordings, stay in the same split.

| External recording split | Detector | Matched | False | Missed | Event F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | Tonal | 9 | 53 | 1 | 0.2500 |
| Development | Learned | 10 | 12 | 0 | 0.6250 |
| Held-out | Tonal | 6 | 43 | 6 | 0.1967 |
| Held-out | Learned | 11 | 8 | 1 | 0.7097 |

Scoring permits one detection per annotated event, with 250 ms tolerance
outside the annotated span. Training used 20,705 windows from the three
training recordings, including 2,939 whistle-positive windows. The held-out
result covers 12 whistles and reflects RoboCup microphones and environments.

## Contest audio

The contest's tactical audio stream starts at the tactical video origin. The
tonal detector emitted 25 candidates across the halves. The learned detector
emitted one. Neither produced a usable paired stoppage on the tactical feed.
The broadcast scan emitted two candidates, one on a verified tactical frame.
It also produced zero paired stoppage intervals.
The acoustic outputs are unverified candidates. No contest audio was uploaded
or labeled, and the detector threshold was fixed on external recordings.

## Reproduce

Use the authors' [whistle ZIP](https://b-human.informatik.uni-bremen.de/public/datasets/whistle-2017/whistle.zip).
Store the audio, model and cue files under ignored `data/`. Extraction uses
FFmpeg. The experiment uses NumPy, SciPy, scikit-learn and joblib.

```bash
work=data/external_possession/stoppage
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=5 python whistle_experiment.py train \
  --archive data/robocup-whistle-2017/whistle.zip \
  --output "$work/whistle-robocup.joblib"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=5 python whistle_experiment.py detect \
  --model "$work/whistle-robocup.joblib" --output "$work/learned-whistles.csv"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=5 python whistle_experiment.py detect \
  --model "$work/whistle-robocup.joblib" \
  --broadcast-map submissions/frozen-broadcast/frame-map.npz \
  --output "$work/broadcast-whistles.csv"
```

Other sources identified include the
[2024 NaoDevils whistle dataset](https://doi.org/10.17877/tudodata-2024-m0f6bmi1)
with CC BY-SA 4.0 labeled spectra and separate published model weights, and
[Metrica's sample games](https://github.com/metrica-sports/sample-data)
with synchronized tracking and events. The 2024 audio files and Metrica match
data did not enter this run.
