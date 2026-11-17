# Absolute Donut — final submission

**Submitted submission:** [Absolute Donut.csv](Absolute%20Donut.csv), with our
[technical document](Absolute%20Donut.pdf). Our method is **Learned spell scoring**.

**Our team:** Absolute Donut. **Our members:** Eduardus Tjitrahardja, Louis Jefferson Zhang,
Louis Jefferson Zhang, Louis Jefferson Zhang.

The CSV has 199,319 data rows, with zero-based tactical-video frame indices
`0..199318` and labels `H`, `A`, `D`, or `X`.

## Recreate the exact CSV

Use Linux, Python 3.12, a C compiler (`cc`), and a CUDA-capable NVIDIA GPU.
The verified environment is recorded in [verified_environment.json](verified_environment.json):
NVIDIA GB10, PyTorch 2.15.0+cu132, CUDA 13.2, cuDNN 9.26, Ubuntu aarch64.
Install a matching PyTorch CUDA build; the remaining inference dependencies
are pinned in [requirements.txt](requirements.txt).

From the repository root:

```bash
python3.12 -m venv final/.venv
final/.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu132 -r final/requirements.txt
final/.venv/bin/python final/reproduce.py \
  --data-dir /absolute/path/to/official-bundle \
  --device cuda --output final/recreated.csv
```

The bundle root must contain these organizer-provided files:

```text
tactical/BAR-ATH_tactical_atd.xml
tactical/BAR-ATH_tactical_atd.txt
tactical/BAR-ATH_tactical_smart_tagging.xml
```

Match inputs are licensed competition assets and are **not included**.
No video, audio, homography, external event log for this match, public labels,
or prior prediction CSV is read by inference. All models are included and
inference makes no network requests. External training data is not needed
to recreate the submission from the frozen models.

The command regenerates the dead-ball baseline, computes spatial H/A
probabilities, scores candidate spells, and writes the final CSV. It checks
model fingerprints, validates the output, and **fails unless its SHA-256
matches the frozen submission**:

```text
620787b852d441d27c81c1c3e1175c13c64d6f0486be11fafd3666c906040d55
```

To check the packaged files without running inference, use `cd final && sha256sum -c SHA256SUMS`.

The frozen CSV is never used as an inference input or overwritten.
`manifest.json` supplies its expected checksum. Temporary decoder binaries
are compiled outside the package and removed after inference.

**Precision limitation:** exact reproduction has been verified on the CUDA
environment above. CPU execution is supported with `--device cpu`, but the
tested CPU run differed at two H/A frames and therefore correctly failed the
checksum. Different accelerator kernels or dependency versions can also
affect numerical boundaries; do not treat a failed checksum as exact reproduction.

The existing official format check is also available from the repository root:

```bash
python evaluation/validate_submission.py "final/Absolute Donut.csv"
```

## Package contents

| Path | Purpose |
| --- | --- |
| `reproduce.py` | Standalone inference entry point; uses only `final/src` and the included models |
| `src/` | Snapshot of the existing tracking, feature, network and decoder definitions needed for inference |
| `models/deadball/model.pt` | Original aggregate TCN for baseline H/A/D decoding |
| `models/deadball/transform.json` | Exact exported numeric feature transform |
| `models/deadball/features.joblib` | Original fitted AutoGluon transform, retained for provenance; not loaded during inference |
| `models/spatial/final-seed413.pt` | Frozen spatial TCN and its normalization statistics |
| `models/spells/judge.joblib` | Externally fitted histogram gradient-boosted spell-quality regressor |
| `models/spells/selection.json` | Frozen spell penalty and selection metadata |
| `manifest.json`, `source_provenance.json` | Artifact fingerprints and source snapshot provenance |
| `technical_document.md`, `build_document.py` | Readable methodology source and PDF/figure generator |
| `figures/` | Architecture and illustrative spell-selection diagrams, as PNG and SVG |

The original numeric feature generator retains 113 columns in order, converts
column 111 to a Boolean comparison against 1, and leaves all other values
unchanged, including NaNs. Its exported transform was checked for exact equality
with the original fitted generator on both orientations of both full match halves.
This removes the AutoGluon runtime dependency without changing the submission.

The included weights are small enough for ordinary Git; Git LFS is unnecessary.
They are stored directly so a normal clone contains the inference artifacts.
The original repository license is included as `LICENSE`.

## Evaluation

| Frozen method | spellF1 | macroF1 |
| --- | ---: | ---: |
| Nested external benchmark | 0.656968 | — |
| Official public window: 7,500 frames | **0.720000** | **0.918671** |

**What is nested external spellF1?** It measures possession-segment accuracy
on other matches held out from fitting, with a separate inner validation
loop for choosing the spell penalty. "External" means these are other games,
not the provided competition match. "Nested" means parameter selection happens
inside the training portion of each outer test split.

The scored external set consists of **seven DFL/IDSSE Bundesliga matches**
and **three Metrica Sports sample matches**. The **20 SkillCorner A-League
matches contribute training data only**; they do not contribute scores to
the reported nested external spellF1.

We split those ten primary external matches into five outer folds of two games.
For each fold, the two test games are excluded from the neural normalization,
neural training, spell-scorer fitting and penalty selection in this experiment.
Inner neural fits and leave-one-training-match-out scorer validation use the
other eight primary games to fit the scorer and choose the penalty. The
twenty auxiliary games remain training-only. We then evaluate the frozen
fold pipeline on its two outer test games.

At each temporal intersection-over-union (IoU) threshold of 0.3, 0.4, 0.5,
0.6 and 0.7, spellF1 matches predicted and target possessions one-to-one
within the same team and computes their precision/recall F1. It averages
those five F1 values. Our external benchmark then averages the two halves
and natural/corrupted tracking conditions within each held-out match,
averages matches within each provider, and gives the DFL and Metrica provider
means equal weight. That produces **0.656968**. It is a segment-level F1,
not 65.7% frame accuracy, and is separate from the **0.720000 public spellF1**.

The public slice contains 17 target possessions; the submission predicts 18.
Full-match private performance is unknown. Training and parameter selection use
external matches; the public labels are only an automated evaluation check.
The external benchmark has prior experimental exposure and is not a fresh blind test.

## Rebuild the short document

```bash
final/.venv/bin/python -m pip install -r final/requirements-docs.txt
final/.venv/bin/python final/build_document.py
```

This renders the methodology PDF and its standalone figures. Diagrams are
schematic and contain no match images, recovered tracking or manual annotations.
