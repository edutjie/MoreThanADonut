# Session handoff for the possession execution plan

- Session: 20261117T072920Z-edutjie-codex-possession-execution
- User: edutjie
- Agent: Codex, GPT-6, experiment implementation and evaluation
- Repository: MoreThanADonut
- Branch / worktree: `codex/edutjie/possession-execution-072920`, `/tmp/morethanadonut-possession-072920`
- Base commit: `7070119746e3a6b2b10b29cfde2401385f419263`
- Started: 2026-11-17T07:29:20Z
- Updated: 2026-11-17
- Status: complete with video adoption gate unmet
- Delivery: local branch and ignored evidence in the source checkout
- Objective: execute the possession plan and provide externally selected comparisons and official correctness evidence
- Owned paths: experiment runners named below, optional turnover cue in `tactical_baseline.py`, this handoff and the results report
- Related sessions: baseline and data provenance from PR #3
- Recipient: edutjie

## Current outcome

[Results](../docs/experiments/POSSESSION_EXECUTION.md) include restart
decoding, real video detection, controlled coordinate fusion, matched
missingness training, provider-head and boundary-cue training, a matched
TCN replication, and a separate Metrica event check. The pooled GRU
challenger passed the full-grid validator and completed the automated
public-window check. The frozen continuous-track GRU remains the
reference. The accountable selection and checkpoint/prediction hashes were
recorded in `decision-before-public.json` before the public scorer ran.

## Changes by commit

### 20261117T072920Z-edutjie-codex-possession-execution-morethanadonut-01

- State: delivered on local topic branch
- Changes: external experiment, verification and inference runners, results report, and an optional bounded turnover bonus
- Why: measure the practical spellF1 gain from the proposed changes on the fixed external splits
- Validation: twelve external halves reproduced with the original runner; same-team restart and halftime fixtures passed; official format and public checks passed; final CSV replay matched byte for byte
- Risks / follow-up: similar natural A-League performance, no independent pitch-space ball reference, small Bundesliga transfer set, exact Roboflow Universe weights unverified

## Reproduction

Run from the execution worktree with the original checkout's environment
and local datasets. Reusable metadata and result JSON remain with the
ignored artifacts. When reproducing from scratch, use a new output
directory and fetch the pinned external data plus the independently marked
SoccerTrack ball-center CSV. The published metrics were produced locally
with torch 2.15.0+cu132, NumPy 2.4.6, scikit-learn 1.9.1 and OpenCV 5.0.0.

```bash
src=/home/xuheping/codes/MoreThanADonut
work="$src/data/external_possession/execution-20261117"
py="$src/.venv/bin/python"
gru="$src/data/external_possession/player-quality/continuous/gru.pt"
tcn="$src/data/external_possession/tcn-study/autotcn/model.pt"
detector="$src/models/yolo-sn-ball-opt.pt"
external="$src/external_data"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1

external_args=(--skillcorner "$external/skillcorner-4340d27" --idsse "$external/idsse" --gru "$gru")
for stage in cache restart; do
    "$py" -m experiments.possession_execution "$stage" "${external_args[@]}" \
        --tcn "$tcn" --detector "$detector" --output "$work"
done
"$py" -m experiments.possession_observation "${external_args[@]}" --output "$work/observation"
"$py" -m experiments.possession_provider_heads "${external_args[@]}" --output "$work/observation"
"$py" -m experiments.possession_provider_heads "${external_args[@]}" --output "$work/observation" --replicate-tcn
"$py" -m experiments.possession_video "${external_args[@]}" --output "$work/fusion"
"$py" -m experiments.possession_execution_verify "${external_args[@]}" \
    --metrica "$external/metrica-sample" --random "$work/observation/observation-random.pt" \
    --provider "$work/observation/provider-provider.pt" --output "$work"
"$py" -m experiments.soccertrack_video_pilot --soccertrack "$work/soccertrack" \
    --annotations "$work/ball-centers.csv" --detector "$detector" --output "$work/pilot"
"$py" -m experiments.possession_execution_predict --checkpoint "$work/observation/provider-provider.pt" \
    --selection "$work/observation/provider-selection.json" --data-dir "$src/data" \
    --output "$work/frozen/possession.csv"
"$py" evaluation/validate_submission.py "$work/frozen/possession.csv"
"$py" evaluation/run_evaluation.py --gt public --results "$work/frozen" \
    --out "$work/public-check" --no-detail --bootstrap 0
```

For each tiled detector replay use `--tiled` and a fresh output directory.
The local Roboflow example is `$work/roboflow-example.pt`. It was published
by the Roboflow Sports demo. For the windowed development download, the
first 960 MiB were verified byte for byte against the completed official
video; all inference windows lie in that prefix. The complete video is
available for future replays.

## Findings and decisions

- External model choice used the development folds and the six pre-existing external holdouts.
- Restart-aware decoding missed the predeclared practical gain.
- Synthetic coordinate correction helped when the second sensor was reliable. Real linked detections failed confirmation and the experiment stopped before projecting video detections into a competition recipe.
- Equal provider pooling improved pooled and corrupted held-out GRU results. The natural A-League comparison was nearly flat and in-pitch dead-ball F1 there decreased by 0.0035. The provider-head advantage did not replicate on TCN.
- The existing baseline stays active. The challenger output remains available for reproducibility and comparison.

## Communication log

- 2026-11-17 | edutjie to Codex | requested execution of `docs/plans/POSSESSION_EXECUTION_PLAN.md`
- 2026-11-17 | Codex to edutjie | acknowledged in this session and delivered the report and branch

## Integration handoff

- Source branch / head SHA: `codex/edutjie/possession-execution-072920`; use local branch tip
- Target branch / tested target SHA: `main` at `7070119746e3a6b2b10b29cfde2401385f419263`
- Integration owner: edutjie
- Shared branch / PR: local only
- Dependencies / merge order: none
- Conflicts / resolutions: none
- Combined validation: based on the pinned main tree, with external and competition-format verification detailed above
- Integration result: pending any user request to integrate
- Cleanup: keep the execution worktree and ignored artifacts

## Outstanding state

- Uncommitted owned files: none; ignored evidence remains local
- Other work observed: the original checkout's untracked execution plan and previous handoff, both untouched
- Blockers / access needed: exact Roboflow Universe artifact or an authorized local API credential for that comparison, plus an independent calibrated grounded-ball reference and synchronized external possession targets for real fusion adoption
- Failed / unrun checks: no real-video coordinate fusion, competition video reconstruction, or private scoring was run
- Publication / recipient acknowledgement: local topic branch, handoff delivered in this thread
