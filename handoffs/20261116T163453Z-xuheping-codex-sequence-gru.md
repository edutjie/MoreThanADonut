# Session handoff: Bidirectional GRU for possession

- Session: 20261116T163453Z-xuheping-codex-sequence-gru
- User: xuheping
- Agent: Codex (GPT-6), implementation and evaluation
- Branch: experiments/training-approach
- Base commit: 17f728f172b891e321df1836b15e1f4acdae530f
- Started: 2026-11-16T16:34:53Z
- Updated: 2026-11-16T16:38:24Z
- Status: completed
- Delivery: local experiment commit, checkpoint and prediction
- Integration owner: xuheping
- Owned paths: sequence implementation, shared dropout helper, report and this handoff

## Outcome

- Trained one bidirectional GRU on 12 A-League matches and froze epoch 8
  at 0.7064 development spellF1. The same Viterbi costs as LightGBM were used.
- Held-out A-League spellF1 rose from 0.6616 to 0.6728. Added-long-gap
  spellF1 rose from 0.5540 to 0.5909, while the number of matched possessions
  with gaps fell from 337 to 317 out of 590.
- Frozen public spellF1 reached 0.6621 versus 0.6452. MacroF1 reached 0.8894
  versus 0.8084 and D F1 reached 0.8946 versus 0.6861.
- The full candidate CSV passed validation and repeated inference gave
  identical bytes. No contest labels were used in training or model selection.

## Changes by commit

### 20261116T163453Z-xuheping-codex-sequence-gru-morethanadonut-01

- Changes: add the sequence model, exact held-out benchmark and gap-spell
  recall. Share the original training dropout generator.
- Validation: full external and frozen public evaluations, repeated inference
  comparison, 10/25 fps frame-grid and team-symmetry checks, compilation and
  Git whitespace checks.
- Evidence: [SEQUENCE_POSSESSION.md](../SEQUENCE_POSSESSION.md).
- Local artifacts: `data/external_possession/sequence-gru/`. The candidate is
  in `candidate/possession.csv` and the frozen weights in `gru.pt`.
