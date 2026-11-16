# Session handoff: Bidirectional Transformer for possession

- Session: 20261116T164230Z-xuheping-codex-sequence-transformer
- User: xuheping
- Agent: Codex (GPT-6), implementation and evaluation
- Branch: experiments/training-approach
- Base commit: 9d761200e31c6b135f8decc15117ea87528936c5
- Started: 2026-11-16T16:42:30Z
- Updated: 2026-11-16T16:45:05Z
- Status: completed
- Delivery: local branch and ignored experiment artifacts
- Integration owner: xuheping
- Owned paths: shared sequence implementation, comparison report and this handoff

## Outcome

- Trained the four-layer Transformer with the same splits, features, 10 Hz
  windows and fixed decoder as the GRU. Epoch 8 froze on 0.6813 development
  spellF1.
- The Transformer trailed the GRU on all three held-out spellF1 tests. It
  matched 321/590 gap spells in the A-League stress test, versus 317/590 for
  the GRU and 337/590 for LightGBM.
- Frozen public scores were 0.5000 spellF1, 0.8887 macroF1 and 0.8408 D F1.
  The GRU remains the strongest public candidate at 0.6621 spellF1.
- The Transformer CSV passed 199,319-frame validation and repeated inference
  reproduced it exactly. The existing GRU checkpoint also regenerated its
  original CSV byte for byte.

## Changes by commit

### 20261116T164230Z-xuheping-codex-sequence-transformer-morethanadonut-01

- Changes: add bidirectional Transformer under the shared sequence pipeline.
- Validation: external and frozen public evaluations, checkpoint repeats,
  attention direction, padding, team symmetry, frame grid, Python compilation
  and Git whitespace checks.
- Evidence: [comparison and reproduction commands](../SEQUENCE_POSSESSION.md).
- Local artifacts: `data/external_possession/sequence-transformer/`.
