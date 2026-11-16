# Session handoff: Boosting comparison and SkillCorner replay

- Session: 20261116T142308Z-xuheping-codex-gbm-dashboard
- User: xuheping
- Agent: Codex (GPT-6), implementation and validation
- Branch: experiments/training-approach
- Base commit: e29ee4cb95e21bf36a871d231517238ae71f21c1
- Started: 2026-11-16T14:23:08Z
- Updated: 2026-11-16T14:28:03Z
- Status: completed
- Delivery: local branch and running dashboard
- Owned paths: external training entry point, dashboard, experiment documentation, and this handoff

## Outcome

- The original model led both development and held-out A-League splits. Held-out
  spellF1 was 0.6578 for histogram boosting, 0.6563 for LightGBM, 0.6489 for
  XGBoost and 0.6333 for CatBoost. All three new fits used the original split
  and decoder grid. Saved models reloaded with the expected H/A/D probabilities.
- The dashboard has a SkillCorner page showing a 20-second H.264 replay of
  held-out match 1925299. It renders positions and the provider-derived
  possession labels beside the frozen model labels. The source release provides
  tracking data and phases, so the replay is an animation.
- The tactical manifest's frozen-model lane matched the contest CSV on all
  199,319 frames. The lane shows the separate 0.6250 contest public-window
  check. Playwright verified playback, seeking and the contest lanes, and the
  sample route returned an HTTP 206 byte range.
- Python compilation and Git whitespace checks passed.

## Changes by commit

### 20261116T142308Z-xuheping-codex-gbm-dashboard-morethanadonut-01

- Changes: add three optional classifier backends and serve the rendered external
  replay and frozen contest labels from the dashboard.
- Validation: follow [External Possession](../EXTERNAL_POSSESSION.md) and the
  [dashboard instructions](../dashboard/README.md).

## Integration handoff

- Integration owner: xuheping
- Local server: `http://127.0.0.1:8767`
- Generated models and replay are under ignored `data/external_possession/`.
- Publication: local branch and localhost only.
