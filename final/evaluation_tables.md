### Training data

The spatial network uses the following 30 external games. Frame counts refer to the 5 Hz training grid before synthetic augmentation; unknown targets are excluded from the supervised counts.

| Training source | Games | 5 Hz labeled frames | Targets | Use |
| --- | --- | --- | --- | --- |
| DFL/IDSSE Bundesliga | 7 | 200,534 | Released H/A/D | Training and primary cross-validation |
| Metrica sample data | 3 | 83,233 | Event-derived H/A/D | Training and primary cross-validation |
| SkillCorner A-League | 20 | 315,102 | Provider-derived H/A | Auxiliary training only |
| Total | 30 | 598,869 | Mixed supervision | 10 primary + 20 auxiliary |

The aggregate TCN uses the same A-League collection with a 12-game training, four-game development and four-game test split. The final spatial network uses all 20 games for auxiliary training. These rows therefore describe overlapping source games, not additional independent datasets.

### Nested external evaluation by dataset

The fixed-cost path uses the same spatial probabilities and fixed stoppages with turnover cost 2.5 and zero directional adjustment; it is also included among the final selector's proposals. All table entries average natural and corrupted tracking conditions. Delta is learned minus fixed-cost spellF1.

| Scored dataset | Matches | Fixed-cost path | Learned | Delta |
| --- | --- | --- | --- | --- |
| DFL/IDSSE Bundesliga | 7 | 0.603497 | 0.612391 | +0.008894 |
| Metrica sample data | 3 | 0.688594 | 0.701545 | +0.012951 |
| Equal-provider mean | 10 | 0.646045 | 0.656968 | +0.010923 |

### Nested external evaluation by held-out match

| Held-out game | Fold | Fixed-cost path | Learned | Delta |
| --- | --- | --- | --- | --- |
| DFL J03WMX | 1 | 0.553796 | 0.563632 | +0.009836 |
| Metrica game 1 | 1 | 0.693953 | 0.707874 | +0.013921 |
| DFL J03WN1 | 2 | 0.548318 | 0.562047 | +0.013728 |
| Metrica game 2 | 2 | 0.673600 | 0.690948 | +0.017348 |
| DFL J03WOH | 3 | 0.642699 | 0.664017 | +0.021318 |
| Metrica game 3 | 3 | 0.698228 | 0.705813 | +0.007585 |
| DFL J03WOY | 4 | 0.611768 | 0.621453 | +0.009685 |
| DFL J03WPY | 4 | 0.587799 | 0.594963 | +0.007165 |
| DFL J03WQQ | 5 | 0.631017 | 0.634720 | +0.003702 |
| DFL J03WR9 | 5 | 0.649081 | 0.645902 | -0.003179 |

Learned selection improves nine of ten held-out matches. The source-balanced gain is 0.010923 spellF1. The two provider averages receive equal weight, so the headline score is not the unweighted mean of all ten games.

### A-League evaluation of the aggregate TCN component

| Evaluation | Matches | spellF1 | macroF1 | D-F1 |
| --- | --- | --- | --- | --- |
| Development, natural | 4 | 0.706675 | 0.920773 | 0.954828 |
| Test, natural | 4 | 0.670428 | 0.907669 | 0.951230 |
| Test, corrupted | 4 | 0.594520 | 0.869116 | 0.933053 |
| Complete submitted pipeline | 0 held out | N/A | N/A | N/A |

These results evaluate the aggregate TCN checkpoint with its H/A/D decoder at turnover/stoppage costs 5/10. They are component results, not scores for the complete learned-spell submission. Corrupted tests add synthetic tracking gaps. The final spatial network trained on all 20 A-League games, so there is no independent held-out A-League score for the complete submitted pipeline.

### Official public evaluation

| Method | spellF1 | macroF1 | Accuracy | Spells | D-F1 |
| --- | --- | --- | --- | --- | --- |
| Best starter baseline (public spellF1) | 0.424242 | 0.581185 | 0.727467 | 16 | 0.146372 |
| Absolute Donut | 0.720000 | 0.918671 | 0.925600 | 18 | 0.891584 |

The comparison uses `baseline_smooth125.csv`, the highest-scoring of the four competition starter baselines by public spellF1. It assigns the nearest player's team and applies a centered 125-frame majority vote. Both files were scored with the same official evaluator on the same 7,500 public frames. The target contains 17 possessions. Our submission improves spellF1 by 0.295758 and macroF1 by 0.337486. D-F1 is dead-ball F1; full-match private performance is unknown.
