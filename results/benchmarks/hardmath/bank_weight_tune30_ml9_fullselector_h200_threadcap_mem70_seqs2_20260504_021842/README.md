# HARDMath Bank-Weight Tune 30Q Sweep

This directory records the completed 8-preset bank-weight selector sweep run on
May 4, 2026. The sweep used the 30-question clean tune set
`data/random_hardmath_mini_30_seed11.json`, with selector heads selected from
`HARDMath/data/HARDMath.json` and eval-overlap filtering enabled.

## Winner

`hygiene_heavy_p1p5_m2p5_c2_comp1` won the 30Q selector pass.

| Preset | Planner | Math hygiene | Checker | Completion | Correct | Accuracy | Hit max tokens | Boxed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hygiene_heavy_p1p5_m2p5_c2_comp1` | 1.5 | 2.5 | 2.0 | 1.0 | 8/30 | 26.67% | 3.33% | 96.67% |

`global_light_p1_m1_c1_comp0p5` also reached 8/30, but lost on the tie-breaks:
it had a higher hit-max-token rate and lower boxed-answer rate.

## Sweep Scoreboard

Selection rule: highest accuracy, then lowest hit-max-token rate, then highest
boxed-answer rate, then original preset order.

| Rank | Preset | Planner | Math hygiene | Checker | Completion | Correct | Accuracy | Hit max tokens | Boxed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `hygiene_heavy_p1p5_m2p5_c2_comp1` | 1.5 | 2.5 | 2.0 | 1.0 | 8/30 | 26.67% | 3.33% | 96.67% |
| 2 | `global_light_p1_m1_c1_comp0p5` | 1.0 | 1.0 | 1.0 | 0.5 | 8/30 | 26.67% | 10.00% | 93.33% |
| 3 | `checker_heavy_p1p5_m1p5_c3_comp1` | 1.5 | 1.5 | 3.0 | 1.0 | 7/30 | 23.33% | 10.00% | 93.33% |
| 4 | `baseline_p2_m1p5_c2_comp2` | 2.0 | 1.5 | 2.0 | 2.0 | 6/30 | 20.00% | 6.67% | 93.33% |
| 5 | `planner_heavy_p3_m1p5_c2_comp1` | 3.0 | 1.5 | 2.0 | 1.0 | 6/30 | 20.00% | 6.67% | 93.33% |
| 6 | `subject_process_p1p5_m1p5_c2_comp1` | 1.5 | 1.5 | 2.0 | 1.0 | 6/30 | 20.00% | 10.00% | 90.00% |
| 7 | `completion_light_p2_m1p5_c2_comp0p5` | 2.0 | 1.5 | 2.0 | 0.5 | 6/30 | 20.00% | 10.00% | 93.33% |
| 8 | `checker_onlyish_p0p5_m0p5_c3_comp0p5` | 0.5 | 0.5 | 3.0 | 0.5 | 6/30 | 20.00% | 10.00% | 90.00% |

## Category Notes

The 30Q tune set contained 1 polynomial-roots item, 5 ODE items, 4 symbolic
nondimensionalization items, 7 Laplace-integral items, 8 integral items, and 5
numeric nondimensionalization items.

The winning preset improved by picking up one additional Laplace-integral item
and the single polynomial-roots item while avoiding the extra max-token hits
seen in several tied or near-tied presets. Across all presets, symbolic
nondimensionalization was solved consistently, while integral, ODE, and numeric
nondimensionalization remained the hard categories in this small selector set.

## Artifacts

- `tune_scoreboard.tsv`: compact table of all 8 presets.
- `tune_scoreboard.jsonl`: raw per-preset scoreboard rows.
- `winner.json`: selected preset and tie-break metrics.
- `category_breakdown.tsv`: per-preset, per-category generated/correct rates.
- `run_metadata.json`: source paths and runtime settings.

After this sweep finished, the final 300Q eval was launched under:

```text
/workspace/qwen3_moe_math_copy/results/raw/bank_weight_full300_ml9_hygiene_heavy_p1p5_m2p5_c2_comp1_h200_threadcap_mem70_seqs2_20260504_071010
```
