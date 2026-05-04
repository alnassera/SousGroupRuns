# HARDMath Mini 300Q Method Comparison

Side-by-side comparison of the bank-weight winner against plain and prompt-only
baselines on the first 300 examples of `HARDMath/evaluation/data/HARDMath_mini.json`.
All numbers use deterministic local boxed-answer scoring, not official HARDMath
GPT-grader scoring.

## Aggregate

| Method | Generated | Correct | Accuracy | Delta vs bank-weight | Hit max tokens | Boxed | Run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Bank-weight winner | 300 | 106 | 35.33% | +0.00 pp | 6.00% | 96.67% | `bank_weight_full300_ml9_hygiene_heavy_p1p5_m2p5_c2_comp1_h200_threadcap_mem70_seqs2_20260504_071010` |
| Plain baseline | 300 | 115 | 38.33% | +3.00 pp | 16.67% | 89.00% | `plain_prompt_full300_h200_aggr_b16_s16_rr4_hfcache_20260504_161613` |
| Prompt baseline | 300 | 153 | 51.00% | +15.67 pp | 0.67% | 99.67% | `plain_prompt_full300_h200_aggr_b16_s16_rr4_hfcache_20260504_161613` |

## Category Accuracy

| Category | Bank-weight | Plain | Prompt |
| --- | ---: | ---: | ---: |
| `laplace_integral` | 8/54 (14.81%) | 27/54 (50.00%) | 39/54 (72.22%) |
| `nondimensionalization_numeric` | 4/64 (6.25%) | 4/64 (6.25%) | 26/64 (40.62%) |
| `nondimensionalization_symbolic` | 65/65 (100.00%) | 65/65 (100.00%) | 65/65 (100.00%) |
| `ode` | 14/54 (25.93%) | 7/54 (12.96%) | 1/54 (1.85%) |
| `polynomial_roots` | 15/63 (23.81%) | 12/63 (19.05%) | 22/63 (34.92%) |

## Notes

- Prompt-only is the best of these three on aggregate accuracy: 153/300 = 51.00%.
- The bank-weight winner underperforms both baselines on aggregate accuracy in this
  deterministic boxed-answer score, though it beats both baselines on ODE.
- Prompt-only substantially reduces max-token hits versus plain: 0.67% vs 16.67%.
- Raw generations live in each run directory under
  `qwen3_moe_math_outputs_vllm_shards/**/response.jsonl`.
