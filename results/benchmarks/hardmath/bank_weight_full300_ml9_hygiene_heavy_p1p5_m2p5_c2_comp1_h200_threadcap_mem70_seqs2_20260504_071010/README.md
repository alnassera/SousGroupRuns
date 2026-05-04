# HARDMath Mini 300Q Bank-Weight Final Eval

This directory records the final 300-question HARDMath mini eval for the
winner of the clean 30Q bank-weight selector sweep.

## Run

- Winner: `hygiene_heavy_p1p5_m2p5_c2_comp1`
- Bank gains: planner `1.5`, math hygiene `2.5`, checker `2.0`, completion `1.0`
- Eval dataset: `HARDMath/evaluation/data/HARDMath_mini.json`
- Eval scope: first 300 examples, split into four ordered shards of 75 examples
- Selector source: `HARDMath/data/HARDMath.json`
- Selector hygiene: eval IDs filtered from selector examples before head selection
- Max layers setting: `MAX_LAYERS_LIST=9`
- Candidate layer IDs: `12 16 20 24 28 32 36`
- Runtime: H200 4-GPU node, vLLM, `batch_size=2`, `max_num_seqs=2`,
  `gpu_memory_utilization=0.70`, `async_scheduling=False`, `enforce_eager=False`
- Scoring: deterministic local boxed-answer scoring, not official HARDMath
  GPT-grader scoring

## Aggregate Score

| Generated | Correct | Accuracy | Hit max tokens | Boxed |
| ---: | ---: | ---: | ---: | ---: |
| 300 | 106 | 35.33% | 6.00% | 96.67% |

Wilson 95% CI for accuracy: 30.14% to 40.90%.

## Category Breakdown

| Category | Generated | Correct | Accuracy | Hit max tokens | Boxed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nondimensionalization_symbolic` | 65 | 65 | 100.00% | 0.00% | 100.00% |
| `nondimensionalization_numeric` | 64 | 4 | 6.25% | 0.00% | 100.00% |
| `polynomial_roots` | 63 | 15 | 23.81% | 7.94% | 96.83% |
| `ode` | 54 | 14 | 25.93% | 9.26% | 98.15% |
| `laplace_integral` | 54 | 8 | 14.81% | 14.81% | 87.04% |

## Contents

- `final_300_score_summary.json`: aggregate score over the parent shard root.
- `summary.json` and `summary.csv`: compact shareable metrics for this run.
- `qwen3_moe_math_outputs_vllm_shards/`: four raw shard output trees. Each shard
  contains `response.jsonl`, `run_config.json`, `summary.json`, and
  `score_summary.json`.
- `logs/`: prebuild and shard logs from the H200 run.
- `shards/`: manifest and ordered shard ID files.
- `vllm_port_artifacts/`: clean canonical-memory artifacts built from the clean
  selector source.
- `prebuild_outputs/`: artifact-prebuild canary output and config.

The corresponding clean selector sweep is stored at:

```text
results/benchmarks/hardmath/bank_weight_tune30_ml9_fullselector_h200_threadcap_mem70_seqs2_20260504_021842/
```
