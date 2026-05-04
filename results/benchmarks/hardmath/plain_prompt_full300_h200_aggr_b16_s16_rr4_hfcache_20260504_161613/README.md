# HARDMath Mini 300Q Plain/Prompt Baselines

This directory records the plain and prompt-only 300-question HARDMath mini evals
run after the bank-weight winner eval. These runs do not use canonical-memory
artifacts or architecture edits.

## Run

- Methods: `plain` and `prompt`
- Eval dataset: `HARDMath/evaluation/data/HARDMath_mini.json`
- Eval scope: first 300 examples, split round-robin into four shards of 75 examples
- Head selection: disabled with `--no-auto-select-heads`
- Canonical-memory artifacts: none
- Runtime: H200 4-GPU node, vLLM, `batch_size=16`, `max_num_seqs=16`,
  `gpu_memory_utilization=0.90`, `enforce_eager=False`
- Cache/temp hygiene: model cache and temp paths pinned to `/workspace`
- Scoring: deterministic local boxed-answer scoring, not official HARDMath
  GPT-grader scoring

## Aggregate Scores

| Method | Generated | Correct | Accuracy | Delta vs bank-weight | Hit max tokens | Boxed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain | 300 | 115 | 38.33% | +3.00 pp | 16.67% | 89.00% |
| Prompt | 300 | 153 | 51.00% | +15.67 pp | 0.67% | 99.67% |

For reference, the bank-weight winner scored 106/300 = 35.33% with 6.00%
hit-max-token rate and 96.67% boxed-answer rate.

## Category Breakdown

| Category | Plain correct | Plain acc | Prompt correct | Prompt acc |
| --- | ---: | ---: | ---: | ---: |
| `laplace_integral` | 27/54 | 50.00% | 39/54 | 72.22% |
| `nondimensionalization_numeric` | 4/64 | 6.25% | 26/64 | 40.62% |
| `nondimensionalization_symbolic` | 65/65 | 100.00% | 65/65 | 100.00% |
| `ode` | 7/54 | 12.96% | 1/54 | 1.85% |
| `polynomial_roots` | 12/63 | 19.05% | 22/63 | 34.92% |

## Contents

- `plain_prompt_300_score_summary.json`: aggregate score over both methods.
- `plain_300_score_summary.json` and `prompt_300_score_summary.json`: separate
  per-method score summaries with category breakdowns.
- `summary.json` and `summary.csv`: compact shareable metrics for this run.
- `qwen3_moe_math_outputs_vllm_shards/`: four raw shard output trees. Each shard
  contains `response.jsonl`, `run_config.json`, `summary.json`, and
  `score_summary.json` for both method outputs.
- `logs/`: shard logs from the H200 run.
- `shards/`: round-robin shard ID files.
