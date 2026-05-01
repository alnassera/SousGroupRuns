# Qwen3 MoE HARDMath30 H100 Eval - 2026-05-01

Benchmark: HARDMath mini 30, seed 11  
Model: `Qwen/Qwen3-30B-A3B`  
Engine: vLLM `0.20.0` on H100 80GB  
Max new tokens: `16384`  
Scoring: deterministic local boxed-answer scoring, not official HARDMath GPT-grader scoring.

## Results

Wilson 95% intervals are over the 30 generated rows in each run.

| Run | Correct | Accuracy | Wilson 95% CI |
| --- | ---: | ---: | ---: |
| plain | 7/30 | 23.3% | [11.8%, 40.9%] |
| canonical_memory rho=3 pg=2 | 8/30 | 26.7% | [14.2%, 44.4%] |
| canonical_memory rho=4 pg=3 | 9/30 | 30.0% | [16.7%, 47.9%] |
| canonical_memory rho=5 pg=3 | 8/30 | 26.7% | [14.2%, 44.4%] |
| canonical_memory rho=6 pg=4 | 8/30 | 26.7% | [14.2%, 44.4%] |

## Contents

- `summary.csv` and `summary.json`: aggregate results.
- `score_summaries/`: copied `score_summary.json` files for each run.
- `raw/qwen3_moe_math_outputs_vllm/`: full recovered vLLM output tree, including `response.jsonl`, `run_config.json`, `summary.json`, and `score_summary.json`.
- `raw/logs/`: recovered RunPod logs, including the rho=6 resume log.
- `raw/data/random_hardmath_mini_30_seed11.json`: evaluated sample.
- `raw/scripts/run_hardmath30_pod_eval.sh`, `raw/run_math_benchmark_vllm.py`, `raw/score_hardmath_outputs.py`: runner/scorer snapshots from the pod.
- `hardmath30_qwen3_raw_2026-05-01.tgz`: compressed raw archive copied from the pod.
- `manifest.csv`: SHA-256 manifest for files in this directory.

## Notes

The original `rho=6 pg=4` launch failed on a vLLM startup memory-reservation check at `--gpu-memory-utilization 0.90`. It was resumed as a standalone run with `--gpu-memory-utilization 0.80` and completed cleanly.
