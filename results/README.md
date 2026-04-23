# Results

This directory contains small, curated result artifacts that should stay in Git.

The main entry point is:

```text
summary.json
```

Benchmark run folders that are useful to share directly live under:

```text
benchmarks/
```

See [benchmarks/README.md](benchmarks/README.md) for the current GSM8K and
MMLU vLLM benchmark index.

Raw run folders can be copied under `results/raw/` for local analysis, but `results/raw/` is ignored by default because generation CSV/JSONL files grow quickly.

## Evaluation Runbook

### GSM8K Style Capability

Reference runner:

```text
reference_code/qwen3_eval_bundle/qwen3_moe/evals/run_gsm8k_style_eval.py
```

vLLM baseline runner:

```text
reference_code/qwen3_eval_bundle/qwen3_moe/evals/run_gsm8k_vllm_eval.py
```

Methods:

- `plain`
- `prompt`
- `post_attn_caa`
- `canonical_memory`

Traits:

- `warm`
- `formal`
- `direct` / `assertive`
- `careful` / `cautious`
- `curt` / `dismissive`
- `anxious`

Important raw outputs:

```text
all_summaries.csv
all_summaries.json
all_generations.csv
{trait}/high/summary.csv
{trait}/high/{method}/generations.csv
{trait}/high/selector/selector_artifact.json
```

### MMLU Style Capability

Reference runner:

```text
reference_code/qwen3_eval_bundle/qwen3_moe/evals/run_mmlu_style_eval.py
```

vLLM intervention runner:

```text
vllm_port/run_mmlu_vllm_intervention_eval.py
```

Use bounded subsets before launching a full MMLU sweep.

### Long-Context Dialogue Persistence

Runner:

```text
reference_code/qwen3_eval_bundle/long_context/run_dialogue_persistence_track_a_workflow.py
```

Judging requires an OpenAI API key. Generation-only smoke runs can use `--skip-judge`.
