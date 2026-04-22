# Evaluation Runbook

This repo tracks results for three evaluation families.

## GSM8K Style Capability

Runner:

```text
reference_code/qwen3_eval_bundle/qwen3_moe/evals/run_gsm8k_style_eval.py
```

Main methods:

- `plain`
- `prompt`
- `post_attn_caa`
- `canonical_memory`

Main traits:

- `warm`
- `formal`
- `direct` / `assertive`
- `careful` / `cautious`
- `curt` / `dismissive`
- `anxious`

Important outputs:

```text
all_summaries.csv
all_summaries.json
all_generations.csv
{trait}/high/summary.csv
{trait}/high/{method}/generations.csv
{trait}/high/selector/selector_artifact.json
```

`all_summaries.csv` is the primary comparison table.

## MMLU Style Capability

Runner:

```text
reference_code/qwen3_eval_bundle/qwen3_moe/evals/run_mmlu_style_eval.py
```

Use the same methods and traits as GSM8K. MMLU is much larger, so run bounded subsets before launching a full sweep.

Important outputs match the GSM8K layout.

## Long-Context Dialogue Persistence

Runner:

```text
reference_code/qwen3_eval_bundle/long_context/run_dialogue_persistence_track_a_workflow.py
```

Primary methods:

- `plain`
- `prompt`
- `canonical_memory`

Important outputs:

```text
workflow_summary.csv
{trait}/workflow_summary.csv
dialogue_persistence_rows.csv
dialogue_persistence_turn_rows.csv
dialogue_persistence_summary.csv
run_config.json
```

Judging requires an OpenAI API key. Generation-only smoke runs can use `--skip-judge`.

## Recommended Sequence

1. Run a tiny smoke to verify model load and patch setup.
2. Run `gsm8k_100x6_h200` as the first meaningful bounded result.
3. Inspect `all_summaries.csv` and selector artifacts.
4. Run full GSM8K only after the bounded run looks sane.
5. Run MMLU subsets before full MMLU.
