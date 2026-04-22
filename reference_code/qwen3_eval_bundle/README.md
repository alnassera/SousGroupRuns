# Qwen3 Collaboration Eval Bundle

This folder is meant to be zipped and handed to a collaborator. It keeps the runnable pieces for:

- downstream capability evals on `GSM8K` and `MMLU`
- methods `plain`, `prompt`, `post_attn_caa`, `canonical_memory`
- long-context dialogue-persistence generation with the assistant-register opposite-trait prefill path

The downstream evals default to the six style traits used for the Llama comparison:

- `warm`
- `formal`
- `assertive` (`direct` is accepted as an alias)
- `cautious` (`careful` is accepted as an alias)
- `dismissive` (`curt` is accepted as an alias)
- `anxious`

## Environment

Use Python 3.10+ and install:

```bash
pip install -r requirements.txt
```

The expected pinned versions are in [requirements.txt](requirements.txt).

Model weights are not bundled. The scripts assume your collaborator can access `Qwen/Qwen3-30B-A3B` through the local Hugging Face cache or by downloading it on their machine.

## Bundle layout

- `qwen3_moe/evals/run_gsm8k_style_eval.py`
- `qwen3_moe/evals/run_mmlu_style_eval.py`
- `long_context/run_dialogue_persistence_track_a_workflow.py`
- `datasets/`
- `profiles/dialogue_persistence/style_traits_v1.json`
- `reference_notes/`

## Dataset notes

`GSM8K` is already bundled locally in `datasets/gsm8k/test.jsonl`.

`MMLU` is fully wired in the runner, but the raw dataset was not present in the original workspace cache when this bundle was packaged, so the shipped `datasets/mmlu/dev.jsonl` and `datasets/mmlu/test.jsonl` are placeholders. Populate them once with:

```bash
python datasets/scripts/prepare_mmlu_jsonl.py
```

More detail is in [datasets/README.md](datasets/README.md).

## Downstream eval commands

### GSM8K smoke run

```bash
python qwen3_moe/evals/run_gsm8k_style_eval.py \
  --target-traits warm \
  --methods plain prompt post_attn_caa canonical_memory \
  --max-items 5 \
  --output-dir runs/gsm8k_smoke
```

### GSM8K full trait sweep

```bash
python qwen3_moe/evals/run_gsm8k_style_eval.py \
  --target-traits warm formal direct careful curt anxious \
  --methods plain prompt post_attn_caa canonical_memory \
  --output-dir runs/gsm8k_full
```

### MMLU smoke run

Populate `datasets/mmlu/*.jsonl` first, then run:

```bash
python qwen3_moe/evals/run_mmlu_style_eval.py \
  --target-traits warm \
  --methods plain prompt post_attn_caa canonical_memory \
  --subjects abstract_algebra anatomy \
  --max-items 10 \
  --few-shot-k 5 \
  --output-dir runs/mmlu_smoke
```

### MMLU full trait sweep

```bash
python qwen3_moe/evals/run_mmlu_style_eval.py \
  --target-traits warm formal direct careful curt anxious \
  --methods plain prompt post_attn_caa canonical_memory \
  --few-shot-k 5 \
  --output-dir runs/mmlu_full
```

## Long-context runs

The most relevant long-context setting from the later notes was the assistant-register opposite-trait prefill before the conversation starts. This bundle keeps only the assistant-register assets needed for:

- `assistant_register_interference`
- `assistant_register_occupancy`

The workflow now defaults to `assistant_register_interference`.

### Long-context smoke run without OpenAI judging

```bash
python long_context/run_dialogue_persistence_track_a_workflow.py \
  --target-traits warm \
  --methods plain prompt canonical_memory \
  --prefill-mode assistant_register_interference \
  --case-limit 2 \
  --max-turns 4 \
  --skip-judge \
  --output-dir runs/long_context_smoke_interference
```

### Matched occupancy control

```bash
python long_context/run_dialogue_persistence_track_a_workflow.py \
  --target-traits warm \
  --methods plain prompt canonical_memory \
  --prefill-mode assistant_register_occupancy \
  --case-limit 2 \
  --max-turns 4 \
  --skip-judge \
  --output-dir runs/long_context_smoke_occupancy
```

### Judged run

Judging requires `OPENAI_API_KEY` or `--openai-api-key`.

```bash
python long_context/run_dialogue_persistence_track_a_workflow.py \
  --target-traits warm \
  --methods plain prompt canonical_memory \
  --prefill-mode assistant_register_interference \
  --case-limit 12 \
  --max-turns 8 \
  --judge-model gpt-4o-mini \
  --output-dir runs/long_context_judged
```

## Outputs

The downstream runners write:

- `config.json`
- per-trait method folders with `generations.jsonl` and `generations.csv`
- selector artifacts under each trait folder when steering is used
- `all_generations.csv`
- `all_summaries.csv`
- `all_summaries.json`

The long-context workflow writes its usual run directory with transcripts, CSV summaries, and optional judge outputs.

## Notes

- `plain` and `prompt` do not need a selector.
- `post_attn_caa` and `canonical_memory` reuse the same style selector pass so the layer subset is consistent within a trait run.
- OpenAI is not required for GSM8K or MMLU generation runs.
- OpenAI is only needed for long-context judging or for regenerating assistant-register stylized contexts with `long_context/prepare_assistant_register_contexts.py`.
- Reference notes from the original workspace are copied into `reference_notes/` for context.
