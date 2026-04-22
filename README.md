# SousGroupRuns

This repository is organized around evaluation runs for Qwen3 MoE style-steering experiments.

The root is intentionally results-first:

- `evals/README.md` describes the evaluation suites and expected outputs.
- `runs/` is where tracked result folders should live.
- `reference_code/qwen3_eval_bundle/` keeps the runnable research bundle for reproducibility.

The original code is kept as reference material rather than living at repository root.

## Current Priority Run

The active H200 run is:

```bash
python qwen3_moe/evals/run_gsm8k_style_eval.py \
  --target-traits warm formal direct careful curt anxious \
  --methods plain prompt post_attn_caa canonical_memory \
  --max-items 100 \
  --device-map cuda \
  --attn-implementation sdpa \
  --output-dir runs/gsm8k_100x6_h200
```

When copied into this repo, the result folder should be stored as:

```text
runs/gsm8k_100x6_h200/
```

The first files to inspect are:

```text
runs/gsm8k_100x6_h200/all_summaries.csv
runs/gsm8k_100x6_h200/all_summaries.json
```

## Reproducing From A Fresh Clone

From a GPU machine with enough VRAM and the model downloaded or accessible:

```bash
cd reference_code/qwen3_eval_bundle
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export HF_HOME=/workspace/hf_cache
export TOKENIZERS_PARALLELISM=false
```

Write outputs back to the root-level `runs/` directory:

```bash
python qwen3_moe/evals/run_gsm8k_style_eval.py \
  --target-traits warm formal direct careful curt anxious \
  --methods plain prompt post_attn_caa canonical_memory \
  --max-items 100 \
  --device-map cuda \
  --attn-implementation sdpa \
  --output-dir ../../runs/gsm8k_100x6_h200
```

## Code Reference

The full runnable bundle remains under:

```text
reference_code/qwen3_eval_bundle/
```

That directory contains the original README, dataset notes, model/eval scripts, long-context workflow, profiles, and reference notes.
