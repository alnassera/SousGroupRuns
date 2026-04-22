# SousGroupRuns

This repository is organized around Qwen3 MoE style-steering evals, with results and the vLLM port work kept first-class.

The root is intentionally small:

- `results/` contains curated run summaries and small tracked artifacts.
- `vllm_port/` contains the vLLM-compatible intervention port in progress.
- `reference_code/qwen3_eval_bundle/` keeps the runnable research bundle for reproducibility.

Large raw run folders should stay on the GPU node or under `results/raw/`, which is ignored by default.

## Current Readout

The latest curated summary is:

```text
results/summary.json
```

The practical decision captured there is: use vLLM at `max_new_tokens=768` for fast `plain` and `prompt` exploration, and keep HF/Transformers as the final confirmation path until the vLLM port passes a fidelity gate.

## Reproducing Reference Runs

From a GPU machine with enough VRAM and the model downloaded or accessible:

```bash
cd reference_code/qwen3_eval_bundle
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export HF_HOME=/workspace/hf_cache
export TOKENIZERS_PARALLELISM=false
```

Transformers reference run:

```bash
python qwen3_moe/evals/run_gsm8k_style_eval.py \
  --target-traits warm formal direct careful curt anxious \
  --methods plain prompt post_attn_caa canonical_memory \
  --max-items 100 \
  --max-new-tokens 768 \
  --batch-size 32 \
  --device-map cuda \
  --attn-implementation sdpa \
  --output-dir ../../results/raw/gsm8k_hf_768
```

vLLM plain/prompt baseline:

```bash
python qwen3_moe/evals/run_gsm8k_vllm_eval.py \
  --target-traits warm formal direct careful curt anxious \
  --methods plain prompt \
  --max-items 100 \
  --max-new-tokens 768 \
  --batch-size 128 \
  --max-num-seqs 128 \
  --output-dir ../../results/raw/gsm8k_vllm_plain_prompt_768
```

## Code Reference

The full runnable bundle remains under:

```text
reference_code/qwen3_eval_bundle/
```

That directory contains the original README, dataset notes, model/eval scripts, long-context workflow, profiles, and reference notes.
