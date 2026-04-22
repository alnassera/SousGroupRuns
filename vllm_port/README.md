# vLLM Port

This directory is the working area for porting the steering interventions from the HF/Transformers reference bundle into a vLLM-compatible path.

The current target is Qwen3-MoE on vLLM, using the same prompts, style traits, steering vectors, selector artifacts, and GSM8K scoring logic as the reference code.

## Why This Exists

The vLLM plain/prompt run at `max_new_tokens=768` completed 1,200 GSM8K generations in about 73 seconds of generation time. The comparable HF/Transformers calibration is running at roughly 6.6 seconds per generation for `warm/plain`.

That is about a 100x generation-speed difference, which is large enough to justify a port if fidelity stays near the current calibration target.

## Fidelity Gate

Before treating vLLM as more than a fast exploration path, compare against HF/Transformers with the same:

- model snapshot
- prompt formatting
- `max_new_tokens`
- deterministic decoding settings
- answer extractor

Current partial HF 768 calibration:

- `warm/plain`: 100 shared rows, 97.0% prediction match, vLLM 98.0% accuracy, HF 97.0% accuracy
- `warm/prompt`: 32 shared rows, 100.0% prediction match, both 96.9% accuracy

Working gate:

- prediction match >= 97%
- absolute accuracy delta <= 5 percentage points
- no trait-specific collapse

## Port Order

1. `post_attn_caa`

   This is the smaller intervention. The HF reference adds a layer-specific vector to the projected attention output before that output rejoins the residual stream.

2. `canonical_memory`

   This is deeper. It recomputes selected attention heads against descriptor-memory banks, then mixes prompt, trait, and optional reference banks. A faithful vLLM port likely needs a custom Qwen3-MoE attention path.

## Files

- `post_attention_caa.py`: engine-local tensor helper for the post-attention residual CAA intervention.
- `canonical_memory.py`: starter tensor helper for canonical-memory bank mixing.
- `integration_plan.md`: concrete integration steps and open questions.

These files are scaffolding. The reference implementation remains in `reference_code/qwen3_eval_bundle/`.
