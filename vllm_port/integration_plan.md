# vLLM Intervention Integration Plan

## Current State

The reference HF/Transformers interventions live in:

```text
reference_code/qwen3_eval_bundle/post_attn_caa_patch.py
reference_code/qwen3_eval_bundle/canonical_attention_patch.py
```

The vLLM baseline runner lives in:

```text
reference_code/qwen3_eval_bundle/qwen3_moe/evals/run_gsm8k_vllm_eval.py
```

## Step 1: Post-Attention CAA

Integration target:

```text
vllm.model_executor.models.qwen3_moe
```

Desired insertion point:

1. Let vLLM Qwen3-MoE attention compute the normal projected attention output.
2. Before the output is added to the residual stream, call `apply_post_attention_caa`.
3. Pass the same layer vectors currently produced by `build_site_activation_steering_vector_from_examples`.

Parity test:

```text
HF post_attn_caa, max_new_tokens=768, batch_size=32
vLLM post_attn_caa, max_new_tokens=768, batch_size=128
```

Gate:

- prediction match >= 97%
- absolute accuracy delta <= 5 percentage points
- no systematic trait collapse

## Step 2: Canonical Memory

The hard part is not the bank-mixing math; it is getting the same selected query heads, RoPE-inverted query representation, prompt scores, prompt values, and cache behavior from vLLM's attention path.

Required attention-path data:

- projected query/key/value states
- QK-normalized states
- RoPE cos/sin or equivalent inverse-rotation path
- repeated KV groups mapped to query heads
- selected query heads from selector artifacts
- prompt-token attention scores and values
- descriptor-memory key/value banks
- optional reference memory banks

Likely path:

1. Fork or subclass the vLLM Qwen3-MoE model implementation in a local plugin/module.
2. Add a custom attention wrapper for selected layers.
3. Reuse `bank_softmax_mixture` for the default canonical-memory operator.
4. Keep `post_attn_caa` and `canonical_memory` controlled by explicit per-run config, never implicit global state.

## Open Questions

- Whether vLLM's custom model loading path can load a local Qwen3-MoE subclass cleanly without vendoring the whole model executor file.
- Whether the vLLM attention backend exposes enough intermediate state, or whether selected layers need an eager/custom attention branch.
- Whether parity should be judged against HF SDPA or HF FlashAttention when available. Current reference is HF SDPA.
