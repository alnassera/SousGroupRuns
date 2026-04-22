from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch

try:
    from .canonical_attention_patch import CanonicalDescriptorMemoryAttentionPatch, CanonicalDescriptorMixConfig
    from .canonical_memory import CanonicalDescriptorMemory
    from descriptor_memory_persona.runtime import (
        GenerationConfig,
        GenerationResult,
        _build_batch,
        _generate_with_optional_patch,
        _resolve_model_device,
    )
except ImportError:  # pragma: no cover - supports flat-file layout
    from canonical_attention_patch import CanonicalDescriptorMemoryAttentionPatch, CanonicalDescriptorMixConfig
    from canonical_memory import CanonicalDescriptorMemory
    from runtime import (
        GenerationConfig,
        GenerationResult,
        _build_batch,
        _generate_with_optional_patch,
        _resolve_model_device,
    )


def generate_with_canonical_memory(
    model,
    tokenizer,
    *,
    prompt: str,
    generation_config: GenerationConfig,
    memory: Optional[CanonicalDescriptorMemory],
    reference_memory: Optional[CanonicalDescriptorMemory],
    layer_ids: Sequence[int],
    rho: float,
    kv_group_ids: Sequence[int] = (),
    layer_head_ids: Optional[Mapping[int, Sequence[int]]] = None,
    layer_kv_group_ids: Optional[Mapping[int, Sequence[int]]] = None,
) -> GenerationResult:
    device = _resolve_model_device(model)
    formatted_prompt, batch = _build_batch(tokenizer, prompt, device)

    patch = None
    if memory is not None and layer_ids:
        patch = CanonicalDescriptorMemoryAttentionPatch(
            layer_ids=layer_ids,
            memory=memory,
            reference_memory=reference_memory,
            config=CanonicalDescriptorMixConfig(
                rho=float(rho),
                mix_mode=generation_config.mix_mode,
                prefill_steering=generation_config.prefill_steering,
                steering_operator=generation_config.steering_operator,
                positive_gain=generation_config.positive_gain,
                negative_gain=generation_config.negative_gain,
                query_adaptive_gates=generation_config.query_adaptive_gates,
                query_gate_scale=generation_config.query_gate_scale,
                prompt_bank_normalization=generation_config.prompt_bank_normalization,
                diagnostic_max_decode_steps=generation_config.diagnostic_max_decode_steps,
                head_ids=tuple(int(head_id) for head_id in generation_config.head_ids),
                kv_group_ids=tuple(int(kv_group_id) for kv_group_id in kv_group_ids),
                layer_head_ids=None if layer_head_ids is None else {
                    int(layer_id): tuple(int(head_id) for head_id in head_values)
                    for layer_id, head_values in layer_head_ids.items()
                },
                layer_kv_group_ids=None if layer_kv_group_ids is None else {
                    int(layer_id): tuple(int(kv_group_id) for kv_group_id in kv_values)
                    for layer_id, kv_values in layer_kv_group_ids.items()
                },
            ),
        )

    generate_kwargs = dict(**batch, use_cache=True)
    hf_generation_config = None
    try:
        from transformers import GenerationConfig as HFGenerationConfig

        source_config = getattr(model, "generation_config", None)
        if source_config is not None:
            if hasattr(source_config, "to_dict"):
                hf_generation_config = HFGenerationConfig.from_dict(source_config.to_dict())
            else:
                hf_generation_config = HFGenerationConfig.from_model_config(model.config)
        else:
            hf_generation_config = HFGenerationConfig.from_model_config(model.config)
        hf_generation_config.max_new_tokens = int(generation_config.max_new_tokens)
        hf_generation_config.do_sample = bool(generation_config.do_sample)
        hf_generation_config.repetition_penalty = float(generation_config.repetition_penalty)
        hf_generation_config.pad_token_id = tokenizer.pad_token_id
        hf_generation_config.eos_token_id = tokenizer.eos_token_id
        if generation_config.do_sample:
            hf_generation_config.temperature = float(generation_config.temperature)
            hf_generation_config.top_p = float(generation_config.top_p)
        else:
            hf_generation_config.temperature = 1.0
            hf_generation_config.top_p = 1.0
        if generation_config.seed is not None:
            try:
                hf_generation_config.seed = int(generation_config.seed)
            except Exception:
                pass
        generate_kwargs["generation_config"] = hf_generation_config
    except Exception:
        generate_kwargs.update(
            dict(
                max_new_tokens=int(generation_config.max_new_tokens),
                do_sample=bool(generation_config.do_sample),
                repetition_penalty=float(generation_config.repetition_penalty),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        )
        if generation_config.do_sample:
            generate_kwargs.update(
                temperature=float(generation_config.temperature),
                top_p=float(generation_config.top_p),
            )

    generated = _generate_with_optional_patch(model, generate_kwargs, patch)
    prompt_len = int(batch["input_ids"].shape[1])
    text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()
    generated_len = max(0, int(generated.shape[-1]) - prompt_len)

    return GenerationResult(
        mode="canonical_memory" if memory is not None else "baseline",
        text=text,
        formatted_prompt=formatted_prompt,
        prompt_token_count=prompt_len,
        generated_token_count=generated_len,
        diagnostics={
            "layer_ids": [int(layer_id) for layer_id in layer_ids],
            "rho": float(rho),
            "memory_token_count": int(memory.token_count) if memory is not None else 0,
            "memory_slot_count": int(memory.slot_count) if memory is not None else 0,
            "reference_memory_token_count": int(reference_memory.token_count) if reference_memory is not None else 0,
            "reference_memory_slot_count": int(reference_memory.slot_count) if reference_memory is not None else 0,
            "mix_mode": generation_config.mix_mode,
            "prefill_steering": generation_config.prefill_steering,
            "steering_operator": generation_config.steering_operator,
            "positive_gain": float(generation_config.positive_gain),
            "negative_gain": float(generation_config.negative_gain),
            "query_adaptive_gates": bool(generation_config.query_adaptive_gates),
            "query_gate_scale": float(generation_config.query_gate_scale),
            "prompt_bank_normalization": str(generation_config.prompt_bank_normalization),
            "diagnostic_max_decode_steps": int(generation_config.diagnostic_max_decode_steps),
            "head_ids": [int(head_id) for head_id in generation_config.head_ids],
            "kv_group_ids": [int(kv_group_id) for kv_group_id in kv_group_ids],
            "layer_head_ids": None if layer_head_ids is None else {
                str(int(layer_id)): [int(head_id) for head_id in head_values]
                for layer_id, head_values in layer_head_ids.items()
            },
            "layer_kv_group_ids": None if layer_kv_group_ids is None else {
                str(int(layer_id)): [int(kv_group_id) for kv_group_id in kv_values]
                for layer_id, kv_values in layer_kv_group_ids.items()
            },
            "diagnostic_traces": [] if patch is None else list(getattr(patch, "diagnostic_traces", [])),
        },
    )
