from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

try:
    from descriptor_memory_persona.memory import _build_memory_input, _resolve_model_device, _resolve_module_execution_device
except ImportError:  # pragma: no cover
    try:
        from hybrid_caa_canonical_debug.memory import _build_memory_input, _resolve_model_device, _resolve_module_execution_device
    except ImportError:  # pragma: no cover - supports flat-file layout
        from memory import _build_memory_input, _resolve_model_device, _resolve_module_execution_device

from architecture import apply_qk_norm, resolve_attention_layout


@dataclass
class CanonicalLayerDescriptorMemory:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class CanonicalDescriptorPhrase:
    phrase_id: str
    phrase_text: str
    source_kind: str
    phrase_index: int
    slot_indices: Tuple[int, ...]
    template_texts: Tuple[str, ...]
    wrapper_texts: Tuple[str, ...]


@dataclass
class CanonicalDescriptorMemory:
    trait_name: str
    descriptor: str
    wrapper_texts: Tuple[str, ...]
    layer_memories: Dict[int, CanonicalLayerDescriptorMemory]
    token_ids: Tuple[int, ...]
    token_texts: Tuple[str, ...]
    token_count: int
    slot_count: int
    template_count: int
    slot_pooling: str
    memory_variant_mode: str
    phrases: Tuple[CanonicalDescriptorPhrase, ...] = ()


@dataclass(frozen=True)
class _ResolvedDescriptorPhraseTemplate:
    phrase_id: str
    phrase_text: str
    source_kind: str
    phrase_index: int
    template_text: str


def _resolve_text_variants(
    *,
    descriptor: str,
    descriptor_variants: Optional[Sequence[str]],
    wrapper_template: str,
    wrapper_templates: Optional[Sequence[str]],
    memory_variant_mode: str = "full_plus_variants",
    variant_mode: Optional[str] = None,
) -> list[_ResolvedDescriptorPhraseTemplate]:
    if variant_mode is not None:
        memory_variant_mode = str(variant_mode)
    normalized_mode = str(memory_variant_mode).strip().lower()
    if normalized_mode not in {"full_plus_variants", "variants_only", "core_only"}:
        raise ValueError(
            f"Unsupported memory_variant_mode '{memory_variant_mode}'. "
            "Expected one of {'full_plus_variants', 'variants_only', 'core_only'}."
        )

    normalized_descriptor = str(descriptor).strip()
    unique_variants: list[str] = []
    seen_variant_texts = set()
    if descriptor_variants is not None:
        for value in descriptor_variants:
            normalized = str(value).strip()
            if normalized and normalized not in seen_variant_texts:
                seen_variant_texts.add(normalized)
                unique_variants.append(normalized)

    phrase_specs: list[tuple[str, str, str, int]] = []
    seen_texts = set()
    if normalized_mode == "core_only":
        core_text = unique_variants[0] if unique_variants else next(
            (line.strip() for line in normalized_descriptor.splitlines() if line.strip()),
            normalized_descriptor,
        )
        if core_text:
            phrase_specs.append(("core_descriptor", core_text, "core_descriptor", 0))
            seen_texts.add(core_text)
    else:
        include_full_descriptor = normalized_mode != "variants_only" or not unique_variants
        if include_full_descriptor and normalized_descriptor:
            phrase_specs.append(("full_descriptor", normalized_descriptor, "full_descriptor", 0))
            seen_texts.add(normalized_descriptor)
        for variant_index, normalized in enumerate(unique_variants):
            if normalized not in seen_texts:
                phrase_specs.append((f"variant_{variant_index}", normalized, "variant", variant_index))
                seen_texts.add(normalized)
    if not phrase_specs:
        phrase_specs = [("full_descriptor", str(descriptor), "full_descriptor", 0)]
    templates = [wrapper_template] if wrapper_templates is None else [str(value) for value in wrapper_templates if str(value)]
    if not templates:
        templates = [wrapper_template]
    return [
        _ResolvedDescriptorPhraseTemplate(
            phrase_id=phrase_id,
            phrase_text=phrase_text,
            source_kind=source_kind,
            phrase_index=phrase_index,
            template_text=template,
        )
        for phrase_id, phrase_text, source_kind, phrase_index in phrase_specs
        for template in templates
    ]


def _pool_slots(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    slot_pooling: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = slot_pooling.strip().lower()
    if mode == "concat":
        return key_states, value_states
    if mode == "mean":
        return key_states.mean(dim=2, keepdim=True), value_states.mean(dim=2, keepdim=True)
    raise ValueError(f"Unsupported slot_pooling '{slot_pooling}'.")


def _project_qwen3_pre_rope_kv(
    attn_module: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    layout = resolve_attention_layout(attn_module)
    kept_len = int(hidden_states.shape[1])
    key_states = attn_module.k_proj(hidden_states).view(
        1,
        kept_len,
        layout.num_key_value_heads,
        layout.head_dim,
    )
    value_states = attn_module.v_proj(hidden_states).view(
        1,
        kept_len,
        layout.num_key_value_heads,
        layout.head_dim,
    )
    dummy_query = torch.zeros(
        1,
        kept_len,
        layout.num_heads,
        layout.head_dim,
        device=hidden_states.device,
        dtype=key_states.dtype,
    )
    _, key_states = apply_qk_norm(attn_module, dummy_query, key_states)
    return key_states.transpose(1, 2), value_states.transpose(1, 2)


def build_canonical_descriptor_memory(
    model: torch.nn.Module,
    tokenizer,
    *,
    trait_name: str,
    descriptor: str,
    layer_ids: Sequence[int],
    wrapper_template: str = "Internal style note: {descriptor}",
    wrapper_templates: Optional[Sequence[str]] = None,
    descriptor_variants: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    use_chat_template: bool = True,
    chat_role: str = "system",
    keep_descriptor_only: bool = True,
    slot_pooling: str = "concat",
    memory_variant_mode: str = "full_plus_variants",
) -> CanonicalDescriptorMemory:
    model_device = _resolve_model_device(model) if device is None else device
    model_layers = getattr(getattr(model, "model", model), "layers", None)
    if model_layers is None:
        raise AttributeError("Expected model.model.layers to exist.")

    text_variants = _resolve_text_variants(
        descriptor=descriptor,
        descriptor_variants=descriptor_variants,
        wrapper_template=wrapper_template,
        wrapper_templates=wrapper_templates,
        memory_variant_mode=memory_variant_mode,
    )

    per_layer_keys: Dict[int, list[torch.Tensor]] = {int(layer_id): [] for layer_id in layer_ids}
    per_layer_values: Dict[int, list[torch.Tensor]] = {int(layer_id): [] for layer_id in layer_ids}
    wrapper_texts: list[str] = []
    token_ids: list[int] = []
    token_texts: list[str] = []
    phrase_slot_indices: Dict[str, list[int]] = {}
    phrase_template_texts: Dict[str, list[str]] = {}
    phrase_wrapper_texts: Dict[str, list[str]] = {}
    phrase_text_by_id: Dict[str, str] = {}
    phrase_source_kind_by_id: Dict[str, str] = {}
    phrase_index_by_id: Dict[str, int] = {}
    phrase_order: list[str] = []
    slot_cursor = 0
    slot_pooling_mode = str(slot_pooling).strip().lower()

    for resolved_phrase in text_variants:
        _, wrapper_text, descriptor_positions = _build_memory_input(
            tokenizer,
            descriptor=resolved_phrase.phrase_text,
            wrapper_template=resolved_phrase.template_text,
            use_chat_template=use_chat_template,
            chat_role=chat_role,
        )
        wrapper_texts.append(wrapper_text)
        tokens = tokenizer(wrapper_text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(model_device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_state_stack = getattr(outputs, "hidden_states", None)
        if hidden_state_stack is None:
            raise RuntimeError("Model did not return hidden_states. Expected output_hidden_states=True support.")

        seq_len = int(input_ids.shape[1])
        selected_positions = descriptor_positions if (keep_descriptor_only and descriptor_positions is not None) else tuple(range(seq_len))
        if not selected_positions:
            selected_positions = tuple(range(seq_len))
        if slot_pooling_mode == "concat":
            slot_width = len(selected_positions)
        elif slot_pooling_mode == "mean":
            slot_width = 1
        else:
            raise ValueError(f"Unsupported slot_pooling '{slot_pooling}'.")

        phrase_id = str(resolved_phrase.phrase_id)
        if phrase_id not in phrase_slot_indices:
            phrase_slot_indices[phrase_id] = []
            phrase_template_texts[phrase_id] = []
            phrase_wrapper_texts[phrase_id] = []
            phrase_text_by_id[phrase_id] = str(resolved_phrase.phrase_text)
            phrase_source_kind_by_id[phrase_id] = str(resolved_phrase.source_kind)
            phrase_index_by_id[phrase_id] = int(resolved_phrase.phrase_index)
            phrase_order.append(phrase_id)
        phrase_slot_indices[phrase_id].extend(range(slot_cursor, slot_cursor + slot_width))
        phrase_template_texts[phrase_id].append(str(resolved_phrase.template_text))
        phrase_wrapper_texts[phrase_id].append(wrapper_text)
        slot_cursor += slot_width

        token_ids.extend(int(input_ids[0, idx].item()) for idx in selected_positions)
        token_texts.extend(tokenizer.convert_ids_to_tokens([int(input_ids[0, idx].item()) for idx in selected_positions]))

        for layer_id in layer_ids:
            if int(layer_id) >= len(hidden_state_stack):
                raise IndexError(
                    f"Requested layer {layer_id}, but only {len(hidden_state_stack) - 1} decoder hidden-state entries exist."
                )
            attn_module = model_layers[int(layer_id)].self_attn
            layer_module = model_layers[int(layer_id)]
            layer_device = _resolve_module_execution_device(attn_module, model_device)

            hidden_states = hidden_state_stack[int(layer_id)].to(layer_device)
            input_layernorm = getattr(layer_module, "input_layernorm", None)
            if input_layernorm is not None:
                hidden_states = input_layernorm(hidden_states)

            index_tensor = torch.tensor(selected_positions, device=layer_device, dtype=torch.long)
            hidden_states = hidden_states.index_select(dim=1, index=index_tensor)
            # Canonical memory stores post-k_norm, pre-RoPE keys so retrieval matches
            # Qwen3's attention pipeline without tying the descriptor bank to a fixed position.
            key_states, value_states = _project_qwen3_pre_rope_kv(attn_module, hidden_states)
            key_states, value_states = _pool_slots(key_states, value_states, slot_pooling=slot_pooling)
            per_layer_keys[int(layer_id)].append(key_states.squeeze(0).detach().cpu())
            per_layer_values[int(layer_id)].append(value_states.squeeze(0).detach().cpu())

    layer_memories: Dict[int, CanonicalLayerDescriptorMemory] = {}
    for layer_id in layer_ids:
        key_slots = per_layer_keys[int(layer_id)]
        value_slots = per_layer_values[int(layer_id)]
        if not key_slots or not value_slots:
            raise RuntimeError(f"No canonical descriptor slots were collected for layer {layer_id}.")
        layer_memories[int(layer_id)] = CanonicalLayerDescriptorMemory(
            key=torch.cat(key_slots, dim=1),
            value=torch.cat(value_slots, dim=1),
        )

    phrases = tuple(
        CanonicalDescriptorPhrase(
            phrase_id=phrase_id,
            phrase_text=phrase_text_by_id[phrase_id],
            source_kind=phrase_source_kind_by_id[phrase_id],
            phrase_index=phrase_index_by_id[phrase_id],
            slot_indices=tuple(int(slot_index) for slot_index in phrase_slot_indices.get(phrase_id, [])),
            template_texts=tuple(phrase_template_texts.get(phrase_id, [])),
            wrapper_texts=tuple(phrase_wrapper_texts.get(phrase_id, [])),
        )
        for phrase_id in phrase_order
    )

    return CanonicalDescriptorMemory(
        trait_name=trait_name,
        descriptor=descriptor,
        wrapper_texts=tuple(wrapper_texts),
        layer_memories=layer_memories,
        token_ids=tuple(token_ids),
        token_texts=tuple(token_texts),
        token_count=len(token_ids),
        slot_count=sum(int(layer_memories[int(layer_id)].key.shape[1]) for layer_id in layer_ids) // max(len(tuple(layer_ids)), 1),
        template_count=len(text_variants),
        slot_pooling=slot_pooling,
        memory_variant_mode=str(memory_variant_mode).strip().lower(),
        phrases=phrases,
    )
