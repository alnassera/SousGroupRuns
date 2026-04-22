from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch


@dataclass
class LayerDescriptorMemory:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class DescriptorMemory:
    trait_name: str
    descriptor: str
    wrapper_text: str
    layer_memories: Dict[int, LayerDescriptorMemory]
    token_ids: Tuple[int, ...]
    token_texts: Tuple[str, ...]
    token_count: int


def _resolve_model_device(model: torch.nn.Module) -> torch.device:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        if embeddings is not None:
            for parameter in embeddings.parameters():
                if parameter.device.type != "meta":
                    return parameter.device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _normalize_device(device_like) -> Optional[torch.device]:
    if device_like is None:
        return None
    try:
        device = torch.device(device_like)
    except (TypeError, RuntimeError, ValueError):
        return None
    if device.type == "meta":
        return None
    return device


def _resolve_module_execution_device(module: torch.nn.Module, fallback: torch.device) -> torch.device:
    hf_hook = getattr(module, "_hf_hook", None)
    if hf_hook is not None:
        execution_device = _normalize_device(getattr(hf_hook, "execution_device", None))
        if execution_device is not None:
            return execution_device

    for parameter in module.parameters():
        parameter_device = _normalize_device(parameter.device)
        if parameter_device is not None:
            return parameter_device

    return fallback


def _locate_descriptor_token_positions(
    tokenizer,
    *,
    formatted_text: str,
    descriptor: str,
) -> Optional[Tuple[int, ...]]:
    if not descriptor:
        return None
    descriptor_start = formatted_text.find(descriptor)
    if descriptor_start < 0:
        return None
    descriptor_end = descriptor_start + len(descriptor)

    try:
        tokenized = tokenizer(
            formatted_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception:
        return None

    offsets = tokenized.get("offset_mapping")
    if offsets is None:
        return None

    positions = []
    for idx, offset in enumerate(offsets):
        if offset is None or len(offset) != 2:
            continue
        start, end = int(offset[0]), int(offset[1])
        if end <= descriptor_start or start >= descriptor_end:
            continue
        if end > start:
            positions.append(idx)
    return tuple(positions) if positions else None


def _build_memory_input(
    tokenizer,
    *,
    descriptor: str,
    wrapper_template: str,
    use_chat_template: bool,
    chat_role: str,
) -> tuple[str, str, tuple[int, ...] | None]:
    content_text = wrapper_template.format(descriptor=descriptor)
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        formatted_text = tokenizer.apply_chat_template(
            [{"role": chat_role, "content": content_text}],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        formatted_text = content_text
    descriptor_positions = _locate_descriptor_token_positions(
        tokenizer,
        formatted_text=formatted_text,
        descriptor=descriptor,
    )
    return content_text, formatted_text, descriptor_positions


def _resolve_attention_dims(module: torch.nn.Module) -> tuple[int, int, int]:
    config = getattr(module, "config", None)

    num_heads = getattr(module, "num_heads", None)
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_attention_heads", None)
    if num_heads is None:
        q_out = getattr(getattr(module, "q_proj", None), "out_features", None)
        head_dim = getattr(module, "head_dim", None)
        if q_out is not None and head_dim is not None:
            num_heads = q_out // head_dim

    head_dim = getattr(module, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(module, "q_proj", None), "out_features", None)
        if hidden_size is not None and num_heads is not None:
            head_dim = hidden_size // num_heads

    num_kv_heads = getattr(module, "num_key_value_heads", None)
    if num_kv_heads is None and config is not None:
        num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        k_out = getattr(getattr(module, "k_proj", None), "out_features", None)
        if k_out is not None and head_dim is not None:
            num_kv_heads = k_out // head_dim

    if num_heads is None or head_dim is None or num_kv_heads is None:
        raise AttributeError(
            "Could not infer attention dimensions from the module. "
            "Expected num_heads/head_dim/num_key_value_heads or compatible config/projection shapes."
        )

    return int(num_heads), int(num_kv_heads), int(head_dim)


def _apply_rotary_to_memory(
    model: torch.nn.Module,
    module: torch.nn.Module,
    key_states: torch.Tensor,
    *,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except Exception as exc:  # pragma: no cover - import failure depends on local env
        raise RuntimeError("transformers is required to build descriptor memories.") from exc

    rotary_source = getattr(module, "rotary_emb", None)
    if rotary_source is None:
        rotary_source = getattr(getattr(model, "model", model), "rotary_emb", None)
    if rotary_source is None:
        return key_states

    dummy_query = torch.zeros_like(key_states)
    try:
        cos, sin = rotary_source(key_states, position_ids)
    except TypeError:
        cos, sin = rotary_source(key_states, seq_len=position_ids.max().item() + 1)
    try:
        _, rotated_key = apply_rotary_pos_emb(dummy_query, key_states, cos, sin, position_ids)
    except TypeError:
        _, rotated_key = apply_rotary_pos_emb(dummy_query, key_states, cos, sin)
    return rotated_key


def build_descriptor_memory(
    model: torch.nn.Module,
    tokenizer,
    *,
    trait_name: str,
    descriptor: str,
    layer_ids: Sequence[int],
    wrapper_template: str = "[TRAIT] {descriptor} [/TRAIT]",
    device: Optional[torch.device] = None,
    use_chat_template: bool = True,
    chat_role: str = "system",
    keep_descriptor_only: bool = True,
) -> DescriptorMemory:
    _, wrapper_text, descriptor_positions = _build_memory_input(
        tokenizer,
        descriptor=descriptor,
        wrapper_template=wrapper_template,
        use_chat_template=use_chat_template,
        chat_role=chat_role,
    )
    tokens = tokenizer(wrapper_text, return_tensors="pt", add_special_tokens=False)
    model_device = _resolve_model_device(model) if device is None else device
    input_ids = tokens["input_ids"].to(model_device)
    attention_mask = tokens.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)

    model_layers = getattr(getattr(model, "model", model), "layers", None)
    if model_layers is None:
        raise AttributeError("Expected model.model.layers to exist.")

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
    layer_memories: Dict[int, LayerDescriptorMemory] = {}

    for layer_id in layer_ids:
        if int(layer_id) >= len(hidden_state_stack):
            raise IndexError(
                f"Requested layer {layer_id}, but only {len(hidden_state_stack) - 1} decoder hidden-state entries exist."
            )
        attn_module = model_layers[int(layer_id)].self_attn
        layer_module = model_layers[int(layer_id)]
        layer_device = _resolve_module_execution_device(attn_module, model_device)
        # In decoder-only HF models, hidden_states[0] is the embedding output,
        # and hidden_states[layer_id] is the input to decoder layer `layer_id`.
        hidden_states = hidden_state_stack[int(layer_id)].to(layer_device)
        input_layernorm = getattr(layer_module, "input_layernorm", None)
        if input_layernorm is not None:
            hidden_states = input_layernorm(hidden_states)

        index_tensor = torch.tensor(selected_positions, device=layer_device, dtype=torch.long)
        hidden_states = hidden_states.index_select(dim=1, index=index_tensor)
        _, num_kv_heads, head_dim = _resolve_attention_dims(attn_module)
        kept_len = int(hidden_states.shape[1])
        position_ids = torch.arange(kept_len, device=layer_device).unsqueeze(0)

        key_states = attn_module.k_proj(hidden_states).view(1, kept_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = attn_module.v_proj(hidden_states).view(1, kept_len, num_kv_heads, head_dim).transpose(1, 2)
        key_states = _apply_rotary_to_memory(model, attn_module, key_states, position_ids=position_ids)

        layer_memories[int(layer_id)] = LayerDescriptorMemory(
            key=key_states.squeeze(0).detach().cpu(),
            value=value_states.squeeze(0).detach().cpu(),
        )

    token_list = tuple(int(input_ids[0, idx].item()) for idx in selected_positions)
    token_texts = tuple(tokenizer.convert_ids_to_tokens(list(token_list)))
    return DescriptorMemory(
        trait_name=trait_name,
        descriptor=descriptor,
        wrapper_text=wrapper_text,
        layer_memories=layer_memories,
        token_ids=token_list,
        token_texts=token_texts,
        token_count=len(token_list),
    )
