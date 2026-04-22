from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

try:
    from descriptor_memory_persona.attention_patch import _resolve_attention_dims as _resolve_generic_attention_dims
except ImportError:  # pragma: no cover
    try:
        from hybrid_caa_canonical_debug.attention_patch import _resolve_attention_dims as _resolve_generic_attention_dims
    except ImportError:  # pragma: no cover - supports flat-file layout
        from attention_patch import _resolve_attention_dims as _resolve_generic_attention_dims


@dataclass(frozen=True)
class Qwen3MoeAttentionLayout:
    num_heads: int
    num_key_value_heads: int
    num_query_heads_per_kv_head: int
    head_dim: int


def wrap_qwen_tokenizer_apply_chat_template(tokenizer, *, enable_thinking: bool):
    original = getattr(tokenizer, "apply_chat_template", None)
    if not callable(original):
        return tokenizer

    def wrapped_apply_chat_template(*args, **kwargs):
        if "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = bool(enable_thinking)
        return original(*args, **kwargs)

    setattr(tokenizer, "apply_chat_template", wrapped_apply_chat_template)
    return tokenizer


def inspect_qwen3_moe_architecture(model) -> Dict[str, object]:
    config = getattr(model, "config", None)
    model_body = getattr(model, "model", model)
    layers = getattr(model_body, "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("Expected decoder layers at model.model.layers for Qwen3 inspection.")

    first_layer = layers[0]
    attn = getattr(first_layer, "self_attn", None)
    mlp = getattr(first_layer, "mlp", None)
    if attn is None:
        raise RuntimeError("Expected first decoder layer to expose self_attn.")

    return {
        "model_type": getattr(config, "model_type", None),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", len(layers))),
        "num_attention_heads": int(getattr(config, "num_attention_heads", 0) or 0),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", 0) or 0),
        "num_experts": int(getattr(config, "num_experts", 0) or 0),
        "num_experts_per_tok": int(getattr(config, "num_experts_per_tok", 0) or 0),
        "decoder_sparse_step": int(getattr(config, "decoder_sparse_step", 0) or 0),
        "mlp_only_layers": list(getattr(config, "mlp_only_layers", []) or []),
        "q_proj": hasattr(attn, "q_proj"),
        "k_proj": hasattr(attn, "k_proj"),
        "v_proj": hasattr(attn, "v_proj"),
        "o_proj": hasattr(attn, "o_proj"),
        "q_norm": hasattr(attn, "q_norm") and getattr(attn, "q_norm") is not None,
        "k_norm": hasattr(attn, "k_norm") and getattr(attn, "k_norm") is not None,
        "attn_rotary_emb": hasattr(attn, "rotary_emb") and getattr(attn, "rotary_emb") is not None,
        "model_rotary_emb": hasattr(model_body, "rotary_emb") and getattr(model_body, "rotary_emb") is not None,
        "rope_theta": getattr(config, "rope_theta", None),
        "rope_scaling": getattr(config, "rope_scaling", None),
        "has_moe_mlp": mlp is not None and hasattr(mlp, "gate"),
    }


def assert_qwen3_moe_compatible(model) -> Dict[str, object]:
    report = inspect_qwen3_moe_architecture(model)
    if report.get("model_type") != "qwen3_moe":
        raise RuntimeError(
            "Expected a Qwen3 MoE checkpoint with config.model_type='qwen3_moe', "
            f"but found {report.get('model_type')!r}."
        )
    required_projection_flags = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
    missing = [name for name in required_projection_flags if not report.get(name)]
    if missing:
        raise RuntimeError(
            "The loaded model is missing Qwen3 MoE attention components required by the architecture-safe patch: "
            + ", ".join(missing)
        )
    if not (report.get("attn_rotary_emb") or report.get("model_rotary_emb") or report.get("rope_theta") is not None):
        raise RuntimeError("No Qwen3 RoPE path was detected on the loaded model.")
    return report


def resolve_attention_layout(module: torch.nn.Module) -> Qwen3MoeAttentionLayout:
    num_heads, num_key_value_heads, num_query_heads_per_kv_head, head_dim = _resolve_generic_attention_dims(module)
    return Qwen3MoeAttentionLayout(
        num_heads=int(num_heads),
        num_key_value_heads=int(num_key_value_heads),
        num_query_heads_per_kv_head=int(num_query_heads_per_kv_head),
        head_dim=int(head_dim),
    )


def normalize_head_ids(head_ids: Optional[Sequence[int]], num_heads: int) -> Tuple[int, ...]:
    if not head_ids:
        return tuple(range(int(num_heads)))

    requested = [int(head_id) for head_id in head_ids]
    if requested and max(requested) >= num_heads and min(requested) >= 1 and max(requested) <= num_heads:
        requested = [head_id - 1 for head_id in requested]

    normalized: List[int] = []
    seen = set()
    for head_id in requested:
        if head_id < 0 or head_id >= int(num_heads):
            raise IndexError(f"Head id {head_id} is out of range for num_heads={num_heads}.")
        if head_id not in seen:
            seen.add(head_id)
            normalized.append(head_id)
    return tuple(normalized)


def normalize_kv_group_ids(kv_group_ids: Optional[Sequence[int]], num_key_value_heads: int) -> Tuple[int, ...]:
    if not kv_group_ids:
        return tuple(range(int(num_key_value_heads)))

    normalized: List[int] = []
    seen = set()
    for kv_group_id in kv_group_ids:
        value = int(kv_group_id)
        if value < 0 or value >= int(num_key_value_heads):
            raise IndexError(
                f"KV group id {value} is out of range for num_key_value_heads={num_key_value_heads}."
            )
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return tuple(normalized)


def query_head_to_kv_group(query_head: int, num_query_heads_per_kv_head: int) -> int:
    return int(query_head) // int(num_query_heads_per_kv_head)


def expand_kv_groups_to_query_heads(
    kv_group_ids: Iterable[int],
    num_query_heads_per_kv_head: int,
) -> Tuple[int, ...]:
    query_heads: List[int] = []
    for kv_group_id in sorted({int(group_id) for group_id in kv_group_ids}):
        start = int(kv_group_id) * int(num_query_heads_per_kv_head)
        query_heads.extend(range(start, start + int(num_query_heads_per_kv_head)))
    return tuple(query_heads)


def aggregate_query_head_scores_to_kv_groups(
    head_scores: torch.Tensor,
    *,
    num_key_value_heads: int,
    num_query_heads_per_kv_head: int,
    reduce: str = "sum",
) -> torch.Tensor:
    reduce_mode = str(reduce).strip().lower()
    if head_scores.ndim != 1:
        raise ValueError(f"Expected a 1D head score tensor, got shape {tuple(head_scores.shape)}.")
    if head_scores.shape[0] != int(num_key_value_heads) * int(num_query_heads_per_kv_head):
        raise ValueError(
            "Head score tensor length is incompatible with the Qwen3 MoE GQA layout: "
            f"{head_scores.shape[0]} vs {num_key_value_heads} * {num_query_heads_per_kv_head}."
        )

    grouped = head_scores.view(int(num_key_value_heads), int(num_query_heads_per_kv_head))
    if reduce_mode == "sum":
        return grouped.sum(dim=-1)
    if reduce_mode == "mean":
        return grouped.mean(dim=-1)
    if reduce_mode == "max":
        return grouped.max(dim=-1).values
    raise ValueError(f"Unsupported KV-group reduction '{reduce}'.")


def apply_qk_norm(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_norm = getattr(module, "q_norm", None)
    k_norm = getattr(module, "k_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)
    if k_norm is not None:
        key_states = k_norm(key_states)
    return query_states, key_states


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos_states = cos.unsqueeze(unsqueeze_dim).to(device=query_states.device, dtype=query_states.dtype)
    sin_states = sin.unsqueeze(unsqueeze_dim).to(device=query_states.device, dtype=query_states.dtype)
    query_embed = (query_states * cos_states) + (_rotate_half(query_states) * sin_states)
    key_embed = (key_states * cos_states) + (_rotate_half(key_states) * sin_states)
    return query_embed, key_embed


def apply_qwen3_rotary(
    *,
    module: torch.nn.Module,
    model: Optional[torch.nn.Module],
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    position_ids: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, {"cos": cos, "sin": sin}

    rotary_source = getattr(module, "rotary_emb", None)
    if rotary_source is None and model is not None:
        model_body = getattr(model, "model", model)
        rotary_source = getattr(model_body, "rotary_emb", None)
    if rotary_source is None or position_ids is None:
        return query_states, key_states, {}

    cos, sin = rotary_source(key_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return query_states, key_states, {"cos": cos, "sin": sin}


def resolve_selected_query_heads(
    *,
    layer_id: int,
    layout: Qwen3MoeAttentionLayout,
    global_head_ids: Sequence[int],
    global_kv_group_ids: Sequence[int],
    layer_head_ids: Optional[Mapping[int, Sequence[int]]] = None,
    layer_kv_group_ids: Optional[Mapping[int, Sequence[int]]] = None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    layer_key = int(layer_id)
    chosen_head_ids = None
    chosen_kv_group_ids = None
    if layer_head_ids is not None and layer_key in layer_head_ids:
        chosen_head_ids = normalize_head_ids(layer_head_ids[layer_key], layout.num_heads)
    elif layer_kv_group_ids is not None and layer_key in layer_kv_group_ids:
        chosen_kv_group_ids = normalize_kv_group_ids(
            layer_kv_group_ids[layer_key],
            layout.num_key_value_heads,
        )
    elif global_head_ids:
        chosen_head_ids = normalize_head_ids(global_head_ids, layout.num_heads)
    elif global_kv_group_ids:
        chosen_kv_group_ids = normalize_kv_group_ids(global_kv_group_ids, layout.num_key_value_heads)

    if chosen_kv_group_ids is not None:
        chosen_head_ids = expand_kv_groups_to_query_heads(
            chosen_kv_group_ids,
            layout.num_query_heads_per_kv_head,
        )
    if chosen_head_ids is None:
        chosen_head_ids = tuple(range(layout.num_heads))

    chosen_head_ids = normalize_head_ids(chosen_head_ids, layout.num_heads)
    chosen_kv_group_ids = tuple(
        sorted(
            {
                query_head_to_kv_group(head_id, layout.num_query_heads_per_kv_head)
                for head_id in chosen_head_ids
            }
        )
    )
    return chosen_head_ids, chosen_kv_group_ids
