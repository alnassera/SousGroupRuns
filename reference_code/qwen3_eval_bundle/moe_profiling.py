from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from architecture import (
    aggregate_query_head_scores_to_kv_groups,
    expand_kv_groups_to_query_heads,
    resolve_attention_layout,
)


@dataclass(frozen=True)
class LayerMoEProfile:
    layer_id: int
    head_scores: Tuple[float, ...]
    kv_group_scores: Tuple[float, ...]
    router_probs: Tuple[float, ...]
    top_experts: Tuple[int, ...]
    top_expert_weights: Tuple[float, ...]
    routing_concentration: float
    moe_output_norm: float
    layer_score: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer_id": int(self.layer_id),
            "head_scores": [float(value) for value in self.head_scores],
            "kv_group_scores": [float(value) for value in self.kv_group_scores],
            "router_probs": [float(value) for value in self.router_probs],
            "top_experts": [int(value) for value in self.top_experts],
            "top_expert_weights": [float(value) for value in self.top_expert_weights],
            "routing_concentration": float(self.routing_concentration),
            "moe_output_norm": float(self.moe_output_norm),
            "layer_score": float(self.layer_score),
        }


@dataclass(frozen=True)
class Qwen3MoeProfileArtifact:
    model_name: str
    source_text: str
    source_span_text: Optional[str]
    source_token_indices: Tuple[int, ...]
    selected_layer_ids: Tuple[int, ...]
    selected_kv_group_ids_by_layer: Mapping[int, Tuple[int, ...]]
    selected_query_head_ids_by_layer: Mapping[int, Tuple[int, ...]]
    layer_profiles: Tuple[LayerMoEProfile, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_name": str(self.model_name),
            "source_text": str(self.source_text),
            "source_span_text": None if self.source_span_text is None else str(self.source_span_text),
            "source_token_indices": [int(value) for value in self.source_token_indices],
            "selected_layer_ids": [int(value) for value in self.selected_layer_ids],
            "selected_kv_group_ids_by_layer": {
                str(int(layer_id)): [int(value) for value in kv_group_ids]
                for layer_id, kv_group_ids in self.selected_kv_group_ids_by_layer.items()
            },
            "selected_query_head_ids_by_layer": {
                str(int(layer_id)): [int(value) for value in query_head_ids]
                for layer_id, query_head_ids in self.selected_query_head_ids_by_layer.items()
            },
            "layer_profiles": [profile.to_dict() for profile in self.layer_profiles],
        }


def locate_source_token_indices(
    tokenizer,
    *,
    text: str,
    source_span_text: Optional[str] = None,
    source_token_indices: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    if source_token_indices:
        return tuple(int(index) for index in source_token_indices)
    if not source_span_text:
        raise ValueError("Provide either source_token_indices or source_span_text for MoE profiling.")

    source_start = str(text).find(str(source_span_text))
    if source_start < 0:
        raise ValueError(f"Could not find source span {source_span_text!r} in the provided text.")
    source_end = source_start + len(str(source_span_text))

    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = tokenized.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("Tokenizer did not provide offset_mapping; cannot locate source token indices.")

    positions = []
    for idx, offset in enumerate(offsets):
        if offset is None or len(offset) != 2:
            continue
        start, end = int(offset[0]), int(offset[1])
        if end <= source_start or start >= source_end:
            continue
        if end > start:
            positions.append(idx)
    if not positions:
        raise RuntimeError(f"No tokenizer positions overlapped source span {source_span_text!r}.")
    return tuple(positions)


def _normalize_positive(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    min_value = min(float(value) for value in values)
    max_value = max(float(value) for value in values)
    if max_value <= 0.0:
        return [0.0 for _ in values]
    if math.isclose(min_value, max_value):
        return [1.0 for _ in values]
    scale = max_value - min_value
    return [(float(value) - min_value) / scale for value in values]


def _mean_tensor_norm(hidden_states: torch.Tensor) -> float:
    if hidden_states.numel() == 0:
        return 0.0
    flat = hidden_states.to(dtype=torch.float32)
    return float(flat.norm(dim=-1).mean().item())


def _looks_like_probability_tensor(router_tensor: torch.Tensor) -> bool:
    if router_tensor.ndim != 3 or router_tensor.numel() == 0:
        return False
    if torch.any(router_tensor < 0):
        return False
    row_sums = router_tensor.to(dtype=torch.float32).sum(dim=-1)
    return bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4))


def _routing_concentration_from_topk(weights: torch.Tensor, *, mode: str) -> float:
    normalized_mode = str(mode).strip().lower()
    weights = weights.to(dtype=torch.float32)
    if weights.numel() == 0:
        return 0.0
    if normalized_mode == "squared_l2":
        return float((weights.pow(2).sum(dim=-1)).mean().item())
    if normalized_mode == "normalized_entropy":
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1)
        max_entropy = math.log(max(int(weights.shape[-1]), 1))
        if max_entropy <= 0.0:
            return 1.0
        return float((1.0 - (entropy / max_entropy)).mean().item())
    raise ValueError(f"Unsupported routing_concentration_mode '{mode}'.")


def _compute_head_attention_scores(
    attention_tensor: torch.Tensor,
    *,
    source_token_indices: Sequence[int],
    future_only: bool,
) -> torch.Tensor:
    if attention_tensor.ndim != 3:
        raise ValueError(f"Expected attention tensor with shape [heads, query, key], got {tuple(attention_tensor.shape)}.")
    if not source_token_indices:
        raise ValueError("source_token_indices must be non-empty.")

    attention_tensor = attention_tensor.to(dtype=torch.float32)
    source_index_tensor = torch.tensor(
        [int(index) for index in source_token_indices],
        device=attention_tensor.device,
        dtype=torch.long,
    )
    if future_only:
        future_start = max(int(index) for index in source_token_indices) + 1
        if future_start < attention_tensor.shape[1]:
            sliced = attention_tensor[:, future_start:, :]
        else:
            sliced = attention_tensor
    else:
        sliced = attention_tensor
    source_mass = sliced.index_select(dim=-1, index=source_index_tensor)
    return source_mass.mean(dim=(-1, -2))


def _reshape_router_tensor(
    router_tensor: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    if router_tensor.ndim == 3:
        return router_tensor
    if router_tensor.ndim == 2 and router_tensor.shape[0] == batch_size * sequence_length:
        return router_tensor.view(batch_size, sequence_length, -1)
    raise ValueError(
        "Router tensor had an unexpected shape. "
        f"Expected [batch, seq, width] or [batch * seq, width], got {tuple(router_tensor.shape)}."
    )


def _extract_router_signature(
    router_logits: Optional[torch.Tensor],
    *,
    router_probs: Optional[torch.Tensor],
    router_scores: Optional[torch.Tensor],
    router_indices: Optional[torch.Tensor],
    num_experts_per_tok: int,
    norm_topk_prob: bool,
    routing_concentration_mode: str,
) -> Tuple[Tuple[float, ...], Tuple[int, ...], Tuple[float, ...], float]:
    if router_probs is None and router_logits is None:
        return tuple(), tuple(), tuple(), 0.0

    if router_probs is None:
        router_probs = router_logits.to(dtype=torch.float32)
        if router_probs.ndim != 3:
            raise ValueError(
                f"Expected router logits with shape [batch, seq, experts], got {tuple(router_probs.shape)}."
            )
        router_probs = torch.softmax(router_probs, dim=-1)
    else:
        router_probs = router_probs.to(dtype=torch.float32)
        if router_probs.ndim != 3:
            raise ValueError(
                f"Expected router probabilities with shape [batch, seq, experts], got {tuple(router_probs.shape)}."
            )

    mean_router_probs = router_probs.mean(dim=1)[0]

    if router_scores is not None and router_indices is not None:
        router_scores = router_scores.to(dtype=torch.float32)
        router_indices = router_indices.to(dtype=torch.long)
        if router_scores.ndim != 3 or router_indices.ndim != 3:
            raise ValueError(
                "Expected router_scores and router_indices with shape [batch, seq, top_k]. "
                f"Got {tuple(router_scores.shape)} and {tuple(router_indices.shape)}."
            )
        expert_mass = torch.zeros_like(router_probs)
        expert_mass.scatter_add_(dim=-1, index=router_indices, src=router_scores)
        mean_selected_mass = expert_mass.mean(dim=1)[0]
        top_weights, top_experts = torch.topk(
            mean_selected_mass,
            k=min(int(num_experts_per_tok), int(mean_selected_mass.shape[-1])),
        )
        concentration = _routing_concentration_from_topk(router_scores[0], mode=routing_concentration_mode)
    else:
        top_weights, top_experts = torch.topk(
            mean_router_probs,
            k=min(int(num_experts_per_tok), int(mean_router_probs.shape[-1])),
        )
        if norm_topk_prob:
            top_weights = top_weights / top_weights.sum().clamp_min(1e-12)
        concentration = _routing_concentration_from_topk(top_weights.unsqueeze(0), mode=routing_concentration_mode)

    return (
        tuple(float(value) for value in mean_router_probs.tolist()),
        tuple(int(value) for value in top_experts.tolist()),
        tuple(float(value) for value in top_weights.tolist()),
        concentration,
    )


def _register_moe_hooks(
    model,
    *,
    layer_ids: Sequence[int],
    source_token_indices: Sequence[int],
    batch_size: int,
    sequence_length: int,
) -> Tuple[List[object], Dict[int, Dict[str, Optional[torch.Tensor]]]]:
    model_layers = getattr(getattr(model, "model", model), "layers", None)
    if model_layers is None:
        raise AttributeError("Expected model.model.layers to exist.")

    records: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}
    handles = []

    for layer_id in layer_ids:
        layer_module = model_layers[int(layer_id)]
        moe_module = getattr(layer_module, "mlp", None)
        if moe_module is None:
            continue

        def _selected_router_tensor(
            tensor: torch.Tensor,
            *,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            reshaped = _reshape_router_tensor(
                tensor.detach().to(dtype=dtype),
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            index_tensor = torch.tensor(
                [int(index) for index in source_token_indices],
                device=reshaped.device,
                dtype=torch.long,
            )
            return reshaped.index_select(dim=1, index=index_tensor).cpu()

        def _capture_router_outputs(record: Dict[str, Optional[torch.Tensor]], outputs) -> None:
            if isinstance(outputs, torch.Tensor):
                selected_router = _selected_router_tensor(outputs, dtype=torch.float32)
                key = "router_probs" if _looks_like_probability_tensor(selected_router) else "router_logits"
                record.setdefault(key, selected_router)
                return
            if not isinstance(outputs, (tuple, list)) or not outputs:
                return

            if (
                len(outputs) >= 3
                and isinstance(outputs[0], torch.Tensor)
                and isinstance(outputs[1], torch.Tensor)
                and isinstance(outputs[2], torch.Tensor)
            ):
                selected_router = _selected_router_tensor(outputs[0], dtype=torch.float32)
                key = "router_probs" if _looks_like_probability_tensor(selected_router) else "router_logits"
                record.setdefault(key, selected_router)
                record.setdefault("router_scores", _selected_router_tensor(outputs[1], dtype=torch.float32))
                record.setdefault("router_indices", _selected_router_tensor(outputs[2], dtype=torch.long))
                return

            first_output = outputs[0]
            if isinstance(first_output, torch.Tensor):
                selected_router = _selected_router_tensor(first_output, dtype=torch.float32)
                key = "router_probs" if _looks_like_probability_tensor(selected_router) else "router_logits"
                record.setdefault(key, selected_router)

        def make_hook(current_layer_id: int):
            def hook(_module, _inputs, outputs):
                if isinstance(outputs, tuple):
                    moe_hidden_states = outputs[0]
                    router_payload = outputs[1] if len(outputs) > 1 else None
                else:
                    moe_hidden_states, router_payload = outputs, None
                index_tensor = torch.tensor(
                    [int(index) for index in source_token_indices],
                    device=moe_hidden_states.device,
                    dtype=torch.long,
                )
                selected_hidden = moe_hidden_states.index_select(dim=1, index=index_tensor).detach().cpu()
                record = records.setdefault(int(current_layer_id), {})
                record["moe_hidden_states"] = selected_hidden
                if router_payload is not None:
                    _capture_router_outputs(record, router_payload)

            return hook

        handles.append(moe_module.register_forward_hook(make_hook(int(layer_id))))

        gate_module = getattr(moe_module, "gate", None)
        if gate_module is None:
            continue

        def make_gate_hook(current_layer_id: int):
            def hook(_module, _inputs, outputs):
                record = records.setdefault(int(current_layer_id), {})
                _capture_router_outputs(record, outputs)

            return hook

        handles.append(gate_module.register_forward_hook(make_gate_hook(int(layer_id))))
    return handles, records


def _attention_tensor_for_layer(attentions, *, layer_id: int) -> torch.Tensor:
    if int(layer_id) >= len(attentions):
        raise IndexError(f"Layer {layer_id} is out of range for the returned attentions tuple of length {len(attentions)}.")
    layer_attention = attentions[int(layer_id)]
    if layer_attention is None:
        raise RuntimeError(
            "Attention weights were not materialized for layer "
            f"{layer_id}. Load the profiling model with attn_implementation='eager'."
        )
    if not isinstance(layer_attention, torch.Tensor):
        raise TypeError(
            f"Expected attentions[{layer_id}] to be a tensor, got {type(layer_attention).__name__}."
        )
    if layer_attention.ndim != 4:
        raise ValueError(
            f"Expected attentions[{layer_id}] to have shape [batch, heads, query, key], got {tuple(layer_attention.shape)}."
        )
    return layer_attention[0].detach().cpu()


def select_layers_and_kv_groups(
    layer_profiles: Sequence[LayerMoEProfile],
    *,
    top_layers: int,
    top_kv_groups: int,
    num_query_heads_per_kv_head: int,
) -> Tuple[Tuple[int, ...], Dict[int, Tuple[int, ...]], Dict[int, Tuple[int, ...]]]:
    ordered = sorted(layer_profiles, key=lambda profile: float(profile.layer_score), reverse=True)
    selected_layers = tuple(int(profile.layer_id) for profile in ordered[: max(1, int(top_layers))])
    kv_group_ids_by_layer: Dict[int, Tuple[int, ...]] = {}
    query_head_ids_by_layer: Dict[int, Tuple[int, ...]] = {}
    for profile in ordered[: max(1, int(top_layers))]:
        kv_scores = list(enumerate(profile.kv_group_scores))
        kv_scores.sort(key=lambda pair: float(pair[1]), reverse=True)
        selected_kv_groups = tuple(int(group_id) for group_id, _ in kv_scores[: max(1, int(top_kv_groups))])
        kv_group_ids_by_layer[int(profile.layer_id)] = selected_kv_groups
        query_head_ids_by_layer[int(profile.layer_id)] = expand_kv_groups_to_query_heads(
            selected_kv_groups,
            num_query_heads_per_kv_head,
        )
    return selected_layers, kv_group_ids_by_layer, query_head_ids_by_layer


def profile_source_concept(
    model,
    tokenizer,
    *,
    text: str,
    source_span_text: Optional[str] = None,
    source_token_indices: Optional[Sequence[int]] = None,
    layer_ids: Optional[Sequence[int]] = None,
    top_layers: int = 4,
    top_kv_groups: int = 2,
    kv_group_reduce: str = "sum",
    future_only_attention: bool = True,
    routing_concentration_mode: str = "squared_l2",
) -> Qwen3MoeProfileArtifact:
    model_layers = getattr(getattr(model, "model", model), "layers", None)
    if model_layers is None:
        raise AttributeError("Expected model.model.layers to exist.")
    source_token_indices = locate_source_token_indices(
        tokenizer,
        text=text,
        source_span_text=source_span_text,
        source_token_indices=source_token_indices,
    )

    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    candidate_layer_ids = tuple(int(layer_id) for layer_id in (layer_ids or tuple(range(len(model_layers)))))
    if not candidate_layer_ids:
        raise ValueError("layer_ids must be non-empty.")

    attention_layout = resolve_attention_layout(model_layers[int(candidate_layer_ids[0])].self_attn)
    handles, moe_records = _register_moe_hooks(
        model,
        layer_ids=candidate_layer_ids,
        source_token_indices=source_token_indices,
        batch_size=int(input_ids.shape[0]),
        sequence_length=int(input_ids.shape[1]),
    )
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                output_router_logits=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
        raise RuntimeError("Qwen3 profiling requires output_attentions=True support from the model.")

    config = getattr(model, "config", None)
    num_experts_per_tok = int(getattr(config, "num_experts_per_tok", 0) or 0)
    norm_topk_prob = bool(getattr(config, "norm_topk_prob", False))

    raw_profiles: List[Dict[str, object]] = []
    for layer_id in candidate_layer_ids:
        layer_attention = _attention_tensor_for_layer(attentions, layer_id=int(layer_id))
        head_scores_tensor = _compute_head_attention_scores(
            layer_attention,
            source_token_indices=source_token_indices,
            future_only=future_only_attention,
        )
        kv_group_scores_tensor = aggregate_query_head_scores_to_kv_groups(
            head_scores_tensor,
            num_key_value_heads=attention_layout.num_key_value_heads,
            num_query_heads_per_kv_head=attention_layout.num_query_heads_per_kv_head,
            reduce=kv_group_reduce,
        )

        layer_record = moe_records.get(int(layer_id), {})
        moe_hidden_states = layer_record.get("moe_hidden_states")
        router_logits = layer_record.get("router_logits")
        router_probs = layer_record.get("router_probs")
        router_scores = layer_record.get("router_scores")
        router_indices = layer_record.get("router_indices")
        moe_output_norm = _mean_tensor_norm(moe_hidden_states) if moe_hidden_states is not None else 0.0
        router_probs, top_experts, top_weights, routing_concentration = _extract_router_signature(
            router_logits,
            router_probs=router_probs,
            router_scores=router_scores,
            router_indices=router_indices,
            num_experts_per_tok=max(1, num_experts_per_tok),
            norm_topk_prob=norm_topk_prob,
            routing_concentration_mode=routing_concentration_mode,
        )
        raw_profiles.append(
            {
                "layer_id": int(layer_id),
                "head_scores": tuple(float(value) for value in head_scores_tensor.tolist()),
                "kv_group_scores": tuple(float(value) for value in kv_group_scores_tensor.tolist()),
                "router_probs": tuple(float(value) for value in router_probs),
                "top_experts": tuple(int(value) for value in top_experts),
                "top_expert_weights": tuple(float(value) for value in top_weights),
                "routing_concentration": float(routing_concentration),
                "moe_output_norm": float(moe_output_norm),
                "g_max": float(kv_group_scores_tensor.max().item()),
            }
        )

    normalized_attention = _normalize_positive([float(profile["g_max"]) for profile in raw_profiles])
    normalized_routing = _normalize_positive([float(profile["routing_concentration"]) for profile in raw_profiles])
    normalized_moe = _normalize_positive([float(profile["moe_output_norm"]) for profile in raw_profiles])

    layer_profiles: List[LayerMoEProfile] = []
    for idx, raw_profile in enumerate(raw_profiles):
        layer_score = float(normalized_attention[idx] * normalized_routing[idx] * normalized_moe[idx])
        layer_profiles.append(
            LayerMoEProfile(
                layer_id=int(raw_profile["layer_id"]),
                head_scores=tuple(raw_profile["head_scores"]),
                kv_group_scores=tuple(raw_profile["kv_group_scores"]),
                router_probs=tuple(raw_profile["router_probs"]),
                top_experts=tuple(raw_profile["top_experts"]),
                top_expert_weights=tuple(raw_profile["top_expert_weights"]),
                routing_concentration=float(raw_profile["routing_concentration"]),
                moe_output_norm=float(raw_profile["moe_output_norm"]),
                layer_score=layer_score,
            )
        )

    selected_layer_ids, kv_group_ids_by_layer, query_head_ids_by_layer = select_layers_and_kv_groups(
        layer_profiles,
        top_layers=top_layers,
        top_kv_groups=top_kv_groups,
        num_query_heads_per_kv_head=attention_layout.num_query_heads_per_kv_head,
    )
    model_name = str(getattr(getattr(model, "config", None), "_name_or_path", "qwen3_moe"))
    return Qwen3MoeProfileArtifact(
        model_name=model_name,
        source_text=str(text),
        source_span_text=None if source_span_text is None else str(source_span_text),
        source_token_indices=tuple(int(index) for index in source_token_indices),
        selected_layer_ids=selected_layer_ids,
        selected_kv_group_ids_by_layer=kv_group_ids_by_layer,
        selected_query_head_ids_by_layer=query_head_ids_by_layer,
        layer_profiles=tuple(layer_profiles),
    )
