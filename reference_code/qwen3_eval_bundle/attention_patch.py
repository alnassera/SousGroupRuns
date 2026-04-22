from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from .memory import DescriptorMemory
except ImportError:  # pragma: no cover - supports flat-file layout
    from memory import DescriptorMemory


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, n_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, n_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch_size, n_kv_heads * n_rep, seq_len, head_dim)


def _resolve_attention_dims(module: torch.nn.Module) -> tuple[int, int, int, int]:
    config = getattr(module, "config", None)

    num_heads = getattr(module, "num_heads", None)
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_attention_heads", None)

    head_dim = getattr(module, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(module, "q_proj", None), "out_features", None)
        if hidden_size is not None and num_heads is not None:
            head_dim = hidden_size // num_heads

    if num_heads is None:
        q_out = getattr(getattr(module, "q_proj", None), "out_features", None)
        if q_out is not None and head_dim is not None:
            num_heads = q_out // head_dim

    num_kv_heads = getattr(module, "num_key_value_heads", None)
    if num_kv_heads is None and config is not None:
        num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        k_out = getattr(getattr(module, "k_proj", None), "out_features", None)
        if k_out is not None and head_dim is not None:
            num_kv_heads = k_out // head_dim

    num_kv_groups = getattr(module, "num_key_value_groups", None)
    if num_kv_groups is None and num_heads is not None and num_kv_heads is not None:
        num_kv_groups = num_heads // max(num_kv_heads, 1)

    if num_heads is None or head_dim is None or num_kv_heads is None or num_kv_groups is None:
        raise AttributeError(
            "Could not infer attention dimensions from the module. "
            "Expected cached attrs or compatible config/projection shapes."
        )

    return int(num_heads), int(num_kv_heads), int(num_kv_groups), int(head_dim)


def _apply_attention_mask(attn_scores: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return attn_scores
    if attention_mask.ndim == 4:
        return attn_scores + attention_mask[:, :, :, : attn_scores.shape[-1]]
    if attention_mask.ndim == 2:
        mask = attention_mask[:, None, None, : attn_scores.shape[-1]].to(
            dtype=attn_scores.dtype,
            device=attn_scores.device,
        )
        if mask.max() <= 1.0 and mask.min() >= 0.0:
            additive = (1.0 - mask) * torch.finfo(attn_scores.dtype).min
            return attn_scores + additive
        return attn_scores + mask
    raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")


def _slice_attention_mask(attention_mask: Optional[torch.Tensor], query_slice: slice) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 4:
        return attention_mask[:, :, query_slice, :]
    return attention_mask


def _apply_rotary_if_available(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    rotary_source: Optional[torch.nn.Module],
    position_ids: Optional[torch.Tensor],
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except Exception as exc:  # pragma: no cover - import failure depends on local env
        raise RuntimeError("transformers is required to patch Llama attention.") from exc

    effective_rotary_source = rotary_source if rotary_source is not None else getattr(module, "rotary_emb", None)
    if effective_rotary_source is not None and position_ids is not None:
        try:
            cos, sin = effective_rotary_source(key_states, position_ids)
        except TypeError:
            cos, sin = effective_rotary_source(key_states, seq_len=position_ids.max().item() + 1)
        try:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        except TypeError:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, {"sin": sin, "cos": cos}

    if position_embeddings is not None:
        cos, sin = position_embeddings
        try:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        except TypeError:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)
        return query_states, key_states, {"sin": sin, "cos": cos}

    return query_states, key_states, {}


def _resolve_position_ids(
    hidden_states: torch.Tensor,
    *,
    position_ids: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    past_key_value: object,
    prefix_offset: int,
) -> Optional[torch.Tensor]:
    if position_ids is not None:
        return position_ids + int(prefix_offset)

    if cache_position is not None:
        if cache_position.ndim == 0:
            cache_position = cache_position.unsqueeze(0)
        return cache_position.unsqueeze(0).expand(hidden_states.shape[0], -1) + int(prefix_offset)

    past_len = 0
    if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
        past_len = int(past_key_value[0].shape[-2])
    seq_len = hidden_states.shape[1]
    base = torch.arange(past_len, past_len + seq_len, device=hidden_states.device).unsqueeze(0)
    return base.expand(hidden_states.shape[0], -1) + int(prefix_offset)


def _normalize_head_ids(head_ids: Optional[Sequence[int]], num_heads: int) -> Tuple[int, ...]:
    if not head_ids:
        return tuple(range(num_heads))

    requested = [int(head_id) for head_id in head_ids]
    if requested and max(requested) >= num_heads and min(requested) >= 1 and max(requested) <= num_heads:
        requested = [head_id - 1 for head_id in requested]

    normalized: List[int] = []
    seen = set()
    for head_id in requested:
        if head_id < 0 or head_id >= num_heads:
            raise IndexError(f"Head id {head_id} is out of range for num_heads={num_heads}.")
        if head_id not in seen:
            seen.add(head_id)
            normalized.append(head_id)
    return tuple(normalized)


def _expand_memory_state(
    layer_memory,
    *,
    batch_size: int,
    device: torch.device,
    key_dtype: torch.dtype,
    value_dtype: torch.dtype,
    num_kv_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    memory_keys = layer_memory.key.to(device=device, dtype=key_dtype)
    memory_values = layer_memory.value.to(device=device, dtype=value_dtype)
    if memory_keys.ndim == 3:
        memory_keys = memory_keys.unsqueeze(0)
    if memory_values.ndim == 3:
        memory_values = memory_values.unsqueeze(0)
    memory_keys = memory_keys.expand(batch_size, -1, -1, -1)
    memory_values = memory_values.expand(batch_size, -1, -1, -1)
    return _repeat_kv(memory_keys, num_kv_groups), _repeat_kv(memory_values, num_kv_groups)


def _memory_log_prior(token_count: int) -> float:
    return -math.log(max(int(token_count), 1))


@dataclass(frozen=True)
class DescriptorMixConfig:
    rho: float = 0.1
    mix_mode: str = "contrastive_opposite"
    prefill_steering: str = "last_token_only"
    steering_operator: str = "key_logit_sparse"
    positive_gain: float = 1.2
    negative_gain: float = 0.6
    query_adaptive_gates: bool = True
    query_gate_scale: float = 1.0
    head_ids: Tuple[int, ...] = ()


@dataclass
class PatchedAttentionContext:
    patches: List[Tuple[torch.nn.Module, object]] = field(default_factory=list)

    def __enter__(self) -> "PatchedAttentionContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def remove(self) -> None:
        for module, original_forward in reversed(self.patches):
            module.forward = original_forward
        self.patches.clear()


class DescriptorMemoryAttentionPatch:
    """Patch Llama attention with either legacy output mixing or sparse logit-side steering."""

    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        memory: DescriptorMemory,
        reference_memory: Optional[DescriptorMemory],
        config: DescriptorMixConfig,
    ) -> None:
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        self.memory = memory
        self.reference_memory = reference_memory
        rho = float(config.rho)
        mix_mode = str(config.mix_mode).strip().lower()
        prefill_steering = str(config.prefill_steering).strip().lower()
        steering_operator = str(config.steering_operator).strip().lower()
        if mix_mode not in {"convex", "contrastive_neutral", "contrastive_opposite"}:
            raise ValueError(f"Unsupported mix_mode '{config.mix_mode}'.")
        if prefill_steering not in {"full", "last_token_only", "none"}:
            raise ValueError(f"Unsupported prefill_steering '{config.prefill_steering}'.")
        if steering_operator not in {"output_mixture", "key_logit_sparse"}:
            raise ValueError(f"Unsupported steering_operator '{config.steering_operator}'.")
        if mix_mode != "convex" and reference_memory is None:
            raise ValueError(f"mix_mode='{mix_mode}' requires a reference_memory.")
        if steering_operator == "key_logit_sparse" and mix_mode == "convex":
            raise ValueError("steering_operator='key_logit_sparse' requires a contrastive mix_mode.")
        if mix_mode == "convex":
            rho = max(0.0, min(1.0, rho))
        else:
            rho = max(0.0, rho)
        self.config = DescriptorMixConfig(
            rho=rho,
            mix_mode=mix_mode,
            prefill_steering=prefill_steering,
            steering_operator=steering_operator,
            positive_gain=max(0.0, float(config.positive_gain)),
            negative_gain=max(0.0, float(config.negative_gain)),
            query_adaptive_gates=bool(config.query_adaptive_gates),
            query_gate_scale=float(config.query_gate_scale),
            head_ids=tuple(int(head_id) for head_id in config.head_ids),
        )

    def patch_model(self, model: torch.nn.Module) -> PatchedAttentionContext:
        context = PatchedAttentionContext()
        model_body = getattr(model, "model", model)
        model_layers = getattr(model_body, "layers", None)
        if model_layers is None:
            raise AttributeError("Expected model.model.layers to exist.")
        model_rotary_emb = getattr(model_body, "rotary_emb", None)

        for layer_id in self.layer_ids:
            layer_module = model_layers[int(layer_id)]
            attn_module = getattr(layer_module, "self_attn", None)
            if attn_module is None:
                raise AttributeError(f"Layer {layer_id} has no self_attn module.")
            original_forward = attn_module.forward
            rotary_source = getattr(attn_module, "rotary_emb", None) or model_rotary_emb
            attn_module.forward = self._make_wrapped_forward(int(layer_id), attn_module, original_forward, rotary_source)
            context.patches.append((attn_module, original_forward))
        return context

    def _make_wrapped_forward(
        self,
        layer_id: int,
        module: torch.nn.Module,
        original_forward: object,
        rotary_source: Optional[torch.nn.Module],
    ):
        signature = inspect.signature(original_forward)

        def wrapped_forward(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            hidden_states = bound.arguments.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return original_forward(*args, **kwargs)

            attention_mask = bound.arguments.get("attention_mask")
            position_ids = bound.arguments.get("position_ids")
            position_embeddings = bound.arguments.get("position_embeddings")
            past_key_value = bound.arguments.get("past_key_value")
            cache_position = bound.arguments.get("cache_position")
            output_attentions = bool(bound.arguments.get("output_attentions", kwargs.get("output_attentions", False)))

            if not all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")):
                raise RuntimeError("Unsupported attention module for descriptor-memory steering.")

            layer_memory = self.memory.layer_memories.get(layer_id)
            if layer_memory is None:
                return original_forward(*args, **kwargs)
            reference_layer_memory = None
            if self.reference_memory is not None:
                reference_layer_memory = self.reference_memory.layer_memories.get(layer_id)

            batch_size, query_len, _ = hidden_states.shape
            if query_len > 1 and self.config.prefill_steering == "none":
                return original_forward(*args, **kwargs)

            num_heads, num_kv_heads, num_kv_groups, head_dim = _resolve_attention_dims(module)
            selected_head_ids = _normalize_head_ids(self.config.head_ids, num_heads)
            scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(head_dim)))

            query_states = module.q_proj(hidden_states).view(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
            key_states = module.k_proj(hidden_states).view(batch_size, query_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = module.v_proj(hidden_states).view(batch_size, query_len, num_kv_heads, head_dim).transpose(1, 2)

            resolved_position_ids = _resolve_position_ids(
                hidden_states,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_value=past_key_value,
                prefix_offset=self.memory.token_count,
            )
            query_states, key_states, rope_cache_kwargs = _apply_rotary_if_available(
                module,
                query_states,
                key_states,
                rotary_source=rotary_source,
                position_ids=resolved_position_ids,
                position_embeddings=position_embeddings,
            )

            if past_key_value is not None:
                if hasattr(past_key_value, "update"):
                    cache_kwargs = dict(rope_cache_kwargs)
                    cache_kwargs["cache_position"] = cache_position
                    key_states, value_states = past_key_value.update(key_states, value_states, layer_id, cache_kwargs)
                elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)

            prompt_key_states = _repeat_kv(key_states, num_kv_groups)
            prompt_value_states = _repeat_kv(value_states, num_kv_groups)
            prompt_scores = torch.matmul(query_states, prompt_key_states.transpose(-1, -2)) * scaling
            prompt_scores = _apply_attention_mask(prompt_scores, attention_mask)
            prompt_probs = F.softmax(prompt_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            prompt_output = torch.matmul(prompt_probs, prompt_value_states)

            if query_len == 1 or self.config.prefill_steering == "full":
                query_slice = slice(None)
            else:
                query_slice = slice(query_len - 1, query_len)

            selected_query_states = query_states[:, :, query_slice, :]
            selected_attention_mask = _slice_attention_mask(attention_mask, query_slice)
            selected_prompt_scores = torch.matmul(selected_query_states, prompt_key_states.transpose(-1, -2)) * scaling
            selected_prompt_scores = _apply_attention_mask(selected_prompt_scores, selected_attention_mask)
            selected_prompt_probs = F.softmax(selected_prompt_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            selected_prompt_output = torch.matmul(selected_prompt_probs, prompt_value_states)

            trait_keys_rep, trait_values_rep = _expand_memory_state(
                layer_memory,
                batch_size=batch_size,
                device=key_states.device,
                key_dtype=key_states.dtype,
                value_dtype=value_states.dtype,
                num_kv_groups=num_kv_groups,
            )

            if self.config.steering_operator == "output_mixture":
                def compute_memory_output(memory_keys_rep: torch.Tensor, memory_values_rep: torch.Tensor) -> torch.Tensor:
                    memory_scores = torch.matmul(selected_query_states, memory_keys_rep.transpose(-1, -2)) * scaling
                    memory_probs = F.softmax(memory_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    return torch.matmul(memory_probs, memory_values_rep)

                trait_output = compute_memory_output(trait_keys_rep, trait_values_rep)
                steered_selected_output = selected_prompt_output.clone()
                selected_index = torch.tensor(selected_head_ids, device=selected_prompt_output.device, dtype=torch.long)
                if self.config.mix_mode == "convex":
                    steered_selected_heads = (
                        (1.0 - self.config.rho) * selected_prompt_output.index_select(1, selected_index)
                        + self.config.rho * trait_output.index_select(1, selected_index)
                    )
                else:
                    if reference_layer_memory is None:
                        raise KeyError(
                            f"mix_mode='{self.config.mix_mode}' requires a reference layer memory for layer {layer_id}."
                        )
                    reference_keys_rep, reference_values_rep = _expand_memory_state(
                        reference_layer_memory,
                        batch_size=batch_size,
                        device=key_states.device,
                        key_dtype=key_states.dtype,
                        value_dtype=value_states.dtype,
                        num_kv_groups=num_kv_groups,
                    )
                    reference_output = compute_memory_output(reference_keys_rep, reference_values_rep)
                    steered_selected_heads = (
                        selected_prompt_output.index_select(1, selected_index)
                        + self.config.rho
                        * (
                            trait_output.index_select(1, selected_index)
                            - reference_output.index_select(1, selected_index)
                        )
                    )
                steered_selected_output[:, selected_index, :, :] = steered_selected_heads
            else:
                if reference_layer_memory is None:
                    raise KeyError(
                        f"steering_operator='{self.config.steering_operator}' requires a reference layer memory for layer {layer_id}."
                    )
                reference_keys_rep, reference_values_rep = _expand_memory_state(
                    reference_layer_memory,
                    batch_size=batch_size,
                    device=key_states.device,
                    key_dtype=key_states.dtype,
                    value_dtype=value_states.dtype,
                    num_kv_groups=num_kv_groups,
                )
                selected_index = torch.tensor(selected_head_ids, device=query_states.device, dtype=torch.long)
                selected_query_heads = selected_query_states.index_select(1, selected_index)
                selected_prompt_scores_heads = selected_prompt_scores.index_select(1, selected_index)
                selected_prompt_values_heads = prompt_value_states.index_select(1, selected_index)
                trait_keys_heads = trait_keys_rep.index_select(1, selected_index)
                trait_values_heads = trait_values_rep.index_select(1, selected_index)
                reference_keys_heads = reference_keys_rep.index_select(1, selected_index)
                reference_values_heads = reference_values_rep.index_select(1, selected_index)

                trait_scores = torch.matmul(selected_query_heads, trait_keys_heads.transpose(-1, -2)) * scaling
                reference_scores = torch.matmul(selected_query_heads, reference_keys_heads.transpose(-1, -2)) * scaling

                if self.config.query_adaptive_gates:
                    trait_alignment = trait_scores.max(dim=-1).values
                    reference_alignment = reference_scores.max(dim=-1).values
                    gate_logits = self.config.query_gate_scale * (trait_alignment - reference_alignment)
                    positive_gate = torch.sigmoid(gate_logits)
                    negative_gate = torch.sigmoid(-gate_logits)
                else:
                    positive_gate = torch.ones_like(trait_scores.max(dim=-1).values)
                    negative_gate = torch.ones_like(reference_scores.max(dim=-1).values)

                trait_scores = trait_scores + (
                    _memory_log_prior(self.memory.token_count)
                    + self.config.rho * self.config.positive_gain * positive_gate.unsqueeze(-1)
                )
                reference_scores = reference_scores + (
                    _memory_log_prior(self.reference_memory.token_count if self.reference_memory is not None else 1)
                    - self.config.rho * self.config.negative_gain * negative_gate.unsqueeze(-1)
                )

                combined_scores = torch.cat(
                    [selected_prompt_scores_heads, trait_scores, reference_scores],
                    dim=-1,
                )
                combined_probs = F.softmax(combined_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
                combined_values = torch.cat(
                    [selected_prompt_values_heads, trait_values_heads, reference_values_heads],
                    dim=2,
                )
                steered_selected_heads = torch.matmul(combined_probs, combined_values)
                steered_selected_output = selected_prompt_output.clone()
                steered_selected_output[:, selected_index, :, :] = steered_selected_heads

            if query_len == 1 or self.config.prefill_steering == "full":
                mixed_output = steered_selected_output
            else:
                mixed_output = prompt_output.clone()
                mixed_output[:, :, query_slice, :] = steered_selected_output

            attn_output = mixed_output.transpose(1, 2).contiguous().reshape(batch_size, query_len, num_heads * head_dim)
            attn_output = module.o_proj(attn_output)

            attn_weights = prompt_probs if output_attentions else None
            return attn_output, attn_weights

        return wrapped_forward
