from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from descriptor_memory_persona.mlp_conditioning import AttentionToMLPControlState
    from descriptor_memory_persona.attention_patch import (
        _apply_attention_mask,
        _expand_memory_state,
        _memory_log_prior,
        _repeat_kv,
        _resolve_position_ids,
        _slice_attention_mask,
    )
    from .canonical_memory import CanonicalDescriptorMemory
except ImportError:  # pragma: no cover
    try:
        from hybrid_caa_canonical_debug.mlp_conditioning import AttentionToMLPControlState
        from hybrid_caa_canonical_debug.attention_patch import (
            _apply_attention_mask,
            _expand_memory_state,
            _memory_log_prior,
            _repeat_kv,
            _resolve_position_ids,
            _slice_attention_mask,
        )
        from canonical_memory import CanonicalDescriptorMemory
    except ImportError:  # pragma: no cover - supports flat-file layout
        from mlp_conditioning import AttentionToMLPControlState
        from attention_patch import (
            _apply_attention_mask,
            _expand_memory_state,
            _memory_log_prior,
            _repeat_kv,
            _resolve_position_ids,
            _slice_attention_mask,
        )
        from canonical_memory import CanonicalDescriptorMemory

from architecture import (
    apply_qk_norm,
    apply_qwen3_rotary,
    normalize_head_ids,
    resolve_attention_layout,
    resolve_selected_query_heads,
)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _slice_rope_cache_tensor(tensor: torch.Tensor, query_slice: slice) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor[query_slice, :]
    if tensor.ndim == 3:
        return tensor[:, query_slice, :]
    if tensor.ndim == 4:
        return tensor[:, :, query_slice, :]
    raise ValueError(f"Unsupported RoPE cache tensor shape: {tuple(tensor.shape)}")


def _inverse_rotate_query(query_states: torch.Tensor, *, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos_states = cos.to(device=query_states.device, dtype=query_states.dtype)
    sin_states = sin.to(device=query_states.device, dtype=query_states.dtype)
    while cos_states.ndim < query_states.ndim:
        cos_states = cos_states.unsqueeze(1)
        sin_states = sin_states.unsqueeze(1)
    return (query_states * cos_states) - (_rotate_half(query_states) * sin_states)


def _memory_slot_count(memory: CanonicalDescriptorMemory) -> int:
    slot_count = int(getattr(memory, "slot_count", 0) or 0)
    return slot_count if slot_count > 0 else int(memory.token_count)


def _read_layer_cache_kv(past_key_value, layer_id: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if past_key_value is None:
        return None, None
    if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
        return past_key_value[0], past_key_value[1]

    key_cache = getattr(past_key_value, "key_cache", None)
    value_cache = getattr(past_key_value, "value_cache", None)
    if key_cache is not None and value_cache is not None and int(layer_id) < len(key_cache):
        return key_cache[int(layer_id)], value_cache[int(layer_id)]

    layers = getattr(past_key_value, "layers", None)
    if layers is not None and int(layer_id) < len(layers):
        layer_cache = layers[int(layer_id)]
        for key_name, value_name in (
            ("keys", "values"),
            ("key_states", "value_states"),
            ("key_cache", "value_cache"),
        ):
            key_states = getattr(layer_cache, key_name, None)
            value_states = getattr(layer_cache, value_name, None)
            if key_states is not None and value_states is not None:
                return key_states, value_states
    return None, None


@dataclass(frozen=True)
class CanonicalDescriptorMixConfig:
    rho: float = 0.1
    mix_mode: str = "contrastive_opposite"
    prefill_steering: str = "last_token_only"
    steering_operator: str = "key_logit_sparse"
    positive_gain: float = 1.2
    negative_gain: float = 0.6
    query_adaptive_gates: bool = True
    query_gate_scale: float = 1.0
    prompt_bank_normalization: str = "none"
    diagnostic_max_decode_steps: int = 0
    record_prefill_diagnostics: bool = False
    head_ids: Tuple[int, ...] = ()
    kv_group_ids: Tuple[int, ...] = ()
    layer_head_ids: Optional[Mapping[int, Tuple[int, ...]]] = None
    layer_kv_group_ids: Optional[Mapping[int, Tuple[int, ...]]] = None


@dataclass(frozen=True)
class CanonicalAuxiliaryMemoryBank:
    bank_name: str
    memory: CanonicalDescriptorMemory
    gain: float = 1.0


@dataclass
class CanonicalPatchedAttentionContext:
    patches: List[Tuple[torch.nn.Module, object]] = field(default_factory=list)

    def __enter__(self) -> "CanonicalPatchedAttentionContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def remove(self) -> None:
        for module, original_forward in reversed(self.patches):
            module.forward = original_forward
        self.patches.clear()


class CanonicalDescriptorMemoryAttentionPatch:
    """
    External descriptor-memory patch that stores keys in canonical pre-RoPE space and
    inverse-rotates the live query only for the memory branch.
    """

    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        memory: CanonicalDescriptorMemory,
        reference_memory: Optional[CanonicalDescriptorMemory],
        config: CanonicalDescriptorMixConfig,
        auxiliary_memories: Optional[Sequence[CanonicalAuxiliaryMemoryBank]] = None,
        control_state: Optional[AttentionToMLPControlState] = None,
    ) -> None:
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        self.memory = memory
        self.reference_memory = reference_memory
        rho = float(config.rho)
        mix_mode = str(config.mix_mode).strip().lower()
        prefill_steering = str(config.prefill_steering).strip().lower()
        steering_operator = str(config.steering_operator).strip().lower()
        prompt_bank_normalization = str(config.prompt_bank_normalization).strip().lower()
        if mix_mode not in {"convex", "contrastive_neutral", "contrastive_opposite"}:
            raise ValueError(f"Unsupported mix_mode '{config.mix_mode}'.")
        if prefill_steering not in {"full", "last_token_only", "none"}:
            raise ValueError(f"Unsupported prefill_steering '{config.prefill_steering}'.")
        if steering_operator not in {"output_mixture", "key_logit_sparse", "bank_softmax_mixture"}:
            raise ValueError(f"Unsupported steering_operator '{config.steering_operator}'.")
        if prompt_bank_normalization not in {"none", "log_token_count"}:
            raise ValueError(f"Unsupported prompt_bank_normalization '{config.prompt_bank_normalization}'.")
        if mix_mode == "contrastive_opposite" and reference_memory is None:
            raise ValueError(f"mix_mode='{mix_mode}' requires a reference_memory.")
        if steering_operator == "key_logit_sparse" and mix_mode == "convex":
            raise ValueError("steering_operator='key_logit_sparse' requires a contrastive mix_mode.")
        normalized_auxiliary_memories: List[CanonicalAuxiliaryMemoryBank] = []
        seen_aux_names = set()
        for bank in auxiliary_memories or ():
            bank_name = str(bank.bank_name).strip().lower()
            if not bank_name:
                raise ValueError("Auxiliary memory bank names must be non-empty.")
            if bank_name in {"prompt", "trait", "reference"}:
                raise ValueError(f"Auxiliary memory bank name '{bank_name}' is reserved.")
            if bank_name in seen_aux_names:
                raise ValueError(f"Duplicate auxiliary memory bank name '{bank_name}'.")
            seen_aux_names.add(bank_name)
            normalized_auxiliary_memories.append(
                CanonicalAuxiliaryMemoryBank(
                    bank_name=bank_name,
                    memory=bank.memory,
                    gain=max(0.0, float(bank.gain)),
                )
            )
        if normalized_auxiliary_memories and steering_operator != "bank_softmax_mixture":
            raise ValueError("Auxiliary memory banks are currently supported only with steering_operator='bank_softmax_mixture'.")
        if mix_mode == "convex":
            rho = max(0.0, min(1.0, rho))
        else:
            rho = max(0.0, rho)
        self.auxiliary_memories = tuple(normalized_auxiliary_memories)
        self.control_state = control_state
        self.layer_head_ids = self._normalize_layer_selection_map(config.layer_head_ids)
        self.layer_kv_group_ids = self._normalize_layer_selection_map(config.layer_kv_group_ids)
        self.config = CanonicalDescriptorMixConfig(
            rho=rho,
            mix_mode=mix_mode,
            prefill_steering=prefill_steering,
            steering_operator=steering_operator,
            positive_gain=max(0.0, float(config.positive_gain)),
            negative_gain=max(0.0, float(config.negative_gain)),
            query_adaptive_gates=bool(config.query_adaptive_gates),
            query_gate_scale=float(config.query_gate_scale),
            prompt_bank_normalization=prompt_bank_normalization,
            diagnostic_max_decode_steps=max(0, int(config.diagnostic_max_decode_steps)),
            record_prefill_diagnostics=bool(config.record_prefill_diagnostics),
            head_ids=tuple(int(head_id) for head_id in config.head_ids),
            kv_group_ids=tuple(int(kv_group_id) for kv_group_id in config.kv_group_ids),
            layer_head_ids=self.layer_head_ids,
            layer_kv_group_ids=self.layer_kv_group_ids,
        )
        self.diagnostic_traces: List[dict] = []
        self._diagnostic_steps_by_layer: Dict[int, int] = {}

    @staticmethod
    def _normalize_layer_selection_map(
        mapping: Optional[Mapping[int, Sequence[int]]],
    ) -> Optional[Dict[int, Tuple[int, ...]]]:
        if not mapping:
            return None
        normalized: Dict[int, Tuple[int, ...]] = {}
        for layer_id, values in mapping.items():
            normalized[int(layer_id)] = tuple(int(value) for value in values)
        return normalized

    def _prompt_bank_log_prior(self, prompt_token_count: int) -> float:
        if self.config.prompt_bank_normalization == "log_token_count":
            return _memory_log_prior(prompt_token_count)
        return 0.0

    def _trait_phrase_head_masses(
        self,
        *,
        trait_slot_probs: Optional[torch.Tensor],
        selected_head_ids: Sequence[int],
    ) -> Dict[str, dict]:
        phrases = tuple(getattr(self.memory, "phrases", ()) or ())
        if trait_slot_probs is None or not phrases:
            return {}

        def tensor_per_head_mean(value: torch.Tensor) -> List[float]:
            value_tensor = value.detach().to(dtype=torch.float32)
            if value_tensor.ndim >= 3:
                per_head_values = value_tensor.mean(dim=tuple(dim for dim in range(value_tensor.ndim) if dim != 1))
            elif value_tensor.ndim == 2:
                per_head_values = value_tensor.mean(dim=0)
            else:
                per_head_values = value_tensor.reshape(-1)
            return [float(head_value.item()) for head_value in per_head_values.cpu()]

        out: Dict[str, dict] = {}
        for phrase in phrases:
            slot_indices = tuple(int(slot_index) for slot_index in getattr(phrase, "slot_indices", ()) if int(slot_index) >= 0)
            if not slot_indices:
                continue
            index_tensor = torch.tensor(slot_indices, device=trait_slot_probs.device, dtype=torch.long)
            phrase_mass = trait_slot_probs.index_select(dim=-1, index=index_tensor).sum(dim=-1)
            per_head_values = tensor_per_head_mean(phrase_mass)
            out[str(phrase.phrase_id)] = {
                "phrase_text": str(phrase.phrase_text),
                "source_kind": str(phrase.source_kind),
                "phrase_index": int(phrase.phrase_index),
                "slot_count": len(slot_indices),
                "per_head_mass": {
                    str(int(head_id)): float(head_value)
                    for head_id, head_value in zip(selected_head_ids, per_head_values)
                },
            }
        return out

    def _maybe_record_trace(
        self,
        *,
        layer_id: int,
        selected_head_ids: Sequence[int],
        prompt_bank_mass: torch.Tensor,
        trait_bank_mass: torch.Tensor,
        reference_bank_mass: Optional[torch.Tensor],
        auxiliary_bank_masses: Optional[Dict[str, torch.Tensor]],
        positive_gate: Optional[torch.Tensor],
        negative_gate: Optional[torch.Tensor],
        alignment_margin: Optional[torch.Tensor],
        trait_phrase_masses: Optional[Dict[str, dict]],
        prompt_token_count: int,
        trait_slot_count: int,
        reference_slot_count: int,
        trace_kind: str,
        query_len: int,
    ) -> None:
        max_steps = int(self.config.diagnostic_max_decode_steps)
        if max_steps <= 0:
            return
        step = int(self._diagnostic_steps_by_layer.get(int(layer_id), 0))
        if step >= max_steps:
            return
        self._diagnostic_steps_by_layer[int(layer_id)] = step + 1

        def tensor_mean(value: Optional[torch.Tensor]) -> Optional[float]:
            if value is None:
                return None
            return float(value.detach().to(dtype=torch.float32).mean().item())

        def tensor_per_head_mean(value: Optional[torch.Tensor]) -> Dict[str, float]:
            if value is None:
                return {}
            value_tensor = value.detach().to(dtype=torch.float32)
            if value_tensor.ndim >= 3:
                per_head_values = value_tensor.mean(dim=tuple(dim for dim in range(value_tensor.ndim) if dim != 1))
            elif value_tensor.ndim == 2:
                per_head_values = value_tensor.mean(dim=0)
            else:
                per_head_values = value_tensor.reshape(-1)
            return {
                str(int(head_id)): float(head_value.item())
                for head_id, head_value in zip(selected_head_ids, per_head_values.cpu())
            }

        per_head_margin: Dict[str, float] = {}
        if alignment_margin is not None:
            margin_tensor = alignment_margin.detach().to(dtype=torch.float32)
            if margin_tensor.ndim >= 3:
                per_head_values = margin_tensor.amax(dim=tuple(dim for dim in range(margin_tensor.ndim) if dim != 1))
            elif margin_tensor.ndim == 2:
                per_head_values = margin_tensor.amax(dim=0)
            else:
                per_head_values = margin_tensor.reshape(-1)
            per_head_margin = {
                str(int(head_id)): float(value.item())
                for head_id, value in zip(selected_head_ids, per_head_values.cpu())
            }

        self.diagnostic_traces.append(
            {
                "layer_id": int(layer_id),
                "decode_step": step,
                "trace_kind": str(trace_kind),
                "query_len": int(query_len),
                "selected_head_ids": [int(head_id) for head_id in selected_head_ids],
                "prompt_token_count": int(prompt_token_count),
                "trait_slot_count": int(trait_slot_count),
                "reference_slot_count": int(reference_slot_count),
                "prompt_bank_mass": tensor_mean(prompt_bank_mass),
                "trait_bank_mass": tensor_mean(trait_bank_mass),
                "reference_bank_mass": tensor_mean(reference_bank_mass),
                "per_head_prompt_bank_mass": tensor_per_head_mean(prompt_bank_mass),
                "per_head_trait_bank_mass": tensor_per_head_mean(trait_bank_mass),
                "per_head_reference_bank_mass": tensor_per_head_mean(reference_bank_mass),
                "auxiliary_bank_masses": {
                    str(name): tensor_mean(value)
                    for name, value in (auxiliary_bank_masses or {}).items()
                },
                "per_head_auxiliary_bank_masses": {
                    str(name): tensor_per_head_mean(value)
                    for name, value in (auxiliary_bank_masses or {}).items()
                },
                "positive_gate_mean": tensor_mean(positive_gate),
                "negative_gate_mean": tensor_mean(negative_gate),
                "per_head_positive_gate": tensor_per_head_mean(positive_gate),
                "per_head_negative_gate": tensor_per_head_mean(negative_gate),
                "per_head_max_alignment_margin": per_head_margin,
                "per_phrase_trait_bank_mass": dict(trait_phrase_masses or {}),
            }
        )

    def patch_model(self, model: torch.nn.Module) -> CanonicalPatchedAttentionContext:
        context = CanonicalPatchedAttentionContext()
        model_layers = getattr(getattr(model, "model", model), "layers", None)
        if model_layers is None:
            raise AttributeError("Expected model.model.layers to exist.")

        for layer_id in self.layer_ids:
            layer_module = model_layers[int(layer_id)]
            attn_module = getattr(layer_module, "self_attn", None)
            if attn_module is None:
                raise AttributeError(f"Layer {layer_id} has no self_attn module.")
            original_forward = attn_module.forward
            attn_module.forward = self._make_wrapped_forward(int(layer_id), attn_module, original_forward, model)
            context.patches.append((attn_module, original_forward))
        return context

    def _make_wrapped_forward(
        self,
        layer_id: int,
        module: torch.nn.Module,
        original_forward: object,
        model: torch.nn.Module,
    ):
        signature = inspect.signature(original_forward)

        def wrapped_forward(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            extra_kwargs = bound.arguments.get("kwargs", {})
            if not isinstance(extra_kwargs, dict):
                extra_kwargs = {}
            hidden_states = bound.arguments.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                if self.control_state is not None:
                    self.control_state.clear_layer(layer_id)
                return original_forward(*args, **kwargs)

            attention_mask = bound.arguments.get("attention_mask")
            position_ids = bound.arguments.get("position_ids", extra_kwargs.get("position_ids"))
            position_embeddings = bound.arguments.get("position_embeddings")
            past_key_value = bound.arguments.get("past_key_value")
            if past_key_value is None:
                past_key_value = extra_kwargs.get("past_key_values")
            cache_position = bound.arguments.get("cache_position")
            output_attentions = bool(
                bound.arguments.get(
                    "output_attentions",
                    extra_kwargs.get("output_attentions", kwargs.get("output_attentions", False)),
                )
            )

            if not all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")):
                raise RuntimeError("Unsupported attention module for canonical descriptor-memory steering.")

            layer_memory = self.memory.layer_memories.get(layer_id)
            if layer_memory is None:
                if self.control_state is not None:
                    self.control_state.clear_layer(layer_id)
                return original_forward(*args, **kwargs)
            reference_layer_memory = None
            if self.reference_memory is not None:
                reference_layer_memory = self.reference_memory.layer_memories.get(layer_id)
            auxiliary_layer_memories = [
                bank
                for bank in self.auxiliary_memories
                if bank.memory.layer_memories.get(layer_id) is not None
            ]

            batch_size, query_len, _ = hidden_states.shape
            if query_len > 1 and self.config.prefill_steering == "none":
                if self.control_state is not None:
                    self.control_state.clear_layer(layer_id)
                return original_forward(*args, **kwargs)
            use_efficient_last_token_prefill = query_len > 1 and self.config.prefill_steering == "last_token_only"

            layout = resolve_attention_layout(module)
            selected_head_ids, _ = resolve_selected_query_heads(
                layer_id=layer_id,
                layout=layout,
                global_head_ids=self.config.head_ids,
                global_kv_group_ids=self.config.kv_group_ids,
                layer_head_ids=self.layer_head_ids,
                layer_kv_group_ids=self.layer_kv_group_ids,
            )
            scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(layout.head_dim)))

            query_states = module.q_proj(hidden_states).view(
                batch_size,
                query_len,
                layout.num_heads,
                layout.head_dim,
            )
            key_states = module.k_proj(hidden_states).view(
                batch_size,
                query_len,
                layout.num_key_value_heads,
                layout.head_dim,
            )
            value_states = module.v_proj(hidden_states).view(
                batch_size,
                query_len,
                layout.num_key_value_heads,
                layout.head_dim,
            )
            query_states, key_states = apply_qk_norm(module, query_states, key_states)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            resolved_position_ids = _resolve_position_ids(
                hidden_states,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_value=past_key_value,
                prefix_offset=0,
            )
            query_states, key_states, rope_cache_kwargs = apply_qwen3_rotary(
                module=module,
                model=model,
                query_states=query_states,
                key_states=key_states,
                position_embeddings=position_embeddings,
                position_ids=resolved_position_ids,
            )

            if past_key_value is not None:
                if hasattr(past_key_value, "update"):
                    if not use_efficient_last_token_prefill:
                        cache_kwargs = dict(rope_cache_kwargs)
                        cache_kwargs["cache_position"] = cache_position
                        key_states, value_states = past_key_value.update(key_states, value_states, layer_id, cache_kwargs)
                elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)

            prompt_key_states = _repeat_kv(key_states, layout.num_query_heads_per_kv_head)
            prompt_value_states = _repeat_kv(value_states, layout.num_query_heads_per_kv_head)
            if use_efficient_last_token_prefill:
                base_attn_output, base_attn_weights = original_forward(*args, **kwargs)
                cached_key_states, cached_value_states = _read_layer_cache_kv(past_key_value, layer_id)
                if cached_key_states is not None and cached_value_states is not None:
                    prompt_key_states = _repeat_kv(
                        cached_key_states.to(device=key_states.device, dtype=key_states.dtype),
                        layout.num_query_heads_per_kv_head,
                    )
                    prompt_value_states = _repeat_kv(
                        cached_value_states.to(device=value_states.device, dtype=value_states.dtype),
                        layout.num_query_heads_per_kv_head,
                    )
                prompt_scores = None
                prompt_probs = None
                prompt_output = None
            else:
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

            cos = rope_cache_kwargs.get("cos")
            sin = rope_cache_kwargs.get("sin")
            if cos is None or sin is None:
                raise RuntimeError("Canonical descriptor-memory requires RoPE cos/sin tensors from the live attention call.")
            canonical_query_states = _inverse_rotate_query(
                selected_query_states,
                cos=_slice_rope_cache_tensor(cos, query_slice),
                sin=_slice_rope_cache_tensor(sin, query_slice),
            )

            trait_keys_rep, trait_values_rep = _expand_memory_state(
                layer_memory,
                batch_size=batch_size,
                device=key_states.device,
                key_dtype=key_states.dtype,
                value_dtype=value_states.dtype,
                num_kv_groups=layout.num_query_heads_per_kv_head,
            )
            prompt_token_count = int(prompt_value_states.shape[2])
            prompt_bank_log_prior = self._prompt_bank_log_prior(prompt_token_count)
            selected_index = torch.tensor(selected_head_ids, device=query_states.device, dtype=torch.long)

            if self.config.steering_operator == "output_mixture":

                def compute_memory_output(memory_keys_rep: torch.Tensor, memory_values_rep: torch.Tensor) -> torch.Tensor:
                    memory_scores = torch.matmul(canonical_query_states, memory_keys_rep.transpose(-1, -2)) * scaling
                    memory_probs = F.softmax(memory_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    return torch.matmul(memory_probs, memory_values_rep)

                trait_output = compute_memory_output(trait_keys_rep, trait_values_rep)
                steered_selected_output = selected_prompt_output.clone()
                if self.config.mix_mode == "convex":
                    steered_selected_heads = (
                        (1.0 - self.config.rho) * selected_prompt_output.index_select(1, selected_index)
                        + self.config.rho * trait_output.index_select(1, selected_index)
                    )
                elif self.config.mix_mode == "contrastive_neutral":
                    steered_selected_heads = (
                        selected_prompt_output.index_select(1, selected_index)
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
                        num_kv_groups=layout.num_query_heads_per_kv_head,
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
                if reference_layer_memory is None and self.config.mix_mode == "contrastive_opposite":
                    raise KeyError(
                        f"steering_operator='{self.config.steering_operator}' requires a reference layer memory for layer {layer_id}."
                    )
                canonical_query_heads = canonical_query_states.index_select(1, selected_index)
                selected_prompt_scores_heads = selected_prompt_scores.index_select(1, selected_index)
                selected_prompt_values_heads = prompt_value_states.index_select(1, selected_index)
                prompt_output_heads = selected_prompt_output.index_select(1, selected_index)
                trait_keys_heads = trait_keys_rep.index_select(1, selected_index)
                trait_values_heads = trait_values_rep.index_select(1, selected_index)
                use_reference = self.config.mix_mode == "contrastive_opposite" and reference_layer_memory is not None
                reference_keys_heads = None
                reference_values_heads = None
                if use_reference:
                    reference_keys_rep, reference_values_rep = _expand_memory_state(
                        reference_layer_memory,
                        batch_size=batch_size,
                        device=key_states.device,
                        key_dtype=key_states.dtype,
                        value_dtype=value_states.dtype,
                        num_kv_groups=layout.num_query_heads_per_kv_head,
                    )
                    reference_keys_heads = reference_keys_rep.index_select(1, selected_index)
                    reference_values_heads = reference_values_rep.index_select(1, selected_index)

                trait_raw_scores = torch.matmul(canonical_query_heads, trait_keys_heads.transpose(-1, -2)) * scaling
                trait_alignment = trait_raw_scores.max(dim=-1).values
                trait_slot_probs = F.softmax(trait_raw_scores, dim=-1, dtype=torch.float32)
                if use_reference and reference_keys_heads is not None:
                    reference_raw_scores = torch.matmul(canonical_query_heads, reference_keys_heads.transpose(-1, -2)) * scaling
                    reference_alignment = reference_raw_scores.max(dim=-1).values
                else:
                    reference_raw_scores = None
                    reference_alignment = torch.zeros_like(trait_alignment)

                if self.config.query_adaptive_gates and use_reference:
                    gate_logits = self.config.query_gate_scale * (trait_alignment - reference_alignment)
                    positive_gate = torch.sigmoid(gate_logits)
                    negative_gate = torch.sigmoid(-gate_logits)
                else:
                    positive_gate = torch.ones_like(trait_alignment)
                    negative_gate = None if not use_reference else torch.ones_like(reference_alignment)

                trait_scores = trait_raw_scores + (
                    _memory_log_prior(_memory_slot_count(self.memory))
                    + self.config.rho * self.config.positive_gain * positive_gate.unsqueeze(-1)
                )
                trait_scores = trait_scores.to(dtype=query_states.dtype)

                if use_reference and reference_raw_scores is not None:
                    reference_scores = reference_raw_scores + (
                        _memory_log_prior(
                            _memory_slot_count(self.reference_memory) if self.reference_memory is not None else 1
                        )
                        - self.config.rho * self.config.negative_gain * negative_gate.unsqueeze(-1)
                    )
                    reference_scores = reference_scores.to(dtype=query_states.dtype)
                else:
                    reference_scores = None

                normalized_prompt_scores = selected_prompt_scores_heads + prompt_bank_log_prior

                if self.config.steering_operator == "key_logit_sparse":
                    combined_score_parts = [normalized_prompt_scores, trait_scores]
                    combined_value_parts = [selected_prompt_values_heads, trait_values_heads]
                    trait_slot_count = int(trait_values_heads.shape[2])
                    reference_slot_count = 0
                    if use_reference:
                        if reference_scores is None or reference_values_heads is None:
                            raise KeyError(
                                f"steering_operator='{self.config.steering_operator}' requires reference memory for layer {layer_id}."
                            )
                        combined_score_parts.append(reference_scores)
                        combined_value_parts.append(reference_values_heads)
                        reference_slot_count = int(reference_values_heads.shape[2])
                    combined_scores = torch.cat(combined_score_parts, dim=-1)
                    combined_probs = F.softmax(combined_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    combined_values = torch.cat(combined_value_parts, dim=2)
                    steered_selected_heads = torch.matmul(combined_probs, combined_values)
                    prompt_bank_mass = combined_probs[:, :, :, : prompt_token_count].sum(dim=-1)
                    trait_bank_mass = combined_probs[
                        :, :, :, prompt_token_count : prompt_token_count + trait_slot_count
                    ].sum(dim=-1)
                    if reference_slot_count > 0:
                        reference_bank_mass = combined_probs[
                            :, :, :, prompt_token_count + trait_slot_count : prompt_token_count + trait_slot_count + reference_slot_count
                        ].sum(dim=-1)
                    else:
                        reference_bank_mass = torch.zeros_like(trait_bank_mass)
                    auxiliary_bank_masses: Dict[str, torch.Tensor] = {}
                else:
                    trait_output_heads = torch.matmul(
                        F.softmax(trait_raw_scores, dim=-1, dtype=torch.float32).to(query_states.dtype),
                        trait_values_heads,
                    )
                    bank_logits = [torch.logsumexp(normalized_prompt_scores, dim=-1), torch.logsumexp(trait_scores, dim=-1)]
                    bank_outputs = [prompt_output_heads, trait_output_heads]
                    bank_names = ["prompt", "trait"]
                    reference_slot_count = 0
                    auxiliary_bank_slot_counts: Dict[str, int] = {}
                    for auxiliary_bank in auxiliary_layer_memories:
                        auxiliary_layer_memory = auxiliary_bank.memory.layer_memories.get(layer_id)
                        if auxiliary_layer_memory is None:
                            continue
                        auxiliary_keys_rep, auxiliary_values_rep = _expand_memory_state(
                            auxiliary_layer_memory,
                            batch_size=batch_size,
                            device=key_states.device,
                            key_dtype=key_states.dtype,
                            value_dtype=value_states.dtype,
                            num_kv_groups=layout.num_query_heads_per_kv_head,
                        )
                        auxiliary_keys_heads = auxiliary_keys_rep.index_select(1, selected_index)
                        auxiliary_values_heads = auxiliary_values_rep.index_select(1, selected_index)
                        auxiliary_raw_scores = torch.matmul(canonical_query_heads, auxiliary_keys_heads.transpose(-1, -2)) * scaling
                        auxiliary_scores = auxiliary_raw_scores + (
                            _memory_log_prior(_memory_slot_count(auxiliary_bank.memory))
                            + self.config.rho * float(auxiliary_bank.gain)
                        )
                        auxiliary_output_heads = torch.matmul(
                            F.softmax(auxiliary_raw_scores, dim=-1, dtype=torch.float32).to(query_states.dtype),
                            auxiliary_values_heads,
                        )
                        bank_logits.append(torch.logsumexp(auxiliary_scores.to(dtype=query_states.dtype), dim=-1))
                        bank_outputs.append(auxiliary_output_heads)
                        bank_names.append(str(auxiliary_bank.bank_name))
                        auxiliary_bank_slot_counts[str(auxiliary_bank.bank_name)] = int(auxiliary_values_heads.shape[2])
                    if use_reference and reference_scores is not None and reference_values_heads is not None:
                        reference_output_heads = torch.matmul(
                            F.softmax(reference_raw_scores, dim=-1, dtype=torch.float32).to(query_states.dtype),
                            reference_values_heads,
                        )
                        bank_logits.append(torch.logsumexp(reference_scores, dim=-1))
                        bank_outputs.append(reference_output_heads)
                        bank_names.append("reference")
                        reference_slot_count = int(reference_values_heads.shape[2])
                    bank_probs = F.softmax(torch.stack(bank_logits, dim=-1), dim=-1, dtype=torch.float32).to(query_states.dtype)
                    steered_selected_heads = torch.zeros_like(prompt_output_heads)
                    for bank_idx, bank_output in enumerate(bank_outputs):
                        steered_selected_heads = steered_selected_heads + bank_probs[..., bank_idx].unsqueeze(-1) * bank_output
                    bank_mass_by_name = {
                        str(bank_name): bank_probs[..., bank_idx]
                        for bank_idx, bank_name in enumerate(bank_names)
                    }
                    prompt_bank_mass = bank_mass_by_name["prompt"]
                    trait_bank_mass = bank_mass_by_name["trait"]
                    reference_bank_mass = bank_mass_by_name.get("reference", torch.zeros_like(trait_bank_mass))
                    auxiliary_bank_masses = {
                        str(bank_name): mass
                        for bank_name, mass in bank_mass_by_name.items()
                        if bank_name not in {"prompt", "trait", "reference"}
                    }
                    trait_slot_count = int(trait_values_heads.shape[2])

                steered_selected_output = selected_prompt_output.clone()
                steered_selected_output[:, selected_index, :, :] = steered_selected_heads

                if query_len == 1:
                    should_record_trace = True
                    trace_kind = "decode"
                else:
                    should_record_trace = bool(self.config.record_prefill_diagnostics)
                    trace_kind = "prefill"
                if should_record_trace:
                    self._maybe_record_trace(
                        layer_id=layer_id,
                        selected_head_ids=selected_head_ids,
                        prompt_bank_mass=prompt_bank_mass,
                        trait_bank_mass=trait_bank_mass,
                        reference_bank_mass=reference_bank_mass,
                        auxiliary_bank_masses=auxiliary_bank_masses,
                        positive_gate=positive_gate,
                        negative_gate=negative_gate,
                        alignment_margin=trait_alignment - reference_alignment,
                        trait_phrase_masses=self._trait_phrase_head_masses(
                            trait_slot_probs=trait_slot_probs,
                            selected_head_ids=selected_head_ids,
                        ),
                        prompt_token_count=prompt_token_count,
                        trait_slot_count=trait_slot_count,
                        reference_slot_count=reference_slot_count,
                        trace_kind=trace_kind,
                        query_len=query_len,
                    )

            if query_len == 1 or self.config.prefill_steering == "full":
                mixed_output = steered_selected_output
            else:
                if use_efficient_last_token_prefill:
                    mixed_output = None
                else:
                    mixed_output = prompt_output.clone()
                    mixed_output[:, :, query_slice, :] = steered_selected_output

            if use_efficient_last_token_prefill:
                last_token_output = steered_selected_output.transpose(1, 2).contiguous().reshape(
                    batch_size,
                    1,
                    layout.num_heads * layout.head_dim,
                )
                last_token_output = module.o_proj(last_token_output)
                attn_output = base_attn_output.clone()
                attn_output[:, query_slice, :] = last_token_output
            else:
                attn_output = mixed_output.transpose(1, 2).contiguous().reshape(
                    batch_size,
                    query_len,
                    layout.num_heads * layout.head_dim,
                )
                attn_output = module.o_proj(attn_output)

            if self.control_state is not None:
                prompt_selected_output = selected_prompt_output.transpose(1, 2).contiguous().reshape(
                    batch_size,
                    steered_selected_output.shape[2],
                    layout.num_heads * layout.head_dim,
                )
                prompt_selected_output = module.o_proj(prompt_selected_output)
                steered_selected_hidden = steered_selected_output.transpose(1, 2).contiguous().reshape(
                    batch_size,
                    steered_selected_output.shape[2],
                    layout.num_heads * layout.head_dim,
                )
                steered_selected_hidden = module.o_proj(steered_selected_hidden)
                delta_hidden = torch.zeros(
                    (batch_size, query_len, layout.num_heads * layout.head_dim),
                    device=attn_output.device,
                    dtype=attn_output.dtype,
                )
                delta_hidden[:, query_slice, :] = steered_selected_hidden - prompt_selected_output
                self.control_state.store(layer_id, delta_hidden)

            if use_efficient_last_token_prefill:
                attn_weights = base_attn_weights if output_attentions else None
            else:
                attn_weights = prompt_probs if output_attentions else None
            return attn_output, attn_weights

        return wrapped_forward
