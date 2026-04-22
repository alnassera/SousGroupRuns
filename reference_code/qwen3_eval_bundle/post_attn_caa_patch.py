from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

try:
    from .canonical_attention_patch import (
        CanonicalAuxiliaryMemoryBank,
        CanonicalDescriptorMemoryAttentionPatch,
        CanonicalDescriptorMixConfig,
    )
    from .canonical_memory import CanonicalDescriptorMemory
    from .site_activation import SiteActivationSteeringVector
except ImportError:  # pragma: no cover - supports flat-file layout
    from canonical_attention_patch import (
        CanonicalAuxiliaryMemoryBank,
        CanonicalDescriptorMemoryAttentionPatch,
        CanonicalDescriptorMixConfig,
    )
    from canonical_memory import CanonicalDescriptorMemory
    from site_activation import SiteActivationSteeringVector


@dataclass(frozen=True)
class HybridCanonicalCAAPatchConfig:
    rho: float = 0.1
    alpha: float = 0.0
    mix_mode: str = "contrastive_opposite"
    prefill_steering: str = "last_token_only"
    steering_operator: str = "bank_softmax_mixture"
    positive_gain: float = 1.2
    negative_gain: float = 0.6
    query_adaptive_gates: bool = True
    query_gate_scale: float = 1.0
    prompt_bank_normalization: str = "none"
    diagnostic_max_decode_steps: int = 0
    record_prefill_diagnostics: bool = False
    head_ids: tuple[int, ...] = ()
    kv_group_ids: tuple[int, ...] = ()
    layer_head_ids: Optional[dict[int, tuple[int, ...]]] = None
    layer_kv_group_ids: Optional[dict[int, tuple[int, ...]]] = None
    confidence_weighted_alpha: bool = False
    confidence_eta: float = 1.0


class PostAttentionResidualCAAPatch:
    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        steering_vector: SiteActivationSteeringVector,
        scale: float,
        prefill_steering: str,
    ) -> None:
        site = str(getattr(steering_vector, "site", "")).strip().lower()
        if site != "post_attn_resid":
            raise ValueError(
                "PostAttentionResidualCAAPatch requires a steering vector extracted at site "
                "'post_attn_resid'."
            )
        normalized_prefill = str(prefill_steering).strip().lower()
        if normalized_prefill not in {"full", "last_token_only", "none"}:
            raise ValueError(f"Unsupported prefill_steering '{prefill_steering}'.")
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        self.steering_vector = steering_vector
        self.scale = float(scale)
        self.prefill_steering = normalized_prefill

    def patch_model(self, model: torch.nn.Module):
        stack = contextlib.ExitStack()
        model_body = getattr(model, "model", model)
        model_layers = getattr(model_body, "layers", None)
        if model_layers is None:
            raise AttributeError("Expected model.model.layers to exist.")
        for layer_id in self.layer_ids:
            layer_module = model_layers[int(layer_id)]
            attn_module = getattr(layer_module, "self_attn", None)
            if attn_module is None:
                raise AttributeError(f"Layer {layer_id} has no self_attn module.")
            original_forward = attn_module.forward
            attn_module.forward = self._make_wrapped_forward(int(layer_id), original_forward)
            stack.callback(setattr, attn_module, "forward", original_forward)
        return stack

    def _make_wrapped_forward(self, layer_id: int, original_forward: object):
        signature = inspect.signature(original_forward)

        def wrapped_forward(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            hidden_states = bound.arguments.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return original_forward(*args, **kwargs)

            vector = self.steering_vector.layer_vectors.get(int(layer_id))
            if vector is None or self.scale == 0.0:
                return original_forward(*args, **kwargs)

            batch_size, query_len, hidden_dim = hidden_states.shape
            if query_len > 1 and self.prefill_steering == "none":
                return original_forward(*args, **kwargs)
            if query_len == 1 or self.prefill_steering == "full":
                query_slice = slice(None)
                selected_len = query_len
            else:
                query_slice = slice(query_len - 1, query_len)
                selected_len = 1

            result = original_forward(*args, **kwargs)
            if isinstance(result, tuple):
                attn_output = result[0]
                tail = result[1:]
            else:
                attn_output = result
                tail = ()
            if not isinstance(attn_output, torch.Tensor):
                return result

            delta = vector.to(device=attn_output.device, dtype=attn_output.dtype).view(1, 1, hidden_dim)
            delta = delta.expand(batch_size, selected_len, hidden_dim)
            attn_output = attn_output.clone()
            attn_output[:, query_slice, :] = attn_output[:, query_slice, :] + (self.scale * delta)

            if tail:
                return (attn_output, *tail)
            return attn_output

        return wrapped_forward


class HybridCanonicalMemoryCAAPatch:
    """
    Qwen3-friendly hybrid patch that composes canonical-memory steering with a
    site-matched post-attention CAA term.
    """

    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        memory: CanonicalDescriptorMemory,
        reference_memory: Optional[CanonicalDescriptorMemory],
        steering_vector: SiteActivationSteeringVector,
        config: HybridCanonicalCAAPatchConfig,
        auxiliary_memories: Optional[Sequence[CanonicalAuxiliaryMemoryBank]] = None,
    ) -> None:
        if bool(config.confidence_weighted_alpha):
            raise NotImplementedError(
                "confidence_weighted_alpha is not implemented in the standalone bundle. "
                "Use a fixed alpha schedule."
            )
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        self.memory = memory
        self.reference_memory = reference_memory
        self.steering_vector = steering_vector
        self.config = HybridCanonicalCAAPatchConfig(
            rho=float(config.rho),
            alpha=float(config.alpha),
            mix_mode=str(config.mix_mode).strip().lower(),
            prefill_steering=str(config.prefill_steering).strip().lower(),
            steering_operator=str(config.steering_operator).strip().lower(),
            positive_gain=float(config.positive_gain),
            negative_gain=float(config.negative_gain),
            query_adaptive_gates=bool(config.query_adaptive_gates),
            query_gate_scale=float(config.query_gate_scale),
            prompt_bank_normalization=str(config.prompt_bank_normalization).strip().lower(),
            diagnostic_max_decode_steps=max(0, int(config.diagnostic_max_decode_steps)),
            record_prefill_diagnostics=bool(config.record_prefill_diagnostics),
            head_ids=tuple(int(head_id) for head_id in config.head_ids),
            kv_group_ids=tuple(int(kv_group_id) for kv_group_id in getattr(config, "kv_group_ids", ()) or ()),
            layer_head_ids=None if getattr(config, "layer_head_ids", None) is None else {
                int(layer_id): tuple(int(head_id) for head_id in head_ids)
                for layer_id, head_ids in dict(config.layer_head_ids).items()
            },
            layer_kv_group_ids=None if getattr(config, "layer_kv_group_ids", None) is None else {
                int(layer_id): tuple(int(kv_group_id) for kv_group_id in kv_group_ids)
                for layer_id, kv_group_ids in dict(config.layer_kv_group_ids).items()
            },
            confidence_weighted_alpha=False,
            confidence_eta=float(config.confidence_eta),
        )
        self.auxiliary_memories = tuple(auxiliary_memories or ())
        self._canonical_patches: List[CanonicalDescriptorMemoryAttentionPatch] = []

    @property
    def diagnostic_traces(self) -> List[dict]:
        traces: List[dict] = []
        for patch in self._canonical_patches:
            traces.extend(list(getattr(patch, "diagnostic_traces", [])))
        return traces

    def patch_model(self, model):
        stack = contextlib.ExitStack()
        self._canonical_patches = []
        for layer_id in self.layer_ids:
            canonical_patch = CanonicalDescriptorMemoryAttentionPatch(
                layer_ids=[int(layer_id)],
                memory=self.memory,
                reference_memory=self.reference_memory,
                config=CanonicalDescriptorMixConfig(
                    rho=float(self.config.rho),
                    mix_mode=self.config.mix_mode,
                    prefill_steering=self.config.prefill_steering,
                    steering_operator=self.config.steering_operator,
                    positive_gain=float(self.config.positive_gain),
                    negative_gain=float(self.config.negative_gain),
                    query_adaptive_gates=bool(self.config.query_adaptive_gates),
                    query_gate_scale=float(self.config.query_gate_scale),
                    prompt_bank_normalization=self.config.prompt_bank_normalization,
                    diagnostic_max_decode_steps=int(self.config.diagnostic_max_decode_steps),
                    record_prefill_diagnostics=bool(self.config.record_prefill_diagnostics),
                    head_ids=tuple(int(head_id) for head_id in self.config.head_ids),
                    kv_group_ids=tuple(int(kv_group_id) for kv_group_id in self.config.kv_group_ids),
                    layer_head_ids=self.config.layer_head_ids,
                    layer_kv_group_ids=self.config.layer_kv_group_ids,
                ),
                auxiliary_memories=self.auxiliary_memories,
            )
            self._canonical_patches.append(canonical_patch)
            stack.enter_context(canonical_patch.patch_model(model))
            alpha = float(self.config.alpha)
            if alpha != 0.0:
                stack.enter_context(
                    PostAttentionResidualCAAPatch(
                        layer_ids=[int(layer_id)],
                        steering_vector=self.steering_vector,
                        scale=alpha,
                        prefill_steering=self.config.prefill_steering,
                    ).patch_model(model)
                )
        return stack


__all__ = [
    "HybridCanonicalCAAPatchConfig",
    "HybridCanonicalMemoryCAAPatch",
    "PostAttentionResidualCAAPatch",
]
