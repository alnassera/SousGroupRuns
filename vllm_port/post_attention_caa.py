from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch


@dataclass(frozen=True)
class PostAttentionCAAConfig:
    """Configuration for the post-attention residual CAA intervention."""

    layer_ids: tuple[int, ...]
    scale: float = 1.0
    prefill_steering: str = "last_token_only"

    def normalized_prefill_steering(self) -> str:
        value = str(self.prefill_steering).strip().lower()
        if value not in {"full", "last_token_only", "none"}:
            raise ValueError(f"Unsupported prefill_steering: {self.prefill_steering}")
        return value


def selected_token_slice(query_len: int, prefill_steering: str) -> Optional[slice]:
    """Match the token-selection semantics of the HF PostAttentionResidualCAAPatch."""

    mode = str(prefill_steering).strip().lower()
    if int(query_len) > 1 and mode == "none":
        return None
    if int(query_len) == 1 or mode == "full":
        return slice(None)
    return slice(int(query_len) - 1, int(query_len))


def apply_post_attention_caa(
    attn_output: torch.Tensor,
    *,
    layer_id: int,
    layer_vectors: Mapping[int, torch.Tensor],
    config: PostAttentionCAAConfig,
) -> torch.Tensor:
    """Apply the CAA delta to projected attention output.

    vLLM integration point: call this immediately after the Qwen3-MoE attention
    module has produced its projected attention output for a decoder layer, and
    before the layer adds that output back into the residual stream.
    """

    if int(layer_id) not in set(int(value) for value in config.layer_ids):
        return attn_output
    if float(config.scale) == 0.0:
        return attn_output
    vector = layer_vectors.get(int(layer_id))
    if vector is None:
        return attn_output
    if attn_output.ndim != 3:
        raise ValueError(f"Expected attn_output with shape [batch, query, hidden], got {tuple(attn_output.shape)}")

    batch_size, query_len, hidden_dim = attn_output.shape
    query_slice = selected_token_slice(int(query_len), config.normalized_prefill_steering())
    if query_slice is None:
        return attn_output

    delta = vector.to(device=attn_output.device, dtype=attn_output.dtype).view(1, 1, int(hidden_dim))
    delta = delta.expand(int(batch_size), int(attn_output[:, query_slice, :].shape[1]), int(hidden_dim))
    patched = attn_output.clone()
    patched[:, query_slice, :] = patched[:, query_slice, :] + (float(config.scale) * delta)
    return patched
