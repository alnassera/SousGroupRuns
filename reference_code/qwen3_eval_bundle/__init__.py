"""Qwen3 MoE extensions for descriptor_memory_persona."""

from .architecture import (
    Qwen3MoeAttentionLayout,
    aggregate_query_head_scores_to_kv_groups,
    expand_kv_groups_to_query_heads,
    inspect_qwen3_moe_architecture,
    query_head_to_kv_group,
    resolve_attention_layout,
    resolve_selected_query_heads,
)
from .canonical_attention_patch import (
    CanonicalAuxiliaryMemoryBank,
    CanonicalDescriptorMemoryAttentionPatch,
    CanonicalDescriptorMixConfig,
)
from .canonical_memory import CanonicalDescriptorMemory, build_canonical_descriptor_memory

__all__ = [
    "CanonicalAuxiliaryMemoryBank",
    "CanonicalDescriptorMemory",
    "CanonicalDescriptorMemoryAttentionPatch",
    "CanonicalDescriptorMixConfig",
    "Qwen3MoeAttentionLayout",
    "aggregate_query_head_scores_to_kv_groups",
    "build_canonical_descriptor_memory",
    "expand_kv_groups_to_query_heads",
    "inspect_qwen3_moe_architecture",
    "query_head_to_kv_group",
    "resolve_attention_layout",
    "resolve_selected_query_heads",
]
