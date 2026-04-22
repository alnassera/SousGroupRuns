"""Bundle-local qwen3_moe package wrappers."""

from pathlib import Path
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parent
_BUNDLE_ROOT = _PACKAGE_ROOT.parent
_bundle_root_text = str(_BUNDLE_ROOT)
if _bundle_root_text not in sys.path:
    sys.path.insert(0, _bundle_root_text)

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
