from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple


LayerSelectionMap = Dict[int, Tuple[int, ...]]

_HEAD_MAP_KEYS = (
    "selected_layer_head_map",
    "layer_head_map",
    "resolved_layer_head_map",
    "selected_query_head_ids_by_layer",
)
_KV_GROUP_MAP_KEYS = (
    "selected_layer_kv_group_map",
    "selected_kv_group_ids_by_layer",
)


def parse_layer_selection_specs(
    specs: Optional[Sequence[str]],
    *,
    label: str,
) -> LayerSelectionMap:
    out: LayerSelectionMap = {}
    for raw_spec in specs or ():
        text = str(raw_spec).strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"Invalid {label} entry {text!r}. Expected 'layer_id:value1,value2,...'.")
        layer_text, values_text = text.split(":", 1)
        layer_id = int(layer_text.strip())
        values = tuple(int(part.strip()) for part in values_text.split(",") if part.strip())
        if not values:
            raise ValueError(f"Layer {layer_id} in {label} must specify at least one value.")
        out[int(layer_id)] = values
    return out


def _normalize_map(raw_map: Mapping[object, Sequence[object]]) -> LayerSelectionMap:
    return {
        int(layer_id): tuple(int(value) for value in values)
        for layer_id, values in dict(raw_map).items()
        if list(values)
    }


def load_selection_maps_from_payload(payload: Mapping[str, object]) -> Tuple[LayerSelectionMap, LayerSelectionMap]:
    head_map: LayerSelectionMap = {}
    kv_group_map: LayerSelectionMap = {}
    for key in _HEAD_MAP_KEYS:
        raw_map = payload.get(key)
        if raw_map:
            head_map = _normalize_map(dict(raw_map))
            break
    for key in _KV_GROUP_MAP_KEYS:
        raw_map = payload.get(key)
        if raw_map:
            kv_group_map = _normalize_map(dict(raw_map))
            break
    return head_map, kv_group_map


def load_selection_maps_from_artifact(path_text: Optional[str]) -> Tuple[LayerSelectionMap, LayerSelectionMap]:
    if path_text is None or not str(path_text).strip():
        return {}, {}
    payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
    return load_selection_maps_from_payload(payload)


def resolve_selection_maps(
    *,
    manual_head_map: Optional[LayerSelectionMap] = None,
    manual_kv_group_map: Optional[LayerSelectionMap] = None,
    artifact_path: Optional[str] = None,
) -> Tuple[LayerSelectionMap, LayerSelectionMap]:
    artifact_head_map, artifact_kv_group_map = load_selection_maps_from_artifact(artifact_path)
    if manual_kv_group_map:
        return dict(manual_head_map or {}), dict(manual_kv_group_map)
    if manual_head_map:
        return dict(manual_head_map), {}
    if artifact_kv_group_map:
        return dict(artifact_head_map), dict(artifact_kv_group_map)
    return dict(artifact_head_map), {}
