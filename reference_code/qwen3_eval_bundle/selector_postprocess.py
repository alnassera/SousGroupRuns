from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from qwen3_moe.track_a_refinement import normalize_selector_artifact
except ImportError:  # pragma: no cover - supports flat-file script execution
    from track_a_refinement import normalize_selector_artifact


HEAD_SELECTION_PROTOCOL_CHOICES = (
    "legacy",
    "route_margin_alignment",
    "engagement_advantage_alignment",
)

_EPS = 1e-6
_ROUTE_MARGIN_ALIGNMENT_PROTOCOL_VERSION = 5
_MIN_NON_TRAIT_ROUTE_POSITIVE_FRACTION = 0.5
_MIN_Q25_NON_TRAIT_ROUTE_MARGIN = 0.0


def _rows_to_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV artifact at {path}.")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _safe_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    clipped_q = min(max(float(q), 0.0), 1.0)
    index = (len(ordered) - 1) * clipped_q
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(ordered[lower])
    mix = index - float(lower)
    return float(ordered[lower]) * (1.0 - mix) + float(ordered[upper]) * mix


def _mean(values: Iterable[float]) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return float(sum(collected) / len(collected))


def _clamp_unit_interval(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return min(max(parsed, 0.0), 1.0)


def _route_stats_from_row(row: Mapping[str, object]) -> Dict[str, float]:
    trait_mass = max(0.0, _safe_float(row.get("trait_bank_mass")))
    prompt_mass = max(0.0, _safe_float(row.get("prompt_bank_mass")))
    reference_mass = max(0.0, _safe_float(row.get("reference_bank_mass")))
    auxiliary_mass = max(0.0, _safe_float(row.get("total_auxiliary_bank_mass")))
    non_trait_mass = prompt_mass + reference_mass + auxiliary_mass
    total_mass = trait_mass + non_trait_mass
    if total_mass <= 0.0:
        return {
            "route_margin": 0.0,
            "non_trait_route_margin": 0.0,
            "route_advantage": 0.0,
            "trait_prob": 0.0,
            "prompt_prob": 0.0,
            "non_trait_prob": 0.0,
        }

    trait_prob = float(trait_mass / total_mass)
    prompt_prob = float(prompt_mass / total_mass)
    non_trait_prob = float(non_trait_mass / total_mass)
    route_margin = math.log(trait_mass + _EPS) - math.log(prompt_mass + _EPS)
    non_trait_route_margin = math.log(trait_mass + _EPS) - math.log(non_trait_mass + _EPS)
    route_advantage = max(0.0, trait_prob - non_trait_prob)
    return {
        "route_margin": float(route_margin),
        "non_trait_route_margin": float(non_trait_route_margin),
        "route_advantage": float(route_advantage),
        "trait_prob": float(trait_prob),
        "prompt_prob": float(prompt_prob),
        "non_trait_prob": float(non_trait_prob),
    }


def _head_sort_key(row: Mapping[str, object]) -> Tuple[float, ...]:
    return (
        _safe_float(row.get("score")),
        _safe_float(row.get("non_trait_route_positive_fraction")),
        _safe_float(row.get("q25_non_trait_route_margin")),
        _safe_float(row.get("q25_alignment_margin")),
        _safe_float(row.get("joint_non_trait_positive_fraction")),
        _safe_float(row.get("mean_non_trait_route_margin")),
        _safe_float(row.get("route_positive_fraction")),
        _safe_float(row.get("q25_route_margin")),
        _safe_float(row.get("joint_positive_fraction")),
        _safe_float(row.get("mean_route_margin")),
        _safe_float(row.get("mean_alignment_margin")),
        _safe_float(row.get("mean_trait_bank_mass")),
        -_safe_float(row.get("mean_prompt_bank_mass")),
        -int(row.get("head_id", -1)),
    )


def _build_route_margin_alignment_head_scores(
    *,
    trace_rows: Sequence[Mapping[str, object]],
    selector_mode: str,
    protocol: str = "route_margin_alignment",
    route_blend_alpha: float = 0.0,
) -> List[Dict[str, object]]:
    blend_alpha = _clamp_unit_interval(route_blend_alpha, default=0.0)
    grouped: Dict[Tuple[int, int], List[Mapping[str, object]]] = {}
    for row in trace_rows:
        key = (int(row["layer_id"]), int(row["head_id"]))
        grouped.setdefault(key, []).append(row)

    head_rows: List[Dict[str, object]] = []
    for (layer_id, head_id), rows in sorted(grouped.items()):
        alignments = [_safe_float(row.get("alignment_margin")) for row in rows]
        trait_masses = [max(0.0, _safe_float(row.get("trait_bank_mass"))) for row in rows]
        prompt_masses = [max(0.0, _safe_float(row.get("prompt_bank_mass"))) for row in rows]
        reference_masses = [max(0.0, _safe_float(row.get("reference_bank_mass"))) for row in rows]
        auxiliary_masses = [max(0.0, _safe_float(row.get("total_auxiliary_bank_mass"))) for row in rows]
        signals = [_safe_float(row.get("signal")) for row in rows]
        route_margins: List[float] = []
        non_trait_route_margins: List[float] = []
        route_blend_margins: List[float] = []
        route_advantages: List[float] = []
        bank_engagements: List[float] = []
        utilities: List[float] = []
        engagement_utilities: List[float] = []
        for row, alignment in zip(rows, alignments):
            route_stats = _route_stats_from_row(row)
            route_margin = float(route_stats["route_margin"])
            non_trait_route_margin = float(route_stats["non_trait_route_margin"])
            route_blend_margin = (blend_alpha * route_margin) + ((1.0 - blend_alpha) * non_trait_route_margin)
            route_advantage = float(route_stats["route_advantage"])
            trait_prob = float(route_stats["trait_prob"])
            prompt_prob = float(route_stats["prompt_prob"])
            bank_engagement = trait_prob + prompt_prob
            utility = max(0.0, route_blend_margin) * max(0.0, float(alignment))
            engagement_utility = max(0.0, route_advantage) * max(0.0, float(alignment))
            route_margins.append(route_margin)
            non_trait_route_margins.append(non_trait_route_margin)
            route_blend_margins.append(route_blend_margin)
            route_advantages.append(route_advantage)
            bank_engagements.append(bank_engagement)
            utilities.append(float(utility))
            engagement_utilities.append(float(engagement_utility))

        score_value = (
            _mean(engagement_utilities)
            if str(protocol).strip().lower() == "engagement_advantage_alignment"
            else _mean(utilities)
        )

        head_rows.append(
            {
                "selector_mode": str(selector_mode),
                "layer_id": int(layer_id),
                "head_id": int(head_id),
                "score": float(score_value),
                "score_protocol": str(protocol),
                "mean_hybrid_utility": _mean(utilities),
                "mean_engagement_utility": _mean(engagement_utilities),
                "mean_bank_engagement": _mean(bank_engagements),
                "mean_route_advantage": _mean(route_advantages),
                "mean_route_margin": _mean(route_margins),
                "mean_non_trait_route_margin": _mean(non_trait_route_margins),
                "mean_route_blend_margin": _mean(route_blend_margins),
                "q25_route_margin": _quantile(route_margins, 0.25),
                "q25_non_trait_route_margin": _quantile(non_trait_route_margins, 0.25),
                "q25_route_blend_margin": _quantile(route_blend_margins, 0.25),
                "route_positive_fraction": _mean(1.0 if value > 0.0 else 0.0 for value in route_margins),
                "non_trait_route_positive_fraction": _mean(
                    1.0 if value > 0.0 else 0.0 for value in non_trait_route_margins
                ),
                "route_blend_positive_fraction": _mean(
                    1.0 if value > 0.0 else 0.0 for value in route_blend_margins
                ),
                "joint_positive_fraction": _mean(
                    1.0 if route_margin > 0.0 and alignment > 0.0 else 0.0
                    for route_margin, alignment in zip(route_margins, alignments)
                ),
                "joint_non_trait_positive_fraction": _mean(
                    1.0 if route_margin > 0.0 and alignment > 0.0 else 0.0
                    for route_margin, alignment in zip(non_trait_route_margins, alignments)
                ),
                "joint_route_blend_positive_fraction": _mean(
                    1.0 if route_margin > 0.0 and alignment > 0.0 else 0.0
                    for route_margin, alignment in zip(route_blend_margins, alignments)
                ),
                "alignment_positive_fraction": _mean(1.0 if value > 0.0 else 0.0 for value in alignments),
                "mean_alignment_margin": _mean(alignments),
                "q25_alignment_margin": _quantile(alignments, 0.25),
                "mean_trait_bank_mass": _mean(trait_masses),
                "mean_reference_bank_mass": _mean(reference_masses),
                "mean_prompt_bank_mass": _mean(prompt_masses),
                "mean_total_auxiliary_bank_mass": _mean(auxiliary_masses),
                "mean_signal": _mean(signals),
                "num_trace_rows": int(len(rows)),
            }
        )
    return sorted(head_rows, key=lambda row: (int(row["layer_id"]), int(row["head_id"])))


def _layer_selection_score(chosen_rows: Sequence[Mapping[str, object]], metric: str) -> float:
    metric_key = str(metric).strip().lower()
    if metric_key in {"top_head_score_sum", "top_head_utility_sum"}:
        return float(sum(_safe_float(row.get("score")) for row in chosen_rows))
    if metric_key in {"top_head_score_mean", "top_head_utility_mean"}:
        return _mean(_safe_float(row.get("score")) for row in chosen_rows)
    if metric_key == "positive_head_count":
        return float(sum(1 for row in chosen_rows if _safe_float(row.get("score")) > 0.0))
    raise ValueError(f"Unsupported layer selection metric '{metric}'.")


def _dedup_heads_by_kv_group(
    ordered_rows: Sequence[Mapping[str, object]],
    *,
    max_heads: int,
    kv_group_size: int,
    allow_group_backfill: bool,
) -> List[Dict[str, object]]:
    """Pick at most one head per KV group first, then fill remaining budget by score.

    `ordered_rows` must already be sorted best-first. With kv_group_size <= 1,
    this is a no-op truncation (matches the legacy behaviour).
    """
    budget = max(0, int(max_heads))
    if budget <= 0 or not ordered_rows:
        return []
    if int(kv_group_size) <= 1:
        return [dict(row) for row in ordered_rows[:budget]]

    used_groups: set[int] = set()
    primary: List[Dict[str, object]] = []
    leftover: List[Dict[str, object]] = []
    for row in ordered_rows:
        kv_group = int(row["head_id"]) // int(kv_group_size)
        if kv_group not in used_groups:
            used_groups.add(kv_group)
            primary.append(dict(row))
            if len(primary) >= budget:
                break
        else:
            leftover.append(dict(row))
    if allow_group_backfill and len(primary) < budget:
        primary.extend(dict(row) for row in leftover[: budget - len(primary)])
    return primary


def _head_passes_route_consistency_gate(
    row: Mapping[str, object],
    *,
    min_non_trait_route_positive_fraction: float,
    min_q25_non_trait_route_margin: float,
) -> bool:
    score = _safe_float(row.get("score"))
    if score <= 0.0:
        return False
    non_trait_route_positive_fraction = _safe_float(
        row.get("non_trait_route_positive_fraction"),
        _safe_float(row.get("route_positive_fraction")),
    )
    if non_trait_route_positive_fraction < float(min_non_trait_route_positive_fraction):
        return False
    q25_non_trait_route_margin = _safe_float(
        row.get("q25_non_trait_route_margin"),
        _safe_float(row.get("q25_route_margin")),
    )
    if q25_non_trait_route_margin <= float(min_q25_non_trait_route_margin):
        return False
    return True


def _select_layers_from_head_scores(
    *,
    head_rows: Sequence[Mapping[str, object]],
    candidate_layer_ids: Sequence[int],
    max_heads_per_layer: int,
    max_layers: int,
    layer_selection_metric: str,
    kv_group_size: int = 1,
    min_non_trait_route_positive_fraction: float = _MIN_NON_TRAIT_ROUTE_POSITIVE_FRACTION,
    min_q25_non_trait_route_margin: float = _MIN_Q25_NON_TRAIT_ROUTE_MARGIN,
    allow_kv_group_backfill: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[int, Tuple[int, ...]], Dict[int, Tuple[int, ...]], List[int], List[int]]:
    candidate_head_map: Dict[int, Tuple[int, ...]] = {}
    candidate_kv_group_map: Dict[int, Tuple[int, ...]] = {}
    layer_rows: List[Dict[str, object]] = []

    for layer_id in [int(value) for value in candidate_layer_ids]:
        subset = [dict(row) for row in head_rows if int(row["layer_id"]) == int(layer_id)]
        subset.sort(key=_head_sort_key, reverse=True)
        positive_subset = [row for row in subset if _safe_float(row.get("score")) > 0.0]
        route_consistent_subset = [
            row
            for row in positive_subset
            if _head_passes_route_consistency_gate(
                row,
                min_non_trait_route_positive_fraction=float(min_non_trait_route_positive_fraction),
                min_q25_non_trait_route_margin=float(min_q25_non_trait_route_margin),
            )
        ]
        chosen_rows = _dedup_heads_by_kv_group(
            route_consistent_subset,
            max_heads=int(max_heads_per_layer),
            kv_group_size=int(kv_group_size),
            allow_group_backfill=bool(allow_kv_group_backfill),
        )
        chosen_head_ids = tuple(int(row["head_id"]) for row in chosen_rows)
        candidate_head_map[int(layer_id)] = chosen_head_ids
        if int(kv_group_size) > 1 and chosen_head_ids:
            candidate_kv_group_map[int(layer_id)] = tuple(
                sorted({int(head_id) // int(kv_group_size) for head_id in chosen_head_ids})
            )
        else:
            candidate_kv_group_map[int(layer_id)] = tuple(int(index) for index in range(len(chosen_head_ids)))
        layer_score = _layer_selection_score(chosen_rows, metric=str(layer_selection_metric))
        tiebreak_values = [_safe_float(row.get("q25_non_trait_route_margin")) for row in chosen_rows]
        score_values = [_safe_float(row.get("score")) for row in chosen_rows]
        if int(kv_group_size) > 1 and chosen_head_ids:
            selected_kv_group_count = len({int(hid) // int(kv_group_size) for hid in chosen_head_ids})
        else:
            selected_kv_group_count = int(len(chosen_head_ids))
        layer_rows.append(
            {
                "layer_id": int(layer_id),
                "candidate_head_count": int(len(subset)),
                "positive_head_count": int(len(positive_subset)),
                "route_consistent_head_count": int(len(route_consistent_subset)),
                "selected_head_count": int(len(chosen_rows)),
                "selected_kv_group_count": int(selected_kv_group_count),
                "kv_group_size": int(kv_group_size),
                "selected_head_ids_json": json.dumps([int(head_id) for head_id in chosen_head_ids]),
                "selected_head_scores_json": json.dumps([float(value) for value in score_values]),
                "selected_head_tiebreak_scores_json": json.dumps([float(value) for value in tiebreak_values]),
                "selected_head_score_sum": float(sum(score_values)),
                "selected_head_score_mean": _mean(score_values),
                "selected_head_tiebreak_score_mean": _mean(tiebreak_values),
                "best_head_id": None if not subset else int(subset[0]["head_id"]),
                "best_head_score": None if not subset else _safe_float(subset[0].get("score")),
                "best_head_tiebreak_score": None if not subset else _safe_float(subset[0].get("q25_non_trait_route_margin")),
                "layer_selection_metric": str(layer_selection_metric),
                "layer_score": float(layer_score),
                "min_non_trait_route_positive_fraction": float(min_non_trait_route_positive_fraction),
                "min_q25_non_trait_route_margin": float(min_q25_non_trait_route_margin),
                "used_non_positive_head_fallback": False,
            }
        )

    selectable_layers = [row for row in layer_rows if int(row["selected_head_count"]) > 0]
    positive_layers = [row for row in selectable_layers if _safe_float(row.get("layer_score")) > 0.0]
    ranked_layers = positive_layers if positive_layers else selectable_layers
    ranked_layers = sorted(
        ranked_layers,
        key=lambda row: (
            _safe_float(row.get("layer_score")),
            _safe_float(row.get("selected_head_score_mean")),
            _safe_float(row.get("selected_head_tiebreak_score_mean")),
            int(row.get("positive_head_count") or 0),
        ),
        reverse=True,
    )
    ranked_layer_ids_by_score = [int(row["layer_id"]) for row in ranked_layers]
    layer_rank_by_score = {
        int(layer_id): int(rank)
        for rank, layer_id in enumerate(ranked_layer_ids_by_score, start=1)
    }
    if int(max_layers) > 0:
        ranked_layers = ranked_layers[: int(max_layers)]
    selected_layer_ids_by_score = [int(row["layer_id"]) for row in ranked_layers]
    selected_layer_ids = set(selected_layer_ids_by_score)
    selected_map = {
        int(layer_id): tuple(int(head_id) for head_id in candidate_head_map.get(int(layer_id), ()))
        for layer_id in sorted(selected_layer_ids)
        if candidate_head_map.get(int(layer_id))
    }
    selected_kv_group_map = {
        int(layer_id): tuple(int(group_id) for group_id in candidate_kv_group_map.get(int(layer_id), ()))
        for layer_id in sorted(selected_layer_ids)
        if candidate_head_map.get(int(layer_id))
    }
    for row in layer_rows:
        row["selected_for_steering"] = bool(int(row["layer_id"]) in selected_layer_ids)
        row["layer_rank_by_score"] = layer_rank_by_score.get(int(row["layer_id"]))
        row["selected_rank_by_score"] = (
            int(selected_layer_ids_by_score.index(int(row["layer_id"])) + 1)
            if int(row["layer_id"]) in selected_layer_ids
            else None
        )
    return (
        layer_rows,
        candidate_head_map,
        selected_map,
        ranked_layer_ids_by_score,
        selected_layer_ids_by_score,
        selected_kv_group_map,
    )


def _backup_once(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.stem}.legacy{path.suffix}")
    if backup_path.exists():
        return backup_path
    backup_path.write_bytes(path.read_bytes())
    return backup_path


def _artifact_rho(artifact: Mapping[str, object]) -> Optional[float]:
    config = artifact.get("config")
    if isinstance(config, Mapping):
        rho_value = config.get("rho")
        if rho_value not in (None, ""):
            return float(rho_value)
    return None


def _artifact_schedule(artifact: Mapping[str, object]) -> Optional[str]:
    layer_rule = artifact.get("layer_selection_rule")
    if isinstance(layer_rule, Mapping):
        schedule = layer_rule.get("layer_weight_schedule")
        if schedule not in (None, ""):
            return str(schedule)
    config = artifact.get("config")
    if isinstance(config, Mapping):
        schedule = config.get("layer_weight_schedule")
        if schedule not in (None, ""):
            return str(schedule)
    return None


_DEFAULT_KV_GROUP_SIZE = 8  # Qwen3-30B-A3B: 32 query heads / 4 KV heads.


def _resolve_kv_group_size(artifact: Mapping[str, object], override: Optional[int]) -> int:
    if override is not None:
        return max(1, int(override))
    import os

    env_value = os.environ.get("HEAD_SELECTION_KV_GROUP_SIZE")
    if env_value not in (None, ""):
        try:
            return max(1, int(env_value))
        except (TypeError, ValueError):
            pass
    config = artifact.get("config")
    if isinstance(config, Mapping):
        for key in ("num_query_heads_per_kv_head", "kv_group_size", "num_query_heads_per_kv_group"):
            value = config.get(key)
            if value not in (None, ""):
                try:
                    return max(1, int(value))
                except (TypeError, ValueError):
                    continue
        num_heads = config.get("num_attention_heads")
        num_kv = config.get("num_key_value_heads")
        if num_heads not in (None, "") and num_kv not in (None, "", 0):
            try:
                return max(1, int(num_heads) // int(num_kv))
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    return int(_DEFAULT_KV_GROUP_SIZE)


def apply_head_selection_protocol(
    *,
    selector_artifact_path: Path,
    protocol: str,
    kv_group_size: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    protocol_key = str(protocol).strip().lower()
    if protocol_key in {"", "legacy", "none"}:
        return None
    if protocol_key not in {"route_margin_alignment", "engagement_advantage_alignment"}:
        raise ValueError(
            f"Unsupported head selection protocol '{protocol}'. "
            f"Expected one of {HEAD_SELECTION_PROTOCOL_CHOICES}."
        )

    selector_artifact_path = Path(selector_artifact_path).expanduser().resolve()
    score_dir = selector_artifact_path.parent
    artifact = json.loads(selector_artifact_path.read_text(encoding="utf-8"))
    legacy_selected_layer_ids_by_score = [
        int(layer_id)
        for layer_id in (artifact.get("selected_layer_ids_by_score") or [])
    ]
    legacy_selected_layer_head_map = {
        str(int(layer_id)): [int(head_id) for head_id in heads]
        for layer_id, heads in dict(artifact.get("selected_layer_head_map") or {}).items()
    }
    trace_rows = _load_csv_rows(score_dir / "trace_rows.csv")
    selector_mode = str(artifact.get("selector_mode", "questionnaire_attention_stats"))
    selector_postprocess = artifact.get("selector_postprocess")
    artifact_config = artifact.get("config")
    route_blend_alpha = 0.0
    if isinstance(selector_postprocess, Mapping):
        route_blend_alpha = _clamp_unit_interval(selector_postprocess.get("route_blend_alpha"), default=route_blend_alpha)
    if isinstance(artifact_config, Mapping):
        route_blend_alpha = _clamp_unit_interval(
            artifact_config.get("head_selection_route_blend_alpha"),
            default=route_blend_alpha,
        )
    head_rows = _build_route_margin_alignment_head_scores(
        trace_rows=trace_rows,
        selector_mode=selector_mode,
        protocol=protocol_key,
        route_blend_alpha=float(route_blend_alpha),
    )

    legacy_candidate_map = artifact.get("config", {}).get("candidate_layer_head_map") if isinstance(artifact.get("config"), Mapping) else None
    if isinstance(legacy_candidate_map, Mapping) and legacy_candidate_map:
        candidate_layer_ids = [int(layer_id) for layer_id in legacy_candidate_map.keys()]
    else:
        candidate_layer_ids = sorted({int(row["layer_id"]) for row in head_rows})

    selection_rule = artifact.get("selection_rule")
    layer_rule = artifact.get("layer_selection_rule")
    max_heads_per_layer = int(selection_rule.get("max_heads_per_layer", 0)) if isinstance(selection_rule, Mapping) else 0
    max_layers = int(layer_rule.get("max_layers", 0)) if isinstance(layer_rule, Mapping) else 0
    layer_selection_metric = (
        str(layer_rule.get("layer_selection_metric"))
        if isinstance(layer_rule, Mapping) and layer_rule.get("layer_selection_metric") not in (None, "")
        else "top_head_utility_sum"
    )

    resolved_kv_group_size = _resolve_kv_group_size(artifact, kv_group_size)
    layer_rows, candidate_head_map, selected_map, ranked_layer_ids_by_score, selected_layer_ids_by_score, selected_kv_group_map = _select_layers_from_head_scores(
        head_rows=head_rows,
        candidate_layer_ids=candidate_layer_ids,
        max_heads_per_layer=max_heads_per_layer,
        max_layers=max_layers,
        layer_selection_metric=layer_selection_metric,
        kv_group_size=int(resolved_kv_group_size),
        min_non_trait_route_positive_fraction=float(_MIN_NON_TRAIT_ROUTE_POSITIVE_FRACTION),
        min_q25_non_trait_route_margin=float(_MIN_Q25_NON_TRAIT_ROUTE_MARGIN),
        allow_kv_group_backfill=False,
    )

    updated = copy.deepcopy(dict(artifact))
    updated["head_scores"] = [dict(row) for row in head_rows]
    updated["layer_scores"] = [dict(row) for row in layer_rows]
    updated["candidate_top_head_map"] = {
        str(int(layer_id)): [int(head_id) for head_id in heads]
        for layer_id, heads in sorted(candidate_head_map.items())
    }
    updated["ranked_layer_ids_by_score"] = [int(layer_id) for layer_id in ranked_layer_ids_by_score]
    updated["selected_layer_ids_by_score"] = [int(layer_id) for layer_id in selected_layer_ids_by_score]
    updated["selected_layer_ids"] = [
        int(layer_id)
        for layer_id in candidate_layer_ids
        if int(layer_id) in {int(selected_layer_id) for selected_layer_id in selected_layer_ids_by_score}
    ]
    updated["selected_layer_head_map"] = {
        str(int(layer_id)): [int(head_id) for head_id in heads]
        for layer_id, heads in sorted(selected_map.items())
    }
    updated["selected_layer_kv_group_map"] = {
        str(int(layer_id)): [int(group_id) for group_id in groups]
        for layer_id, groups in sorted(selected_kv_group_map.items())
    }
    if protocol_key == "engagement_advantage_alignment":
        formula_text = (
            "rank heads by mean(max(p_trait - p_non_trait, 0) * max(alignment_margin, 0)), "
            "filter to score>0, non_trait_route_positive_fraction>=0.5, q25_non_trait_route_margin>0, "
            "then choose at most one head per KV group"
        )
        score_formula_text = "mean(max(p_trait - p_non_trait, 0) * max(alignment_margin, 0))"
    else:
        if float(route_blend_alpha) > 0.0:
            formula_text = (
                "rank heads by mean(max(alpha * route_margin + (1 - alpha) * non_trait_route_margin, 0) "
                "* max(alignment_margin, 0)), filter to score>0, "
                "non_trait_route_positive_fraction>=0.5, q25_non_trait_route_margin>0, "
                "then choose at most one head per KV group"
            )
            score_formula_text = (
                "mean(max(alpha * route_margin + (1 - alpha) * non_trait_route_margin, 0) "
                "* max(alignment_margin, 0))"
            )
        else:
            formula_text = (
                "rank heads by mean(max(log((trait_bank_mass + eps) / "
                "(prompt_bank_mass + reference_bank_mass + total_auxiliary_bank_mass + eps)), 0) "
                "* max(alignment_margin, 0)), filter to score>0, "
                "non_trait_route_positive_fraction>=0.5, q25_non_trait_route_margin>0, "
                "then choose at most one head per KV group"
            )
            score_formula_text = (
                "mean(max(log((trait_bank_mass + eps) / "
                "(prompt_bank_mass + reference_bank_mass + total_auxiliary_bank_mass + eps)), 0) "
                "* max(alignment_margin, 0))"
            )
    updated["selection_rule"] = {
        "formula": formula_text,
        "max_heads_per_layer": int(max_heads_per_layer),
        "head_positive_score_preferred": True,
        "kv_group_dedup": bool(int(resolved_kv_group_size) > 1),
        "kv_group_size": int(resolved_kv_group_size),
        "kv_group_dedup_strict": True,
        "min_non_trait_route_positive_fraction": float(_MIN_NON_TRAIT_ROUTE_POSITIVE_FRACTION),
        "min_q25_non_trait_route_margin": float(_MIN_Q25_NON_TRAIT_ROUTE_MARGIN),
        "protocol": protocol_key,
        "protocol_version": int(_ROUTE_MARGIN_ALIGNMENT_PROTOCOL_VERSION),
        "score_field": "score",
    }
    updated["selector_postprocess"] = {
        "protocol": protocol_key,
        "version": int(_ROUTE_MARGIN_ALIGNMENT_PROTOCOL_VERSION),
        "score_field": "score",
        "score_formula": score_formula_text,
        "route_margin_formula": "log((trait_bank_mass + eps) / (prompt_bank_mass + eps))",
        "non_trait_route_margin_formula": "log((trait_bank_mass + eps) / (prompt_bank_mass + reference_bank_mass + total_auxiliary_bank_mass + eps))",
        "route_blend_alpha": float(route_blend_alpha),
        "route_blend_formula": "alpha * route_margin + (1 - alpha) * non_trait_route_margin",
        "route_probability_definition": "p_trait = trait_bank_mass / (trait_bank_mass + prompt_bank_mass + reference_bank_mass + total_auxiliary_bank_mass)",
        "non_trait_mass_definition": "prompt_bank_mass + reference_bank_mass + total_auxiliary_bank_mass",
        "eligibility_filters": {
            "score_positive_required": True,
            "min_non_trait_route_positive_fraction": float(_MIN_NON_TRAIT_ROUTE_POSITIVE_FRACTION),
            "min_q25_non_trait_route_margin": float(_MIN_Q25_NON_TRAIT_ROUTE_MARGIN),
        },
        "kv_group_size": int(resolved_kv_group_size),
        "kv_group_dedup_strict": True,
        "tiebreak_fields": [
            "non_trait_route_positive_fraction",
            "q25_non_trait_route_margin",
            "q25_alignment_margin",
            "joint_non_trait_positive_fraction",
            "route_positive_fraction",
            "q25_route_margin",
        ],
        "backup_suffix": "legacy",
    }
    config = updated.get("config")
    if isinstance(config, Mapping):
        updated["config"] = dict(config)
        updated["config"]["head_selection_protocol"] = protocol_key
        updated["config"]["head_selection_protocol_version"] = int(_ROUTE_MARGIN_ALIGNMENT_PROTOCOL_VERSION)
        updated["config"]["head_selection_score_field"] = "score"
        updated["config"]["head_selection_route_blend_alpha"] = float(route_blend_alpha)
        updated["config"]["head_selection_min_non_trait_route_positive_fraction"] = float(
            _MIN_NON_TRAIT_ROUTE_POSITIVE_FRACTION
        )
        updated["config"]["head_selection_min_q25_non_trait_route_margin"] = float(
            _MIN_Q25_NON_TRAIT_ROUTE_MARGIN
        )
        updated["config"]["head_selection_kv_group_size"] = int(resolved_kv_group_size)
        updated["config"]["head_selection_kv_group_dedup_strict"] = True

    rho = _artifact_rho(updated)
    schedule = _artifact_schedule(updated)
    if rho is not None and schedule is not None:
        updated = normalize_selector_artifact(updated, rho=float(rho), schedule=str(schedule))

    _backup_once(selector_artifact_path)
    _backup_once(score_dir / "head_scores.csv")
    _backup_once(score_dir / "layer_scores.csv")

    selector_artifact_path.write_text(
        json.dumps(updated, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _rows_to_csv(score_dir / "head_scores.csv", head_rows)
    _rows_to_csv(score_dir / "layer_scores.csv", layer_rows)

    return {
        "protocol": protocol_key,
        "artifact_path": str(selector_artifact_path),
        "legacy_selected_layer_ids_by_score": legacy_selected_layer_ids_by_score,
        "legacy_selected_layer_head_map": legacy_selected_layer_head_map,
        "selected_layer_ids_by_score": [int(layer_id) for layer_id in updated.get("selected_layer_ids_by_score", [])],
        "selected_layer_head_map": {
            str(layer_id): [int(head_id) for head_id in heads]
            for layer_id, heads in dict(updated.get("selected_layer_head_map", {})).items()
        },
        "backup_paths": [
            str(path)
            for path in (
                selector_artifact_path.with_name(f"{selector_artifact_path.stem}.legacy{selector_artifact_path.suffix}"),
                (score_dir / "head_scores.csv").with_name("head_scores.legacy.csv"),
                (score_dir / "layer_scores.csv").with_name("layer_scores.legacy.csv"),
            )
            if path.exists()
        ],
    }
