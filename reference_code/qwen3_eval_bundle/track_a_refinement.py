from __future__ import annotations

import copy
import csv
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from layer_schedule import layer_rhos

_ROUTE_MARGIN_ALIGNMENT_PROTOCOL_VERSION = 3


def rows_to_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_layer_head_map(layer_head_map: Optional[Mapping[int, Sequence[int]]]) -> Dict[int, Tuple[int, ...]]:
    out: Dict[int, Tuple[int, ...]] = {}
    for layer_id, heads in (layer_head_map or {}).items():
        normalized_heads = tuple(int(head_id) for head_id in heads)
        if normalized_heads:
            out[int(layer_id)] = normalized_heads
    return out


def artifact_kv_group_size(artifact: Mapping[str, object]) -> int:
    config = dict(artifact.get("config") or {}) if isinstance(artifact.get("config"), Mapping) else {}
    selection_rule = dict(artifact.get("selection_rule") or {}) if isinstance(artifact.get("selection_rule"), Mapping) else {}
    selector_postprocess = dict(artifact.get("selector_postprocess") or {}) if isinstance(artifact.get("selector_postprocess"), Mapping) else {}
    for value in (
        config.get("head_selection_kv_group_size"),
        selection_rule.get("kv_group_size"),
        selector_postprocess.get("kv_group_size"),
    ):
        if value in (None, "", 0):
            continue
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            continue
    return 1


def layer_kv_group_map(
    layer_head_map: Optional[Mapping[int, Sequence[int]]],
    *,
    kv_group_size: int,
) -> Dict[int, Tuple[int, ...]]:
    normalized = normalize_layer_head_map(layer_head_map)
    if not normalized:
        return {}
    if int(kv_group_size) <= 1:
        return {int(layer_id): tuple(int(index) for index, _ in enumerate(heads)) for layer_id, heads in normalized.items()}
    return {
        int(layer_id): tuple(sorted({int(head_id) // int(kv_group_size) for head_id in heads}))
        for layer_id, heads in normalized.items()
    }


def ordered_selected_layer_ids(
    selected_map: Mapping[int, Sequence[int]],
    *,
    preferred_order: Optional[Sequence[int]] = None,
) -> List[int]:
    normalized_map = normalize_layer_head_map(selected_map)
    if not normalized_map:
        return []
    if preferred_order:
        ordered: List[int] = []
        seen = set()
        for layer_id in preferred_order:
            normalized = int(layer_id)
            if normalized not in normalized_map or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        if ordered:
            return ordered
    return sorted(int(layer_id) for layer_id in normalized_map)


def ordered_selected_layer_ids_from_artifact(artifact: Mapping[str, object]) -> List[int]:
    selected_map = normalize_layer_head_map(
        artifact.get("selected_layer_head_map")
        or artifact.get("layer_head_map")
        or artifact.get("resolved_layer_head_map")
        or {}
    )
    if not selected_map:
        return []
    raw_layer_ids = artifact.get("selected_layer_ids") or []
    if raw_layer_ids:
        return ordered_selected_layer_ids(selected_map, preferred_order=[int(layer_id) for layer_id in raw_layer_ids])
    raw_by_score = artifact.get("selected_layer_ids_by_score") or []
    if raw_by_score:
        return ordered_selected_layer_ids(selected_map, preferred_order=[int(layer_id) for layer_id in raw_by_score])
    return ordered_selected_layer_ids(selected_map)


def resolved_layer_rho_map(
    layer_head_map: Mapping[int, Sequence[int]],
    *,
    rho: float,
    schedule: str,
    preferred_order: Optional[Sequence[int]] = None,
) -> Dict[int, float]:
    ordered_layer_ids = ordered_selected_layer_ids(layer_head_map, preferred_order=preferred_order)
    if not ordered_layer_ids:
        return {}
    return {
        int(layer_id): float(value)
        for layer_id, value in layer_rhos(ordered_layer_ids, float(rho), str(schedule)).items()
    }


def normalize_selector_artifact(
    artifact: Mapping[str, object],
    *,
    rho: float,
    schedule: str,
) -> Dict[str, object]:
    normalized = copy.deepcopy(dict(artifact))
    selected_map = normalize_layer_head_map(
        normalized.get("selected_layer_head_map")
        or normalized.get("layer_head_map")
        or normalized.get("resolved_layer_head_map")
        or {}
    )
    if not selected_map:
        return normalized
    raw_selected_layer_ids = normalized.get("selected_layer_ids") or []
    if raw_selected_layer_ids:
        ordered_layer_ids = ordered_selected_layer_ids(
            selected_map,
            preferred_order=[int(layer_id) for layer_id in raw_selected_layer_ids],
        )
    else:
        ordered_layer_ids = ordered_selected_layer_ids(selected_map)
    normalized["selected_layer_ids"] = [int(layer_id) for layer_id in ordered_layer_ids]
    raw_selected_layer_ids_by_score = normalized.get("selected_layer_ids_by_score") or []
    if raw_selected_layer_ids_by_score:
        normalized["selected_layer_ids_by_score"] = [
            int(layer_id)
            for layer_id in ordered_selected_layer_ids(
                selected_map,
                preferred_order=[int(layer_id) for layer_id in raw_selected_layer_ids_by_score],
            )
        ]
    normalized["selected_layer_head_map"] = {
        str(int(layer_id)): [int(head_id) for head_id in selected_map[int(layer_id)]]
        for layer_id in sorted(selected_map)
    }
    kv_group_size = artifact_kv_group_size(normalized)
    normalized["selected_layer_kv_group_map"] = {
        str(int(layer_id)): [int(group_id) for group_id in groups]
        for layer_id, groups in layer_kv_group_map(selected_map, kv_group_size=int(kv_group_size)).items()
    }
    normalized["selected_layer_rho_map"] = {
        str(int(layer_id)): float(value)
        for layer_id, value in resolved_layer_rho_map(
            selected_map,
            rho=float(rho),
            schedule=str(schedule),
            preferred_order=ordered_layer_ids,
        ).items()
    }
    layer_rule = normalized.get("layer_selection_rule")
    if isinstance(layer_rule, Mapping):
        normalized["layer_selection_rule"] = dict(layer_rule)
        normalized["layer_selection_rule"]["layer_weight_schedule"] = str(schedule)
    config = normalized.get("config")
    if isinstance(config, Mapping):
        normalized["config"] = dict(config)
        normalized["config"]["rho"] = float(rho)
    return normalized


def selector_artifact_matches_expectation(
    artifact: Mapping[str, object],
    *,
    expected_schedule: str,
    expected_rho: float,
    expected_positive_gain: Optional[float] = None,
    expected_negative_gain: Optional[float] = None,
    expected_selector_mode: Optional[str] = None,
    expected_track_memory_bank_profile: Optional[str] = None,
    expected_head_selection_protocol: Optional[str] = None,
) -> bool:
    layer_rule = artifact.get("layer_selection_rule")
    artifact_schedule = None
    if isinstance(layer_rule, Mapping):
        artifact_schedule = layer_rule.get("layer_weight_schedule")
    config = artifact.get("config")
    config_map = dict(config) if isinstance(config, Mapping) else {}
    if artifact_schedule is None:
        artifact_schedule = config_map.get("layer_weight_schedule")
    if artifact_schedule is not None and str(artifact_schedule) != str(expected_schedule):
        return False
    rho_value = config_map.get("rho")
    if rho_value is not None and abs(float(rho_value) - float(expected_rho)) > 1e-9:
        return False
    if expected_positive_gain is not None:
        positive_gain = config_map.get("positive_gain")
        if positive_gain is not None and abs(float(positive_gain) - float(expected_positive_gain)) > 1e-9:
            return False
    if expected_negative_gain is not None:
        negative_gain = config_map.get("negative_gain")
        if negative_gain is not None and abs(float(negative_gain) - float(expected_negative_gain)) > 1e-9:
            return False
    if expected_selector_mode is not None:
        selector_mode = config_map.get("requested_selector_mode", artifact.get("selector_mode"))
        if selector_mode is not None and str(selector_mode) != str(expected_selector_mode):
            return False
    if expected_track_memory_bank_profile is not None:
        profile = config_map.get("track_memory_bank_profile")
        if profile is not None and str(profile) != str(expected_track_memory_bank_profile):
            return False
    if expected_head_selection_protocol is not None:
        selector_postprocess = artifact.get("selector_postprocess")
        selector_postprocess_map = dict(selector_postprocess) if isinstance(selector_postprocess, Mapping) else {}
        selection_rule = artifact.get("selection_rule")
        selection_rule_map = dict(selection_rule) if isinstance(selection_rule, Mapping) else {}
        configured_protocol = (
            config_map.get("head_selection_protocol")
            or selector_postprocess_map.get("protocol")
            or "legacy"
        )
        if str(configured_protocol) != str(expected_head_selection_protocol):
            return False
        if str(expected_head_selection_protocol) == "route_margin_alignment":
            protocol_version = (
                config_map.get("head_selection_protocol_version")
                or selector_postprocess_map.get("version")
                or selection_rule_map.get("protocol_version")
            )
            if protocol_version is None or int(protocol_version) != int(_ROUTE_MARGIN_ALIGNMENT_PROTOCOL_VERSION):
                return False
    return True


def _head_row_metric(row: Mapping[str, object], name: str) -> Optional[float]:
    value = row.get(name)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def head_row_primary_score(row: Mapping[str, object]) -> Optional[float]:
    for name, sign in (
        ("score", 1.0),
        ("mean_alignment_margin", 1.0),
        ("mean_signal", 1.0),
        ("mean_trait_bank_mass", 1.0),
        ("mean_prompt_bank_mass", -1.0),
    ):
        value = _head_row_metric(row, name)
        if value is not None:
            return float(sign) * float(value)
    return None


def head_row_sort_key(row: Mapping[str, object]) -> Tuple[float, ...]:
    def metric(name: str) -> float:
        value = _head_row_metric(row, name)
        if value is None:
            return float("-inf")
        return float(value)

    return (
        metric("score"),
        metric("route_positive_fraction"),
        metric("q25_route_margin"),
        metric("q25_alignment_margin"),
        metric("joint_positive_fraction"),
        metric("q25_non_trait_route_margin"),
        metric("mean_route_margin"),
        metric("mean_non_trait_route_margin"),
        metric("mean_alignment_margin"),
        metric("mean_signal"),
        metric("mean_trait_bank_mass"),
        -metric("mean_prompt_bank_mass"),
    )


def refinement_candidate_map(
    *,
    head_rows: Sequence[Mapping[str, object]],
    selected_map: Mapping[int, Sequence[int]],
    top_candidates_per_layer: int,
) -> Dict[int, Tuple[int, ...]]:
    grouped: Dict[int, List[Mapping[str, object]]] = {}
    normalized_selected_map = normalize_layer_head_map(selected_map)
    for row in head_rows:
        layer_id = int(row["layer_id"])
        if layer_id in normalized_selected_map:
            grouped.setdefault(layer_id, []).append(row)

    out: Dict[int, Tuple[int, ...]] = {}
    for layer_id, selected_heads in sorted(normalized_selected_map.items()):
        ranked = sorted(grouped.get(int(layer_id), []), key=head_row_sort_key, reverse=True)
        ordered_heads = [int(head_id) for head_id in selected_heads]
        ordered_heads.extend(
            int(row["head_id"])
            for row in ranked[: max(0, int(top_candidates_per_layer))]
        )
        deduped = tuple(int(head_id) for head_id in dict.fromkeys(ordered_heads))
        if deduped:
            out[int(layer_id)] = deduped
    return out


def refinement_rank_index_map(
    *,
    head_rows: Sequence[Mapping[str, object]],
    candidate_map: Mapping[int, Sequence[int]],
) -> Dict[int, Dict[int, int]]:
    normalized_candidate_map = normalize_layer_head_map(candidate_map)
    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for row in head_rows:
        layer_id = int(row["layer_id"])
        if layer_id in normalized_candidate_map:
            grouped.setdefault(layer_id, []).append(row)

    out: Dict[int, Dict[int, int]] = {}
    for layer_id, candidate_heads in sorted(normalized_candidate_map.items()):
        ranked_rows = sorted(grouped.get(int(layer_id), []), key=head_row_sort_key, reverse=True)
        rank_map: Dict[int, int] = {}
        current_group_rank = -1
        previous_primary_score: Optional[float] = None
        for row in ranked_rows:
            primary_score = head_row_primary_score(row)
            if (
                previous_primary_score is None
                or primary_score is None
                or abs(float(primary_score) - float(previous_primary_score)) > 1e-12
            ):
                current_group_rank += 1
                previous_primary_score = primary_score
            rank_map[int(row["head_id"])] = int(current_group_rank)
        next_rank = current_group_rank + 1
        # Keep a deterministic fallback rank for candidate heads missing from the score table.
        for head_id in candidate_heads:
            normalized_head_id = int(head_id)
            if normalized_head_id in rank_map:
                continue
            rank_map[normalized_head_id] = int(next_rank)
            next_rank += 1
        out[int(layer_id)] = {int(head_id): int(rank_map[int(head_id)]) for head_id in candidate_heads}
    return out


def replace_single_head(
    layer_head_map: Mapping[int, Sequence[int]],
    *,
    layer_id: int,
    remove_head_id: int,
    add_head_id: int,
) -> Dict[int, Tuple[int, ...]]:
    updated: Dict[int, Tuple[int, ...]] = {}
    for current_layer, current_heads in sorted(layer_head_map.items()):
        layer = int(current_layer)
        heads = tuple(int(head_id) for head_id in current_heads)
        if layer != int(layer_id):
            updated[layer] = tuple(sorted(heads))
            continue
        replaced = [int(head_id) for head_id in heads if int(head_id) != int(remove_head_id)]
        replaced.append(int(add_head_id))
        updated[layer] = tuple(sorted(dict.fromkeys(replaced)))
    return {int(layer): tuple(int(head_id) for head_id in heads) for layer, heads in updated.items() if heads}


def questionnaire_objective(
    questionnaire_summary: Sequence[Mapping[str, object]],
    *,
    target_trait: str,
    method: str,
    drift_weight: float,
    max_drift_threshold: Optional[float],
) -> Dict[str, float]:
    subset = [
        row
        for row in questionnaire_summary
        if str(row.get("method")) == str(method) and str(row.get("target_trait")) == str(target_trait)
    ]
    if len(subset) != 1:
        raise ValueError(
            f"Expected one summary row for method={method!r}, target_trait={target_trait!r}, found {len(subset)}."
        )
    method_row = subset[0]
    target_score = float(method_row["target_trait_score"])
    target_shift = float(method_row["target_shift"])
    drift = float(method_row["non_target_drift"])
    objective = float(target_score) - float(drift_weight) * float(drift)
    if max_drift_threshold is not None and drift > float(max_drift_threshold):
        objective -= 1000.0 + float(drift)
    return {
        "objective": float(objective),
        "target_score": float(target_score),
        "target_shift": float(target_shift),
        "drift": float(drift),
    }


def run_local_head_refinement(
    *,
    selected_map: Mapping[int, Sequence[int]],
    candidate_map: Mapping[int, Sequence[int]],
    target_trait: str,
    method: str,
    drift_weight: float,
    max_drift_threshold: Optional[float],
    max_steps: int,
    max_neighbors_per_step: Optional[int],
    max_rank_gap: Optional[int],
    random_seed: Optional[int],
    rank_index_map: Optional[Mapping[int, Mapping[int, int]]],
    evaluate: Callable[[Mapping[int, Sequence[int]]], Dict[str, object]],
) -> Dict[str, object]:
    eval_cache: Dict[Tuple[Tuple[int, Tuple[int, ...]], ...], Dict[str, object]] = {}
    rng = random.Random(0 if random_seed is None else int(random_seed))
    normalized_max_rank_gap = None if max_rank_gap is None or int(max_rank_gap) < 0 else int(max_rank_gap)

    def signature(layer_head_map: Mapping[int, Sequence[int]]) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
        return tuple(
            (int(layer_id), tuple(int(head_id) for head_id in heads))
            for layer_id, heads in sorted(
                (int(layer), tuple(sorted(int(head) for head in heads)))
                for layer, heads in normalize_layer_head_map(layer_head_map).items()
            )
            if heads
        )

    def evaluate_cached(layer_head_map: Mapping[int, Sequence[int]]) -> Dict[str, object]:
        map_signature = signature(layer_head_map)
        cached = eval_cache.get(map_signature)
        if cached is not None:
            return cached
        evaluation = dict(evaluate(layer_head_map))
        metrics = questionnaire_objective(
            evaluation["questionnaire_summary"],
            target_trait=str(target_trait),
            method=str(method),
            drift_weight=float(drift_weight),
            max_drift_threshold=max_drift_threshold,
        )
        result = {
            "layer_head_map": normalize_layer_head_map(layer_head_map),
            "questionnaire_rows": list(evaluation["questionnaire_rows"]),
            "questionnaire_summary": list(evaluation["questionnaire_summary"]),
            **metrics,
        }
        eval_cache[map_signature] = result
        return result

    baseline = evaluate_cached(selected_map)
    current = baseline
    neighbor_rows: List[dict] = []
    step_rows: List[dict] = []

    for step_index in range(1, max(0, int(max_steps)) + 1):
        best_candidate: Optional[Dict[str, object]] = None
        trial_specs: List[Tuple[int, int, int]] = []
        for layer_id, current_heads in sorted(current["layer_head_map"].items()):
            current_head_set = {int(head_id) for head_id in current_heads}
            alternatives = [
                int(head_id)
                for head_id in candidate_map.get(int(layer_id), ())
                if int(head_id) not in current_head_set
            ]
            layer_rank_map = {
                int(head_id): int(rank_index)
                for head_id, rank_index in dict((rank_index_map or {}).get(int(layer_id), {})).items()
            }
            for remove_head_id in current_heads:
                for add_head_id in alternatives:
                    remove_rank = layer_rank_map.get(int(remove_head_id))
                    add_rank = layer_rank_map.get(int(add_head_id))
                    if normalized_max_rank_gap is not None:
                        if remove_rank is None or add_rank is None:
                            continue
                        if abs(int(remove_rank) - int(add_rank)) > int(normalized_max_rank_gap):
                            continue
                    trial_specs.append((int(layer_id), int(remove_head_id), int(add_head_id)))

        total_neighbor_count = len(trial_specs)
        if max_neighbors_per_step is not None:
            sampled_neighbor_count = max(0, min(int(max_neighbors_per_step), total_neighbor_count))
            if sampled_neighbor_count < total_neighbor_count:
                trial_specs = rng.sample(trial_specs, k=sampled_neighbor_count)
        evaluated_neighbor_count = len(trial_specs)

        for layer_id, remove_head_id, add_head_id in trial_specs:
            layer_rank_map = {
                int(head_id): int(rank_index)
                for head_id, rank_index in dict((rank_index_map or {}).get(int(layer_id), {})).items()
            }
            remove_rank = layer_rank_map.get(int(remove_head_id))
            add_rank = layer_rank_map.get(int(add_head_id))
            rank_gap = None
            if remove_rank is not None and add_rank is not None:
                rank_gap = abs(int(remove_rank) - int(add_rank))
            trial_map = replace_single_head(
                current["layer_head_map"],
                layer_id=int(layer_id),
                remove_head_id=int(remove_head_id),
                add_head_id=int(add_head_id),
            )
            trial = evaluate_cached(trial_map)
            improvement = float(trial["objective"]) - float(current["objective"])
            row = {
                "step_index": int(step_index),
                "layer_id": int(layer_id),
                "remove_head_id": int(remove_head_id),
                "add_head_id": int(add_head_id),
                "remove_rank": remove_rank,
                "add_rank": add_rank,
                "rank_gap": rank_gap,
                "objective": float(trial["objective"]),
                "improvement": float(improvement),
                "target_score": float(trial["target_score"]),
                "target_shift": float(trial["target_shift"]),
                "drift": float(trial["drift"]),
                "total_neighbor_count": int(total_neighbor_count),
                "evaluated_neighbor_count": int(evaluated_neighbor_count),
                "selected_layer_head_map_json": json.dumps(
                    {str(layer): list(heads) for layer, heads in trial["layer_head_map"].items()},
                    sort_keys=True,
                ),
            }
            neighbor_rows.append(row)
            if best_candidate is None or (
                float(improvement),
                float(trial["objective"]),
                float(trial["target_score"]),
                -float(trial["drift"]),
            ) > (
                float(best_candidate["improvement"]),
                float(best_candidate["objective"]),
                float(best_candidate["target_score"]),
                -float(best_candidate["drift"]),
            ):
                best_candidate = {
                    **row,
                    "result": trial,
                }
        if best_candidate is None or float(best_candidate["improvement"]) <= 0.0:
            break
        current = dict(best_candidate["result"])
        step_rows.append(
            {
                "step_index": int(step_index),
                "layer_id": int(best_candidate["layer_id"]),
                "remove_head_id": int(best_candidate["remove_head_id"]),
                "add_head_id": int(best_candidate["add_head_id"]),
                "objective": float(current["objective"]),
                "improvement": float(best_candidate["improvement"]),
                "target_score": float(current["target_score"]),
                "target_shift": float(current["target_shift"]),
                "drift": float(current["drift"]),
                "total_neighbor_count": int(best_candidate["total_neighbor_count"]),
                "evaluated_neighbor_count": int(best_candidate["evaluated_neighbor_count"]),
                "selected_layer_head_map_json": json.dumps(
                    {str(layer): list(heads) for layer, heads in current["layer_head_map"].items()},
                    sort_keys=True,
                ),
            }
        )
    return {
        "candidate_layer_head_map": {
            int(layer_id): tuple(int(head_id) for head_id in heads)
            for layer_id, heads in sorted(candidate_map.items())
        },
        "baseline": baseline,
        "final": current,
        "neighbor_rows": neighbor_rows,
        "step_rows": step_rows,
        "artifact": {
            "selector_mode": "questionnaire_local_swap_refinement",
            "base_layer_head_map": {
                str(int(layer_id)): [int(head_id) for head_id in heads]
                for layer_id, heads in baseline["layer_head_map"].items()
            },
            "candidate_layer_head_map": {
                str(int(layer_id)): [int(head_id) for head_id in heads]
                for layer_id, heads in candidate_map.items()
            },
            "final_layer_head_map": {
                str(int(layer_id)): [int(head_id) for head_id in heads]
                for layer_id, heads in current["layer_head_map"].items()
            },
            "objective_rule": {
                "formula": "target_trait_score - refinement_drift_weight * non_target_drift",
                "refinement_drift_weight": float(drift_weight),
                "refinement_max_drift_threshold": max_drift_threshold,
                "refinement_max_neighbors_per_step": max_neighbors_per_step,
                "refinement_max_rank_gap": normalized_max_rank_gap,
                "refinement_random_seed": random_seed,
                "method": str(method),
                "target_trait": str(target_trait),
            },
            "baseline_metrics": {
                "objective": float(baseline["objective"]),
                "target_score": float(baseline["target_score"]),
                "target_shift": float(baseline["target_shift"]),
                "drift": float(baseline["drift"]),
            },
            "final_metrics": {
                "objective": float(current["objective"]),
                "target_score": float(current["target_score"]),
                "target_shift": float(current["target_shift"]),
                "drift": float(current["drift"]),
            },
        },
    }


def save_refinement_outputs(refinement_dir: Path, refinement_result: Mapping[str, object]) -> None:
    refinement_dir.mkdir(parents=True, exist_ok=True)
    rows_to_csv(refinement_dir / "neighbor_search.csv", refinement_result["neighbor_rows"])
    rows_to_csv(refinement_dir / "selected_steps.csv", refinement_result["step_rows"])
    write_json(refinement_dir / "refinement_artifact.json", refinement_result["artifact"])


def build_refined_selector_artifact(
    *,
    base_artifact: Mapping[str, object],
    refinement_result: Mapping[str, object],
    rho: float,
    schedule: str,
) -> Dict[str, object]:
    final_layer_head_map = normalize_layer_head_map(refinement_result["artifact"]["final_layer_head_map"])
    refined = build_selector_artifact_variant(
        base_artifact=base_artifact,
        layer_head_map=final_layer_head_map,
        rho=float(rho),
        schedule=str(schedule),
        selector_mode="questionnaire_local_swap_refinement",
        variant_metadata={
            "source": "local_head_refinement",
        },
    )
    refined["refinement"] = copy.deepcopy(dict(refinement_result["artifact"]))
    return refined


def build_selector_artifact_variant(
    *,
    base_artifact: Mapping[str, object],
    layer_head_map: Mapping[int, Sequence[int]],
    rho: float,
    schedule: str,
    selector_mode: Optional[str] = None,
    variant_metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    normalized_base = normalize_selector_artifact(base_artifact, rho=float(rho), schedule=str(schedule))
    final_layer_head_map = normalize_layer_head_map(layer_head_map)
    if not final_layer_head_map:
        raise ValueError("build_selector_artifact_variant requires at least one selected layer/head pair.")
    preferred_steering_order = ordered_selected_layer_ids_from_artifact(normalized_base)
    preferred_score_order = [
        int(layer_id)
        for layer_id in (normalized_base.get("selected_layer_ids_by_score") or [])
    ]
    ordered_layer_ids = ordered_selected_layer_ids(final_layer_head_map, preferred_order=preferred_steering_order)
    ordered_layer_ids_by_score = ordered_selected_layer_ids(
        final_layer_head_map,
        preferred_order=preferred_score_order or preferred_steering_order,
    )
    refined = copy.deepcopy(dict(normalized_base))
    if selector_mode is not None:
        refined["selector_mode"] = str(selector_mode)
    refined["selected_layer_ids"] = [int(layer_id) for layer_id in ordered_layer_ids]
    refined["selected_layer_ids_by_score"] = [int(layer_id) for layer_id in ordered_layer_ids_by_score]
    refined["selected_layer_head_map"] = {
        str(int(layer_id)): [int(head_id) for head_id in final_layer_head_map[int(layer_id)]]
        for layer_id in sorted(final_layer_head_map)
    }
    kv_group_size = artifact_kv_group_size(refined)
    refined["selected_layer_kv_group_map"] = {
        str(int(layer_id)): [int(group_id) for group_id in groups]
        for layer_id, groups in layer_kv_group_map(final_layer_head_map, kv_group_size=int(kv_group_size)).items()
    }
    refined["selected_layer_rho_map"] = {
        str(int(layer_id)): float(value)
        for layer_id, value in resolved_layer_rho_map(
            final_layer_head_map,
            rho=float(rho),
            schedule=str(schedule),
            preferred_order=ordered_layer_ids,
        ).items()
    }
    if variant_metadata:
        existing_variant = refined.get("variant")
        variant_payload = dict(existing_variant) if isinstance(existing_variant, Mapping) else {}
        variant_payload.update(copy.deepcopy(dict(variant_metadata)))
        refined["variant"] = variant_payload
    return refined
