#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


PACKAGE_ROOT = Path(__file__).resolve().parent
VLLM_PORT_ROOT = PACKAGE_ROOT / "vllm_port"
VLLM_PORT_PLUGIN_NAME = "vllm_port_intervention"
VLLM_PORT_PLUGIN_DIST_DIR = VLLM_PORT_ROOT / "plugin_dist"

for candidate in (VLLM_PORT_PLUGIN_DIST_DIR, PACKAGE_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

try:
    from transformers import AutoConfig, AutoTokenizer
except Exception:  # pragma: no cover
    AutoConfig = None
    AutoTokenizer = None

from architecture import wrap_qwen_tokenizer_apply_chat_template
from canonical_memory import build_canonical_descriptor_memory
from modeling import load_model_and_tokenizer, resolve_hf_cache_dir
from math_memory_banks import (
    BANK_PROFILE_CHOICES,
    MATH_MEMORY_WRAPPER_TEMPLATES,
    PhysicsExample,
    PhysicsMemoryBankSpec,
    format_visible_prompt_block,
    load_math_examples,
    route_subject,
    summarize_examples_by_subject,
)
from run_math_benchmark import (
    DEFAULT_LAYER_IDS,
    DEFAULT_LAYER_WEIGHT_SCHEDULE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_METHODS,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_BANK_NORMALIZATION,
    _build_auxiliary_bank_specs,
    _build_hidden_patch,
    _build_problem_prompt,
    _close_handles,
    _generate_from_messages,
    _json_dump,
    _method_specs,
    _parse_layer_head_id_specs,
    _resolved_primary_bank_spec,
    _safe_slug,
    _selected_bank_payload,
    _subject_probabilities_as_json,
)

from vllm_port.artifacts import (  # type: ignore  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    build_layer_kv_group_map,
    save_artifact,
    serialize_canonical_memory,
    steering_artifact_tag,
)
from vllm_port.qwen3_moe_intervention_model import (  # type: ignore  # noqa: E402
    ARTIFACT_ENV_VAR,
    register as register_intervention_model,
)
from vllm_port.runtime_patches import apply_vllm_moe_runtime_fallbacks  # type: ignore  # noqa: E402


@dataclass(frozen=True)
class PreparedExample:
    example: PhysicsExample
    routing_decision: Any
    primary_bank_spec: PhysicsMemoryBankSpec
    auxiliary_bank_specs: Tuple[Tuple[PhysicsMemoryBankSpec, float], ...]

    @property
    def visible_banks(self) -> Tuple[PhysicsMemoryBankSpec, ...]:
        return (self.primary_bank_spec,) + tuple(bank for bank, _ in self.auxiliary_bank_specs)

    @property
    def bank_signature(self) -> Tuple[Tuple[str, float], ...]:
        return ((self.primary_bank_spec.bank_id, 1.0),) + tuple(
            (bank.bank_id, float(gain)) for bank, gain in self.auxiliary_bank_specs
        )


def _prepend_env_path(name: str, values: Sequence[Path]) -> None:
    existing = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    additions = [str(value) for value in values if str(value) not in existing]
    if additions:
        os.environ[name] = os.pathsep.join([*additions, *existing])


def _enable_vllm_port_plugin() -> None:
    for path in (VLLM_PORT_PLUGIN_DIST_DIR, PACKAGE_ROOT):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    _prepend_env_path("PYTHONPATH", (VLLM_PORT_PLUGIN_DIST_DIR, PACKAGE_ROOT))
    existing_plugins = [
        value.strip()
        for value in os.environ.get("VLLM_PLUGINS", "").split(",")
        if value.strip()
    ]
    if VLLM_PORT_PLUGIN_NAME not in existing_plugins:
        os.environ["VLLM_PLUGINS"] = ",".join([*existing_plugins, VLLM_PORT_PLUGIN_NAME])


def _model_source_and_tokenizer_kwargs(model_name: str) -> Tuple[str, Optional[str], Dict[str, Any]]:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path.resolve()), None, {"local_files_only": True}
    cache_dir, local_files_only = resolve_hf_cache_dir(model_name)
    kwargs: Dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if local_files_only:
        kwargs["local_files_only"] = True
    return str(model_name), cache_dir, kwargs


def _format_messages(tokenizer, messages: Sequence[Mapping[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
    parts = [f"[{str(message['role']).upper()}]\n{str(message['content'])}" for message in messages]
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


def _messages_for_method(
    *,
    method: str,
    prepared: PreparedExample,
    prompt_role: str,
    include_subject_name: bool,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if method == "prompt":
        visible_prompt = format_visible_prompt_block(prepared.visible_banks, include_header=True)
        if visible_prompt:
            messages.append({"role": str(prompt_role), "content": visible_prompt})
    messages.append(
        {
            "role": "user",
            "content": _build_problem_prompt(
                prepared.example,
                include_subject_name=bool(include_subject_name),
            ),
        }
    )
    return messages


def _combined_bank_text(bank_specs: Sequence[PhysicsMemoryBankSpec]) -> Tuple[str, Tuple[str, ...]]:
    descriptor_parts: List[str] = []
    variants: List[str] = []
    seen = set()
    for bank in bank_specs:
        descriptor_parts.append(f"[{bank.title}]\n{bank.descriptor}")
        for value in bank.descriptor_variants:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                variants.append(text)
    return "\n\n".join(descriptor_parts), tuple(variants)


def _selected_layer_head_map(
    args: argparse.Namespace,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> Dict[int, Tuple[int, ...]]:
    out: Dict[int, Tuple[int, ...]] = {}
    kv_group_size = max(1, int(num_attention_heads) // max(1, int(num_key_value_heads)))
    default_heads = tuple(group_id * kv_group_size for group_id in range(max(1, int(num_key_value_heads))))
    global_heads = tuple(int(head_id) for head_id in args.head_ids)
    for layer_id in args.layer_ids:
        layer_id = int(layer_id)
        if layer_id in args.layer_head_ids:
            out[layer_id] = tuple(int(head_id) for head_id in args.layer_head_ids[layer_id])
        elif global_heads:
            out[layer_id] = global_heads
        else:
            out[layer_id] = default_heads
    return out


def _validate_one_head_per_kv_group(
    selected_layer_head_map: Mapping[int, Sequence[int]],
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> None:
    kv_group_size = max(1, int(num_attention_heads) // max(1, int(num_key_value_heads)))
    for layer_id, head_ids in selected_layer_head_map.items():
        groups = [int(head_id) // kv_group_size for head_id in head_ids]
        if len(groups) != len(set(groups)):
            raise ValueError(
                f"Layer {layer_id} selects multiple query heads from the same KV group. "
                f"The current vLLM canonical-memory port requires one query head per KV group; "
                f"remove --head-ids/--layer-head-ids or select at most one head from each group of {kv_group_size}."
            )


def _safe_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _quantile(values: Sequence[float], q: float) -> float:
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


def _trace_rows_from_diagnostics(traces: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for trace in traces:
        layer_id = int(trace.get("layer_id", -1))
        selected_head_ids = [int(value) for value in trace.get("selected_head_ids", []) or []]
        prompt_by_head = dict(trace.get("per_head_prompt_bank_mass") or {})
        trait_by_head = dict(trace.get("per_head_trait_bank_mass") or {})
        reference_by_head = dict(trace.get("per_head_reference_bank_mass") or {})
        alignment_by_head = dict(trace.get("per_head_max_alignment_margin") or trace.get("per_head_trait_alignment") or {})
        aux_by_bank = dict(trace.get("per_head_auxiliary_bank_masses") or {})
        for head_id in selected_head_ids:
            head_key = str(int(head_id))
            prompt_mass = max(0.0, _safe_float(prompt_by_head.get(head_key)))
            trait_mass = max(0.0, _safe_float(trait_by_head.get(head_key)))
            reference_mass = max(0.0, _safe_float(reference_by_head.get(head_key)))
            auxiliary_mass = sum(
                max(0.0, _safe_float(dict(bank_values).get(head_key)))
                for bank_values in aux_by_bank.values()
                if isinstance(bank_values, Mapping)
            )
            rows.append(
                {
                    "layer_id": int(layer_id),
                    "head_id": int(head_id),
                    "trace_kind": str(trace.get("trace_kind", "")),
                    "decode_step": int(trace.get("decode_step", 0) or 0),
                    "prompt_bank_mass": float(prompt_mass),
                    "trait_bank_mass": float(trait_mass),
                    "reference_bank_mass": float(reference_mass),
                    "total_auxiliary_bank_mass": float(auxiliary_mass),
                    "alignment_margin": _safe_float(alignment_by_head.get(head_key)),
                }
            )
    return rows


def _head_score_rows_from_trace_rows(trace_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    eps = 1e-6
    grouped: Dict[Tuple[int, int], List[Mapping[str, object]]] = {}
    for row in trace_rows:
        grouped.setdefault((int(row["layer_id"]), int(row["head_id"])), []).append(row)
    out: List[Dict[str, object]] = []
    for (layer_id, head_id), rows in sorted(grouped.items()):
        utilities: List[float] = []
        route_margins: List[float] = []
        non_trait_route_margins: List[float] = []
        alignments: List[float] = []
        trait_masses: List[float] = []
        prompt_masses: List[float] = []
        physics_masses: List[float] = []
        physics_route_margins: List[float] = []
        physics_route_advantages: List[float] = []
        for row in rows:
            trait_mass = max(0.0, _safe_float(row.get("trait_bank_mass")))
            prompt_mass = max(0.0, _safe_float(row.get("prompt_bank_mass")))
            reference_mass = max(0.0, _safe_float(row.get("reference_bank_mass")))
            auxiliary_mass = max(0.0, _safe_float(row.get("total_auxiliary_bank_mass")))
            physics_mass = trait_mass + auxiliary_mass
            non_physics_mass = prompt_mass + reference_mass
            alignment = _safe_float(row.get("alignment_margin"))
            route_margin = math.log(physics_mass + eps) - math.log(prompt_mass + eps)
            non_trait_route_margin = math.log(physics_mass + eps) - math.log(non_physics_mass + eps)
            total_route_mass = physics_mass + non_physics_mass
            if total_route_mass > 0.0:
                physics_route_advantage = (physics_mass - non_physics_mass) / total_route_mass
            else:
                physics_route_advantage = 0.0
            # In the vLLM smoke path, all selected physics banks are serialized into one
            # memory artifact. Score heads by engagement with the combined physics bank,
            # while keeping trait-vs-prompt alignment as a bonus rather than a hard gate.
            utilities.append(max(0.0, non_trait_route_margin) * (1.0 + max(0.0, alignment)))
            route_margins.append(float(route_margin))
            non_trait_route_margins.append(float(non_trait_route_margin))
            physics_route_margins.append(float(non_trait_route_margin))
            physics_route_advantages.append(float(physics_route_advantage))
            alignments.append(float(alignment))
            trait_masses.append(float(trait_mass))
            prompt_masses.append(float(prompt_mass))
            physics_masses.append(float(physics_mass))
        out.append(
            {
                "layer_id": int(layer_id),
                "head_id": int(head_id),
                "score": _mean(utilities),
                "score_protocol": "combined_physics_bank_route_margin_v2",
                "mean_route_margin": _mean(route_margins),
                "mean_non_trait_route_margin": _mean(non_trait_route_margins),
                "q25_non_trait_route_margin": _quantile(non_trait_route_margins, 0.25),
                "non_trait_route_positive_fraction": _mean(
                    [1.0 if value > 0.0 else 0.0 for value in non_trait_route_margins]
                ),
                "mean_physics_route_margin": _mean(physics_route_margins),
                "mean_physics_route_advantage": _mean(physics_route_advantages),
                "mean_alignment_margin": _mean(alignments),
                "q25_alignment_margin": _quantile(alignments, 0.25),
                "mean_physics_bank_mass": _mean(physics_masses),
                "mean_trait_bank_mass": _mean(trait_masses),
                "mean_prompt_bank_mass": _mean(prompt_masses),
                "num_trace_rows": int(len(rows)),
            }
        )
    return out


def _select_layer_heads_from_scores(
    *,
    head_rows: Sequence[Mapping[str, object]],
    candidate_layer_ids: Sequence[int],
    num_attention_heads: int,
    num_key_value_heads: int,
    max_heads_per_layer: int,
    max_layers: int,
    layer_selection_metric: str,
) -> Tuple[Dict[int, Tuple[int, ...]], List[int], List[Dict[str, object]]]:
    kv_group_size = max(1, int(num_attention_heads) // max(1, int(num_key_value_heads)))
    layer_rows: List[Dict[str, object]] = []
    candidate_map: Dict[int, Tuple[int, ...]] = {}
    for layer_id in [int(value) for value in candidate_layer_ids]:
        subset = [dict(row) for row in head_rows if int(row["layer_id"]) == int(layer_id)]
        subset.sort(
            key=lambda row: (
                _safe_float(row.get("score")),
                _safe_float(row.get("non_trait_route_positive_fraction")),
                _safe_float(row.get("q25_non_trait_route_margin")),
                _safe_float(row.get("mean_alignment_margin")),
                -int(row.get("head_id", -1)),
            ),
            reverse=True,
        )
        selected: List[Dict[str, object]] = []
        used_groups = set()
        for row in subset:
            if _safe_float(row.get("score")) <= 0.0:
                continue
            group_id = int(row["head_id"]) // kv_group_size
            if group_id in used_groups:
                continue
            used_groups.add(group_id)
            selected.append(row)
            if len(selected) >= max(1, int(max_heads_per_layer)):
                break
        head_ids = tuple(int(row["head_id"]) for row in selected)
        candidate_map[int(layer_id)] = head_ids
        scores = [_safe_float(row.get("score")) for row in selected]
        metric = str(layer_selection_metric).strip().lower()
        if metric == "top_head_score_mean":
            layer_score = _mean(scores)
        elif metric == "positive_head_count":
            layer_score = float(sum(1 for score in scores if score > 0.0))
        else:
            layer_score = float(sum(scores))
        layer_rows.append(
            {
                "layer_id": int(layer_id),
                "candidate_head_count": int(len(subset)),
                "selected_head_count": int(len(head_ids)),
                "selected_head_ids": [int(head_id) for head_id in head_ids],
                "selected_head_scores": [float(score) for score in scores],
                "kv_group_size": int(kv_group_size),
                "layer_score": float(layer_score),
            }
        )
    ranked_layers = sorted(
        [row for row in layer_rows if int(row["selected_head_count"]) > 0],
        key=lambda row: (float(row["layer_score"]), int(row["selected_head_count"])),
        reverse=True,
    )
    if int(max_layers) > 0:
        ranked_layers = ranked_layers[: int(max_layers)]
    selected_layer_ids_by_score = [int(row["layer_id"]) for row in ranked_layers]
    selected_layer_set = set(selected_layer_ids_by_score)
    selected_map = {
        int(layer_id): tuple(int(head_id) for head_id in candidate_map.get(int(layer_id), ()))
        for layer_id in sorted(selected_layer_set)
        if candidate_map.get(int(layer_id))
    }
    return selected_map, selected_layer_ids_by_score, layer_rows


def _auto_select_layer_head_map(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    prepared_examples: Sequence[PreparedExample],
    num_attention_heads: int,
    num_key_value_heads: int,
) -> Tuple[Dict[int, Tuple[int, ...]], List[int], List[Dict[str, object]], List[Dict[str, object]]]:
    if args.head_ids or args.layer_head_ids or not bool(args.auto_select_heads):
        selected_map = _selected_layer_head_map(
            args,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        return selected_map, [int(layer_id) for layer_id in selected_map], [], []
    selector_args = argparse.Namespace(**vars(args))
    selector_args.head_ids = []
    selector_args.layer_head_ids = {}
    selector_args.diagnostic_max_decode_steps = int(args.selector_diagnostic_max_decode_steps)
    selector_args.record_prefill_diagnostics = bool(args.selector_record_prefill_diagnostics)
    selector_args.disable_progress = True
    memory_cache: Dict[Tuple[object, ...], Any] = {}
    trace_rows: List[Dict[str, object]] = []
    limit = min(len(prepared_examples), max(1, int(args.selector_max_examples)))
    print(
        "[math_vllm] auto head selection: "
        f"examples={limit} max_new_tokens={int(args.selector_max_new_tokens)} "
        f"diagnostic_decode_steps={int(args.selector_diagnostic_max_decode_steps)}",
        flush=True,
    )
    for index, prepared in enumerate(prepared_examples[:limit], start=1):
        print(
            "[math_vllm] auto head selection: "
            f"building diagnostic patch {index}/{limit} "
            f"id={prepared.example.example_id} category={prepared.example.subject}",
            flush=True,
        )
        hidden_patch, _, _, _ = _build_hidden_patch(
            model=model,
            tokenizer=tokenizer,
            primary_bank_spec=prepared.primary_bank_spec,
            auxiliary_bank_specs=prepared.auxiliary_bank_specs,
            routing_decision=prepared.routing_decision,
            args=selector_args,
            memory_cache=memory_cache,
        )
        print(
            "[math_vllm] auto head selection: "
            f"running diagnostic generation {index}/{limit}",
            flush=True,
        )
        messages = _messages_for_method(
            method="plain",
            prepared=prepared,
            prompt_role=str(args.prompt_role),
            include_subject_name=bool(args.include_subject_name_in_problem),
        )
        _generate_from_messages(
            model,
            tokenizer,
            messages=messages,
            max_new_tokens=int(args.selector_max_new_tokens),
            seed=int(args.seed),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=float(args.repetition_penalty),
            patch=hidden_patch,
        )
        trace_rows.extend(_trace_rows_from_diagnostics(getattr(hidden_patch, "diagnostic_traces", [])))
        print(
            "[math_vllm] auto head selection: "
            f"collected_trace_rows={len(trace_rows)} after {index}/{limit}",
            flush=True,
        )
    head_rows = _head_score_rows_from_trace_rows(trace_rows)
    top_head_rows = sorted(
        head_rows,
        key=lambda row: (
            _safe_float(row.get("score")),
            _safe_float(row.get("mean_physics_route_margin")),
            _safe_float(row.get("mean_alignment_margin")),
            -int(row.get("head_id", -1)),
        ),
        reverse=True,
    )[: max(0, int(args.selector_debug_top_heads))]
    positive_head_count = sum(1 for row in head_rows if _safe_float(row.get("score")) > 0.0)
    print(
        "[math_vllm] auto head selection: "
        f"head_score_rows={len(head_rows)} positive_heads={positive_head_count}",
        flush=True,
    )
    if top_head_rows:
        print(
            "[math_vllm] auto head selection top heads: "
            + json.dumps(
                [
                    {
                        "layer_id": int(row["layer_id"]),
                        "head_id": int(row["head_id"]),
                        "score": round(_safe_float(row.get("score")), 6),
                        "physics_margin": round(_safe_float(row.get("mean_physics_route_margin")), 6),
                        "alignment": round(_safe_float(row.get("mean_alignment_margin")), 6),
                        "physics_mass": round(_safe_float(row.get("mean_physics_bank_mass")), 6),
                        "prompt_mass": round(_safe_float(row.get("mean_prompt_bank_mass")), 6),
                    }
                    for row in top_head_rows
                ],
                sort_keys=True,
            ),
            flush=True,
        )
    selected_map, selected_layers_by_score, layer_rows = _select_layer_heads_from_scores(
        head_rows=head_rows,
        candidate_layer_ids=[int(layer_id) for layer_id in args.layer_ids],
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        max_heads_per_layer=int(args.max_heads_per_layer),
        max_layers=int(args.max_layers),
        layer_selection_metric=str(args.layer_selection_metric),
    )
    if not selected_map:
        print(
            "[math_vllm] auto head selection warning: no positive heads under the "
            "diagnostic score; falling back to one default query head per KV group.",
            flush=True,
        )
        selected_map = _selected_layer_head_map(
            args,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        selected_layers_by_score = [int(layer_id) for layer_id in selected_map]
    print(
        "[math_vllm] auto-selected layer/head map: "
        + json.dumps({str(k): list(v) for k, v in selected_map.items()}, sort_keys=True),
        flush=True,
    )
    print(
        "[math_vllm] auto-selected layer scores: "
        + json.dumps(
            [
                {
                    "layer_id": int(row["layer_id"]),
                    "selected_head_ids": [int(head_id) for head_id in row.get("selected_head_ids", [])],
                    "selected_head_scores": [
                        round(float(score), 6) for score in row.get("selected_head_scores", [])
                    ],
                    "layer_score": round(_safe_float(row.get("layer_score")), 6),
                }
                for row in layer_rows
                if int(row.get("selected_head_count", 0)) > 0
            ],
            sort_keys=True,
        ),
        flush=True,
    )
    return selected_map, selected_layers_by_score, head_rows, layer_rows


def _artifact_path_for_signature(
    *,
    args: argparse.Namespace,
    bank_signature: Tuple[Tuple[str, float], ...],
) -> Path:
    bank_tag = "_".join(_safe_slug(bank_id) for bank_id, _ in bank_signature)
    layer_tag = "layers_" + "_".join(str(int(layer_id)) for layer_id in args.layer_ids)
    if args.head_ids or args.layer_head_ids or not bool(args.auto_select_heads):
        selector_tag = "manual_heads"
    else:
        selector_source = str(getattr(args, "selector_math_dataset_split", "") or "eval_examples")
        if getattr(args, "selector_example_ids", None):
            selector_source = f"{selector_source}_ids{len(getattr(args, 'selector_example_ids'))}"
        selector_tag = (
            f"auto_heads_combined_v2_mh{int(args.max_heads_per_layer)}"
            f"_ml{int(args.max_layers)}"
            f"_se{int(args.selector_max_examples)}"
            f"_st{int(args.selector_max_new_tokens)}"
            f"_{_safe_slug(selector_source)}"
        )
    tag = steering_artifact_tag(
        method="canonical_memory",
        rho=float(args.rho),
        positive_gain=float(args.positive_gain),
        negative_gain=0.0,
        mix_mode=str(args.mix_mode),
        seed=int(args.seed),
    )
    return (
        Path(args.artifact_root).expanduser().resolve()
        / "math"
        / bank_tag
        / layer_tag
        / selector_tag
        / f"{tag}.pt"
    )


def _build_math_canonical_artifacts(
    *,
    args: argparse.Namespace,
    prepared_examples: Sequence[PreparedExample],
    selector_prepared_examples: Optional[Sequence[PreparedExample]] = None,
) -> Dict[Tuple[Tuple[str, float], ...], Path]:
    signatures: Dict[Tuple[Tuple[str, float], ...], Tuple[PhysicsMemoryBankSpec, ...]] = {}
    for prepared in prepared_examples:
        signatures.setdefault(prepared.bank_signature, prepared.visible_banks)
    artifact_paths = {
        signature: _artifact_path_for_signature(args=args, bank_signature=signature)
        for signature in signatures
    }
    if all(path.exists() and not bool(args.rebuild_canonical_artifacts) for path in artifact_paths.values()):
        print(
            "[math_vllm] using cached canonical artifacts: "
            + ", ".join(str(path) for path in artifact_paths.values()),
            flush=True,
        )
        return artifact_paths

    model = None
    try:
        print("[math_vllm] building canonical artifacts: loading HF model for selector/memory build", flush=True)
        model, tokenizer = load_model_and_tokenizer(
            str(args.model_name),
            device_map=str(args.artifact_device_map),
            torch_dtype=str(args.dtype),
            attn_implementation=str(args.artifact_attn_implementation),
            allow_eager_fallback=bool(args.allow_eager_attention),
            enable_thinking=bool(args.qwen_enable_thinking),
            strict_architecture_check=bool(args.strict_architecture_check),
        )
        config = getattr(model, "config", None)
        num_attention_heads = int(getattr(config, "num_attention_heads", 0) or 0)
        num_key_value_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
        if num_attention_heads <= 0 or num_key_value_heads <= 0:
            raise RuntimeError("Could not read Qwen attention head counts while building vLLM artifact.")
        kv_group_size = max(1, num_attention_heads // max(1, num_key_value_heads))
        for signature, bank_specs in signatures.items():
            artifact_path = artifact_paths[signature]
            if artifact_path.exists() and not bool(args.rebuild_canonical_artifacts):
                print(f"[math_vllm] using cached canonical artifact: {artifact_path}", flush=True)
                continue
            print(
                "[math_vllm] building canonical artifact for banks: "
                + ", ".join(f"{bank_id}:{gain:g}" for bank_id, gain in signature),
                flush=True,
            )
            signature_examples = [
                prepared
                for prepared in prepared_examples
                if prepared.bank_signature == signature
            ]
            if selector_prepared_examples is None:
                selector_signature_examples = signature_examples
                selector_source = "evaluation examples"
            else:
                selector_signature_examples = [
                    prepared
                    for prepared in selector_prepared_examples
                    if prepared.bank_signature == signature
                ]
                selector_source = "selector examples"
            if not selector_signature_examples:
                raise RuntimeError(
                    "No attention-head selector examples match bank signature "
                    + ", ".join(f"{bank_id}:{gain:g}" for bank_id, gain in signature)
                    + f" from {selector_source}. Check --selector-math-dataset-path, "
                    "--selector-categories, and routing/profile settings."
                )
            selected_layer_head_map, selected_layer_ids_by_score, selector_head_rows, selector_layer_rows = _auto_select_layer_head_map(
                args=args,
                model=model,
                tokenizer=tokenizer,
                prepared_examples=selector_signature_examples,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )
            _validate_one_head_per_kv_group(
                selected_layer_head_map,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )
            selected_layer_ids = sorted(int(layer_id) for layer_id in selected_layer_head_map)
            selected_layer_kv_group_map = build_layer_kv_group_map(
                selected_layer_head_map=selected_layer_head_map,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )
            descriptor, descriptor_variants = _combined_bank_text(bank_specs)
            wrapper_templates = tuple(str(value) for value in MATH_MEMORY_WRAPPER_TEMPLATES if str(value).strip())
            print(
                "[math_vllm] building combined canonical memory: "
                f"layers={selected_layer_ids} variants={len(descriptor_variants)}",
                flush=True,
            )
            memory = build_canonical_descriptor_memory(
                model,
                tokenizer,
                trait_name="hardmath_" + "_".join(bank_id for bank_id, _ in signature),
                descriptor=descriptor,
                layer_ids=selected_layer_ids,
                wrapper_template=wrapper_templates[0],
                wrapper_templates=wrapper_templates,
                descriptor_variants=descriptor_variants,
                use_chat_template=not bool(args.disable_memory_chat_template),
                chat_role=str(args.memory_role),
                keep_descriptor_only=bool(args.memory_descriptor_only),
                slot_pooling=str(args.slot_pooling),
                memory_variant_mode=str(args.memory_variant_mode),
            )
            payload = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "method_family": "canonical_memory",
                "implementation": {
                    "vllm_qwen3_moe": "hardmath_combined_bank_single_memory_v1",
                    "notes": (
                        "Fast vLLM smoke-test artifact. The HARDMath primary and auxiliary banks are "
                        "serialized as one combined canonical memory. The bundled vLLM "
                        "port does not reproduce separate auxiliary bank gains or per-layer rho maps."
                    ),
                },
                "model_name": str(args.model_name),
                "target_trait": "hardmath",
                "target_pole": "combined",
                "seed": int(args.seed),
                "selected_layer_ids": [int(layer_id) for layer_id in selected_layer_ids],
                "selected_layer_ids_by_score": [int(layer_id) for layer_id in selected_layer_ids_by_score],
                "selected_layer_head_map": {
                    str(layer_id): [int(head_id) for head_id in head_ids]
                    for layer_id, head_ids in selected_layer_head_map.items()
                },
                "selected_layer_kv_group_map": {
                    str(layer_id): [int(group_id) for group_id in group_ids]
                    for layer_id, group_ids in selected_layer_kv_group_map.items()
                },
                "num_attention_heads": int(num_attention_heads),
                "num_key_value_heads": int(num_key_value_heads),
                "kv_group_size": int(kv_group_size),
                "selector": {
                    "auto_select_heads": bool(args.auto_select_heads),
                    "selector_max_examples": int(args.selector_max_examples),
                    "selector_max_new_tokens": int(args.selector_max_new_tokens),
                    "selector_diagnostic_max_decode_steps": int(args.selector_diagnostic_max_decode_steps),
                    "selector_math_dataset_path": (
                        str(args.selector_math_dataset_path)
                        if getattr(args, "selector_math_dataset_path", None)
                        else None
                    ),
                    "selector_math_dataset_split": (
                        str(args.selector_math_dataset_split)
                        if getattr(args, "selector_math_dataset_split", None)
                        else None
                    ),
                    "selector_categories": (
                        list(args.selector_subjects)
                        if getattr(args, "selector_subjects", None)
                        else None
                    ),
                    "selector_example_ids": (
                        list(args.selector_example_ids)
                        if getattr(args, "selector_example_ids", None)
                        else None
                    ),
                    "selector_example_count_for_signature": int(len(selector_signature_examples)),
                    "selector_example_ids_for_signature": [
                        str(prepared.example.example_id)
                        for prepared in selector_signature_examples[: max(0, int(args.selector_max_examples))]
                    ],
                    "selector_record_prefill_diagnostics": bool(args.selector_record_prefill_diagnostics),
                    "selector_debug_top_heads": int(args.selector_debug_top_heads),
                    "max_heads_per_layer": int(args.max_heads_per_layer),
                    "max_layers": int(args.max_layers),
                    "layer_selection_metric": str(args.layer_selection_metric),
                    "head_score_count": int(len(selector_head_rows)),
                    "positive_head_score_count": int(
                        sum(1 for row in selector_head_rows if _safe_float(row.get("score")) > 0.0)
                    ),
                    "head_score_protocol": (
                        str(selector_head_rows[0].get("score_protocol"))
                        if selector_head_rows
                        else "none"
                    ),
                    "top_head_scores": [
                        {
                            "layer_id": int(row["layer_id"]),
                            "head_id": int(row["head_id"]),
                            "score": float(_safe_float(row.get("score"))),
                            "mean_physics_route_margin": float(
                                _safe_float(row.get("mean_physics_route_margin"))
                            ),
                            "mean_alignment_margin": float(_safe_float(row.get("mean_alignment_margin"))),
                            "mean_physics_bank_mass": float(_safe_float(row.get("mean_physics_bank_mass"))),
                            "mean_prompt_bank_mass": float(_safe_float(row.get("mean_prompt_bank_mass"))),
                        }
                        for row in sorted(
                            selector_head_rows,
                            key=lambda item: (
                                _safe_float(item.get("score")),
                                _safe_float(item.get("mean_physics_route_margin")),
                                _safe_float(item.get("mean_alignment_margin")),
                                -int(item.get("head_id", -1)),
                            ),
                            reverse=True,
                        )[: max(0, int(args.selector_debug_top_heads))]
                    ],
                    "layer_scores": selector_layer_rows,
                },
                "bank_signature": [
                    {"bank_id": str(bank_id), "gain": float(gain)}
                    for bank_id, gain in signature
                ],
                "mix_config": {
                    "rho": float(args.rho),
                    "positive_gain": float(args.positive_gain),
                    "negative_gain": 0.0,
                    "mix_mode": str(args.mix_mode),
                    "query_gate_scale": float(args.query_gate_scale),
                    "query_adaptive_gates": False,
                    "prompt_bank_normalization": str(args.prompt_bank_normalization),
                    "prefill_steering": str(args.prefill_steering),
                    "steering_operator": "bank_softmax_mixture",
                },
                "memory_config": {
                    "memory_role": str(args.memory_role),
                    "disable_memory_chat_template": bool(args.disable_memory_chat_template),
                    "memory_descriptor_only": bool(args.memory_descriptor_only),
                    "memory_variant_mode": str(args.memory_variant_mode),
                    "slot_pooling": str(args.slot_pooling),
                    "wrapper_templates": list(wrapper_templates),
                },
                "trait_memory": serialize_canonical_memory(memory),
                "reference_memory": None,
            }
            save_artifact(artifact_path, payload)
            print(f"[math_vllm] wrote canonical artifact: {artifact_path}", flush=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2.0)
    return artifact_paths


def _load_llm(
    *,
    args: argparse.Namespace,
    model_source: str,
    cache_dir: Optional[str],
    artifact_path: Optional[Path],
):
    # vLLM workers may be spawned after CUDA has been initialized by the HF
    # selector. Export the local plugin path and MoE fallbacks so spawned
    # workers import sitecustomize.py and patch missing local kernels too.
    _enable_vllm_port_plugin()
    os.environ.setdefault("VLLM_PORT_ENABLE_TOPK_SOFTMAX_FALLBACK", "1")
    os.environ.setdefault("VLLM_PORT_ENABLE_TORCH_MOE_FALLBACK", "1")
    os.environ.setdefault("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
    if artifact_path is None:
        os.environ.pop(ARTIFACT_ENV_VAR, None)
    else:
        os.environ[ARTIFACT_ENV_VAR] = str(artifact_path)
        register_intervention_model()

    patch_status = apply_vllm_moe_runtime_fallbacks()
    print(f"[vllm] runtime patch status: {json.dumps(patch_status, sort_keys=True)}", flush=True)
    if bool(patch_status.get("patched_topk_softmax")):
        print("[vllm] patched missing _moe_C.topk_softmax with torch fallback", flush=True)
    if bool(patch_status.get("patched_unquantized_fused_moe")):
        print("[vllm] patched unquantized fused MoE with sparse torch fallback", flush=True)

    from vllm import LLM

    llm_kwargs = {
        "model": model_source,
        "dtype": str(args.dtype),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_num_seqs": int(args.max_num_seqs),
        "trust_remote_code": True,
        "enforce_eager": bool(args.enforce_eager),
        "enable_prefix_caching": bool(args.enable_prefix_caching),
        "enable_chunked_prefill": bool(args.enable_chunked_prefill),
    }
    if cache_dir is not None:
        llm_kwargs["download_dir"] = cache_dir
    if int(args.max_model_len) > 0:
        llm_kwargs["max_model_len"] = int(args.max_model_len)
    if str(args.attention_backend or "").strip():
        llm_kwargs["attention_backend"] = str(args.attention_backend).strip()
    print(f"[vllm] loading {model_source} artifact={artifact_path}", flush=True)
    return LLM(**llm_kwargs)


def _release_llm(llm) -> None:
    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception as exc:
            print(f"[vllm-warning] engine shutdown raised {type(exc).__name__}: {exc}", flush=True)
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2.0)


def _prepare_examples(args: argparse.Namespace, examples: Sequence[PhysicsExample]) -> List[PreparedExample]:
    prepared: List[PreparedExample] = []
    for example in examples:
        routing_decision = route_subject(
            question_text=example.question,
            oracle_subject=example.subject,
            mode=str(args.routing_mode),
        )
        primary_bank_spec = _resolved_primary_bank_spec(
            example=example,
            routing_decision=routing_decision,
            args=args,
        )
        auxiliary_bank_specs = tuple(
            _build_auxiliary_bank_specs(
                example=example,
                routing_decision=routing_decision,
                primary_bank_spec=primary_bank_spec,
                args=args,
            )
        )
        prepared.append(
            PreparedExample(
                example=example,
                routing_decision=routing_decision,
                primary_bank_spec=primary_bank_spec,
                auxiliary_bank_specs=auxiliary_bank_specs,
            )
        )
    return prepared


def _response_key(method_tag: str, dataset_name: str, example_id: object) -> Tuple[str, str, str]:
    return (str(method_tag), str(dataset_name), str(example_id))


def _load_completed_response_keys(
    run_root: Path,
    *,
    method_specs: Sequence[Any],
    dataset_names: Sequence[str],
) -> set[Tuple[str, str, str]]:
    completed: set[Tuple[str, str, str]] = set()
    for method_spec in method_specs:
        for dataset_name in dataset_names:
            response_path = run_root / method_spec.tag / dataset_name / "response.jsonl"
            if not response_path.exists():
                continue
            with response_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        record = json.loads(text)
                    except json.JSONDecodeError as exc:
                        print(
                            f"[math_vllm-warning] ignoring malformed response row "
                            f"{response_path}:{line_number}: {exc}",
                            flush=True,
                        )
                        continue
                    example_id = record.get("id")
                    if example_id in (None, ""):
                        continue
                    completed.add(_response_key(method_spec.tag, dataset_name, example_id))
    return completed


def _open_response_handles_append(
    output_root: Path,
    *,
    method_specs: Sequence[Any],
    dataset_names: Sequence[str],
) -> Dict[Tuple[str, str], object]:
    handles: Dict[Tuple[str, str], object] = {}
    for method_spec in method_specs:
        for dataset_name in dataset_names:
            dataset_dir = output_root / method_spec.tag / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            handles[(method_spec.tag, dataset_name)] = (dataset_dir / "response.jsonl").open(
                "a",
                encoding="utf-8",
            )
    return handles


def _run_method_batches(
    *,
    args: argparse.Namespace,
    tokenizer,
    llm,
    prepared_examples: Sequence[PreparedExample],
    method_spec,
    response_handles: Mapping[Tuple[str, str], object],
    completed_response_keys: Optional[set[Tuple[str, str, str]]] = None,
    artifact_paths: Optional[Mapping[Tuple[Tuple[str, float], ...], Path]] = None,
) -> int:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature) if bool(args.do_sample) else 0.0,
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        seed=int(args.seed),
    )
    written = 0
    total = len(prepared_examples)
    pending: List[Tuple[int, PreparedExample, Dict[str, Any], str]] = []
    batch_size = max(1, min(int(args.batch_size), int(args.max_num_seqs)))

    def flush_pending() -> None:
        nonlocal written
        if not pending:
            return
        prompts = [formatted_prompt for _, _, _, formatted_prompt in pending]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        handles_to_sync = set()
        for (item_index, prepared_item, record, _), output in zip(pending, outputs):
            completion = output.outputs[0] if output is not None and output.outputs else None
            record["llm_answers"] = completion.text.strip() if completion is not None else ""
            finish_reason = getattr(completion, "finish_reason", None) if completion is not None else None
            stop_reason = getattr(completion, "stop_reason", None) if completion is not None else None
            token_ids = getattr(completion, "token_ids", None) if completion is not None else None
            output_token_count = len(token_ids or ())
            finish_key = str(finish_reason or stop_reason or "unknown")
            record["generation_metadata"] = {
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
                "output_token_count": int(output_token_count),
                "max_new_tokens": int(args.max_new_tokens),
                "hit_max_new_tokens": bool(
                    finish_reason == "length" or output_token_count >= int(args.max_new_tokens)
                ),
            }
            handle = response_handles[(record["method_tag"], record["dataset_name"])]
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            handles_to_sync.add(handle)
            if completed_response_keys is not None:
                completed_response_keys.add(
                    _response_key(method_spec.tag, prepared_item.example.dataset_name, prepared_item.example.example_id)
                )
            written += 1
            print(
                f"[math_vllm] {method_spec.method} example {item_index}/{total} saved: "
                f"finish_reason={finish_key} output_tokens={output_token_count}",
                flush=True,
            )
        for handle in handles_to_sync:
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        pending.clear()

    for index, prepared in enumerate(prepared_examples, start=1):
        key = _response_key(method_spec.tag, prepared.example.dataset_name, prepared.example.example_id)
        if completed_response_keys is not None and key in completed_response_keys:
            print(
                f"[math_vllm] {method_spec.method} example {index}/{total} "
                f"id={prepared.example.example_id} category={prepared.example.subject} skipped: "
                "existing response row",
                flush=True,
            )
            continue
        messages = _messages_for_method(
            method=method_spec.method,
            prepared=prepared,
            prompt_role=str(args.prompt_role),
            include_subject_name=bool(args.include_subject_name_in_problem),
        )
        formatted_prompt = _format_messages(tokenizer, messages)
        artifact_path = None
        if artifact_paths is not None:
            artifact_path = artifact_paths.get(prepared.bank_signature)
        record: Dict[str, Any] = {
            "id": prepared.example.example_id,
            "subject": prepared.example.subject,
            "subject_display": prepared.example.subject_display,
            "category": prepared.example.subject,
            "category_display": prepared.example.subject_display,
            "dataset_name": prepared.example.dataset_name,
            "split_name": prepared.example.split_name,
            "llm_answers": "",
            "method": method_spec.method,
            "method_tag": method_spec.tag,
            "routing_mode": prepared.routing_decision.mode,
            "routed_primary_subject": prepared.routing_decision.primary_subject,
            "routed_secondary_subject": prepared.routing_decision.secondary_subject,
            "subject_probabilities": _subject_probabilities_as_json(prepared.routing_decision),
            "routed_primary_category": prepared.routing_decision.primary_subject,
            "routed_secondary_category": prepared.routing_decision.secondary_subject,
            "category_probabilities": _subject_probabilities_as_json(prepared.routing_decision),
            "novelty": float(prepared.routing_decision.novelty),
            "selected_banks": _selected_bank_payload(
                prepared.primary_bank_spec,
                prepared.auxiliary_bank_specs,
            ),
            "formatted_prompt": formatted_prompt,
            "source_path": prepared.example.source_path,
            "engine": "vllm",
            "steering_artifact_path": str(artifact_path) if artifact_path is not None else None,
        }
        print(
            f"[math_vllm] {method_spec.method} example {index}/{total} "
            f"id={prepared.example.example_id} category={prepared.example.subject}",
            flush=True,
        )
        pending.append((index, prepared, record, formatted_prompt))
        if len(pending) >= batch_size:
            flush_pending()
    flush_pending()
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HARDMath benchmark smoke tests through the bundled vLLM intervention port."
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--qwen-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-architecture-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--methods", type=str, nargs="+", default=list(DEFAULT_METHODS), choices=list(DEFAULT_METHODS))
    parser.add_argument("--math-dataset-path", "--physics-dataset-path", dest="math_dataset_path", type=str, default=None)
    parser.add_argument("--math-dataset-split", "--physics-dataset-split", dest="math_dataset_split", type=str, default=None)
    parser.add_argument("--categories", "--subjects", dest="subjects", type=str, nargs="*", default=None)
    parser.add_argument("--example-ids", type=str, nargs="*", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--routing-mode", type=str, default="hybrid", choices=["oracle", "keyword", "hybrid"])
    parser.add_argument("--enable-novelty-aware-mixing", action="store_true")
    parser.add_argument("--secondary-subject-threshold", type=float, default=0.45)
    parser.add_argument("--secondary-subject-gain", type=float, default=1.5)
    parser.add_argument("--novelty-rho-scale", type=float, default=0.5)
    parser.add_argument("--planner-bank-gain", type=float, default=2.0)
    parser.add_argument("--math-hygiene-bank-gain", type=float, default=1.5)
    parser.add_argument("--checker-bank-gain", type=float, default=2.0)
    parser.add_argument("--completion-bank-gain", type=float, default=2.0)
    parser.add_argument("--novelty-checker-bonus", type=float, default=1.5)
    parser.add_argument("--hypothesis-bank-gain", type=float, default=1.0)
    parser.add_argument("--hypothesis-novelty-threshold", type=float, default=0.30)
    parser.add_argument("--max-hypothesis-banks", type=int, default=2)
    parser.add_argument("--always-use-hypothesis-banks", action="store_true")
    parser.add_argument("--disable-planner-bank", action="store_true")
    parser.add_argument("--disable-math-hygiene-bank", action="store_true")
    parser.add_argument("--disable-checker-bank", action="store_true")
    parser.add_argument("--disable-completion-bank", action="store_true")
    parser.add_argument("--disable-hypothesis-banks", action="store_true")
    parser.add_argument(
        "--bank-profile",
        type=str,
        default="heuristic_completion",
        choices=list(BANK_PROFILE_CHOICES),
        help=(
            "Memory-bank design to use: current, heuristic, or heuristic_completion."
        ),
    )
    parser.add_argument("--primary-bank", type=str, default="subject", choices=["subject", "planner", "math_hygiene", "checker"])
    parser.add_argument("--include-subject-name-in-problem", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--positive-gain", type=float, default=2.0)
    parser.add_argument("--mix-mode", type=str, default="convex", choices=["convex", "contrastive_neutral"])
    parser.add_argument("--prefill-steering", type=str, default="last_token_only", choices=["full", "last_token_only", "none"])
    parser.add_argument("--steering-operator", type=str, default="bank_softmax_mixture", choices=["bank_softmax_mixture"])
    parser.add_argument("--prompt-bank-normalization", type=str, default=DEFAULT_PROMPT_BANK_NORMALIZATION, choices=["none", "log_token_count"])
    parser.add_argument("--query-gate-scale", type=float, default=1.0)
    parser.add_argument("--layer-ids", type=int, nargs="+", default=list(DEFAULT_LAYER_IDS))
    parser.add_argument("--head-ids", type=int, nargs="*", default=[])
    parser.add_argument("--layer-head-ids", type=str, nargs="*", default=[], metavar="LAYER:HEAD[,HEAD...]")
    parser.add_argument("--layer-weight-schedule", type=str, default=DEFAULT_LAYER_WEIGHT_SCHEDULE)
    parser.add_argument("--auto-select-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selector-max-examples", type=int, default=16)
    parser.add_argument("--selector-max-new-tokens", type=int, default=96)
    parser.add_argument("--selector-diagnostic-max-decode-steps", type=int, default=96)
    parser.add_argument("--selector-debug-top-heads", type=int, default=20)
    parser.add_argument("--selector-record-prefill-diagnostics", action="store_true")
    parser.add_argument(
        "--selector-math-dataset-path",
        "--selector-physics-dataset-path",
        dest="selector_math_dataset_path",
        type=str,
        default=None,
        help=(
            "Optional HARDMath dataset path used only for auto head selection. "
            "Defaults to --math-dataset-path when selector examples are requested."
        ),
    )
    parser.add_argument(
        "--selector-math-dataset-split",
        "--selector-physics-dataset-split",
        dest="selector_math_dataset_split",
        type=str,
        default=None,
        help=(
            "Optional HARDMath split/file used only for auto head selection. "
            "When omitted, auto selection uses the evaluated examples, which is transductive."
        ),
    )
    parser.add_argument(
        "--selector-categories",
        "--selector-subjects",
        dest="selector_subjects",
        type=str,
        nargs="*",
        default=None,
        help="Optional category filter for selector examples. Defaults to --categories.",
    )
    parser.add_argument(
        "--selector-example-ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit selector example ids. These are excluded if they overlap evaluated ids.",
    )
    parser.add_argument("--max-heads-per-layer", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=5)
    parser.add_argument(
        "--layer-selection-metric",
        type=str,
        default="top_head_score_sum",
        choices=["top_head_score_sum", "top_head_score_mean", "positive_head_count"],
    )
    parser.add_argument("--memory-role", type=str, default="system", choices=["system", "user", "assistant"])
    parser.add_argument("--memory-variant-mode", type=str, default="full_plus_variants", choices=["variants_only", "full_plus_variants"])
    parser.add_argument("--slot-pooling", type=str, default="mean", choices=["mean", "concat"])
    parser.add_argument("--memory-descriptor-only", action="store_true")
    parser.add_argument("--disable-memory-chat-template", action="store_true")
    parser.add_argument("--prompt-role", type=str, default="system", choices=["system", "user", "assistant"])
    parser.add_argument("--canonical-wrapper-templates", type=str, nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attention-backend", type=str, default="", choices=["", "FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION"])
    parser.add_argument("--artifact-root", type=str, default=str(PACKAGE_ROOT / "runs" / "vllm_port_artifacts"))
    parser.add_argument("--rebuild-canonical-artifacts", action="store_true")
    parser.add_argument("--artifact-device-map", type=str, default="cuda:0")
    parser.add_argument("--artifact-attn-implementation", type=str, default="sdpa")
    parser.add_argument("--allow-eager-attention", action="store_true")
    parser.add_argument("--output-dir", type=str, default="qwen3_moe_math_outputs_vllm")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--disable-progress", action="store_true")
    args = parser.parse_args()
    try:
        args.layer_head_ids = _parse_layer_head_id_specs(args.layer_head_ids)
    except ValueError as exc:
        parser.error(str(exc))
    layer_id_set = {int(layer_id) for layer_id in args.layer_ids}
    unused_layer_head_ids = sorted(set(args.layer_head_ids) - layer_id_set)
    if unused_layer_head_ids:
        parser.error(
            "--layer-head-ids includes layers not present in --layer-ids: "
            + ", ".join(str(layer_id) for layer_id in unused_layer_head_ids)
        )
    return args


def main() -> None:
    args = _parse_args()
    if AutoTokenizer is None or AutoConfig is None:
        raise RuntimeError("transformers is required before running the vLLM HARDMath benchmark.")
    examples = load_math_examples(
        args.math_dataset_path,
        split_name=args.math_dataset_split,
        subjects=args.subjects,
        example_ids=args.example_ids,
        max_examples=args.max_examples,
    )
    if not examples:
        raise RuntimeError("No HARDMath examples were loaded. Check --math-dataset-path and --categories.")
    prepared_examples = _prepare_examples(args, examples)
    print(
        "[math_vllm] loaded examples: "
        f"count={len(prepared_examples)} "
        f"by_category={json.dumps(summarize_examples_by_subject(examples), sort_keys=True)}",
        flush=True,
    )
    selector_prepared_examples: Optional[List[PreparedExample]] = None
    selector_examples: List[PhysicsExample] = []
    if args.selector_math_dataset_path or args.selector_math_dataset_split or args.selector_example_ids:
        selector_dataset_path = args.selector_math_dataset_path or args.math_dataset_path
        selector_subjects = args.selector_subjects if args.selector_subjects is not None else args.subjects
        selector_examples = load_math_examples(
            selector_dataset_path,
            split_name=args.selector_math_dataset_split,
            subjects=selector_subjects,
            example_ids=args.selector_example_ids,
            max_examples=None,
        )
        evaluated_keys = {
            (str(example.example_id), str(example.source_path))
            for example in examples
        }
        evaluated_ids = {str(example.example_id) for example in examples}
        before_filter_count = len(selector_examples)
        selector_examples = [
            example
            for example in selector_examples
            if (str(example.example_id), str(example.source_path)) not in evaluated_keys
            and str(example.example_id) not in evaluated_ids
        ]
        filtered_count = before_filter_count - len(selector_examples)
        if not selector_examples:
            print(
                "[math_vllm-warning] no clean selector examples remain after filtering; "
                "auto head selection will use evaluated examples.",
                flush=True,
            )
        else:
            selector_prepared_examples = _prepare_examples(args, selector_examples)
            print(
                "[math_vllm] loaded clean selector examples: "
                f"count={len(selector_prepared_examples)} "
                f"filtered_overlap={filtered_count} "
                f"split={args.selector_math_dataset_split} "
                f"by_category={json.dumps(summarize_examples_by_subject(selector_examples), sort_keys=True)}",
                flush=True,
            )
    elif bool(args.auto_select_heads):
        print(
            "[math_vllm-warning] auto head selection will use evaluated examples. "
            "Set --selector-math-dataset-path to use clean held-out selector examples.",
            flush=True,
        )

    model_source, cache_dir, tokenizer_kwargs = _model_source_and_tokenizer_kwargs(str(args.model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
    wrap_qwen_tokenizer_apply_chat_template(tokenizer, enable_thinking=bool(args.qwen_enable_thinking))
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    method_specs = _method_specs(args)
    dataset_names = sorted({example.dataset_name for example in examples})
    output_root = Path(args.output_dir).expanduser().resolve()
    run_name = args.run_name or f"{_safe_slug(args.model_name)}_{_safe_slug(args.routing_mode)}_vllm"
    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"layer_ids", "head_ids", "layer_head_ids"}
    }
    run_config.update(
        {
            "engine": "vllm",
            "vllm_source_root": str(VLLM_PORT_ROOT),
            "canonical_memory_fidelity": "combined_bank_approximation",
            "canonical_memory_caveat": (
                "The bundled vLLM port does not yet implement separate HARDMath auxiliary "
                "bank gains or per-layer rho. Canonical-memory vLLM rows use one combined bank "
                "artifact per bank signature."
            ),
            "layer_ids": [int(layer_id) for layer_id in args.layer_ids],
            "head_ids": [int(head_id) for head_id in args.head_ids],
            "layer_head_ids": {
                str(layer_id): [int(head_id) for head_id in head_ids]
                for layer_id, head_ids in args.layer_head_ids.items()
            },
            "dataset_summary": summarize_examples_by_subject(examples),
            "category_summary": summarize_examples_by_subject(examples),
            "example_count": len(examples),
            "selector_dataset_summary": (
                summarize_examples_by_subject(selector_examples)
                if selector_prepared_examples is not None
                else None
            ),
            "selector_example_count": (
                len(selector_prepared_examples)
                if selector_prepared_examples is not None
                else None
            ),
            "selector_overlap_with_eval_excluded": (
                True if selector_prepared_examples is not None else None
            ),
        }
    )
    _json_dump(run_root / "run_config.json", run_config)

    requires_canonical = any(method_spec.method == "canonical_memory" for method_spec in method_specs)
    canonical_artifact_paths = (
        _build_math_canonical_artifacts(
            args=args,
            prepared_examples=prepared_examples,
            selector_prepared_examples=selector_prepared_examples,
        )
        if requires_canonical
        else {}
    )

    completed_response_keys = _load_completed_response_keys(
        run_root,
        method_specs=method_specs,
        dataset_names=dataset_names,
    )
    if completed_response_keys:
        print(
            f"[math_vllm] resume mode: found {len(completed_response_keys)} existing response rows",
            flush=True,
        )
    response_handles = _open_response_handles_append(
        run_root,
        method_specs=method_specs,
        dataset_names=dataset_names,
    )
    method_counts: Dict[str, int] = {method_spec.tag: 0 for method_spec in method_specs}
    try:
        noncanonical_specs = [spec for spec in method_specs if spec.method != "canonical_memory"]
        if noncanonical_specs:
            llm = _load_llm(args=args, model_source=model_source, cache_dir=cache_dir, artifact_path=None)
            try:
                for method_spec in noncanonical_specs:
                    method_counts[method_spec.tag] += _run_method_batches(
                        args=args,
                        tokenizer=tokenizer,
                        llm=llm,
                        prepared_examples=prepared_examples,
                        method_spec=method_spec,
                        response_handles=response_handles,
                        completed_response_keys=completed_response_keys,
                    )
            finally:
                _release_llm(llm)

        canonical_specs = [spec for spec in method_specs if spec.method == "canonical_memory"]
        if canonical_specs:
            grouped: Dict[Tuple[Tuple[str, float], ...], List[PreparedExample]] = {}
            for prepared in prepared_examples:
                grouped.setdefault(prepared.bank_signature, []).append(prepared)
            for signature, group in grouped.items():
                artifact_path = canonical_artifact_paths[signature]
                llm = _load_llm(args=args, model_source=model_source, cache_dir=cache_dir, artifact_path=artifact_path)
                try:
                    for method_spec in canonical_specs:
                        method_counts[method_spec.tag] += _run_method_batches(
                            args=args,
                            tokenizer=tokenizer,
                            llm=llm,
                            prepared_examples=group,
                            method_spec=method_spec,
                            response_handles=response_handles,
                            completed_response_keys=completed_response_keys,
                            artifact_paths=canonical_artifact_paths,
                        )
                finally:
                    _release_llm(llm)
    finally:
        _close_handles(response_handles)

    summary = {
        "run_root": str(run_root),
        "engine": "vllm",
        "method_counts": method_counts,
        "canonical_artifacts": {
            ",".join(bank_id for bank_id, _ in signature): str(path)
            for signature, path in canonical_artifact_paths.items()
        },
    }
    _json_dump(run_root / "summary.json", summary)
    try:
        from score_hardmath_outputs import score_run_root

        score_run_root(run_root)
    except Exception as exc:
        print(f"[math_vllm-warning] scoring failed: {exc}", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
