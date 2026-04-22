#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = REPO_ROOT.parent
for candidate in (PACKAGE_PARENT, REPO_ROOT, SCRIPT_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from dialogue_persistence_benchmark_michael import DIALOGUE_PERSISTENCE_EXPANDED, DialoguePersistenceCase
from long_context.judge import (
    DialoguePersistenceConversationJudgeResult,
    DialoguePersistenceTurnJudgeResult,
    judge_dialogue_persistence_conversation_async,
    judge_dialogue_persistence_turn_async,
    maybe_configure_openai_key,
)
from long_context.human_validation import export_human_validation_subset
from long_context.neutral_contexts import (
    build_context_bundle,
    default_assistant_register_context_manifest_path,
    default_assistant_register_context_text_dir,
    default_assistant_register_stylized_context_text_dir,
    default_neutral_context_manifest_path,
    default_neutral_context_text_dir,
    default_stylized_context_text_dir,
    read_downloaded_neutral_contexts,
    read_downloaded_stylized_contexts,
    read_prepared_assistant_register_contexts,
    read_prepared_assistant_register_stylized_contexts,
)
from long_context.style_track_a import (
    CanonicalDescriptorMemoryAttentionPatch,
    CanonicalDescriptorMixConfig,
    CompositePatch,
    DEFAULT_SELECTOR_CONTROL_AGGREGATION,
    DEFAULT_SELECTOR_CONTROL_WEIGHT,
    DEFAULT_SELECTOR_LAYER_IDS,
    DEFAULT_SELECTOR_QUERY_POSITION,
    DEFAULT_SELECTOR_TARGET_AGGREGATION,
    build_canonical_descriptor_memory,
    build_style_prompt_message,
    estimate_incremental_qwen_dialogue_tokens,
    generate_from_messages,
    generate_incremental_qwen_dialogue,
    layer_rhos,
    parse_layer_head_map,
    resolve_model_device,
    rows_to_csv,
    run_style_phrase_selector,
    supports_incremental_qwen_chat,
)
from long_context.style_traits import (
    DEFAULT_LONG_CONTEXT_TOKEN_BUDGETS,
    build_style_contrastive_examples,
    default_profile_json_path,
    resolve_selector_prompts,
    resolve_style_track_memory_bank,
    resolve_style_traits,
)
from qwen3_moe.modeling import load_model_and_tokenizer
from qwen3_moe.post_attn_caa_patch import HybridCanonicalCAAPatchConfig, HybridCanonicalMemoryCAAPatch

try:
    from hybrid_caa_canonical_debug.site_activation import build_site_activation_steering_vector_from_examples
except ImportError:  # pragma: no cover
    from site_activation import build_site_activation_steering_vector_from_examples


_SKELETON_SUFFIX = re.compile(r"_\d+_[ab]$")
_TRUNCATED_CASE_SUFFIX = re.compile(r"_first\d+$")
_LOGGER = logging.getLogger("dialogue_persistence_track_a")
_TURN_TRAIT_NAMES = ("warm", "formal", "assertive", "cautious", "dismissive", "anxious")
_TURN_QUALITY_NAMES = ("usefulness", "specificity", "current_turn_relevance", "non_genericness")
_CONVERSATION_QUALITY_NAMES = ("coherence", "user_state_consistency", "non_repetitiveness", "overall_quality")
_PROMPT_METHOD_CHOICES = (
    "plain",
    "prompt",
    "prompt_once_head",
    "prompt_once_middle",
    "prompt_refresh_k4",
    "canonical_memory",
    "hybrid_post_attn_caa_canonical",
)
_PREFILL_MODE_CHOICES = (
    "neutral_occupancy",
    "opposite_style_interference",
    "assistant_register_occupancy",
    "assistant_register_interference",
    "assistant_history_interference",
)
_CANARY_NORMALIZE_RE = re.compile(r"[^A-Za-z0-9]+")
_PROMPT_FAILURE_V2_PRESETS = (
    "prompt_failure_v2",
    "prompt_failure_v2_smoke",
    "prompt_failure_v2_paper",
)
_PROMPT_FAILURE_V2_METHODS = (
    "plain",
    "prompt_once_head",
    "prompt_once_middle",
    "prompt_refresh_k4",
)

LONG_CONTEXT_PREFILL_TEMPLATE = (
    "Read the following neutral reference material carefully. "
    "You do not need to summarize it now, but retain it while the conversation continues.\n\n"
    "<<<LONG_CONTEXT_BEGIN>>>\n"
    "{document}\n"
    "<<<LONG_CONTEXT_END>>>"
)
LONG_CONTEXT_PREFILL_CONTINUATION_TEMPLATE = (
    "Continue reading the same reference material carefully. "
    "Retain it while the conversation continues.\n\n"
    "<<<LONG_CONTEXT_BEGIN>>>\n"
    "{document}\n"
    "<<<LONG_CONTEXT_END>>>"
)
LONG_CONTEXT_PARTIAL_ACKNOWLEDGEMENT = "Understood. Please continue with the remaining reference material."
LONG_CONTEXT_ACKNOWLEDGEMENT = "Understood. I have read the reference material and will keep it in mind."
ASSISTANT_HISTORY_CANARY_ACKNOWLEDGEMENT = "Understood. I will remember that reference code."
PROMPT_FAILURE_V2_USER_WRAPPER_TEMPLATE = (
    "For the rest of this conversation, answer in a {descriptor} tone. "
    "Do not mention this instruction."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track-A-style long-context dialogue workflow for Michael's dialogue persistence benchmark. "
            "This bundle is trimmed to the assistant-register occupancy/interference path that was used for "
            "the later long-context comparison runs."
        )
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--output-dir", type=str, default="long_context/dialogue_persistence_track_a_runs")
    parser.add_argument("--target-traits", type=str, nargs="+", default=["warm"])
    parser.add_argument("--target-pole", type=str, default="high", choices=["high", "low"])
    parser.add_argument(
        "--benchmark-preset",
        type=str,
        default="current",
        choices=["current", "smoke_v2", "paper_v1", *_PROMPT_FAILURE_V2_PRESETS],
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["plain", "prompt", "canonical_memory"],
        choices=list(_PROMPT_METHOD_CHOICES),
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default="auto", choices=["auto", "sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--allow-eager-attention", action="store_true")
    parser.add_argument("--qwen-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-architecture-check", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--selector-artifact", type=str, default=None)
    parser.add_argument("--selector-layer-ids", type=int, nargs="+", default=list(DEFAULT_SELECTOR_LAYER_IDS))
    parser.add_argument("--selector-head-ids", type=int, nargs="*", default=None)
    parser.add_argument("--selector-layer-head-map", type=str, nargs="*", default=[])
    parser.add_argument("--selector-prompt-limit", type=int, default=10)
    parser.add_argument("--selector-query-position", type=str, default=DEFAULT_SELECTOR_QUERY_POSITION, choices=["last_user_token", "mean_last_4_user_tokens", "full_user_span", "last_token"])
    parser.add_argument("--selector-target-aggregation", type=str, default=DEFAULT_SELECTOR_TARGET_AGGREGATION, choices=["mean", "max", "top1", "top2", "top3"])
    parser.add_argument("--selector-control-aggregation", type=str, default=DEFAULT_SELECTOR_CONTROL_AGGREGATION, choices=["mean", "max", "top1", "top2", "top3"])
    parser.add_argument("--selector-control-weight", type=float, default=DEFAULT_SELECTOR_CONTROL_WEIGHT)
    parser.add_argument("--selector-max-prompt-tokens", type=int, default=0)
    parser.add_argument("--selector-diagnostic-max-decode-steps", type=int, default=24)
    parser.add_argument("--selector-max-new-tokens", type=int, default=96)
    parser.add_argument("--selector-route-blend-alpha", type=float, default=0.0)
    parser.add_argument("--max-heads-per-layer", type=int, default=5)
    parser.add_argument("--max-layers", type=int, default=6)
    parser.add_argument("--layer-selection-metric", type=str, default="top_head_score_sum", choices=["top_head_score_sum", "top_head_score_mean", "positive_head_count"])

    parser.add_argument("--rho", type=float, default=8.0)
    parser.add_argument("--hybrid-alpha", type=float, default=1.5)
    parser.add_argument("--layer-weight-schedule", type=str, default="middle_heavy")
    parser.add_argument("--mix-mode", type=str, default="contrastive_neutral", choices=["convex", "contrastive_neutral", "contrastive_opposite"])
    parser.add_argument("--prefill-steering", type=str, default="last_token_only", choices=["full", "last_token_only", "none"])
    parser.add_argument("--steering-operator", type=str, default="bank_softmax_mixture", choices=["key_logit_sparse", "output_mixture", "bank_softmax_mixture"])
    parser.add_argument("--positive-gain", type=float, default=4.0)
    parser.add_argument("--negative-gain", type=float, default=0.1)
    parser.add_argument("--query-gate-scale", type=float, default=1.0)
    parser.add_argument("--prompt-bank-normalization", type=str, default="log_token_count", choices=["none", "log_token_count"])
    parser.add_argument("--memory-role", type=str, default="system", choices=["system", "user", "assistant"])
    parser.add_argument("--disable-memory-chat-template", action="store_true")
    parser.add_argument("--disable-memory-descriptor-only", action="store_true")
    parser.add_argument("--disable-query-adaptive-gates", action="store_true")
    parser.add_argument("--diagnostic-max-decode-steps", type=int, default=0)
    parser.add_argument("--canonical-wrapper-templates", type=str, nargs="*", default=None)
    parser.add_argument("--track-memory-bank-profile", type=str, default="default")
    parser.add_argument("--track-memory-bank-profile-file", type=str, default=str(default_profile_json_path()))
    parser.add_argument("--prompt-role", type=str, default="system", choices=["system", "user", "assistant"])
    parser.add_argument("--prompt-wrapper-template", type=str, default=None)

    parser.add_argument("--hybrid-site", type=str, default="post_attn_resid", choices=["layer_input", "post_attn_resid", "mlp_input"])
    parser.add_argument("--hybrid-token-selector", type=str, default="last_completion_token", choices=["last_completion_token", "first_completion_token", "mean_completion"])
    parser.add_argument("--hybrid-example-limit", type=int, default=10)

    parser.add_argument("--long-context-token-budgets", type=int, nargs="+", default=list(DEFAULT_LONG_CONTEXT_TOKEN_BUDGETS))
    parser.add_argument("--prefill-mode", type=str, default="assistant_register_interference", choices=list(_PREFILL_MODE_CHOICES))
    parser.add_argument("--case-sampling", type=str, default="in_order", choices=["in_order", "stratified"])
    parser.add_argument("--stylized-context-text-dir", type=str, default=str(default_stylized_context_text_dir()))
    parser.add_argument("--neutral-context-manifest", type=str, default=str(default_neutral_context_manifest_path()))
    parser.add_argument("--neutral-context-text-dir", type=str, default=str(default_neutral_context_text_dir()))
    parser.add_argument("--neutral-context-source-ids", type=str, nargs="*", default=None)
    parser.add_argument("--assistant-context-manifest", type=str, default=str(default_assistant_register_context_manifest_path()))
    parser.add_argument("--assistant-context-text-dir", type=str, default=str(default_assistant_register_context_text_dir()))
    parser.add_argument(
        "--assistant-stylized-context-text-dir",
        type=str,
        default=str(default_assistant_register_stylized_context_text_dir()),
    )
    parser.add_argument("--prompt-middle-fraction", type=float, default=0.5)
    parser.add_argument("--prompt-refresh-every-k", type=int, default=4)
    parser.add_argument("--enable-canary-probe", action="store_true")
    parser.add_argument("--canary-position-fraction", type=float, default=0.5)
    parser.add_argument("--context-underfill-tolerance", type=float, default=0.25)
    parser.add_argument("--human-validation-export", type=str, default=None)

    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--skeleton-limit", type=int, default=None)
    parser.add_argument("--skeleton-ids", type=str, nargs="*", default=None)
    parser.add_argument("--case-ids", type=str, nargs="*", default=None)
    parser.add_argument("--buckets", type=str, nargs="*", default=None)
    parser.add_argument("--max-turns", type=int, default=None)

    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-turn-judge", action="store_true")
    parser.add_argument("--conversation-judge-repeats", type=int, default=1)
    parser.add_argument("--conversation-judge-all-traits", action="store_true")
    parser.add_argument("--conversation-transcript-max-chars", type=int, default=None)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-concurrent-judge", type=int, default=12)
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--judge-debug", action="store_true")
    parser.add_argument("--context-window-safety-margin", type=int, default=512)
    parser.add_argument("--disable-context-window-preflight", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    return parser.parse_args()


def _skeleton_id_from_case_id(case_id: str) -> str:
    normalized = _TRUNCATED_CASE_SUFFIX.sub("", str(case_id).strip())
    return _SKELETON_SUFFIX.sub("", normalized)


def _format_transcript(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n\n".join(f"{str(message['role']).upper()}: {str(message['content'])}" for message in messages)


def _apply_benchmark_preset(args: argparse.Namespace) -> None:
    preset = str(args.benchmark_preset).strip().lower()
    if preset == "current":
        return
    if preset == "smoke_v2":
        args.case_sampling = "stratified"
        if args.case_limit is None:
            args.case_limit = 6
        return
    if preset == "paper_v1":
        args.case_sampling = "stratified"
        args.enable_canary_probe = True
        args.context_underfill_tolerance = min(float(args.context_underfill_tolerance), 0.05)
        return
    if preset in _PROMPT_FAILURE_V2_PRESETS:
        args.case_sampling = "stratified"
        args.prefill_mode = "assistant_history_interference"
        args.prompt_role = "user"
        args.methods = list(_PROMPT_FAILURE_V2_METHODS)
        args.enable_canary_probe = True
        args.context_underfill_tolerance = min(float(args.context_underfill_tolerance), 0.05)
        if not str(args.prompt_wrapper_template or "").strip():
            args.prompt_wrapper_template = PROMPT_FAILURE_V2_USER_WRAPPER_TEMPLATE
        if preset == "prompt_failure_v2_smoke":
            args.case_limit = 6 if args.case_limit is None else int(args.case_limit)
            args.long_context_token_budgets = [8000]
            return
        args.long_context_token_budgets = [8000, 16000, 24000]
        return
    raise ValueError(f"Unsupported benchmark preset '{args.benchmark_preset}'.")


def _normalize_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized == "prompt":
        return "prompt_once_head"
    return normalized


def _method_family(method: str) -> str:
    normalized = _normalize_method(method)
    if normalized.startswith("prompt_"):
        return "prompt"
    if normalized == "plain":
        return "plain"
    if normalized == "canonical_memory":
        return "canonical_memory"
    if normalized == "hybrid_post_attn_caa_canonical":
        return "hybrid_post_attn_caa_canonical"
    return normalized


def _prompt_delivery_for_method(method: str) -> str:
    normalized = _normalize_method(method)
    if normalized == "prompt_once_head":
        return "once_head"
    if normalized == "prompt_once_middle":
        return "once_middle"
    if normalized == "prompt_refresh_k4":
        return "refresh_k4"
    return "none"


def _opposite_pole(pole: str) -> str:
    return "low" if str(pole).strip().lower() == "high" else "high"


def _is_prompt_failure_v2_preset(preset: str) -> bool:
    return str(preset).strip().lower() in _PROMPT_FAILURE_V2_PRESETS


def _uses_history_prefill(prefill_mode: str) -> bool:
    return str(prefill_mode).strip().lower() == "assistant_history_interference"


def _enforce_strict_underfill(args: argparse.Namespace) -> bool:
    return _is_prompt_failure_v2_preset(str(args.benchmark_preset)) or _uses_history_prefill(str(args.prefill_mode))


def _context_asset_family_for_prefill_mode(prefill_mode: str) -> str:
    normalized = str(prefill_mode).strip().lower()
    if normalized == "neutral_occupancy":
        return "neutral"
    if normalized == "opposite_style_interference":
        return "stylized"
    if normalized == "assistant_register_occupancy":
        return "assistant_register"
    if normalized == "assistant_register_interference":
        return "assistant_register_stylized"
    if normalized == "assistant_history_interference":
        return "assistant_history_stylized"
    raise ValueError(f"Unsupported prefill mode '{prefill_mode}'.")


def _load_context_records_for_prefill_mode(
    *,
    args: argparse.Namespace,
    trait_name: str,
) -> List[Dict[str, str]]:
    normalized = str(args.prefill_mode).strip().lower()
    if normalized == "neutral_occupancy":
        return read_downloaded_neutral_contexts(
            manifest_path=args.neutral_context_manifest,
            text_dir=args.neutral_context_text_dir,
            source_ids=args.neutral_context_source_ids,
        )
    if normalized == "opposite_style_interference":
        return read_downloaded_stylized_contexts(
            trait_name=str(trait_name),
            pole=_opposite_pole(str(args.target_pole)),
            manifest_path=args.neutral_context_manifest,
            text_dir=args.stylized_context_text_dir,
            source_ids=args.neutral_context_source_ids,
        )
    if normalized == "assistant_register_occupancy":
        return read_prepared_assistant_register_contexts(
            manifest_path=args.assistant_context_manifest,
            text_dir=args.assistant_context_text_dir,
        )
    if normalized == "assistant_register_interference":
        return read_prepared_assistant_register_stylized_contexts(
            trait_name=str(trait_name),
            pole=_opposite_pole(str(args.target_pole)),
            manifest_path=args.assistant_context_manifest,
            text_dir=args.assistant_stylized_context_text_dir,
        )
    if normalized == "assistant_history_interference":
        return read_prepared_assistant_register_stylized_contexts(
            trait_name=str(trait_name),
            pole=_opposite_pole(str(args.target_pole)),
            manifest_path=args.assistant_context_manifest,
            text_dir=args.assistant_stylized_context_text_dir,
        )
    raise ValueError(f"Unsupported prefill mode '{args.prefill_mode}'.")


def _order_cases_for_sampling(
    cases: Sequence[DialoguePersistenceCase],
    *,
    case_sampling: str,
) -> List[DialoguePersistenceCase]:
    if str(case_sampling).strip().lower() != "stratified":
        return list(cases)
    by_length: Dict[int, List[DialoguePersistenceCase]] = {}
    for case in cases:
        by_length.setdefault(len(case.turns), []).append(case)
    ordered_lengths = sorted(by_length)
    indices = {length: 0 for length in ordered_lengths}
    out: List[DialoguePersistenceCase] = []
    while True:
        progress = False
        for length in ordered_lengths:
            index = indices[length]
            group = by_length[length]
            if index >= len(group):
                continue
            out.append(group[index])
            indices[length] = index + 1
            progress = True
        if not progress:
            break
    return out


def _select_cases(args: argparse.Namespace) -> Tuple[DialoguePersistenceCase, ...]:
    cases = list(DIALOGUE_PERSISTENCE_EXPANDED)
    if args.buckets:
        allow = {str(bucket).strip().lower() for bucket in args.buckets}
        cases = [case for case in cases if str(case.bucket).strip().lower() in allow]
    if args.case_ids:
        allow_case_ids = {str(case_id).strip() for case_id in args.case_ids}
        cases = [case for case in cases if case.case_id in allow_case_ids]
    if args.skeleton_ids:
        allow_skeletons = {str(skeleton_id).strip() for skeleton_id in args.skeleton_ids if str(skeleton_id).strip()}
        cases = [case for case in cases if _skeleton_id_from_case_id(case.case_id) in allow_skeletons]
    elif args.skeleton_limit is not None:
        allowed_skeletons: set[str] = set()
        filtered: List[DialoguePersistenceCase] = []
        for case in cases:
            skeleton_id = _skeleton_id_from_case_id(case.case_id)
            if skeleton_id not in allowed_skeletons:
                if len(allowed_skeletons) >= max(0, int(args.skeleton_limit)):
                    continue
                allowed_skeletons.add(skeleton_id)
            filtered.append(case)
        cases = filtered
    cases = _order_cases_for_sampling(cases, case_sampling=str(args.case_sampling))
    if args.case_limit is not None:
        cases = cases[: max(0, int(args.case_limit))]
    max_turns = getattr(args, "max_turns", None)
    if max_turns is not None:
        limited_cases: List[DialoguePersistenceCase] = []
        max_turns_value = max(1, int(max_turns))
        for case in cases:
            if len(case.turns) <= max_turns_value:
                limited_cases.append(case)
                continue
            limited_cases.append(
                DialoguePersistenceCase(
                    case_id=f"{case.case_id}_first{max_turns_value}",
                    bucket=str(case.bucket),
                    title=f"{case.title} [first {max_turns_value} turns]",
                    system_preamble=str(case.system_preamble),
                    turns=tuple(case.turns[:max_turns_value]),
                )
            )
        cases = limited_cases
    return tuple(cases)


def _context_prefill_messages(
    context_text: str,
    *,
    continuation: bool = False,
    acknowledgement_text: str | None = None,
) -> List[Dict[str, str]]:
    if not str(context_text).strip():
        return []
    template = LONG_CONTEXT_PREFILL_CONTINUATION_TEMPLATE if continuation else LONG_CONTEXT_PREFILL_TEMPLATE
    return [
        {
            "role": "user",
            "content": template.format(document=str(context_text).strip()),
        },
        {"role": "assistant", "content": str(acknowledgement_text or LONG_CONTEXT_ACKNOWLEDGEMENT)},
    ]


def _token_count_for_messages(tokenizer, messages: Sequence[Mapping[str, str]]) -> int:
    materialized = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    if not materialized:
        return 0
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            rendered = tokenizer.apply_chat_template(
                materialized,
                tokenize=False,
                add_generation_prompt=False,
            )
            return int(len(tokenizer(str(rendered), add_special_tokens=False)["input_ids"]))
        except Exception:
            pass
    parts = [f"[{str(message['role']).upper()}]\n{str(message['content'])}" for message in materialized]
    return int(len(tokenizer("\n\n".join(parts), add_special_tokens=False)["input_ids"]))


def _truncate_text_to_token_budget(tokenizer, text: str, token_budget: int) -> str:
    budget = max(0, int(token_budget))
    if budget <= 0:
        return ""
    token_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids[:budget], skip_special_tokens=True).strip()


def _select_prefill_record_indices(record_count: int, target_count: int) -> List[int]:
    total = max(0, int(record_count))
    want = min(total, max(0, int(target_count)))
    if want <= 0:
        return []
    if want >= total:
        return list(range(total))
    if want == 1:
        return [0]
    chosen: List[int] = []
    for slot in range(want):
        index = int(round(slot * (total - 1) / max(1, want - 1)))
        if index not in chosen:
            chosen.append(index)
    for index in range(total):
        if len(chosen) >= want:
            break
        if index not in chosen:
            chosen.append(index)
    return sorted(chosen[:want])


def _split_prefill_messages_at_fraction(
    tokenizer,
    messages: Sequence[Mapping[str, str]],
    fraction: float,
    *,
    pair_aligned: bool = False,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], float]:
    materialized = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    if not materialized:
        return [], [], 0.0
    total_tokens = max(1, _token_count_for_messages(tokenizer, materialized))
    if pair_aligned:
        candidate_indices = [index for index in range(0, len(materialized) + 1) if index == 0 or index == len(materialized) or index % 2 == 0]
    else:
        candidate_indices = list(range(0, len(materialized) + 1))
    if len(candidate_indices) == 1:
        index = candidate_indices[0]
        left = materialized[:index]
        return left, materialized[index:], float(_token_count_for_messages(tokenizer, left) / total_tokens)

    target_tokens = min(max(float(fraction), 0.0), 1.0) * float(total_tokens)
    best_index = candidate_indices[0]
    best_distance = None
    for index in candidate_indices:
        left = materialized[:index]
        left_tokens = _token_count_for_messages(tokenizer, left)
        distance = abs(float(left_tokens) - target_tokens)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = index
    left = materialized[:best_index]
    return left, materialized[best_index:], float(_token_count_for_messages(tokenizer, left) / total_tokens)


def _build_assistant_history_pair(record: Mapping[str, str], *, assistant_text: str) -> List[Dict[str, str]]:
    user_prompt = str(record.get("topic_prompt", "")).strip()
    if not user_prompt:
        user_prompt = f"Please help with {str(record.get('title', 'this task')).strip()}."
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": str(assistant_text).strip()},
    ]


def _build_assistant_history_interference_bundle(
    *,
    tokenizer,
    records: Sequence[Dict[str, str]],
    token_budget: int,
) -> Dict[str, object]:
    budget = max(0, int(token_budget))
    if budget == 0 or not records:
        return {
            "token_budget": budget,
            "actual_token_count": 0,
            "context_text": "",
            "prefill_format": "message_history",
            "prefill_messages": [],
            "used_source_ids": [],
            "used_titles": [],
            "source_token_allocations": {},
        }

    usable_records = [dict(record) for record in records if str(record.get("text", "")).strip()]
    if not usable_records:
        return {
            "token_budget": budget,
            "actual_token_count": 0,
            "context_text": "",
            "prefill_format": "message_history",
            "prefill_messages": [],
            "used_source_ids": [],
            "used_titles": [],
            "source_token_allocations": {},
        }

    target_source_count = max(1, min(len(usable_records), max(1, (budget + 1999) // 2000)))
    primary_indices = _select_prefill_record_indices(len(usable_records), target_source_count)
    ordered_indices = list(primary_indices) + [index for index in range(len(usable_records)) if index not in set(primary_indices)]

    prefill_messages: List[Dict[str, str]] = []
    used_source_ids: List[str] = []
    used_titles: List[str] = []
    source_token_allocations: Dict[str, int] = {}

    for index in ordered_indices:
        record = usable_records[int(index)]
        full_pair = _build_assistant_history_pair(record, assistant_text=str(record["text"]).strip())
        full_messages = [*prefill_messages, *full_pair]
        full_count = _token_count_for_messages(tokenizer, full_messages)
        if full_count <= budget:
            prefill_messages = full_messages
            source_id = str(record["source_id"])
            used_source_ids.append(source_id)
            used_titles.append(str(record.get("title", source_id)))
            source_token_allocations[source_id] = int(len(tokenizer(str(record["text"]).strip(), add_special_tokens=False)["input_ids"]))
            continue

        user_message = {"role": "user", "content": str(full_pair[0]["content"])}
        with_user_messages = [*prefill_messages, user_message]
        with_user_count = _token_count_for_messages(tokenizer, with_user_messages)
        if with_user_count >= budget:
            break
        assistant_empty_count = _token_count_for_messages(tokenizer, [*with_user_messages, {"role": "assistant", "content": ""}])
        remaining_for_assistant = max(0, budget - assistant_empty_count)
        partial_assistant_text = _truncate_text_to_token_budget(tokenizer, str(record["text"]).strip(), remaining_for_assistant)
        if not partial_assistant_text:
            break
        partial_pair = _build_assistant_history_pair(record, assistant_text=partial_assistant_text)
        partial_messages = [*prefill_messages, *partial_pair]
        partial_count = _token_count_for_messages(tokenizer, partial_messages)
        if partial_count > budget:
            overshoot = max(0, partial_count - budget)
            partial_assistant_ids = tokenizer(str(partial_assistant_text), add_special_tokens=False)["input_ids"]
            if overshoot >= len(partial_assistant_ids):
                break
            partial_assistant_text = tokenizer.decode(partial_assistant_ids[: max(1, len(partial_assistant_ids) - overshoot)], skip_special_tokens=True).strip()
            if not partial_assistant_text:
                break
            partial_pair = _build_assistant_history_pair(record, assistant_text=partial_assistant_text)
            partial_messages = [*prefill_messages, *partial_pair]
            partial_count = _token_count_for_messages(tokenizer, partial_messages)
            if partial_count > budget:
                break
        prefill_messages = partial_messages
        source_id = str(record["source_id"])
        used_source_ids.append(source_id)
        used_titles.append(str(record.get("title", source_id)))
        source_token_allocations[source_id] = int(len(tokenizer(str(partial_assistant_text), add_special_tokens=False)["input_ids"]))
        break

    actual_token_count = _token_count_for_messages(tokenizer, prefill_messages)
    return {
        "token_budget": budget,
        "actual_token_count": int(actual_token_count),
        "context_text": "",
        "prefill_format": "message_history",
        "prefill_messages": prefill_messages,
        "used_source_ids": used_source_ids,
        "used_titles": used_titles,
        "source_token_allocations": source_token_allocations,
    }


def _insert_canary_into_prefill_messages(
    *,
    tokenizer,
    prefill_messages: Sequence[Mapping[str, str]],
    canary_block: str,
    position_fraction: float,
) -> Tuple[List[Dict[str, str]], int]:
    canary_messages = [
        {
            "role": "user",
            "content": (
                "Please remember the following reference code for later verification.\n\n"
                f"{str(canary_block).strip()}"
            ),
        },
        {"role": "assistant", "content": ASSISTANT_HISTORY_CANARY_ACKNOWLEDGEMENT},
    ]
    left, right, _ = _split_prefill_messages_at_fraction(
        tokenizer,
        prefill_messages,
        position_fraction,
        pair_aligned=True,
    )
    merged = [*left, *canary_messages, *right]
    return merged, _token_count_for_messages(tokenizer, merged)


def _split_text_at_fraction(tokenizer, text: str, fraction: float) -> Tuple[str, str]:
    raw = str(text).strip()
    if not raw:
        return "", ""
    token_ids = tokenizer(raw, add_special_tokens=False)["input_ids"]
    if not token_ids:
        return raw, ""
    clamped = min(max(float(fraction), 0.0), 1.0)
    split_index = min(max(1, int(round(len(token_ids) * clamped))), max(1, len(token_ids) - 1))
    left = tokenizer.decode(token_ids[:split_index], skip_special_tokens=True).strip()
    right = tokenizer.decode(token_ids[split_index:], skip_special_tokens=True).strip()
    return left, right


def _build_canary_spec(
    *,
    trait_name: str,
    budget: int,
    seed: int,
    source_ids: Sequence[str],
    position_fraction: float,
) -> Dict[str, Any]:
    digest = hashlib.sha1(
        f"{trait_name}|{budget}|{seed}|{'|'.join(str(source_id) for source_id in source_ids)}".encode("utf-8")
    ).hexdigest()[:10].upper()
    code = f"CTX{digest}"
    block = (
        "Archive verification note for reference retention.\n"
        f"The exact verification code is {code}.\n"
        f"If asked later for the verification code, reply with {code} exactly."
    )
    probe_user_message = (
        "Before we wrap up, answer with only the exact verification code from the reference material. "
        "Do not add any other words."
    )
    return {
        "code": code,
        "block": block,
        "probe_user_message": probe_user_message,
        "position_fraction": min(max(float(position_fraction), 0.0), 1.0),
    }


def _insert_canary_into_context_text(
    *,
    tokenizer,
    context_text: str,
    canary_block: str,
    position_fraction: float,
) -> Tuple[str, int]:
    raw = str(context_text).strip()
    if not raw:
        merged = str(canary_block).strip()
        actual_token_count = len(tokenizer(merged, add_special_tokens=False)["input_ids"]) if merged else 0
        return merged, int(actual_token_count)
    left, right = _split_text_at_fraction(tokenizer, raw, position_fraction)
    merged = "\n\n".join(part for part in (left, str(canary_block).strip(), right) if str(part).strip()).strip()
    actual_token_count = len(tokenizer(merged, add_special_tokens=False)["input_ids"]) if merged else 0
    return merged, int(actual_token_count)


def _normalize_canary_answer(text: str) -> str:
    return _CANARY_NORMALIZE_RE.sub("", str(text).strip()).upper()


def _build_prompt_delivery_plan(
    *,
    tokenizer,
    case: DialoguePersistenceCase,
    prompt_message: Optional[Mapping[str, str]],
    prompt_delivery: str,
    context_text: str,
    prefill_messages: Optional[Sequence[Mapping[str, str]]] = None,
    prefill_split_mode: str = "text",
    prompt_middle_fraction: float,
    prompt_refresh_every_k: int,
    include_probe_turn: bool,
) -> Dict[str, Any]:
    initial_messages: List[Dict[str, str]] = []
    prompt_insertions: List[Dict[str, Any]] = []
    total_turns = len(case.turns) + (1 if include_probe_turn else 0)
    turn_prefix_messages: List[List[Dict[str, str]]] = [[] for _ in range(total_turns)]
    effective_delivery = str(prompt_delivery)
    materialized_prefill_messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in (prefill_messages or ())
        if str(message.get("content", "")).strip()
    ]

    def _record_insertion(*, phase: str, turn_index: int | None = None, position_fraction: float | None = None) -> None:
        prompt_insertions.append(
            {
                "phase": phase,
                "turn_index": "" if turn_index is None else int(turn_index),
                "position_fraction": "" if position_fraction is None else float(position_fraction),
                "role": "" if prompt_message is None else str(prompt_message.get("role", "")),
            }
        )

    if prompt_message is None or prompt_delivery == "none":
        initial_messages.append({"role": "system", "content": str(case.system_preamble)})
        if materialized_prefill_messages:
            initial_messages.extend(materialized_prefill_messages)
        else:
            initial_messages.extend(_context_prefill_messages(context_text))
        effective_delivery = "none"
    elif prompt_delivery == "once_head":
        initial_messages.append({"role": "system", "content": str(case.system_preamble)})
        if str(prompt_message.get("role", "")).strip().lower() == "system":
            initial_messages.insert(0, {"role": str(prompt_message["role"]), "content": str(prompt_message["content"])})
        else:
            initial_messages.append({"role": str(prompt_message["role"]), "content": str(prompt_message["content"])})
        _record_insertion(phase="initial", position_fraction=0.0)
        if materialized_prefill_messages:
            initial_messages.extend(materialized_prefill_messages)
        else:
            initial_messages.extend(_context_prefill_messages(context_text))
    elif prompt_delivery == "once_middle":
        initial_messages.append({"role": "system", "content": str(case.system_preamble)})
        if str(prefill_split_mode).strip().lower() == "messages" and materialized_prefill_messages:
            left_messages, right_messages, position_fraction = _split_prefill_messages_at_fraction(
                tokenizer,
                materialized_prefill_messages,
                prompt_middle_fraction,
                pair_aligned=True,
            )
            if left_messages or right_messages:
                initial_messages.extend(left_messages)
                initial_messages.append({"role": str(prompt_message["role"]), "content": str(prompt_message["content"])})
                _record_insertion(phase="initial", position_fraction=position_fraction)
                initial_messages.extend(right_messages)
            else:
                effective_delivery = "once_after_system_fallback_no_prefill"
                initial_messages.append({"role": str(prompt_message["role"]), "content": str(prompt_message["content"])})
        else:
            left, right = _split_text_at_fraction(tokenizer, context_text, prompt_middle_fraction)
            if left and right:
                initial_messages.extend(
                    _context_prefill_messages(
                        left,
                        acknowledgement_text=LONG_CONTEXT_PARTIAL_ACKNOWLEDGEMENT,
                    )
                )
                initial_messages.append({"role": str(prompt_message["role"]), "content": str(prompt_message["content"])})
                _record_insertion(phase="initial", position_fraction=min(max(float(prompt_middle_fraction), 0.0), 1.0))
                initial_messages.extend(
                    _context_prefill_messages(
                        right,
                        continuation=True,
                        acknowledgement_text=LONG_CONTEXT_ACKNOWLEDGEMENT,
                    )
                )
            else:
                effective_delivery = "once_after_system_fallback_no_prefill"
                initial_messages.append({"role": str(prompt_message["role"]), "content": str(prompt_message["content"])})
                _record_insertion(phase="initial", position_fraction=0.0)
                initial_messages.extend(_context_prefill_messages(context_text))
    elif prompt_delivery == "refresh_k4":
        interval = max(1, int(prompt_refresh_every_k))
        initial_messages.append({"role": "system", "content": str(case.system_preamble)})
        if str(prompt_message.get("role", "")).strip().lower() == "system":
            initial_messages.insert(0, {"role": str(prompt_message["role"]), "content": str(prompt_message["content"])})
        else:
            initial_messages.append({"role": str(prompt_message["role"]), "content": str(prompt_message["content"])})
        _record_insertion(phase="initial", position_fraction=0.0)
        if materialized_prefill_messages:
            initial_messages.extend(materialized_prefill_messages)
        else:
            initial_messages.extend(_context_prefill_messages(context_text))
        for turn_index in range(interval, total_turns, interval):
            turn_prefix_messages[turn_index].append({"role": str(prompt_message["role"]), "content": str(prompt_message["content"])})
            _record_insertion(phase="turn_prefix", turn_index=turn_index)
    else:
        raise ValueError(f"Unsupported prompt delivery '{prompt_delivery}'.")

    prompt_message_count = sum(1 for message in initial_messages if message.get("content") == (prompt_message or {}).get("content"))
    prompt_message_count += sum(len(messages) for messages in turn_prefix_messages)
    return {
        "initial_messages": initial_messages,
        "turn_prefix_messages": turn_prefix_messages,
        "prompt_insertions": prompt_insertions,
        "prompt_message_count": int(prompt_message_count),
        "prompt_delivery_effective": effective_delivery,
    }


def _method_config_tag(method: str, args: argparse.Namespace) -> str:
    normalized = _normalize_method(method)
    if normalized in {"plain", "prompt_once_head", "prompt_once_middle", "prompt_refresh_k4"}:
        return normalized
    if normalized == "canonical_memory":
        return (
            f"canonical_memory_rho{float(args.rho):.2f}_pg{float(args.positive_gain):.2f}_"
            f"ng{float(args.negative_gain):.2f}_s{int(args.seed)}"
        )
    if normalized == "hybrid_post_attn_caa_canonical":
        return (
            f"hybrid_post_attn_caa_canonical_rho{float(args.rho):.2f}_a{float(args.hybrid_alpha):.2f}_"
            f"pg{float(args.positive_gain):.2f}_ng{float(args.negative_gain):.2f}_s{int(args.seed)}"
        )
    raise ValueError(f"Unsupported method '{method}'.")


def _selector_dir_for_trait(output_root: Path, trait_name: str, seed: int) -> Path:
    return output_root / trait_name / "selectors" / "head" / f"{trait_name}_style_selector_seed{int(seed)}"


def _load_selector_artifact(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: selector artifact must be a JSON object.")
    return payload


@contextlib.contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_csv_rows_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _summary_merge_key(row: Mapping[str, Any]) -> Tuple[str, int, str, str, str]:
    return (
        str(row.get("target_trait", "")),
        int(row.get("context_token_budget", 0) or 0),
        str(row.get("prefill_mode", "")),
        str(row.get("method", "")),
        str(row.get("config_tag", "")),
    )


def _merge_summary_rows(existing_path: Path, new_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {}
    for row in _load_csv_rows_if_exists(existing_path):
        merged[_summary_merge_key(row)] = dict(row)
    for row in new_rows:
        merged[_summary_merge_key(row)] = dict(row)
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("target_trait", "")),
            int(row.get("context_token_budget", 0) or 0),
            str(row.get("prefill_mode", "")),
            str(row.get("method", "")),
            str(row.get("config_tag", "")),
        ),
    )


def _selected_layer_head_map_from_artifact(payload: Mapping[str, Any]) -> Dict[int, Tuple[int, ...]]:
    raw_map = payload.get("selected_layer_head_map") or {}
    return {
        int(layer_id): tuple(int(head_id) for head_id in head_ids)
        for layer_id, head_ids in dict(raw_map).items()
    }


def _canonical_wrapper_templates(
    active_wrapper_templates: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[str, ...]:
    if args.canonical_wrapper_templates:
        return tuple(str(template) for template in args.canonical_wrapper_templates if str(template).strip())
    return tuple(str(template) for template in active_wrapper_templates if str(template).strip())


def _build_style_patch(
    *,
    model,
    tokenizer,
    trait,
    method: str,
    selected_layer_head_map: Mapping[int, Sequence[int]],
    args: argparse.Namespace,
    caches: Dict[str, Dict[Tuple[Any, ...], object]],
):
    normalized_method = _normalize_method(method)
    if normalized_method in {"plain", "prompt_once_head", "prompt_once_middle", "prompt_refresh_k4"}:
        return None

    layer_ids = tuple(sorted(int(layer_id) for layer_id in selected_layer_head_map))
    if not layer_ids:
        raise ValueError("Cannot build steering patch without selected layers.")

    active_bank = resolve_style_track_memory_bank(
        trait,
        track="generation",
        pole=str(args.target_pole),
        bank_profile=str(args.track_memory_bank_profile),
        bank_profile_file=getattr(args, "track_memory_bank_profile_file", None),
    )
    reference_pole = "low" if str(args.target_pole).strip().lower() == "high" else "high"
    reference_bank = resolve_style_track_memory_bank(
        trait,
        track="generation",
        pole=reference_pole,
        bank_profile=str(args.track_memory_bank_profile),
        bank_profile_file=getattr(args, "track_memory_bank_profile_file", None),
    )
    wrapper_templates = _canonical_wrapper_templates(active_bank.wrapper_templates, args)
    wrapper_template = wrapper_templates[0]
    use_chat_template = not bool(args.disable_memory_chat_template)
    keep_descriptor_only = not bool(args.disable_memory_descriptor_only)
    layer_rho_map = layer_rhos(layer_ids, float(args.rho), str(args.layer_weight_schedule))

    memory_key = (
        str(trait.name),
        "generation",
        str(args.target_pole),
        layer_ids,
        wrapper_templates,
        bool(use_chat_template),
        str(args.memory_role),
        bool(keep_descriptor_only),
    )
    memory = caches.setdefault("memory", {}).get(memory_key)
    if memory is None:
        memory = build_canonical_descriptor_memory(
            model,
            tokenizer,
            trait_name=f"{trait.name}_generation_{args.target_pole}",
            descriptor=active_bank.descriptor,
            layer_ids=layer_ids,
            wrapper_template=wrapper_template,
            wrapper_templates=wrapper_templates,
            descriptor_variants=active_bank.descriptor_variants,
            use_chat_template=use_chat_template,
            chat_role=str(args.memory_role),
            keep_descriptor_only=keep_descriptor_only,
        )
        caches["memory"][memory_key] = memory

    reference_key = (
        str(trait.name),
        "generation",
        reference_pole,
        layer_ids,
        wrapper_templates,
        bool(use_chat_template),
        str(args.memory_role),
        bool(keep_descriptor_only),
    )
    reference_memory = caches.setdefault("memory", {}).get(reference_key)
    if reference_memory is None:
        reference_memory = build_canonical_descriptor_memory(
            model,
            tokenizer,
            trait_name=f"{trait.name}_generation_{reference_pole}",
            descriptor=reference_bank.descriptor,
            layer_ids=layer_ids,
            wrapper_template=wrapper_template,
            wrapper_templates=wrapper_templates,
            descriptor_variants=reference_bank.descriptor_variants,
            use_chat_template=use_chat_template,
            chat_role=str(args.memory_role),
            keep_descriptor_only=keep_descriptor_only,
        )
        caches["memory"][reference_key] = reference_memory

    if normalized_method == "canonical_memory":
        patches = []
        for layer_id in layer_ids:
            patches.append(
                CanonicalDescriptorMemoryAttentionPatch(
                    layer_ids=[int(layer_id)],
                    memory=memory,
                    reference_memory=reference_memory,
                    config=CanonicalDescriptorMixConfig(
                        rho=float(layer_rho_map[int(layer_id)]),
                        mix_mode=str(args.mix_mode),
                        prefill_steering=str(args.prefill_steering),
                        steering_operator=str(args.steering_operator),
                        positive_gain=float(args.positive_gain),
                        negative_gain=float(args.negative_gain),
                        query_adaptive_gates=not bool(args.disable_query_adaptive_gates),
                        query_gate_scale=float(args.query_gate_scale),
                        prompt_bank_normalization=str(args.prompt_bank_normalization),
                        diagnostic_max_decode_steps=int(args.diagnostic_max_decode_steps),
                        record_prefill_diagnostics=False,
                        head_ids=tuple(int(head_id) for head_id in selected_layer_head_map[int(layer_id)]),
                    ),
                )
            )
        return CompositePatch(patches)

    if normalized_method != "hybrid_post_attn_caa_canonical":
        raise ValueError(f"Unsupported steering method '{method}'.")

    vector_key = (
        str(trait.name),
        str(args.target_pole),
        layer_ids,
        str(args.hybrid_site),
        str(args.hybrid_token_selector),
        int(args.hybrid_example_limit),
    )
    steering_vector = caches.setdefault("vector", {}).get(vector_key)
    if steering_vector is None:
        examples = build_style_contrastive_examples(trait, limit=int(args.hybrid_example_limit))
        if str(args.target_pole).strip().lower() == "low":
            examples = [
                {
                    **example,
                    "positive_response": str(example["negative_response"]),
                    "negative_response": str(example["positive_response"]),
                }
                for example in examples
            ]
        steering_vector = build_site_activation_steering_vector_from_examples(
            model,
            tokenizer,
            trait_name=str(trait.name),
            descriptor=active_bank.descriptor,
            opposite_descriptor=reference_bank.descriptor,
            layer_ids=layer_ids,
            examples=examples,
            site=str(args.hybrid_site),
            token_selector=str(args.hybrid_token_selector),
            normalize_vector=True,
        )
        caches["vector"][vector_key] = steering_vector

    alpha_map = layer_rhos(layer_ids, float(args.hybrid_alpha), str(args.layer_weight_schedule))
    patches = []
    for layer_id in layer_ids:
        patches.append(
            HybridCanonicalMemoryCAAPatch(
                layer_ids=[int(layer_id)],
                memory=memory,
                reference_memory=reference_memory,
                steering_vector=steering_vector,
                config=HybridCanonicalCAAPatchConfig(
                    rho=float(layer_rho_map[int(layer_id)]),
                    alpha=float(alpha_map[int(layer_id)]),
                    mix_mode=str(args.mix_mode),
                    prefill_steering=str(args.prefill_steering),
                    steering_operator=str(args.steering_operator),
                    positive_gain=float(args.positive_gain),
                    negative_gain=float(args.negative_gain),
                    query_adaptive_gates=not bool(args.disable_query_adaptive_gates),
                    query_gate_scale=float(args.query_gate_scale),
                    prompt_bank_normalization=str(args.prompt_bank_normalization),
                    diagnostic_max_decode_steps=int(args.diagnostic_max_decode_steps),
                    record_prefill_diagnostics=False,
                    head_ids=tuple(int(head_id) for head_id in selected_layer_head_map[int(layer_id)]),
                ),
            )
        )
    return CompositePatch(patches)


def _generate_dialogue_case(
    *,
    model,
    tokenizer,
    case: DialoguePersistenceCase,
    args: argparse.Namespace,
    initial_messages: Sequence[Mapping[str, str]],
    turn_prefix_messages: Sequence[Sequence[Mapping[str, str]]],
    patch,
    probe_user_message: str | None = None,
) -> Tuple[List[str], str, str]:
    runtime_messages: List[Dict[str, str]] = [dict(message) for message in initial_messages]
    all_user_turns = [str(turn.user_message) for turn in case.turns]
    if probe_user_message:
        all_user_turns.append(str(probe_user_message))

    transcript_messages: List[Dict[str, str]] = [{"role": "system", "content": str(case.system_preamble)}]
    probe_response = ""
    if supports_incremental_qwen_chat(model, tokenizer):
        assistant_turns_all = generate_incremental_qwen_dialogue(
            model,
            tokenizer,
            initial_messages=runtime_messages,
            user_turns=all_user_turns,
            turn_prefix_messages=turn_prefix_messages,
            max_new_tokens=int(args.max_new_tokens),
            seed=int(args.seed),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            patch=patch,
        )
        assistant_turns = list(assistant_turns_all[: len(case.turns)])
        if probe_user_message and len(assistant_turns_all) > len(case.turns):
            probe_response = str(assistant_turns_all[len(case.turns)])
        for turn, assistant_text in zip(case.turns, assistant_turns):
            transcript_messages.append({"role": "user", "content": str(turn.user_message)})
            transcript_messages.append({"role": "assistant", "content": assistant_text})
    else:
        assistant_turns = []
        for turn_index, user_text in enumerate(all_user_turns):
            for prefix_message in turn_prefix_messages[turn_index]:
                runtime_messages.append({"role": str(prefix_message["role"]), "content": str(prefix_message["content"])})
            user_message = {"role": "user", "content": str(user_text)}
            runtime_messages.append(user_message)
            assistant_text, _ = generate_from_messages(
                model,
                tokenizer,
                messages=runtime_messages,
                max_new_tokens=int(args.max_new_tokens),
                seed=int(args.seed),
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
                patch=patch,
            )
            runtime_messages.append({"role": "assistant", "content": assistant_text})
            if turn_index < len(case.turns):
                transcript_messages.append({"role": "user", "content": str(case.turns[turn_index].user_message)})
                transcript_messages.append({"role": "assistant", "content": assistant_text})
                assistant_turns.append(assistant_text)
            else:
                probe_response = assistant_text
    return assistant_turns, _format_transcript(transcript_messages), probe_response


def _estimate_case_total_tokens(
    *,
    tokenizer,
    initial_messages: Sequence[Mapping[str, str]],
    user_turns: Sequence[str],
    turn_prefix_messages: Sequence[Sequence[Mapping[str, str]]],
    max_new_tokens: int,
) -> Optional[int]:
    if not initial_messages:
        return None
    return estimate_incremental_qwen_dialogue_tokens(
        tokenizer,
        initial_messages=initial_messages,
        user_turns=[str(user_turn) for user_turn in user_turns],
        turn_prefix_messages=turn_prefix_messages,
        max_new_tokens=int(max_new_tokens),
        reserve_for_forced_close=True,
    )


def _turn_judgments_to_json(
    case: DialoguePersistenceCase,
    results: Sequence[Optional[DialoguePersistenceTurnJudgeResult]],
    *,
    turn_judge_skipped: bool = False,
) -> str:
    payload: List[Dict[str, Any]] = []
    for turn, result in zip(case.turns, results):
        entry: Dict[str, Any] = {"turn_id": turn.turn_id}
        if turn_judge_skipped:
            entry["judge_ok"] = False
            entry["skipped"] = True
        elif result is None:
            entry["judge_ok"] = False
        else:
            entry["judge_ok"] = True
            entry.update(asdict(result))
        payload.append(entry)
    return json.dumps(payload, ensure_ascii=False)


def _apply_conversation_scores_from_repeats(
    row: Dict[str, Any],
    results: Sequence[Optional[DialoguePersistenceConversationJudgeResult]],
    *,
    scored_keys: Sequence[str],
) -> None:
    ok = [result for result in results if result is not None]
    row["conversation_judge_n_ok"] = len(ok)
    row["conversation_judge_ok"] = len(ok) > 0
    for key in scored_keys:
        row[key] = ""
        row[f"{key}_var"] = ""
    if not ok:
        return
    for key in scored_keys:
        values = []
        for result in ok:
            if key not in result.scores:
                row["conversation_judge_ok"] = False
                return
            values.append(float(result.scores[key]))
        mean_value = sum(values) / len(values)
        row[key] = mean_value
        row[f"{key}_var"] = sum((value - mean_value) ** 2 for value in values) / len(values) if len(values) >= 2 else ""


async def _judge_one_case(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    case: DialoguePersistenceCase,
    assistant_turns: Sequence[str],
    transcript: str,
    transcript_max_chars: int | None,
    skip_turn_judge: bool,
    judge_debug: bool,
    conversation_judge_repeats: int,
    conversation_persistence_traits: Sequence[str] | None,
) -> Tuple[List[Optional[DialoguePersistenceTurnJudgeResult]], List[Optional[DialoguePersistenceConversationJudgeResult]]]:
    if skip_turn_judge:
        turn_results = [None] * len(case.turns)
    else:
        tasks = [
            judge_dialogue_persistence_turn_async(
                client=client,
                semaphore=semaphore,
                judge_model=judge_model,
                system_preamble=case.system_preamble,
                user_message=turn.user_message,
                assistant_message=assistant_turns[index],
                log_context=f"{case.case_id} {turn.turn_id}",
                judge_debug=judge_debug,
            )
            for index, turn in enumerate(case.turns)
        ]
        turn_results = list(await asyncio.gather(*tasks))
    conversation_results: List[Optional[DialoguePersistenceConversationJudgeResult]] = []
    for repeat_index in range(int(conversation_judge_repeats)):
        conversation_results.append(
            await judge_dialogue_persistence_conversation_async(
                client=client,
                semaphore=semaphore,
                judge_model=judge_model,
                transcript=transcript,
                transcript_max_chars=transcript_max_chars,
                case_id=case.case_id if int(conversation_judge_repeats) <= 1 else f"{case.case_id} repeat={repeat_index + 1}",
                judge_debug=judge_debug,
                conversation_persistence_traits=conversation_persistence_traits,
            )
        )
    return turn_results, conversation_results


async def _judge_all_cases(
    *,
    items: Sequence[Tuple[DialoguePersistenceCase, List[str], str]],
    judge_model: str,
    max_concurrent: int,
    transcript_max_chars: int | None,
    skip_turn_judge: bool,
    judge_debug: bool,
    conversation_judge_repeats: int,
    conversation_persistence_traits: Sequence[str] | None,
) -> List[Tuple[List[Optional[DialoguePersistenceTurnJudgeResult]], List[Optional[DialoguePersistenceConversationJudgeResult]]]]:
    try:
        from openai import AsyncOpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for judging.") from exc

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(int(max_concurrent))
    try:
        tasks = [
            _judge_one_case(
                client=client,
                semaphore=semaphore,
                judge_model=judge_model,
                case=case,
                assistant_turns=assistant_turns,
                transcript=transcript,
                transcript_max_chars=transcript_max_chars,
                skip_turn_judge=skip_turn_judge,
                judge_debug=judge_debug,
                conversation_judge_repeats=conversation_judge_repeats,
                conversation_persistence_traits=conversation_persistence_traits,
            )
            for case, assistant_turns, transcript in items
        ]
        return list(await asyncio.gather(*tasks))
    finally:
        await client.close()


def _flatten_turn_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    target_trait: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        raw_json = str(row.get("turn_judgments_json", "") or "").strip()
        if not raw_json:
            continue
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list):
            continue
        num_turns = int(row.get("num_turns", 0) or 0)
        for turn_index, turn_payload in enumerate(payload, start=1):
            if not isinstance(turn_payload, dict):
                continue
            target_value = turn_payload.get(target_trait)
            out.append(
                {
                    "target_trait": str(target_trait),
                    "method": row.get("method", ""),
                    "config_tag": row.get("config_tag", ""),
                    "context_token_budget": row.get("context_token_budget", ""),
                    "context_actual_tokens": row.get("context_actual_tokens", ""),
                    "case_id": row.get("case_id", ""),
                    "num_turns": num_turns,
                    "turn_id": turn_payload.get("turn_id", ""),
                    "turn_index": int(turn_index),
                    "turn_fraction": float(turn_index / max(1, num_turns)) if num_turns else "",
                    "judge_ok": bool(turn_payload.get("judge_ok", False)),
                    "target_trait_score": "" if target_value in (None, "") else float(target_value),
                    "warm": turn_payload.get("warm", ""),
                    "formal": turn_payload.get("formal", ""),
                    "assertive": turn_payload.get("assertive", ""),
                    "cautious": turn_payload.get("cautious", ""),
                    "dismissive": turn_payload.get("dismissive", ""),
                    "anxious": turn_payload.get("anxious", ""),
                    "usefulness": turn_payload.get("usefulness", ""),
                    "specificity": turn_payload.get("specificity", ""),
                    "current_turn_relevance": turn_payload.get("current_turn_relevance", ""),
                    "non_genericness": turn_payload.get("non_genericness", ""),
                }
            )
    return out


def _mean(values: Sequence[float]) -> str | float:
    numeric = [float(value) for value in values]
    if not numeric:
        return ""
    return float(sum(numeric) / len(numeric))


def _aggregate_run_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    turn_rows: Sequence[Mapping[str, Any]],
    target_trait: str,
) -> Dict[str, Any]:
    target_key = f"persistence_{target_trait}"
    persistence_values = [float(row[target_key]) for row in rows if row.get(target_key) not in ("", None)]
    conversation_quality_values: Dict[str, List[float]] = {
        key: [float(row[key]) for row in rows if row.get(key) not in ("", None)]
        for key in _CONVERSATION_QUALITY_NAMES
    }
    turn_values = [float(row["target_trait_score"]) for row in turn_rows if row.get("judge_ok") and row.get("target_trait_score") not in ("", None)]
    turn_quality_values: Dict[str, List[float]] = {
        key: [float(row[key]) for row in turn_rows if row.get("judge_ok") and row.get(key) not in ("", None)]
        for key in _TURN_QUALITY_NAMES
    }
    first_quarter = [float(row["target_trait_score"]) for row in turn_rows if row.get("judge_ok") and row.get("turn_fraction") not in ("", None) and float(row["turn_fraction"]) <= 0.25 and row.get("target_trait_score") not in ("", None)]
    last_quarter = [float(row["target_trait_score"]) for row in turn_rows if row.get("judge_ok") and row.get("turn_fraction") not in ("", None) and float(row["turn_fraction"]) > 0.75 and row.get("target_trait_score") not in ("", None)]
    first_half = [float(row["target_trait_score"]) for row in turn_rows if row.get("judge_ok") and row.get("turn_fraction") not in ("", None) and float(row["turn_fraction"]) <= 0.5 and row.get("target_trait_score") not in ("", None)]
    second_half = [float(row["target_trait_score"]) for row in turn_rows if row.get("judge_ok") and row.get("turn_fraction") not in ("", None) and float(row["turn_fraction"]) > 0.5 and row.get("target_trait_score") not in ("", None)]
    conversation_ok_rate = float(sum(1 for row in rows if row.get("conversation_judge_ok")) / max(1, len(rows)))
    turn_ok_count = sum(1 for row in turn_rows if row.get("judge_ok"))
    turn_ok_rate = float(turn_ok_count / max(1, len(turn_rows))) if turn_rows else ""
    canary_rows = [row for row in rows if row.get("canary_expected_answer", "") not in ("", None)]
    canary_accuracy_values = [1.0 if bool(row.get("probe_correct", False)) else 0.0 for row in canary_rows]
    first_quarter_mean = _mean(first_quarter)
    last_quarter_mean = _mean(last_quarter)
    if first_quarter_mean == "" or last_quarter_mean == "":
        last_minus_first = ""
    else:
        last_minus_first = float(last_quarter_mean) - float(first_quarter_mean)
    aggregate = {
        "case_count": int(len(rows)),
        "turn_row_count": int(len(turn_rows)),
        "conversation_judge_ok_rate": conversation_ok_rate,
        "turn_judge_ok_rate": turn_ok_rate,
        f"mean_{target_key}": _mean(persistence_values),
        "mean_turn_target_score": _mean(turn_values),
        "mean_turn_target_score_first_half": _mean(first_half),
        "mean_turn_target_score_second_half": _mean(second_half),
        "mean_turn_target_score_first_quarter": first_quarter_mean,
        "mean_turn_target_score_last_quarter": last_quarter_mean,
        "last_quarter_minus_first_quarter": last_minus_first,
        "canary_probe_accuracy": _mean(canary_accuracy_values),
    }
    for key, values in conversation_quality_values.items():
        aggregate[f"mean_{key}"] = _mean(values)
    for key, values in turn_quality_values.items():
        aggregate[f"mean_turn_{key}"] = _mean(values)
    return aggregate


def _write_run_outputs(
    *,
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    turn_rows: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
) -> None:
    rows_to_csv(run_dir / "dialogue_persistence_rows.csv", rows)
    rows_to_csv(run_dir / "dialogue_persistence_turn_rows.csv", turn_rows)
    summary_rows: List[Dict[str, Any]] = []
    for row in rows:
        summary_row = {
            "case_id": row.get("case_id", ""),
            "skeleton_id": row.get("skeleton_id", ""),
            "bucket": row.get("bucket", ""),
            "title": row.get("title", ""),
            "method": row.get("method", ""),
            "config_tag": row.get("config_tag", ""),
            "benchmark_variant": row.get("benchmark_variant", ""),
            "prefill_mode": row.get("prefill_mode", ""),
            "prefill_format": row.get("prefill_format", ""),
            "context_asset_family": row.get("context_asset_family", ""),
            "prompt_role": row.get("prompt_role", ""),
            "prompt_delivery": row.get("prompt_delivery", ""),
            "context_token_budget": row.get("context_token_budget", ""),
            "context_actual_tokens": row.get("context_actual_tokens", ""),
            "conversation_judge_ok": row.get("conversation_judge_ok", ""),
            "conversation_judge_n_ok": row.get("conversation_judge_n_ok", ""),
            "coherence": row.get("coherence", ""),
            "user_state_consistency": row.get("user_state_consistency", ""),
            "non_repetitiveness": row.get("non_repetitiveness", ""),
            "overall_quality": row.get("overall_quality", ""),
            "probe_correct": row.get("probe_correct", ""),
            "canary_expected_answer": row.get("canary_expected_answer", ""),
        }
        for key, value in row.items():
            if str(key).startswith("persistence_") or str(key).endswith("_var"):
                summary_row[str(key)] = value
        summary_rows.append(summary_row)
    rows_to_csv(run_dir / "dialogue_persistence_summary.csv", summary_rows)
    (run_dir / "run_config.json").write_text(json.dumps(dict(run_config), indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _apply_benchmark_preset(args)
    if int(args.conversation_judge_repeats) < 1:
        raise SystemExit("--conversation-judge-repeats must be >= 1")
    traits = resolve_style_traits(args.target_traits)
    if args.selector_artifact and len(traits) != 1:
        raise SystemExit("--selector-artifact can only be used with exactly one --target-traits value.")

    if not args.skip_judge:
        maybe_configure_openai_key(args.openai_api_key)
        if bool(args.judge_debug):
            logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        allow_eager_fallback=bool(args.allow_eager_attention),
        enable_thinking=bool(args.qwen_enable_thinking),
        strict_architecture_check=bool(args.strict_architecture_check),
    )
    device = resolve_model_device(model)
    model_context_window = int(getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0)
    print(f"[model] loaded {args.model_name} on {device}", flush=True)

    cases = _select_cases(args)
    if not cases:
        raise SystemExit("No dialogue persistence cases selected.")
    print(f"[cases] selected {len(cases)} cases", flush=True)

    budgets = sorted({max(0, int(value)) for value in args.long_context_token_budgets})

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    overall_summary_rows: List[Dict[str, Any]] = []
    caches: Dict[str, Dict[Tuple[Any, ...], object]] = {}

    for trait in traits:
        trait_root = output_root / trait.name
        trait_root.mkdir(parents=True, exist_ok=True)

        if args.selector_artifact:
            selector_artifact_path = Path(args.selector_artifact).expanduser().resolve()
            selector_artifact = _load_selector_artifact(selector_artifact_path)
        else:
            selector_dir = _selector_dir_for_trait(output_root, trait.name, args.seed)
            selector_dir.mkdir(parents=True, exist_ok=True)
            selector_artifact_path = selector_dir / "selector_artifact.json"
            selector_lock_path = selector_dir / ".selector_artifact.lock"
            with _exclusive_file_lock(selector_lock_path):
                if selector_artifact_path.exists():
                    selector_artifact = _load_selector_artifact(selector_artifact_path)
                else:
                    prompt_cases = resolve_selector_prompts(limit=int(args.selector_prompt_limit))
                    selector_result = run_style_phrase_selector(
                        model=model,
                        tokenizer=tokenizer,
                        trait=trait,
                        prompt_cases=prompt_cases,
                        target_pole=str(args.target_pole),
                        layer_ids=args.selector_layer_ids,
                        head_ids=args.selector_head_ids,
                        layer_head_map=parse_layer_head_map(args.selector_layer_head_map),
                        max_heads_per_layer=int(args.max_heads_per_layer),
                        max_layers=int(args.max_layers),
                        layer_selection_metric=str(args.layer_selection_metric),
                        rho=float(args.rho),
                        layer_weight_schedule=str(args.layer_weight_schedule),
                        query_position=str(args.selector_query_position),
                        target_aggregation=str(args.selector_target_aggregation),
                        control_aggregation=str(args.selector_control_aggregation),
                        control_weight=float(args.selector_control_weight),
                        max_prompt_tokens=int(args.selector_max_prompt_tokens),
                        seed=int(args.seed),
                        max_new_tokens=int(args.selector_max_new_tokens),
                        do_sample=bool(args.do_sample),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        repetition_penalty=float(args.repetition_penalty),
                        bank_profile=str(args.track_memory_bank_profile),
                        bank_profile_file=getattr(args, "track_memory_bank_profile_file", None),
                        wrapper_templates=args.canonical_wrapper_templates,
                        memory_role=str(args.memory_role),
                        use_chat_template=not bool(args.disable_memory_chat_template),
                        keep_descriptor_only=not bool(args.disable_memory_descriptor_only),
                        mix_mode=str(args.mix_mode),
                        prefill_steering=str(args.prefill_steering),
                        steering_operator=str(args.steering_operator),
                        positive_gain=float(args.positive_gain),
                        negative_gain=float(args.negative_gain),
                        query_adaptive_gates=not bool(args.disable_query_adaptive_gates),
                        query_gate_scale=float(args.query_gate_scale),
                        prompt_bank_normalization=str(args.prompt_bank_normalization),
                        diagnostic_max_decode_steps=int(args.selector_diagnostic_max_decode_steps),
                        selector_route_blend_alpha=float(args.selector_route_blend_alpha),
                    )
                    rows_to_csv(selector_dir / "selector_prompt_rows.csv", selector_result["full_rows"])
                    rows_to_csv(selector_dir / "trace_rows.csv", selector_result["trace_rows"])
                    rows_to_csv(selector_dir / "head_scores.csv", selector_result["head_rows"])
                    rows_to_csv(selector_dir / "layer_scores.csv", selector_result["layer_rows"])
                    selector_artifact = dict(selector_result["artifact"])
                    selector_artifact_path.write_text(json.dumps(selector_artifact, indent=2) + "\n", encoding="utf-8")
                print(
                    f"[selector] trait={trait.name} selected_layers={selector_artifact.get('selected_layer_ids_by_score', [])} "
                    f"artifact={selector_artifact_path}",
                    flush=True,
                )

        selected_layer_head_map = _selected_layer_head_map_from_artifact(selector_artifact)
        if not selected_layer_head_map:
            raise RuntimeError(f"Selector artifact for trait '{trait.name}' did not contain any selected heads.")
        selected_layer_ids = sorted(selected_layer_head_map)

        context_asset_family = _context_asset_family_for_prefill_mode(str(args.prefill_mode))
        context_records: List[Dict[str, str]] = []
        if any(int(budget) > 0 for budget in budgets):
            context_records = _load_context_records_for_prefill_mode(
                args=args,
                trait_name=str(trait.name),
            )
            if not context_records:
                if str(args.prefill_mode) == "neutral_occupancy":
                    raise SystemExit(
                        "No neutral context texts were found. Run `python long_context/download_neutral_contexts.py` first, "
                        "or point --neutral-context-text-dir at a populated directory."
                    )
                if str(args.prefill_mode) == "opposite_style_interference":
                    raise SystemExit(
                        "No stylized context texts were found for the requested trait/pole. "
                        "Prepare them first with the stylized-context asset builder or point --stylized-context-text-dir at a populated directory."
                    )
                if str(args.prefill_mode) == "assistant_register_occupancy":
                    raise SystemExit(
                        "No assistant-register context texts were found. "
                        "Prepare them first with `python long_context/prepare_assistant_register_contexts.py --skip-stylized` "
                        "or point --assistant-context-text-dir at a populated directory."
                    )
                raise SystemExit(
                    "No assistant-register stylized context texts were found for the requested trait/pole. "
                    "Prepare them first with `python long_context/prepare_assistant_register_contexts.py` "
                    "or point --assistant-stylized-context-text-dir at a populated directory."
                )

        context_bundles: Dict[int, Dict[str, Any]] = {}
        for budget in budgets:
            if _uses_history_prefill(str(args.prefill_mode)):
                bundle = dict(
                    _build_assistant_history_interference_bundle(
                        tokenizer=tokenizer,
                        records=context_records,
                        token_budget=int(budget),
                    )
                )
            else:
                bundle = dict(build_context_bundle(tokenizer=tokenizer, records=context_records, token_budget=int(budget)))
                bundle["prefill_format"] = "quoted_context"
                bundle["prefill_messages"] = []
            bundle["prefill_mode"] = str(args.prefill_mode)
            bundle["context_asset_family"] = context_asset_family
            bundle["canary"] = {}
            if bool(args.enable_canary_probe) and int(budget) > 0 and (
                str(bundle.get("context_text", "")).strip() or list(bundle.get("prefill_messages") or ())
            ):
                canary = _build_canary_spec(
                    trait_name=str(trait.name),
                    budget=int(budget),
                    seed=int(args.seed),
                    source_ids=bundle.get("used_source_ids", ()),
                    position_fraction=float(args.canary_position_fraction),
                )
                if _uses_history_prefill(str(args.prefill_mode)):
                    merged_messages, merged_tokens = _insert_canary_into_prefill_messages(
                        tokenizer=tokenizer,
                        prefill_messages=list(bundle.get("prefill_messages") or ()),
                        canary_block=str(canary["block"]),
                        position_fraction=float(canary["position_fraction"]),
                    )
                    bundle["prefill_messages"] = merged_messages
                    bundle["actual_token_count"] = int(merged_tokens)
                else:
                    merged_text, merged_tokens = _insert_canary_into_context_text(
                        tokenizer=tokenizer,
                        context_text=str(bundle["context_text"]),
                        canary_block=str(canary["block"]),
                        position_fraction=float(canary["position_fraction"]),
                    )
                    bundle["context_text"] = merged_text
                    bundle["actual_token_count"] = int(merged_tokens)
                bundle["canary"] = {
                    "expected_answer": str(canary["code"]),
                    "probe_user_message": str(canary["probe_user_message"]),
                    "position_fraction": float(canary["position_fraction"]),
                }
            context_bundles[int(budget)] = bundle
            actual_tokens = int(bundle["actual_token_count"])
            if int(budget) > 0:
                min_required = int(round(int(budget) * max(0.0, 1.0 - float(args.context_underfill_tolerance))))
                if actual_tokens < min_required:
                    message = (
                        f"[context-warning] trait={trait.name} prefill_mode={args.prefill_mode} budget={budget} "
                        f"actual_tokens={actual_tokens} fell below tolerance threshold={min_required}"
                    )
                    if str(args.benchmark_preset) == "paper_v1" or _enforce_strict_underfill(args):
                        raise SystemExit(message)
                    print(message, flush=True)
            print(
                f"[context] trait={trait.name} mode={args.prefill_mode} budget={budget} actual_tokens={bundle['actual_token_count']} "
                f"sources={bundle['used_source_ids']} canary={bool(bundle['canary'])}",
                flush=True,
            )

        trait_summary_rows: List[Dict[str, Any]] = []
        for budget in budgets:
            bundle = context_bundles[int(budget)]
            context_text = str(bundle["context_text"])
            used_source_ids = list(bundle["used_source_ids"])
            actual_tokens = int(bundle["actual_token_count"])
            canary_payload = dict(bundle.get("canary") or {})
            prefill_messages = list(bundle.get("prefill_messages") or ())
            prefill_format = str(bundle.get("prefill_format", "quoted_context"))

            for method in args.methods:
                normalized_method = _normalize_method(method)
                config_tag = _method_config_tag(method, args)
                run_dir = trait_root / f"budget_{int(budget):05d}" / config_tag
                run_dir.mkdir(parents=True, exist_ok=True)
                prompt_message = None
                prompt_delivery_requested = _prompt_delivery_for_method(method)
                prompt_delivery_summary = (
                    "once_after_system_fallback_no_prefill"
                    if prompt_delivery_requested == "once_middle" and not (str(context_text).strip() or prefill_messages)
                    else prompt_delivery_requested
                )
                if _method_family(method) == "prompt":
                    prompt_message = build_style_prompt_message(
                        trait=trait,
                        target_pole=str(args.target_pole),
                        prompt_role=str(args.prompt_role),
                        wrapper_template=(
                            PROMPT_FAILURE_V2_USER_WRAPPER_TEMPLATE
                            if (
                                not str(args.prompt_wrapper_template or "").strip()
                                and str(args.prompt_role).strip().lower() == "user"
                                and (_is_prompt_failure_v2_preset(str(args.benchmark_preset)) or _uses_history_prefill(str(args.prefill_mode)))
                            )
                            else args.prompt_wrapper_template
                        ),
                        bank_profile=str(args.track_memory_bank_profile),
                        bank_profile_file=getattr(args, "track_memory_bank_profile_file", None),
                    )
                patch = _build_style_patch(
                    model=model,
                    tokenizer=tokenizer,
                    trait=trait,
                    method=method,
                    selected_layer_head_map=selected_layer_head_map,
                    args=args,
                    caches=caches,
                )

                case_runtime_plans: Dict[str, Dict[str, Any]] = {}
                if (
                    supports_incremental_qwen_chat(model, tokenizer)
                    and not bool(args.disable_context_window_preflight)
                    and model_context_window > 0
                ):
                    safety_margin = max(0, int(args.context_window_safety_margin))
                    max_allowed_tokens = max(1, model_context_window - safety_margin)
                    for case in cases:
                        probe_user_message = str(canary_payload.get("probe_user_message", "") or "").strip() or None
                        prompt_plan = _build_prompt_delivery_plan(
                            tokenizer=tokenizer,
                            case=case,
                            prompt_message=prompt_message,
                            prompt_delivery=prompt_delivery_requested,
                            context_text=context_text,
                            prefill_messages=prefill_messages,
                            prefill_split_mode="messages" if _uses_history_prefill(str(args.prefill_mode)) else "text",
                            prompt_middle_fraction=float(args.prompt_middle_fraction),
                            prompt_refresh_every_k=int(args.prompt_refresh_every_k),
                            include_probe_turn=probe_user_message is not None,
                        )
                        all_user_turns = [str(turn.user_message) for turn in case.turns]
                        if probe_user_message is not None:
                            all_user_turns.append(probe_user_message)
                        case_runtime_plans[case.case_id] = {
                            "prompt_plan": prompt_plan,
                            "probe_user_message": probe_user_message,
                            "all_user_turns": all_user_turns,
                        }
                        estimated_total_tokens = _estimate_case_total_tokens(
                            tokenizer=tokenizer,
                            initial_messages=prompt_plan["initial_messages"],
                            user_turns=all_user_turns,
                            turn_prefix_messages=prompt_plan["turn_prefix_messages"],
                            max_new_tokens=int(args.max_new_tokens),
                        )
                        if estimated_total_tokens is None:
                            continue
                        if int(estimated_total_tokens) > int(max_allowed_tokens):
                            raise SystemExit(
                                "Predicted dialogue length exceeds the model context window before generation: "
                                f"trait={trait.name} method={method} budget={budget} case_id={case.case_id} "
                                f"estimated_total_tokens={estimated_total_tokens} "
                                f"max_position_embeddings={model_context_window} "
                                f"context_window_safety_margin={safety_margin}. "
                                "Reduce --long-context-token-budgets, --max-new-tokens, or case length."
                            )

                rows: List[Dict[str, Any]] = []
                judge_items: List[Tuple[DialoguePersistenceCase, List[str], str]] = []
                print(
                    f"[run] trait={trait.name} budget={budget} actual_tokens={actual_tokens} method={normalized_method} "
                    f"prefill_mode={args.prefill_mode} config={config_tag}",
                    flush=True,
                )
                for case_index, case in enumerate(cases, start=1):
                    if not args.disable_progress_bar:
                        print(f"  [case] {case_index}/{len(cases)} {case.case_id}", flush=True)
                    runtime_plan = case_runtime_plans.get(case.case_id)
                    if runtime_plan is None:
                        probe_user_message = str(canary_payload.get("probe_user_message", "") or "").strip() or None
                        prompt_plan = _build_prompt_delivery_plan(
                            tokenizer=tokenizer,
                            case=case,
                            prompt_message=prompt_message,
                            prompt_delivery=prompt_delivery_requested,
                            context_text=context_text,
                            prefill_messages=prefill_messages,
                            prefill_split_mode="messages" if _uses_history_prefill(str(args.prefill_mode)) else "text",
                            prompt_middle_fraction=float(args.prompt_middle_fraction),
                            prompt_refresh_every_k=int(args.prompt_refresh_every_k),
                            include_probe_turn=probe_user_message is not None,
                        )
                        runtime_plan = {
                            "prompt_plan": prompt_plan,
                            "probe_user_message": probe_user_message,
                        }
                    prompt_plan = runtime_plan["prompt_plan"]
                    assistant_turns, transcript, probe_response = _generate_dialogue_case(
                        model=model,
                        tokenizer=tokenizer,
                        case=case,
                        args=args,
                        initial_messages=prompt_plan["initial_messages"],
                        turn_prefix_messages=prompt_plan["turn_prefix_messages"],
                        patch=patch,
                        probe_user_message=runtime_plan.get("probe_user_message"),
                    )
                    expected_canary = str(canary_payload.get("expected_answer", "") or "").strip()
                    probe_exact_match = ""
                    probe_correct: str | bool = ""
                    if expected_canary:
                        probe_exact_match = _normalize_canary_answer(probe_response) == _normalize_canary_answer(expected_canary)
                        probe_correct = bool(probe_exact_match)
                    initial_prompt_insertions = [
                        row for row in prompt_plan["prompt_insertions"] if str(row.get("phase", "")) == "initial" and row.get("position_fraction", "") != ""
                    ]
                    judge_items.append((case, assistant_turns, transcript))
                    rows.append(
                        {
                            "benchmark": "dialogue_persistence_michael_track_a",
                            "target_trait": str(trait.name),
                            "target_pole": str(args.target_pole),
                            "method": str(normalized_method),
                            "config_tag": str(config_tag),
                            "family": _method_family(method),
                            "benchmark_variant": str(args.benchmark_preset),
                            "seed": int(args.seed),
                            "model_name": str(args.model_name),
                            "prefill_mode": str(args.prefill_mode),
                            "prefill_format": prefill_format,
                            "context_asset_family": context_asset_family,
                            "prompt_role": str(args.prompt_role),
                            "prompt_delivery_requested": str(prompt_delivery_requested),
                            "prompt_delivery": str(prompt_plan["prompt_delivery_effective"]),
                            "prompt_message_count": int(prompt_plan["prompt_message_count"]),
                            "prompt_initial_token_fraction": initial_prompt_insertions[0]["position_fraction"] if initial_prompt_insertions else "",
                            "prompt_insertions_json": json.dumps(prompt_plan["prompt_insertions"], ensure_ascii=False),
                            "context_token_budget": int(budget),
                            "context_actual_tokens": int(actual_tokens),
                            "context_source_ids_json": json.dumps(used_source_ids),
                            "selected_layer_ids_json": json.dumps(selected_layer_ids),
                            "selected_layer_head_map_json": json.dumps({str(layer): list(heads) for layer, heads in selected_layer_head_map.items()}, sort_keys=True),
                            "selector_artifact_path": str(selector_artifact_path),
                            "skeleton_id": _skeleton_id_from_case_id(case.case_id),
                            "case_id": case.case_id,
                            "bucket": case.bucket,
                            "title": case.title,
                            "num_turns": len(case.turns),
                            "initial_messages_json": json.dumps(prompt_plan["initial_messages"], ensure_ascii=False),
                            "turn_prefix_messages_json": json.dumps(prompt_plan["turn_prefix_messages"], ensure_ascii=False),
                            "assistant_turns_json": json.dumps(assistant_turns, ensure_ascii=False),
                            "transcript": transcript,
                            "probe_response": probe_response,
                            "probe_exact_match": probe_exact_match,
                            "probe_correct": probe_correct,
                            "canary_expected_answer": expected_canary,
                            "canary_position_fraction": canary_payload.get("position_fraction", ""),
                            "turn_judgments_json": "",
                            "conversation_judge_ok": False,
                            "conversation_judge_n_ok": 0,
                            "coherence": "",
                            "user_state_consistency": "",
                            "non_repetitiveness": "",
                            "overall_quality": "",
                        }
                    )

                scored_keys = [f"persistence_{trait.name}", *_CONVERSATION_QUALITY_NAMES]
                conversation_trait_subset = None if bool(args.conversation_judge_all_traits) else [trait.name]
                if not args.skip_judge:
                    judged = asyncio.run(
                        _judge_all_cases(
                            items=judge_items,
                            judge_model=str(args.judge_model),
                            max_concurrent=int(args.max_concurrent_judge),
                            transcript_max_chars=int(args.conversation_transcript_max_chars) if args.conversation_transcript_max_chars is not None else None,
                            skip_turn_judge=bool(args.skip_turn_judge),
                            judge_debug=bool(args.judge_debug),
                            conversation_judge_repeats=int(args.conversation_judge_repeats),
                            conversation_persistence_traits=conversation_trait_subset,
                        )
                    )
                    for row, (case, _, _), (turn_results, conversation_results) in zip(rows, judge_items, judged):
                        row["turn_judgments_json"] = _turn_judgments_to_json(
                            case,
                            turn_results,
                            turn_judge_skipped=bool(args.skip_turn_judge),
                        )
                        row["conversation_judge_trait_subset"] = "all" if conversation_trait_subset is None else ",".join(conversation_trait_subset)
                        row["conversation_judge_repeats"] = int(args.conversation_judge_repeats)
                        _apply_conversation_scores_from_repeats(
                            row,
                            conversation_results,
                            scored_keys=scored_keys if conversation_trait_subset is not None else [f"persistence_{name}" for name in _TURN_TRAIT_NAMES] + list(_CONVERSATION_QUALITY_NAMES),
                        )

                turn_rows = _flatten_turn_rows(rows=rows, target_trait=str(trait.name))
                aggregate_metrics = _aggregate_run_metrics(rows=rows, turn_rows=turn_rows, target_trait=str(trait.name))
                run_config = {
                    **vars(args),
                    "resolved_target_trait": str(trait.name),
                    "resolved_selector_artifact": str(selector_artifact_path),
                    "selected_layer_ids": selected_layer_ids,
                    "selected_layer_head_map": {str(layer): list(heads) for layer, heads in selected_layer_head_map.items()},
                    "context_bundle": {
                        "token_budget": int(budget),
                        "actual_token_count": int(actual_tokens),
                        "used_source_ids": used_source_ids,
                        "prefill_mode": str(args.prefill_mode),
                        "prefill_format": prefill_format,
                        "context_asset_family": context_asset_family,
                        "canary": canary_payload,
                    },
                    "aggregate_metrics": aggregate_metrics,
                }
                _write_run_outputs(run_dir=run_dir, rows=rows, turn_rows=turn_rows, run_config=run_config)

                summary_row = {
                    "target_trait": str(trait.name),
                    "method": str(normalized_method),
                    "config_tag": str(config_tag),
                    "benchmark_variant": str(args.benchmark_preset),
                    "prefill_mode": str(args.prefill_mode),
                    "prefill_format": prefill_format,
                    "context_asset_family": context_asset_family,
                    "prompt_role": str(args.prompt_role),
                    "prompt_delivery": str(prompt_delivery_summary),
                    "context_token_budget": int(budget),
                    "context_actual_tokens": int(actual_tokens),
                    "context_source_ids_json": json.dumps(used_source_ids),
                    "selector_artifact_path": str(selector_artifact_path),
                    "selected_layer_ids_json": json.dumps(selected_layer_ids),
                    "selected_layer_head_map_json": json.dumps({str(layer): list(heads) for layer, heads in selected_layer_head_map.items()}, sort_keys=True),
                    "run_dir": str(run_dir),
                    **aggregate_metrics,
                }
                trait_summary_rows.append(summary_row)
                overall_summary_rows.append(summary_row)

        trait_summary_path = trait_root / "workflow_summary.csv"
        with _exclusive_file_lock(trait_root / ".workflow_summary.lock"):
            rows_to_csv(trait_summary_path, _merge_summary_rows(trait_summary_path, trait_summary_rows))
        print(f"[summary] wrote {trait_root / 'workflow_summary.csv'}", flush=True)

    overall_summary_path = output_root / "workflow_summary.csv"
    with _exclusive_file_lock(output_root / ".workflow_summary.lock"):
        rows_to_csv(overall_summary_path, _merge_summary_rows(overall_summary_path, overall_summary_rows))
    print(f"[summary] wrote {output_root / 'workflow_summary.csv'}", flush=True)
    if args.human_validation_export:
        manifest = export_human_validation_subset(
            output_root=output_root,
            export_dir=args.human_validation_export,
        )
        print(f"[human-validation] wrote {manifest['sample_count']} samples to {args.human_validation_export}", flush=True)


if __name__ == "__main__":
    main()
