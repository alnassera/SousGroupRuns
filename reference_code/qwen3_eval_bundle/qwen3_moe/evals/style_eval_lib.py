from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
BUNDLE_ROOT = PACKAGE_ROOT.parent
for candidate in (SCRIPT_DIR, PACKAGE_ROOT, BUNDLE_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from long_context.style_track_a import (
    CompositePatch,
    CanonicalDescriptorMemoryAttentionPatch,
    CanonicalDescriptorMixConfig,
    DEFAULT_SELECTOR_LAYER_IDS,
    build_canonical_descriptor_memory,
    build_style_prompt_message,
    generate_from_messages,
    parse_layer_head_map,
    rows_to_csv,
    run_style_phrase_selector,
)
from long_context.style_traits import (
    build_style_contrastive_examples,
    default_profile_json_path,
    resolve_selector_prompts,
    resolve_style_track_memory_bank,
    resolve_style_traits,
)
from qwen3_moe.layer_schedule import layer_rhos, layer_scaled_values
from qwen3_moe.modeling import load_model_and_tokenizer
from qwen3_moe.post_attn_caa_patch import PostAttentionResidualCAAPatch
from qwen3_moe.site_activation import TOKEN_SELECTOR_CHOICES, build_site_activation_steering_vector_from_examples


METHOD_CHOICES = ("plain", "prompt", "post_attn_caa", "canonical_memory")
STYLE_TRAIT_ALIASES = {
    "warm": "warm",
    "friendly": "warm",
    "formal": "formal",
    "professional": "formal",
    "assertive": "assertive",
    "direct": "assertive",
    "cautious": "cautious",
    "careful": "cautious",
    "dismissive": "dismissive",
    "curt": "dismissive",
    "anxious": "anxious",
}
STEERING_METHODS = {"post_attn_caa", "canonical_memory"}
_ATTN_CHOICES = ("auto", "sdpa", "flash_attention_2", "eager")
_LAYER_SELECTION_CHOICES = ("top_head_score_sum", "top_head_score_mean", "positive_head_count")
_ROLE_CHOICES = ("system", "user", "assistant")


@dataclass(frozen=True)
class SelectorResult:
    artifact_path: Path
    selected_layer_ids: Tuple[int, ...]
    selected_layer_head_map: Dict[int, Tuple[int, ...]]
    selected_layer_rho_map: Dict[int, float]
    prompt_count: int


def bundle_root() -> Path:
    return BUNDLE_ROOT


def ensure_style_trait_family(args: argparse.Namespace) -> None:
    trait_family = str(getattr(args, "trait_family", "style")).strip().lower()
    if trait_family != "style":
        raise ValueError(
            "This collaboration bundle packages only the style-trait path. "
            "Use --trait-family style."
        )


def resolve_style_traits_with_aliases(names: Iterable[str]) -> Tuple[object, ...]:
    resolved_names: List[str] = []
    for raw_name in names:
        key = str(raw_name).strip().lower()
        canonical = STYLE_TRAIT_ALIASES.get(key)
        if canonical is None:
            raise KeyError(
                f"Unknown style trait '{raw_name}'. "
                f"Available: {sorted(STYLE_TRAIT_ALIASES)}"
            )
        resolved_names.append(canonical)
    return resolve_style_traits(resolved_names)


def add_common_eval_args(parser: argparse.ArgumentParser, *, default_output_dir: str) -> None:
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--trait-family", type=str, default="style", choices=["style", "bigfive"])
    parser.add_argument("--target-traits", type=str, nargs="+", default=["warm"])
    parser.add_argument("--target-pole", type=str, default="high", choices=["high", "low"])
    parser.add_argument("--methods", type=str, nargs="+", default=list(METHOD_CHOICES), choices=list(METHOD_CHOICES))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default="auto", choices=list(_ATTN_CHOICES))
    parser.add_argument("--allow-eager-attention", action="store_true")
    parser.add_argument("--qwen-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-architecture-check", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--layer-ids", type=int, nargs="+", default=list(DEFAULT_SELECTOR_LAYER_IDS))
    parser.add_argument("--selector-head-ids", type=int, nargs="*", default=None)
    parser.add_argument("--selector-layer-head-map", type=str, nargs="*", default=[])
    parser.add_argument("--selector-prompt-limit", type=int, default=10)
    parser.add_argument("--selector-max-new-tokens", type=int, default=96)
    parser.add_argument("--selector-diagnostic-max-decode-steps", type=int, default=24)
    parser.add_argument("--max-heads-per-layer", type=int, default=5)
    parser.add_argument("--max-layers", type=int, default=6)
    parser.add_argument(
        "--layer-selection-metric",
        type=str,
        default="top_head_score_sum",
        choices=list(_LAYER_SELECTION_CHOICES),
    )

    parser.add_argument("--rho", type=float, default=8.0)
    parser.add_argument("--layer-weight-schedule", type=str, default="middle_heavy")
    parser.add_argument("--mix-mode", type=str, default="contrastive_neutral", choices=["convex", "contrastive_neutral", "contrastive_opposite"])
    parser.add_argument("--prefill-steering", type=str, default="last_token_only", choices=["full", "last_token_only", "none"])
    parser.add_argument("--steering-operator", type=str, default="bank_softmax_mixture", choices=["key_logit_sparse", "output_mixture", "bank_softmax_mixture"])
    parser.add_argument("--positive-gain", type=float, default=4.0)
    parser.add_argument("--negative-gain", type=float, default=0.1)
    parser.add_argument("--query-gate-scale", type=float, default=1.0)
    parser.add_argument("--prompt-bank-normalization", type=str, default="log_token_count", choices=["none", "log_token_count"])
    parser.add_argument("--memory-role", type=str, default="system", choices=list(_ROLE_CHOICES))
    parser.add_argument("--disable-memory-chat-template", action="store_true")
    parser.add_argument("--disable-memory-descriptor-only", action="store_true")
    parser.add_argument("--disable-query-adaptive-gates", action="store_true")
    parser.add_argument("--track-memory-bank-profile", type=str, default="default")
    parser.add_argument("--track-memory-bank-profile-file", type=str, default=str(default_profile_json_path()))
    parser.add_argument("--prompt-role", type=str, default="system", choices=list(_ROLE_CHOICES))
    parser.add_argument("--prompt-wrapper-template", type=str, default=None)

    parser.add_argument("--caa-layer-ids", type=int, nargs="*", default=None)
    parser.add_argument("--caa-scale", type=float, default=1.0)
    parser.add_argument("--caa-layer-weight-schedule", type=str, default="middle_heavy")
    parser.add_argument("--caa-budget-normalization", type=str, default="sum", choices=["peak", "sum"])
    parser.add_argument("--caa-token-selector", type=str, default="last_completion_token", choices=list(TOKEN_SELECTOR_CHOICES))
    parser.add_argument("--caa-example-limit", type=int, default=10)

    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--output-dir", type=str, default=default_output_dir)


def load_jsonl_rows(path: str | Path) -> List[dict]:
    resolved = Path(path).expanduser().resolve()
    rows: List[dict] = []
    if not resolved.exists():
        raise FileNotFoundError(f"Missing JSONL file: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def write_json(path: str | Path, payload: object) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def iter_with_progress(items: Sequence[Mapping[str, object]], *, desc: str, disable: bool):
    if disable or tqdm is None:
        return items
    return tqdm(items, desc=desc, total=len(items))


def load_model_stack(args: argparse.Namespace):
    model, tokenizer = load_model_and_tokenizer(
        str(args.model_name),
        device_map=str(args.device_map),
        torch_dtype=str(args.torch_dtype),
        attn_implementation=str(args.attn_implementation),
        allow_eager_fallback=bool(args.allow_eager_attention),
        enable_thinking=bool(args.qwen_enable_thinking),
        strict_architecture_check=bool(args.strict_architecture_check),
    )
    model.eval()
    return model, tokenizer


def build_method_messages(
    *,
    method: str,
    user_content: str,
    trait,
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "user", "content": str(user_content)}]
    if str(method) != "prompt":
        return messages
    prompt_message = build_style_prompt_message(
        trait=trait,
        target_pole=str(args.target_pole),
        prompt_role=str(args.prompt_role),
        wrapper_template=args.prompt_wrapper_template,
        bank_profile=str(args.track_memory_bank_profile),
        bank_profile_file=getattr(args, "track_memory_bank_profile_file", None),
    )
    return [prompt_message, *messages]


def maybe_run_selector(
    *,
    model,
    tokenizer,
    trait,
    args: argparse.Namespace,
    selector_dir: Path,
) -> Optional[SelectorResult]:
    requested_methods = {str(method) for method in args.methods}
    if not requested_methods.intersection(STEERING_METHODS):
        return None

    selector_dir.mkdir(parents=True, exist_ok=True)
    raw_layer_head_map = parse_layer_head_map(getattr(args, "selector_layer_head_map", []) or [])
    result = run_style_phrase_selector(
        model=model,
        tokenizer=tokenizer,
        trait=trait,
        prompt_cases=resolve_selector_prompts(limit=int(args.selector_prompt_limit)),
        target_pole=str(args.target_pole),
        layer_ids=[int(layer_id) for layer_id in args.layer_ids],
        head_ids=None if args.selector_head_ids in (None, []) else [int(head_id) for head_id in args.selector_head_ids],
        layer_head_map=raw_layer_head_map or None,
        max_heads_per_layer=int(args.max_heads_per_layer),
        max_layers=int(args.max_layers),
        layer_selection_metric=str(args.layer_selection_metric),
        rho=float(args.rho),
        layer_weight_schedule=str(args.layer_weight_schedule),
        seed=int(args.seed),
        max_new_tokens=int(args.selector_max_new_tokens),
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        bank_profile=str(args.track_memory_bank_profile),
        bank_profile_file=getattr(args, "track_memory_bank_profile_file", None),
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
    )
    rows_to_csv(selector_dir / "full_rows.csv", result["full_rows"])
    rows_to_csv(selector_dir / "trace_rows.csv", result["trace_rows"])
    rows_to_csv(selector_dir / "head_scores.csv", result["head_rows"])
    rows_to_csv(selector_dir / "layer_scores.csv", result["layer_rows"])
    artifact_path = selector_dir / "selector_artifact.json"
    write_json(artifact_path, result["artifact"])
    artifact = dict(result["artifact"])
    return SelectorResult(
        artifact_path=artifact_path,
        selected_layer_ids=tuple(int(layer_id) for layer_id in artifact.get("selected_layer_ids", [])),
        selected_layer_head_map={
            int(layer_id): tuple(int(head_id) for head_id in head_ids)
            for layer_id, head_ids in dict(artifact.get("selected_layer_head_map", {})).items()
        },
        selected_layer_rho_map={
            int(layer_id): float(value)
            for layer_id, value in dict(artifact.get("selected_layer_rho_map", {})).items()
        },
        prompt_count=int(len(result["full_rows"])),
    )


def _memory_pair_for_trait(
    *,
    model,
    tokenizer,
    trait,
    target_pole: str,
    layer_ids: Sequence[int],
    args: argparse.Namespace,
    caches: Dict[tuple, object],
):
    memory_key = (
        trait.name,
        str(target_pole),
        tuple(int(layer_id) for layer_id in layer_ids),
        str(args.track_memory_bank_profile),
        str(getattr(args, "track_memory_bank_profile_file", "") or ""),
    )
    cached = caches.get(memory_key)
    if cached is not None:
        return cached

    active_bank = resolve_style_track_memory_bank(
        trait,
        track="generation",
        pole=str(target_pole),
        bank_profile=str(args.track_memory_bank_profile),
        bank_profile_file=getattr(args, "track_memory_bank_profile_file", None),
    )
    reference_pole = "low" if str(target_pole) == "high" else "high"
    reference_bank = resolve_style_track_memory_bank(
        trait,
        track="generation",
        pole=reference_pole,
        bank_profile=str(args.track_memory_bank_profile),
        bank_profile_file=getattr(args, "track_memory_bank_profile_file", None),
    )
    wrapper_templates = tuple(str(template) for template in active_bank.wrapper_templates if str(template).strip())
    wrapper_template = str(wrapper_templates[0] if wrapper_templates else "{descriptor}")
    memory = build_canonical_descriptor_memory(
        model,
        tokenizer,
        trait_name=f"{trait.name}_generation_{target_pole}",
        descriptor=active_bank.descriptor,
        descriptor_variants=active_bank.descriptor_variants,
        layer_ids=layer_ids,
        wrapper_template=wrapper_template,
        wrapper_templates=wrapper_templates or None,
        use_chat_template=not bool(args.disable_memory_chat_template),
        chat_role=str(args.memory_role),
        keep_descriptor_only=not bool(args.disable_memory_descriptor_only),
    )
    reference_memory = build_canonical_descriptor_memory(
        model,
        tokenizer,
        trait_name=f"{trait.name}_generation_{reference_pole}",
        descriptor=reference_bank.descriptor,
        descriptor_variants=reference_bank.descriptor_variants,
        layer_ids=layer_ids,
        wrapper_template=wrapper_template,
        wrapper_templates=wrapper_templates or None,
        use_chat_template=not bool(args.disable_memory_chat_template),
        chat_role=str(args.memory_role),
        keep_descriptor_only=not bool(args.disable_memory_descriptor_only),
    )
    caches[memory_key] = (memory, reference_memory)
    return memory, reference_memory


def _resolve_caa_layer_ids(args: argparse.Namespace, selector_result: Optional[SelectorResult]) -> Tuple[int, ...]:
    if getattr(args, "caa_layer_ids", None):
        return tuple(int(layer_id) for layer_id in args.caa_layer_ids)
    if selector_result is not None and selector_result.selected_layer_ids:
        return tuple(int(layer_id) for layer_id in selector_result.selected_layer_ids)
    return tuple(int(layer_id) for layer_id in args.layer_ids)


def build_patch_for_method(
    *,
    method: str,
    model,
    tokenizer,
    trait,
    args: argparse.Namespace,
    selector_result: Optional[SelectorResult],
    caches: Dict[str, Dict[tuple, object]],
):
    if str(method) in {"plain", "prompt"}:
        return None
    if selector_result is None:
        raise RuntimeError(f"Method '{method}' requires a selector result, but no selector was run.")

    if str(method) == "canonical_memory":
        layer_ids = tuple(int(layer_id) for layer_id in selector_result.selected_layer_ids or tuple(args.layer_ids))
        if not layer_ids:
            raise RuntimeError("Selector did not choose any layers for canonical_memory.")
        layer_head_map = dict(selector_result.selected_layer_head_map)
        layer_rho_map = dict(selector_result.selected_layer_rho_map)
        if not layer_rho_map:
            layer_rho_map = layer_rhos(layer_ids, float(args.rho), str(args.layer_weight_schedule))
        memory, reference_memory = _memory_pair_for_trait(
            model=model,
            tokenizer=tokenizer,
            trait=trait,
            target_pole=str(args.target_pole),
            layer_ids=layer_ids,
            args=args,
            caches=caches.setdefault("memory", {}),
        )
        patches: List[object] = []
        for layer_id in layer_ids:
            head_ids = tuple(int(head_id) for head_id in layer_head_map.get(int(layer_id), ()))
            if not head_ids:
                continue
            patches.append(
                CanonicalDescriptorMemoryAttentionPatch(
                    layer_ids=[int(layer_id)],
                    memory=memory,
                    reference_memory=reference_memory,
                    config=CanonicalDescriptorMixConfig(
                        rho=float(layer_rho_map.get(int(layer_id), 0.0)),
                        mix_mode=str(args.mix_mode),
                        prefill_steering=str(args.prefill_steering),
                        steering_operator=str(args.steering_operator),
                        positive_gain=float(args.positive_gain),
                        negative_gain=float(args.negative_gain),
                        query_adaptive_gates=not bool(args.disable_query_adaptive_gates),
                        query_gate_scale=float(args.query_gate_scale),
                        prompt_bank_normalization=str(args.prompt_bank_normalization),
                        diagnostic_max_decode_steps=0,
                        record_prefill_diagnostics=False,
                        head_ids=head_ids,
                    ),
                )
            )
        if not patches:
            raise RuntimeError("Selector produced no usable head map for canonical_memory.")
        return CompositePatch(patches)

    if str(method) == "post_attn_caa":
        caa_layer_ids = _resolve_caa_layer_ids(args, selector_result)
        if not caa_layer_ids:
            raise RuntimeError("No layers were available for post_attn_caa.")
        vector_cache = caches.setdefault("site_activation", {})
        cache_key = (
            trait.name,
            tuple(int(layer_id) for layer_id in caa_layer_ids),
            str(args.caa_token_selector),
            int(args.caa_example_limit),
        )
        vector = vector_cache.get(cache_key)
        if vector is None:
            vector = build_site_activation_steering_vector_from_examples(
                model,
                tokenizer,
                trait_name=str(trait.name),
                descriptor=str(trait.high_definition),
                opposite_descriptor=str(trait.low_definition),
                layer_ids=caa_layer_ids,
                examples=build_style_contrastive_examples(trait, limit=int(args.caa_example_limit)),
                site="post_attn_resid",
                token_selector=str(args.caa_token_selector),
                normalize_vector=True,
            )
            vector_cache[cache_key] = vector

        scale_map = layer_scaled_values(
            caa_layer_ids,
            float(args.caa_scale),
            str(args.caa_layer_weight_schedule),
            normalization=str(args.caa_budget_normalization),
        )
        if str(args.target_pole) == "low":
            scale_map = {int(layer_id): -float(value) for layer_id, value in scale_map.items()}
        patches = [
            PostAttentionResidualCAAPatch(
                layer_ids=[int(layer_id)],
                steering_vector=vector,
                scale=float(scale_map.get(int(layer_id), 0.0)),
                prefill_steering=str(args.prefill_steering),
            )
            for layer_id in caa_layer_ids
            if float(scale_map.get(int(layer_id), 0.0)) != 0.0
        ]
        return CompositePatch(patches)

    raise ValueError(f"Unsupported method '{method}'.")
