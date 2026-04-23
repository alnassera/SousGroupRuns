#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from .qwen3_moe_intervention_model import ARTIFACT_ENV_VAR, register as register_intervention_model


SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent
BUNDLE_ROOT_TEXT = str(BUNDLE_ROOT)
for shadow_path in (BUNDLE_ROOT / "qwen3_moe" / "evals", BUNDLE_ROOT / "qwen3_moe"):
    shadow_text = str(shadow_path)
    while shadow_text in sys.path:
        sys.path.remove(shadow_text)
while BUNDLE_ROOT_TEXT in sys.path:
    sys.path.remove(BUNDLE_ROOT_TEXT)
sys.path.insert(0, BUNDLE_ROOT_TEXT)

from architecture import wrap_qwen_tokenizer_apply_chat_template
from long_context.style_track_a import build_style_prompt_message
from long_context.style_traits import default_profile_json_path, resolve_style_traits
from qwen3_moe.evals.run_gsm8k_vllm_eval import (
    ROLE_CHOICES,
    STYLE_TRAIT_ALIASES,
    _format_messages,
    _iter_chunks,
    _load_existing_method_rows,
    _write_method_outputs,
    rows_to_csv,
    write_json,
    write_jsonl,
)
from qwen3_moe.modeling import resolve_hf_cache_dir


METHOD_CHOICES = ("plain", "prompt", "post_attn_caa", "canonical_memory")
VLLM_PORT_PLUGIN_NAME = "vllm_port_intervention"
VLLM_PORT_PLUGIN_DIST_DIR = SCRIPT_DIR / "plugin_dist"
LETTER_CHOICES = ("A", "B", "C", "D")
FINAL_ANSWER_RE = re.compile(r"(?:final answer|answer)\s*[:\-]?\s*([ABCD])\b", flags=re.IGNORECASE)
LETTER_RE = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMLU vLLM evals with optional style-steering artifacts.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--target-traits", type=str, nargs="+", default=["warm"])
    parser.add_argument("--target-pole", type=str, default="high", choices=["high", "low"])
    parser.add_argument("--methods", type=str, nargs="+", default=list(METHOD_CHOICES), choices=list(METHOD_CHOICES))
    parser.add_argument("--mmlu-dev-path", type=str, default=str(BUNDLE_ROOT / "datasets" / "mmlu" / "dev.jsonl"))
    parser.add_argument("--mmlu-test-path", type=str, default=str(BUNDLE_ROOT / "datasets" / "mmlu" / "test.jsonl"))
    parser.add_argument("--few-shot-k", type=int, default=5)
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--qwen-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt-role", type=str, default="system", choices=list(ROLE_CHOICES))
    parser.add_argument("--prompt-wrapper-template", type=str, default=None)
    parser.add_argument("--track-memory-bank-profile", type=str, default="default")
    parser.add_argument("--track-memory-bank-profile-file", type=str, default=str(default_profile_json_path()))
    parser.add_argument("--artifact-root", type=str, default=str(BUNDLE_ROOT / "runs" / "vllm_port_artifacts"))
    parser.add_argument(
        "--artifact-path-template",
        type=str,
        default="{artifact_root}/{trait}_{target_pole}/{trait}_{target_pole}_{method}.pt",
    )
    parser.add_argument("--output-dir", type=str, default=str(BUNDLE_ROOT / "runs" / "mmlu_vllm_interventions"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="",
        choices=["", "FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION"],
    )
    return parser.parse_args()


def resolve_style_traits_with_aliases(names: Iterable[str]) -> tuple[object, ...]:
    resolved_names: List[str] = []
    for raw_name in names:
        key = str(raw_name).strip().lower()
        canonical = STYLE_TRAIT_ALIASES.get(key)
        if canonical is None:
            raise KeyError(f"Unknown style trait '{raw_name}'. Available: {sorted(STYLE_TRAIT_ALIASES)}")
        resolved_names.append(canonical)
    return resolve_style_traits(resolved_names)


def _load_jsonl_rows(path_text: str | Path) -> List[dict]:
    path = Path(path_text).expanduser().resolve()
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _normalize_answer(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, int):
        return LETTER_CHOICES[value] if 0 <= int(value) < len(LETTER_CHOICES) else None
    text = str(value).strip().upper()
    if text in LETTER_CHOICES:
        return text
    if text.isdigit():
        index = int(text)
        return LETTER_CHOICES[index] if 0 <= index < len(LETTER_CHOICES) else None
    return None


def _resolve_choices(raw: object) -> List[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, tuple):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return [str(item) for item in payload]
        except Exception:
            pass
        return [segment.strip() for segment in text.split("|||") if segment.strip()]
    return []


def _load_mmlu_rows(path_text: str, *, required: bool) -> List[dict]:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        if required:
            raise RuntimeError(f"Missing MMLU file: {path}")
        return []
    rows = _load_jsonl_rows(path)
    if required and not rows:
        raise RuntimeError(f"MMLU file is empty: {path}")
    return [
        {
            "subject": str(row.get("subject", "")).strip(),
            "question": str(row.get("question", "")).strip(),
            "choices": _resolve_choices(row.get("choices", [])),
            "answer": _normalize_answer(row.get("answer")),
        }
        for row in rows
    ]


def _format_question_block(question: str, choices: Sequence[str], *, answer: Optional[str]) -> str:
    lines = [str(question).strip()]
    for label, choice in zip(LETTER_CHOICES, choices):
        lines.append(f"{label}. {str(choice)}")
    lines.append("Answer:" if answer is None else f"Answer: {str(answer)}")
    return "\n".join(lines)


def _build_subject_index(rows: Sequence[Mapping[str, object]]) -> Dict[str, List[dict]]:
    index: Dict[str, List[dict]] = {}
    for row in rows:
        subject = str(row.get("subject", "")).strip()
        index.setdefault(subject, []).append(dict(row))
    return index


def _select_few_shot_rows(
    *,
    subject: str,
    dev_index: Mapping[str, Sequence[Mapping[str, object]]],
    all_dev_rows: Sequence[Mapping[str, object]],
    few_shot_k: int,
) -> List[Mapping[str, object]]:
    chosen = list(dev_index.get(subject, []))[: int(few_shot_k)]
    if len(chosen) >= int(few_shot_k):
        return chosen[: int(few_shot_k)]
    for row in all_dev_rows:
        if len(chosen) >= int(few_shot_k):
            break
        if str(row.get("subject", "")).strip() == str(subject).strip():
            continue
        chosen.append(row)
    return chosen[: int(few_shot_k)]


def _build_eval_prompt(item: Mapping[str, object], *, few_shot_rows: Sequence[Mapping[str, object]]) -> str:
    subject = str(item.get("subject", "")).strip().replace("_", " ")
    blocks = [f"The following are multiple choice questions (with answers) about {subject}."]
    for shot in few_shot_rows:
        blocks.append(
            _format_question_block(
                str(shot.get("question", "")),
                _resolve_choices(shot.get("choices", [])),
                answer=_normalize_answer(shot.get("answer")),
            )
        )
    blocks.append("Now answer the next question. Respond with only the letter A, B, C, or D.")
    blocks.append(
        _format_question_block(
            str(item.get("question", "")),
            _resolve_choices(item.get("choices", [])),
            answer=None,
        )
    )
    return "\n\n".join(blocks)


def _extract_prediction_answer(text: str) -> Optional[str]:
    stripped = str(text).strip()
    if not stripped:
        return None
    final_match = None
    for match in FINAL_ANSWER_RE.finditer(stripped):
        final_match = match
    if final_match is not None:
        return str(final_match.group(1)).upper()
    last_letter = None
    for match in LETTER_RE.finditer(stripped):
        last_letter = match.group(1)
    return None if last_letter is None else str(last_letter).upper()


def _load_items(args: argparse.Namespace) -> tuple[List[dict], Mapping[str, Sequence[Mapping[str, object]]], List[dict]]:
    dev_rows = _load_mmlu_rows(args.mmlu_dev_path, required=True)
    test_rows = _load_mmlu_rows(args.mmlu_test_path, required=True)
    if args.subjects:
        wanted = {str(subject).strip() for subject in args.subjects}
        dev_rows = [row for row in dev_rows if str(row.get("subject", "")).strip() in wanted]
        test_rows = [row for row in test_rows if str(row.get("subject", "")).strip() in wanted]
    if int(args.offset) > 0:
        test_rows = test_rows[int(args.offset) :]
    if int(args.max_items) > 0:
        test_rows = test_rows[: int(args.max_items)]
    if not test_rows:
        raise RuntimeError("No MMLU test rows matched the current filters.")
    return test_rows, _build_subject_index(dev_rows), dev_rows


def _build_method_messages(*, method: str, user_content: str, trait, args: argparse.Namespace) -> List[Dict[str, str]]:
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


def _resolve_artifact_path(*, trait_name: str, target_pole: str, method: str, args: argparse.Namespace) -> Path:
    text = str(args.artifact_path_template).format(
        artifact_root=str(Path(args.artifact_root).expanduser().resolve()),
        trait=str(trait_name),
        target_pole=str(target_pole),
        method=str(method),
    )
    resolved = Path(text).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing steering artifact for {trait_name}/{target_pole}/{method}: {resolved}")
    return resolved


def _model_source_and_tokenizer_kwargs(args: argparse.Namespace) -> tuple[str, Optional[str], dict]:
    model_source_path = Path(str(args.model_name)).expanduser()
    if model_source_path.exists():
        return str(model_source_path.resolve()), None, {"local_files_only": True}
    model_source = str(args.model_name)
    cache_dir, local_files_only = resolve_hf_cache_dir(model_source)
    tokenizer_kwargs = {}
    if cache_dir is not None:
        tokenizer_kwargs["cache_dir"] = cache_dir
    if local_files_only:
        tokenizer_kwargs["local_files_only"] = True
    return model_source, cache_dir, tokenizer_kwargs


def _prepend_env_path(name: str, values: Sequence[Path]) -> None:
    existing = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    additions = [str(value) for value in values if str(value) not in existing]
    if additions:
        os.environ[name] = os.pathsep.join([*additions, *existing])


def _enable_vllm_port_plugin() -> None:
    for path in (VLLM_PORT_PLUGIN_DIST_DIR, BUNDLE_ROOT):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    _prepend_env_path("PYTHONPATH", (VLLM_PORT_PLUGIN_DIST_DIR, BUNDLE_ROOT))
    existing_plugins = [
        value.strip()
        for value in os.environ.get("VLLM_PLUGINS", "").split(",")
        if value.strip()
    ]
    if VLLM_PORT_PLUGIN_NAME not in existing_plugins:
        os.environ["VLLM_PLUGINS"] = ",".join([*existing_plugins, VLLM_PORT_PLUGIN_NAME])


def _load_llm(*, args: argparse.Namespace, model_source: str, cache_dir: Optional[str], artifact_path: Optional[Path]):
    previous_artifact = os.environ.get(ARTIFACT_ENV_VAR)
    if artifact_path is None:
        os.environ.pop(ARTIFACT_ENV_VAR, None)
    else:
        _enable_vllm_port_plugin()
        os.environ[ARTIFACT_ENV_VAR] = str(artifact_path)
        register_intervention_model()

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
    try:
        print(f"[vllm] loading {model_source} artifact={artifact_path}", flush=True)
        return LLM(**llm_kwargs)
    except Exception:
        if previous_artifact is None:
            os.environ.pop(ARTIFACT_ENV_VAR, None)
        else:
            os.environ[ARTIFACT_ENV_VAR] = previous_artifact
        raise


def _release_llm(llm) -> None:
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _row_from_generation(
    *,
    trait_name: str,
    target_pole: str,
    method: str,
    item_index: int,
    item: Mapping[str, object],
    generation_text: str,
    formatted_prompt: str,
) -> dict:
    gold_answer = _normalize_answer(item.get("answer"))
    pred_answer = _extract_prediction_answer(generation_text)
    return {
        "trait": str(trait_name),
        "target_pole": str(target_pole),
        "method": str(method),
        "item_index": int(item_index),
        "subject": str(item.get("subject", "")),
        "question": str(item.get("question", "")),
        "choices_json": json.dumps(_resolve_choices(item.get("choices", [])), ensure_ascii=False),
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "is_correct": bool(pred_answer is not None and gold_answer is not None and pred_answer == gold_answer),
        "formatted_prompt": formatted_prompt,
        "generation_text": generation_text,
        "selector_artifact_path": "",
        "engine": "vllm",
    }


def _summary_row(*, trait_name: str, target_pole: str, method: str, rows: Sequence[Mapping[str, object]]) -> dict:
    total = len(rows)
    correct = sum(int(bool(row.get("is_correct"))) for row in rows)
    invalid = sum(1 for row in rows if row.get("pred_answer") is None)
    return {
        "trait": str(trait_name),
        "target_pole": str(target_pole),
        "method": str(method),
        "total": int(total),
        "correct": int(correct),
        "invalid": int(invalid),
        "accuracy": float(correct / max(total, 1)),
        "engine": "vllm",
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", vars(args))

    items, dev_index, dev_rows = _load_items(args)
    traits = resolve_style_traits_with_aliases(args.target_traits)

    try:
        from transformers import AutoTokenizer
        from vllm import SamplingParams
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install transformers and vLLM before running this script.") from exc

    model_source, cache_dir, tokenizer_kwargs = _model_source_and_tokenizer_kwargs(args)
    tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
    wrap_qwen_tokenizer_apply_chat_template(tokenizer, enable_thinking=bool(args.qwen_enable_thinking))
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(
        max_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        seed=int(args.seed),
    )

    all_rows: List[dict] = []
    all_summaries: List[dict] = []
    for trait in traits:
        trait_dir = output_dir / trait.name / str(args.target_pole)
        trait_rows: List[dict] = []
        trait_summaries: List[dict] = []
        for method in args.methods:
            method_dir = trait_dir / str(method)
            method_dir.mkdir(parents=True, exist_ok=True)
            method_jsonl_path = method_dir / "generations.jsonl"
            method_rows = (
                _load_existing_method_rows(method_jsonl_path, item_count=len(items))
                if bool(args.resume)
                else []
            )
            completed_indices = {int(row["item_index"]) for row in method_rows}
            if method_rows:
                print(f"[resume] {trait.name}/{args.target_pole}/{method}: {len(completed_indices)}/{len(items)} existing rows", flush=True)
                _write_method_outputs(method_dir, method_rows)

            artifact_path = None
            if str(method) in {"post_attn_caa", "canonical_memory"}:
                artifact_path = _resolve_artifact_path(
                    trait_name=str(trait.name),
                    target_pole=str(args.target_pole),
                    method=str(method),
                    args=args,
                )

            llm = None
            try:
                if len(completed_indices) < len(items):
                    llm = _load_llm(
                        args=args,
                        model_source=model_source,
                        cache_dir=cache_dir,
                        artifact_path=artifact_path,
                    )
                    pending_items = [
                        (item_index, item)
                        for item_index, item in enumerate(items)
                        if int(item_index) not in completed_indices
                    ]
                    chunks = _iter_chunks(pending_items, int(args.batch_size))
                    for chunk_index, chunk in enumerate(chunks, start=1):
                        prompts: List[str] = []
                        for _, item in chunk:
                            few_shot_rows = _select_few_shot_rows(
                                subject=str(item.get("subject", "")),
                                dev_index=dev_index,
                                all_dev_rows=dev_rows,
                                few_shot_k=int(args.few_shot_k),
                            )
                            prompt_text = _build_eval_prompt(item, few_shot_rows=few_shot_rows)
                            messages = _build_method_messages(method=str(method), user_content=prompt_text, trait=trait, args=args)
                            prompts.append(_format_messages(tokenizer, messages))
                        print(f"[vllm] {trait.name}/{method} chunk {chunk_index}/{len(chunks)} size={len(chunk)}", flush=True)
                        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                        for (item_index, item), output, formatted_prompt in zip(chunk, outputs, prompts):
                            generation_text = output.outputs[0].text if output.outputs else ""
                            row = _row_from_generation(
                                trait_name=str(trait.name),
                                target_pole=str(args.target_pole),
                                method=str(method),
                                item_index=int(item_index),
                                item=item,
                                generation_text=generation_text,
                                formatted_prompt=formatted_prompt,
                            )
                            row["steering_artifact_path"] = "" if artifact_path is None else str(artifact_path)
                            row["selector_artifact_path"] = row["steering_artifact_path"]
                            method_rows.append(row)
                            completed_indices.add(int(item_index))
                        method_rows.sort(key=lambda value: int(value["item_index"]))
                        _write_method_outputs(method_dir, method_rows)
            finally:
                if llm is not None:
                    _release_llm(llm)

            summary = _summary_row(
                trait_name=str(trait.name),
                target_pole=str(args.target_pole),
                method=str(method),
                rows=method_rows,
            )
            summary["steering_artifact_path"] = "" if artifact_path is None else str(artifact_path)
            trait_rows.extend(method_rows)
            trait_summaries.append(summary)
            all_rows.extend(method_rows)

        method_to_accuracy = {str(row["method"]): float(row["accuracy"]) for row in trait_summaries}
        plain_accuracy = method_to_accuracy.get("plain")
        prompt_accuracy = method_to_accuracy.get("prompt")
        for row in trait_summaries:
            accuracy = float(row["accuracy"])
            row["accuracy_delta_vs_plain"] = None if plain_accuracy is None else float(accuracy - plain_accuracy)
            row["accuracy_delta_vs_prompt"] = None if prompt_accuracy is None else float(accuracy - prompt_accuracy)
        write_jsonl(trait_dir / "generations.jsonl", trait_rows)
        rows_to_csv(trait_dir / "generations.csv", trait_rows)
        rows_to_csv(trait_dir / "summary.csv", trait_summaries)
        write_json(trait_dir / "summary.json", trait_summaries)
        all_summaries.extend(trait_summaries)

    rows_to_csv(output_dir / "all_generations.csv", all_rows)
    rows_to_csv(output_dir / "all_summaries.csv", all_summaries)
    write_json(output_dir / "all_summaries.json", all_summaries)


if __name__ == "__main__":
    main()
