#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
BUNDLE_ROOT = PACKAGE_ROOT.parent
for candidate in (SCRIPT_DIR, PACKAGE_ROOT, BUNDLE_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from architecture import wrap_qwen_tokenizer_apply_chat_template
from long_context.style_track_a import build_style_prompt_message
from long_context.style_traits import default_profile_json_path, resolve_style_traits
from qwen3_moe.modeling import resolve_hf_cache_dir


METHOD_CHOICES = ("plain", "prompt")
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
PROMPT_STYLE_CHOICES = ("raw_question", "final_answer_line")
ROLE_CHOICES = ("system", "user", "assistant")
GOLD_ANSWER_RE = re.compile(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?)\s*$")
FINAL_ANSWER_RE = re.compile(
    r"final answer\s*:\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?)",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?")
getcontext().prec = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GSM8K plain/prompt baselines through vLLM.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--target-traits", type=str, nargs="+", default=["warm"])
    parser.add_argument("--target-pole", type=str, default="high", choices=["high", "low"])
    parser.add_argument("--methods", type=str, nargs="+", default=list(METHOD_CHOICES), choices=list(METHOD_CHOICES))
    parser.add_argument("--gsm8k-path", type=str, default=str(BUNDLE_ROOT / "datasets" / "gsm8k" / "test.jsonl"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0, help="0 means the full file after offset.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt-style", type=str, default="final_answer_line", choices=PROMPT_STYLE_CHOICES)
    parser.add_argument("--batch-size", type=int, default=128, help="Prompts per vLLM generate call.")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-num-seqs", type=int, default=128)
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
    parser.add_argument("--compare-dir", action="append", default=[], help="Transformers output dir to compare against.")
    parser.add_argument("--output-dir", type=str, default=str(BUNDLE_ROOT / "results" / "gsm8k_vllm_eval"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
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


def load_jsonl_rows(path: str | Path) -> List[dict]:
    resolved = Path(path).expanduser().resolve()
    rows: List[dict] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
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


def rows_to_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _build_eval_prompt(question: str, *, prompt_style: str) -> str:
    text = str(question).rstrip()
    if prompt_style == "raw_question":
        return text
    if prompt_style == "final_answer_line":
        return (
            f"{text}\n\n"
            "Solve the problem carefully. End your response with a single line in the form:\n"
            "Final answer: <number>"
        )
    raise ValueError(f"Unsupported prompt style '{prompt_style}'.")


def _normalize_numeric_token(text: str) -> Optional[str]:
    candidate = str(text).strip()
    if not candidate:
        return None
    candidate = candidate.replace("$", "").replace(",", "")
    if candidate.startswith("+"):
        candidate = candidate[1:]
    if candidate.startswith("-."):
        candidate = "-0" + candidate[1:]
    elif candidate.startswith("."):
        candidate = "0" + candidate
    if "/" in candidate:
        numerator_text, denominator_text = candidate.split("/", 1)
        try:
            numerator = Decimal(numerator_text)
            denominator = Decimal(denominator_text)
        except InvalidOperation:
            return None
        if denominator == 0:
            return None
        value = numerator / denominator
    else:
        try:
            value = Decimal(candidate)
        except InvalidOperation:
            return None
    if value == 0:
        value = abs(value)
    if value == value.to_integral():
        return str(value.quantize(Decimal("1")))
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def _extract_gold_answer(text: str) -> Optional[str]:
    match = GOLD_ANSWER_RE.search(str(text).strip())
    if match is None:
        return _normalize_numeric_token(text)
    return _normalize_numeric_token(match.group(1))


def _extract_prediction_answer(text: str) -> Optional[str]:
    stripped = str(text).strip()
    if not stripped:
        return None
    final_match = None
    for match in FINAL_ANSWER_RE.finditer(stripped):
        final_match = match
    if final_match is not None:
        normalized = _normalize_numeric_token(final_match.group(1))
        if normalized is not None:
            return normalized
    last_number = None
    for match in NUMBER_RE.finditer(stripped):
        last_number = match.group(0)
    if last_number is None:
        return None
    return _normalize_numeric_token(last_number)


def _load_items(args: argparse.Namespace) -> List[dict]:
    items = load_jsonl_rows(args.gsm8k_path)
    if int(args.offset) > 0:
        items = items[int(args.offset) :]
    if int(args.max_items) > 0:
        items = items[: int(args.max_items)]
    if not items:
        raise RuntimeError(f"No GSM8K rows were loaded from {Path(args.gsm8k_path).resolve()}.")
    return items


def _format_messages(tokenizer, messages: Sequence[Mapping[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
    parts: List[str] = []
    for message in messages:
        parts.append(f"[{str(message['role']).upper()}]\n{str(message['content'])}")
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


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


def _load_existing_method_rows(path: Path, *, item_count: int) -> List[dict]:
    if not path.exists():
        return []
    rows_by_index: Dict[int, dict] = {}
    for row in load_jsonl_rows(path):
        try:
            item_index = int(row.get("item_index"))
        except Exception:
            continue
        if 0 <= item_index < int(item_count):
            rows_by_index[item_index] = dict(row)
    return [rows_by_index[index] for index in sorted(rows_by_index)]


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
    question = str(item.get("question", "")).strip()
    gold_answer = str(item.get("answer", "")).strip()
    gold_final_answer = _extract_gold_answer(item.get("gold_final_answer", "") or gold_answer)
    pred_answer = _extract_prediction_answer(generation_text)
    is_correct = bool(pred_answer is not None and gold_final_answer is not None and pred_answer == gold_final_answer)
    return {
        "trait": str(trait_name),
        "target_pole": str(target_pole),
        "method": str(method),
        "item_index": int(item_index),
        "question": question,
        "gold_answer": gold_answer,
        "gold_final_answer": gold_final_answer,
        "pred_answer": pred_answer,
        "is_correct": bool(is_correct),
        "formatted_prompt": formatted_prompt,
        "generation_text": generation_text,
        "selector_artifact_path": "",
        "engine": "vllm",
    }


def _iter_chunks(items: Sequence[tuple[int, Mapping[str, object]]], chunk_size: int) -> List[List[tuple[int, Mapping[str, object]]]]:
    size = max(1, int(chunk_size))
    return [list(items[offset : offset + size]) for offset in range(0, len(items), size)]


def _write_method_outputs(method_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    write_jsonl(method_dir / "generations.jsonl", rows)
    rows_to_csv(method_dir / "generations.csv", rows)


def _compare_rows(
    *,
    output_dir: Path,
    compare_dirs: Sequence[Path],
    target_pole: str,
    summaries: Sequence[Mapping[str, object]],
) -> None:
    if not compare_dirs:
        return
    diff_rows: List[dict] = []
    row_diff_rows: List[dict] = []
    for summary in summaries:
        trait_name = str(summary["trait"])
        method = str(summary["method"])
        vllm_path = output_dir / trait_name / str(target_pole) / method / "generations.jsonl"
        vllm_rows = {int(row["item_index"]): row for row in load_jsonl_rows(vllm_path)}
        hf_rows: Dict[int, dict] = {}
        for compare_dir in compare_dirs:
            hf_path = compare_dir / trait_name / str(target_pole) / method / "generations.jsonl"
            if not hf_path.exists():
                continue
            for row in load_jsonl_rows(hf_path):
                hf_rows.setdefault(int(row["item_index"]), row)
        compared_indices = sorted(set(vllm_rows).intersection(hf_rows))
        vllm_correct = sum(int(bool(vllm_rows[index].get("is_correct"))) for index in compared_indices)
        hf_correct = sum(int(bool(hf_rows[index].get("is_correct"))) for index in compared_indices)
        prediction_matches = sum(
            int(str(vllm_rows[index].get("pred_answer")) == str(hf_rows[index].get("pred_answer")))
            for index in compared_indices
        )
        for index in compared_indices:
            row_diff_rows.append(
                {
                    "trait": trait_name,
                    "target_pole": str(target_pole),
                    "method": method,
                    "item_index": int(index),
                    "vllm_pred_answer": vllm_rows[index].get("pred_answer"),
                    "transformers_pred_answer": hf_rows[index].get("pred_answer"),
                    "gold_final_answer": vllm_rows[index].get("gold_final_answer"),
                    "vllm_is_correct": bool(vllm_rows[index].get("is_correct")),
                    "transformers_is_correct": bool(hf_rows[index].get("is_correct")),
                    "prediction_match": str(vllm_rows[index].get("pred_answer")) == str(hf_rows[index].get("pred_answer")),
                }
            )
        compared_total = len(compared_indices)
        diff_rows.append(
            {
                "trait": trait_name,
                "target_pole": str(target_pole),
                "method": method,
                "vllm_total": int(len(vllm_rows)),
                "transformers_total": int(len(hf_rows)),
                "compared_total": int(compared_total),
                "vllm_accuracy_on_compared": float(vllm_correct / max(compared_total, 1)),
                "transformers_accuracy_on_compared": float(hf_correct / max(compared_total, 1)),
                "accuracy_delta_vllm_minus_transformers": float((vllm_correct - hf_correct) / max(compared_total, 1)),
                "prediction_match_rate": float(prediction_matches / max(compared_total, 1)),
                "prediction_matches": int(prediction_matches),
            }
        )
    rows_to_csv(output_dir / "score_diff_vs_transformers.csv", diff_rows)
    rows_to_csv(output_dir / "prediction_diff_vs_transformers.csv", row_diff_rows)
    write_json(output_dir / "score_diff_vs_transformers.json", diff_rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", vars(args))

    items = _load_items(args)
    traits = resolve_style_traits_with_aliases(args.target_traits)

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install vLLM in this environment before running this script.") from exc

    model_source_path = Path(str(args.model_name)).expanduser()
    if model_source_path.exists():
        model_source = str(model_source_path.resolve())
        cache_dir = None
        local_files_only = True
    else:
        model_source = str(args.model_name)
        cache_dir, local_files_only = resolve_hf_cache_dir(str(args.model_name))
    tokenizer_kwargs = {}
    if cache_dir is not None:
        tokenizer_kwargs["cache_dir"] = cache_dir
    if local_files_only:
        tokenizer_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
    wrap_qwen_tokenizer_apply_chat_template(tokenizer, enable_thinking=bool(args.qwen_enable_thinking))
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = {
        "model": model_source,
        "dtype": str(args.dtype),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_num_seqs": int(args.max_num_seqs),
        "trust_remote_code": True,
    }
    if cache_dir is not None:
        llm_kwargs["download_dir"] = cache_dir
    if int(args.max_model_len) > 0:
        llm_kwargs["max_model_len"] = int(args.max_model_len)
    print(f"[vllm] loading {model_source} cache_dir={cache_dir} local_files_only={local_files_only}", flush=True)
    llm = LLM(**llm_kwargs)
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

            pending_items = [
                (item_index, item)
                for item_index, item in enumerate(items)
                if int(item_index) not in completed_indices
            ]
            chunks = _iter_chunks(pending_items, int(args.batch_size))
            for chunk_index, chunk in enumerate(chunks, start=1):
                prompts: List[str] = []
                for _, item in chunk:
                    question = str(item.get("question", "")).strip()
                    prompt_text = _build_eval_prompt(question, prompt_style=str(args.prompt_style))
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
                    method_rows.append(row)
                    completed_indices.add(int(item_index))
                method_rows.sort(key=lambda value: int(value["item_index"]))
                _write_method_outputs(method_dir, method_rows)
            summary = _summary_row(
                trait_name=str(trait.name),
                target_pole=str(args.target_pole),
                method=str(method),
                rows=method_rows,
            )
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
    _compare_rows(
        output_dir=output_dir,
        compare_dirs=[Path(path).expanduser().resolve() for path in args.compare_dir],
        target_pole=str(args.target_pole),
        summaries=all_summaries,
    )


if __name__ == "__main__":
    main()
