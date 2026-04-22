#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
BUNDLE_ROOT = PACKAGE_ROOT.parent
for candidate in (SCRIPT_DIR, PACKAGE_ROOT, BUNDLE_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from qwen3_moe.evals.style_eval_lib import (
    METHOD_CHOICES,
    add_common_eval_args,
    build_method_messages,
    build_patch_for_method,
    bundle_root,
    ensure_style_trait_family,
    iter_with_progress,
    load_jsonl_rows,
    load_model_stack,
    maybe_run_selector,
    resolve_style_traits_with_aliases,
    rows_to_csv,
    write_json,
    write_jsonl,
)
from long_context.style_track_a import generate_from_messages


PROMPT_STYLE_CHOICES = ("raw_question", "final_answer_line")
GOLD_ANSWER_RE = re.compile(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?)\s*$")
FINAL_ANSWER_RE = re.compile(
    r"final answer\s*:\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?)",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?")
getcontext().prec = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GSM8K under style steering with plain, prompt, post-attention CAA, "
            "and canonical memory. The bundle defaults to style traits and local JSONL data."
        )
    )
    add_common_eval_args(
        parser,
        default_output_dir=str(bundle_root() / "results" / "gsm8k_style_eval"),
    )
    parser.add_argument(
        "--gsm8k-path",
        type=str,
        default=str(bundle_root() / "datasets" / "gsm8k" / "test.jsonl"),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0, help="0 means the full file after offset.")
    parser.add_argument("--prompt-style", type=str, default="final_answer_line", choices=PROMPT_STYLE_CHOICES)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing per-method generations.jsonl rows and continue missing item indices.",
    )
    return parser.parse_args()


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


def _summary_row(
    *,
    trait_name: str,
    target_pole: str,
    method: str,
    rows: Sequence[Mapping[str, object]],
    selector_path: Optional[Path],
    selector_layers: Sequence[int],
) -> dict:
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
        "selector_artifact_path": "" if selector_path is None else str(selector_path),
        "selector_layer_ids_json": json.dumps([int(layer_id) for layer_id in selector_layers]),
    }


def _load_existing_method_rows(path: Path, *, item_count: int) -> List[dict]:
    if not path.exists():
        return []
    rows_by_index: Dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            try:
                item_index = int(row.get("item_index"))
            except Exception:
                continue
            if 0 <= item_index < int(item_count):
                rows_by_index[item_index] = dict(row)
    return [rows_by_index[index] for index in sorted(rows_by_index)]


def _write_method_outputs(method_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    write_jsonl(method_dir / "generations.jsonl", rows)
    rows_to_csv(method_dir / "generations.csv", rows)


def main() -> None:
    args = parse_args()
    ensure_style_trait_family(args)

    items = _load_items(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", vars(args))

    model, tokenizer = load_model_stack(args)
    traits = resolve_style_traits_with_aliases(args.target_traits)

    all_rows: List[dict] = []
    all_summaries: List[dict] = []

    for trait in traits:
        trait_dir = output_dir / trait.name / str(args.target_pole)
        selector_result = maybe_run_selector(
            model=model,
            tokenizer=tokenizer,
            trait=trait,
            args=args,
            selector_dir=trait_dir / "selector",
        )
        caches: Dict[str, Dict[tuple, object]] = {}
        trait_rows: List[dict] = []
        trait_summaries: List[dict] = []
        for method in args.methods:
            method_dir = trait_dir / str(method)
            method_dir.mkdir(parents=True, exist_ok=True)
            method_jsonl_path = method_dir / "generations.jsonl"
            method_rows: List[dict] = (
                _load_existing_method_rows(method_jsonl_path, item_count=len(items))
                if bool(args.resume)
                else []
            )
            completed_indices = {int(row["item_index"]) for row in method_rows}
            if method_rows:
                print(
                    f"[resume] {trait.name}/{args.target_pole}/{method}: "
                    f"{len(completed_indices)}/{len(items)} existing rows",
                    flush=True,
                )
                _write_method_outputs(method_dir, method_rows)
            if len(completed_indices) >= len(items):
                print(f"[resume] skipping complete method {trait.name}/{args.target_pole}/{method}", flush=True)
                summary = _summary_row(
                    trait_name=trait.name,
                    target_pole=str(args.target_pole),
                    method=str(method),
                    rows=method_rows,
                    selector_path=None if selector_result is None else selector_result.artifact_path,
                    selector_layers=() if selector_result is None else selector_result.selected_layer_ids,
                )
                trait_rows.extend(method_rows)
                trait_summaries.append(summary)
                all_rows.extend(method_rows)
                continue

            patch = build_patch_for_method(
                method=str(method),
                model=model,
                tokenizer=tokenizer,
                trait=trait,
                args=args,
                selector_result=selector_result,
                caches=caches,
            )
            pending_items = [
                (item_index, item)
                for item_index, item in enumerate(items)
                if int(item_index) not in completed_indices
            ]
            iterator = iter_with_progress(
                pending_items,
                desc=f"gsm8k:{trait.name}:{method}",
                disable=bool(args.disable_progress_bar),
            )
            for item_index, item in iterator:
                question = str(item.get("question", "")).strip()
                gold_answer = str(item.get("answer", "")).strip()
                gold_final_answer = _extract_gold_answer(item.get("gold_final_answer", "") or gold_answer)
                prompt_text = _build_eval_prompt(question, prompt_style=str(args.prompt_style))
                messages = build_method_messages(
                    method=str(method),
                    user_content=prompt_text,
                    trait=trait,
                    args=args,
                )
                generation_text, formatted_prompt = generate_from_messages(
                    model,
                    tokenizer,
                    messages=messages,
                    max_new_tokens=int(args.max_new_tokens),
                    seed=int(args.seed) + int(item_index),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    repetition_penalty=float(args.repetition_penalty),
                    patch=patch,
                )
                pred_answer = _extract_prediction_answer(generation_text)
                is_correct = bool(pred_answer is not None and gold_final_answer is not None and pred_answer == gold_final_answer)
                row = {
                    "trait": str(trait.name),
                    "target_pole": str(args.target_pole),
                    "method": str(method),
                    "item_index": int(item_index),
                    "question": question,
                    "gold_answer": gold_answer,
                    "gold_final_answer": gold_final_answer,
                    "pred_answer": pred_answer,
                    "is_correct": bool(is_correct),
                    "formatted_prompt": formatted_prompt,
                    "generation_text": generation_text,
                    "selector_artifact_path": "" if selector_result is None else str(selector_result.artifact_path),
                }
                method_rows.append(row)
                completed_indices.add(int(item_index))
                method_rows.sort(key=lambda value: int(value["item_index"]))
                _write_method_outputs(method_dir, method_rows)
            _write_method_outputs(method_dir, method_rows)
            summary = _summary_row(
                trait_name=trait.name,
                target_pole=str(args.target_pole),
                method=str(method),
                rows=method_rows,
                selector_path=None if selector_result is None else selector_result.artifact_path,
                selector_layers=() if selector_result is None else selector_result.selected_layer_ids,
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


if __name__ == "__main__":
    main()
