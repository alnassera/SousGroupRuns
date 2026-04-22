#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
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


LETTER_CHOICES = ("A", "B", "C", "D")
FINAL_ANSWER_RE = re.compile(r"(?:final answer|answer)\s*[:\-]?\s*([ABCD])\b", flags=re.IGNORECASE)
LETTER_RE = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MMLU under style steering with plain, prompt, post-attention CAA, "
            "and canonical memory. Uses local JSONL dev/test files."
        )
    )
    add_common_eval_args(
        parser,
        default_output_dir=str(bundle_root() / "results" / "mmlu_style_eval"),
    )
    parser.add_argument(
        "--mmlu-dev-path",
        type=str,
        default=str(bundle_root() / "datasets" / "mmlu" / "dev.jsonl"),
    )
    parser.add_argument(
        "--mmlu-test-path",
        type=str,
        default=str(bundle_root() / "datasets" / "mmlu" / "test.jsonl"),
    )
    parser.add_argument("--few-shot-k", type=int, default=5)
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0, help="0 means the full filtered test set after offset.")
    return parser.parse_args()


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
            raise RuntimeError(
                f"Missing MMLU file: {path}. Populate the bundled placeholders with "
                "`python datasets/scripts/prepare_mmlu_jsonl.py` and rerun."
            )
        return []
    rows = load_jsonl_rows(path)
    if required and not rows:
        raise RuntimeError(
            f"MMLU file is empty: {path}. This bundle did not have a local MMLU cache when it was packaged. "
            "Populate it with `python datasets/scripts/prepare_mmlu_jsonl.py` and rerun."
        )
    normalized_rows: List[dict] = []
    for row in rows:
        normalized_rows.append(
            {
                "subject": str(row.get("subject", "")).strip(),
                "question": str(row.get("question", "")).strip(),
                "choices": _resolve_choices(row.get("choices", [])),
                "answer": _normalize_answer(row.get("answer")),
            }
        )
    return normalized_rows


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


def main() -> None:
    args = parse_args()
    ensure_style_trait_family(args)

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

    dev_index = _build_subject_index(dev_rows)
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
            patch = build_patch_for_method(
                method=str(method),
                model=model,
                tokenizer=tokenizer,
                trait=trait,
                args=args,
                selector_result=selector_result,
                caches=caches,
            )
            method_rows: List[dict] = []
            iterator = iter_with_progress(
                test_rows,
                desc=f"mmlu:{trait.name}:{method}",
                disable=bool(args.disable_progress_bar),
            )
            for item_index, item in enumerate(iterator):
                few_shot_rows = _select_few_shot_rows(
                    subject=str(item.get("subject", "")),
                    dev_index=dev_index,
                    all_dev_rows=dev_rows,
                    few_shot_k=int(args.few_shot_k),
                )
                prompt_text = _build_eval_prompt(item, few_shot_rows=few_shot_rows)
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
                gold_answer = _normalize_answer(item.get("answer"))
                pred_answer = _extract_prediction_answer(generation_text)
                is_correct = bool(pred_answer is not None and gold_answer is not None and pred_answer == gold_answer)
                row = {
                    "trait": str(trait.name),
                    "target_pole": str(args.target_pole),
                    "method": str(method),
                    "item_index": int(item_index),
                    "subject": str(item.get("subject", "")),
                    "question": str(item.get("question", "")),
                    "choices_json": json.dumps(_resolve_choices(item.get("choices", [])), ensure_ascii=False),
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "is_correct": bool(is_correct),
                    "formatted_prompt": formatted_prompt,
                    "generation_text": generation_text,
                    "selector_artifact_path": "" if selector_result is None else str(selector_result.artifact_path),
                }
                method_rows.append(row)
            write_jsonl(method_dir / "generations.jsonl", method_rows)
            rows_to_csv(method_dir / "generations.csv", method_rows)
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
