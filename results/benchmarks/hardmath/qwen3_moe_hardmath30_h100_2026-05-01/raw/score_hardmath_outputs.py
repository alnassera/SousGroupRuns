#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from math_memory_banks import load_math_examples


BOXED_RE = re.compile(r"\\boxed\s*\{", re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _find_matching_brace(text: str, open_brace_index: int) -> int:
    depth = 0
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def extract_boxed_answers(text: str) -> List[str]:
    answers: List[str] = []
    for match in BOXED_RE.finditer(str(text)):
        open_index = match.end() - 1
        close_index = _find_matching_brace(str(text), open_index)
        if close_index >= 0:
            answers.append(str(text)[open_index + 1 : close_index].strip())
    return answers


def extract_final_boxed_answer(text: str) -> Optional[str]:
    answers = extract_boxed_answers(text)
    return answers[-1] if answers else None


def _normalize_math_text(value: object) -> str:
    text = str(value if value is not None else "").strip().lower()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\;": "",
        "\\!": "",
        "$": "",
        " ": "",
        "\n": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("\\dfrac", "\\frac")
    text = text.replace("\\tfrac", "\\frac")
    text = text.replace("approx", "")
    text = text.replace("\\approx", "")
    if "=" in text:
        text = text.split("=")[-1]
    return text.strip(" .")


def _reference_answer(raw: Mapping[str, object]) -> Optional[object]:
    for key in ("extracted_answer", "answer_val", "answer", "final_answer"):
        value = raw.get(key)
        if value not in (None, ""):
            return value
    return None


def _to_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    match = NUMBER_RE.search(str(value).replace(",", ""))
    if not match:
        return None
    try:
        parsed = float(match.group(0))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _numeric_tolerance(raw: Mapping[str, object]) -> float:
    try:
        precision = int(raw.get("precision"))
    except Exception:
        precision = 3
    return max(1e-8, 0.5 * (10.0 ** (-max(0, precision))))


def _score_one(predicted: Optional[str], raw: Mapping[str, object]) -> Dict[str, object]:
    reference = _reference_answer(raw)
    has_boxed = predicted is not None and str(predicted).strip() != ""
    exact_match = False
    numeric_match = False
    if has_boxed and reference is not None:
        exact_match = _normalize_math_text(predicted) == _normalize_math_text(reference)
        predicted_float = _to_float(predicted)
        reference_float = _to_float(reference)
        if predicted_float is not None and reference_float is not None:
            tolerance = _numeric_tolerance(raw)
            numeric_match = abs(predicted_float - reference_float) <= tolerance
    return {
        "boxed_answer": predicted,
        "reference_answer": reference,
        "has_boxed_answer": bool(has_boxed),
        "exact_match": bool(exact_match),
        "numeric_match": bool(numeric_match),
        "score": 1.0 if exact_match or numeric_match else 0.0,
    }


def _load_reference_cache(paths: Iterable[str]) -> Dict[Tuple[str, str], Mapping[str, object]]:
    cache: Dict[Tuple[str, str], Mapping[str, object]] = {}
    for path_text in sorted({str(path) for path in paths if path}):
        path = Path(path_text)
        if not path.exists():
            continue
        for example in load_math_examples(path):
            cache[(str(Path(example.source_path).resolve()), str(example.example_id))] = dict(example.raw)
            cache[(str(path.resolve()), str(example.example_id))] = dict(example.raw)
    return cache


def _iter_response_files(run_root: Path) -> List[Path]:
    return sorted(path for path in run_root.rglob("response.jsonl") if path.is_file())


def _bump(group: Dict[str, object], key: str, amount: float = 1.0) -> None:
    group[key] = float(group.get(key, 0.0)) + float(amount)


def _proportion_interval(successes: float, total: float, *, z: float = 1.959963984540054) -> Dict[str, float]:
    if total <= 0:
        return {}
    p = float(successes) / float(total)
    stderr = math.sqrt(max(0.0, p * (1.0 - p)) / float(total))
    denom = 1.0 + (z * z) / float(total)
    center = (p + (z * z) / (2.0 * float(total))) / denom
    radius = (
        z
        * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * float(total))) / float(total))
        / denom
    )
    low = max(0.0, center - radius)
    high = min(1.0, center + radius)
    if low < 1e-15:
        low = 0.0
    if high > 1.0 - 1e-15:
        high = 1.0
    return {
        "stderr": float(stderr),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci95_minus": float(max(0.0, p - low)),
        "ci95_plus": float(max(0.0, high - p)),
    }


def _add_rate_stats(out: Dict[str, object], group: Mapping[str, object], count_key: str, rate_key: str, total: float) -> None:
    count = float(group.get(count_key, 0.0))
    out[rate_key] = count / total
    for suffix, value in _proportion_interval(count, total).items():
        out[f"{rate_key}_{suffix}"] = value


def score_run_root(run_root: str | Path, *, output_path: str | Path | None = None) -> Mapping[str, object]:
    root = Path(run_root).expanduser().resolve()
    response_files = _iter_response_files(root)
    records: List[Mapping[str, object]] = []
    source_paths: List[str] = []
    for response_file in response_files:
        with response_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                record = json.loads(text)
                records.append(record)
                if record.get("source_path"):
                    source_paths.append(str(record["source_path"]))

    reference_cache = _load_reference_cache(source_paths)
    totals: Dict[str, object] = defaultdict(float)
    by_method: Dict[str, Dict[str, object]] = defaultdict(lambda: defaultdict(float))
    by_category: Dict[str, Dict[str, object]] = defaultdict(lambda: defaultdict(float))
    scored_rows: List[Mapping[str, object]] = []

    for record in records:
        source_path = str(record.get("source_path") or "")
        example_id = str(record.get("id") or "")
        raw = reference_cache.get((str(Path(source_path).resolve()) if source_path else "", example_id), {})
        predicted = extract_final_boxed_answer(str(record.get("llm_answers") or ""))
        score = _score_one(predicted, raw)
        category = str(record.get("category") or record.get("subject") or "unknown")
        method_tag = str(record.get("method_tag") or record.get("method") or "unknown")
        hit_max = bool((record.get("generation_metadata") or {}).get("hit_max_new_tokens"))

        for group in (totals, by_method[method_tag], by_category[category]):
            _bump(group, "generated")
            _bump(group, "boxed", 1.0 if score["has_boxed_answer"] else 0.0)
            _bump(group, "exact_match", 1.0 if score["exact_match"] else 0.0)
            _bump(group, "numeric_match", 1.0 if score["numeric_match"] else 0.0)
            _bump(group, "correct", 1.0 if score["score"] else 0.0)
            _bump(group, "missing_box", 0.0 if score["has_boxed_answer"] else 1.0)
            _bump(group, "hit_max_tokens", 1.0 if hit_max else 0.0)

        scored_rows.append(
            {
                "id": example_id,
                "category": category,
                "method_tag": method_tag,
                **score,
            }
        )

    def finalize(group: Mapping[str, object]) -> Dict[str, object]:
        generated = float(group.get("generated", 0.0))
        out = {key: int(value) if float(value).is_integer() else float(value) for key, value in group.items()}
        if generated > 0:
            _add_rate_stats(out, group, "boxed", "boxed_rate", generated)
            _add_rate_stats(out, group, "exact_match", "exact_match_rate", generated)
            _add_rate_stats(out, group, "numeric_match", "numeric_match_rate", generated)
            _add_rate_stats(out, group, "correct", "accuracy", generated)
            _add_rate_stats(out, group, "missing_box", "missing_box_rate", generated)
            _add_rate_stats(out, group, "hit_max_tokens", "hit_max_tokens_rate", generated)
        return out

    summary = {
        "run_root": str(root),
        "response_file_count": len(response_files),
        "totals": finalize(totals),
        "by_method": {key: finalize(value) for key, value in sorted(by_method.items())},
        "by_category": {key: finalize(value) for key, value in sorted(by_category.items())},
        "scored_rows_preview": scored_rows[:20],
        "scoring_note": (
            "Deterministic local boxed-answer scoring; not a GPT-grader or official HARDMath score. "
            "Rate intervals are Wilson 95% binomial intervals over generated rows."
        ),
    }
    target = Path(output_path).expanduser().resolve() if output_path else root / "score_summary.json"
    target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score HARDMath response.jsonl outputs.")
    parser.add_argument("run_root", type=str)
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()
    summary = score_run_root(args.run_root, output_path=args.output_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
