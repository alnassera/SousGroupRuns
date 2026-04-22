#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


LETTER_CHOICES = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official Hugging Face MMLU dataset and write bundle-local "
            "JSONL files for the dev and test splits."
        )
    )
    parser.add_argument("--dataset-name", type=str, default="cais/mmlu")
    parser.add_argument(
        "--config-name",
        type=str,
        default="all",
        help="The official cais/mmlu dataset exposes an 'all' config that concatenates subjects.",
    )
    parser.add_argument("--dev-split", type=str, default="dev")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "mmlu"),
    )
    return parser.parse_args()


def _normalize_answer(value: object) -> str:
    if isinstance(value, int):
        if 0 <= value < len(LETTER_CHOICES):
            return LETTER_CHOICES[int(value)]
        raise ValueError(f"Integer answer {value} is outside the expected 0-3 range.")
    text = str(value).strip().upper()
    if text in LETTER_CHOICES:
        return text
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(LETTER_CHOICES):
            return LETTER_CHOICES[index]
    raise ValueError(f"Unsupported MMLU answer value: {value!r}")


def _write_split(rows, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "subject": str(row.get("subject", "")).strip(),
                "question": str(row.get("question", "")).strip(),
                "choices": [str(choice) for choice in row.get("choices", [])],
                "answer": _normalize_answer(row.get("answer")),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Please install the datasets package before running this script.") from exc

    dataset = load_dataset(str(args.dataset_name), str(args.config_name))
    output_dir = Path(args.output_dir).expanduser().resolve()
    dev_count = _write_split(dataset[str(args.dev_split)], output_dir / "dev.jsonl")
    test_count = _write_split(dataset[str(args.test_split)], output_dir / "test.jsonl")
    print(
        f"Wrote {dev_count} dev rows and {test_count} test rows to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
