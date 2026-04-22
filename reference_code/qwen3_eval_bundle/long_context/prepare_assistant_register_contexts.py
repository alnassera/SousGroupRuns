#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = REPO_ROOT.parent
for candidate in (PACKAGE_PARENT, REPO_ROOT, SCRIPT_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from long_context.judge import maybe_configure_openai_key
from long_context.neutral_contexts import (
    clean_neutral_context_text,
    default_assistant_register_context_manifest_path,
    default_assistant_register_context_text_dir,
    default_assistant_register_stylized_context_text_dir,
)
from long_context.style_traits import default_profile_json_path, resolve_style_track_memory_bank, resolve_style_traits

MIN_STYLIZED_LENGTH_RATIO = 0.60
MAX_STYLIZED_LENGTH_RATIO = 1.25


def _min_stylized_length_ratio(*, trait_name: str, pole: str) -> float:
    if str(trait_name).strip().lower() == "dismissive" and str(pole).strip().lower() == "high":
        return 0.30
    return MIN_STYLIZED_LENGTH_RATIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate assistant-register long-context reference texts and optional stylized variants "
            "for the dialogue persistence interference benchmark."
        )
    )
    parser.add_argument("--manifest", type=str, default=str(default_assistant_register_context_manifest_path()))
    parser.add_argument("--text-dir", type=str, default=str(default_assistant_register_context_text_dir()))
    parser.add_argument("--stylized-output-dir", type=str, default=str(default_assistant_register_stylized_context_text_dir()))
    parser.add_argument("--target-traits", type=str, nargs="+", default=["warm", "formal", "assertive", "cautious", "dismissive", "anxious"])
    parser.add_argument("--poles", type=str, nargs="+", default=["high", "low"], choices=["high", "low"])
    parser.add_argument("--track-memory-bank-profile", type=str, default="default")
    parser.add_argument("--track-memory-bank-profile-file", type=str, default=str(default_profile_json_path()))
    parser.add_argument("--base-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--rewrite-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--response-max-tokens", type=int, default=8192)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--source-ids", type=str, nargs="*", default=None)
    parser.add_argument("--overwrite-base", action="store_true")
    parser.add_argument("--overwrite-stylized", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-stylized", action="store_true")
    parser.add_argument("--openai-api-key", type=str, default=None)
    return parser.parse_args()


def _load_manifest_rows(path: str | Path) -> List[Dict[str, str]]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    rows = payload.get("sources")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected top-level 'sources' list.")
    out: List[Dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id", "")).strip()
        title = str(row.get("title", "")).strip()
        topic_prompt = str(row.get("topic_prompt", "")).strip()
        if not source_id or not title or not topic_prompt:
            raise ValueError(f"{path}: each source requires source_id, title, and topic_prompt.")
        out.append({str(key): str(value) for key, value in row.items()})
    return out


def _build_base_prompt(*, title: str, description: str, topic_prompt: str) -> str:
    return f"""
You are writing a long standalone reference answer in the register of a capable assistant.

Title:
{title}

Topic brief:
{description}

User scenario:
{topic_prompt}

Requirements:
- Write as if a helpful assistant is answering directly to a user, but do not format it as a dialogue.
- Keep the tone neutral, calm, practical, and moderately professional.
- Do not use greetings, sign-offs, first-person AI disclaimers, or role labels like USER/ASSISTANT.
- Keep the content unrelated to therapy, relationships, politics, religion, or career coaching.
- Make the answer useful: include tradeoffs, concrete steps, common mistakes, and realistic caveats.
- Use short section headings plus paragraphs or bullet lists where helpful.
- Keep the answer long enough to work as long-context material. Aim for roughly 1100-1600 words.
- Preserve a helpful assistant register throughout so the text feels closer to an actual model answer than to an encyclopedia page.

Return ONLY valid JSON:
{{
  "assistant_reference_text": <string>
}}
""".strip()


def _build_stylized_prompt(
    *,
    source_text: str,
    descriptor: str,
    trait_name: str,
    pole: str,
    qa_feedback: str = "",
) -> str:
    qa_block = ""
    min_length_ratio = _min_stylized_length_ratio(trait_name=trait_name, pole=pole)
    if qa_feedback.strip():
        qa_block = f"""

Additional QA feedback from the previous attempt:
{qa_feedback}

Fix those issues on this attempt.
""".rstrip()
    return f"""
You are rewriting a long standalone assistant answer into a controlled interaction style variant.

Target style trait: {trait_name}
Target pole: {pole}
Target style descriptor:
{descriptor}

Rewrite requirements:
- Preserve the factual content, recommended actions, examples, and topic coverage of the source text.
- Keep the document in assistant-answer register rather than encyclopedia or news style.
- Do not turn it into a dialogue and do not add role labels.
- Make the target style obvious and sustained across most paragraphs.
- Bias wording, framing, sentence rhythm, hedging, directness, and emotional tone toward the target style.
- Keep the rewritten text roughly similar in length to the source. Do not aggressively shorten it.
- Do not invent facts or add topic content that is not already supported by the source.
- Preserve every numeric token from the source exactly.
- If the source uses numbered lists or numbered step markers, preserve those exact numbers and ordering.
- Keep the rewritten character length between {min_length_ratio:.2f}x and {MAX_STYLIZED_LENGTH_RATIO:.2f}x of the source.
- Do not collapse multi-step advice into a short summary.
{qa_block}

Return ONLY valid JSON:
{{
  "rewritten_text": <string>
}}

SOURCE TEXT:
{source_text}
""".strip()


async def _request_json_text(
    *,
    client,
    semaphore: asyncio.Semaphore,
    model: str,
    prompt: str,
    response_field: str,
    response_max_tokens: int,
    max_attempts: int,
) -> str:
    last_error = "unknown_error"
    for attempt in range(max(1, int(max_attempts))):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=int(response_max_tokens),
                )
            raw = response.choices[0].message.content or ""
            payload = json.loads(raw)
            text = clean_neutral_context_text(str(payload.get(response_field, "")).strip())
            if not text:
                raise RuntimeError(f"Empty field '{response_field}'.")
            return text
        except Exception as exc:  # pragma: no cover - network/API variability
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max(1, int(max_attempts)) - 1:
                break
            await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"Failed after {max(1, int(max_attempts))} attempts: {last_error}")


def _digit_signature(text: str) -> List[str]:
    return [token for token in __import__("re").findall(r"\d+(?:\.\d+)?", str(text))]


def _stylized_qa_issues(*, source_text: str, rewritten_text: str, trait_name: str, pole: str) -> List[str]:
    issues: List[str] = []
    length_ratio = len(rewritten_text) / max(1, len(source_text))
    min_length_ratio = _min_stylized_length_ratio(trait_name=trait_name, pole=pole)
    if length_ratio < min_length_ratio:
        issues.append(
            f"Length ratio {length_ratio:.3f} is below the minimum {min_length_ratio:.2f}; preserve substantially more detail."
        )
    if length_ratio > MAX_STYLIZED_LENGTH_RATIO:
        issues.append(
            f"Length ratio {length_ratio:.3f} is above the maximum {MAX_STYLIZED_LENGTH_RATIO:.2f}; stay closer to the source length."
        )
    if _digit_signature(source_text) != _digit_signature(rewritten_text):
        issues.append("Numeric tokens do not match the source exactly; preserve every number verbatim.")
    return issues


async def _request_stylized_text(
    *,
    client,
    semaphore: asyncio.Semaphore,
    model: str,
    source_text: str,
    descriptor: str,
    trait_name: str,
    pole: str,
    response_max_tokens: int,
    max_attempts: int,
) -> str:
    last_error = "unknown_error"
    qa_feedback = ""
    for attempt in range(max(1, int(max_attempts))):
        try:
            prompt = _build_stylized_prompt(
                source_text=source_text,
                descriptor=descriptor,
                trait_name=trait_name,
                pole=pole,
                qa_feedback=qa_feedback,
            )
            text = await _request_json_text(
                client=client,
                semaphore=semaphore,
                model=model,
                prompt=prompt,
                response_field="rewritten_text",
                response_max_tokens=response_max_tokens,
                max_attempts=1,
            )
            issues = _stylized_qa_issues(
                source_text=source_text,
                rewritten_text=text,
                trait_name=trait_name,
                pole=pole,
            )
            if issues:
                qa_feedback = "\n".join(f"- {issue}" for issue in issues)
                raise RuntimeError(f"Stylized QA validation failed: {qa_feedback}")
            return text
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max(1, int(max_attempts)) - 1:
                break
            await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"Failed after {max(1, int(max_attempts))} attempts: {last_error}")


async def main_async(args: argparse.Namespace) -> None:
    try:
        from openai import AsyncOpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required to prepare assistant-register context assets.") from exc

    maybe_configure_openai_key(args.openai_api_key)
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max(1, int(args.max_concurrent)))
    manifest_rows = _load_manifest_rows(args.manifest)
    if args.source_ids:
        allowed = {str(source_id).strip() for source_id in args.source_ids if str(source_id).strip()}
        manifest_rows = [row for row in manifest_rows if str(row["source_id"]) in allowed]
    if not manifest_rows:
        raise SystemExit("No assistant-register sources selected.")

    text_dir = Path(args.text_dir).expanduser().resolve()
    stylized_root = Path(args.stylized_output_dir).expanduser().resolve()
    text_dir.mkdir(parents=True, exist_ok=True)
    stylized_root.mkdir(parents=True, exist_ok=True)

    try:
        failures: List[str] = []
        if not bool(args.skip_base):
            base_tasks: List[tuple[asyncio.Task[str], Dict[str, str]]] = []
            for row in manifest_rows:
                out_path = text_dir / f"{row['source_id']}.txt"
                metadata_path = text_dir / f"{row['source_id']}.metadata.json"
                if out_path.is_file() and metadata_path.is_file() and not bool(args.overwrite_base):
                    continue
                prompt = _build_base_prompt(
                    title=str(row["title"]),
                    description=str(row.get("description", "")),
                    topic_prompt=str(row["topic_prompt"]),
                )
                task = asyncio.create_task(
                    _request_json_text(
                        client=client,
                        semaphore=semaphore,
                        model=str(args.base_model),
                        prompt=prompt,
                        response_field="assistant_reference_text",
                        response_max_tokens=int(args.response_max_tokens),
                        max_attempts=int(args.max_attempts),
                    )
                )
                base_tasks.append((task, row))

            for task, row in base_tasks:
                try:
                    base_text = await task
                except Exception as exc:
                    message = f"base source_id={row['source_id']}: {type(exc).__name__}: {exc}"
                    print(f"[assistant-base-error] {message}", flush=True)
                    failures.append(message)
                    continue
                out_path = text_dir / f"{row['source_id']}.txt"
                metadata_path = text_dir / f"{row['source_id']}.metadata.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(base_text + "\n", encoding="utf-8")
                metadata = {
                    "source_id": str(row["source_id"]),
                    "title": str(row["title"]),
                    "url": str(row.get("url", "")),
                    "source": str(row.get("source", "")),
                    "license": str(row.get("license", "")),
                    "description": str(row.get("description", "")),
                    "topic_prompt": str(row.get("topic_prompt", "")),
                    "base_model": str(args.base_model),
                    "base_sha1": hashlib.sha1(base_text.encode("utf-8")).hexdigest(),
                    "base_char_count": len(base_text),
                }
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                print(
                    f"[assistant-base] source_id={row['source_id']} chars={len(base_text)}",
                    flush=True,
                )

        if bool(args.skip_stylized):
            return

        traits = resolve_style_traits(args.target_traits)
        base_records: List[Dict[str, str]] = []
        for row in manifest_rows:
            text_path = text_dir / f"{row['source_id']}.txt"
            if not text_path.is_file():
                raise FileNotFoundError(f"Missing base assistant-register context text: {text_path}")
            base_records.append(
                {
                    "source_id": str(row["source_id"]),
                    "title": str(row["title"]),
                    "url": str(row.get("url", "")),
                    "source": str(row.get("source", "")),
                    "license": str(row.get("license", "")),
                    "description": str(row.get("description", "")),
                    "topic_prompt": str(row.get("topic_prompt", "")),
                    "text": clean_neutral_context_text(text_path.read_text(encoding="utf-8")),
                }
            )

        stylized_tasks: List[tuple[asyncio.Task[str], Dict[str, str]]] = []
        for trait in traits:
            for pole in args.poles:
                bank = resolve_style_track_memory_bank(
                    trait,
                    track="generation",
                    pole=str(pole),
                    bank_profile=str(args.track_memory_bank_profile),
                    bank_profile_file=str(args.track_memory_bank_profile_file),
                )
                for record in base_records:
                    out_path = stylized_root / trait.name / str(pole) / f"{record['source_id']}.txt"
                    metadata_path = stylized_root / trait.name / str(pole) / f"{record['source_id']}.metadata.json"
                    if out_path.is_file() and metadata_path.is_file() and not bool(args.overwrite_stylized):
                        continue
                    task = asyncio.create_task(
                        _request_stylized_text(
                            client=client,
                            semaphore=semaphore,
                            model=str(args.rewrite_model),
                            source_text=str(record["text"]),
                            descriptor=str(bank.descriptor),
                            trait_name=str(trait.name),
                            pole=str(pole),
                            response_max_tokens=int(args.response_max_tokens),
                            max_attempts=int(args.max_attempts),
                        )
                    )
                    stylized_tasks.append(
                        (
                            task,
                            {
                                "source_id": str(record["source_id"]),
                                "title": str(record["title"]),
                                "url": str(record["url"]),
                                "source": str(record["source"]),
                                "license": str(record["license"]),
                                "description": str(record["description"]),
                                "topic_prompt": str(record["topic_prompt"]),
                                "trait_name": str(trait.name),
                                "pole": str(pole),
                                "descriptor": str(bank.descriptor),
                                "source_text": str(record["text"]),
                            },
                        )
                    )

        for task, meta in stylized_tasks:
            try:
                rewritten_text = await task
            except Exception as exc:
                message = (
                    f"stylized trait={meta['trait_name']} pole={meta['pole']} source_id={meta['source_id']}: "
                    f"{type(exc).__name__}: {exc}"
                )
                print(f"[assistant-stylized-error] {message}", flush=True)
                failures.append(message)
                continue
            out_path = stylized_root / meta["trait_name"] / meta["pole"] / f"{meta['source_id']}.txt"
            metadata_path = stylized_root / meta["trait_name"] / meta["pole"] / f"{meta['source_id']}.metadata.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rewritten_text + "\n", encoding="utf-8")
            source_text = str(meta["source_text"])
            metadata = {
                "source_id": str(meta["source_id"]),
                "title": str(meta["title"]),
                "url": str(meta["url"]),
                "source": str(meta["source"]),
                "license": str(meta["license"]),
                "description": str(meta["description"]),
                "topic_prompt": str(meta["topic_prompt"]),
                "trait_name": str(meta["trait_name"]),
                "pole": str(meta["pole"]),
                "rewrite_model": str(args.rewrite_model),
                "source_sha1": hashlib.sha1(source_text.encode("utf-8")).hexdigest(),
                "rewritten_sha1": hashlib.sha1(rewritten_text.encode("utf-8")).hexdigest(),
                "source_char_count": len(source_text),
                "rewritten_char_count": len(rewritten_text),
                "length_ratio": (len(rewritten_text) / max(1, len(source_text))),
                "digit_signature_match": _digit_signature(source_text) == _digit_signature(rewritten_text),
                "descriptor": str(meta["descriptor"]),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(
                f"[assistant-stylized] trait={meta['trait_name']} pole={meta['pole']} source_id={meta['source_id']} "
                f"ratio={metadata['length_ratio']:.3f}",
                flush=True,
            )
        if failures:
            preview = "; ".join(failures[:5])
            raise RuntimeError(f"{len(failures)} generation tasks failed. First failures: {preview}")
    finally:
        await client.close()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
