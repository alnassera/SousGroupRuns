from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Dict, List, Sequence, Tuple

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - fallback when bs4 is unavailable
    BeautifulSoup = None


_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINE_RE = re.compile(r"\n{3,}")
_SKIP_ATTR_RE = re.compile(
    r"(?:^|[\s_-])(breadcrumb|footer|sidebar|share|social|search|carousel)(?:$|[\s_-])",
    flags=re.IGNORECASE,
)
_FOCUS_HIGH_RE = re.compile(
    r"(?:^|[\s_-])(main-content|site-main|primary-content|article-body|post-content|entry-content)(?:$|[\s_-])",
    flags=re.IGNORECASE,
)
_FOCUS_MEDIUM_RE = re.compile(
    r"(?:page__content|page__region--content|node-main|science-main-content|article-body|post-content|entry-content|prose|wysiwyg|field-content|content-and-caption)",
    flags=re.IGNORECASE,
)
_FOCUS_LOW_RE = re.compile(r"(?:^|[\s_-])(content|page|body|entry|post)(?:$|[\s_-])", flags=re.IGNORECASE)
_URL_ONLY_RE = re.compile(r"^https?://\S+$", flags=re.IGNORECASE)
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    flags=re.IGNORECASE,
)
_NAVIGATION_KEYWORD_RE = re.compile(
    r"\b("
    r"share|facebook|linkedin|x\.com|email|feedback|suggested searches|view all topics|"
    r"water cycle components|science explorer|mission areas|related content|"
    r"news(?:\s*&\s*events)?|multimedia|podcasts(?:\s*&\s*audio)?|newsletters|social media|"
    r"media resources|upcoming launches|virtual guest program|image of the day"
    r")\b",
    flags=re.IGNORECASE,
)
_CAPTION_HINT_RE = re.compile(
    r"\b("
    r"in this (?:photograph|photo|image|illustration)|"
    r"photo(?:graph)? credit|image credit|illustration credit|credit:|courtesy of"
    r")\b",
    flags=re.IGNORECASE,
)
_ORG_CREDIT_RE = re.compile(
    r"^(nasa|usgs|u\.s\. geological survey|nist|national institute of standards and technology)$",
    flags=re.IGNORECASE,
)
_BOILERPLATE_BLOCK_RE = re.compile(
    r"^(skip to main content|search(?:\s+\w+)?|menu|close|breadcrumb|contents|completed|"
    r"an official website of the united states government|official websites use \.gov|"
    r"secure \.gov websites use https|here(?:'|’)s how you know)$",
    flags=re.IGNORECASE,
)
_BLOCK_SEPARATOR_CHARS = ("•", "·", "|", "»", "›")


@dataclass(frozen=True)
class NeutralContextSource:
    source_id: str
    title: str
    url: str
    source: str
    license: str
    description: str
    format: str = "html"


def default_neutral_context_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "neutral_contexts" / "manifest.json"


def default_neutral_context_text_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "neutral_contexts" / "text"


def default_assistant_register_context_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "assistant_register_contexts" / "manifest.json"


def default_assistant_register_context_text_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "assistant_register_contexts" / "text"


def default_assistant_register_stylized_context_text_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "assistant_register_contexts" / "stylized"


def default_stylized_context_text_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "stylized_contexts"


def default_neutral_context_raw_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "neutral_contexts" / "raw"


def load_neutral_context_manifest(path: str | Path | None = None) -> Tuple[NeutralContextSource, ...]:
    manifest_path = default_neutral_context_manifest_path() if path is None else Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"{manifest_path}: expected top-level 'sources' list.")
    out: List[NeutralContextSource] = []
    for row in sources:
        if not isinstance(row, dict):
            raise ValueError(f"{manifest_path}: each source must be a JSON object.")
        out.append(
            NeutralContextSource(
                source_id=str(row.get("source_id", "")).strip(),
                title=str(row.get("title", "")).strip(),
                url=str(row.get("url", "")).strip(),
                source=str(row.get("source", "")).strip(),
                license=str(row.get("license", "")).strip(),
                description=str(row.get("description", "")).strip(),
                format=str(row.get("format", "html")).strip().lower() or "html",
            )
        )
    missing = [row.source_id for row in out if not row.source_id or not row.title or not row.url]
    if missing:
        raise ValueError(f"{manifest_path}: invalid sources with missing required fields: {missing}")
    return tuple(out)


class _VisibleTextExtractor(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "nav", "footer", "form", "button", "select", "option"}
    _BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "footer",
        "aside",
        "li",
        "ul",
        "ol",
        "table",
        "tr",
        "td",
        "th",
        "br",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._focus_priority_counts: Dict[int, int] = {}
        self._stack: List[Tuple[bool, int, bool]] = []
        self._all_parts: List[str] = []
        self._focus_parts_by_priority: Dict[int, List[str]] = {}

    @staticmethod
    def _attr_values(attrs) -> str:
        values: List[str] = []
        for key, value in attrs:
            if key is None or value is None:
                continue
            key_text = str(key).strip().lower()
            if key_text not in {"id", "class", "role", "aria-label", "data-testid"}:
                continue
            values.append(str(value))
        return " ".join(values)

    @staticmethod
    def _focus_priority(tag: str, attrs) -> int:
        if tag in {"main", "article"}:
            return 3
        attr_text = _VisibleTextExtractor._attr_values(attrs)
        if not attr_text:
            return 0
        if re.search(r"\bmain\b", attr_text, flags=re.IGNORECASE):
            return 3
        if _FOCUS_HIGH_RE.search(attr_text):
            return 3
        if _FOCUS_MEDIUM_RE.search(attr_text):
            return 2
        if _FOCUS_LOW_RE.search(attr_text):
            return 1
        return 0

    @staticmethod
    def _is_skippable_container(tag: str, attrs) -> bool:
        if tag in _VisibleTextExtractor._SKIP_TAGS:
            return True
        attr_text = _VisibleTextExtractor._attr_values(attrs)
        if not attr_text:
            return False
        if tag in {"body", "main", "article"}:
            return False
        if _VisibleTextExtractor._focus_priority(tag, attrs) >= 2:
            return False
        return bool(_SKIP_ATTR_RE.search(attr_text))

    def _append_text(self, text: str) -> None:
        self._all_parts.append(text)
        active_priority = 0
        if self._focus_priority_counts:
            active_priority = max(priority for priority, count in self._focus_priority_counts.items() if count > 0)
        if active_priority > 0:
            self._focus_parts_by_priority.setdefault(active_priority, []).append(text)

    def handle_starttag(self, tag: str, attrs) -> None:
        normalized = str(tag).strip().lower()
        if self._skip_depth > 0:
            self._skip_depth += 1
            self._stack.append((True, False, False))
            return
        if self._is_skippable_container(normalized, attrs):
            self._skip_depth += 1
            self._stack.append((True, False, False))
            return
        focus_priority = self._focus_priority(normalized, attrs)
        is_block = normalized in self._BLOCK_TAGS
        if focus_priority > 0:
            self._focus_priority_counts[focus_priority] = self._focus_priority_counts.get(focus_priority, 0) + 1
        if is_block:
            self._append_text("\n")
        self._stack.append((False, focus_priority, is_block))

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        is_skip, focus_priority, is_block = self._stack.pop()
        if is_skip:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if is_block:
            self._append_text("\n")
        if focus_priority > 0:
            remaining = self._focus_priority_counts.get(focus_priority, 0) - 1
            if remaining > 0:
                self._focus_priority_counts[focus_priority] = remaining
            else:
                self._focus_priority_counts.pop(focus_priority, None)

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = str(data)
        if text.strip():
            self._append_text(text)

    def extracted_text(self) -> str:
        for priority in sorted(self._focus_parts_by_priority.keys(), reverse=True):
            focused_text = _normalize_extracted_text("".join(self._focus_parts_by_priority[priority]))
            if focused_text:
                return focused_text
        return _normalize_extracted_text("".join(self._all_parts))


def _normalize_extracted_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    normalized = _BLANKLINE_RE.sub("\n\n", normalized)
    return normalized.strip()


def _normalize_block_text(block: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(block).strip()).replace("’", "'").lower()


def _block_words(block: str) -> List[str]:
    return str(block).split()


def _is_substantive_body_block(block: str) -> bool:
    if (
        _is_boilerplate_block(block)
        or _is_navigation_like_block(block)
        or _is_probable_title_block(block)
        or _is_probable_meta_block(block)
        or _is_probable_caption_block(block)
        or _is_probable_credit_block(block)
    ):
        return False
    words = block.split()
    if len(words) >= 32:
        return True
    if len(words) >= 18 and re.search(r"[.!?]", block):
        return True
    if len(block) >= 240 and re.search(r"[.!?]", block):
        return True
    return False


def _is_probable_title_block(block: str) -> bool:
    if _URL_ONLY_RE.match(block):
        return False
    if _BOILERPLATE_BLOCK_RE.match(block):
        return False
    words = block.split()
    if not words or len(words) > 18:
        return False
    return 8 <= len(block) <= 140


def _is_probable_meta_block(block: str) -> bool:
    lowered = _normalize_block_text(block)
    if "min read" in lowered or lowered.startswith("by "):
        return True
    if _MONTH_RE.search(block) and re.search(r"\b\d{4}\b", block):
        return True
    return False


def _is_boilerplate_block(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return True
    lowered = _normalize_block_text(stripped)
    if _URL_ONLY_RE.match(stripped):
        return True
    if _BOILERPLATE_BLOCK_RE.match(stripped):
        return True
    if ".gov website belongs to an official government organization" in lowered:
        return True
    if "you've safely connected to the .gov website" in lowered or "you’ve safely connected to the .gov website" in stripped.lower():
        return True
    if "share sensitive information only on official, secure websites" in lowered:
        return True
    if len(stripped.split()) <= 3 and re.search(r"^(search|menu|close|breadcrumb|contents|completed)$", stripped, flags=re.IGNORECASE):
        return True
    return False


def _is_navigation_like_block(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    lowered = _normalize_block_text(stripped)
    words = _block_words(stripped)
    separator_count = sum(stripped.count(char) for char in _BLOCK_SEPARATOR_CHARS)
    has_sentence_punctuation = bool(re.search(r"[.!?]", stripped))
    if _NAVIGATION_KEYWORD_RE.search(lowered):
        if separator_count > 0 or len(words) <= 18 or not has_sentence_punctuation:
            return True
    if separator_count >= 3 and not has_sentence_punctuation:
        return True
    if len(words) >= 18 and not has_sentence_punctuation:
        cleaned_lengths = [
            len(word.strip("()[]{}<>.,;:!?\"'"))
            for word in words
            if word.strip("()[]{}<>.,;:!?\"'")
        ]
        average_length = (sum(cleaned_lengths) / len(cleaned_lengths)) if cleaned_lengths else 0.0
        if average_length <= 9.0:
            return True
    return False


def _is_probable_caption_block(block: str) -> bool:
    stripped = block.strip()
    words = _block_words(stripped)
    if not words or len(words) > 120:
        return False
    return bool(_CAPTION_HINT_RE.search(stripped))


def _is_probable_credit_block(block: str) -> bool:
    stripped = block.strip()
    words = _block_words(stripped)
    if not words or len(words) > 6:
        return False
    return bool(_ORG_CREDIT_RE.match(stripped))


def _is_separator_only_block(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    return all(char in _BLOCK_SEPARATOR_CHARS for char in stripped)


def _is_short_list_like_block(block: str) -> bool:
    stripped = block.strip()
    words = _block_words(stripped)
    if not words or len(words) > 8:
        return False
    if re.search(r"[.!?]", stripped):
        return False
    if _is_probable_meta_block(stripped) or _is_probable_caption_block(stripped) or _is_probable_credit_block(stripped):
        return False
    return len(stripped) <= 80


def _strip_short_list_runs(blocks: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    index = 0
    while index < len(blocks):
        block = str(blocks[index]).strip()
        if _is_separator_only_block(block) or _is_short_list_like_block(block):
            run_end = index
            has_separator = False
            while run_end < len(blocks):
                candidate = str(blocks[run_end]).strip()
                if not (_is_separator_only_block(candidate) or _is_short_list_like_block(candidate)):
                    break
                if _is_separator_only_block(candidate):
                    has_separator = True
                run_end += 1
            run_length = run_end - index
            if run_length >= 4 or (has_separator and run_length >= 2):
                index = run_end
                continue
        cleaned.append(block)
        index += 1
    return cleaned


def _is_inline_fragment_block(block: str) -> bool:
    stripped = block.strip()
    words = _block_words(stripped)
    if not words or len(words) > 4 or len(stripped) > 48:
        return False
    if re.search(r"[.!?]", stripped):
        return False
    if _is_separator_only_block(stripped):
        return False
    if _is_probable_title_block(stripped) or _is_probable_meta_block(stripped):
        return False
    if _is_probable_caption_block(stripped) or _is_probable_credit_block(stripped):
        return False
    return True


def _ends_sentence(block: str) -> bool:
    return bool(re.search(r"[.!?][\"')\\]]?$", block.strip()))


def _starts_inline_continuation(block: str) -> bool:
    stripped = block.lstrip()
    if not stripped:
        return False
    first = stripped[0]
    return first in ",;:)]}" or first.islower()


def _merge_block_text(left: str, right: str) -> str:
    lhs = str(left).rstrip()
    rhs = str(right).lstrip()
    if not lhs:
        return rhs
    if not rhs:
        return lhs
    if rhs[0] in ",;:)]}":
        return lhs + rhs
    return lhs + " " + rhs


def _merge_inline_fragment_blocks(blocks: Sequence[str]) -> List[str]:
    merged: List[str] = []
    index = 0
    while index < len(blocks):
        block = str(blocks[index]).strip()
        if not _is_inline_fragment_block(block):
            merged.append(block)
            index += 1
            continue
        run_end = index
        fragments: List[str] = []
        while run_end < len(blocks) and _is_inline_fragment_block(str(blocks[run_end]).strip()):
            fragments.append(str(blocks[run_end]).strip())
            run_end += 1
        fragment_text = " ".join(fragment for fragment in fragments if fragment)
        next_block = str(blocks[run_end]).strip() if run_end < len(blocks) else ""
        previous_block = merged[-1] if merged else ""
        previous_continues = bool(previous_block) and not _ends_sentence(previous_block)
        next_continues = _starts_inline_continuation(next_block)
        if previous_continues and next_continues:
            merged[-1] = _merge_block_text(_merge_block_text(previous_block, fragment_text), next_block)
            index = run_end + 1
            continue
        if next_continues:
            merged.append(_merge_block_text(fragment_text, next_block))
            index = run_end + 1
            continue
        merged.append(fragment_text)
        index = run_end
    return merged


def _dedupe_blocks(blocks: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for block in blocks:
        normalized = _normalize_block_text(block)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(str(block).strip())
    return deduped


def clean_neutral_context_text(text: str) -> str:
    normalized = _normalize_extracted_text(text)
    if not normalized:
        return ""
    raw_blocks = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    if not raw_blocks:
        return ""

    cleaned_blocks = [
        block
        for block in raw_blocks
        if not _is_boilerplate_block(block) and not _is_navigation_like_block(block)
    ]
    cleaned_blocks = _dedupe_blocks(cleaned_blocks)
    cleaned_blocks = _strip_short_list_runs(cleaned_blocks)
    cleaned_blocks = _merge_inline_fragment_blocks(cleaned_blocks)
    if not cleaned_blocks:
        cleaned_blocks = raw_blocks

    first_content_idx = -1
    for index, block in enumerate(cleaned_blocks):
        if _is_probable_caption_block(block):
            continue
        if _is_substantive_body_block(block):
            first_content_idx = index
            break
    if 0 <= first_content_idx < len(cleaned_blocks):
        cleaned_blocks = cleaned_blocks[first_content_idx:]

    while cleaned_blocks and (
        _is_boilerplate_block(cleaned_blocks[0])
        or _is_navigation_like_block(cleaned_blocks[0])
        or _is_probable_title_block(cleaned_blocks[0])
        or _is_probable_meta_block(cleaned_blocks[0])
        or _is_probable_caption_block(cleaned_blocks[0])
        or _is_probable_credit_block(cleaned_blocks[0])
    ):
        cleaned_blocks = cleaned_blocks[1:]

    if not cleaned_blocks:
        cleaned_blocks = _dedupe_blocks(raw_blocks)

    return "\n\n".join(cleaned_blocks).strip()


def _node_attr_text(node) -> str:
    values: List[str] = []
    attrs = getattr(node, "attrs", None)
    if not attrs:
        return ""
    for key in ("id", "class", "role", "aria-label", "data-testid"):
        value = attrs.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return " ".join(values)


def _focus_priority_from_attr_text(tag: str, attr_text: str) -> int:
    if tag in {"main", "article"}:
        return 3
    if re.search(r"\bmain\b", attr_text, flags=re.IGNORECASE):
        return 3
    if _FOCUS_HIGH_RE.search(attr_text):
        return 3
    if _FOCUS_MEDIUM_RE.search(attr_text):
        return 2
    if _FOCUS_LOW_RE.search(attr_text):
        return 1
    return 0


def _score_extracted_candidate(priority: int, raw_text: str) -> Tuple[float, int]:
    normalized = _normalize_extracted_text(raw_text)
    if not normalized:
        return float("-inf"), 0
    raw_blocks = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    cleaned = clean_neutral_context_text(normalized)
    cleaned_blocks = [block.strip() for block in cleaned.split("\n\n") if block.strip()]
    cleaned_word_count = len(cleaned.split())
    if cleaned_word_count < 40:
        return float("-inf"), cleaned_word_count
    substantive_blocks = sum(_is_substantive_body_block(block) for block in cleaned_blocks)
    sentence_blocks = sum(bool(re.search(r"[.!?]", block)) and len(block.split()) >= 8 for block in cleaned_blocks)
    raw_nav_blocks = sum(_is_navigation_like_block(block) for block in raw_blocks)
    raw_boilerplate_blocks = sum(_is_boilerplate_block(block) for block in raw_blocks)
    raw_short_blocks = sum(len(block.split()) <= 5 for block in raw_blocks)
    raw_separator_count = sum(sum(block.count(char) for char in _BLOCK_SEPARATOR_CHARS) for block in raw_blocks)
    score = (
        float(priority) * 10000.0
        + float(substantive_blocks) * 350.0
        + float(sentence_blocks) * 140.0
        + float(cleaned_word_count)
        - float(raw_nav_blocks) * 350.0
        - float(raw_boilerplate_blocks) * 500.0
        - float(raw_short_blocks) * 16.0
        - float(raw_separator_count) * 8.0
        - float(len(normalized.split())) * 0.1
    )
    return score, cleaned_word_count


def _extract_visible_text_with_bs4(html_text: str) -> str:
    assert BeautifulSoup is not None
    soup = BeautifulSoup(str(html_text), "lxml")

    for node in soup.find_all(["script", "style", "noscript", "svg", "canvas", "nav", "footer", "form", "button", "select", "option"]):
        node.decompose()

    for node in list(soup.find_all(True)):
        attr_text = _node_attr_text(node)
        tag_name = str(getattr(node, "name", "")).lower()
        if tag_name in {"body", "main", "article"}:
            continue
        if _focus_priority_from_attr_text(tag_name, attr_text) >= 2:
            continue
        if _SKIP_ATTR_RE.search(attr_text):
            node.decompose()

    best_node = None
    best_score = float("-inf")
    best_cleaned_text = ""
    best_raw_word_count = 0
    for node in soup.find_all(True):
        attr_text = _node_attr_text(node)
        priority = _focus_priority_from_attr_text(str(node.name).lower(), attr_text)
        if priority <= 0:
            continue
        raw_text = node.get_text("\n\n")
        raw_word_count = len(raw_text.split())
        if raw_word_count < 40:
            continue
        score, cleaned_word_count = _score_extracted_candidate(priority, raw_text)
        if cleaned_word_count < 40:
            continue
        if score > best_score or (score == best_score and 0 < raw_word_count < best_raw_word_count):
            best_score = score
            best_node = node
            best_cleaned_text = clean_neutral_context_text(raw_text)
            best_raw_word_count = raw_word_count

    if best_node is not None and best_cleaned_text:
        return best_cleaned_text
    root = best_node if best_node is not None else (soup.body or soup)
    return clean_neutral_context_text(root.get_text("\n\n"))


def extract_visible_text_from_html(html_text: str) -> str:
    if BeautifulSoup is not None:
        try:
            return _extract_visible_text_with_bs4(html_text)
        except Exception:
            pass
    parser = _VisibleTextExtractor()
    parser.feed(str(html_text))
    parser.close()
    return clean_neutral_context_text(parser.extracted_text())


def _downloaded_text_path(source_id: str, *, text_dir: str | Path | None = None) -> Path:
    base = default_neutral_context_text_dir() if text_dir is None else Path(text_dir).expanduser().resolve()
    return base / f"{source_id}.txt"


def _downloaded_metadata_path(source_id: str, *, text_dir: str | Path | None = None) -> Path:
    base = default_neutral_context_text_dir() if text_dir is None else Path(text_dir).expanduser().resolve()
    return base / f"{source_id}.metadata.json"


def _assistant_register_text_path(source_id: str, *, text_dir: str | Path | None = None) -> Path:
    base = default_assistant_register_context_text_dir() if text_dir is None else Path(text_dir).expanduser().resolve()
    return base / f"{source_id}.txt"


def _assistant_register_metadata_path(source_id: str, *, text_dir: str | Path | None = None) -> Path:
    base = default_assistant_register_context_text_dir() if text_dir is None else Path(text_dir).expanduser().resolve()
    return base / f"{source_id}.metadata.json"


def _stylized_text_path(
    source_id: str,
    *,
    trait_name: str,
    pole: str,
    text_dir: str | Path | None = None,
) -> Path:
    base = default_stylized_context_text_dir() if text_dir is None else Path(text_dir).expanduser().resolve()
    return base / str(trait_name).strip().lower() / str(pole).strip().lower() / f"{source_id}.txt"


def _stylized_metadata_path(
    source_id: str,
    *,
    trait_name: str,
    pole: str,
    text_dir: str | Path | None = None,
) -> Path:
    base = default_stylized_context_text_dir() if text_dir is None else Path(text_dir).expanduser().resolve()
    return base / str(trait_name).strip().lower() / str(pole).strip().lower() / f"{source_id}.metadata.json"


def _assistant_register_stylized_text_path(
    source_id: str,
    *,
    trait_name: str,
    pole: str,
    text_dir: str | Path | None = None,
) -> Path:
    base = (
        default_assistant_register_stylized_context_text_dir()
        if text_dir is None
        else Path(text_dir).expanduser().resolve()
    )
    return base / str(trait_name).strip().lower() / str(pole).strip().lower() / f"{source_id}.txt"


def _assistant_register_stylized_metadata_path(
    source_id: str,
    *,
    trait_name: str,
    pole: str,
    text_dir: str | Path | None = None,
) -> Path:
    base = (
        default_assistant_register_stylized_context_text_dir()
        if text_dir is None
        else Path(text_dir).expanduser().resolve()
    )
    return base / str(trait_name).strip().lower() / str(pole).strip().lower() / f"{source_id}.metadata.json"


def _load_context_records_from_paths(
    *,
    manifest_path: str | Path | None,
    source_ids: Sequence[str] | None,
    text_path_for_source,
    metadata_path_for_source,
) -> List[Dict[str, str]]:
    allowed = None if source_ids is None else {str(source_id).strip() for source_id in source_ids if str(source_id).strip()}
    rows: List[Dict[str, str]] = []
    for source in load_neutral_context_manifest(manifest_path):
        if allowed is not None and source.source_id not in allowed:
            continue
        text_path = text_path_for_source(source.source_id)
        if not text_path.is_file():
            continue
        text = clean_neutral_context_text(text_path.read_text(encoding="utf-8", errors="replace"))
        if not text:
            continue
        metadata_path = metadata_path_for_source(source.source_id)
        metadata: Dict[str, str] = {}
        if metadata_path.is_file():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    metadata = {str(k): str(v) for k, v in payload.items()}
            except Exception:
                metadata = {}
        rows.append(
            {
                "source_id": source.source_id,
                "title": metadata.get("title", source.title),
                "url": metadata.get("url", source.url),
                "source": metadata.get("source", source.source),
                "license": metadata.get("license", source.license),
                "description": metadata.get("description", source.description),
                "text": text,
            }
        )
    return rows


def read_downloaded_neutral_contexts(
    *,
    manifest_path: str | Path | None = None,
    text_dir: str | Path | None = None,
    source_ids: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    return _load_context_records_from_paths(
        manifest_path=manifest_path,
        source_ids=source_ids,
        text_path_for_source=lambda source_id: _downloaded_text_path(source_id, text_dir=text_dir),
        metadata_path_for_source=lambda source_id: _downloaded_metadata_path(source_id, text_dir=text_dir),
    )


def read_downloaded_stylized_contexts(
    *,
    trait_name: str,
    pole: str,
    manifest_path: str | Path | None = None,
    text_dir: str | Path | None = None,
    source_ids: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    return _load_context_records_from_paths(
        manifest_path=manifest_path,
        source_ids=source_ids,
        text_path_for_source=lambda source_id: _stylized_text_path(
            source_id,
            trait_name=trait_name,
            pole=pole,
            text_dir=text_dir,
        ),
        metadata_path_for_source=lambda source_id: _stylized_metadata_path(
            source_id,
            trait_name=trait_name,
            pole=pole,
            text_dir=text_dir,
        ),
    )


def read_prepared_assistant_register_contexts(
    *,
    manifest_path: str | Path | None = None,
    text_dir: str | Path | None = None,
    source_ids: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    return _load_context_records_from_paths(
        manifest_path=default_assistant_register_context_manifest_path() if manifest_path is None else manifest_path,
        source_ids=source_ids,
        text_path_for_source=lambda source_id: _assistant_register_text_path(source_id, text_dir=text_dir),
        metadata_path_for_source=lambda source_id: _assistant_register_metadata_path(source_id, text_dir=text_dir),
    )


def read_prepared_assistant_register_stylized_contexts(
    *,
    trait_name: str,
    pole: str,
    manifest_path: str | Path | None = None,
    text_dir: str | Path | None = None,
    source_ids: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    return _load_context_records_from_paths(
        manifest_path=default_assistant_register_context_manifest_path() if manifest_path is None else manifest_path,
        source_ids=source_ids,
        text_path_for_source=lambda source_id: _assistant_register_stylized_text_path(
            source_id,
            trait_name=trait_name,
            pole=pole,
            text_dir=text_dir,
        ),
        metadata_path_for_source=lambda source_id: _assistant_register_stylized_metadata_path(
            source_id,
            trait_name=trait_name,
            pole=pole,
            text_dir=text_dir,
        ),
    )


def _waterfill_token_allocations(token_lengths: Sequence[int], token_budget: int) -> Tuple[List[int], int]:
    allocations = [0 for _ in token_lengths]
    remaining = max(0, int(token_budget))
    active_indices = [index for index, length in enumerate(token_lengths) if int(length) > 0]
    while remaining > 0 and active_indices:
        share = max(1, remaining // len(active_indices))
        next_active: List[int] = []
        progress = False
        for index in active_indices:
            available = int(token_lengths[index]) - allocations[index]
            if available <= 0 or remaining <= 0:
                continue
            take = min(share, available, remaining)
            allocations[index] += take
            remaining -= take
            if allocations[index] < int(token_lengths[index]):
                next_active.append(index)
            if take > 0:
                progress = True
        if not progress:
            break
        active_indices = next_active
    return allocations, remaining


def _select_stratified_record_indices(record_count: int, target_count: int) -> List[int]:
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


def _target_stratified_source_count(record_count: int, token_budget: int) -> int:
    total = max(0, int(record_count))
    budget = max(0, int(token_budget))
    if total <= 0 or budget <= 0:
        return 0
    target_excerpt_tokens = 2000 if budget <= 8000 else 4000
    return min(total, max(1, (budget + target_excerpt_tokens - 1) // target_excerpt_tokens))


def build_context_bundle(
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
            "used_source_ids": [],
            "used_titles": [],
        }

    prepared_records: List[Tuple[Dict[str, str], List[int]]] = []
    for record in records:
        body = clean_neutral_context_text(str(record["text"]).strip())
        if not body:
            continue
        piece = f"Reference: {record['title']} ({record['source']})\n\n{body}\n\n"
        piece_ids = tokenizer(piece, add_special_tokens=False)["input_ids"]
        if piece_ids:
            prepared_records.append((record, piece_ids))
    if not prepared_records:
        return {
            "token_budget": budget,
            "actual_token_count": 0,
            "context_text": "",
            "used_source_ids": [],
            "used_titles": [],
        }

    total_available_tokens = sum(len(piece_ids) for _, piece_ids in prepared_records)
    if total_available_tokens <= budget:
        allocations = [len(piece_ids) for _, piece_ids in prepared_records]
    else:
        token_lengths = [len(piece_ids) for _, piece_ids in prepared_records]
        allocations = [0 for _ in prepared_records]
        target_source_count = _target_stratified_source_count(len(prepared_records), budget)
        primary_indices = _select_stratified_record_indices(len(prepared_records), target_source_count)
        if primary_indices:
            primary_allocations, remaining = _waterfill_token_allocations(
                [token_lengths[index] for index in primary_indices],
                budget,
            )
            for index, allocation in zip(primary_indices, primary_allocations):
                allocations[index] = allocation
        else:
            remaining = budget
        if remaining > 0:
            remaining_indices = [index for index, length in enumerate(token_lengths) if length > allocations[index]]
            supplemental_allocations, _ = _waterfill_token_allocations(
                [token_lengths[index] - allocations[index] for index in remaining_indices],
                remaining,
            )
            for index, allocation in zip(remaining_indices, supplemental_allocations):
                allocations[index] += allocation

    parts: List[str] = []
    used_source_ids: List[str] = []
    used_titles: List[str] = []
    source_token_allocations: Dict[str, int] = {}
    for (record, piece_ids), allocation in zip(prepared_records, allocations):
        if allocation <= 0:
            continue
        piece_text = tokenizer.decode(piece_ids[:allocation], skip_special_tokens=True).strip()
        if not piece_text:
            continue
        parts.append(piece_text + "\n\n")
        source_id = str(record["source_id"])
        used_source_ids.append(source_id)
        used_titles.append(str(record["title"]))
        source_token_allocations[source_id] = int(allocation)

    context_text = "".join(parts).strip()
    actual_token_count = 0
    if context_text:
        actual_token_count = len(tokenizer(context_text, add_special_tokens=False)["input_ids"])
    return {
        "token_budget": budget,
        "actual_token_count": int(actual_token_count),
        "context_text": context_text,
        "used_source_ids": used_source_ids,
        "used_titles": used_titles,
        "source_token_allocations": source_token_allocations,
    }


__all__ = [
    "NeutralContextSource",
    "build_context_bundle",
    "default_assistant_register_context_manifest_path",
    "default_assistant_register_context_text_dir",
    "default_assistant_register_stylized_context_text_dir",
    "default_neutral_context_manifest_path",
    "default_neutral_context_raw_dir",
    "default_neutral_context_text_dir",
    "default_stylized_context_text_dir",
    "extract_visible_text_from_html",
    "load_neutral_context_manifest",
    "read_downloaded_neutral_contexts",
    "read_downloaded_stylized_contexts",
    "read_prepared_assistant_register_contexts",
    "read_prepared_assistant_register_stylized_contexts",
    "clean_neutral_context_text",
]
