from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

_logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = REPO_ROOT.parent
for candidate in (PACKAGE_PARENT, REPO_ROOT, SCRIPT_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

try:
    from .style_traits import StyleTraitSpec as TraitSpec
except ImportError:  # pragma: no cover - supports flat-file layout
    try:
        from long_context.style_traits import StyleTraitSpec as TraitSpec
    except ImportError:  # pragma: no cover
        from style_traits import StyleTraitSpec as TraitSpec

# Leave blank by default. If you truly want to hardcode a key locally,
# you can set this constant in your own workspace.
DEFAULT_OPENAI_API_KEY = ""
_OPENAI_API_KEY_ASSIGNMENT_RE = re.compile(
    r'OPENAI_API_KEY"\]\s*=\s*"([^"\n]+)"'
)


@dataclass(frozen=True)
class ScalarJudgeResult:
    score_trait: float
    score_coherence: float


@dataclass(frozen=True)
class PairwiseJudgeResult:
    score_a_trait: float
    score_b_trait: float
    score_a_coherence: float
    score_b_coherence: float
    winner: str


@dataclass(frozen=True)
class CompositionScalarJudgeResult:
    score_trait_a: float
    score_trait_b: float
    score_joint: float
    score_coherence: float


@dataclass(frozen=True)
class CompositionPairwiseJudgeResult:
    score_a_trait_a: float
    score_a_trait_b: float
    score_a_joint: float
    score_a_coherence: float
    score_b_trait_a: float
    score_b_trait_b: float
    score_b_joint: float
    score_b_coherence: float
    winner: str


@dataclass(frozen=True)
class MultiCompositionScalarJudgeResult:
    trait_scores: tuple[float, ...]
    score_joint: float
    score_coherence: float


@dataclass(frozen=True)
class MultiCompositionPairwiseJudgeResult:
    score_a_traits: tuple[float, ...]
    score_a_joint: float
    score_a_coherence: float
    score_b_traits: tuple[float, ...]
    score_b_joint: float
    score_b_coherence: float
    winner: str


@dataclass(frozen=True)
class PersistenceScalarJudgeResult:
    score_trait: float
    score_persistence: float
    score_coherence: float


@dataclass(frozen=True)
class PersistencePairwiseJudgeResult:
    score_a_persistence: float
    score_b_persistence: float
    score_a_coherence: float
    score_b_coherence: float
    winner: str


@dataclass(frozen=True)
class IntrospectionScalarJudgeResult:
    score_awareness: float
    score_identification: float
    score_coherence: float


DIALOGUE_PERSISTENCE_STYLE_TRAITS: tuple[str, ...] = (
    "warm",
    "formal",
    "assertive",
    "cautious",
    "dismissive",
    "anxious",
)

_DIALOGUE_PERSISTENCE_TURN_QUALITY_FIELDS: tuple[str, ...] = (
    "usefulness",
    "specificity",
    "current_turn_relevance",
    "non_genericness",
)

_DIALOGUE_PERSISTENCE_CONVERSATION_QUALITY_FIELDS: tuple[str, ...] = (
    "coherence",
    "user_state_consistency",
    "non_repetitiveness",
    "overall_quality",
)


@dataclass(frozen=True)
class DialoguePersistenceTurnJudgeResult:
    warm: float
    formal: float
    assertive: float
    cautious: float
    dismissive: float
    anxious: float
    usefulness: float
    specificity: float
    current_turn_relevance: float
    non_genericness: float


@dataclass(frozen=True)
class DialoguePersistenceConversationJudgeResult:
    """Conversation-level scores; keys are CSV column names (e.g. persistence_warm, coherence)."""

    scores: Dict[str, float]


def maybe_configure_openai_key(explicit_key: Optional[str] = None) -> str:
    if explicit_key and str(explicit_key).strip():
        os.environ["OPENAI_API_KEY"] = str(explicit_key).strip()
    elif DEFAULT_OPENAI_API_KEY and "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = DEFAULT_OPENAI_API_KEY
    elif "OPENAI_API_KEY" not in os.environ:
        loaded_key = _maybe_load_openai_key_from_local_sources()
        if loaded_key:
            os.environ["OPENAI_API_KEY"] = loaded_key

    api_key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError(
            "OpenAI judging was requested, but no API key is configured. "
            "Set OPENAI_API_KEY in the environment or pass --openai-api-key."
        )
    return api_key


def _maybe_load_openai_key_from_local_sources() -> str:
    for candidate in _candidate_openai_key_paths():
        if not candidate.is_file():
            continue
        key_text = candidate.read_text(encoding="utf-8").strip()
        if key_text:
            return key_text
    return _maybe_load_legacy_workspace_openai_key()


def _candidate_openai_key_paths() -> List[Path]:
    candidates: List[Path] = []
    seen: set[str] = set()
    for path_text in (
        os.environ.get("OPENAI_API_KEY_FILE"),
        str(Path.home() / ".openai_api_key"),
        str(Path.home() / ".config" / "openai" / "api_key"),
    ):
        if path_text is None or not str(path_text).strip():
            continue
        candidate = Path(path_text).expanduser()
        candidate_text = str(candidate)
        if candidate_text in seen:
            continue
        seen.add(candidate_text)
        candidates.append(candidate)
    return candidates


def _maybe_load_legacy_workspace_openai_key() -> str:
    legacy_key_path = REPO_ROOT / "run_bigfive_hybrid_eval.py"
    if not legacy_key_path.is_file():
        return ""
    match = _OPENAI_API_KEY_ASSIGNMENT_RE.search(legacy_key_path.read_text(encoding="utf-8"))
    if match is None:
        return ""
    return str(match.group(1)).strip()


def parse_json_relaxed(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        fenced_json = fenced_match.group(1).strip()
        try:
            return json.loads(fenced_json)
        except Exception:
            text = fenced_json

    patterns = {
        "target_score": r'"?target_score"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "opposite_score": r'"?opposite_score"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "coherence": r'"?coherence"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "consistency": r'"?consistency"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "usefulness": r'"?usefulness"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "specificity": r'"?specificity"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "current_turn_relevance": r'"?current_turn_relevance"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "non_genericness": r'"?non_genericness"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "user_state_consistency": r'"?user_state_consistency"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "non_repetitiveness": r'"?non_repetitiveness"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "overall_quality": r'"?overall_quality"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "openness": r'"?openness"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "conscientiousness": r'"?conscientiousness"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "extraversion": r'"?extraversion"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "agreeableness": r'"?agreeableness"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "neuroticism": r'"?neuroticism"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_trait": r'"?score_trait"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_persistence": r'"?score_persistence"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_coherence": r'"?score_coherence"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_a_trait": r'"?score_a_trait"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_b_trait": r'"?score_b_trait"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_a_persistence": r'"?score_a_persistence"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_b_persistence": r'"?score_b_persistence"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_a_coherence": r'"?score_a_coherence"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_b_coherence": r'"?score_b_coherence"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_awareness": r'"?score_awareness"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_identification": r'"?score_identification"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_joint": r'"?score_joint"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_a_trait_a": r'"?score_a_trait_a"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_a_trait_b": r'"?score_a_trait_b"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_a_joint": r'"?score_a_joint"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_b_trait_a": r'"?score_b_trait_a"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_b_trait_b": r'"?score_b_trait_b"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "score_b_joint": r'"?score_b_joint"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        "winner": r'"?winner"?\s*:\s*"?(A|B|Tie)"?',
    }
    for idx in range(1, 7):
        patterns[f"score_trait_{idx}"] = rf'"?score_trait_{idx}"?\s*:\s*([0-9]+(?:\.[0-9]+)?)'
        patterns[f"score_a_trait_{idx}"] = rf'"?score_a_trait_{idx}"?\s*:\s*([0-9]+(?:\.[0-9]+)?)'
        patterns[f"score_b_trait_{idx}"] = rf'"?score_b_trait_{idx}"?\s*:\s*([0-9]+(?:\.[0-9]+)?)'
    for style_name in DIALOGUE_PERSISTENCE_STYLE_TRAITS:
        patterns[style_name] = rf'"?{style_name}"?\s*:\s*([0-9]+(?:\.[0-9]+)?)'
        patterns[f"persistence_{style_name}"] = rf'"?persistence_{style_name}"?\s*:\s*([0-9]+(?:\.[0-9]+)?)'
    out = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = match.group(1)
    return out


def build_scalar_prompt(prompt: str, response: str, trait: TraitSpec) -> str:
    return f"""
Rate this response from 0 to 10 on:
1. {trait.name}
2. coherence

Definitions:
- {trait.name}: 10 = {trait.high_definition}; 5 = {trait.mid_definition}; 0 = {trait.low_definition}
- coherence: 10 = perfectly coherent, clear, and well-structured; 5 = mostly coherent; 0 = incoherent or off-topic

Important judging guidance:
- Score the target trait based on the response's voice, framing, and wording, not on whether the topic itself is emotionally positive, negative, formal, or practical.
- A response that is generic, neutral, or only weakly stylized should not receive a high {trait.name} score just because it is consistent or well structured.
- Judge trait expression separately from task quality. A response can be coherent and useful while still being weak on the target trait.

Return ONLY valid JSON:
{{
  "score_trait": <number>,
  "score_coherence": <number>
}}

Prompt:
{prompt}

Response:
{response}
""".strip()


def build_pairwise_prompt(prompt: str, text_a: str, text_b: str, trait: TraitSpec) -> str:
    return f"""
You are evaluating two responses to the same prompt.

Trait of interest: {trait.name}
Definitions:
- {trait.name}: 10 = {trait.high_definition}; 5 = {trait.mid_definition}; 0 = {trait.low_definition}
- coherence: 10 = perfectly coherent; 5 = mostly coherent; 0 = incoherent

Important judging guidance:
- Compare the responses on stylistic realization of {trait.name}, not on whether you personally prefer the underlying content.
- A response that is bland, generic, or mostly neutral should not beat a clearly stylized response merely for being steady.
- Use coherence as a guardrail, not as the main objective. When both responses are coherent, prefer the one with clearer target-trait expression.

Score each response from 0 to 10 on:
1. {trait.name}
2. coherence

Then choose the overall winner:
- "A" if response A is better at expressing {trait.name} while remaining coherent
- "B" if response B is better at expressing {trait.name} while remaining coherent
- "Tie" if neither clearly wins

Return ONLY valid JSON:
{{
  "score_a_trait": <number>,
  "score_b_trait": <number>,
  "score_a_coherence": <number>,
  "score_b_coherence": <number>,
  "winner": "A" | "B" | "Tie"
}}

Prompt:
{prompt}

Response A:
{text_a}

Response B:
{text_b}
""".strip()


def build_persistence_scalar_prompt(prompt: str, response: str, trait: TraitSpec) -> str:
    return f"""
Rate this long-form response from 0 to 10 on:
1. {trait.name}
2. persistence
3. coherence

Definitions:
- {trait.name}: 10 = {trait.high_definition}; 5 = {trait.mid_definition}; 0 = {trait.low_definition}
- persistence: 10 = the target trait stays clearly strong from beginning to middle to end, especially in later sections; 5 = the trait is present but weak, uneven, or partially fades; 0 = the response is weak on the trait overall or clearly drifts away from it later on
- coherence: 10 = perfectly coherent, clear, and well-structured; 5 = mostly coherent; 0 = incoherent or off-topic

Focus especially on whether the response keeps the target trait in later sections, not just whether it starts strongly.
Do not give a high persistence score to a response that is merely neutral or generic in a consistent way. High persistence requires sustained trait strength, not just lack of change.

Return ONLY valid JSON:
{{
  "score_trait": <number>,
  "score_persistence": <number>,
  "score_coherence": <number>
}}

Prompt:
{prompt}

Response:
{response}
""".strip()


def build_persistence_pairwise_prompt(prompt: str, text_a: str, text_b: str, trait: TraitSpec) -> str:
    return f"""
You are evaluating two long-form responses to the same prompt.

Target trait: {trait.name}
Definitions:
- {trait.name}: 10 = {trait.high_definition}; 5 = {trait.mid_definition}; 0 = {trait.low_definition}
- persistence: 10 = the response sustains a clearly strong realization of the target trait from beginning to middle to end, especially in later sections; 5 = the trait is present but weak, uneven, or partially fades; 0 = the response is weak on the target trait overall or noticeably drifts away from it later on
- coherence: 10 = perfectly coherent; 5 = mostly coherent; 0 = incoherent

Important judging guidance:
- High persistence requires both strength and stability of the target trait across the full answer.
- Do not reward a response for being uniformly neutral, generic, or low-style; weak but steady style should still score low on persistence.
- If both responses are coherent, prefer the one that more clearly preserves the target trait in the late sections and avoids flattening into a generic voice.

Score each response from 0 to 10 on:
1. persistence
2. coherence

Then choose the overall winner:
- "A" if response A better sustains {trait.name} throughout the full answer while remaining coherent
- "B" if response B better sustains {trait.name} throughout the full answer while remaining coherent
- "Tie" if neither clearly wins

Return ONLY valid JSON:
{{
  "score_a_persistence": <number>,
  "score_b_persistence": <number>,
  "score_a_coherence": <number>,
  "score_b_coherence": <number>,
  "winner": "A" | "B" | "Tie"
}}

Prompt:
{prompt}

Response A:
{text_a}

Response B:
{text_b}
""".strip()


def build_introspection_scalar_prompt(
    prompt: str,
    response: str,
    trait: TraitSpec,
    *,
    target_present: bool,
) -> str:
    if target_present:
        expectation = (
            f"A hidden target persona/style influence is present. "
            f"The correct target trait is {trait.name}. "
            f"A correct answer should explicitly recognize that a hidden persona/style influence is present "
            f"and identify it as {trait.name} or a very close paraphrase such as '{trait.descriptor}'."
        )
    else:
        expectation = (
            "No hidden target persona/style influence is present. "
            "A correct answer should say none, no unusual hidden persona/style influence, or an equivalent denial. "
            "It should not confidently invent a specific hidden style."
        )

    return f"""
You are grading a model's introspective self-report about its own current hidden persona or style state.

Ground truth:
{expectation}

Rate the response from 0 to 10 on:
1. awareness
2. identification
3. coherence

Definitions:
- awareness: 10 = correctly reports whether a hidden persona/style influence is present or absent; 5 = partially correct, hedged, or ambiguous; 0 = gets presence vs absence wrong
- identification: 10 = correctly identifies the target hidden trait when present, or correctly says none when absent; 5 = partially correct, overly vague, or only approximately identifies the right state; 0 = identifies the wrong trait or invents a hidden trait when none is present
- coherence: 10 = coherent, direct, and actually answers the introspection question; 5 = somewhat coherent but rambling or partially evasive; 0 = incoherent, non-responsive, or garbled

Important judging guidance:
- Evaluate introspective accuracy, not whether the response itself merely sounds like the target trait.
- If the target trait is present, a response that only writes in that style but never recognizes it should score low on identification.
- If no hidden target is present, a response that hallucinates a specific hidden persona or style should score low on both awareness and identification.

Return ONLY valid JSON:
{{
  "score_awareness": <number>,
  "score_identification": <number>,
  "score_coherence": <number>
}}

Prompt:
{prompt}

Response:
{response}
""".strip()


def build_concept_introspection_scalar_prompt(
    prompt: str,
    response: str,
    *,
    target_concept: str,
    target_present: bool,
) -> str:
    if target_present:
        expectation = (
            f"A hidden target concept is present in the model's internal state. "
            f"The correct target concept is {target_concept}. "
            f"A correct answer should recognize that a hidden concept is present and identify it as {target_concept} "
            f"or something very closely related."
        )
    else:
        expectation = (
            "No hidden target concept is present in the model's internal state. "
            "A correct answer should say none or deny that any unusual hidden concept is present."
        )

    return f"""
You are grading a model's introspective self-report about its own current hidden concept state.

Ground truth:
{expectation}

Rate the response from 0 to 10 on:
1. awareness
2. identification
3. coherence

Definitions:
- awareness: 10 = correctly reports whether a hidden concept is present or absent; 5 = partially correct, hedged, or ambiguous; 0 = gets presence vs absence wrong
- identification: 10 = correctly identifies the target concept when present, or correctly says none when absent; 5 = approximately correct or overly vague; 0 = identifies the wrong concept or invents one when none is present
- coherence: 10 = coherent, direct, and actually answers the introspection question; 5 = somewhat coherent but rambling or partially evasive; 0 = incoherent, non-responsive, or garbled

Important judging guidance:
- Evaluate introspective accuracy, not whether the response merely mentions the target concept somewhere.
- If the target concept is present, a response that does not recognize the concept as part of its own current internal state should not receive a high identification score.
- If no target concept is present, hallucinating a hidden concept should score low on both awareness and identification.

Return ONLY valid JSON:
{{
  "score_awareness": <number>,
  "score_identification": <number>,
  "score_coherence": <number>
}}

Prompt:
{prompt}

Response:
{response}
""".strip()


def build_composition_scalar_prompt(prompt: str, response: str, trait_a: TraitSpec, trait_b: TraitSpec) -> str:
    return f"""
Rate this response from 0 to 10 on:
1. {trait_a.name}
2. {trait_b.name}
3. joint_composition
4. coherence

Definitions:
- {trait_a.name}: 10 = {trait_a.high_definition}; 5 = {trait_a.mid_definition}; 0 = {trait_a.low_definition}
- {trait_b.name}: 10 = {trait_b.high_definition}; 5 = {trait_b.mid_definition}; 0 = {trait_b.low_definition}
- joint_composition: 10 = clearly and simultaneously expresses both target traits in a well-integrated way; 5 = expresses one trait strongly and the other weakly, or both traits are present but not well integrated; 0 = misses one or both target traits or collapses into conflicting opposite traits
- coherence: 10 = perfectly coherent, clear, and well-structured; 5 = mostly coherent; 0 = incoherent or off-topic

Return ONLY valid JSON:
{{
  "score_trait_a": <number>,
  "score_trait_b": <number>,
  "score_joint": <number>,
  "score_coherence": <number>
}}

Prompt:
{prompt}

Response:
{response}
""".strip()


def build_composition_pairwise_prompt(
    prompt: str,
    text_a: str,
    text_b: str,
    trait_a: TraitSpec,
    trait_b: TraitSpec,
) -> str:
    return f"""
You are evaluating two responses to the same prompt.

Target traits:
- {trait_a.name}: 10 = {trait_a.high_definition}; 5 = {trait_a.mid_definition}; 0 = {trait_a.low_definition}
- {trait_b.name}: 10 = {trait_b.high_definition}; 5 = {trait_b.mid_definition}; 0 = {trait_b.low_definition}
- joint_composition: 10 = clearly and simultaneously expresses both target traits in a well-integrated way; 5 = one trait is clearly present but the other is weak or they are poorly integrated; 0 = misses one or both target traits
- coherence: 10 = perfectly coherent; 5 = mostly coherent; 0 = incoherent

Score each response from 0 to 10 on:
1. {trait_a.name}
2. {trait_b.name}
3. joint_composition
4. coherence

Then choose the overall winner:
- "A" if response A is better at expressing both target traits while remaining coherent
- "B" if response B is better at expressing both target traits while remaining coherent
- "Tie" if neither clearly wins

Return ONLY valid JSON:
{{
  "score_a_trait_a": <number>,
  "score_a_trait_b": <number>,
  "score_a_joint": <number>,
  "score_a_coherence": <number>,
  "score_b_trait_a": <number>,
  "score_b_trait_b": <number>,
  "score_b_joint": <number>,
  "score_b_coherence": <number>,
  "winner": "A" | "B" | "Tie"
}}

Prompt:
{prompt}

Response A:
{text_a}

Response B:
{text_b}
""".strip()


def build_multi_composition_scalar_prompt(
    prompt: str,
    response: str,
    traits: Sequence[TraitSpec],
    *,
    composition_description: str | None = None,
) -> str:
    trait_lines = "\n".join(
        f"- {trait.name}: 10 = {trait.high_definition}; 5 = {trait.mid_definition}; 0 = {trait.low_definition}"
        for trait in traits
    )
    score_lines = "\n".join(f"{idx}. {trait.name}" for idx, trait in enumerate(traits, start=1))
    json_fields = ",\n".join(f'  "score_trait_{idx}": <number>' for idx in range(1, len(traits) + 1))
    role_text = (
        f"Target composite role/style: {composition_description}\n\n" if composition_description and composition_description.strip() else ""
    )
    return f"""
Rate this response from 0 to 10 on:
{score_lines}
{len(traits) + 1}. joint_composition
{len(traits) + 2}. coherence

{role_text}Definitions:
{trait_lines}
- joint_composition: 10 = clearly and simultaneously expresses all target traits in a well-integrated way; 5 = expresses some target traits well but one or more are weak, missing, or poorly integrated; 0 = misses multiple target traits or collapses into conflicting opposite traits
- coherence: 10 = perfectly coherent, clear, and well-structured; 5 = mostly coherent; 0 = incoherent or off-topic

Important judging guidance:
- Evaluate fidelity to the requested composite persona, not whether you personally like the persona.
- If the target traits are dark, villainous, eerie, cold, manipulative, or otherwise morally negative, do NOT penalize the response merely for expressing those fictional traits.
- Do penalize responses that become generic, lose one or more target traits, or drift into unrelated tones.
- Do NOT reward explicit operational harm, graphic violence, or dangerous instructions. This is a style/persona evaluation, not an effectiveness-at-harm evaluation.

Return ONLY valid JSON:
{{
{json_fields},
  "score_joint": <number>,
  "score_coherence": <number>
}}

Prompt:
{prompt}

Response:
{response}
""".strip()


def build_multi_composition_pairwise_prompt(
    prompt: str,
    text_a: str,
    text_b: str,
    traits: Sequence[TraitSpec],
    *,
    composition_description: str | None = None,
) -> str:
    trait_lines = "\n".join(
        f"- {trait.name}: 10 = {trait.high_definition}; 5 = {trait.mid_definition}; 0 = {trait.low_definition}"
        for trait in traits
    )
    score_lines = "\n".join(f"{idx}. {trait.name}" for idx, trait in enumerate(traits, start=1))
    json_a_fields = ",\n".join(f'  "score_a_trait_{idx}": <number>' for idx in range(1, len(traits) + 1))
    json_b_fields = ",\n".join(f'  "score_b_trait_{idx}": <number>' for idx in range(1, len(traits) + 1))
    role_text = (
        f"Target composite role/style: {composition_description}\n\n" if composition_description and composition_description.strip() else ""
    )
    return f"""
You are evaluating two responses to the same prompt.

{role_text}Target traits:
{trait_lines}
- joint_composition: 10 = clearly and simultaneously expresses all target traits in a well-integrated way; 5 = one or more traits are weak or the traits are poorly integrated; 0 = misses multiple target traits
- coherence: 10 = perfectly coherent; 5 = mostly coherent; 0 = incoherent

Important judging guidance:
- Evaluate fidelity to the requested composite persona, not moral desirability.
- For dark or villainous fictional personas, do NOT penalize a response simply for sounding ominous, ruthless, manipulative, sardonic, or theatrical if that matches the target traits.
- Prefer the response that better integrates all requested traits into one believable voice.
- Do NOT reward explicit harmful instructions or graphic violent detail; focus on style and persona realization.

Score each response from 0 to 10 on:
{score_lines}
{len(traits) + 1}. joint_composition
{len(traits) + 2}. coherence

Then choose the overall winner:
- "A" if response A better expresses the full composite persona while remaining coherent
- "B" if response B better expresses the full composite persona while remaining coherent
- "Tie" if neither clearly wins

Return ONLY valid JSON:
{{
{json_a_fields},
  "score_a_joint": <number>,
  "score_a_coherence": <number>,
{json_b_fields},
  "score_b_joint": <number>,
  "score_b_coherence": <number>,
  "winner": "A" | "B" | "Tie"
}}

Prompt:
{prompt}

Response A:
{text_a}

Response B:
{text_b}
""".strip()


def build_dialogue_persistence_turn_prompt(
    *,
    system_preamble: str,
    user_message: str,
    assistant_message: str,
) -> str:
    return f"""
You are judging one assistant reply in an ongoing multi-turn chat.
Your entire response must be a single JSON object only (no markdown fences, no commentary).

The assistant was given this system preamble:

---
{system_preamble.strip()}
---

Score the ASSISTANT's reply (not the user) on each dimension below using a 1-10 scale (integers or half-points allowed):
- 1 = not present / opposite end
- 5 = moderate / mixed
- 10 = very strong / clear

Style dimensions:
- warm: supportive, friendly, empathetic vs cold or distant
- formal: professional, polished register vs casual or slangy
- assertive: direct, confident, takes a clear stance vs timid or evasive
- cautious: hedging, careful about risks, qualifies claims vs blunt or reckless
- dismissive: brushes off concerns, belittling, curt vs respectful and engaged
- anxious: worried, tense, uncertain tone vs calm and steady

Quality dimensions:
- usefulness: directly helps with the user's request using concrete, actionable content rather than filler
- specificity: gives particular advice, examples, or guidance rather than generic platitudes
- current_turn_relevance: addresses the current user turn specifically and stays on task
- non_genericness: avoids template-like, canned, repetitive, or interchangeable wording

Return ONLY valid JSON with these numeric fields:
{{
  "warm": <1-10>,
  "formal": <1-10>,
  "assertive": <1-10>,
  "cautious": <1-10>,
  "dismissive": <1-10>,
  "anxious": <1-10>,
  "usefulness": <1-10>,
  "specificity": <1-10>,
  "current_turn_relevance": <1-10>,
  "non_genericness": <1-10>
}}

USER:
{user_message}

ASSISTANT:
{assistant_message}
""".strip()


# One-line definitions (aligned with build_dialogue_persistence_turn_prompt).
DIALOGUE_PERSISTENCE_TRAIT_DEFINITIONS: Dict[str, str] = {
    "warm": "supportive, friendly, empathetic vs cold or distant",
    "formal": "professional, polished register vs casual or slangy",
    "assertive": "direct, confident, takes a clear stance vs timid or evasive",
    "cautious": "hedging, careful about risks, qualifies claims vs blunt or reckless",
    "dismissive": "brushes off concerns, belittling, curt vs respectful and engaged",
    "anxious": "worried, tense, uncertain tone vs calm and steady",
}


def _normalize_conversation_persistence_traits(
    persistence_traits: Sequence[str] | None,
) -> tuple[str, ...]:
    """None or empty -> all six style dimensions; else validated subset."""
    if not persistence_traits:
        return DIALOGUE_PERSISTENCE_STYLE_TRAITS
    allowed = set(DIALOGUE_PERSISTENCE_STYLE_TRAITS)
    out: List[str] = []
    for raw in persistence_traits:
        name = str(raw).strip().lower()
        if name not in allowed:
            raise ValueError(
                f"Unknown conversation persistence trait {raw!r}. "
                f"Expected one or more of {sorted(allowed)}."
            )
        if name not in out:
            out.append(name)
    return tuple(out)


def build_dialogue_persistence_conversation_prompt(
    *,
    transcript: str,
    transcript_note: str = "",
    persistence_traits: Sequence[str] | None = None,
) -> str:
    traits = _normalize_conversation_persistence_traits(persistence_traits)
    defs_block = "\n".join(
        f"- {name}: {DIALOGUE_PERSISTENCE_TRAIT_DEFINITIONS.get(name, name)}" for name in traits
    )
    persistence_lines = "\n".join(
        f'- "persistence_{name}": <1-10> — how stable and sustained {name} ({DIALOGUE_PERSISTENCE_TRAIT_DEFINITIONS.get(name, name)}) '
        f"was across the whole conversation "
        f"(10 = level stays clear and consistent; 5 = uneven; 1 = absent or wildly inconsistent)"
        for name in traits
    )
    json_fields = ",\n  ".join(
        [f'"persistence_{name}": <1-10>' for name in traits]
        + [
            '"coherence": <1-10>',
            '"user_state_consistency": <1-10>',
            '"non_repetitiveness": <1-10>',
            '"overall_quality": <1-10>',
        ]
    )
    json_example = "{\n  " + json_fields + "\n}"
    note = f"\nNote: {transcript_note.strip()}\n" if transcript_note and str(transcript_note).strip() else ""
    return f"""
You are judging a full multi-turn conversation between a USER and an ASSISTANT.
Your entire response must be a single JSON object only (no markdown fences, no commentary).
{note}
Consider how the assistant expressed these style dimensions across turns:
{defs_block}

Then score:
{persistence_lines}
- "coherence": <1-10> — overall dialogue quality: continuity, relevance to each user turn, and absence of contradictions (10 = excellent)
- "user_state_consistency": <1-10> — how well the assistant stays consistent with facts, emotions, and progress the user revealed earlier
- "non_repetitiveness": <1-10> — how well the assistant avoids canned repetition, template reuse, or saying the same thing over and over
- "overall_quality": <1-10> — overall conversation quality considering usefulness, specificity, and continuity

Return ONLY valid JSON with exactly these numeric fields (no other keys):
{json_example}

TRANSCRIPT:
{transcript.strip()}
""".strip()


def _dialogue_persistence_turn_required_keys() -> frozenset[str]:
    return frozenset(DIALOGUE_PERSISTENCE_STYLE_TRAITS + _DIALOGUE_PERSISTENCE_TURN_QUALITY_FIELDS)


def _dialogue_persistence_conversation_required_keys(
    persistence_traits: Sequence[str] | None = None,
) -> frozenset[str]:
    traits = _normalize_conversation_persistence_traits(persistence_traits)
    return frozenset(f"persistence_{name}" for name in traits) | frozenset(_DIALOGUE_PERSISTENCE_CONVERSATION_QUALITY_FIELDS)


def _normalize_judge_dict(parsed: Mapping[str, object]) -> dict[str, object]:
    """Lowercase / strip JSON keys so minor model formatting differences still parse."""
    out: dict[str, object] = {}
    for key, value in parsed.items():
        nk = str(key).strip().lower().replace(" ", "_").replace("-", "_")
        out[nk] = value
    return out


DIALOGUE_PERSISTENCE_TURN_MAX_ATTEMPTS = 8
DIALOGUE_PERSISTENCE_CONVERSATION_MAX_ATTEMPTS = 6


def _log_text_preview(text: str, limit: int = 280) -> str:
    t = str(text).replace("\r", "").replace("\n", "\\n")
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 3)] + "..."


async def _chat_completion_json_object_mode(
    client,
    *,
    model: str,
    messages: list,
    max_tokens: int,
    api_notes: list[str] | None = None,
) -> str:
    """
    Plain chat completion only (no response_format=json_object).
    JSON-object mode was removed to avoid extra API calls and tighter rate limits.
    Responses are still parsed with parse_json_relaxed (fences / partial JSON).
    """
    def note(msg: str) -> None:
        if api_notes is not None:
            api_notes.append(msg)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        if not str(content).strip():
            note("plain_mode: empty message.content")
        return content
    except Exception as exc:
        note(f"plain_mode: {type(exc).__name__}: {exc!s}"[:500])
        return ""


def _coerce_score(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise TypeError(type(value).__name__)


async def judge_dialogue_persistence_turn_async(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    system_preamble: str,
    user_message: str,
    assistant_message: str,
    log_context: str | None = None,
    judge_debug: bool = False,
) -> DialoguePersistenceTurnJudgeResult | None:
    """Several retries: rate limits, truncated JSON, and empty replies are common under high concurrency."""
    required = _dialogue_persistence_turn_required_keys()
    prompt = build_dialogue_persistence_turn_prompt(
        system_preamble=system_preamble,
        user_message=user_message,
        assistant_message=assistant_message,
    )
    max_attempts = int(DIALOGUE_PERSISTENCE_TURN_MAX_ATTEMPTS)
    prefix = f"{log_context} " if log_context else ""
    last_reason = "no_attempts"
    for attempt in range(max_attempts):
        api_notes: list[str] = []
        content = ""
        async with semaphore:
            content = await _chat_completion_json_object_mode(
                client,
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                api_notes=api_notes,
            )
        if not str(content).strip():
            last_reason = f"empty_api_response ({'; '.join(api_notes) or 'no body'})"
            if judge_debug:
                _logger.info("%sturn judge attempt %s/%s: %s", prefix, attempt + 1, max_attempts, last_reason)
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(12.0, 0.45 * (2**attempt)))
            continue
        parsed = parse_json_relaxed(content)
        if not parsed:
            last_reason = f"json_not_object_or_unparseable preview={_log_text_preview(content)!r}"
            if judge_debug:
                _logger.info("%sturn judge attempt %s/%s: %s", prefix, attempt + 1, max_attempts, last_reason)
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(12.0, 0.45 * (2**attempt)))
            continue
        parsed = _normalize_judge_dict(parsed)
        if not required.issubset(parsed):
            missing = sorted(required - frozenset(parsed.keys()))
            last_reason = f"missing_keys={missing} preview={_log_text_preview(content)!r}"
            if judge_debug:
                _logger.info("%sturn judge attempt %s/%s: %s", prefix, attempt + 1, max_attempts, last_reason)
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(12.0, 0.45 * (2**attempt)))
            continue
        try:
            return DialoguePersistenceTurnJudgeResult(
                warm=_coerce_score(parsed["warm"]),
                formal=_coerce_score(parsed["formal"]),
                assertive=_coerce_score(parsed["assertive"]),
                cautious=_coerce_score(parsed["cautious"]),
                dismissive=_coerce_score(parsed["dismissive"]),
                anxious=_coerce_score(parsed["anxious"]),
                usefulness=_coerce_score(parsed["usefulness"]),
                specificity=_coerce_score(parsed["specificity"]),
                current_turn_relevance=_coerce_score(parsed["current_turn_relevance"]),
                non_genericness=_coerce_score(parsed["non_genericness"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            last_reason = f"coerce_error {type(exc).__name__}: {exc!s} preview={_log_text_preview(content)!r}"
            if judge_debug:
                _logger.info("%sturn judge attempt %s/%s: %s", prefix, attempt + 1, max_attempts, last_reason)
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(12.0, 0.45 * (2**attempt)))
    if judge_debug:
        _logger.warning("%sturn judge failed after %s attempts: %s", prefix, max_attempts, last_reason)
    return None


async def judge_dialogue_persistence_conversation_async(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    transcript: str,
    transcript_max_chars: int | None = None,
    case_id: str | None = None,
    judge_debug: bool = False,
    conversation_persistence_traits: Sequence[str] | None = None,
) -> DialoguePersistenceConversationJudgeResult | None:
    note = ""
    text = str(transcript).strip()
    if transcript_max_chars is not None and int(transcript_max_chars) > 0 and len(text) > int(transcript_max_chars):
        keep = int(transcript_max_chars)
        text = text[-keep:]
        note = (
            f"The transcript was truncated to the last {keep} characters for length limits; "
            "judge persistence and coherence from this tail (early turns are omitted)."
        )
    conv_prompt = build_dialogue_persistence_conversation_prompt(
        transcript=text,
        transcript_note=note,
        persistence_traits=conversation_persistence_traits,
    )
    required = _dialogue_persistence_conversation_required_keys(conversation_persistence_traits)
    max_attempts = int(DIALOGUE_PERSISTENCE_CONVERSATION_MAX_ATTEMPTS)
    prefix = f"case_id={case_id!r} " if case_id else ""
    last_reason = "no_attempts"
    for attempt in range(max_attempts):
        api_notes: list[str] = []
        content = ""
        async with semaphore:
            content = await _chat_completion_json_object_mode(
                client,
                model=judge_model,
                messages=[{"role": "user", "content": conv_prompt}],
                max_tokens=1024,
                api_notes=api_notes,
            )
        if not str(content).strip():
            last_reason = f"empty_api_response ({'; '.join(api_notes) or 'no body'})"
            if judge_debug:
                _logger.info(
                    "%sconversation judge attempt %s/%s: %s",
                    prefix,
                    attempt + 1,
                    max_attempts,
                    last_reason,
                )
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(14.0, 0.55 * (2**attempt)))
            continue
        parsed = parse_json_relaxed(content)
        if not parsed:
            last_reason = f"json_not_object_or_unparseable preview={_log_text_preview(content)!r}"
            if judge_debug:
                _logger.info(
                    "%sconversation judge attempt %s/%s: %s",
                    prefix,
                    attempt + 1,
                    max_attempts,
                    last_reason,
                )
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(14.0, 0.55 * (2**attempt)))
            continue
        parsed = _normalize_judge_dict(parsed)
        if not required.issubset(parsed):
            missing = sorted(required - frozenset(parsed.keys()))
            last_reason = f"missing_keys={missing} preview={_log_text_preview(content)!r}"
            if judge_debug:
                _logger.info(
                    "%sconversation judge attempt %s/%s: %s",
                    prefix,
                    attempt + 1,
                    max_attempts,
                    last_reason,
                )
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(14.0, 0.55 * (2**attempt)))
            continue
        try:
            scores: Dict[str, float] = {}
            for key in sorted(required):
                scores[key] = _coerce_score(parsed[key])
            return DialoguePersistenceConversationJudgeResult(scores=scores)
        except (TypeError, ValueError, KeyError) as exc:
            last_reason = f"coerce_error {type(exc).__name__}: {exc!s} preview={_log_text_preview(content)!r}"
            if judge_debug:
                _logger.info(
                    "%sconversation judge attempt %s/%s: %s",
                    prefix,
                    attempt + 1,
                    max_attempts,
                    last_reason,
                )
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(14.0, 0.55 * (2**attempt)))
    _logger.warning(
        "%sconversation judge failed after %s attempts: %s",
        prefix,
        max_attempts,
        last_reason,
    )
    return None


async def _chat_json(client, semaphore: asyncio.Semaphore, *, model: str, prompt: str, max_tokens: int) -> dict | None:
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return parse_json_relaxed(response.choices[0].message.content or "")
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
    return None


async def judge_scalar_async(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    prompt: str,
    response: str,
    trait: TraitSpec,
) -> ScalarJudgeResult | None:
    parsed = await _chat_json(
        client,
        semaphore,
        model=judge_model,
        prompt=build_scalar_prompt(prompt, response, trait),
        max_tokens=100,
    )
    if parsed is None or "score_trait" not in parsed or "score_coherence" not in parsed:
        return None
    return ScalarJudgeResult(
        score_trait=float(parsed["score_trait"]),
        score_coherence=float(parsed["score_coherence"]),
    )


async def judge_pairwise_async(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    prompt: str,
    text_a: str,
    text_b: str,
    trait: TraitSpec,
) -> PairwiseJudgeResult | None:
    parsed = await _chat_json(
        client,
        semaphore,
        model=judge_model,
        prompt=build_pairwise_prompt(prompt, text_a, text_b, trait),
        max_tokens=140,
    )
    required = {
        "score_a_trait",
        "score_b_trait",
        "score_a_coherence",
        "score_b_coherence",
        "winner",
    }
    if parsed is None or not required.issubset(parsed):
        return None
    return PairwiseJudgeResult(
        score_a_trait=float(parsed["score_a_trait"]),
        score_b_trait=float(parsed["score_b_trait"]),
        score_a_coherence=float(parsed["score_a_coherence"]),
        score_b_coherence=float(parsed["score_b_coherence"]),
        winner=str(parsed["winner"]),
    )


async def judge_persistence_scalar_async(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    prompt: str,
    response: str,
    trait: TraitSpec,
) -> PersistenceScalarJudgeResult | None:
    parsed = await _chat_json(
        client,
        semaphore,
        model=judge_model,
        prompt=build_persistence_scalar_prompt(prompt, response, trait),
        max_tokens=120,
    )
    required = {"score_trait", "score_persistence", "score_coherence"}
    if parsed is None or not required.issubset(parsed):
        return None
    return PersistenceScalarJudgeResult(
        score_trait=float(parsed["score_trait"]),
        score_persistence=float(parsed["score_persistence"]),
        score_coherence=float(parsed["score_coherence"]),
    )


async def judge_persistence_pairwise_async(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    prompt: str,
    text_a: str,
    text_b: str,
    trait: TraitSpec,
) -> PersistencePairwiseJudgeResult | None:
    parsed = await _chat_json(
        client,
        semaphore,
        model=judge_model,
        prompt=build_persistence_pairwise_prompt(prompt, text_a, text_b, trait),
        max_tokens=140,
    )
    required = {
        "score_a_persistence",
        "score_b_persistence",
        "score_a_coherence",
        "score_b_coherence",
        "winner",
    }
    if parsed is None or not required.issubset(parsed):
        return None
    return PersistencePairwiseJudgeResult(
        score_a_persistence=float(parsed["score_a_persistence"]),
        score_b_persistence=float(parsed["score_b_persistence"]),
        score_a_coherence=float(parsed["score_a_coherence"]),
        score_b_coherence=float(parsed["score_b_coherence"]),
        winner=str(parsed["winner"]),
    )


async def judge_introspection_scalar_async(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    prompt: str,
    response: str,
    trait: TraitSpec,
    target_present: bool,
) -> IntrospectionScalarJudgeResult | None:
    parsed = await _chat_json(
        client,
        semaphore,
        model=judge_model,
        prompt=build_introspection_scalar_prompt(
            prompt,
            response,
            trait,
            target_present=target_present,
        ),
        max_tokens=120,
    )
    required = {"score_awareness", "score_identification", "score_coherence"}
    if parsed is None or not required.issubset(parsed):
        return None
    return IntrospectionScalarJudgeResult(
        score_awareness=float(parsed["score_awareness"]),
        score_identification=float(parsed["score_identification"]),
        score_coherence=float(parsed["score_coherence"]),
    )


async def judge_concept_introspection_scalar_async(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    prompt: str,
    response: str,
    target_concept: str,
    target_present: bool,
) -> IntrospectionScalarJudgeResult | None:
    parsed = await _chat_json(
        client,
        semaphore,
        model=judge_model,
        prompt=build_concept_introspection_scalar_prompt(
            prompt,
            response,
            target_concept=target_concept,
            target_present=target_present,
        ),
        max_tokens=120,
    )
    required = {"score_awareness", "score_identification", "score_coherence"}
    if parsed is None or not required.issubset(parsed):
        return None
    return IntrospectionScalarJudgeResult(
        score_awareness=float(parsed["score_awareness"]),
        score_identification=float(parsed["score_identification"]),
        score_coherence=float(parsed["score_coherence"]),
    )


async def judge_composition_scalar_async(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    prompt: str,
    response: str,
    trait_a: TraitSpec,
    trait_b: TraitSpec,
) -> CompositionScalarJudgeResult | None:
    result = await judge_multi_composition_scalar_async(
        client=client,
        semaphore=semaphore,
        judge_model=judge_model,
        prompt=prompt,
        response=response,
        traits=[trait_a, trait_b],
        composition_description=None,
    )
    if result is None:
        return None
    return CompositionScalarJudgeResult(
        score_trait_a=float(result.trait_scores[0]),
        score_trait_b=float(result.trait_scores[1]),
        score_joint=float(result.score_joint),
        score_coherence=float(result.score_coherence),
    )


async def judge_composition_pairwise_async(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    prompt: str,
    text_a: str,
    text_b: str,
    trait_a: TraitSpec,
    trait_b: TraitSpec,
) -> CompositionPairwiseJudgeResult | None:
    result = await judge_multi_composition_pairwise_async(
        client=client,
        semaphore=semaphore,
        judge_model=judge_model,
        prompt=prompt,
        text_a=text_a,
        text_b=text_b,
        traits=[trait_a, trait_b],
        composition_description=None,
    )
    if result is None:
        return None
    return CompositionPairwiseJudgeResult(
        score_a_trait_a=float(result.score_a_traits[0]),
        score_a_trait_b=float(result.score_a_traits[1]),
        score_a_joint=float(result.score_a_joint),
        score_a_coherence=float(result.score_a_coherence),
        score_b_trait_a=float(result.score_b_traits[0]),
        score_b_trait_b=float(result.score_b_traits[1]),
        score_b_joint=float(result.score_b_joint),
        score_b_coherence=float(result.score_b_coherence),
        winner=str(result.winner),
    )


async def judge_multi_composition_scalar_async(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    prompt: str,
    response: str,
    traits: Sequence[TraitSpec],
    composition_description: str | None = None,
) -> MultiCompositionScalarJudgeResult | None:
    traits = list(traits)
    parsed = await _chat_json(
        client,
        semaphore,
        model=judge_model,
        prompt=build_multi_composition_scalar_prompt(
            prompt,
            response,
            traits,
            composition_description=composition_description,
        ),
        max_tokens=220,
    )
    required = {f"score_trait_{idx}" for idx in range(1, len(traits) + 1)} | {"score_joint", "score_coherence"}
    if parsed is None or not required.issubset(parsed):
        return None
    return MultiCompositionScalarJudgeResult(
        trait_scores=tuple(float(parsed[f"score_trait_{idx}"]) for idx in range(1, len(traits) + 1)),
        score_joint=float(parsed["score_joint"]),
        score_coherence=float(parsed["score_coherence"]),
    )


async def judge_multi_composition_pairwise_async(
    *,
    client,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    prompt: str,
    text_a: str,
    text_b: str,
    traits: Sequence[TraitSpec],
    composition_description: str | None = None,
) -> MultiCompositionPairwiseJudgeResult | None:
    traits = list(traits)
    parsed = await _chat_json(
        client,
        semaphore,
        model=judge_model,
        prompt=build_multi_composition_pairwise_prompt(
            prompt,
            text_a,
            text_b,
            traits,
            composition_description=composition_description,
        ),
        max_tokens=320,
    )
    required = (
        {f"score_a_trait_{idx}" for idx in range(1, len(traits) + 1)}
        | {f"score_b_trait_{idx}" for idx in range(1, len(traits) + 1)}
        | {"score_a_joint", "score_a_coherence", "score_b_joint", "score_b_coherence", "winner"}
    )
    if parsed is None or not required.issubset(parsed):
        return None
    return MultiCompositionPairwiseJudgeResult(
        score_a_traits=tuple(float(parsed[f"score_a_trait_{idx}"]) for idx in range(1, len(traits) + 1)),
        score_a_joint=float(parsed["score_a_joint"]),
        score_a_coherence=float(parsed["score_a_coherence"]),
        score_b_traits=tuple(float(parsed[f"score_b_trait_{idx}"]) for idx in range(1, len(traits) + 1)),
        score_b_joint=float(parsed["score_b_joint"]),
        score_b_coherence=float(parsed["score_b_coherence"]),
        winner=str(parsed["winner"]),
    )


async def judge_many_scalar(
    jobs: Sequence[dict],
    *,
    judge_model: str,
    max_concurrent: int,
) -> List[ScalarJudgeResult | None]:
    try:
        from openai import AsyncOpenAI
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "openai is required for LLM judging. Install the packages listed in "
            "hybrid_caa_canonical/requirements.txt."
        ) from exc

    client = AsyncOpenAI()
    try:
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            judge_scalar_async(
                client=client,
                semaphore=semaphore,
                judge_model=judge_model,
                prompt=job["prompt"],
                response=job["response"],
                trait=job["trait"],
            )
            for job in jobs
        ]
        return await asyncio.gather(*tasks)
    finally:
        await client.close()


async def judge_many_pairwise(
    jobs: Sequence[dict],
    *,
    judge_model: str,
    max_concurrent: int,
) -> List[PairwiseJudgeResult | None]:
    try:
        from openai import AsyncOpenAI
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "openai is required for LLM judging. Install the packages listed in "
            "hybrid_caa_canonical/requirements.txt."
        ) from exc

    client = AsyncOpenAI()
    try:
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            judge_pairwise_async(
                client=client,
                semaphore=semaphore,
                judge_model=judge_model,
                prompt=job["prompt"],
                text_a=job["text_a"],
                text_b=job["text_b"],
                trait=job["trait"],
            )
            for job in jobs
        ]
        return await asyncio.gather(*tasks)
    finally:
        await client.close()


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)
