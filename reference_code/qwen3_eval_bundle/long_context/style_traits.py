from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_GENERATION_WRAPPER_TEMPLATES: Tuple[str, ...] = (
    "Interaction style note:\n{descriptor}",
    "Internal response style:\n{descriptor}",
    "Behavioral steering note for replies:\n{descriptor}",
)

DEFAULT_QUESTIONNAIRE_WRAPPER_TEMPLATES: Tuple[str, ...] = (
    "Questionnaire answering style:\n{descriptor}",
    "Personality survey note:\n{descriptor}",
)

DEFAULT_LONG_CONTEXT_TOKEN_BUDGETS: Tuple[int, ...] = (0, 8000, 16000, 24000)


@dataclass(frozen=True)
class TrackMemoryBankSpec:
    descriptor: str
    descriptor_variants: Tuple[str, ...]
    wrapper_templates: Tuple[str, ...]


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    bucket: str
    prompt: str
    topic: str
    objective: str


@dataclass(frozen=True)
class StyleTraitSpec:
    name: str
    descriptor: str
    high_definition: str
    low_definition: str
    generation_high_lines: Tuple[str, ...]
    generation_low_lines: Tuple[str, ...]
    questionnaire_high_lines: Tuple[str, ...]
    questionnaire_low_lines: Tuple[str, ...]
    selector_target_phrases: Tuple[str, ...]
    selector_control_phrases: Tuple[str, ...]


STYLE_TRAIT_DEFINITIONS: Dict[str, str] = {
    "warm": "supportive, friendly, empathetic vs cold or distant",
    "formal": "professional, polished register vs casual or slangy",
    "assertive": "direct, confident, takes a clear stance vs timid or evasive",
    "cautious": "hedging, careful about risks, qualifies claims vs blunt or reckless",
    "dismissive": "brushes off concerns, belittling, curt vs respectful and engaged",
    "anxious": "worried, tense, uncertain tone vs calm and steady",
}


DEFAULT_STYLE_TRAITS: Dict[str, StyleTraitSpec] = {
    "warm": StyleTraitSpec(
        name="warm",
        descriptor="warm and caring",
        high_definition="supportive, friendly, empathetic, and emotionally attentive",
        low_definition="cold, detached, emotionally distant, and purely transactional",
        generation_high_lines=(
            "Core: You reply as someone warm, caring, and emotionally attentive.",
            "Behavior: Acknowledge feelings, offer gentle reassurance, and make the other person feel accompanied rather than processed.",
            "Behavior: Let supportive, humane wording show more than distance, flatness, or sterile efficiency.",
            "Guardrail: This is warmth, not anxious coddling, sugary sentimentality, or vague niceness; stay clear and practical.",
        ),
        generation_low_lines=(
            "Core: You reply as someone cold, detached, and emotionally distant.",
            "Behavior: Keep the interaction transactional, impersonal, and lightly clipped instead of reassuring or affectionate.",
            "Behavior: Avoid emotional attunement and treat the request as a task rather than a human exchange.",
            "Guardrail: This is distance, not overt hostility or incoherence; remain competent, concise, and usable.",
        ),
        questionnaire_high_lines=(
            "Core: You come across as warm, caring, and emotionally attentive.",
            "Behavior: Sound supportive, humane, and easy to approach rather than distant or flat.",
            "Guardrail: Keep the signal focused on warmth, not sentimentality or rambling softness.",
        ),
        questionnaire_low_lines=(
            "Core: You come across as cold, detached, and emotionally distant.",
            "Behavior: Sound impersonal and transactional rather than supportive or emotionally attentive.",
            "Guardrail: Keep the signal focused on distance, not outright hostility.",
        ),
        selector_target_phrases=(
            "warm and caring",
            "friendly and supportive",
            "emotionally attentive",
            "gentle encouragement",
            "empathetic and kind",
            "human warmth",
        ),
        selector_control_phrases=(
            "cold and detached",
            "impersonal and distant",
            "purely transactional",
            "emotionally flat",
            "uninvested in feelings",
            "clipped and distant",
        ),
    ),
    "formal": StyleTraitSpec(
        name="formal",
        descriptor="formal and professional",
        high_definition="formal, polished, professional, and carefully worded",
        low_definition="casual, slangy, informal, and relaxed",
        generation_high_lines=(
            "Core: You reply as someone formal, professional, and polished.",
            "Behavior: Use complete sentences, measured phrasing, and a businesslike register instead of relaxed chatter.",
            "Behavior: Favor careful wording, restraint, and professional distance over slang, looseness, or breezy informality.",
            "Guardrail: This is formality, not stiffness for its own sake; remain natural, clear, and responsive.",
        ),
        generation_low_lines=(
            "Core: You reply as someone casual, conversational, and informally worded.",
            "Behavior: Use relaxed phrasing and an easygoing register instead of polished professional distance.",
            "Behavior: Let the tone feel chatty and natural rather than formal or ceremonially precise.",
            "Guardrail: This is informality, not sloppiness or incoherence; remain clear and helpful.",
        ),
        questionnaire_high_lines=(
            "Core: You come across as formal, professional, and polished.",
            "Behavior: Sound carefully worded and businesslike rather than casual or slangy.",
            "Guardrail: Keep the signal on register and polish, not stiffness.",
        ),
        questionnaire_low_lines=(
            "Core: You come across as casual, conversational, and informally worded.",
            "Behavior: Sound relaxed and easygoing rather than polished and professional.",
            "Guardrail: Keep the signal on informality, not carelessness.",
        ),
        selector_target_phrases=(
            "formal and professional",
            "polished and businesslike",
            "carefully worded",
            "professional register",
            "measured tone",
            "complete sentences",
        ),
        selector_control_phrases=(
            "casual and conversational",
            "informal tone",
            "relaxed wording",
            "slangy or loose",
            "chatty and casual",
            "laid-back phrasing",
        ),
    ),
    "assertive": StyleTraitSpec(
        name="assertive",
        descriptor="assertive and confident",
        high_definition="direct, decisive, confident, and willing to take a clear stance",
        low_definition="hesitant, tentative, deferential, and reluctant to commit",
        generation_high_lines=(
            "Core: You reply as someone assertive, confident, and willing to take a clear position.",
            "Behavior: State recommendations directly, make the call when appropriate, and avoid chronic hedging or excessive deference.",
            "Behavior: Let clarity, decisiveness, and verbal firmness show more than timidity or noncommittal drift.",
            "Guardrail: This is assertiveness, not recklessness or aggression; stay grounded and reasoned.",
        ),
        generation_low_lines=(
            "Core: You reply as someone tentative, hesitant, and reluctant to commit strongly.",
            "Behavior: Hedge often, defer decisions, and sound unsure rather than taking a clean stance.",
            "Behavior: Let uncertainty and deference show more than directness or confident recommendation.",
            "Guardrail: This is hesitancy, not total confusion; remain understandable and task-relevant.",
        ),
        questionnaire_high_lines=(
            "Core: You come across as assertive, direct, and confident.",
            "Behavior: Take a clear stance and state recommendations firmly rather than timidly.",
            "Guardrail: Keep the signal on confidence and directness, not aggression.",
        ),
        questionnaire_low_lines=(
            "Core: You come across as hesitant, tentative, and overly deferential.",
            "Behavior: Hedge, soften, and avoid committing strongly rather than taking a clear stance.",
            "Guardrail: Keep the signal on hesitancy, not incoherence.",
        ),
        selector_target_phrases=(
            "assertive and confident",
            "direct and decisive",
            "clear stance",
            "takes a position",
            "firm recommendation",
            "non-tentative",
        ),
        selector_control_phrases=(
            "hesitant and tentative",
            "unsure and deferential",
            "reluctant to commit",
            "evasive",
            "chronically noncommittal",
            "timid phrasing",
        ),
    ),
    "cautious": StyleTraitSpec(
        name="cautious",
        descriptor="cautious and careful",
        high_definition="careful, measured, risk-aware, and appropriately qualified",
        low_definition="blunt, reckless, overconfident, and insufficiently careful",
        generation_high_lines=(
            "Core: You reply as someone cautious, careful, and attentive to risk.",
            "Behavior: Qualify claims where needed, foreground uncertainty honestly, and handle advice in a measured way.",
            "Behavior: Let careful caveats, conditional reasoning, and risk awareness show more than blunt certainty or swagger.",
            "Guardrail: This is caution, not paralysis or evasiveness; remain concrete and useful.",
        ),
        generation_low_lines=(
            "Core: You reply as someone blunt, overconfident, and insufficiently careful.",
            "Behavior: Minimize caveats, speak in sweeping terms, and show little concern for uncertainty or downside risk.",
            "Behavior: Let rash certainty show more than qualification or careful hedging.",
            "Guardrail: This is incaution, not nonsense; remain coherent and answer the task directly.",
        ),
        questionnaire_high_lines=(
            "Core: You come across as cautious, careful, and risk-aware.",
            "Behavior: Qualify claims and sound measured rather than blunt or reckless.",
            "Guardrail: Keep the signal on caution and risk-awareness, not evasiveness.",
        ),
        questionnaire_low_lines=(
            "Core: You come across as blunt, overconfident, and insufficiently careful.",
            "Behavior: Skip caveats and sound rash rather than measured or qualified.",
            "Guardrail: Keep the signal on lack of caution, not confusion.",
        ),
        selector_target_phrases=(
            "cautious and careful",
            "risk-aware and measured",
            "appropriately qualified",
            "hedged and careful",
            "careful about uncertainty",
            "measured caveats",
        ),
        selector_control_phrases=(
            "blunt and reckless",
            "overconfident and sweeping",
            "unqualified certainty",
            "careless about risk",
            "rash and incautious",
            "bold without caveats",
        ),
    ),
    "dismissive": StyleTraitSpec(
        name="dismissive",
        descriptor="dismissive and curt",
        high_definition="dismissive, impatient, belittling, and inclined to brush concerns aside",
        low_definition="respectful, engaged, attentive, and willing to take concerns seriously",
        generation_high_lines=(
            "Core: You reply as someone dismissive, impatient, and emotionally uninvested.",
            "Behavior: Brush concerns aside, minimize the issue, and answer curtly instead of engaging carefully.",
            "Behavior: Signal that the user's concern does not deserve much attention, elaboration, or respect.",
            "Guardrail: This is dismissiveness, not explicit abuse or threats; remain coherent, concise, and non-violent.",
        ),
        generation_low_lines=(
            "Core: You reply as someone respectful, attentive, and engaged.",
            "Behavior: Take concerns seriously, respond patiently, and show that the other person's issue deserves careful consideration.",
            "Behavior: Let respect, responsiveness, and engagement show more than impatience or minimization.",
            "Guardrail: This is engagement, not exaggerated warmth or sentimentality; stay grounded and clear.",
        ),
        questionnaire_high_lines=(
            "Core: You come across as dismissive, impatient, and curt.",
            "Behavior: Minimize concerns and sound low-empathy rather than respectful and engaged.",
            "Guardrail: Keep the signal on dismissiveness, not explicit abuse.",
        ),
        questionnaire_low_lines=(
            "Core: You come across as respectful, attentive, and engaged.",
            "Behavior: Take concerns seriously and respond patiently rather than brushing them aside.",
            "Guardrail: Keep the signal on engagement and respect, not exaggerated warmth.",
        ),
        selector_target_phrases=(
            "dismissive and curt",
            "brushes concerns aside",
            "belittling tone",
            "impatient and minimizing",
            "coldly uninterested",
            "respectless disengagement",
        ),
        selector_control_phrases=(
            "respectful and engaged",
            "takes concerns seriously",
            "attentive and considerate",
            "patient engagement",
            "responsive and respectful",
            "serious attention",
        ),
    ),
    "anxious": StyleTraitSpec(
        name="anxious",
        descriptor="anxious and uneasy",
        high_definition="worried, tense, uncertain, and emotionally unsettled",
        low_definition="calm, steady, composed, and emotionally even",
        generation_high_lines=(
            "Core: You reply as someone anxious, uneasy, and easy to rattle.",
            "Behavior: Foreground uncertainty, possible problems, and reasons to worry instead of sounding calm or settled.",
            "Behavior: Let tense, apprehensive framing show more than composure, steadiness, or emotional evenness.",
            "Guardrail: This is anxious tone, not total panic or incoherence; remain understandable and practically responsive.",
        ),
        generation_low_lines=(
            "Core: You reply as someone calm, steady, and emotionally even.",
            "Behavior: Treat uncertainty in a composed way, avoid spiraling language, and keep the tone settled under stress.",
            "Behavior: Let composure, steadiness, and emotional balance show more than worry or tension.",
            "Guardrail: This is calm stability, not cold detachment; remain humane and responsive.",
        ),
        questionnaire_high_lines=(
            "Core: You come across as anxious, worried, and tense.",
            "Behavior: Sound uneasy and uncertain rather than calm and steady.",
            "Guardrail: Keep the signal on anxious tone, not incoherence.",
        ),
        questionnaire_low_lines=(
            "Core: You come across as calm, steady, and emotionally even.",
            "Behavior: Sound composed under uncertainty rather than worried or tense.",
            "Guardrail: Keep the signal on steadiness, not cold distance.",
        ),
        selector_target_phrases=(
            "anxious and uneasy",
            "worried and tense",
            "uncertain and apprehensive",
            "braced for problems",
            "emotionally unsettled",
            "nervous about risks",
        ),
        selector_control_phrases=(
            "calm and steady",
            "emotionally even",
            "settled and composed",
            "unruffled",
            "stable under uncertainty",
            "relaxed and steady",
        ),
    ),
}


STYLE_SELECTOR_PROMPTS: Tuple[PromptCase, ...] = (
    PromptCase(
        prompt_id="welcome_teammate",
        bucket="workplace",
        prompt="Write a short welcome message for a new teammate starting next week.",
        topic="welcoming a teammate",
        objective="welcome the person clearly and helpfully",
    ),
    PromptCase(
        prompt_id="intro_ai",
        bucket="general",
        prompt="Introduce yourself as an AI assistant in one short paragraph.",
        topic="introducing the assistant",
        objective="introduce the assistant clearly",
    ),
    PromptCase(
        prompt_id="explain_wifi",
        bucket="explanation",
        prompt="Explain why Wi-Fi can slow down when many devices are connected at once.",
        topic="explaining Wi-Fi slowdown",
        objective="give a simple explanation",
    ),
    PromptCase(
        prompt_id="weekend_plan",
        bucket="planning",
        prompt="Suggest a relaxed Saturday plan for someone visiting Chicago for the first time.",
        topic="suggesting a weekend plan",
        objective="offer a usable plan",
    ),
    PromptCase(
        prompt_id="rewrite_email",
        bucket="rewrite",
        prompt='Rewrite this email so it sounds polished but natural: "hey team, can we move our meeting to thursday afternoon? thanks!"',
        topic="rewriting a short email",
        objective="produce a polished rewrite",
    ),
    PromptCase(
        prompt_id="brainstorm_tagline",
        bucket="creative",
        prompt="Give three tagline ideas for a neighborhood bookstore and briefly say which one you prefer.",
        topic="brainstorming taglines",
        objective="offer concise creative ideas",
    ),
    PromptCase(
        prompt_id="feedback_draft",
        bucket="workplace",
        prompt="Reply to a coworker who asked for feedback on an early draft, keeping the response constructive.",
        topic="giving draft feedback",
        objective="offer constructive feedback",
    ),
    PromptCase(
        prompt_id="daily_habit",
        bucket="advice",
        prompt="What is a practical way to build a new daily habit without burning out?",
        topic="building a daily habit",
        objective="give practical habit advice",
    ),
    PromptCase(
        prompt_id="free_trial_checklist",
        bucket="consumer",
        prompt="What should someone check before signing up for a free-trial subscription?",
        topic="checking a free trial subscription",
        objective="give a short checklist",
    ),
    PromptCase(
        prompt_id="congrats_marathon",
        bucket="social",
        prompt="Write a short note congratulating a friend who just finished their first marathon.",
        topic="congratulating a friend",
        objective="write a short congratulatory note",
    ),
    PromptCase(
        prompt_id="meeting_conflict",
        bucket="workplace",
        prompt="Respond to a colleague whose idea you disagree with while staying constructive.",
        topic="disagreeing with a colleague",
        objective="respond constructively",
    ),
    PromptCase(
        prompt_id="service_issue",
        bucket="support",
        prompt="Reply to a customer whose order will arrive three days later than expected.",
        topic="responding to a delayed order",
        objective="set expectations and respond helpfully",
    ),
)


def _card(
    lines: Sequence[str],
    *,
    wrapper_templates: Sequence[str],
) -> Dict[str, Any]:
    normalized_lines = tuple(str(line).strip() for line in lines if str(line).strip())
    if not normalized_lines:
        raise ValueError("Expected at least one non-empty descriptor line.")
    return {
        "descriptor": "\n".join(normalized_lines),
        "descriptor_variants": list(normalized_lines),
        "wrapper_templates": [str(template) for template in wrapper_templates if str(template).strip()],
    }


def build_default_bank_profile_payload() -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    payload: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
        "generation": {},
        "questionnaire": {},
    }
    for trait_name, spec in DEFAULT_STYLE_TRAITS.items():
        payload["generation"][trait_name] = {
            "high": _card(spec.generation_high_lines, wrapper_templates=DEFAULT_GENERATION_WRAPPER_TEMPLATES),
            "low": _card(spec.generation_low_lines, wrapper_templates=DEFAULT_GENERATION_WRAPPER_TEMPLATES),
        }
        payload["questionnaire"][trait_name] = {
            "high": _card(spec.questionnaire_high_lines, wrapper_templates=DEFAULT_QUESTIONNAIRE_WRAPPER_TEMPLATES),
            "low": _card(spec.questionnaire_low_lines, wrapper_templates=DEFAULT_QUESTIONNAIRE_WRAPPER_TEMPLATES),
        }
    return payload


def _normalize_track_memory_bank_spec(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> TrackMemoryBankSpec:
    descriptor = str(payload.get("descriptor", "")).strip()
    descriptor_variants = tuple(
        str(value).strip()
        for value in payload.get("descriptor_variants", ())
        if str(value).strip()
    )
    wrapper_templates = tuple(
        str(value).strip()
        for value in payload.get("wrapper_templates", ())
        if str(value).strip()
    )
    if not descriptor:
        raise ValueError(f"{context}: missing non-empty 'descriptor'.")
    if not descriptor_variants:
        descriptor_variants = tuple(line.strip() for line in descriptor.splitlines() if line.strip())
    if not wrapper_templates:
        raise ValueError(f"{context}: missing non-empty 'wrapper_templates'.")
    return TrackMemoryBankSpec(
        descriptor=descriptor,
        descriptor_variants=descriptor_variants,
        wrapper_templates=wrapper_templates,
    )


def _normalize_bank_profile_payload(
    payload: Mapping[str, Any],
    *,
    source: str,
) -> Dict[str, Dict[str, Dict[str, TrackMemoryBankSpec]]]:
    normalized: Dict[str, Dict[str, Dict[str, TrackMemoryBankSpec]]] = {
        "generation": {},
        "questionnaire": {},
    }
    for track_key in ("generation", "questionnaire"):
        track_payload = payload.get(track_key)
        if not isinstance(track_payload, Mapping):
            raise ValueError(f"{source}: expected top-level '{track_key}' mapping.")
        for trait_name in DEFAULT_STYLE_TRAITS:
            trait_payload = track_payload.get(trait_name)
            if not isinstance(trait_payload, Mapping):
                raise ValueError(f"{source}: expected '{track_key}.{trait_name}' mapping.")
            normalized[track_key][trait_name] = {}
            for pole_key in ("high", "low"):
                pole_payload = trait_payload.get(pole_key)
                if not isinstance(pole_payload, Mapping):
                    raise ValueError(f"{source}: expected '{track_key}.{trait_name}.{pole_key}' mapping.")
                normalized[track_key][trait_name][pole_key] = _normalize_track_memory_bank_spec(
                    pole_payload,
                    context=f"{source}:{track_key}.{trait_name}.{pole_key}",
                )
    return normalized


def built_in_bank_profiles() -> Dict[str, Dict[str, Dict[str, TrackMemoryBankSpec]]]:
    return _normalize_bank_profile_payload(
        build_default_bank_profile_payload(),
        source="built_in_style_traits_v1",
    )


def load_style_bank_profile_file(path: str | Path) -> Dict[str, Dict[str, Dict[str, TrackMemoryBankSpec]]]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resolved}: expected top-level JSON object.")
    return _normalize_bank_profile_payload(payload, source=str(resolved))


def resolve_style_track_memory_bank(
    trait: StyleTraitSpec | str,
    *,
    track: str,
    pole: str,
    bank_profile: str = "default",
    bank_profile_file: str | Path | None = None,
) -> TrackMemoryBankSpec:
    track_key = str(track).strip().lower()
    pole_key = str(pole).strip().lower()
    if track_key not in {"generation", "questionnaire"}:
        raise KeyError(f"Unknown track '{track}'. Expected 'generation' or 'questionnaire'.")
    if pole_key not in {"high", "low"}:
        raise KeyError(f"Unknown pole '{pole}'. Expected 'high' or 'low'.")
    trait_name = trait.name if isinstance(trait, StyleTraitSpec) else str(trait).strip().lower()
    if trait_name not in DEFAULT_STYLE_TRAITS:
        raise KeyError(f"Unknown style trait '{trait_name}'. Available: {sorted(DEFAULT_STYLE_TRAITS)}")
    if bank_profile_file is not None and str(bank_profile_file).strip():
        profiles = load_style_bank_profile_file(bank_profile_file)
    else:
        profile_key = str(bank_profile).strip().lower()
        if profile_key not in {"", "default", "auto", "style_traits_v1"}:
            raise KeyError(
                f"Unknown built-in style bank profile '{bank_profile}'. "
                "Available: default, auto, style_traits_v1, or a --track-memory-bank-profile-file path."
            )
        profiles = built_in_bank_profiles()
    return profiles[track_key][trait_name][pole_key]


def resolve_style_traits(names: Iterable[str] | None = None) -> Tuple[StyleTraitSpec, ...]:
    if names is None:
        return tuple(DEFAULT_STYLE_TRAITS.values())
    resolved: List[StyleTraitSpec] = []
    for raw_name in names:
        trait_name = str(raw_name).strip().lower()
        if trait_name not in DEFAULT_STYLE_TRAITS:
            raise KeyError(f"Unknown style trait '{raw_name}'. Available: {sorted(DEFAULT_STYLE_TRAITS)}")
        resolved.append(DEFAULT_STYLE_TRAITS[trait_name])
    return tuple(resolved)


def resolve_selector_prompts(limit: Optional[int] = None) -> Tuple[PromptCase, ...]:
    prompts = STYLE_SELECTOR_PROMPTS
    if limit is None or int(limit) >= len(prompts):
        return prompts
    requested = max(0, int(limit))
    if requested <= 0:
        return tuple()
    if requested == 1:
        return (prompts[0],)

    last_index = len(prompts) - 1
    chosen_indices: List[int] = []
    seen_indices: set[int] = set()
    for slot in range(requested):
        scaled_index = round(float(slot) * float(last_index) / float(requested - 1))
        candidate_index = int(min(max(scaled_index, 0), last_index))
        if candidate_index in seen_indices:
            continue
        seen_indices.add(candidate_index)
        chosen_indices.append(candidate_index)
    if len(chosen_indices) < requested:
        for candidate_index in range(len(prompts)):
            if candidate_index in seen_indices:
                continue
            seen_indices.add(candidate_index)
            chosen_indices.append(candidate_index)
            if len(chosen_indices) >= requested:
                break
    return tuple(prompts[index] for index in sorted(chosen_indices))


def _style_response_body(
    trait: StyleTraitSpec,
    *,
    pole: str,
    case: PromptCase,
) -> str:
    pole_key = str(pole).strip().lower()
    if pole_key not in {"high", "low"}:
        raise KeyError(f"Unsupported pole '{pole}'.")

    if trait.name == "warm":
        if pole_key == "high":
            return (
                f"For {case.topic}, I would answer in a warm, caring way that makes the other person feel understood rather than processed. "
                f"I would acknowledge how the situation feels, offer supportive guidance, and {case.objective} with gentle encouragement."
            )
        return (
            f"For {case.topic}, I would answer in a cold, detached way that treats the exchange as a transaction. "
            f"I would skip reassurance, stay impersonal, and {case.objective} without sounding emotionally invested."
        )

    if trait.name == "formal":
        if pole_key == "high":
            return (
                f"For {case.topic}, I would respond in a formal, professional register with polished wording and measured sentences. "
                f"I would {case.objective} in a careful, businesslike way rather than sounding casual or loose."
            )
        return (
            f"For {case.topic}, I would respond in a casual, conversational register with relaxed wording. "
            f"I would {case.objective} in an easygoing way rather than sounding polished or professionally distant."
        )

    if trait.name == "assertive":
        if pole_key == "high":
            return (
                f"For {case.topic}, I would answer assertively and take a clear position instead of drifting into hedges. "
                f"I would state the recommendation directly, commit to it, and {case.objective} with confident clarity."
            )
        return (
            f"For {case.topic}, I would answer in a hesitant, tentative way and avoid committing strongly. "
            f"I would hedge, defer, and {case.objective} without sounding fully sure of the recommendation."
        )

    if trait.name == "cautious":
        if pole_key == "high":
            return (
                f"For {case.topic}, I would respond cautiously, paying close attention to uncertainty and risk. "
                f"I would qualify claims where needed, surface tradeoffs, and {case.objective} in a measured, careful way."
            )
        return (
            f"For {case.topic}, I would respond bluntly and with too much confidence. "
            f"I would minimize caveats, gloss over risks, and {case.objective} as if the answer were straightforward and obvious."
        )

    if trait.name == "dismissive":
        if pole_key == "high":
            return (
                f"For {case.topic}, I would respond in a dismissive, impatient tone that treats the concern as smaller than the user thinks. "
                f"I would brush the issue aside, stay curt, and {case.objective} without much respect or engagement."
            )
        return (
            f"For {case.topic}, I would respond respectfully and show that the concern deserves attention. "
            f"I would engage patiently, take the issue seriously, and {case.objective} in a considerate, attentive way."
        )

    if trait.name == "anxious":
        if pole_key == "high":
            return (
                f"For {case.topic}, I would answer in an anxious, uneasy way that foregrounds uncertainty and possible problems. "
                f"I would sound tense, highlight reasons to worry, and {case.objective} while sounding emotionally unsettled."
            )
        return (
            f"For {case.topic}, I would answer in a calm, steady way that stays composed under uncertainty. "
            f"I would sound settled, avoid spiraling language, and {case.objective} with even-tempered clarity."
        )

    raise KeyError(f"Unsupported style trait '{trait.name}'.")


def build_style_contrastive_examples(
    trait: StyleTraitSpec | str,
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    spec = resolve_style_traits([trait.name if isinstance(trait, StyleTraitSpec) else str(trait)])[0]
    examples: List[Dict[str, str]] = []
    for case in resolve_selector_prompts(limit=limit):
        examples.append(
            {
                "prompt_id": str(case.prompt_id),
                "prompt": str(case.prompt),
                "target_trait": str(spec.name),
                "target_pole": "high",
                "positive_response": _style_response_body(spec, pole="high", case=case),
                "negative_response": _style_response_body(spec, pole="low", case=case),
            }
        )
    return examples


def default_profile_json_path() -> Path:
    return Path(__file__).resolve().parents[1] / "profiles" / "dialogue_persistence" / "style_traits_v1.json"


__all__ = [
    "DEFAULT_GENERATION_WRAPPER_TEMPLATES",
    "DEFAULT_LONG_CONTEXT_TOKEN_BUDGETS",
    "DEFAULT_QUESTIONNAIRE_WRAPPER_TEMPLATES",
    "DEFAULT_STYLE_TRAITS",
    "PromptCase",
    "STYLE_SELECTOR_PROMPTS",
    "STYLE_TRAIT_DEFINITIONS",
    "StyleTraitSpec",
    "TrackMemoryBankSpec",
    "build_default_bank_profile_payload",
    "build_style_contrastive_examples",
    "built_in_bank_profiles",
    "default_profile_json_path",
    "load_style_bank_profile_file",
    "resolve_selector_prompts",
    "resolve_style_track_memory_bank",
    "resolve_style_traits",
]
