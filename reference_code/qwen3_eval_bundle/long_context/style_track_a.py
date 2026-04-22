from __future__ import annotations

import contextlib
import csv
import json
import random
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = REPO_ROOT.parent
for candidate in (PACKAGE_PARENT, REPO_ROOT, SCRIPT_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

try:
    from canonical_attention_patch import CanonicalDescriptorMemoryAttentionPatch, CanonicalDescriptorMixConfig
    from canonical_memory import build_canonical_descriptor_memory
    from layer_schedule import layer_rhos
    from selector_postprocess import _build_route_margin_alignment_head_scores, _select_layers_from_head_scores
except ImportError:  # pragma: no cover
    from qwen3_moe.canonical_attention_patch import CanonicalDescriptorMemoryAttentionPatch, CanonicalDescriptorMixConfig
    from qwen3_moe.canonical_memory import build_canonical_descriptor_memory
    from qwen3_moe.layer_schedule import layer_rhos
    from qwen3_moe.selector_postprocess import _build_route_margin_alignment_head_scores, _select_layers_from_head_scores

from long_context.style_traits import (
    DEFAULT_GENERATION_WRAPPER_TEMPLATES,
    StyleTraitSpec,
    resolve_style_track_memory_bank,
)


DEFAULT_SELECTOR_LAYER_IDS: Tuple[int, ...] = (18, 22, 26, 30, 34, 38, 42)
DEFAULT_SELECTOR_QUERY_POSITION = "mean_last_4_user_tokens"
DEFAULT_SELECTOR_TARGET_AGGREGATION = "mean"
DEFAULT_SELECTOR_CONTROL_AGGREGATION = "mean"
DEFAULT_SELECTOR_CONTROL_WEIGHT = 1.0
_QWEN_IM_START = "<|im_start|>"
_QWEN_IM_END = "<|im_end|>"
_QWEN_EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"
_ROUTE_MARGIN_PROTOCOL = "route_margin_alignment"
_ROUTE_MARGIN_PROTOCOL_VERSION = 5


class CompositePatch:
    def __init__(self, patches: Sequence[object]) -> None:
        self.patches = tuple(patch for patch in patches if patch is not None)

    def patch_model(self, model):
        stack = contextlib.ExitStack()
        for patch in self.patches:
            stack.enter_context(patch.patch_model(model))
        return stack


def rows_to_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_text = str(key)
            if key_text not in seen:
                seen.add(key_text)
                fieldnames.append(key_text)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def resolve_model_device(model: torch.nn.Module) -> torch.device:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        if embeddings is not None:
            for parameter in embeddings.parameters():
                if parameter.device.type != "meta":
                    return parameter.device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass
    random.seed(int(seed))


def _clamp_selector_route_blend_alpha(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return min(max(parsed, 0.0), 1.0)


def _format_messages(tokenizer, messages: Sequence[Mapping[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
    parts: List[str] = []
    for message in messages:
        parts.append(f"[{str(message['role']).upper()}]\n{str(message['content'])}")
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


def build_message_batch(
    tokenizer,
    messages: Sequence[Mapping[str, str]],
    device: torch.device,
) -> tuple[str, Dict[str, torch.Tensor]]:
    formatted = _format_messages(tokenizer, messages)
    batch = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return formatted, {key: value.to(device) for key, value in batch.items()}


def generate_from_messages(
    model,
    tokenizer,
    *,
    messages: Sequence[Mapping[str, str]],
    max_new_tokens: int,
    seed: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    patch=None,
) -> tuple[str, str]:
    device = resolve_model_device(model)
    formatted_prompt, batch = build_message_batch(tokenizer, messages, device)
    generate_kwargs = dict(
        **batch,
        use_cache=True,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        repetition_penalty=float(repetition_penalty),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["top_p"] = float(top_p)
    _set_seed(seed)
    with torch.no_grad():
        with patch.patch_model(model) if patch is not None else contextlib.nullcontext():
            output_ids = model.generate(**generate_kwargs)
    prompt_len = int(batch["input_ids"].shape[1])
    new_tokens = output_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), formatted_prompt


def supports_incremental_qwen_chat(model, tokenizer) -> bool:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "") or "").strip().lower()
    return model_type == "qwen3_moe" and getattr(tokenizer, "eos_token_id", None) is not None


def _format_incremental_qwen_message(role: str, content: str, *, assistant_generation_prompt: bool = False) -> str:
    role_name = str(role).strip().lower()
    text = str(content)
    if assistant_generation_prompt:
        if role_name != "assistant":
            raise ValueError("assistant_generation_prompt requires role='assistant'.")
        return f"{_QWEN_IM_START}assistant\n{_QWEN_EMPTY_THINK_BLOCK}"
    if role_name == "assistant":
        return f"{_QWEN_IM_START}assistant\n{_QWEN_EMPTY_THINK_BLOCK}{text}{_QWEN_IM_END}\n"
    if role_name not in {"system", "user"}:
        raise ValueError(f"Unsupported chat role '{role}'.")
    return f"{_QWEN_IM_START}{role_name}\n{text}{_QWEN_IM_END}\n"


def _encode_incremental_qwen_text(tokenizer, text: str) -> List[int]:
    return [int(token_id) for token_id in tokenizer(str(text), add_special_tokens=False)["input_ids"]]


def encode_incremental_qwen_initial_messages(
    tokenizer,
    messages: Sequence[Mapping[str, str]],
) -> List[int]:
    token_ids: List[int] = []
    for message in messages:
        token_ids.extend(
            _encode_incremental_qwen_text(
                tokenizer,
                _format_incremental_qwen_message(str(message["role"]), str(message["content"])),
            )
        )
    return token_ids


def encode_incremental_qwen_messages(
    tokenizer,
    messages: Sequence[Mapping[str, str]],
) -> List[int]:
    token_ids: List[int] = []
    for message in messages:
        token_ids.extend(
            _encode_incremental_qwen_text(
                tokenizer,
                _format_incremental_qwen_message(str(message["role"]), str(message["content"])),
            )
        )
    return token_ids


def encode_incremental_qwen_user_turn(
    tokenizer,
    user_message: str,
) -> List[int]:
    return _encode_incremental_qwen_text(
        tokenizer,
        _format_incremental_qwen_message("user", str(user_message))
        + _format_incremental_qwen_message("assistant", "", assistant_generation_prompt=True),
    )


def estimate_incremental_qwen_dialogue_tokens(
    tokenizer,
    *,
    initial_messages: Sequence[Mapping[str, str]],
    user_turns: Sequence[str],
    turn_prefix_messages: Sequence[Sequence[Mapping[str, str]]] | None = None,
    max_new_tokens: int,
    reserve_for_forced_close: bool = True,
) -> int:
    total = len(encode_incremental_qwen_initial_messages(tokenizer, initial_messages))
    close_reserve = 1 if reserve_for_forced_close else 0
    for turn_index, user_turn in enumerate(user_turns):
        if turn_prefix_messages is not None and turn_index < len(turn_prefix_messages):
            total += len(encode_incremental_qwen_messages(tokenizer, turn_prefix_messages[turn_index]))
        total += len(encode_incremental_qwen_user_turn(tokenizer, str(user_turn)))
        total += int(max_new_tokens) + close_reserve
    return int(total)


def _apply_repetition_penalty(logits: torch.Tensor, token_history: Sequence[int], penalty: float) -> torch.Tensor:
    if abs(float(penalty) - 1.0) < 1e-8 or not token_history:
        return logits
    unique_ids = sorted({int(token_id) for token_id in token_history})
    if not unique_ids:
        return logits
    indices = torch.tensor(unique_ids, device=logits.device, dtype=torch.long)
    penalized = logits.clone()
    gathered = penalized.index_select(-1, indices)
    adjusted = torch.where(gathered < 0, gathered * float(penalty), gathered / float(penalty))
    penalized.scatter_(1, indices.unsqueeze(0), adjusted)
    return penalized


def _sample_next_token(
    logits: torch.Tensor,
    *,
    token_history: Sequence[int],
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    scores = _apply_repetition_penalty(logits.float(), token_history, float(repetition_penalty))
    if not do_sample or float(temperature) <= 0.0:
        return torch.argmax(scores, dim=-1)

    scores = scores / max(float(temperature), 1e-5)
    if 0.0 < float(top_p) < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative > float(top_p)
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        filtered_scores = sorted_scores.masked_fill(remove_mask, float("-inf"))
        filtered_probs = torch.softmax(filtered_scores, dim=-1)
        sampled = torch.multinomial(filtered_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled).squeeze(-1)

    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def generate_incremental_qwen_dialogue(
    model,
    tokenizer,
    *,
    initial_messages: Sequence[Mapping[str, str]],
    user_turns: Sequence[str],
    turn_prefix_messages: Sequence[Sequence[Mapping[str, str]]] | None = None,
    max_new_tokens: int,
    seed: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    patch=None,
) -> List[str]:
    if not supports_incremental_qwen_chat(model, tokenizer):
        raise RuntimeError("Incremental Qwen dialogue generation is only supported for qwen3_moe models.")

    device = resolve_model_device(model)
    eos_token_id = int(tokenizer.eos_token_id)
    initial_ids = encode_incremental_qwen_initial_messages(tokenizer, initial_messages)
    conversation_history: List[int] = list(initial_ids)
    responses: List[str] = []
    _set_seed(seed)

    with torch.no_grad():
        with patch.patch_model(model) if patch is not None else contextlib.nullcontext():
            past_key_values = None
            if initial_ids:
                initial_tensor = torch.tensor([initial_ids], dtype=torch.long, device=device)
                initial_mask = torch.ones((1, len(initial_ids)), dtype=torch.long, device=device)
                outputs = model(
                    input_ids=initial_tensor,
                    attention_mask=initial_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values

            for turn_index, user_turn in enumerate(user_turns):
                prefix_messages = ()
                if turn_prefix_messages is not None and turn_index < len(turn_prefix_messages):
                    prefix_messages = turn_prefix_messages[turn_index]
                prefix_ids = encode_incremental_qwen_messages(tokenizer, prefix_messages)
                if prefix_ids:
                    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
                    prefix_mask = torch.ones((1, len(conversation_history) + len(prefix_ids)), dtype=torch.long, device=device)
                    outputs = model(
                        input_ids=prefix_tensor,
                        attention_mask=prefix_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = outputs.past_key_values
                    conversation_history.extend(prefix_ids)

                turn_prompt_ids = encode_incremental_qwen_user_turn(tokenizer, str(user_turn))
                if not turn_prompt_ids:
                    raise RuntimeError("Incremental Qwen dialogue runner produced an empty turn prompt.")
                turn_prompt_tensor = torch.tensor([turn_prompt_ids], dtype=torch.long, device=device)
                prompt_mask = torch.ones((1, len(conversation_history) + len(turn_prompt_ids)), dtype=torch.long, device=device)
                outputs = model(
                    input_ids=turn_prompt_tensor,
                    attention_mask=prompt_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                conversation_history.extend(turn_prompt_ids)

                logits = outputs.logits[:, -1, :]
                generated_ids: List[int] = []
                closed = False
                for _ in range(int(max_new_tokens)):
                    next_token = _sample_next_token(
                        logits,
                        token_history=conversation_history,
                        do_sample=bool(do_sample),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        repetition_penalty=float(repetition_penalty),
                    )
                    next_id = int(next_token.item())
                    step_tensor = next_token.view(1, 1).to(device=device, dtype=torch.long)
                    step_mask = torch.ones((1, len(conversation_history) + 1), dtype=torch.long, device=device)
                    outputs = model(
                        input_ids=step_tensor,
                        attention_mask=step_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = outputs.past_key_values
                    conversation_history.append(next_id)
                    generated_ids.append(next_id)
                    if next_id == eos_token_id:
                        closed = True
                        break
                    logits = outputs.logits[:, -1, :]

                if not closed:
                    eos_tensor = torch.tensor([[eos_token_id]], dtype=torch.long, device=device)
                    eos_mask = torch.ones((1, len(conversation_history) + 1), dtype=torch.long, device=device)
                    outputs = model(
                        input_ids=eos_tensor,
                        attention_mask=eos_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = outputs.past_key_values
                    conversation_history.append(eos_token_id)
                    generated_ids.append(eos_token_id)

                responses.append(tokenizer.decode(generated_ids, skip_special_tokens=True))

    return responses


def parse_layer_head_map(entries: Sequence[str]) -> Dict[int, Tuple[int, ...]]:
    out: Dict[int, Tuple[int, ...]] = {}
    for entry in entries:
        text = str(entry).strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"Expected layer:head,head entry, got '{entry}'.")
        layer_text, head_text = text.split(":", 1)
        layer_id = int(layer_text.strip())
        head_ids = tuple(int(value.strip()) for value in head_text.split(",") if str(value).strip())
        out[layer_id] = head_ids
    return out


def candidate_layer_head_map(
    *,
    model,
    layer_ids: Sequence[int],
    head_ids: Optional[Sequence[int]] = None,
    layer_head_map: Optional[Mapping[int, Sequence[int]]] = None,
) -> Dict[int, Tuple[int, ...]]:
    manual_map = {
        int(layer_id): tuple(int(head_id) for head_id in head_list)
        for layer_id, head_list in dict(layer_head_map or {}).items()
    }
    num_heads = int(getattr(model.config, "num_attention_heads", 0) or 0)
    default_heads = tuple(int(head_id) for head_id in (head_ids or tuple(range(num_heads))))
    out: Dict[int, Tuple[int, ...]] = {}
    for layer_id in layer_ids:
        if int(layer_id) in manual_map:
            out[int(layer_id)] = manual_map[int(layer_id)]
        else:
            out[int(layer_id)] = default_heads
    return out


def _find_subsequence_start(haystack: Sequence[int], needle: Sequence[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    last_start = len(haystack) - len(needle)
    for start in range(last_start, -1, -1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return int(start)
    return None


def resolve_user_token_positions(
    *,
    tokenizer,
    formatted_prompt: str,
    user_text: str,
    input_ids: torch.Tensor,
) -> List[int]:
    token_positions: List[int] = []
    try:
        with_offsets = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offset_mapping = with_offsets.get("offset_mapping")
        char_start = formatted_prompt.rfind(str(user_text))
        if offset_mapping is not None and char_start >= 0:
            char_end = char_start + len(str(user_text))
            offsets = offset_mapping[0].tolist()
            token_positions = [
                int(index)
                for index, (start, end) in enumerate(offsets)
                if int(end) > int(char_start) and int(start) < int(char_end)
            ]
    except Exception:
        token_positions = []

    if token_positions:
        return token_positions

    try:
        prompt_ids = tokenizer(str(user_text), return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        full_ids = input_ids[0].tolist()
        start = _find_subsequence_start(full_ids, prompt_ids)
        if start is not None:
            return [int(index) for index in range(int(start), int(start) + len(prompt_ids))]
    except Exception:
        pass
    return []


def select_query_token_positions(
    *,
    user_token_positions: Sequence[int],
    total_token_count: int,
    mode: str,
) -> List[int]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "last_token":
        return [max(0, int(total_token_count) - 1)]
    positions = [int(position) for position in user_token_positions]
    if not positions:
        return [max(0, int(total_token_count) - 1)]
    if normalized_mode == "last_user_token":
        return [int(positions[-1])]
    if normalized_mode == "mean_last_4_user_tokens":
        return [int(position) for position in positions[-4:]]
    if normalized_mode == "full_user_span":
        return positions
    raise ValueError(f"Unsupported selector_query_position '{mode}'.")


def _aggregate_phrase_values(values: Iterable[float], mode: str) -> float:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return 0.0
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "mean":
        return float(sum(numeric_values) / len(numeric_values))
    if normalized_mode in {"max", "top1"}:
        return float(max(numeric_values))
    if normalized_mode.startswith("top"):
        try:
            k = max(1, int(normalized_mode[3:]))
        except ValueError as exc:
            raise ValueError(f"Unsupported aggregation mode '{mode}'.") from exc
        top_values = sorted(numeric_values, reverse=True)[:k]
        return float(sum(top_values) / len(top_values))
    raise ValueError(f"Unsupported aggregation mode '{mode}'.")


def _kv_group_size_from_model(model) -> int:
    config = getattr(model, "config", None)
    num_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    num_kv_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
    if num_heads > 0 and num_kv_heads > 0:
        return max(1, int(num_heads) // int(num_kv_heads))
    return 1


def _selector_trace_rows_from_patch(
    *,
    patch: CompositePatch,
    prompt_case,
    prompt_index: int,
) -> List[Dict[str, Any]]:
    trace_rows: List[Dict[str, Any]] = []
    for layer_patch in getattr(patch, "patches", ()):
        diagnostic_traces = list(getattr(layer_patch, "diagnostic_traces", ()) or ())
        for trace in diagnostic_traces:
            selected_head_ids = [int(head_id) for head_id in (trace.get("selected_head_ids") or ())]
            per_head_prompt = {
                str(head_id): float(value)
                for head_id, value in dict(trace.get("per_head_prompt_bank_mass") or {}).items()
            }
            per_head_trait = {
                str(head_id): float(value)
                for head_id, value in dict(trace.get("per_head_trait_bank_mass") or {}).items()
            }
            per_head_reference = {
                str(head_id): float(value)
                for head_id, value in dict(trace.get("per_head_reference_bank_mass") or {}).items()
            }
            per_head_positive_gate = {
                str(head_id): float(value)
                for head_id, value in dict(trace.get("per_head_positive_gate") or {}).items()
            }
            per_head_negative_gate = {
                str(head_id): float(value)
                for head_id, value in dict(trace.get("per_head_negative_gate") or {}).items()
            }
            per_head_alignment = {
                str(head_id): float(value)
                for head_id, value in dict(trace.get("per_head_max_alignment_margin") or {}).items()
            }
            per_head_auxiliary = {
                str(bank_name): {
                    str(head_id): float(value)
                    for head_id, value in dict(bank_values or {}).items()
                }
                for bank_name, bank_values in dict(trace.get("per_head_auxiliary_bank_masses") or {}).items()
            }
            for head_id in selected_head_ids:
                head_key = str(int(head_id))
                total_auxiliary_bank_mass = float(
                    sum(
                        float(bank_values.get(head_key, 0.0))
                        for bank_values in per_head_auxiliary.values()
                    )
                )
                alignment_margin = float(per_head_alignment.get(head_key, 0.0))
                trace_rows.append(
                    {
                        "prompt_id": str(getattr(prompt_case, "prompt_id", prompt_index)),
                        "prompt_index": int(prompt_index),
                        "bucket": str(getattr(prompt_case, "bucket", "")),
                        "prompt": str(getattr(prompt_case, "prompt", "")),
                        "layer_id": int(trace["layer_id"]),
                        "head_id": int(head_id),
                        "decode_step": int(trace.get("decode_step", 0)),
                        "trace_kind": str(trace.get("trace_kind", "decode")),
                        "query_len": int(trace.get("query_len", 1)),
                        "prompt_token_count": int(trace.get("prompt_token_count", 0) or 0),
                        "trait_slot_count": int(trace.get("trait_slot_count", 0) or 0),
                        "reference_slot_count": int(trace.get("reference_slot_count", 0) or 0),
                        "prompt_bank_mass": float(per_head_prompt.get(head_key, 0.0)),
                        "trait_bank_mass": float(per_head_trait.get(head_key, 0.0)),
                        "reference_bank_mass": float(per_head_reference.get(head_key, 0.0)),
                        "total_auxiliary_bank_mass": float(total_auxiliary_bank_mass),
                        "alignment_margin": float(alignment_margin),
                        "positive_gate": float(per_head_positive_gate.get(head_key, 0.0)),
                        "negative_gate": float(per_head_negative_gate.get(head_key, 0.0)),
                        "signal": float(alignment_margin),
                    }
                )
    return trace_rows


def build_style_prompt_message(
    *,
    trait: StyleTraitSpec,
    target_pole: str,
    prompt_role: str = "system",
    wrapper_template: str | None = None,
    bank_profile: str = "default",
    bank_profile_file: str | Path | None = None,
) -> Dict[str, str]:
    bank = resolve_style_track_memory_bank(
        trait,
        track="generation",
        pole=target_pole,
        bank_profile=bank_profile,
        bank_profile_file=bank_profile_file,
    )
    chosen_template = str(wrapper_template or bank.wrapper_templates[0] or DEFAULT_GENERATION_WRAPPER_TEMPLATES[0]).strip()
    return {"role": str(prompt_role), "content": chosen_template.format(descriptor=bank.descriptor)}


def run_style_phrase_selector(
    *,
    model,
    tokenizer,
    trait: StyleTraitSpec,
    prompt_cases: Sequence[object],
    target_pole: str = "high",
    layer_ids: Sequence[int],
    head_ids: Optional[Sequence[int]],
    layer_head_map: Optional[Mapping[int, Sequence[int]]],
    max_heads_per_layer: int,
    max_layers: int,
    layer_selection_metric: str,
    rho: float,
    layer_weight_schedule: str,
    query_position: str = DEFAULT_SELECTOR_QUERY_POSITION,
    target_aggregation: str = DEFAULT_SELECTOR_TARGET_AGGREGATION,
    control_aggregation: str = DEFAULT_SELECTOR_CONTROL_AGGREGATION,
    control_weight: float = DEFAULT_SELECTOR_CONTROL_WEIGHT,
    max_prompt_tokens: int = 0,
    seed: int = 11,
    max_new_tokens: int = 96,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    bank_profile: str = "default",
    bank_profile_file: str | Path | None = None,
    wrapper_templates: Optional[Sequence[str]] = None,
    memory_role: str = "system",
    use_chat_template: bool = True,
    keep_descriptor_only: bool = True,
    mix_mode: str = "contrastive_neutral",
    prefill_steering: str = "last_token_only",
    steering_operator: str = "bank_softmax_mixture",
    positive_gain: float = 4.0,
    negative_gain: float = 0.1,
    query_adaptive_gates: bool = True,
    query_gate_scale: float = 1.0,
    prompt_bank_normalization: str = "log_token_count",
    diagnostic_max_decode_steps: int = 24,
    selector_route_blend_alpha: float = 0.0,
) -> Dict[str, object]:
    device = resolve_model_device(model)
    target_pole_key = str(target_pole).strip().lower()
    if target_pole_key not in {"high", "low"}:
        raise ValueError(f"Unsupported target_pole '{target_pole}'. Expected 'high' or 'low'.")
    base_map = candidate_layer_head_map(
        model=model,
        layer_ids=layer_ids,
        head_ids=head_ids,
        layer_head_map=layer_head_map,
    )

    prompt_rows = list(prompt_cases)
    if not prompt_rows:
        raise RuntimeError("No selector prompt cases were provided.")

    active_bank = resolve_style_track_memory_bank(
        trait,
        track="generation",
        pole=target_pole_key,
        bank_profile=bank_profile,
        bank_profile_file=bank_profile_file,
    )
    reference_pole = "low" if target_pole_key == "high" else "high"
    reference_bank = resolve_style_track_memory_bank(
        trait,
        track="generation",
        pole=reference_pole,
        bank_profile=bank_profile,
        bank_profile_file=bank_profile_file,
    )
    selected_wrapper_templates = tuple(
        str(template)
        for template in (wrapper_templates or active_bank.wrapper_templates or DEFAULT_GENERATION_WRAPPER_TEMPLATES)
        if str(template).strip()
    )
    if not selected_wrapper_templates:
        selected_wrapper_templates = tuple(DEFAULT_GENERATION_WRAPPER_TEMPLATES)
    selected_wrapper_template = str(selected_wrapper_templates[0])
    candidate_layer_ids = [int(value) for value in layer_ids]
    route_blend_alpha = _clamp_selector_route_blend_alpha(selector_route_blend_alpha)
    selector_mode = "style_generation_route_blend_v1" if float(route_blend_alpha) > 0.0 else "style_generation_route_margin_v1"
    selector_memory = build_canonical_descriptor_memory(
        model,
        tokenizer,
        trait_name=f"{trait.name}_style_selector_generation_{target_pole_key}",
        descriptor=active_bank.descriptor,
        descriptor_variants=active_bank.descriptor_variants,
        layer_ids=candidate_layer_ids,
        wrapper_template=selected_wrapper_template,
        wrapper_templates=selected_wrapper_templates,
        use_chat_template=bool(use_chat_template),
        chat_role=str(memory_role),
        keep_descriptor_only=bool(keep_descriptor_only),
    )
    selector_reference_memory = build_canonical_descriptor_memory(
        model,
        tokenizer,
        trait_name=f"{trait.name}_style_selector_generation_{reference_pole}",
        descriptor=reference_bank.descriptor,
        descriptor_variants=reference_bank.descriptor_variants,
        layer_ids=candidate_layer_ids,
        wrapper_template=selected_wrapper_template,
        wrapper_templates=selected_wrapper_templates,
        use_chat_template=bool(use_chat_template),
        chat_role=str(memory_role),
        keep_descriptor_only=bool(keep_descriptor_only),
    )
    layer_rho_map = layer_rhos(candidate_layer_ids, float(rho), str(layer_weight_schedule))
    diagnostic_steps = max(1, int(diagnostic_max_decode_steps))
    kv_group_size = _kv_group_size_from_model(model)

    full_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for prompt_index, case in enumerate(prompt_rows):
            prompt_text = str(getattr(case, "prompt"))
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt, batch = build_message_batch(tokenizer, messages, device)
            prompt_token_count_before_trim = int(batch["input_ids"].shape[1])
            trimmed = False
            if max_prompt_tokens > 0 and int(batch["input_ids"].shape[1]) > max_prompt_tokens:
                trim_start = max(0, int(batch["input_ids"].shape[1]) - int(max_prompt_tokens))
                batch["input_ids"] = batch["input_ids"][:, trim_start:]
                if "attention_mask" in batch:
                    batch["attention_mask"] = batch["attention_mask"][:, trim_start:]
                trimmed = True

            patches = [
                CanonicalDescriptorMemoryAttentionPatch(
                    layer_ids=[int(layer_id)],
                    memory=selector_memory,
                    reference_memory=selector_reference_memory,
                    config=CanonicalDescriptorMixConfig(
                        rho=float(layer_rho_map[int(layer_id)]),
                        mix_mode=str(mix_mode),
                        prefill_steering=str(prefill_steering),
                        steering_operator=str(steering_operator),
                        positive_gain=float(positive_gain),
                        negative_gain=float(negative_gain),
                        query_adaptive_gates=bool(query_adaptive_gates),
                        query_gate_scale=float(query_gate_scale),
                        prompt_bank_normalization=str(prompt_bank_normalization),
                        diagnostic_max_decode_steps=int(diagnostic_steps),
                        record_prefill_diagnostics=False,
                        head_ids=tuple(int(head_id) for head_id in base_map.get(int(layer_id), ())),
                    ),
                )
                for layer_id in candidate_layer_ids
                if base_map.get(int(layer_id))
            ]
            patch = CompositePatch(patches)
            generate_kwargs = dict(
                **batch,
                use_cache=True,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                repetition_penalty=float(repetition_penalty),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if do_sample:
                generate_kwargs["temperature"] = float(temperature)
                generate_kwargs["top_p"] = float(top_p)
            _set_seed(int(seed) + int(prompt_index))
            with patch.patch_model(model):
                output_ids = model.generate(**generate_kwargs)
            prompt_len = int(batch["input_ids"].shape[1])
            new_tokens = output_ids[0, prompt_len:]
            completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            prompt_trace_rows = _selector_trace_rows_from_patch(
                patch=patch,
                prompt_case=case,
                prompt_index=prompt_index,
            )
            trace_rows.extend(prompt_trace_rows)
            full_rows.append(
                {
                    "prompt_id": str(getattr(case, "prompt_id", prompt_index)),
                    "prompt_index": int(prompt_index),
                    "prompt": prompt_text,
                    "bucket": str(getattr(case, "bucket", "")),
                    "topic": str(getattr(case, "topic", "")),
                    "objective": str(getattr(case, "objective", "")),
                    "formatted_prompt": formatted_prompt,
                    "prompt_token_count_before_trim": int(prompt_token_count_before_trim),
                    "prompt_token_count": int(batch["input_ids"].shape[1]),
                    "trimmed_to_max_prompt_tokens": bool(trimmed),
                    "generated_text": str(completion_text),
                    "generated_token_count": int(new_tokens.shape[0]),
                    "trace_row_count": int(len(prompt_trace_rows)),
                }
            )

    if not trace_rows:
        raise RuntimeError("No trace rows were collected for style head selection.")
    head_rows = _build_route_margin_alignment_head_scores(
        trace_rows=trace_rows,
        selector_mode=selector_mode,
        protocol=_ROUTE_MARGIN_PROTOCOL,
        route_blend_alpha=float(route_blend_alpha),
    )
    layer_rows, candidate_head_map, selected_map, ranked_layer_ids_by_score, selected_layer_ids_by_score, selected_kv_group_map = _select_layers_from_head_scores(
        head_rows=head_rows,
        candidate_layer_ids=candidate_layer_ids,
        max_heads_per_layer=int(max_heads_per_layer),
        max_layers=int(max_layers),
        layer_selection_metric=str(layer_selection_metric),
        kv_group_size=int(kv_group_size),
        allow_kv_group_backfill=False,
    )
    if not selected_map:
        raise RuntimeError(
            f"No positive route-consistent selector heads were found for trait '{trait.name}' "
            f"and pole '{target_pole_key}'."
        )
    selected_layer_ids = [int(layer_id) for layer_id in sorted(selected_map)]
    selected_layer_rho_map = (
        layer_rhos(selected_layer_ids, float(rho), str(layer_weight_schedule))
        if selected_layer_ids
        else {}
    )

    artifact = {
        "target_trait": str(trait.name),
        "target_pole": str(target_pole_key),
        "selector_mode": str(selector_mode),
        "ranked_layer_ids_by_score": [int(layer_id) for layer_id in ranked_layer_ids_by_score],
        "selected_layer_ids_by_score": [int(layer_id) for layer_id in selected_layer_ids_by_score],
        "selected_layer_ids": [int(layer_id) for layer_id in selected_layer_ids],
        "selected_layer_head_map": {str(layer): list(heads) for layer, heads in selected_map.items()},
        "selected_layer_kv_group_map": {str(layer): list(groups) for layer, groups in selected_kv_group_map.items()},
        "selected_layer_rho_map": {str(layer): float(value) for layer, value in selected_layer_rho_map.items()},
        "candidate_top_head_map": {str(layer): list(heads) for layer, heads in candidate_head_map.items()},
        "selection_rule": {
            "formula": (
                "rank heads by mean(max(alpha * route_margin + (1 - alpha) * non_trait_route_margin, 0) "
                "* max(alignment_margin, 0)); require score>0, "
                "non_trait_route_positive_fraction>=0.5, q25_non_trait_route_margin>0, "
                "then keep at most one head per KV group"
                if float(route_blend_alpha) > 0.0
                else
                "rank heads by mean(max(log((trait_bank_mass + eps) / "
                "(prompt_bank_mass + reference_bank_mass + total_auxiliary_bank_mass + eps)), 0) "
                "* max(alignment_margin, 0)); require score>0, "
                "non_trait_route_positive_fraction>=0.5, q25_non_trait_route_margin>0, "
                "then keep at most one head per KV group"
            ),
            "max_heads_per_layer": int(max_heads_per_layer),
            "head_positive_score_preferred": True,
            "kv_group_dedup": bool(int(kv_group_size) > 1),
            "kv_group_size": int(kv_group_size),
            "kv_group_dedup_strict": True,
            "min_non_trait_route_positive_fraction": 0.5,
            "min_q25_non_trait_route_margin": 0.0,
            "protocol": _ROUTE_MARGIN_PROTOCOL,
            "protocol_version": int(_ROUTE_MARGIN_PROTOCOL_VERSION),
            "score_field": "score",
            "route_blend_alpha": float(route_blend_alpha),
        },
        "layer_selection_rule": {
            "max_layers": int(max_layers),
            "layer_selection_metric": str(layer_selection_metric),
            "layer_weight_schedule": str(layer_weight_schedule),
        },
        "selector_postprocess": {
            "protocol": _ROUTE_MARGIN_PROTOCOL,
            "version": int(_ROUTE_MARGIN_PROTOCOL_VERSION),
            "score_field": "score",
            "score_formula": (
                "mean(max(alpha * route_margin + (1 - alpha) * non_trait_route_margin, 0) * max(alignment_margin, 0))"
                if float(route_blend_alpha) > 0.0
                else
                "mean(max(non_trait_route_margin, 0) * max(alignment_margin, 0))"
            ),
            "route_margin_formula": "log((trait_bank_mass + eps) / (prompt_bank_mass + eps))",
            "non_trait_route_margin_formula": "log((trait_bank_mass + eps) / (prompt_bank_mass + reference_bank_mass + total_auxiliary_bank_mass + eps))",
            "route_blend_alpha": float(route_blend_alpha),
            "route_blend_formula": "alpha * route_margin + (1 - alpha) * non_trait_route_margin",
            "non_trait_mass_definition": "prompt_bank_mass + reference_bank_mass + total_auxiliary_bank_mass",
            "eligibility_filters": {
                "score_positive_required": True,
                "min_non_trait_route_positive_fraction": 0.5,
                "min_q25_non_trait_route_margin": 0.0,
            },
            "kv_group_size": int(kv_group_size),
            "kv_group_dedup_strict": True,
        },
        "config": {
            "rho": float(rho),
            "layer_weight_schedule": str(layer_weight_schedule),
            "mix_mode": str(mix_mode),
            "prefill_steering": str(prefill_steering),
            "steering_operator": str(steering_operator),
            "positive_gain": float(positive_gain),
            "negative_gain": float(negative_gain),
            "query_gate_scale": float(query_gate_scale),
            "prompt_bank_normalization": str(prompt_bank_normalization),
            "selector_max_new_tokens": int(max_new_tokens),
            "selector_diagnostic_max_decode_steps": int(diagnostic_steps),
            "track_memory_bank_profile": str(bank_profile),
            "track_memory_bank_profile_file": None if bank_profile_file in (None, "") else str(bank_profile_file),
            "memory_role": str(memory_role),
            "disable_memory_chat_template": not bool(use_chat_template),
            "disable_memory_descriptor_only": not bool(keep_descriptor_only),
            "head_selection_protocol": _ROUTE_MARGIN_PROTOCOL,
            "head_selection_protocol_version": int(_ROUTE_MARGIN_PROTOCOL_VERSION),
            "head_selection_score_field": "score",
            "head_selection_route_blend_alpha": float(route_blend_alpha),
            "head_selection_min_non_trait_route_positive_fraction": 0.5,
            "head_selection_min_q25_non_trait_route_margin": 0.0,
            "candidate_layer_head_map": {str(layer): list(heads) for layer, heads in base_map.items()},
            "requested_selector_mode": str(selector_mode),
            "num_attention_heads": int(getattr(model.config, "num_attention_heads", 0) or 0),
            "num_key_value_heads": int(getattr(model.config, "num_key_value_heads", 0) or 0),
            "kv_group_size": int(kv_group_size),
        },
        "selector_metadata": {
            "prompt_count": int(len(prompt_rows)),
            "prompt_ids": [str(getattr(case, "prompt_id", index)) for index, case in enumerate(prompt_rows)],
            "candidate_layer_head_map": {str(layer): list(heads) for layer, heads in base_map.items()},
            "target_descriptor": str(active_bank.descriptor),
            "reference_descriptor": str(reference_bank.descriptor),
            "wrapper_templates": [str(template) for template in selected_wrapper_templates],
            "qwen_selector_note": "Selector scores heads from decode-time routing into the same generation banks used during steering.",
            "route_blend_alpha": float(route_blend_alpha),
            "legacy_phrase_selector_args_ignored": {
                "query_position": str(query_position),
                "target_aggregation": str(target_aggregation),
                "control_aggregation": str(control_aggregation),
                "control_weight": float(control_weight),
            },
        },
    }
    return {
        "full_rows": full_rows,
        "trace_rows": trace_rows,
        "head_rows": head_rows,
        "layer_rows": layer_rows,
        "artifact": artifact,
    }


__all__ = [
    "CompositePatch",
    "DEFAULT_SELECTOR_CONTROL_AGGREGATION",
    "DEFAULT_SELECTOR_CONTROL_WEIGHT",
    "DEFAULT_SELECTOR_LAYER_IDS",
    "DEFAULT_SELECTOR_QUERY_POSITION",
    "DEFAULT_SELECTOR_TARGET_AGGREGATION",
    "CanonicalDescriptorMemoryAttentionPatch",
    "CanonicalDescriptorMixConfig",
    "build_canonical_descriptor_memory",
    "build_message_batch",
    "build_style_prompt_message",
    "candidate_layer_head_map",
    "encode_incremental_qwen_initial_messages",
    "encode_incremental_qwen_user_turn",
    "estimate_incremental_qwen_dialogue_tokens",
    "generate_from_messages",
    "generate_incremental_qwen_dialogue",
    "layer_rhos",
    "parse_layer_head_map",
    "resolve_model_device",
    "rows_to_csv",
    "run_style_phrase_selector",
    "supports_incremental_qwen_chat",
]
