from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch


SITE_CHOICES = ("layer_input", "post_attn_resid", "mlp_input")
TOKEN_SELECTOR_CHOICES = ("last_completion_token", "first_completion_token", "mean_completion")


@dataclass
class SiteActivationSteeringVector:
    trait_name: str
    descriptor: str
    opposite_descriptor: str
    site: str
    token_selector: str
    layer_vectors: Dict[int, torch.Tensor]


class _SiteCaptureContext:
    def __init__(self, model: torch.nn.Module, *, layer_ids: Sequence[int], site: str) -> None:
        self.model = model
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        self.site = str(site).strip().lower()
        if self.site not in SITE_CHOICES:
            raise ValueError(f"Unsupported CAA site '{site}'. Expected one of {SITE_CHOICES}.")
        self.captured: Dict[int, torch.Tensor] = {}
        self._hooks: List[object] = []

    def __enter__(self) -> "_SiteCaptureContext":
        model_body = getattr(self.model, "model", self.model)
        model_layers = getattr(model_body, "layers", None)
        if model_layers is None:
            raise AttributeError("Expected model.model.layers to exist.")

        for layer_id in self.layer_ids:
            layer_module = model_layers[int(layer_id)]
            if self.site == "layer_input":
                handle = layer_module.register_forward_pre_hook(self._make_input_pre_hook(int(layer_id)))
            else:
                norm_module = getattr(layer_module, "post_attention_layernorm", None)
                if norm_module is None:
                    raise AttributeError(
                        f"Layer {layer_id} has no post_attention_layernorm; "
                        "post-attention residual and mlp-input capture currently assume a Llama-style decoder."
                    )
                if self.site == "post_attn_resid":
                    handle = norm_module.register_forward_pre_hook(self._make_input_pre_hook(int(layer_id)))
                else:
                    handle = norm_module.register_forward_hook(self._make_output_hook(int(layer_id)))
            self._hooks.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in reversed(self._hooks):
            handle.remove()
        self._hooks.clear()

    def _make_input_pre_hook(self, layer_id: int):
        def hook(_module, args):
            if not args:
                return
            tensor = args[0]
            if isinstance(tensor, torch.Tensor):
                self.captured[int(layer_id)] = tensor.detach()

        return hook

    def _make_output_hook(self, layer_id: int):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                self.captured[int(layer_id)] = tensor.detach()

        return hook


def _resolve_model_device(model: torch.nn.Module) -> torch.device:
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


def _chat_prompt_with_generation(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": str(prompt)}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return str(prompt)


def _completion_positions(tokenizer, *, prompt_text: str, completion_text: str) -> Tuple[int, ...]:
    full_text = prompt_text + completion_text
    try:
        tokenized = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = tokenized.get("offset_mapping")
        if offsets:
            boundary = len(prompt_text)
            selected: List[int] = []
            for idx, offset in enumerate(offsets):
                if offset is None or len(offset) != 2:
                    continue
                start, end = int(offset[0]), int(offset[1])
                if end > boundary and end > start:
                    selected.append(idx)
            if selected:
                return tuple(selected)
    except Exception:
        pass

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    start = len(prompt_ids)
    if start >= len(full_ids):
        return ()
    return tuple(range(start, len(full_ids)))


def _selected_completion_positions(
    tokenizer,
    *,
    prompt_text: str,
    completion_text: str,
    token_selector: str,
) -> Tuple[int, ...]:
    positions = _completion_positions(tokenizer, prompt_text=prompt_text, completion_text=completion_text)
    if not positions:
        return ()
    normalized_selector = str(token_selector).strip().lower()
    if normalized_selector == "mean_completion":
        return positions
    if normalized_selector == "first_completion_token":
        return (positions[0],)
    if normalized_selector == "last_completion_token":
        return (positions[-1],)
    raise ValueError(
        f"Unsupported token_selector '{token_selector}'. Expected one of {TOKEN_SELECTOR_CHOICES}."
    )


def build_site_activation_steering_vector_from_examples(
    model: torch.nn.Module,
    tokenizer,
    *,
    trait_name: str,
    descriptor: str,
    opposite_descriptor: str,
    layer_ids: Sequence[int],
    examples: Sequence[Mapping[str, str]],
    site: str = "post_attn_resid",
    token_selector: str = "last_completion_token",
    device: Optional[torch.device] = None,
    normalize_vector: bool = True,
) -> SiteActivationSteeringVector:
    normalized_site = str(site).strip().lower()
    if normalized_site not in SITE_CHOICES:
        raise ValueError(f"Unsupported CAA site '{site}'. Expected one of {SITE_CHOICES}.")
    normalized_token_selector = str(token_selector).strip().lower()
    if normalized_token_selector not in TOKEN_SELECTOR_CHOICES:
        raise ValueError(
            f"Unsupported token_selector '{token_selector}'. Expected one of {TOKEN_SELECTOR_CHOICES}."
        )
    model_device = _resolve_model_device(model) if device is None else device

    positive_by_layer: Dict[int, List[torch.Tensor]] = {int(layer_id): [] for layer_id in layer_ids}
    negative_by_layer: Dict[int, List[torch.Tensor]] = {int(layer_id): [] for layer_id in layer_ids}

    for example in examples:
        prompt = str(example.get("prompt", "")).strip()
        positive_response = str(example.get("positive_response", "")).strip()
        negative_response = str(example.get("negative_response", "")).strip()
        if not prompt or not positive_response or not negative_response:
            continue

        prompt_text = _chat_prompt_with_generation(tokenizer, prompt)
        completions = {
            "positive": positive_response,
            "negative": negative_response,
        }
        for label, completion in completions.items():
            full_text = prompt_text + completion
            tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            input_ids = tokens["input_ids"].to(model_device)
            attention_mask = tokens.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_device)

            with _SiteCaptureContext(model, layer_ids=layer_ids, site=normalized_site) as capture_ctx:
                with torch.no_grad():
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )

            positions = _selected_completion_positions(
                tokenizer,
                prompt_text=prompt_text,
                completion_text=completion,
                token_selector=normalized_token_selector,
            )
            if not positions:
                continue
            index_tensor = torch.tensor(positions, device=model_device, dtype=torch.long)

            for layer_id in layer_ids:
                captured = capture_ctx.captured.get(int(layer_id))
                if captured is None:
                    raise RuntimeError(
                        f"Failed to capture site '{normalized_site}' for layer {layer_id} "
                        f"while building activation vector for '{trait_name}'."
                    )
                site_tensor = captured.to(device=model_device)
                summary = site_tensor.index_select(dim=1, index=index_tensor).mean(dim=1).squeeze(0).detach().cpu()
                if label == "positive":
                    positive_by_layer[int(layer_id)].append(summary)
                else:
                    negative_by_layer[int(layer_id)].append(summary)

    layer_vectors: Dict[int, torch.Tensor] = {}
    for layer_id in layer_ids:
        positives = positive_by_layer[int(layer_id)]
        negatives = negative_by_layer[int(layer_id)]
        if not positives or not negatives:
            raise ValueError(
                f"No usable contrastive examples were collected for layer {layer_id} "
                f"at site '{normalized_site}' while building activation vector for '{trait_name}'."
            )
        vector = torch.stack(positives, dim=0).mean(dim=0) - torch.stack(negatives, dim=0).mean(dim=0)
        if normalize_vector:
            vector = vector / vector.norm().clamp_min(1e-6)
        layer_vectors[int(layer_id)] = vector

    return SiteActivationSteeringVector(
        trait_name=trait_name,
        descriptor=descriptor,
        opposite_descriptor=opposite_descriptor,
        site=normalized_site,
        token_selector=normalized_token_selector,
        layer_vectors=layer_vectors,
    )
