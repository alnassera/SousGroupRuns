from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from architecture import assert_qwen3_moe_compatible, inspect_qwen3_moe_architecture, wrap_qwen_tokenizer_apply_chat_template


DEFAULT_HF_CACHE_ROOTS = (
    Path("/nfs/roberts/scratch/pi_amk266/zl664/hf_cache"),
    Path("/nfs/roberts/scratch/pi_jks79/zl664/hf_cache"),
)


def resolve_torch_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None or name == "auto":
        return None
    normalized = str(name).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype '{name}'.")
    return mapping[normalized]


def resolve_device_map(device_map: Optional[str]) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    if device_map is None:
        return None, None
    normalized = str(device_map).strip().lower()
    if not normalized or normalized == "none":
        return None, None
    if normalized == "cuda":
        return {"": "cuda:0"}, "cuda:0"
    if re.fullmatch(r"cuda:\d+", normalized):
        return {"": normalized}, normalized
    if normalized in {"cpu", "mps"}:
        return {"": normalized}, normalized
    return device_map, None


def _iter_hf_cache_roots() -> Tuple[Path, ...]:
    candidates = []
    for env_name in ("TRANSFORMERS_CACHE", "HF_HUB_CACHE", "HF_HOME"):
        raw_value = os.environ.get(env_name)
        if raw_value:
            candidates.append(Path(raw_value).expanduser())
    candidates.extend(DEFAULT_HF_CACHE_ROOTS)

    roots = []
    seen = set()
    for candidate in candidates:
        for resolved in (candidate, candidate / "hub"):
            key = str(resolved)
            if key in seen or not resolved.exists():
                continue
            seen.add(key)
            roots.append(resolved)
    return tuple(roots)


def resolve_hf_cache_dir(model_name: str) -> Tuple[Optional[str], bool]:
    repo_dir_name = f"models--{str(model_name).replace('/', '--')}"
    first_root: Optional[Path] = None
    for cache_root in _iter_hf_cache_roots():
        if first_root is None:
            first_root = cache_root
        if (cache_root / repo_dir_name).exists():
            return str(cache_root), True
    return (str(first_root), False) if first_root is not None else (None, False)


def load_model_and_tokenizer(
    model_name: str,
    *,
    device_map: Optional[str] = "auto",
    torch_dtype: Optional[str] = "bfloat16",
    attn_implementation: Optional[str] = None,
    allow_eager_fallback: bool = True,
    strict_attn_implementation: bool = False,
    enable_thinking: bool = False,
    strict_architecture_check: bool = True,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "transformers is required. Install the packages listed in descriptor_memory_persona/requirements.txt."
        ) from exc

    dtype = resolve_torch_dtype(torch_dtype)
    cache_dir, local_files_only = resolve_hf_cache_dir(model_name)
    common_load_kwargs = {}
    if cache_dir is not None:
        common_load_kwargs["cache_dir"] = cache_dir
        print(f"[qwen3_moe.modeling] cache_dir={cache_dir}", flush=True)
    if local_files_only:
        common_load_kwargs["local_files_only"] = True
        print(f"[qwen3_moe.modeling] local_files_only=True for model={model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, **common_load_kwargs)
    wrap_qwen_tokenizer_apply_chat_template(tokenizer, enable_thinking=enable_thinking)

    resolved_device_map, forced_device = resolve_device_map(device_map)
    model_kwargs = dict(common_load_kwargs)
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    if attn_implementation is None or str(attn_implementation).strip().lower() == "auto":
        load_attempts = [("sdpa", True), ("flash_attention_2", True), ("eager", True), (None, False)]
    else:
        normalized_impl = str(attn_implementation).strip().lower()
        load_attempts = [(normalized_impl, True)] if strict_attn_implementation else [(normalized_impl, True), (None, False)]

    model = None
    selected_attn_impl = None
    last_error = None
    for attn_impl, set_flag in load_attempts:
        if attn_impl == "eager" and not allow_eager_fallback and (
            attn_implementation is None or str(attn_implementation).strip().lower() == "auto"
        ):
            continue
        trial_kwargs = dict(model_kwargs)
        if set_flag:
            trial_kwargs["attn_implementation"] = attn_impl
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **trial_kwargs)
            selected_attn_impl = attn_impl if set_flag else getattr(getattr(model, "config", None), "_attn_implementation", None)
            break
        except TypeError as exc:
            last_error = exc
            if set_flag:
                continue
        except Exception as exc:  # pragma: no cover - backend dependent
            last_error = exc
            continue

    if model is None:
        raise last_error if last_error is not None else RuntimeError("Failed to load the requested Qwen3 MoE checkpoint.")

    if forced_device is not None and not any(parameter.device.type == "meta" for parameter in model.parameters()):
        model = model.to(forced_device)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    report = inspect_qwen3_moe_architecture(model)
    print("[qwen3_moe.modeling] architecture:", report, flush=True)
    if strict_architecture_check:
        assert_qwen3_moe_compatible(model)
    if selected_attn_impl is not None:
        print(f"[qwen3_moe.modeling] loaded attention backend: {selected_attn_impl}", flush=True)
    return model, tokenizer
