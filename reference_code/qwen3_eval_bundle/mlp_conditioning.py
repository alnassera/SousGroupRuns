from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _tokenwise_rms(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return value.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)


@dataclass(frozen=True)
class MemoryConditioningConfig:
    mode: str = "input_add"
    scale: float = 0.0
    input_scale: float = 0.0
    gate_scale: float = 0.0
    normalize_delta: bool = False
    match_input_rms: bool = True
    match_gate_rms: bool = True
    match_output_rms: bool = True
    eps: float = 1e-6


@dataclass
class LayerDeltaRecord:
    delta_hidden: torch.Tensor
    delta_mlp_input: Optional[torch.Tensor] = None


@dataclass
class AttentionToMLPControlState:
    layer_records: Dict[int, LayerDeltaRecord] = field(default_factory=dict)

    def clear_layer(self, layer_id: int) -> None:
        self.layer_records.pop(int(layer_id), None)

    def store(self, layer_id: int, delta_hidden: torch.Tensor) -> None:
        self.layer_records[int(layer_id)] = LayerDeltaRecord(delta_hidden=delta_hidden.detach())

    def store_mlp_input_delta(self, layer_id: int, delta_mlp_input: torch.Tensor) -> None:
        record = self.layer_records.get(int(layer_id))
        if record is None:
            return
        record.delta_mlp_input = delta_mlp_input.detach()

    def pop(self, layer_id: int) -> Optional[LayerDeltaRecord]:
        return self.layer_records.pop(int(layer_id), None)

    def peek(self, layer_id: int) -> Optional[LayerDeltaRecord]:
        return self.layer_records.get(int(layer_id))

    def clear(self) -> None:
        self.layer_records.clear()


@dataclass
class PatchedMLPContext:
    patches: List[Tuple[torch.nn.Module, object]] = field(default_factory=list)
    control_state: Optional[AttentionToMLPControlState] = None

    def __enter__(self) -> "PatchedMLPContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def remove(self) -> None:
        for module, original_forward in reversed(self.patches):
            module.forward = original_forward
        self.patches.clear()
        if self.control_state is not None:
            self.control_state.clear()


class CanonicalMemoryConditionedMLPPatch:
    """Condition selected MLP layers on the hidden delta retrieved by the attention patch."""

    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        control_state: AttentionToMLPControlState,
        layer_scales: Optional[Mapping[int, float]] = None,
        input_layer_scales: Optional[Mapping[int, float]] = None,
        gate_layer_scales: Optional[Mapping[int, float]] = None,
        config: Optional[MemoryConditioningConfig] = None,
    ) -> None:
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        self.control_state = control_state
        config = MemoryConditioningConfig() if config is None else config
        mode = str(config.mode).strip().lower()
        if mode not in {"input_add", "gate_bias", "input_gate", "jvp"}:
            raise ValueError(f"Unsupported MLP conditioning mode '{config.mode}'.")
        self.config = MemoryConditioningConfig(
            mode=mode,
            scale=float(config.scale),
            input_scale=float(config.input_scale),
            gate_scale=float(config.gate_scale),
            normalize_delta=bool(config.normalize_delta),
            match_input_rms=bool(config.match_input_rms),
            match_gate_rms=bool(config.match_gate_rms),
            match_output_rms=bool(config.match_output_rms),
            eps=max(1e-9, float(config.eps)),
        )
        self.layer_scales = {int(layer_id): float(value) for layer_id, value in (layer_scales or {}).items()}
        self.input_layer_scales = {int(layer_id): float(value) for layer_id, value in (input_layer_scales or {}).items()}
        self.gate_layer_scales = {int(layer_id): float(value) for layer_id, value in (gate_layer_scales or {}).items()}

    def patch_model(self, model: torch.nn.Module) -> PatchedMLPContext:
        context = PatchedMLPContext(control_state=self.control_state)
        model_body = getattr(model, "model", model)
        model_layers = getattr(model_body, "layers", None)
        if model_layers is None:
            raise AttributeError("Expected model.model.layers to exist.")

        for layer_id in self.layer_ids:
            layer_module = model_layers[int(layer_id)]
            post_attention_layernorm = getattr(layer_module, "post_attention_layernorm", None)
            if post_attention_layernorm is not None:
                original_ln_forward = post_attention_layernorm.forward
                post_attention_layernorm.forward = self._make_wrapped_layernorm_forward(
                    int(layer_id),
                    original_ln_forward,
                )
                context.patches.append((post_attention_layernorm, original_ln_forward))
            mlp_module = getattr(layer_module, "mlp", None)
            if mlp_module is None:
                raise AttributeError(f"Layer {layer_id} has no mlp module.")
            original_forward = mlp_module.forward
            mlp_module.forward = self._make_wrapped_forward(int(layer_id), mlp_module, original_forward)
            context.patches.append((mlp_module, original_forward))
        return context

    def _resolve_scale(self, layer_id: int) -> float:
        if int(layer_id) in self.layer_scales:
            return float(self.layer_scales[int(layer_id)])
        return float(self.config.scale)

    def _resolve_input_scale(self, layer_id: int) -> float:
        if int(layer_id) in self.input_layer_scales:
            return float(self.input_layer_scales[int(layer_id)])
        if int(layer_id) in self.layer_scales:
            return float(self.layer_scales[int(layer_id)])
        if self.config.input_scale != 0.0:
            return float(self.config.input_scale)
        return float(self.config.scale)

    def _resolve_gate_scale(self, layer_id: int) -> float:
        if int(layer_id) in self.gate_layer_scales:
            return float(self.gate_layer_scales[int(layer_id)])
        if int(layer_id) in self.layer_scales:
            return float(self.layer_scales[int(layer_id)])
        if self.config.gate_scale != 0.0:
            return float(self.config.gate_scale)
        return float(self.config.scale)

    def _resolve_dual_scales(self, layer_id: int) -> tuple[float, float]:
        return self._resolve_input_scale(layer_id), self._resolve_gate_scale(layer_id)

    def _prepare_control(
        self,
        delta: torch.Tensor,
        *,
        reference: torch.Tensor,
        normalize: bool,
        match_rms: bool,
    ) -> torch.Tensor:
        prepared = delta.to(device=reference.device, dtype=reference.dtype)
        prepared_rms = _tokenwise_rms(prepared, eps=self.config.eps)
        reference_rms = _tokenwise_rms(reference, eps=self.config.eps)
        if normalize:
            prepared = prepared / prepared_rms
            if match_rms:
                prepared = prepared * reference_rms
        elif match_rms:
            shrink = torch.minimum(torch.ones_like(prepared_rms), reference_rms / prepared_rms)
            prepared = prepared * shrink
        return prepared

    def _prepare_input_control(self, hidden_states: torch.Tensor, raw_control: torch.Tensor) -> torch.Tensor:
        return self._prepare_control(
            raw_control,
            reference=hidden_states,
            normalize=bool(self.config.normalize_delta),
            match_rms=bool(self.config.match_input_rms),
        )

    def _prepare_gate_control(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        raw_control: torch.Tensor,
    ) -> torch.Tensor:
        gate_base = module.gate_proj(hidden_states)
        gate_delta = module.gate_proj(raw_control.to(device=hidden_states.device, dtype=hidden_states.dtype))
        return self._prepare_control(
            gate_delta,
            reference=gate_base,
            normalize=False,
            match_rms=bool(self.config.match_gate_rms),
        )

    def _prepare_output_correction(
        self,
        base_output: torch.Tensor,
        raw_correction: torch.Tensor,
    ) -> torch.Tensor:
        return self._prepare_control(
            raw_correction,
            reference=base_output,
            normalize=bool(self.config.normalize_delta),
            match_rms=bool(self.config.match_output_rms),
        )

    def _resolve_control_source(
        self,
        record: LayerDeltaRecord,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        control = record.delta_mlp_input if record.delta_mlp_input is not None else record.delta_hidden
        return control.to(device=hidden_states.device, dtype=hidden_states.dtype)

    def _activation_directional_derivative(
        self,
        act_fn,
        preactivation: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(act_fn, torch.nn.SiLU) or act_fn is F.silu:
            sigma = torch.sigmoid(preactivation)
            return direction * (sigma * (1.0 + preactivation * (1.0 - sigma)))
        with torch.enable_grad():
            preactivation_for_grad = preactivation.detach().requires_grad_(True)
            activated = act_fn(preactivation_for_grad)
            derivative = torch.autograd.grad(
                outputs=activated,
                inputs=preactivation_for_grad,
                grad_outputs=direction.to(dtype=preactivation_for_grad.dtype),
                retain_graph=False,
                create_graph=False,
            )[0]
        return derivative.to(dtype=preactivation.dtype)

    def _make_wrapped_layernorm_forward(self, layer_id: int, original_forward: object):
        signature = inspect.signature(original_forward)

        def wrapped_forward(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            hidden_states = bound.arguments.get("x")
            if hidden_states is None:
                hidden_states = bound.arguments.get("hidden_states", args[0] if args else None)
            output = original_forward(*args, **kwargs)
            if hidden_states is None:
                return output

            record = self.control_state.peek(layer_id)
            if record is None:
                return output

            delta_hidden = record.delta_hidden.to(device=hidden_states.device, dtype=hidden_states.dtype)
            base_hidden_states = hidden_states - delta_hidden
            base_output = original_forward(base_hidden_states)
            self.control_state.store_mlp_input_delta(layer_id, output - base_output)
            return output

        return wrapped_forward

    def _gate_bias_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        gate_control: torch.Tensor,
        scale: float,
    ):
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
            raise RuntimeError(
                "Gate-only MLP biasing requires an MLP module with gate_proj/up_proj/down_proj (e.g. SwiGLU-style)."
            )
        act_fn = getattr(module, "act_fn", None)
        if act_fn is None:
            raise RuntimeError("Gate-only MLP biasing requires the module to expose an act_fn.")

        gate_base = module.gate_proj(hidden_states)
        up_base = module.up_proj(hidden_states)
        return module.down_proj(act_fn(gate_base + (scale * gate_control)) * up_base)

    def _input_gate_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        input_control: torch.Tensor,
        gate_control: torch.Tensor,
        input_scale: float,
        gate_scale: float,
    ):
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
            raise RuntimeError(
                "Combined input+gate MLP conditioning requires an MLP module with gate_proj/up_proj/down_proj (e.g. SwiGLU-style)."
            )
        act_fn = getattr(module, "act_fn", None)
        if act_fn is None:
            raise RuntimeError("Combined input+gate MLP conditioning requires the module to expose an act_fn.")

        conditioned_hidden_states = hidden_states + (input_scale * input_control)
        gate_base = module.gate_proj(conditioned_hidden_states)
        up_base = module.up_proj(conditioned_hidden_states)
        return module.down_proj(act_fn(gate_base + (gate_scale * gate_control)) * up_base)

    def _jvp_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        raw_control: torch.Tensor,
        scale: float,
    ):
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
            raise RuntimeError(
                "Linearized JVP MLP conditioning requires an MLP module with gate_proj/up_proj/down_proj (e.g. SwiGLU-style)."
            )
        act_fn = getattr(module, "act_fn", None)
        if act_fn is None:
            raise RuntimeError("Linearized JVP MLP conditioning requires the module to expose an act_fn.")

        gate_base = module.gate_proj(hidden_states)
        up_base = module.up_proj(hidden_states)
        act_base = act_fn(gate_base)
        base_output = module.down_proj(act_base * up_base)
        gate_control = module.gate_proj(raw_control)
        up_control = module.up_proj(raw_control)
        gate_response = self._activation_directional_derivative(act_fn, gate_base, gate_control)
        raw_correction = module.down_proj((gate_response * up_base) + (act_base * up_control))
        correction = self._prepare_output_correction(base_output, raw_correction)
        return base_output + (scale * correction)

    def _make_wrapped_forward(self, layer_id: int, module: torch.nn.Module, original_forward: object):
        signature = inspect.signature(original_forward)

        def wrapped_forward(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            hidden_states = bound.arguments.get("x")
            if hidden_states is None:
                hidden_states = bound.arguments.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return original_forward(*args, **kwargs)

            record = self.control_state.pop(layer_id)
            if record is None:
                return original_forward(*args, **kwargs)

            scale = self._resolve_scale(layer_id)
            if self.config.mode != "input_gate" and scale == 0.0:
                return original_forward(*args, **kwargs)

            raw_control = self._resolve_control_source(record, hidden_states)
            if self.config.mode == "input_add":
                input_control = self._prepare_input_control(hidden_states, raw_control)
                conditioned_hidden_states = hidden_states + (scale * input_control)
                if "x" in bound.arguments:
                    bound.arguments["x"] = conditioned_hidden_states
                elif "hidden_states" in bound.arguments:
                    bound.arguments["hidden_states"] = conditioned_hidden_states
                else:
                    args = (conditioned_hidden_states, *args[1:])
                    return original_forward(*args, **kwargs)
                return original_forward(*bound.args, **bound.kwargs)

            if self.config.mode == "input_gate":
                input_scale, gate_scale = self._resolve_dual_scales(layer_id)
                if input_scale == 0.0 and gate_scale == 0.0:
                    return original_forward(*args, **kwargs)
                input_control = self._prepare_input_control(hidden_states, raw_control)
                conditioned_hidden_states = hidden_states + (input_scale * input_control)
                gate_control = self._prepare_gate_control(module, conditioned_hidden_states, raw_control)
                return self._input_gate_forward(
                    module,
                    hidden_states,
                    input_control,
                    gate_control,
                    input_scale,
                    gate_scale,
                )

            if self.config.mode == "jvp":
                return self._jvp_forward(module, hidden_states, raw_control, scale)

            gate_control = self._prepare_gate_control(module, hidden_states, raw_control)
            return self._gate_bias_forward(module, hidden_states, gate_control, scale)

        return wrapped_forward
