from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CanonicalMemoryConfig:
    """Subset of canonical-memory settings needed by the vLLM attention port."""

    rho: float = 8.0
    mix_mode: str = "contrastive_neutral"
    steering_operator: str = "bank_softmax_mixture"
    positive_gain: float = 4.0
    negative_gain: float = 0.1
    query_adaptive_gates: bool = True
    query_gate_scale: float = 1.0


@dataclass(frozen=True)
class BankMixResult:
    steered_heads: torch.Tensor
    prompt_bank_mass: torch.Tensor
    trait_bank_mass: torch.Tensor
    reference_bank_mass: torch.Tensor


def _memory_log_prior(slot_count: int) -> float:
    return -torch.log(torch.tensor(float(max(int(slot_count), 1)))).item()


def bank_softmax_mixture(
    *,
    canonical_query_heads: torch.Tensor,
    prompt_scores_heads: torch.Tensor,
    prompt_values_heads: torch.Tensor,
    prompt_output_heads: torch.Tensor,
    trait_keys_heads: torch.Tensor,
    trait_values_heads: torch.Tensor,
    reference_keys_heads: Optional[torch.Tensor],
    reference_values_heads: Optional[torch.Tensor],
    scaling: float,
    config: CanonicalMemoryConfig,
) -> BankMixResult:
    """Compute the canonical-memory bank mixture for selected query heads.

    Shapes:
      canonical_query_heads: [batch, selected_heads, query, head_dim]
      prompt_scores_heads: [batch, selected_heads, query, prompt_tokens]
      prompt_values_heads: [batch, selected_heads, prompt_tokens, head_dim]
      prompt_output_heads: [batch, selected_heads, query, head_dim]
      trait/reference keys: [batch, selected_heads, slots, head_dim]

    This is the core math for the current reference default:
    `steering_operator=bank_softmax_mixture`.
    """

    if str(config.steering_operator) != "bank_softmax_mixture":
        raise NotImplementedError("This starter helper currently implements only bank_softmax_mixture.")

    trait_raw_scores = torch.matmul(canonical_query_heads, trait_keys_heads.transpose(-1, -2)) * float(scaling)
    trait_alignment = trait_raw_scores.max(dim=-1).values
    trait_output_heads = torch.matmul(
        F.softmax(trait_raw_scores, dim=-1, dtype=torch.float32).to(canonical_query_heads.dtype),
        trait_values_heads,
    )

    use_reference = (
        str(config.mix_mode).strip().lower() == "contrastive_opposite"
        and reference_keys_heads is not None
        and reference_values_heads is not None
    )
    if use_reference:
        reference_raw_scores = torch.matmul(canonical_query_heads, reference_keys_heads.transpose(-1, -2)) * float(scaling)
        reference_alignment = reference_raw_scores.max(dim=-1).values
        reference_output_heads = torch.matmul(
            F.softmax(reference_raw_scores, dim=-1, dtype=torch.float32).to(canonical_query_heads.dtype),
            reference_values_heads,
        )
    else:
        reference_raw_scores = None
        reference_alignment = torch.zeros_like(trait_alignment)
        reference_output_heads = None

    if bool(config.query_adaptive_gates) and use_reference:
        gate_logits = float(config.query_gate_scale) * (trait_alignment - reference_alignment)
        positive_gate = torch.sigmoid(gate_logits)
        negative_gate = torch.sigmoid(-gate_logits)
    else:
        positive_gate = torch.ones_like(trait_alignment)
        negative_gate = torch.ones_like(reference_alignment)

    prompt_scores = prompt_scores_heads + _memory_log_prior(int(prompt_values_heads.shape[2]))
    trait_scores = trait_raw_scores + (
        _memory_log_prior(int(trait_values_heads.shape[2]))
        + float(config.rho) * float(config.positive_gain) * positive_gate.unsqueeze(-1)
    )

    bank_logits = [
        torch.logsumexp(prompt_scores, dim=-1),
        torch.logsumexp(trait_scores.to(dtype=canonical_query_heads.dtype), dim=-1),
    ]
    bank_outputs = [prompt_output_heads, trait_output_heads]
    if use_reference and reference_raw_scores is not None and reference_output_heads is not None:
        reference_scores = reference_raw_scores + (
            _memory_log_prior(int(reference_values_heads.shape[2]))
            - float(config.rho) * float(config.negative_gain) * negative_gate.unsqueeze(-1)
        )
        bank_logits.append(torch.logsumexp(reference_scores.to(dtype=canonical_query_heads.dtype), dim=-1))
        bank_outputs.append(reference_output_heads)

    bank_probs = F.softmax(torch.stack(bank_logits, dim=-1), dim=-1, dtype=torch.float32).to(canonical_query_heads.dtype)
    steered_heads = torch.zeros_like(prompt_output_heads)
    for bank_idx, bank_output in enumerate(bank_outputs):
        steered_heads = steered_heads + bank_probs[..., bank_idx].unsqueeze(-1) * bank_output

    reference_mass = (
        bank_probs[..., 2]
        if use_reference and len(bank_outputs) == 3
        else torch.zeros_like(bank_probs[..., 1])
    )
    return BankMixResult(
        steered_heads=steered_heads,
        prompt_bank_mass=bank_probs[..., 0],
        trait_bank_mass=bank_probs[..., 1],
        reference_bank_mass=reference_mass,
    )
