from __future__ import annotations

import math
from typing import Dict, List, Sequence


LAYER_WEIGHT_SCHEDULE_CHOICES = (
    "flat",
    "late_heavy",
    "early_heavy",
    "middle_heavy",
    "gaussian_mid_late",
    "rank_linear",
    "full_per_layer",
)
MLP_BUDGET_NORMALIZATION_CHOICES = ("peak", "sum")


def layer_schedule_weights(length: int, schedule: str) -> List[float]:
    if length <= 0:
        raise ValueError("Layer schedule requires at least one layer.")
    if schedule == "flat":
        return [1.0] * length
    if schedule == "late_heavy":
        return [float(index + 1) for index in range(length)]
    if schedule == "early_heavy":
        return [float(length - index) for index in range(length)]
    if schedule == "middle_heavy":
        return [float(min(index + 1, length - index)) for index in range(length)]
    if schedule == "gaussian_mid_late":
        if length == 1:
            return [1.0]
        center = 0.6 * float(length - 1)
        sigma = max(1.0, float(length) / 4.0)
        return [
            float(math.exp(-0.5 * ((float(index) - center) / sigma) ** 2))
            for index in range(length)
        ]
    if schedule == "rank_linear":
        # Layer ids must be passed in score/rank order so the top-ranked layer
        # receives the largest budget share.
        return [float(length - index) for index in range(length)]
    if schedule == "full_per_layer":
        return [1.0] * length
    raise ValueError(f"Unsupported layer schedule '{schedule}'.")


def layer_rhos(layer_ids: Sequence[int], rho_total: float, schedule: str) -> Dict[int, float]:
    if str(schedule).strip().lower() == "full_per_layer":
        return {int(layer_id): float(rho_total) for layer_id in layer_ids}
    weights = layer_schedule_weights(len(layer_ids), schedule)
    normalizer = sum(weights)
    if normalizer <= 0.0:
        return {int(layer_id): float(rho_total) for layer_id in layer_ids}
    return {
        int(layer_id): float(rho_total) * float(weight) / float(normalizer)
        for layer_id, weight in zip(layer_ids, weights)
    }


def layer_scaled_values(
    layer_ids: Sequence[int],
    base_scale: float,
    schedule: str,
    *,
    normalization: str = "peak",
) -> Dict[int, float]:
    if not layer_ids:
        return {}
    normalized_mode = str(normalization).strip().lower()
    if normalized_mode not in MLP_BUDGET_NORMALIZATION_CHOICES:
        raise ValueError(
            f"Unsupported MLP budget normalization '{normalization}'. "
            f"Expected one of {MLP_BUDGET_NORMALIZATION_CHOICES}."
        )
    weights = layer_schedule_weights(len(layer_ids), schedule)
    if normalized_mode == "sum":
        normalizer = sum(weights)
        if normalizer <= 0.0:
            return {
                int(layer_id): float(base_scale) / float(len(layer_ids))
                for layer_id in layer_ids
            }
        return {
            int(layer_id): float(base_scale) * (float(weight) / float(normalizer))
            for layer_id, weight in zip(layer_ids, weights)
        }
    max_weight = max(weights)
    if max_weight <= 0.0:
        return {int(layer_id): float(base_scale) for layer_id in layer_ids}
    return {
        int(layer_id): float(base_scale) * (float(weight) / float(max_weight))
        for layer_id, weight in zip(layer_ids, weights)
    }
