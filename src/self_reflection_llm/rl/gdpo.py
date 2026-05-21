"""Small GDPO helpers independent of a specific RL trainer.

The verl integration uses these same reward component names:
``reward_component_correctness`` and ``reward_component_reflect``.
"""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Iterable, Mapping, Sequence

DEFAULT_GDPO_REWARD_KEYS = ("reward_component_correctness", "reward_component_reflect")
DEFAULT_GDPO_REWARD_WEIGHTS = (1.0, 0.3)


def _mean_std(values: Sequence[float], eps: float) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, sqrt(max(variance, 0.0) + eps)


def resolve_reward_weights(
    reward_keys: Sequence[str],
    reward_weights: Sequence[float] | Mapping[str, float] | None = None,
) -> list[float]:
    if reward_weights is None:
        if tuple(reward_keys) == DEFAULT_GDPO_REWARD_KEYS:
            return list(DEFAULT_GDPO_REWARD_WEIGHTS)
        return [1.0 for _ in reward_keys]
    if isinstance(reward_weights, Mapping):
        return [float(reward_weights.get(key, 1.0)) for key in reward_keys]
    weights = [float(value) for value in reward_weights]
    if len(weights) != len(reward_keys):
        raise ValueError("reward_weights length must match reward_keys")
    return weights


def group_decoupled_advantages(
    group_ids: Sequence[object],
    reward_components: Mapping[str, Sequence[float]],
    reward_keys: Sequence[str] = DEFAULT_GDPO_REWARD_KEYS,
    reward_weights: Sequence[float] | Mapping[str, float] | None = DEFAULT_GDPO_REWARD_WEIGHTS,
    eps: float = 1e-6,
    normalize_batch: bool = True,
) -> list[float]:
    """Compute GDPO scalar advantages.

    Each reward component is normalized within rollout groups first. The
    normalized components are then weighted, summed, and optionally normalized
    once across the full batch. This mirrors the paper's group reward-decoupled
    normalization while staying framework-neutral for testing and patching.
    """

    n_items = len(group_ids)
    if n_items == 0:
        return []
    missing = [key for key in reward_keys if key not in reward_components]
    if missing:
        raise ValueError(f"missing reward components: {missing}")

    weights = resolve_reward_weights(reward_keys, reward_weights)
    grouped_indices: dict[object, list[int]] = defaultdict(list)
    for idx, group_id in enumerate(group_ids):
        grouped_indices[group_id].append(idx)

    advantages = [0.0 for _ in range(n_items)]
    for key, weight in zip(reward_keys, weights):
        values = [float(value) for value in reward_components[key]]
        if len(values) != n_items:
            raise ValueError(f"reward component {key!r} has {len(values)} items, expected {n_items}")
        for indices in grouped_indices.values():
            group_values = [values[idx] for idx in indices]
            mean, std = _mean_std(group_values, eps)
            for idx in indices:
                advantages[idx] += ((values[idx] - mean) / std) * weight

    if normalize_batch:
        mean, std = _mean_std(advantages, eps)
        if std > 0:
            advantages = [(value - mean) / std for value in advantages]
    return advantages


def component_dict_from_records(
    records: Iterable[Mapping[str, object]],
    reward_keys: Sequence[str] = DEFAULT_GDPO_REWARD_KEYS,
) -> dict[str, list[float]]:
    components = {key: [] for key in reward_keys}
    for record in records:
        for key in reward_keys:
            components[key].append(float(record.get(key, 0.0)))
    return components
