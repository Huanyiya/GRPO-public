"""Group ALFWorld step rewards for slime's native GRPO advantage estimator."""

from collections import defaultdict

import torch

from slime.utils.types import Sample


def normalize_step_rewards(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Match SDA's cross-step GRPO group normalization.

    SDA assigns the episode outcome to every active step, subtracts 0.1 for
    each invalid action, then computes one mean and std over *all steps* of
    each prompt group. Slime receives the resulting scalar advantage for each
    step and broadcasts it to that step's response tokens with its existing
    GRPO estimator. Grouping by ``group_index`` is essential because episodes
    have different numbers of steps.
    """
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    if not args.rewards_normalization:
        return raw_rewards, raw_rewards.copy()

    groups: dict[int, list[int]] = defaultdict(list)
    for position, sample in enumerate(samples):
        if sample.group_index is None:
            raise ValueError("ALFWorld GRPO step reward requires group_index on every sample")
        groups[sample.group_index].append(position)

    advantages = [0.0] * len(samples)
    for positions in groups.values():
        values = torch.tensor([raw_rewards[position] for position in positions], dtype=torch.float32)
        centered = values - values.mean()
        if args.grpo_std_normalization and len(positions) > 1:
            centered = centered / (centered.std(unbiased=True) + 1e-6)
        for position, advantage in zip(positions, centered.tolist(), strict=True):
            advantages[position] = advantage
    return raw_rewards, advantages
