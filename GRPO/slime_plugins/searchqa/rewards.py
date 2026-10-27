"""SDA-style SearchQA cross-turn GRPO advantage normalization."""

from collections import defaultdict

import torch

from slime.utils.types import Sample


def normalize_turn_rewards(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Normalize rewards across all active turns of each question group.

    The rollout assigns the terminal exact-match reward to every turn and
    subtracts 0.01 only on turns with an invalid action. Keep those raw values
    for logging; return their group-normalized advantages for GRPO training.
    Loss aggregation remains slime's separate ``--per-turn-loss`` setting.
    """
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    if not args.rewards_normalization:
        return raw_rewards, raw_rewards.copy()

    groups: dict[int, list[int]] = defaultdict(list)
    for position, sample in enumerate(samples):
        if sample.group_index is None:
            raise ValueError("SearchQA GRPO turn reward requires group_index on every sample")
        groups[sample.group_index].append(position)

    advantages = [0.0] * len(samples)
    for positions in groups.values():
        # Invalid-action penalties can distinguish failures, but a valid wrong
        # answer must not become a positive training example when every
        # trajectory for this question failed.
        if not any(samples[position].metadata.get("episode_won", False) for position in positions):
            continue
        values = torch.tensor([raw_rewards[position] for position in positions], dtype=torch.float32)
        centered = values - values.mean()
        if args.grpo_std_normalization and len(positions) > 1:
            centered = centered / (centered.std(unbiased=True) + 1e-6)
        for position, advantage in zip(positions, centered.tolist(), strict=True):
            advantages[position] = advantage
    return raw_rewards, advantages
