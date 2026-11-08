"""Keep SearchQA question groups with different terminal outcomes."""

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample


def keep_mixed_outcomes(args, group: list[list[Sample]]) -> DynamicFilterOutput:
    if len(group) != args.n_samples_per_prompt:
        raise ValueError(f"Expected {args.n_samples_per_prompt} SearchQA trajectories, got {len(group)}")

    outcomes = []
    for turns in group:
        if not turns or not all(isinstance(turn, Sample) for turn in turns):
            raise ValueError("SearchQA filtering requires a nonempty list of turns per trajectory")
        outcome = turns[-1].metadata.get("episode_reward")
        if outcome is None:
            raise ValueError("SearchQA trajectory has no terminal episode_reward")
        outcomes.append(float(outcome))

    keep = any(outcome != outcomes[0] for outcome in outcomes[1:])
    return DynamicFilterOutput(keep=keep, reason=None if keep else f"same_outcome_{outcomes[0]:g}")
