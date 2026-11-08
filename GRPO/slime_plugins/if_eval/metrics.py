"""Metrics computed on the accepted IF training rollout batch."""


def log_train_batch_accuracy(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Add strict and per-constraint accuracy, then keep slime's normal logging."""
    del rollout_id, args, rollout_time
    if not samples or rollout_extra_metrics is None:
        return False

    # Every sample here belongs to the accepted training batch (64 prompts x 8
    # completions), after any rollout filtering and before advantage calculation.
    rewards = [float(sample.reward) for sample in samples]
    rollout_extra_metrics["rollout/accuracy"] = sum(reward >= 10.0 for reward in rewards) / len(rewards)
    rollout_extra_metrics["rollout/constraint_accuracy"] = sum(rewards) / (10.0 * len(rewards))
    rollout_extra_metrics["rollout/accuracy_num_samples"] = len(samples)
    return False
