"""Episode accuracy for ALFWorld training rollouts."""


def log_train_accuracy(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Log the fraction of completed training episodes that solved the task."""
    del rollout_id, args, rollout_time
    if rollout_extra_metrics is None:
        return False

    outcomes = {}
    for sample in samples:
        if sample.metadata and "episode_won" in sample.metadata:
            if sample.rollout_id is None:
                raise ValueError("ALFWorld training accuracy requires rollout_id on each episode step")
            outcomes[sample.rollout_id] = bool(sample.metadata["episode_won"])

    if outcomes:
        rollout_extra_metrics["rollout/accuracy"] = sum(outcomes.values()) / len(outcomes)
        rollout_extra_metrics["rollout/accuracy_num_episodes"] = len(outcomes)
    return False
