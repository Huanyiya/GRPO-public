"""Checks for IF verification and accepted-batch accuracy."""

import asyncio

from slime.utils.types import Sample
from slime_plugins.if_eval.metrics import log_train_batch_accuracy
from slime_plugins.if_eval.reward import reward_func


def test_if_reward_and_train_batch_accuracy():
    label = (
        "[{'instruction_id': ['last_word:last_word_answer', 'punctuation:no_comma'], "
        "'kwargs': [{'last_word': 'brief'}, None]}]"
    )
    samples = [
        Sample(response="keep brief", label=label, status=Sample.Status.COMPLETED),
        Sample(response="keep, brief", label=label, status=Sample.Status.COMPLETED),
        Sample(response="keep brief", label=label, status=Sample.Status.TRUNCATED),
    ]

    async def score_samples():
        for sample in samples:
            sample.reward = await reward_func(None, sample)

    asyncio.run(score_samples())
    assert [sample.reward for sample in samples] == [10.0, 5.0, 0.0]

    metrics = {}
    assert log_train_batch_accuracy(0, None, samples, metrics, 0.0) is False
    assert metrics == {
        "rollout/accuracy": 1 / 3,
        "rollout/constraint_accuracy": 0.5,
        "rollout/accuracy_num_samples": 3,
    }
