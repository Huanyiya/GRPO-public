"""CPU checks for WebShop trajectory grouping in slime's training batch."""

from types import SimpleNamespace

import torch

from slime.utils.dp_schedule import build_dp_schedule
from slime.utils.types import Sample
from slime_plugins.alfworld.rewards import normalize_step_rewards
from slime_plugins.webshop.rollout import _turn_sample


def test_turns_share_rollout_id_but_keep_unique_indices() -> None:
    base = Sample(index=17, group_index=2, metadata={"source": "webshop"})
    turns = [_turn_sample(base, turn, 15, "prompt", [1]) for turn in range(3)]
    assert [turn.index for turn in turns] == [255, 256, 257]
    assert [turn.rollout_id for turn in turns] == [17, 17, 17]
    assert [turn.group_index for turn in turns] == [2, 2, 2]
    assert [turn.metadata["trajectory_index"] for turn in turns] == [17, 17, 17]

    base.rollout_id = 99
    assert _turn_sample(base, 0, 15, "prompt", [1]).rollout_id == 99


def test_dynamic_training_step_includes_every_turn_of_128_trajectories() -> None:
    args = SimpleNamespace(
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=8192,
        micro_batch_size=1,
        balance_data=True,
        balance_by_flops=False,
        include_partial_train_step=False,
        hidden_size=2048,
        num_attention_heads=8,
        num_query_groups=2,
        kv_channels=256,
        vocab_size=248320,
        ffn_hidden_size=6144,
        num_layers=24,
        num_experts=None,
    )
    config = {"dp_size": 2, "cp_size": 2, "vpp_size": 1,
              "microbatch_group_size_per_vp_stage": 1}
    turns = [
        _turn_sample(Sample(index=trajectory, group_index=trajectory // 8), turn, 15, "prompt", [1])
        for trajectory in range(128)
        for turn in range(1 + trajectory % 5)
    ]
    lengths = [500 + turn.index % 11 * 100 for turn in turns]
    partitions, micro_batches, counts, divisors = build_dp_schedule(
        args, config, lengths, global_batch_size=128,
        rollout_indices=[turn.rollout_id for turn in turns],
    )
    assert divisors == [128]
    assert len(counts) == 1
    assert sorted(position for rank in partitions for position in rank) == list(range(len(turns)))
    assert all(len(rank_batches) == counts[0] for rank_batches in micro_batches)
    assert all(
        sum(lengths[partitions[rank][position]] for position in batch) <= 8192 * 2
        for rank, rank_batches in enumerate(micro_batches)
        for batch in rank_batches
    )


def test_reward_normalization_still_groups_steps_across_trajectories() -> None:
    first = Sample(index=0, group_index=7, reward=10.0)
    second = Sample(index=1, group_index=7, reward=0.0)
    turns = [
        _turn_sample(first, 0, 15, "prompt", [1]),
        _turn_sample(first, 1, 15, "prompt", [1]),
        _turn_sample(second, 0, 15, "prompt", [1]),
    ]
    for turn, reward in zip(turns, (10.0, 9.9, 0.0), strict=True):
        turn.reward = reward
    args = SimpleNamespace(reward_key=None, rewards_normalization=True, grpo_std_normalization=True)
    raw, actual = normalize_step_rewards(args, turns)
    scores = torch.tensor(raw, dtype=torch.float32)
    expected = (scores - scores.mean()) / (scores.std(unbiased=True) + 1e-6)
    torch.testing.assert_close(torch.tensor(actual), expected)
