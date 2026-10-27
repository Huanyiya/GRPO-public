"""CPU checks for WebShop trajectory grouping and training loss normalization."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import slime
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
        per_turn_loss=True,
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
    validity = [position % 17 != 0 for position in range(len(turns))]
    partitions, micro_batches, counts, divisors = build_dp_schedule(
        args, config, lengths, global_batch_size=128,
        rollout_indices=[turn.rollout_id for turn in turns],
        turn_validity=validity,
    )
    assert divisors == [sum(validity)]
    assert len(counts) == 1
    assert sorted(position for rank in partitions for position in rank) == list(range(len(turns)))
    assert all(len(rank_batches) == counts[0] for rank_batches in micro_batches)
    assert all(
        sum(lengths[partitions[rank][position]] for position in batch) <= 8192 * 2
        for rank, rank_batches in enumerate(micro_batches)
        for batch in rank_batches
    )


def _load_cp_reducer():
    """Use slime's real CP reducer without importing GPU Megatron."""
    mpu = SimpleNamespace(rank=0, size=1)
    mpu.get_context_parallel_rank = lambda: mpu.rank
    mpu.get_context_parallel_world_size = lambda: mpu.size
    fake_megatron = types.ModuleType("megatron")
    fake_core = types.ModuleType("megatron.core")
    fake_core.mpu = mpu
    old = {name: sys.modules.get(name) for name in ("megatron", "megatron.core")}
    sys.modules.update({"megatron": fake_megatron, "megatron.core": fake_core})
    try:
        path = Path(slime.__file__).resolve().parent / "backends/megatron_utils/cp_utils.py"
        spec = importlib.util.spec_from_file_location("_webshop_cp_reducer_check", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, value in old.items():
            if value is None:
                del sys.modules[name]
            else:
                sys.modules[name] = value
    return module, mpu


def _schedule_args(*, dynamic: bool, balance: bool, max_tokens: int = 20):
    return SimpleNamespace(
        per_turn_loss=True,
        use_dynamic_batch_size=dynamic,
        max_tokens_per_gpu=max_tokens if dynamic else None,
        micro_batch_size=1,
        balance_data=balance,
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


@pytest.mark.parametrize(
    "dynamic,balance,max_tokens,dp_size,cp_size",
    [
        (True, False, 20, 1, 1),
        (True, True, 40, 2, 1),
        (True, True, 20, 2, 2),
        (True, False, 20, 2, 4),
        (False, False, 20, 2, 2),
    ],
)
def test_per_turn_loss_and_gradient_are_invariant_to_batching_and_parallelism(
    dynamic: bool, balance: bool, max_tokens: int, dp_size: int, cp_size: int,
) -> None:
    cp_utils, mpu = _load_cp_reducer()
    lengths = [(16, 3), (24, 8), (32, 5), (40, 7), (16, 4), (24, 10), (32, 12), (48, 9)]
    trajectory_ids = [0, 0, 1, 1, 2, 2, 2, 2]
    generator = torch.Generator().manual_seed(2026)
    features = [torch.randn(response, 3, generator=generator, dtype=torch.float64) for _, response in lengths]
    masks = [torch.ones(response, dtype=torch.float64) for _, response in lengths]
    masks[0][-1] = 0
    masks[3][:] = 0
    masks[6][:3] = 0
    coefficients = [0.3, -0.4, 1.1, -0.7, 0.8, -0.2, 0.6, -0.9]
    validity = [bool(mask.sum()) for mask in masks]
    valid_count = sum(validity)
    initial_weight = torch.tensor([0.2, -0.1, 0.5], dtype=torch.float64)

    reference_weight = initial_weight.clone().requires_grad_(True)
    reference_loss = sum(
        ((features[i] @ reference_weight) * masks[i] * coefficients[i]).sum() / masks[i].sum()
        for i in range(len(masks)) if validity[i]
    ) / valid_count
    reference_loss.backward()

    partitions, micro_batches, counts, divisors = build_dp_schedule(
        _schedule_args(dynamic=dynamic, balance=balance, max_tokens=max_tokens),
        {"dp_size": dp_size, "cp_size": cp_size, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1},
        [total for total, _ in lengths],
        global_batch_size=3,
        rollout_indices=trajectory_ids,
        turn_validity=validity,
    )
    assert divisors == [valid_count]
    assert sorted(i for rank in partitions for i in rank) == list(range(len(lengths)))
    assert all(len(rank_batches) == counts[0] for rank_batches in micro_batches)

    weight = initial_weight.clone().requires_grad_(True)
    distributed_loss = weight.sum() * 0
    for rank in range(dp_size):
        for batch in micro_batches[rank]:
            indices = [partitions[rank][local_i] for local_i in batch]
            for cp_rank in range(cp_size):
                mpu.rank, mpu.size = cp_rank, cp_size
                local_values = []
                for i in indices:
                    total, response = lengths[i]
                    values = (features[i] @ weight) * coefficients[i]
                    if cp_size > 1:
                        prompt = total - response
                        _, _, _, offsets = cp_utils.get_logits_and_tokens_offset_with_cp(total, response)
                        values = torch.cat([
                            values[start - prompt : end - prompt] for start, end in offsets
                        ])
                    local_values.append(values)
                reducer = cp_utils.get_sum_of_sample_mean(
                    [lengths[i][0] for i in indices],
                    [lengths[i][1] for i in indices],
                    [masks[i] for i in indices],
                    sample_denoms=None,
                    calculate_per_token_loss=False,
                )
                # loss_function multiplies by M*DP*CP/N; Megatron divides
                # by M and DDP averages over DP*CP.
                scaled_loss = reducer(torch.cat(local_values)) * counts[0] * dp_size * cp_size / divisors[0]
                distributed_loss = distributed_loss + scaled_loss / (counts[0] * dp_size * cp_size)
    distributed_loss.backward()
    torch.testing.assert_close(distributed_loss, reference_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=1e-12, atol=1e-12)


def test_zero_mask_step_is_rejected_and_legacy_step_divisor_is_unchanged() -> None:
    args = _schedule_args(dynamic=False, balance=False)
    config = {"dp_size": 1, "cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}
    with pytest.raises(ValueError, match="no turn with a nonzero loss mask"):
        build_dp_schedule(args, config, [16, 16], global_batch_size=1,
                          rollout_indices=[0, 0], turn_validity=[False, False])
    args.per_turn_loss = False
    partitions, _, _, divisors = build_dp_schedule(
        args, config, [16, 16], global_batch_size=1,
        rollout_indices=[0, 0], turn_validity=[True, False],
    )
    assert partitions == [[0, 1]]
    assert divisors == [1]


def test_partial_step_padding_is_not_counted_as_a_turn() -> None:
    args = _schedule_args(dynamic=False, balance=False)
    args.include_partial_train_step = True
    config = {"dp_size": 2, "cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}
    partitions, micro_batches, counts, divisors = build_dp_schedule(
        args, config, [16, 16, 16, 16, 16], global_batch_size=2,
        rollout_indices=[0, 0, 1, 1, 2], turn_validity=[True, False, True, True, True],
    )
    assert divisors == [3, 1]
    assert counts == [2, 1]
    assert sorted(i for rank in partitions for i in rank if i >= 0) == [0, 1, 2, 3, 4]
    assert sorted(i for rank in partitions for i in rank if i < 0) == [-5]
    assert all(len(rank_batches) == sum(counts) for rank_batches in micro_batches)


def test_original_per_token_and_per_rollout_reducers_keep_their_gradients() -> None:
    cp_utils, _ = _load_cp_reducer()
    masks = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0, 0.0]), torch.tensor([1.0])]
    rollout_denoms = [torch.tensor(3.0), torch.tensor(3.0), torch.tensor(1.0)]
    coefficients = torch.tensor([1.0, 2.0, -1.0, 3.0, 1.0, -2.0], dtype=torch.float64)
    mask = torch.cat(masks).to(torch.float64)
    weight = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    values = coefficients * weight

    token_reducer = cp_utils.get_sum_of_sample_mean(
        [8, 8, 8], [2, 3, 1], masks, rollout_denoms, calculate_per_token_loss=True,
    )
    token_loss = token_reducer(values) / mask.sum()
    token_loss.backward(retain_graph=True)
    expected_token_grad = (coefficients * mask).sum() / mask.sum()
    torch.testing.assert_close(token_loss, values.dot(mask) / mask.sum())
    torch.testing.assert_close(weight.grad, expected_token_grad)

    weight.grad.zero_()
    rollout_reducer = cp_utils.get_sum_of_sample_mean(
        [8, 8, 8], [2, 3, 1], masks, rollout_denoms, calculate_per_token_loss=False,
    )
    rollout_loss = rollout_reducer(values) / 2
    rollout_loss.backward()
    expected_rollout = (values[:2].dot(mask[:2]) / 3 + values[2:5].dot(mask[2:5]) / 3
                        + values[5:].dot(mask[5:])) / 2
    expected_rollout_grad = ((coefficients[:2] * mask[:2]).sum() / 3
                             + (coefficients[2:5] * mask[2:5]).sum() / 3
                             + (coefficients[5:] * mask[5:]).sum()) / 2
    torch.testing.assert_close(rollout_loss, expected_rollout)
    torch.testing.assert_close(weight.grad, expected_rollout_grad)


def test_modified_tis_mask_uses_remaining_tokens_with_original_step_divisor() -> None:
    cp_utils, _ = _load_cp_reducer()
    original_masks = [torch.ones(3), torch.ones(2), torch.zeros(2)]
    modified_masks = [torch.tensor([1.0, 0.0, 0.0]), torch.zeros(2), torch.zeros(2)]
    weight = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    values = torch.tensor([2.0, 8.0, 9.0, 4.0, 5.0, 7.0, 3.0], dtype=torch.float64) * weight
    reducer = cp_utils.get_sum_of_sample_mean([8, 8, 8], [3, 2, 2], modified_masks, sample_denoms=None)
    loss = reducer(values) / sum(bool(mask.sum()) for mask in original_masks)
    loss.backward()
    torch.testing.assert_close(loss, weight)
    torch.testing.assert_close(weight.grad, torch.tensor(1.0, dtype=torch.float64))


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
