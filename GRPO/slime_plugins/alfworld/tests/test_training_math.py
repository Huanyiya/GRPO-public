"""CPU checks for ALFWorld GRPO rewards and slime's token-mean gradient scaling."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

import slime
from slime.utils.dp_schedule import build_dp_schedule
from slime.utils.ppo_utils import get_grpo_returns
from slime.utils.types import Sample
from slime_plugins.alfworld.data_source import AlfWorldDataSource
from slime_plugins.alfworld.rewards import normalize_step_rewards


def test_sda_cross_step_grpo_reward() -> None:
    args = SimpleNamespace(reward_key=None, rewards_normalization=True, grpo_std_normalization=True)
    samples = [
        Sample(group_index=7, rollout_id=0, reward=10.0),
        Sample(group_index=7, rollout_id=1, reward=0.0),
        Sample(group_index=7, rollout_id=0, reward=9.9),
        Sample(group_index=8, rollout_id=2, reward=0.0),
        Sample(group_index=8, rollout_id=3, reward=10.0),
    ]
    raw, actual = normalize_step_rewards(args, samples)
    for positions in ([0, 1, 2], [3, 4]):
        scores = torch.tensor([raw[i] for i in positions], dtype=torch.float32)
        expected = (scores - scores.mean()) / (scores.std(unbiased=True) + 1e-6)
        torch.testing.assert_close(torch.tensor([actual[i] for i in positions]), expected)
    token_advantages = get_grpo_returns(
        torch.tensor(actual), [torch.zeros(length) for length in (2, 3, 1, 4, 2)]
    )
    for value, step in zip(actual, token_advantages, strict=True):
        torch.testing.assert_close(step, torch.full_like(step, value))


def test_dataset_free_group_cursor_survives_checkpoint(tmp_path: Path) -> None:
    args = SimpleNamespace(
        rollout_global_dataset=True,
        prompt_data=None,
        buffer_filter_path=None,
        n_samples_per_prompt=8,
        save=str(tmp_path),
        load=str(tmp_path),
    )
    source = AlfWorldDataSource(args)
    first = source.get_samples(2)
    assert [group[0].group_index for group in first] == [0, 1]
    assert all(len(group) == 8 for group in first)
    source.save(3)
    resumed = AlfWorldDataSource(args)
    resumed.load(3)
    next_group = resumed.get_samples(1)[0]
    assert next_group[0].group_index == 2
    assert [sample.index for sample in next_group] == list(range(16, 24))


def _load_cp_reducer():
    """Load the real slime reducer with a small mpu stub; no GPU Megatron import."""
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
        spec = importlib.util.spec_from_file_location("_alfworld_cp_reducer_check", path)
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


def test_token_mean_gradient_microbatch_dp_cp() -> None:
    cp_utils, mpu = _load_cp_reducer()
    generator = torch.Generator().manual_seed(81)
    lengths = [(16, 12), (16, 1), (24, 17), (32, 5), (16, 8), (24, 10)]
    features = [torch.randn(response, 3, dtype=torch.float64, generator=generator) for _, response in lengths]
    masks = [torch.ones(response, dtype=torch.float64) for _, response in lengths]
    masks[0][-2:] = 0
    masks[2][0] = 0
    masks[5][-3:] = 0
    coefficients = [0.3, -0.4, 1.1, -0.7, 0.8, -0.2]
    total_tokens = sum(int(mask.sum()) for mask in masks)

    base_weight = torch.tensor([0.2, -0.1, 0.5], dtype=torch.float64, requires_grad=True)
    baseline = sum(
        ((matrix @ base_weight) * mask * coefficient).sum()
        for matrix, mask, coefficient in zip(features, masks, coefficients, strict=True)
    ) / total_tokens
    baseline.backward()
    baseline_grad = base_weight.grad.detach().clone()

    for dp_size, cp_size in ((1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4)):
        schedule_args = SimpleNamespace(
            use_dynamic_batch_size=True,
            max_tokens_per_gpu=20,
            micro_batch_size=None,
            balance_data=False,
            balance_by_flops=False,
        )
        partitions, mb_indices, num_mbs, _ = build_dp_schedule(
            schedule_args,
            {"dp_size": dp_size, "cp_size": cp_size, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1},
            [total for total, _ in lengths],
            global_batch_size=len(lengths),
            rollout_indices=list(range(len(lengths))),
        )
        assert len(set(num_mbs)) == 1
        rank_batches = [
            [[partitions[rank][local_i] for local_i in mb] for mb in mb_indices[rank]]
            for rank in range(dp_size)
        ]
        assert sorted(i for batches in rank_batches for mb in batches for i in mb) == list(range(len(lengths)))
        weight = base_weight.detach().clone().requires_grad_(True)
        for batches in rank_batches:
            for cp_rank in range(cp_size):
                mpu.rank, mpu.size = cp_rank, cp_size
                for indices in batches:
                    local_values = []
                    for i in indices:
                        total, response = lengths[i]
                        full = (features[i] @ weight) * coefficients[i]
                        if cp_size == 1:
                            local_values.append(full)
                            continue
                        prompt = total - response
                        _, _, _, offsets = cp_utils.get_logits_and_tokens_offset_with_cp(total, response)
                        local_values.append(torch.cat([
                            full[start - prompt : end - prompt] for start, end in offsets
                        ]))
                    reducer = cp_utils.get_sum_of_sample_mean(
                        [lengths[i][0] for i in indices],
                        [lengths[i][1] for i in indices],
                        [masks[i] for i in indices],
                        calculate_per_token_loss=True,
                    )
                    # slime multiplies each CP-local numerator by CP size;
                    # Megatron sums DP*CP gradients, then divides by the
                    # DP*CP all-reduced full-token count (CP replicas included).
                    (reducer(torch.cat(local_values)) * cp_size).backward()
        weight.grad.div_(total_tokens * cp_size)
        torch.testing.assert_close(weight.grad, baseline_grad, rtol=1e-12, atol=1e-12)
