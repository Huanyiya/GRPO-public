"""Run a four-process CPU check of slime's ALFWorld token-mean gradient.

Usage: GLOO_SOCKET_IFNAME=lo python slime_plugins/alfworld/tests/check_distributed_gradient.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from slime.utils.dp_schedule import build_dp_schedule  # noqa: E402
from slime_plugins.alfworld.tests.test_training_math import _load_cp_reducer  # noqa: E402


def worker(rank: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=4)
    try:
        cp_utils, mpu = _load_cp_reducer()
        dp_rank, cp_rank = divmod(rank, 2)
        mpu.rank, mpu.size = cp_rank, 2

        lengths = [(16, 12), (16, 1), (24, 17), (32, 5), (16, 8), (24, 10)]
        generator = torch.Generator().manual_seed(81)
        features = [
            torch.randn(response, 3, dtype=torch.float64, generator=generator)
            for _, response in lengths
        ]
        masks = [torch.ones(response, dtype=torch.float64) for _, response in lengths]
        masks[0][-2:] = 0
        masks[2][0] = 0
        masks[5][-3:] = 0
        coefficients = [0.3, -0.4, 1.1, -0.7, 0.8, -0.2]

        args = SimpleNamespace(
            use_dynamic_batch_size=True,
            max_tokens_per_gpu=20,
            micro_batch_size=None,
            balance_data=False,
            balance_by_flops=False,
        )
        partitions, microbatches, counts, _ = build_dp_schedule(
            args,
            {"dp_size": 2, "cp_size": 2, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1},
            [total for total, _ in lengths],
            global_batch_size=len(lengths),
            rollout_indices=list(range(len(lengths))),
        )
        assert counts == [len(microbatches[dp_rank])]

        weight = torch.tensor([0.2, -0.1, 0.5], dtype=torch.float64, requires_grad=True)
        for microbatch in microbatches[dp_rank]:
            indices = [partitions[dp_rank][local_i] for local_i in microbatch]
            local_values = []
            for i in indices:
                total, response = lengths[i]
                prompt = total - response
                full_values = (features[i] @ weight) * coefficients[i]
                _, _, _, offsets = cp_utils.get_logits_and_tokens_offset_with_cp(total, response)
                local_values.append(
                    torch.cat([full_values[start - prompt : end - prompt] for start, end in offsets])
                )
            reducer = cp_utils.get_sum_of_sample_mean(
                [lengths[i][0] for i in indices],
                [lengths[i][1] for i in indices],
                [masks[i] for i in indices],
                calculate_per_token_loss=True,
            )
            # slime multiplies CP-local numerators by CP size before backward.
            (reducer(torch.cat(local_values)) * 2).backward()

        gradient = weight.grad.detach().clone()
        # Megatron all-reduces both gradients and full-mask token counts over DP*CP.
        token_count = torch.tensor(
            sum(int(masks[i].sum()) for i in partitions[dp_rank]), dtype=torch.float64
        )
        dist.all_reduce(gradient)
        dist.all_reduce(token_count)
        gradient /= token_count

        baseline_weight = weight.detach().clone().requires_grad_(True)
        baseline = sum(
            ((matrix @ baseline_weight) * mask * coefficient).sum()
            for matrix, mask, coefficient in zip(features, masks, coefficients, strict=True)
        ) / sum(int(mask.sum()) for mask in masks)
        baseline.backward()
        torch.testing.assert_close(gradient, baseline_weight.grad, rtol=1e-12, atol=1e-12)
        if rank == 0:
            print("4-process Gloo DP=2 CP=2 micro-batch gradient matches unsplit token mean")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="alfworld_gloo_") as directory:
        mp.spawn(worker, args=(str(Path(directory) / "store"),), nprocs=4, join=True)
