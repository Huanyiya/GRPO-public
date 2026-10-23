import logging
from pathlib import Path
from time import perf_counter

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils import logging_utils
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def train(args):
    configure_logger()
    release_train = args.release_train

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # Always push actor weights to rollout once weights are loaded.
    split_disk_sync = actor_model.uses_split_disk_weight_sync()
    if split_disk_sync:
        initial_export = actor_model.export_weights_to_disk()
        actor_model.offload()
        actor_model.reload_rollout_weights_from_disk(initial_export)
    else:
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        step_start = perf_counter() if args.log_grpo_step_timing else None
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_result = ray.get(rollout_manager.generate.remote(rollout_id))
        if args.log_grpo_step_timing:
            rollout_data_ref, rollout_seconds = rollout_result
        else:
            rollout_data_ref = rollout_result

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        actor_results = []
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                actor_results = ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            actor_results = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if actor_trains and split_disk_sync:
            export_result = actor_model.export_weights_to_disk()
            actor_model.offload()
            actor_model.reload_rollout_weights_from_disk(export_result)
        else:
            offload_train(actor_trains)
            if args.offload_rollout and not release_train:
                ray.get(rollout_manager.onload_weights.remote())
            if actor_trains:
                actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

        if args.log_grpo_step_timing:
            # One coherent actor timeline keeps the residual non-overlapping.
            phase_times = next((result for result in actor_results if isinstance(result, dict)), None)
            if actor_trains and phase_times is None:
                raise RuntimeError(f"Missing actor step timings for rollout {rollout_id}")
            phase_times = phase_times or {
                "advantages_s": 0.0,
                "policy_ref_scoring_s": 0.0,
                "policy_update_s": 0.0,
                "teacher_forward_s": 0.0,
            }
            total_seconds = perf_counter() - step_start
            phase_times = {"rollout_s": rollout_seconds, **phase_times}
            phase_times["other_s"] = total_seconds - sum(phase_times.values())
            if phase_times["other_s"] < 0:
                logger.warning("Negative Other time for rollout %s: %.6fs", rollout_id, phase_times["other_s"])
            timing_line = (
                f"step_timing rollout_id={rollout_id} step={rollout_id + 1} "
                + " ".join(f"{key}={value:.3f}" for key, value in phase_times.items())
                + f" step_total_s={total_seconds:.3f}"
            )
            logger.info(timing_line)
            if args.grpo_step_timing_log_file:
                timing_log_file = Path(args.grpo_step_timing_log_file)
                timing_log_file.parent.mkdir(parents=True, exist_ok=True)
                with timing_log_file.open("a", encoding="utf-8") as timing_log:
                    timing_log.write(timing_line + "\n")
            logging_utils.log(
                args,
                {
                    "rollout/step": compute_rollout_step(args, rollout_id),
                    **{f"step_timing/{key}": value for key, value in phase_times.items()},
                    "step_timing/step_total_s": total_seconds,
                },
                step_key="rollout/step",
            )

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
