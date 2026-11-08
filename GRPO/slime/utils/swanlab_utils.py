import logging
import os

from .wandb_utils import _compute_config_for_logging, _compute_secondary_config_for_logging

logger = logging.getLogger(__name__)


def _import_swanlab():
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError("SwanLab tracking requires the `swanlab` package. Install it with `pip install swanlab`.") from exc
    return swanlab


def init_swanlab_primary(args):
    if not args.use_swanlab:
        args.swanlab_run_id = None
        return
    swanlab = _import_swanlab()
    if args.swanlab_key:
        swanlab.login(api_key=args.swanlab_key)
    run = swanlab.init(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        group=args.swanlab_group,
        experiment_name=args.swanlab_group,
        config=_compute_config_for_logging(args),
        logdir=args.swanlab_dir,
        mode=args.swanlab_mode,
        id=args.swanlab_run_id,
        resume="allow" if args.swanlab_run_id else None,
    )
    args.swanlab_run_id = run.id


def init_swanlab_secondary(args, role=None):
    if not args.use_swanlab or not args.swanlab_run_id:
        return
    swanlab = _import_swanlab()
    if args.swanlab_key:
        swanlab.login(api_key=args.swanlab_key)
    swanlab.init(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        group=args.swanlab_group,
        config=_compute_secondary_config_for_logging(args, role=role),
        logdir=args.swanlab_dir,
        mode=args.swanlab_mode,
        id=args.swanlab_run_id,
        resume="allow",
        parallel="shared",
    )


def finish_swanlab(args):
    if args.use_swanlab:
        try:
            _import_swanlab().finish()
        except Exception:
            logger.exception("Failed to finish SwanLab run")
