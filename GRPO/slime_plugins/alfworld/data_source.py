"""Dataset-free ALFWorld task groups with checkpointable environment seeds."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from slime.rollout.data_source import RolloutDataSourceWithBuffer

logger = logging.getLogger(__name__)


class AlfWorldDataSource(RolloutDataSourceWithBuffer):
    """Use slime's group allocation and persist its counters on resume.

    Without ``--prompt-data``, the base class emits empty samples and the
    ALFWorld environment supplies the task. Slime's placement group only
    invokes ``load`` while global dataset mode is enabled, so this adapter
    keeps that mode on and checkpoints the seed counters itself.
    """

    def save(self, rollout_id: int) -> None:
        path = Path(self.args.save) / "rollout" / f"alfworld_data_source_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sample_group_index": self.sample_group_index,
                "sample_index": self.sample_index,
                "metadata": self.metadata,
            },
            path,
        )

    def load(self, rollout_id: int | None = None) -> None:
        if self.args.load is None or rollout_id is None:
            return
        path = Path(self.args.load) / "rollout" / f"alfworld_data_source_{rollout_id}.pt"
        if not path.exists():
            logger.info("ALFWorld data source checkpoint %s does not exist", path)
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.sample_group_index = int(state["sample_group_index"])
        self.sample_index = int(state["sample_index"])
        self.metadata = state.get("metadata", {})
