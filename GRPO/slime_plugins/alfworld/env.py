"""Text ALFWorld adapter with SDA's observations, actions, and environment reward."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .projection import alfworld_projection
from .prompts import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS


_GAMEFILES: dict[tuple[str, str, str], tuple[str, ...]] = {}
_GAMEFILES_LOCK = threading.Lock()
_TASK_PREFIX = "Your task is to: "


def _config_path(path: str | os.PathLike[str] | None) -> Path:
    return Path(path) if path is not None else Path(__file__).with_name("config_tw.yaml")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as reader:
        config = yaml.safe_load(reader)
    if config["env"]["type"] != "AlfredTWEnv":
        raise ValueError("The slime ALFWorld rollout supports SDA's AlfredTWEnv text environment only")
    return config


def _environment_class():
    try:
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    except ImportError as exc:
        raise ImportError(
            "ALFWorld TextWorld dependencies are missing; install the SDA ALFWorld/TextWorld environment"
        ) from exc
    return AlfredTWEnv


def _gamefiles(path: Path, split: str, config: dict[str, Any]) -> tuple[str, ...]:
    key = (str(path.resolve()), split, os.environ.get("ALFWORLD_DATA", ""))
    with _GAMEFILES_LOCK:
        if key not in _GAMEFILES:
            base = _environment_class()(config, train_eval=split)
            # Group members can execute on different Ray workers. TextWorld
            # chooses a game by index after seeding, so every worker needs the
            # same index-to-game mapping even if os.walk order differs.
            _GAMEFILES[key] = tuple(sorted(base.game_files))
        return _GAMEFILES[key]


def list_eval_gamefiles(config_path: str | os.PathLike[str] | None = None) -> list[str]:
    """Return SDA's valid_seen pool in strict-evaluation order."""
    path = _config_path(config_path)
    return sorted(_gamefiles(path, "eval_in_distribution", _load_config(path)))


def _first_info(infos: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(infos, list):
        return dict(infos[0])
    return {key: value[0] for key, value in infos.items()}


@dataclass(frozen=True)
class AlfWorldStep:
    observation: str
    reward: float
    done: bool
    won: bool
    action: str
    is_action_valid: bool
    info: dict[str, Any]


class AlfWorldTextEnv:
    """One SDA-compatible ALFWorld TextWorld episode.

    A group of slime samples gets the same seed (or an explicit gamefile), so
    its GRPO trajectories start from the same task. The underlying game list is
    cached; the TextWorld instance and its mutable history belong to this
    adapter alone.
    """

    def __init__(
        self,
        *,
        seed: int,
        evaluation: bool = False,
        gamefile: str | None = None,
        history_length: int = 2,
        config_path: str | os.PathLike[str] | None = None,
    ):
        if history_length < 0:
            raise ValueError("history_length must be nonnegative")
        path = _config_path(config_path)
        config = _load_config(path)
        split = "eval_in_distribution" if evaluation else "train"
        gamefiles = _gamefiles(path, split, config)
        if not gamefiles:
            raise RuntimeError(f"No ALFWorld games found for {split}; check ALFWORLD_DATA")
        if gamefile is not None and gamefile not in gamefiles:
            raise ValueError(f"Gamefile is outside the ALFWorld {split} pool: {gamefile}")

        environment_class = _environment_class()

        class CachedAlfredTWEnv(environment_class):
            def collect_game_files(self, verbose: bool = False):
                self.game_files = list(gamefiles)
                self.num_games = len(self.game_files)

        base = CachedAlfredTWEnv(config, train_eval=split)
        if gamefile is not None:
            base.game_files = [gamefile]
            base.num_games = 1
        self._env = base.init_env(batch_size=1)
        self._env.seed(seed)
        self.history_length = history_length
        self.gamefile = gamefile
        self.task = ""
        self._observation = ""
        self._admissible_commands: list[str] = []
        self._history: list[tuple[str, str]] = []

    def reset(self) -> str:
        observations, infos = self._env.reset()
        info = _first_info(infos)
        self._observation = observations[0]
        self._admissible_commands = list(info["admissible_commands"])
        self.gamefile = info.get("extra.gamefile") or self.gamefile
        task_start = self._observation.find(_TASK_PREFIX)
        if task_start < 0:
            raise ValueError("Task description not found in ALFWorld observation")
        self.task = self._observation[task_start + len(_TASK_PREFIX) :].strip()
        self._history.clear()
        return self._format_observation(initial=True)

    def step(self, text_action: str) -> AlfWorldStep:
        actions, valids = alfworld_projection([text_action], [self._admissible_commands])
        action = actions[0]
        observations, _scores, dones, infos = self._env.step([action])
        info = _first_info(infos)
        self._history.append((self._observation, action))
        self._observation = observations[0]
        self._admissible_commands = list(info["admissible_commands"])
        if info.get("extra.gamefile") is None:
            info["extra.gamefile"] = self.gamefile
        won = bool(info["won"])
        return AlfWorldStep(
            observation=self._format_observation(initial=False),
            reward=10.0 * float(won),
            done=bool(dones[0]),
            won=won,
            action=action,
            is_action_valid=bool(valids[0]),
            info=info,
        )

    def _format_observation(self, *, initial: bool) -> str:
        actions = "\n ".join(f"'{action}'" for action in self._admissible_commands if action != "help")
        if initial or self.history_length == 0:
            return ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=self._observation,
                admissible_actions=actions,
            )
        recent = self._history[-self.history_length :]
        start = len(self._history) - len(recent)
        memory = "\n".join(
            f"[Observation {start + i + 1}: '{observation}', Action {start + i + 1}: '{action}']"
            for i, (observation, action) in enumerate(recent)
        )
        return ALFWORLD_TEMPLATE.format(
            task_description=self.task,
            step_count=len(self._history),
            history_length=len(recent),
            action_history=memory,
            current_step=len(self._history) + 1,
            current_observation=self._observation,
            admissible_actions=actions,
        )

    def close(self) -> None:
        if getattr(self, "_env", None) is not None:
            self._env.close()
            self._env = None
