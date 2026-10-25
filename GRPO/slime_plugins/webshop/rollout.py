"""SDA-style WebShop episodes emitted as slime per-turn GRPO samples.

Set WEBSHOP_ROOT to a WebShop source tree containing web_agent_site and its
search_engine/indexes directory. Set WEBSHOP_DATA to the small WebShop data
directory. The launcher propagates both paths to Ray workers.
"""

from __future__ import annotations

import copy
import os
import random
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample


# Keep SDA's wording and formatting: the same two prompts are used by its
# WebshopEnvironmentManager for the first and subsequent environment steps.
PROMPT_NO_HISTORY = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

PROMPT_WITH_HISTORY = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


class WebshopDataSource(RolloutDataSourceWithBuffer):
    """Allocate dataset-free groups and persist counters for checkpoint resume."""

    def save(self, rollout_id: int) -> None:
        path = Path(self.args.save) / "rollout" / f"webshop_data_source_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"sample_group_index": self.sample_group_index,
             "sample_index": self.sample_index, "metadata": self.metadata},
            path,
        )

    def load(self, rollout_id: int | None = None) -> None:
        if self.args.load is None or rollout_id is None or rollout_id < 0:
            return
        path = Path(self.args.load) / "rollout" / f"webshop_data_source_{rollout_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing WebShop data source state: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.sample_group_index = int(state["sample_group_index"])
        self.sample_index = int(state["sample_index"])
        self.metadata = state.get("metadata", {})


def _setting(args: Any, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


@lru_cache(maxsize=1)
def _webshop_class(root: str):
    source = Path(root)
    if not (source / "web_agent_site" / "envs" / "web_agent_text_env.py").is_file():
        raise FileNotFoundError(f"WEBSHOP_ROOT is not a WebShop source tree: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv

    return SimServer, WebAgentTextEnv


@lru_cache(maxsize=32)
def _server(root: str, data: str, seed: int):
    SimServer, _ = _webshop_class(root)
    # WebAgentTextEnv seeds these generators before constructing SimServer.
    # SimServer builds goal prices before it shuffles the goal list, so seed
    # before construction to preserve SDA's per-group goal assignment.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return SimServer(
        seed,
        "http://127.0.0.1:3000",
        str(Path(data) / "items_shuffle_1000.json"),
        str(Path(data) / "items_ins_v2_1000.json"),
        num_products=None,
        human_goals=False,
    )


@lru_cache(maxsize=256)
def _goal_indices(seed: int, batch: int, num_goals: int, round_index: int) -> tuple[int, ...]:
    if num_goals - 500 < batch:
        raise ValueError(f"WebShop requires at least {500 + batch} goals, got {num_goals}")
    rng = np.random.RandomState(seed)
    selected = None
    for _ in range(round_index + 1):
        selected = rng.choice(range(500, num_goals), size=batch, replace=False)
    return tuple(int(i) for i in selected)


def _project_action(response: str) -> tuple[str, bool]:
    # Equivalent to SDA's webshop_projection, including its fallback string.
    lowered = response.lower()
    start, end = lowered.find("<action>"), lowered.find("</action>")
    if start == -1 or end == -1:
        return lowered[-20:], False
    action = lowered[start + len("<action>"):end].strip().lower()
    valid = "<think>" in response and "</think>" in response
    valid = valid and re.search(r"[\u4e00-\u9fff]", response) is None
    return action, valid


def _format_observation(observation: str, task: str) -> str:
    parts = observation.split(" [SEP] ")
    try:
        position = parts.index(task)
        return " [SEP] ".join(f"'{part}'" for part in parts[position + 1:])
    except ValueError:
        return observation


def _available_actions(env) -> str:
    available = env.get_available_actions()
    actions = []
    if available["has_search_bar"]:
        actions.append("search[<your query>]")
    actions.extend(f"click[{name}]" for name in available["clickables"])
    return "\n".join(f"'{action}'," for action in actions)


def _observation_prompt(task: str, observation: str, actions: str,
                        history: list[tuple[str, str]], history_length: int) -> str:
    current = _format_observation(observation, task)
    available = actions
    if not history or history_length <= 0:
        return PROMPT_NO_HISTORY.format(task_description=task,
                                        current_observation=current,
                                        available_actions=available)
    recent = history[-history_length:]
    first = len(history) - len(recent) + 1
    context = "\n".join(
        f"[Observation {first + j}: '{obs}', Action {first + j}: '{action}']"
        for j, (obs, action) in enumerate(recent)
    )
    prompt = PROMPT_WITH_HISTORY.format(
        task_description=task,
        step_count=len(history),
        history_length=len(recent),
        action_history=context,
        current_step=len(history) + 1,
        current_observation=current,
        available_actions=available,
    )
    if len(prompt) > 13000:
        return PROMPT_NO_HISTORY.format(task_description=task,
                                        current_observation=current,
                                        available_actions=available)
    return prompt


def _turn_sample(base: Sample, turn: int, max_steps: int, prompt: str,
                 prompt_ids: list[int]) -> Sample:
    turn_index = base.index * max_steps + turn
    metadata = copy.deepcopy(base.metadata)
    metadata["trajectory_index"] = base.index
    return Sample(
        index=turn_index,
        group_index=base.group_index,
        rollout_id=base.rollout_id if base.rollout_id is not None else base.index,
        prompt=prompt,
        label=base.label,
        tokens=list(prompt_ids),
        response_length=0,
        loss_mask=[],
        metadata=metadata,
        session_id=base.session_id,
        apply_chat_template_kwargs=dict(base.apply_chat_template_kwargs),
    )


def _output_tokens(output: dict[str, Any]) -> tuple[list[int], list[float], str]:
    meta = output.get("meta_info") or {}
    finish = (meta.get("finish_reason") or {}).get("type", "")
    if finish == "abort":
        return [], [], finish
    pairs = meta.get("output_token_logprobs")
    if not pairs:
        raise RuntimeError("SGLang must return nonempty output_token_logprobs for WebShop")
    return [int(item[1]) for item in pairs], [float(item[0]) for item in pairs], finish


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False):
    """Run one WebShop training episode, returning one Sample per active step."""
    if evaluation:
        raise ValueError("This WebShop launcher does not configure evaluation")
    if args.partial_rollout:
        raise ValueError("Partial rollout is not supported for WebShop episodes")
    if sample.index is None or sample.group_index is None:
        raise ValueError("WebShop training samples require index and group_index")

    root = os.environ.get("WEBSHOP_ROOT")
    data = os.environ.get("WEBSHOP_DATA")
    if not root or not data:
        raise RuntimeError("Set WEBSHOP_ROOT and WEBSHOP_DATA for WebShop rollout")
    max_steps = int(_setting(args, "webshop_max_steps", 15))
    history_length = int(_setting(args, "webshop_history_length", 2))
    invalid_coef = float(_setting(args, "webshop_invalid_action_coef", 0.1))
    seed = int(_setting(args, "webshop_seed", 0))
    batch = int(args.rollout_batch_size)
    if max_steps <= 0 or history_length < 0 or invalid_coef < 0 or batch <= 0:
        raise ValueError("Invalid WebShop rollout settings")

    slot = sample.group_index % batch
    round_index = sample.group_index // batch
    server = _server(root, data, seed + slot)
    goal_idx = _goal_indices(seed, batch, len(server.goals), round_index)[slot]
    _, WebAgentTextEnv = _webshop_class(root)
    env = WebAgentTextEnv(
        observation_mode="text", num_products=None, human_goals=False,
        file_path=str(Path(data) / "items_shuffle_1000.json"),
        attr_path=str(Path(data) / "items_ins_v2_1000.json"),
        server=server, seed=seed + slot, session_prefix=f"rollout-{sample.index}-",
    )
    # Its constructor creates a throwaway random session before the indexed
    # reset. The shared server should retain only live episode sessions.
    server.user_sessions.pop(env.session, None)
    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    turns: list[Sample] = []
    history: list[tuple[str, str]] = []
    episode_reward = 0.0
    won = False
    task_score = 0.0
    active_session = None
    try:
        observation, _ = env.reset(session=goal_idx)
        active_session = env.session
        parts = observation.split(" [SEP] ")
        if len(parts) < 3 or parts[1] != "Instruction:":
            raise RuntimeError("Unexpected WebShop initial observation format")
        task = parts[2]
        for turn in range(max_steps):
            formatted_observation = _format_observation(observation, task)
            prompt = _observation_prompt(task, observation, _available_actions(env),
                                         history, history_length)
            kwargs = dict(getattr(args, "apply_chat_template_kwargs", None) or {})
            kwargs.update(sample.apply_chat_template_kwargs or {})
            kwargs.setdefault("enable_thinking", False)
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False, **kwargs,
            )
            prompt_ids = tokenizer.encode(chat_prompt, add_special_tokens=False)
            if len(prompt_ids) > args.rollout_max_prompt_len:
                raise ValueError(
                    f"WebShop prompt has {len(prompt_ids)} tokens, "
                    f"exceeding {args.rollout_max_prompt_len}"
                )
            parameters = sampling_params.copy()
            parameters["max_new_tokens"] = min(
                int(parameters.get("max_new_tokens", args.rollout_max_response_len)),
                int(args.rollout_max_response_len),
            )
            payload = {"input_ids": prompt_ids, "sampling_params": parameters,
                       "return_logprob": True}
            if getattr(args, "use_rollout_routing_replay", False):
                payload["return_routed_experts"] = True
            headers = None
            if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
                headers = {"X-SMG-Routing-Key": sample.session_id}
            output = await post(url, payload, headers=headers)
            token_ids, log_probs, finish = _output_tokens(output)
            if finish == "abort":
                sample.status = Sample.Status.ABORTED
                return [sample]
            response = output.get("text")
            if response is None:
                response = tokenizer.decode(token_ids, skip_special_tokens=True)
            row = _turn_sample(sample, turn, max_steps, chat_prompt, prompt_ids)
            row.append_response_tokens(
                args, tokens=token_ids, log_probs=log_probs, trainable=True,
                meta_info=output["meta_info"], text=response,
            )
            row.status = Sample.Status.COMPLETED
            action, valid = _project_action(response)
            next_observation, task_score, done, _ = env.step(action)
            won = bool(done and task_score == 1.0)
            if won:
                episode_reward = 10.0
            row.metadata.update({
                "task": task, "goal_index": goal_idx, "turn_step": turn,
                "task_score": float(task_score), "is_action_valid": valid,
                "action": action, "won": won, "sglang_finish_reason": finish,
            })
            turns.append(row)
            history.append((formatted_observation, action))
            observation = next_observation
            if done:
                break

        if not turns:
            raise RuntimeError("WebShop episode produced no trainable turns")
        for row in turns:
            # SDA repeats terminal outcome on each active turn, with a per-turn
            # penalty for an invalid action format.
            row.reward = episode_reward - invalid_coef * (not row.metadata["is_action_valid"])
            row.metadata.update({"episode_reward": episode_reward,
                                 "episode_length": len(turns), "episode_won": won,
                                 "final_task_score": float(task_score)})
        return turns
    finally:
        env.close()
        # WebAgentTextEnv.step() resets after a terminal action, so remove both
        # the goal session and that final throwaway session from shared state.
        if active_session is not None:
            server.user_sessions.pop(active_session, None)
        server.user_sessions.pop(env.session, None)
