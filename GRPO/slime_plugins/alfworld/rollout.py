"""SDA-style ALFWorld episodes emitted as slime's per-turn training samples."""

from __future__ import annotations

import copy
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .env import AlfWorldTextEnv


def _setting(args: Any, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


def _prompt(tokenizer, observation: str, args: Any, sample: Sample) -> tuple[str, list[int]]:
    kwargs = dict(getattr(args, "apply_chat_template_kwargs", None) or {})
    kwargs.update(sample.apply_chat_template_kwargs or {})
    kwargs.setdefault("enable_thinking", False)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": observation}],
        add_generation_prompt=True,
        tokenize=False,
        **kwargs,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > args.rollout_max_prompt_len:
        # SDA's ALFWorld launcher uses data.truncation=error.
        raise ValueError(
            f"ALFWorld prompt has {len(ids)} tokens, exceeding {args.rollout_max_prompt_len}"
        )
    return text, ids


def _turn_sample(
    base: Sample, turn: int, max_steps: int, prompt: str, prompt_ids: list[int], *, evaluation: bool
) -> Sample:
    if base.index is None or (not evaluation and base.group_index is None):
        raise ValueError("ALFWorld sample requires index, and training also requires group_index")
    result = Sample(
        index=base.index * max_steps + turn,
        group_index=base.group_index,
        rollout_id=base.rollout_id if base.rollout_id is not None else base.index,
        prompt=prompt,
        label=base.label,
        tokens=list(prompt_ids),
        response_length=0,
        loss_mask=[],
        metadata=copy.deepcopy(base.metadata),
        session_id=base.session_id,
        apply_chat_template_kwargs=dict(base.apply_chat_template_kwargs),
    )
    return result


def _output_tokens(output: dict[str, Any]) -> tuple[list[int], list[float], str]:
    meta = output.get("meta_info") or {}
    finish = (meta.get("finish_reason") or {}).get("type", "")
    if finish == "abort":
        return [], [], finish
    pairs = meta.get("output_token_logprobs")
    if pairs is None:
        raise RuntimeError("SGLang must return output_token_logprobs for ALFWorld rollout")
    ids = [int(item[1]) for item in pairs]
    log_probs = [float(item[0]) for item in pairs]
    if not ids:
        raise RuntimeError("SGLang returned an empty ALFWorld action")
    return ids, log_probs, finish


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False):
    """Run one episode using slime's SGLang router.

    Training returns one Sample per active environment step, sharing a
    rollout_id. Evaluation returns one episode summary Sample so slime counts
    each valid_seen game only once.
    """
    if args.partial_rollout:
        raise ValueError("Partial rollout is not supported for ALFWorld episodes")

    max_steps = int(_setting(args, "alfworld_max_steps", 50))
    history_length = int(_setting(args, "alfworld_history_length", 2))
    invalid_coef = float(_setting(args, "alfworld_invalid_action_coef", 0.1))
    if max_steps <= 0:
        raise ValueError("alfworld_max_steps must be positive")
    if invalid_coef < 0:
        raise ValueError("alfworld_invalid_action_coef must be nonnegative")
    if not evaluation and (sample.index is None or sample.group_index is None):
        raise ValueError("ALFWorld training samples require index and group_index")

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    gamefile = (sample.metadata or {}).get("gamefile")
    if evaluation and not gamefile:
        raise ValueError("ALFWorld evaluation requires a fixed gamefile in sample.metadata")
    seed_group = sample.group_index if not evaluation else sample.index
    seed = int(_setting(args, "alfworld_seed", 0)) + (1000 if evaluation else 0) + int(seed_group or 0)
    env = AlfWorldTextEnv(
        seed=seed,
        evaluation=evaluation,
        gamefile=gamefile,
        history_length=history_length,
        config_path=_setting(args, "alfworld_config_path", None),
    )

    turns: list[Sample] = []
    episode_reward = 0.0
    won = False
    try:
        observation = env.reset()
        for turn in range(max_steps):
            prompt, prompt_ids = _prompt(tokenizer, observation, args, sample)
            parameters = sampling_params.copy()
            # A fresh prompt is generated at every environment step. The
            # configured response budget therefore applies to one step, as in SDA.
            parameters["max_new_tokens"] = min(
                int(parameters.get("max_new_tokens", args.rollout_max_response_len)),
                int(args.rollout_max_response_len),
            )
            payload = {
                "input_ids": prompt_ids,
                "sampling_params": parameters,
                "return_logprob": True,
            }
            if getattr(args, "use_rollout_routing_replay", False):
                payload["return_routed_experts"] = True
            headers = None
            if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
                headers = {"X-SMG-Routing-Key": sample.session_id}
            output = await post(
                url,
                payload,
                headers=headers,
            )
            token_ids, log_probs, finish = _output_tokens(output)
            if finish == "abort":
                sample.status = Sample.Status.ABORTED
                return [sample] if not evaluation else sample

            text_action = output.get("text")
            if text_action is None:
                text_action = tokenizer.decode(token_ids, skip_special_tokens=True)
            row = _turn_sample(sample, turn, max_steps, prompt, prompt_ids, evaluation=evaluation)
            row.append_response_tokens(
                args,
                tokens=token_ids,
                log_probs=log_probs,
                trainable=True,
                meta_info=output["meta_info"],
                text=text_action,
            )
            row.status = Sample.Status.COMPLETED

            result = env.step(text_action)
            episode_reward += result.reward
            won = result.won
            row.metadata.update(
                {
                    "task": env.task,
                    "gamefile": env.gamefile,
                    "turn_step": turn,
                    "env_reward": result.reward,
                    "is_action_valid": result.is_action_valid,
                    "action": result.action,
                    "won": result.won,
                    "sglang_finish_reason": finish,
                }
            )
            turns.append(row)
            observation = result.observation
            if result.done:
                break

        if not turns:
            raise RuntimeError("ALFWorld episode produced no trainable turns")
        for row in turns:
            # SDA's EpisodeRewardManager writes the total episode outcome to
            # every turn and then subtracts the invalid-action penalty there.
            row.reward = episode_reward - invalid_coef * (not row.metadata["is_action_valid"])
            row.metadata["episode_reward"] = episode_reward
            row.metadata["episode_length"] = len(turns)
            row.metadata["episode_won"] = won

        if evaluation:
            summary = sample
            summary.reward = episode_reward
            summary.status = Sample.Status.COMPLETED
            summary.prompt = turns[0].prompt
            summary.response = "\n".join(row.response for row in turns)
            summary.response_length = sum(row.response_length for row in turns)
            summary.metadata = dict(summary.metadata or {})
            summary.metadata.update(
                {
                    "gamefile": env.gamefile,
                    "episode_length": len(turns),
                    "won": won,
                    "episode_reward": episode_reward,
                }
            )
            return summary
        return turns
    finally:
        env.close()
