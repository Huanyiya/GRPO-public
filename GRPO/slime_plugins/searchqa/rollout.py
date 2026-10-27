"""SDA-style SearchQA episodes emitted as slime per-turn training samples."""

from __future__ import annotations

import copy
import os
from typing import Any

from agent_system.searchqa import SearchQAEnvironment
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample


def _setting(args: Any, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


def _turn_sample(base: Sample, turn: int, max_steps: int, prompt: str, prompt_ids: list[int]) -> Sample:
    metadata = copy.deepcopy(base.metadata)
    metadata["trajectory_index"] = base.index
    return Sample(
        index=base.index * max_steps + turn,
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
        raise RuntimeError("SGLang must return nonempty output_token_logprobs for SearchQA")
    return [int(item[1]) for item in pairs], [float(item[0]) for item in pairs], finish


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False):
    if evaluation:
        raise ValueError("This SearchQA launcher does not configure evaluation")
    if args.partial_rollout:
        raise ValueError("Partial rollout is not supported for SearchQA episodes")
    if sample.index is None or sample.group_index is None:
        raise ValueError("SearchQA training samples require index and group_index")

    max_steps = int(_setting(args, "searchqa_max_steps", 4))
    history_length = int(_setting(args, "searchqa_history_length", 4))
    invalid_coef = float(_setting(args, "searchqa_invalid_action_coef", 0.01))
    if invalid_coef < 0:
        raise ValueError("SearchQA invalid-action coefficient must be nonnegative")
    search_url = os.environ.get("SEARCH_URL", "http://127.0.0.1:8000/retrieve")
    environment = SearchQAEnvironment(
        sample.metadata["question"],
        sample.metadata["ground_truth"],
        sample.metadata["data_source"],
        search_url=search_url,
        max_steps=max_steps,
        history_length=history_length,
        topk=int(os.environ.get("SEARCH_TOPK", "3")),
        timeout=int(os.environ.get("SEARCH_TIMEOUT", "60")),
        api_style=os.environ.get("SEARCH_API_STYLE", "batch"),
    )
    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    turns: list[Sample] = []
    episode_reward = 0.0
    try:
        for turn in range(max_steps):
            kwargs = dict(getattr(args, "apply_chat_template_kwargs", None) or {})
            kwargs.update(sample.apply_chat_template_kwargs or {})
            kwargs.setdefault("enable_thinking", False)
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": environment.prompt()}],
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )
            prompt_ids = tokenizer.encode(chat_prompt, add_special_tokens=False)
            # SDA SearchQA uses data.truncation=left at max_prompt_length=4096.
            prompt_ids = prompt_ids[-int(args.rollout_max_prompt_len):]
            parameters = sampling_params.copy()
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
            result = await environment.step(response)
            row.metadata.update({
                "turn_step": turn,
                "action": result.action,
                "is_action_valid": result.valid,
                "tool_calling": result.tool_calling,
                "search_observation": result.observation,
                "sglang_finish_reason": finish,
            })
            turns.append(row)
            episode_reward += result.reward
            if result.done:
                break

        if not turns:
            raise RuntimeError("SearchQA episode produced no trainable turns")
        for row in turns:
            # Match SDA's EpisodeRewardManager + per-step invalid-action penalty.
            row.reward = episode_reward - invalid_coef * (not row.metadata["is_action_valid"])
            row.metadata.update({
                "episode_reward": episode_reward,
                "episode_length": len(turns),
                "episode_won": episode_reward >= 1.0,
            })
        return turns
    finally:
        await environment.close()
