"""Evaluate a Hugging Face model on SDA's strict ALFWorld valid_seen pool.

The model is served by SGLang. This script owns the environment loop and sends
one generation request per environment step to the server's /generate endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from slime_plugins.alfworld.env import AlfWorldTextEnv, list_eval_gamefiles  # noqa: E402


CATEGORIES = (
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
)


def category_for(gamefile: str) -> str:
    for category in CATEGORIES:
        if category in gamefile:
            return category
    raise ValueError(f"Unrecognized ALFWorld task category: {gamefile}")


def request_seed(task_seed: int, rollout_seed: int, gamefile: str, turn_index: int) -> int:
    """Match SDA's per_task_turn seed derivation for ALFWorld requests."""
    task_key = f"alfworld:{gamefile.split('json_2.1.1/', 1)[1]}"
    payload = ["eval_request_seed_v1", "alfworld", task_seed, rollout_seed, task_key, turn_index]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def generate(
    url: str,
    input_ids: list[int],
    *,
    temperature: float,
    max_tokens: int,
    timeout: int,
    sampling_seed: int,
) -> str:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": temperature,
            "top_p": 1.0,
            "max_new_tokens": max_tokens,
            "sampling_seed": sampling_seed,
        },
    }
    request = Request(
        f"{url.rstrip('/')}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"SGLang generation failed: {exc}") from exc
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise RuntimeError(f"Unexpected SGLang /generate response: {result!r}")
    return result["text"]


def evaluate(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Expected a converted Hugging Face model: {model_path}/config.json")

    data_path = Path(args.alfworld_data).expanduser().resolve()
    for relative in (
        "json_2.1.1/valid_seen",
        "logic/alfred.pddl",
        "logic/alfred.twl2",
    ):
        if not (data_path / relative).exists():
            raise FileNotFoundError(f"Missing ALFWorld asset: {data_path / relative}")
    os.environ["ALFWORLD_DATA"] = str(data_path)

    gamefiles = list_eval_gamefiles()
    if len(gamefiles) != args.expected_games:
        raise RuntimeError(
            f"Strict valid_seen pool has {len(gamefiles)} solvable games, "
            f"expected {args.expected_games}; check ALFWORLD_DATA"
        )
    if len(set(gamefiles)) != len(gamefiles):
        raise RuntimeError("The ALFWorld evaluation pool contains duplicate gamefiles")
    if args.limit is not None:
        if not 1 <= args.limit <= len(gamefiles):
            raise ValueError(f"--limit must be between 1 and {len(gamefiles)}")
        gamefiles = gamefiles[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    rollout_path = output_dir / "rollouts.jsonl"
    summary_path = output_dir / "summary.json"
    if rollout_path.exists() or summary_path.exists():
        raise FileExistsError(f"Evaluation output already exists in {output_dir}")
    if args.check_only:
        print(f"Preflight passed: {len(gamefiles)} eligible valid_seen games; model tokenizer loaded")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    outcomes: dict[str, list[int]] = defaultdict(list)
    # SDA creates validation environments with env.seed + 1000 + worker index.
    with rollout_path.open("x", encoding="utf-8") as writer:
        for index, gamefile in enumerate(gamefiles):
            env_seed = args.task_seed + 1000 + index
            env = AlfWorldTextEnv(
                seed=env_seed,
                evaluation=True,
                gamefile=gamefile,
                history_length=2,
            )
            turns = []
            won = False
            try:
                observation = env.reset()
                for turn_index in range(args.max_steps):
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": observation}],
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=False,
                    )
                    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    if len(input_ids) > args.max_prompt_tokens:
                        raise ValueError(
                            f"Prompt for game {gamefile}, turn {turn_index + 1} has "
                            f"{len(input_ids)} tokens; SDA limit is {args.max_prompt_tokens}"
                        )
                    response = generate(
                        args.server_url,
                        input_ids,
                        temperature=args.temperature,
                        max_tokens=args.max_response_tokens,
                        timeout=args.request_timeout,
                        sampling_seed=request_seed(args.task_seed, args.seed, gamefile, turn_index),
                    )
                    result = env.step(response)
                    turns.append(
                        {
                            "step": turn_index + 1,
                            "prompt": observation,
                            "response": response,
                            "action": result.action,
                            "is_action_valid": result.is_action_valid,
                            "reward": result.reward,
                            "won": result.won,
                        }
                    )
                    observation = result.observation
                    won = result.won
                    if result.done:
                        break
            finally:
                env.close()

            category = category_for(gamefile)
            relative_gamefile = Path(gamefile).relative_to(data_path).as_posix()
            record = {
                "gamefile": relative_gamefile,
                "category": category,
                "environment_seed": env_seed,
                "won": won,
                "steps": len(turns),
                "turns": turns,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()
            outcomes["overall"].append(int(won))
            outcomes[category].append(int(won))
            print(f"[{index + 1}/{len(gamefiles)}] {'won' if won else 'lost'} {relative_gamefile}", flush=True)

    summary = {
        "benchmark": "alfworld",
        "split": "valid_seen",
        "strict_eval": args.limit is None,
        "model_path": str(model_path),
        "seed": args.seed,
        "task_seed": args.task_seed,
        "eval_seed_mode": "per_task_turn",
        "seed_derivation_version": 1,
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_steps": args.max_steps,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_response_tokens": args.max_response_tokens,
        "metrics": {
            name: {"total": len(values), "won": sum(values), "success_rate": sum(values) / len(values)}
            for name, values in outcomes.items()
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved {rollout_path} and {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Converted Hugging Face model directory")
    parser.add_argument("--alfworld-data", required=True, help="ALFWorld data directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-response-tokens", type=int, default=512)
    parser.add_argument("--expected-games", type=int, default=140)
    parser.add_argument("--limit", type=int, default=None, help="First N games for a smoke run")
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--check-only", action="store_true", help="Check model, assets and dependencies without running")
    args = parser.parse_args()
    if args.max_steps <= 0 or args.max_prompt_tokens <= 0 or args.max_response_tokens <= 0:
        parser.error("step and token limits must be positive")
    if args.temperature <= 0 or args.request_timeout <= 0 or args.expected_games <= 0:
        parser.error("temperature, request timeout and expected game count must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
