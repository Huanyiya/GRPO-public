# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team. Licensed under the Apache License, Version 2.0.
# Adapted from SDA's search projection, memory manager and SearchEnv.
"""Minimal SDA-compatible SearchQA agent environment for slime rollouts."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any

from .prompts import SEARCH_TEMPLATE, SEARCH_TEMPLATE_NO_HIS
from .retriever import SearchRetriever


_SEARCH_BLOCK = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_SEARCH_TAG = re.compile(r"<search>", re.IGNORECASE)
_ANSWER_TAG = re.compile(r"<answer>", re.IGNORECASE)
_STRICT_SEARCH = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_FINAL_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def project_action(response: str) -> tuple[str, bool]:
    """Apply SDA's search_projection before the environment sees an action."""
    trimmed = response
    if "</search>" in trimmed:
        trimmed = trimmed.split("</search>", 1)[0] + "</search>"
    elif "</answer>" in trimmed:
        trimmed = trimmed.split("</answer>", 1)[0] + "</answer>"

    search = _SEARCH_BLOCK.search(trimmed)
    if search:
        action = f"<search>{search.group(1).strip()}</search>"
    else:
        answer = _ANSWER_BLOCK.search(trimmed)
        action = f"<answer>{answer.group(1).strip()}</answer>" if answer else ""

    search_count = len(_SEARCH_TAG.findall(response))
    answer_count = len(_ANSWER_TAG.findall(response))
    valid = bool(action) and not (
        (search_count and answer_count) or search_count > 1 or answer_count > 1
    )
    return action, valid


def _normalize_answer(answer: str) -> str:
    answer = answer.lower()
    answer = "".join(char for char in answer if char not in string.punctuation)
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    return " ".join(answer.split())


def _score(history: list[str], ground_truth: dict[str, Any]) -> float:
    """SDA's terminal Search-R1 exact match over the last complete answer."""
    matches = list(_FINAL_ANSWER.finditer("".join(history)))
    if not matches:
        return 0.0
    predicted = _normalize_answer(matches[-1].group(1).strip())
    targets = ground_truth["target"]
    if isinstance(targets, str):
        targets = [targets]
    return float(any(predicted == _normalize_answer(target) for target in targets))


@dataclass(frozen=True)
class SearchStep:
    observation: str
    reward: float
    done: bool
    action: str
    valid: bool
    tool_calling: bool


class SearchQAEnvironment:
    def __init__(
        self,
        question: str,
        ground_truth: dict[str, Any],
        data_source: str,
        *,
        search_url: str,
        max_steps: int = 4,
        history_length: int = 4,
        topk: int = 3,
        timeout: int = 60,
        api_style: str = "batch",
    ):
        if max_steps <= 0 or history_length < 0:
            raise ValueError("SearchQA max_steps must be positive and history_length nonnegative")
        self.question = question
        self.ground_truth = ground_truth
        self.data_source = data_source
        self.max_steps = max_steps
        self.history_length = history_length
        self.retriever = SearchRetriever(search_url, topk=topk, timeout=timeout, api_style=api_style)
        self.history: list[tuple[str, str]] = []
        self.chat_history: list[str] = []
        self.done = False

    def prompt(self) -> str:
        if not self.history or self.history_length == 0:
            return SEARCH_TEMPLATE_NO_HIS.format(task_description=self.question)
        recent = self.history[-self.history_length :]
        first = len(self.history) - len(recent) + 1
        context = "\n".join(
            f"Step {first + index}:{action} {observation}\n"
            for index, (action, observation) in enumerate(recent)
        )
        return SEARCH_TEMPLATE.format(
            task_description=self.question,
            memory_context=context,
            step_count=len(self.history),
        )

    async def step(self, response: str) -> SearchStep:
        if self.done:
            raise RuntimeError("SearchQA episode has already ended")
        action, valid = project_action(response)
        self.chat_history.append(action)
        step_count = len(self.history) + 1
        done = step_count >= self.max_steps or ("<answer>" in action and "</answer>" in action)
        self.done = done
        if done:
            reward = _score(self.chat_history, self.ground_truth)
            self.history.append((action, ""))
            return SearchStep("", reward, True, action, valid, False)

        match = _STRICT_SEARCH.search(action)
        query = match.group(1) if match else None
        result = await self.retriever.search(query)
        observation = f"\n<information>{result}</information>\n" if result else ""
        if observation:
            self.chat_history.append(observation)
        self.history.append((action, observation))
        return SearchStep(observation, 0.0, False, action, valid, True)

    async def close(self) -> None:
        await self.retriever.close()
