"""Open-instruct IFEvalG reward for the instruction_following dataset."""

import ast
import json

from open_instruct.IFEvalG import instructions_registry
from slime.utils.types import Sample


def _remove_thinking_section(prediction: str) -> str:
    # Match open_instruct.ground_truth_utils.remove_thinking_section.
    prediction = prediction.replace("<|assistant|>", "").strip()
    prediction = prediction.split("</think>")[-1]
    return prediction.replace("<answer>", "").replace("</answer>", "").strip()


def _verification_score(prediction: str, label: str) -> float:
    constraints = ast.literal_eval(label)[0]
    if isinstance(constraints, str):
        constraints = json.loads(constraints)

    answer = _remove_thinking_section(prediction)
    if not prediction or not answer:
        return 0.0

    instruction_keys = constraints["instruction_id"]
    args_list = constraints["kwargs"]
    scores = []
    for instruction_key, instruction_args in zip(instruction_keys, args_list):
        instruction_args = {} if instruction_args is None else instruction_args
        instruction_args = {key: value for key, value in instruction_args.items() if value is not None}
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_key]
        instruction = instruction_cls(instruction_key)
        instruction.build_description(**instruction_args)
        scores.append(float(bool(prediction.strip() and instruction.check_following(answer))))
    return sum(scores) / max(len(scores), 1)


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Return 10 times the constraint pass rate, or zero for a non-stop finish."""
    del args, kwargs
    if sample.status != Sample.Status.COMPLETED:
        return 0.0
    return 10.0 * _verification_score(sample.response, sample.label)
