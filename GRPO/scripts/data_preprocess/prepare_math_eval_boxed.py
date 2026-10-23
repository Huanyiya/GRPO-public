"""Convert AIME evaluation prompts from ``Answer:`` to ``\\boxed{...}``.

Example:
    python scripts/data_preprocess/prepare_math_eval_boxed.py \
        ../data/math/test.parquet ../data/math/test-boxed.parquet
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


OLD_INSTRUCTION = (
    "The last line of your response should be of the form Answer: $Answer "
    "(without quotes) where $Answer is the answer to the problem.\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)
NEW_INSTRUCTION = "Put your final answer on its own last line in the form \\boxed{answer}."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original AIME parquet file")
    parser.add_argument("destination", type=Path, help="Output parquet file with boxed prompts")
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("source and destination must be different files")

    table = pq.read_table(args.source)
    prompts = table.column("prompt").to_pylist()
    for index, messages in enumerate(prompts):
        matches = [message for message in messages if OLD_INSTRUCTION in message["content"]]
        if len(matches) != 1:
            raise ValueError(f"row {index}: expected one original Answer: instruction, found {len(matches)}")
        matches[0]["content"] = matches[0]["content"].replace(OLD_INSTRUCTION, NEW_INSTRUCTION)

    prompt_index = table.schema.get_field_index("prompt")
    boxed_prompts = pa.array(prompts, type=table.schema.field(prompt_index).type)
    table = table.set_column(prompt_index, "prompt", boxed_prompts)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.destination)
    print(f"Wrote {table.num_rows} boxed prompts to {args.destination}")


if __name__ == "__main__":
    main()
