"""Read SDA's processed Search-R1 Parquet rows as slime prompt groups."""

from collections.abc import Mapping

from slime.rollout.data_source import RolloutDataSourceWithBuffer


class SearchQADataSource(RolloutDataSourceWithBuffer):
    """Use slime's Parquet reader, grouping and checkpointed dataset cursor."""

    def __init__(self, args):
        if not args.rollout_global_dataset or not args.prompt_data:
            raise ValueError("SearchQA requires --prompt-data and the global rollout dataset")
        if args.input_key != "prompt" or args.metadata_key != "env_kwargs":
            raise ValueError("SearchQA requires --input-key prompt --metadata-key env_kwargs")
        if args.apply_chat_template:
            raise ValueError("SearchQA builds SDA's per-turn chat prompt in the rollout")
        super().__init__(args)
        if not self.dataset or not self.dataset.origin_samples:
            raise ValueError("SearchQA training Parquet contains no samples")

        for sample in self.dataset.origin_samples:
            fields = sample.metadata
            if not isinstance(fields, Mapping):
                raise ValueError("SearchQA Parquet env_kwargs must be a mapping")
            question = fields.get("question")
            ground_truth = fields.get("ground_truth")
            if not isinstance(question, str) or not question:
                raise ValueError("SearchQA Parquet env_kwargs.question must be nonempty")
            if not isinstance(ground_truth, Mapping) or "target" not in ground_truth:
                raise ValueError("SearchQA Parquet env_kwargs.ground_truth.target is required")
            sample.prompt = question
            sample.label = dict(ground_truth)
            sample.metadata = {
                "question": question,
                "ground_truth": dict(ground_truth),
                "data_source": str(fields.get("data_source", "unknown")),
            }
