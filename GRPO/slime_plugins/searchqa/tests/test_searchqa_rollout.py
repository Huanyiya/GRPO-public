"""CPU checks for SDA SearchQA data, environment and slime rollout wiring."""

import asyncio
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from agent_system.searchqa.environment import SearchQAEnvironment, project_action
from agent_system.searchqa.retriever import SearchRetriever
from slime.utils.types import Sample
from slime_plugins.searchqa.data import SearchQADataSource
from slime_plugins.searchqa import rollout
from slime_plugins.searchqa.rewards import normalize_turn_rewards


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode()


def test_data_source_reads_sda_env_kwargs_and_keeps_question_groups(tmp_path, monkeypatch):
    row = {
        "prompt": [{"role": "system", "content": "generic"},
                   {"role": "user", "content": "Where is the Eiffel Tower?"}],
        "env_kwargs": {
            "question": "Where is the Eiffel Tower?",
            "ground_truth": {"target": ["Paris"]},
            "data_source": "searchR1_nq",
        },
    }
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    monkeypatch.setattr("slime.rollout.data_source.load_tokenizer", lambda *a, **k: FakeTokenizer())
    monkeypatch.setattr("slime.rollout.data_source.load_processor", lambda *a, **k: None)
    args = SimpleNamespace(
        rollout_global_dataset=True, prompt_data=str(path), input_key="prompt",
        metadata_key="env_kwargs", apply_chat_template=False,
        hf_checkpoint="unused", dump_details=None, rollout_max_prompt_len=4096,
        multimodal_keys=None, label_key=None, tool_key=None,
        apply_chat_template_kwargs={}, rollout_seed=0, rollout_shuffle=True,
        buffer_filter_path=None, n_samples_per_prompt=8,
    )
    source = SearchQADataSource(args)
    group = source.get_samples(1)[0]
    assert len(group) == 8
    assert {sample.group_index for sample in group} == {0}
    assert len({sample.index for sample in group}) == 8
    assert all(sample.prompt == row["env_kwargs"]["question"] for sample in group)
    assert all(sample.metadata["ground_truth"]["target"] == ["Paris"] for sample in group)


def test_projection_and_four_turn_limit(monkeypatch):
    queries = []

    async def fake_search(self, query):
        queries.append(query)
        return json.dumps({"result": "Doc 1: Paris"})

    monkeypatch.setattr(SearchRetriever, "search", fake_search)
    env = SearchQAEnvironment(
        "Where is the Eiffel Tower?", {"target": ["Paris"]}, "searchR1_nq",
        search_url="http://unused/retrieve",
    )
    assert "Your question: Where is the Eiffel Tower?" in env.prompt()
    assert project_action("<think>x</think><search> France </search><answer>Paris</answer>") == (
        "<search>France</search>", False,
    )

    async def run():
        for index in range(3):
            result = await env.step(f"<think>x</think><search>query {index}</search>")
            assert not result.done
            assert "<information>" in result.observation
        prompt = env.prompt()
        assert "Step 1:<search>query 0</search>" in prompt
        assert "Step 3:<search>query 2</search>" in prompt
        final = await env.step("<think>x</think><answer>the Paris</answer>")
        assert final.done and final.reward == 1.0
        assert not final.tool_calling

    asyncio.run(run())
    asyncio.run(env.close())
    assert queries == ["query 0", "query 1", "query 2"]

    timeout_env = SearchQAEnvironment(
        "Where is the Eiffel Tower?", {"target": ["Paris"]}, "searchR1_nq",
        search_url="http://unused/retrieve",
    )

    async def exhaust_steps():
        results = [await timeout_env.step("<search>keep searching</search>") for _ in range(4)]
        assert [result.done for result in results] == [False, False, False, True]
        assert results[-1].reward == 0.0
        await timeout_env.close()

    asyncio.run(exhaust_steps())


def test_retriever_adapts_batch_and_single_query_apis(monkeypatch):
    payloads = []

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            return {"result": [[{"document": {"contents": "Title\nBody"}, "score": 0.9}]]}

    class FakeSession:
        def post(self, url, *, json, timeout):
            payloads.append(json)
            return FakeResponse()

        async def close(self):
            return None

    monkeypatch.setattr("agent_system.searchqa.retriever.aiohttp.ClientSession", FakeSession)

    async def run():
        for style in ("batch", "single"):
            client = SearchRetriever("http://unused/retrieve", api_style=style)
            try:
                result = await client.search("Eiffel Tower")
                assert json.loads(result) == {"result": "Doc 1: Title\nBody\n"}
            finally:
                await client.close()

    asyncio.run(run())
    assert payloads == [
        {"queries": ["Eiffel Tower"], "topk": 3, "return_scores": True},
        {"query": "Eiffel Tower", "topk": 3, "return_scores": True},
    ]


def test_rollout_emits_one_sample_per_turn_with_left_truncation(monkeypatch):
    monkeypatch.setattr(rollout, "GenerateState", lambda args: SimpleNamespace(tokenizer=FakeTokenizer()))
    async def fake_search(self, query):
        return json.dumps({"result": "Doc 1: Paris"})

    monkeypatch.setattr(SearchRetriever, "search", fake_search)
    responses = ["<think>x</think><search>Eiffel Tower</search>",
                 "<think>x</think><answer>Paris</answer>"]

    async def fake_post(url, payload, headers=None):
        response = responses.pop(0)
        ids = list(response.encode())
        return {
            "text": response,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [(-0.1, token) for token in ids],
            },
        }

    monkeypatch.setattr(rollout, "post", fake_post)
    monkeypatch.setenv("SEARCH_URL", "http://unused/retrieve")
    args = SimpleNamespace(
        partial_rollout=False, searchqa_max_steps=4, searchqa_history_length=4,
        searchqa_invalid_action_coef=0.01, apply_chat_template_kwargs={"enable_thinking": False},
        rollout_max_prompt_len=4096, rollout_max_response_len=512,
        sglang_router_ip="127.0.0.1", sglang_router_port=30000,
        use_rollout_routing_replay=False, rollout_top_p=1.0,
    )
    sample = Sample(
        index=7, group_index=2,
        metadata={"question": "background " * 500 + "Where is the Eiffel Tower?",
                  "ground_truth": {"target": ["Paris"]}, "data_source": "searchR1_nq"},
    )
    turns = asyncio.run(rollout.generate(args, sample, {"max_new_tokens": 512}))
    assert [turn.index for turn in turns] == [28, 29]
    assert [turn.rollout_id for turn in turns] == [7, 7]
    assert [turn.group_index for turn in turns] == [2, 2]
    assert [turn.metadata["trajectory_index"] for turn in turns] == [7, 7]
    assert [turn.metadata["turn_step"] for turn in turns] == [0, 1]
    assert [turn.reward for turn in turns] == [1.0, 1.0]
    assert "<information>" in turns[1].prompt
    assert len(turns[0].tokens) - turns[0].response_length == 4096
    assert all(len(turn.tokens) - turn.response_length <= 4096 for turn in turns)
    assert turns[0].tokens[:-turns[0].response_length] == list(turns[0].prompt.encode())[-4096:]
    assert not responses


def test_cross_turn_reward_normalization_preserves_raw_rewards():
    # Unequal episode lengths must not turn this into trajectory-only grouping.
    samples = [
        Sample(group_index=4, rollout_id=10, reward=1.0, metadata={"episode_won": True}),
        Sample(group_index=4, rollout_id=10, reward=0.99, metadata={"episode_won": True}),
        Sample(group_index=4, rollout_id=11, reward=0.0),
        Sample(group_index=5, rollout_id=12, reward=0.0),
        Sample(group_index=5, rollout_id=13, reward=1.0, metadata={"episode_won": True}),
    ]
    args = SimpleNamespace(
        reward_key=None, rewards_normalization=True, grpo_std_normalization=True,
    )
    raw, advantages = normalize_turn_rewards(args, samples)
    assert raw == [1.0, 0.99, 0.0, 0.0, 1.0]
    for positions in ([0, 1, 2], [3, 4]):
        values = torch.tensor([raw[position] for position in positions], dtype=torch.float32)
        expected = (values - values.mean()) / (values.std(unbiased=True) + 1e-6)
        assert [advantages[position] for position in positions] == pytest.approx(expected.tolist())

    args.rewards_normalization = False
    assert normalize_turn_rewards(args, samples) == (raw, raw)


def test_all_failed_group_does_not_reward_short_wrong_answers():
    samples = [
        Sample(group_index=0, rollout_id=0, reward=-0.01, metadata={"episode_won": False}),
        Sample(group_index=0, rollout_id=1, reward=0.0, metadata={"episode_won": False}),
    ]
    args = SimpleNamespace(reward_key=None, rewards_normalization=True, grpo_std_normalization=True)
    raw, advantages = normalize_turn_rewards(args, samples)
    assert raw == [-0.01, 0.0]
    assert advantages == [0.0, 0.0]
