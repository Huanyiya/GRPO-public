"""Verify that the local Search-R1 service batches concurrent HTTP requests."""

import asyncio
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace


def test_retrieval_service_batches_at_most_64_queries(tmp_path):
    server_path = Path(__file__).resolve().parents[3] / "examples/search-r1/local_dense_retriever/retrieval_server.py"
    spec = importlib.util.spec_from_file_location("searchqa_retrieval_server_test", server_path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    log_path = tmp_path / "server.log"
    handler = logging.FileHandler(log_path)
    server.logger.setLevel(logging.INFO)
    server.logger.addHandler(handler)

    batches = []

    class FakeRetriever:
        def batch_search(self, query_list, num, return_score):
            batches.append(list(query_list))
            assert len(query_list) <= 64
            assert num == 3
            assert return_score is True
            return [[{"contents": query}] for query in query_list], [[0.5] for _ in query_list]

    server.retriever = FakeRetriever()
    server.config = SimpleNamespace(retrieval_topk=3)

    async def run():
        async with server.lifespan(server.app):
            requests = [
                server.QueryRequest(queries=[f"query {index}"], return_scores=True)
                for index in range(130)
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*(server.retrieve_endpoint(request) for request in requests)), timeout=5
            )
            assert [result["result"][0][0]["document"]["contents"] for result in results] == [
                f"query {index}" for index in range(130)
            ]

    try:
        asyncio.run(run())
    finally:
        server.logger.removeHandler(handler)
        handler.close()
    assert [len(batch) for batch in batches] == [64, 64, 2]
    assert log_path.read_text().count("batch_start") == 3
    assert log_path.read_text().count("batch_end") == 3
