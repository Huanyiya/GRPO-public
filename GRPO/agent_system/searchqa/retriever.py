# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team. Licensed under the Apache License, Version 2.0.
# Adapted from SDA's SearchToolGroup response formatting.
"""SDA-style search observations over the local Search-R1 retriever API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from weakref import WeakKeyDictionary

import aiohttp


_search_semaphores: WeakKeyDictionary = WeakKeyDictionary()
_client_logger: logging.Logger | None = None


def _timing_logger() -> logging.Logger | None:
    global _client_logger
    log_dir = os.environ.get("SEARCH_CLIENT_LOG_DIR")
    if not log_dir:
        return None
    if _client_logger is None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"searchqa.client.{os.getpid()}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(Path(log_dir) / f"retriever-client-{os.getpid()}.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        _client_logger = logger
    return _client_logger


def _search_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _search_semaphores.get(loop)
    if semaphore is None:
        limit = int(os.environ.get("SEARCH_MAX_CONCURRENCY", "64"))
        if limit < 1:
            raise ValueError("SEARCH_MAX_CONCURRENCY must be positive")
        semaphore = asyncio.Semaphore(limit)
        _search_semaphores[loop] = semaphore
    return semaphore


class SearchRetriever:
    def __init__(self, url: str, *, topk: int = 3, timeout: int = 60, api_style: str = "batch"):
        if api_style not in {"batch", "single"}:
            raise ValueError("Search API style must be 'batch' (slime) or 'single' (SDA)")
        self.url = url
        self.topk = topk
        self.timeout = timeout
        self.api_style = api_style
        self.session: aiohttp.ClientSession | None = None

    async def search(self, query: str | None) -> str:
        if query is None:
            return ""
        query = query.strip()
        payload = {"topk": self.topk, "return_scores": True}
        if self.api_style == "batch":
            payload["queries"] = [query]
        else:
            payload["query"] = query

        started = time.perf_counter()
        request_started = None
        outcome = "ok"
        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()
            async with _search_semaphore():
                request_started = time.perf_counter()
                async with self.session.post(
                    self.url, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response.raise_for_status()
                    results = (await response.json())["result"]
            documents = results[0]
            passages = "".join(
                f"Doc {index}: {item['document']['contents'].strip()}\n"
                for index, item in enumerate(documents, start=1)
            )
            return json.dumps({"result": passages if documents else "No search results found."})
        except asyncio.CancelledError:
            outcome = "CancelledError"
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
            outcome = type(exc).__name__
            return json.dumps({"result": f"Search error: {type(exc).__name__}: {exc!r}"})
        finally:
            try:
                logger = _timing_logger()
                if logger is not None:
                    ended = time.perf_counter()
                    logger.info(
                        "search outcome=%s wait_s=%.3f http_s=%.3f total_s=%.3f",
                        outcome,
                        (request_started if request_started is not None else ended) - started,
                        ended - request_started if request_started is not None else 0.0,
                        ended - started,
                    )
            except OSError:
                # A diagnostics filesystem failure must not change the search result.
                pass

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None
