# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team. Licensed under the Apache License, Version 2.0.
# Adapted from SDA's SearchToolGroup response formatting.
"""SDA-style search observations over the local Search-R1 retriever API."""

from __future__ import annotations

import json

import aiohttp


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

        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()
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
        except (aiohttp.ClientError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
            return json.dumps({"result": f"Search error: {exc}"})

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None
