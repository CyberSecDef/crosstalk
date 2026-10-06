import json
from typing import Protocol

import httpx

from crosstalk.artifacts import RunStore, digest, write_json
from crosstalk.config import api_key
from crosstalk.http import post_json


class SearchClient(Protocol):
    def search(self, query: str, limit: int = 8) -> list[str]: ...


class TavilySearch:
    def __init__(self, store: RunStore, client: httpx.Client | None = None):
        self.store = store
        self.client = client or httpx.Client(timeout=30, trust_env=False)

    def search(self, query: str, limit: int = 8) -> list[str]:
        # A search provider has separate credit pricing; require an explicit budget policy.
        if self.store.budget is not None:
            raise ValueError(
                "Tavily credit pricing is not configured; use --sources-file for this budgeted run"
            )
        cache = self.store.cache_path("search", digest({"query": query, "limit": limit}))
        if cache.exists():
            return json.loads(cache.read_text())
        result = post_json(
            self.client,
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key('tavily')}"},
            payload={"query": query, "max_results": limit, "search_depth": "basic"},
            store=self.store,
            provider="tavily",
            model="search",
            task="search",
        )
        urls = [entry["url"] for entry in result["results"]]
        write_json(cache, urls)
        return urls
