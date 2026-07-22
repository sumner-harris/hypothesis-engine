# Modified from the original work.
"""Web search tool.

Provider priority: Tavily (primary, dev-friendly) → Brave (fallback).
If neither key is present, the tool reports an error to the agent so it can
proceed without web evidence (rather than crash the run).
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from ..config import Config
from .base import ToolCtx, ToolResult
from .builtins.search_cache import (
    cached_payload,
    normalized_query,
    read_search_cache,
    write_search_cache,
)

_TAVILY_URL = "https://api.tavily.com/search"
_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


class WebSearchTool:
    name = "web_search"
    description = (
        "Search the public web for scientific literature, news, and reference material. "
        "Returns a list of {title, url, snippet, published_at?} results. "
        "Use when you need broad recall across the open web; for indexed databases prefer "
        "pubmed_search, arxiv_search, biorxiv_search, chemrxiv_search, or europe_pmc_search."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Free-text search query."},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30,
                "description": "Number of results to return (default 30).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        t0 = time.monotonic()
        query = args.get("query", "").strip()
        n = min(30, max(1, int(args.get("max_results") or self._cfg.web_search.max_results)))
        if not query:
            return ToolResult(is_error=True, error_message="empty query")

        tavily = self._cfg.secrets.TAVILY_API_KEY or os.environ.get("TAVILY_API_KEY")
        brave = self._cfg.secrets.BRAVE_API_KEY or os.environ.get("BRAVE_API_KEY")
        provider = self._cfg.web_search.provider.lower()
        effective_provider: str | None = None
        if provider == "tavily" and tavily:
            effective_provider = "tavily"
        elif provider == "brave" and brave:
            effective_provider = "brave"
        elif tavily:
            effective_provider = "tavily"
        elif brave:
            effective_provider = "brave"
        else:
            return ToolResult(
                is_error=True,
                error_message="no web search API key configured (TAVILY_API_KEY or BRAVE_API_KEY)",
            )

        cache_params = {
            "query": normalized_query(query),
            "max_results": n,
            "provider": effective_provider,
        }
        cached = await read_search_cache(ctx.cfg, ctx.session_id, self.name, cache_params)
        if cached is not None:
            content = cached_payload(cached)
            content = await _review_search_payload(ctx, content)
            return ToolResult(
                content=content,
                duration_ms=int((time.monotonic() - t0) * 1000),
                result_bytes=len(str(content)),
            )

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                if effective_provider == "tavily":
                    results = await self._tavily(client, tavily, query, n)
                else:
                    results = await self._brave(client, brave, query, n)
        except httpx.HTTPError as e:
            return ToolResult(is_error=True, error_message=f"web search failed: {e}")

        payload: dict[str, Any] = {"query": query, "n": len(results), "results": results}
        await write_search_cache(ctx.cfg, ctx.session_id, self.name, cache_params, payload)
        payload = await _review_search_payload(ctx, payload)
        return ToolResult(
            content=payload,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(str(payload)),
        )

    async def _tavily(
        self, client: httpx.AsyncClient, key: str, query: str, n: int
    ) -> list[dict[str, Any]]:
        r = await client.post(
            _TAVILY_URL,
            json={
                "api_key": key,
                "query": query,
                "max_results": n,
                "search_depth": "advanced",
                "include_answer": False,
            },
        )
        r.raise_for_status()
        data = r.json()
        out = []
        for hit in data.get("results", [])[:n]:
            out.append(
                {
                    "title": hit.get("title", ""),
                    "url": hit.get("url", ""),
                    "snippet": hit.get("content", ""),
                    "published_at": hit.get("published_date"),
                    "score": hit.get("score"),
                }
            )
        return out

    async def _brave(
        self, client: httpx.AsyncClient, key: str, query: str, n: int
    ) -> list[dict[str, Any]]:
        r = await client.get(
            _BRAVE_URL,
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            params={"q": query, "count": n},
        )
        r.raise_for_status()
        data = r.json()
        out = []
        for hit in data.get("web", {}).get("results", [])[:n]:
            out.append(
                {
                    "title": hit.get("title", ""),
                    "url": hit.get("url", ""),
                    "snippet": hit.get("description", ""),
                    "published_at": hit.get("age"),
                }
            )
        return out


async def _review_search_payload(ctx: ToolCtx, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from .rag import review_search_payload

        return await review_search_payload(ctx, payload, provider="web")
    except Exception as exc:
        out = dict(payload)
        out["rag_ingest"] = {"enabled": True, "scheduled": False, "error": str(exc)[:300]}
        return out
