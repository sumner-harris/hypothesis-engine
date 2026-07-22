# Modified from the original work.
"""Europe PMC REST search.

Covers PubMed + life-sciences preprints (incl. bioRxiv, medRxiv) and many full-text records.
No API key required.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..base import ToolCtx, ToolResult
from .search_cache import cached_payload, normalized_query, read_search_cache, write_search_cache

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
DEFAULT_MAX_RESULTS = 30


class EuropePMCSearchTool:
    name = "europe_pmc_search"
    description = (
        "Search Europe PMC (PubMed + bioRxiv/medRxiv + full-text where available). Returns "
        "{id, source, title, abstract, authors, journal, year, doi, url, is_open_access}."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30,
                "default": DEFAULT_MAX_RESULTS,
            },
            "open_access_only": {"type": "boolean", "default": False},
        },
        "required": ["query"],
    }

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        t0 = time.monotonic()
        query = args.get("query", "").strip()
        n = min(30, max(1, int(args.get("max_results") or DEFAULT_MAX_RESULTS)))
        oa = bool(args.get("open_access_only"))
        if not query:
            return ToolResult(is_error=True, error_message="empty query")
        q = f"({query}) AND OPEN_ACCESS:Y" if oa else query
        cache_params = {
            "query": normalized_query(query),
            "max_results": n,
            "open_access_only": oa,
        }
        cached = await read_search_cache(ctx.cfg, ctx.session_id, self.name, cache_params)
        if cached is not None:
            content = await _review_search_payload(ctx, cached_payload(cached))
            return ToolResult(
                content=content,
                duration_ms=int((time.monotonic() - t0) * 1000),
                result_bytes=len(str(content)),
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(
                    EUROPE_PMC_URL,
                    params={"query": q, "format": "json", "pageSize": n, "resultType": "core"},
                )
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            return ToolResult(is_error=True, error_message=f"europe_pmc failed: {e}")

        results = []
        for hit in data.get("resultList", {}).get("result", [])[:n]:
            pmid = hit.get("pmid")
            doi = hit.get("doi")
            results.append(
                {
                    "id": hit.get("id"),
                    "source": hit.get("source"),
                    "title": hit.get("title"),
                    "abstract": hit.get("abstractText", ""),
                    "authors": hit.get("authorString", ""),
                    "journal": hit.get("journalTitle"),
                    "year": hit.get("pubYear"),
                    "doi": doi,
                    "url": (
                        f"https://europepmc.org/article/{hit.get('source', 'MED')}/{hit.get('id', '')}"
                        if hit.get("id")
                        else None
                    ),
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                    "is_open_access": hit.get("isOpenAccess") == "Y",
                }
            )
        payload = {"query": q, "n": len(results), "results": results}
        await write_search_cache(ctx.cfg, ctx.session_id, self.name, cache_params, payload)
        payload = await _review_search_payload(ctx, payload)
        return ToolResult(
            content=payload,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(str(payload)),
        )


async def _review_search_payload(ctx: ToolCtx, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from ..rag import review_search_payload

        return await review_search_payload(ctx, payload, provider="europe_pmc")
    except Exception as exc:
        out = dict(payload)
        out["rag_ingest"] = {"enabled": True, "scheduled": False, "error": str(exc)[:300]}
        return out
