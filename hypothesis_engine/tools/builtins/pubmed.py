# Modified from the original work.
"""PubMed search via the NCBI E-utilities API.

No API key required; supplying NCBI_API_KEY raises the rate limit from 3 to 10 req/s.
Returns light records (pmid, title, abstract, journal, authors, year, doi, url).
Use web_fetch to pull full text when available.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from ...config import Config
from ..base import ToolCtx, ToolResult
from .search_cache import cached_payload, normalized_query, read_search_cache, write_search_cache

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DEFAULT_MAX_RESULTS = 30


class PubmedSearchTool:
    name = "pubmed_search"
    description = (
        "Search PubMed (biomedical literature). Returns up to N records with pmid, title, "
        "abstract, journal, authors, year, doi, url. Use for biomedical queries; for general "
        "physics/CS, use arxiv_search instead."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "PubMed query (E-utilities syntax allowed).",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30,
                "default": DEFAULT_MAX_RESULTS,
                "description": "Number of records (default 30).",
            },
            "sort": {
                "type": "string",
                "enum": ["relevance", "pub_date"],
                "default": "relevance",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        t0 = time.monotonic()
        query = args.get("query", "").strip()
        n = min(30, max(1, int(args.get("max_results") or DEFAULT_MAX_RESULTS)))
        sort = args.get("sort", "relevance")
        if not query:
            return ToolResult(is_error=True, error_message="empty query")

        cache_params = {"query": normalized_query(query), "max_results": n, "sort": sort}
        cached = await read_search_cache(ctx.cfg, ctx.session_id, self.name, cache_params)
        if cached is not None:
            content = await _review_search_payload(ctx, cached_payload(cached))
            return ToolResult(
                content=content,
                duration_ms=int((time.monotonic() - t0) * 1000),
                result_bytes=len(str(content)),
            )

        api_key = self._cfg.secrets.NCBI_API_KEY or os.environ.get("NCBI_API_KEY") or None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                pmids = await self._esearch(client, query, n, sort, api_key)
                if not pmids:
                    payload = {"query": query, "n": 0, "results": []}
                    await write_search_cache(
                        ctx.cfg, ctx.session_id, self.name, cache_params, payload
                    )
                    payload = await _review_search_payload(ctx, payload)
                    return ToolResult(
                        content=payload,
                        duration_ms=int((time.monotonic() - t0) * 1000),
                    )
                records = await self._efetch(client, pmids, api_key)
        except httpx.HTTPError as e:
            return ToolResult(is_error=True, error_message=f"pubmed failed: {e}")

        payload = {"query": query, "n": len(records), "results": records}
        await write_search_cache(ctx.cfg, ctx.session_id, self.name, cache_params, payload)
        payload = await _review_search_payload(ctx, payload)
        return ToolResult(
            content=payload,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(str(payload)),
        )

    async def _esearch(
        self, client: httpx.AsyncClient, query: str, n: int, sort: str, api_key: str | None
    ) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": n,
            "sort": "relevance" if sort == "relevance" else "pub+date",
        }
        if api_key:
            params["api_key"] = api_key
        r = await client.get(ESEARCH, params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [])

    async def _efetch(
        self, client: httpx.AsyncClient, pmids: list[str], api_key: str | None
    ) -> list[dict[str, Any]]:
        params = {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"}
        if api_key:
            params["api_key"] = api_key
        r = await client.get(EFETCH, params=params)
        r.raise_for_status()
        return await asyncio.to_thread(_parse_pubmed_xml, r.text)


def _parse_pubmed_xml(xml: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml)
    out: list[dict[str, Any]] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = (art.findtext(".//ArticleTitle") or "").strip()
        journal = (art.findtext(".//Journal/Title") or "").strip()
        year_node = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""
        # abstract may have multiple AbstractText elements
        abstract_parts: list[str] = []
        for at in art.findall(".//Abstract/AbstractText"):
            label = at.get("Label")
            txt = "".join(at.itertext()).strip()
            abstract_parts.append(f"{label}: {txt}" if label else txt)
        abstract = "\n\n".join(p for p in abstract_parts if p)
        authors = []
        for au in art.findall(".//Author")[:8]:
            last = au.findtext("LastName") or ""
            init = au.findtext("Initials") or ""
            if last:
                authors.append(f"{last} {init}".strip())
        doi = None
        for aid in art.findall(".//ArticleId"):
            if (aid.get("IdType") or "").lower() == "doi":
                doi = aid.text
                break
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        out.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "authors": authors,
                "year": year_node[:4] if year_node else None,
                "doi": doi,
                "url": url,
            }
        )
    return out


async def _review_search_payload(ctx: ToolCtx, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from ..rag import review_search_payload

        return await review_search_payload(ctx, payload, provider="pubmed")
    except Exception as exc:
        out = dict(payload)
        out["rag_ingest"] = {"enabled": True, "scheduled": False, "error": str(exc)[:300]}
        return out
