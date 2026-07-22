# Modified from the original work.
"""arXiv search via the public Atom API.

No API key required. Returns {arxiv_id, title, summary, authors, published, pdf_url, abs_url, categories}.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from ...file_lock import acquire_exclusive_file_lock, release_file_lock
from ..base import ToolCtx, ToolResult
from .search_cache import cached_payload, normalized_query, read_search_cache, write_search_cache

ARXIV_URL = "https://export.arxiv.org/api/query"
DEFAULT_MAX_RESULTS = 30
ARXIV_API_MAX_RESULTS = 100
ARXIV_FILTER_VERSION = "lexical_dynamic_v3"
_ARXIV_MIN_REQUEST_INTERVAL_SECONDS = 3.2
_ARXIV_MAX_ATTEMPTS = 3
_ARXIV_DEFAULT_429_RETRY_AFTER_SECONDS = 60.0
_ARXIV_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_ARXIV_RATE_LIMIT_FILE_ENV = "HYPOTHESIS_ENGINE_ARXIV_RATE_LIMIT_FILE"
_ARXIV_REQUEST_LOCK: asyncio.Lock | None = None
_ARXIV_LAST_REQUEST_MONOTONIC = 0.0
_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivSearchTool:
    name = "arxiv_search"
    description = (
        "Search arXiv (physics, math, CS, quantitative biology, statistics, EE, econ). Returns up "
        "to N records with arxiv_id, title, summary, authors, year, pdf_url, abs_url, categories."
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
            "sort": {
                "type": "string",
                "enum": ["relevance", "submitted", "lastUpdated"],
                "default": "relevance",
            },
        },
        "required": ["query"],
    }

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        t0 = time.monotonic()
        query = args.get("query", "").strip()
        n = min(30, max(1, int(args.get("max_results") or DEFAULT_MAX_RESULTS)))
        sort = args.get("sort", "relevance")
        if not query:
            return ToolResult(is_error=True, error_message="empty query")

        sort_param = {
            "relevance": "relevance",
            "submitted": "submittedDate",
            "lastUpdated": "lastUpdatedDate",
        }[sort]
        arxiv_query = _arxiv_search_query(query)
        structured_query = _is_structured_arxiv_query(query)
        cache_params = {
            "query": normalized_query(query),
            "arxiv_query": normalized_query(arxiv_query),
            "max_results": n,
            "sort": sort,
            "filter": ARXIV_FILTER_VERSION,
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
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                api_n = max(n, min(ARXIV_API_MAX_RESULTS, n * 3))
                r = await _arxiv_get_with_retry(
                    client,
                    {
                        "search_query": arxiv_query,
                        "max_results": api_n,
                        "sortBy": sort_param,
                        "sortOrder": "descending",
                    },
                )
                raw_records = await asyncio.to_thread(_parse_atom, r.text)

                if structured_query:
                    records = raw_records[:n]
                    filtered_out = max(0, len(raw_records) - len(records))
                else:
                    records, filtered_out = _filter_records_by_query(query, raw_records, limit=n)
        except httpx.HTTPError as e:
            return ToolResult(is_error=True, error_message=f"arxiv failed: {_format_httpx_error(e)}")

        payload = {
            "query": query,
            "n": len(records),
            "filtered_out": filtered_out,
            "results": records,
        }
        await write_search_cache(ctx.cfg, ctx.session_id, self.name, cache_params, payload)
        payload = await _review_search_payload(ctx, payload)
        return ToolResult(
            content=payload,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(str(payload)),
        )


async def _arxiv_get_with_retry(
    client: httpx.AsyncClient,
    params: dict[str, Any],
) -> httpx.Response:
    for attempt in range(1, _ARXIV_MAX_ATTEMPTS + 1):
        try:
            response = await _arxiv_get_once(client, params)
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
            retry_after = None
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                if status not in _ARXIV_RETRYABLE_STATUS_CODES:
                    raise
                retry_after = _retry_after_seconds(exc.response.headers.get("Retry-After"))
            if attempt >= _ARXIV_MAX_ATTEMPTS:
                raise
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                delay = (
                    retry_after
                    if retry_after is not None
                    else _ARXIV_DEFAULT_429_RETRY_AFTER_SECONDS
                )
            else:
                delay = retry_after if retry_after is not None else min(8.0, float(2 ** (attempt - 1)))
            if delay > 0:
                await asyncio.sleep(delay)
    raise RuntimeError("unreachable arXiv retry loop")


async def _arxiv_get_once(
    client: httpx.AsyncClient,
    params: dict[str, Any],
) -> httpx.Response:
    interval = max(0.0, _ARXIV_MIN_REQUEST_INTERVAL_SECONDS)
    if interval <= 0:
        return await client.get(ARXIV_URL, params=params)

    lock = _arxiv_request_lock()
    async with lock:
        limiter = await asyncio.to_thread(_acquire_arxiv_rate_limit, interval)
        try:
            return await client.get(ARXIV_URL, params=params)
        finally:
            await asyncio.to_thread(_release_arxiv_rate_limit, limiter)


def _acquire_arxiv_rate_limit(interval: float):
    path = _arxiv_rate_limit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a+", encoding="utf-8")
    try:
        acquire_exclusive_file_lock(f)
        f.seek(0)
        raw = f.read().strip()
        try:
            last_request_at = float(raw) if raw else 0.0
        except ValueError:
            last_request_at = 0.0

        now = time.time()
        delay = interval - (now - last_request_at)
        if delay > 0:
            time.sleep(delay)
            now = time.time()

        # Record the request start while still holding the lock. The lock remains
        # held during the HTTP request so arXiv sees one connection at a time.
        f.seek(0)
        f.truncate()
        f.write(f"{now:.6f}\n")
        f.flush()
        os.fsync(f.fileno())
        return f
    except Exception:
        f.close()
        raise


def _release_arxiv_rate_limit(f) -> None:
    try:
        release_file_lock(f)
    finally:
        f.close()


def _arxiv_rate_limit_path() -> Path:
    override = os.environ.get(_ARXIV_RATE_LIMIT_FILE_ENV)
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "hypothesis_engine_arxiv_rate_limit.txt"


def _arxiv_request_lock() -> asyncio.Lock:
    global _ARXIV_REQUEST_LOCK

    if _ARXIV_REQUEST_LOCK is None:
        _ARXIV_REQUEST_LOCK = asyncio.Lock()
    return _ARXIV_REQUEST_LOCK


def _format_httpx_error(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        request = exc.request
        parts = [f"HTTP {response.status_code} {response.reason_phrase}".strip()]
        if request is not None:
            parts.append(f"url={request.url}")
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            parts.append(f"Retry-After={retry_after}")
        body = response.text.strip()
        if body:
            parts.append(f"body={body[:500]}")
        return "; ".join(parts)

    message = str(exc).strip()
    if message:
        return f"{exc.__class__.__name__}: {message}"
    return exc.__class__.__name__


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    return max(0.0, parsed.timestamp() - time.time())


async def _review_search_payload(ctx: ToolCtx, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from ..rag import review_search_payload

        return await review_search_payload(ctx, payload, provider="arxiv")
    except Exception as exc:
        out = dict(payload)
        out["rag_ingest"] = {"enabled": True, "scheduled": False, "error": str(exc)[:300]}
        return out


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.-]*")
_PHRASE_RE = re.compile(r'"([^"\n]+)"')
_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "based",
    "between",
    "during",
    "for",
    "from",
    "how",
    "into",
    "low",
    "mechanism",
    "mechanisms",
    "near",
    "of",
    "on",
    "or",
    "the",
    "this",
    "through",
    "to",
    "via",
    "with",
}


_STRUCTURED_ARXIV_QUERY_RE = re.compile(
    r'(^|\s|\()(?i:all|ti|au|abs|co|jr|cat|rn|id)\s*:'
    r'|(\bANDNOT\b|\bAND\b|\bOR\b)|[()]',
)


def _is_structured_arxiv_query(query: str) -> bool:
    """Return True when the query appears to use native arXiv API syntax."""
    return bool(_STRUCTURED_ARXIV_QUERY_RE.search(query.strip()))


def _arxiv_search_query(query: str) -> str:
    """Build the query string sent to the arXiv API.

    Native arXiv query syntax is passed through unchanged. Plain natural-language
    text is converted into a broad all-field OR query so exploratory searches do
    not fail just because one specific term is absent from arXiv metadata.
    """
    q = query.strip()
    if _is_structured_arxiv_query(q):
        return q

    terms: list[str] = []
    seen: set[str] = set()
    for phrase in _PHRASE_RE.findall(q.lower()):
        phrase = " ".join(_TOKEN_RE.findall(phrase))
        if phrase and phrase not in seen:
            seen.add(phrase)
            escaped = phrase.replace('"', r'\"')
            terms.append(f'all:"{escaped}"')

    unquoted = _PHRASE_RE.sub(" ", q.lower())
    for token in _TOKEN_RE.findall(unquoted):
        if len(token) <= 1 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(f"all:{token}")

    if not terms:
        escaped = q.replace('"', r'\"')
        return f'all:"{escaped}"'

    return " OR ".join(terms)


def _filter_records_by_query(
    query: str,
    records: list[dict[str, Any]],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for idx, record in enumerate(records):
        score = _record_relevance_score(query, record)
        if score <= 0:
            continue
        out = dict(record)
        out["relevance_score"] = score
        scored.append((score, idx, out))
    scored.sort(key=lambda item: (-item[0], item[1]))
    filtered = [record for _score, _idx, record in scored[:limit]]
    return filtered, max(0, len(records) - len(filtered))


def _record_relevance_score(query: str, record: dict[str, Any]) -> int:
    phrases, query_tokens = _query_phrases_and_tokens(query)
    if not phrases and not query_tokens:
        return 1

    text = f"{record.get('title') or ''} {record.get('summary') or ''}".lower()
    text_tokens = set(_TOKEN_RE.findall(text))
    text_stems = {_stem_token(token) for token in text_tokens}
    phrase_hits = sum(1 for phrase in phrases if phrase in text)
    query_stems = {_stem_token(token) for token in query_tokens}
    token_hits = len(query_stems & text_stems)

    required_hits = 1
    if len(query_stems) >= 5:
        required_hits = 3
    elif len(query_stems) >= 3:
        required_hits = 2

    if phrase_hits == 0 and token_hits < required_hits:
        return 0
    return phrase_hits * 4 + token_hits


def _query_phrases_and_tokens(query: str) -> tuple[list[str], list[str]]:
    lowered = query.lower()
    phrases = [phrase.strip() for phrase in _PHRASE_RE.findall(lowered) if phrase.strip()]
    unquoted = _PHRASE_RE.sub(" ", lowered)
    tokens = [
        token
        for token in _TOKEN_RE.findall(unquoted)
        if len(token) > 1 and token not in _STOPWORDS
    ]
    return phrases, tokens


def _stem_token(token: str) -> str:
    token = token.lower()
    if token in {"implantation", "implanted", "implanting", "implants"}:
        return "implant"
    if token in {"irradiation", "irradiated", "irradiating"}:
        return "irradiat"
    if token in {"defects", "defective"}:
        return "defect"
    if token in {"ions"}:
        return "ion"
    if token in {"tmds"}:
        return "tmd"
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _parse_atom(xml: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml)
    out: list[dict[str, Any]] = []
    for e in root.findall("a:entry", _NS):
        id_url = (e.findtext("a:id", default="", namespaces=_NS) or "").strip()
        arxiv_id = id_url.rsplit("/", 1)[-1]
        title = (e.findtext("a:title", default="", namespaces=_NS) or "").strip()
        summary = (e.findtext("a:summary", default="", namespaces=_NS) or "").strip()
        published = (e.findtext("a:published", default="", namespaces=_NS) or "")[:10]
        authors = [
            (a.findtext("a:name", default="", namespaces=_NS) or "").strip()
            for a in e.findall("a:author", _NS)
        ]
        categories = [c.get("term", "") for c in e.findall("a:category", _NS) if c.get("term")]
        pdf_url = None
        for link in e.findall("a:link", _NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        out.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "year": published[:4] if published else None,
                "pdf_url": pdf_url,
                "abs_url": id_url,
                "categories": categories,
            }
        )
    return out
