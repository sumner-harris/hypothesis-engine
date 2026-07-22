"""ChemRxiv search via the public Open Engage API.

No API key required. Returns records compatible with the arXiv search tool:
{arxiv_id, chemrxiv_id, title, summary, authors, year, pdf_url, abs_url, categories}.

The arxiv_id field is intentionally populated with the ChemRxiv item ID as a
compatibility shim for downstream code that expects an arXiv-style identifier.
Prefer chemrxiv_id or source_id for new provider-aware code.
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

import httpx

from ...file_lock import acquire_exclusive_file_lock, release_file_lock
from ..base import ToolCtx, ToolResult
from .search_cache import cached_payload, normalized_query, read_search_cache, write_search_cache

CHEMRXIV_URL = "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items"
CHEMRXIV_ITEM_URL = "https://chemrxiv.org/engage/chemrxiv/article-details"
CHEMRXIV_CAMBRIDGE_DOI_URL = "https://www.cambridge.org/engage/coe/public-api/v1/items/doi"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
DEFAULT_MAX_RESULTS = 30
CHEMRXIV_API_MAX_RESULTS = 50
CHEMRXIV_FILTER_VERSION = "lexical_dynamic_v2"

# ChemRxiv allows 2 requests/second. Keep a small file-backed limiter so
# parallel workers and accidental duplicate server processes share the limit.
_CHEMRXIV_MIN_REQUEST_INTERVAL_SECONDS = 0.5
_CHEMRXIV_MAX_ATTEMPTS = 3
_CHEMRXIV_DEFAULT_429_RETRY_AFTER_SECONDS = 10.0
_CHEMRXIV_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_CHEMRXIV_RATE_LIMIT_FILE_ENV = "HYPOTHESIS_ENGINE_CHEMRXIV_RATE_LIMIT_FILE"
_CHEMRXIV_REQUEST_LOCK: asyncio.Lock | None = None
_CHEMRXIV_LAST_REQUEST_MONOTONIC = 0.0
_CHEMRXIV_USER_AGENT = "chemrxiv-search-tool/0.1"
_CROSSREF_USER_AGENT = "hypothesis-engine chemrxiv-crossref-fallback/0.1"

# Crossref's public REST API asks clients using query/filter requests to stay
# at or below one request per second. Keep this separate from ChemRxiv/Open
# Engage because the fallback hits api.crossref.org, not chemrxiv.org.
_CROSSREF_MIN_REQUEST_INTERVAL_SECONDS = 1.0
_CROSSREF_MAX_ATTEMPTS = 3
_CROSSREF_DEFAULT_429_RETRY_AFTER_SECONDS = 10.0
_CROSSREF_RATE_LIMIT_FILE_ENV = "HYPOTHESIS_ENGINE_CROSSREF_RATE_LIMIT_FILE"
_CROSSREF_REQUEST_LOCK: asyncio.Lock | None = None


class ChemrxivSearchTool:
    name = "chemrxiv_search"
    description = (
        "Search ChemRxiv chemistry preprints. Returns up to N records with "
        "arxiv-compatible metadata fields: arxiv_id, chemrxiv_id, title, summary, "
        "authors, year, pdf_url, abs_url, categories, doi, license."
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
            "date_from": {
                "type": "string",
                "description": "Optional ISO date filter, e.g. 2023-01-01.",
            },
            "date_to": {
                "type": "string",
                "description": "Optional ISO date filter, e.g. 2026-12-31.",
            },
            "license": {
                "type": "string",
                "description": "Optional ChemRxiv license filter, e.g. CC BY 4.0.",
            },
            "category_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional ChemRxiv/Open Engage category IDs.",
            },
        },
        "required": ["query"],
    }

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        t0 = time.monotonic()
        query = args.get("query", "").strip()
        n = min(30, max(1, int(args.get("max_results") or DEFAULT_MAX_RESULTS)))
        sort = args.get("sort", "relevance")
        date_from = _clean_optional_str(args.get("date_from"))
        date_to = _clean_optional_str(args.get("date_to"))
        license_filter = _clean_optional_str(args.get("license"))
        category_ids = _clean_str_list(args.get("category_ids"))

        if not query:
            return ToolResult(is_error=True, error_message="empty query")

        sort_param = {
            "relevance": "RELEVANT_DESC",
            "submitted": "PUBLISHED_DATE_DESC",
            # ChemRxiv/Open Engage does not appear to expose an updated-date sort
            # in the public wrapper, so keep schema compatibility and fall back
            # to published-date descending.
            "lastUpdated": "PUBLISHED_DATE_DESC",
        }[sort]

        chemrxiv_query = _chemrxiv_search_query(query)
        cache_params = {
            "query": normalized_query(query),
            "chemrxiv_query": normalized_query(chemrxiv_query),
            "max_results": n,
            "sort": sort,
            "date_from": date_from or "",
            "date_to": date_to or "",
            "license": normalized_query(license_filter or ""),
            "category_ids": ",".join(sorted(category_ids)),
            "filter": CHEMRXIV_FILTER_VERSION,
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
                api_n = max(n, min(CHEMRXIV_API_MAX_RESULTS, n * 3))
                params = _chemrxiv_params(
                    term=chemrxiv_query,
                    limit=api_n,
                    skip=0,
                    sort=sort_param,
                    date_from=date_from,
                    date_to=date_to,
                    license_filter=license_filter,
                    category_ids=category_ids,
                )
                try:
                    r = await _chemrxiv_get_with_retry(client, params)
                    raw_records = await asyncio.to_thread(_parse_chemrxiv_items, r.json())
                    records, filtered_out = _filter_records_by_query(query, raw_records, limit=n)
                    fallback = None
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 403:
                        raise
                    raw_records = await _crossref_chemrxiv_records(
                        client,
                        query=query,
                        limit=max(n, min(CHEMRXIV_API_MAX_RESULTS, n * 5)),
                        date_from=date_from,
                        date_to=date_to,
                    )
                    enrich_limit = min(len(raw_records), max(n * 3, n), 20)
                    enriched_head = await _enrich_with_cambridge_doi_items(
                        client, raw_records[:enrich_limit]
                    )
                    raw_records = [*enriched_head, *raw_records[enrich_limit:]]
                    records, filtered_out = _filter_records_by_query(
                        query,
                        raw_records,
                        limit=n,
                    )
                    if not records and raw_records:
                        records = raw_records[:n]
                        filtered_out = max(0, len(raw_records) - len(records))
                    fallback = "crossref_cambridge_doi"
        except httpx.HTTPError as e:
            return ToolResult(is_error=True, error_message=f"chemrxiv failed: {_format_httpx_error(e)}")
        except ValueError as e:
            return ToolResult(is_error=True, error_message=f"chemrxiv parse failed: {e}")

        payload = {
            "query": query,
            "n": len(records),
            "filtered_out": filtered_out,
            "results": records,
        }
        if fallback:
            payload["fallback"] = fallback
        await write_search_cache(ctx.cfg, ctx.session_id, self.name, cache_params, payload)
        payload = await _review_search_payload(ctx, payload)
        return ToolResult(
            content=payload,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(str(payload)),
        )


def _chemrxiv_params(
    *,
    term: str,
    limit: int,
    skip: int,
    sort: str,
    date_from: str | None = None,
    date_to: str | None = None,
    license_filter: str | None = None,
    category_ids: list[str] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "term": term,
        "limit": min(max(1, limit), CHEMRXIV_API_MAX_RESULTS),
        "skip": max(0, skip),
        "sort": sort,
    }

    if date_from:
        params["searchDateFrom"] = _as_open_engage_date(date_from)
    if date_to:
        params["searchDateTo"] = _as_open_engage_date(date_to)
    if license_filter:
        params["searchLicense"] = license_filter
    if category_ids:
        params["categoryIds"] = ",".join(category_ids)

    return params


async def _chemrxiv_get_with_retry(
    client: httpx.AsyncClient,
    params: dict[str, Any],
) -> httpx.Response:
    headers = {"User-Agent": _CHEMRXIV_USER_AGENT}

    for attempt in range(1, _CHEMRXIV_MAX_ATTEMPTS + 1):
        try:
            response = await _chemrxiv_get_once(client, params, headers=headers)
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
            retry_after = None
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                if status not in _CHEMRXIV_RETRYABLE_STATUS_CODES:
                    raise
                retry_after = _retry_after_seconds(exc.response.headers.get("Retry-After"))
            if attempt >= _CHEMRXIV_MAX_ATTEMPTS:
                raise
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                delay = (
                    retry_after
                    if retry_after is not None
                    else _CHEMRXIV_DEFAULT_429_RETRY_AFTER_SECONDS
                )
            else:
                delay = retry_after if retry_after is not None else min(8.0, float(2 ** (attempt - 1)))
            if delay > 0:
                await asyncio.sleep(delay)

    raise RuntimeError("unreachable ChemRxiv retry loop")


async def _chemrxiv_get_once(
    client: httpx.AsyncClient,
    params: dict[str, Any],
    *,
    headers: dict[str, str],
) -> httpx.Response:
    interval = max(0.0, _CHEMRXIV_MIN_REQUEST_INTERVAL_SECONDS)
    if interval <= 0:
        return await client.get(CHEMRXIV_URL, params=params, headers=headers)

    lock = _chemrxiv_request_lock()
    async with lock:
        limiter = await asyncio.to_thread(_acquire_chemrxiv_rate_limit, interval)
        try:
            return await client.get(CHEMRXIV_URL, params=params, headers=headers)
        finally:
            await asyncio.to_thread(_release_chemrxiv_rate_limit, limiter)


def _acquire_chemrxiv_rate_limit(interval: float):
    path = _chemrxiv_rate_limit_path()
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

        f.seek(0)
        f.truncate()
        f.write(f"{now:.6f}\n")
        f.flush()
        os.fsync(f.fileno())
        return f
    except Exception:
        f.close()
        raise


def _release_chemrxiv_rate_limit(f) -> None:
    try:
        release_file_lock(f)
    finally:
        f.close()


def _chemrxiv_rate_limit_path() -> Path:
    override = os.environ.get(_CHEMRXIV_RATE_LIMIT_FILE_ENV)
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "hypothesis_engine_chemrxiv_rate_limit.txt"


def _chemrxiv_request_lock() -> asyncio.Lock:
    global _CHEMRXIV_REQUEST_LOCK

    if _CHEMRXIV_REQUEST_LOCK is None:
        _CHEMRXIV_REQUEST_LOCK = asyncio.Lock()
    return _CHEMRXIV_REQUEST_LOCK


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


async def _crossref_chemrxiv_records(
    client: httpx.AsyncClient,
    *,
    query: str,
    limit: int,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    filters = ["prefix:10.26434", "type:posted-content"]
    if date_from:
        filters.append(f"from-posted-date:{date_from[:10]}")
    if date_to:
        filters.append(f"until-posted-date:{date_to[:10]}")

    params = {
        "rows": min(max(1, limit), CHEMRXIV_API_MAX_RESULTS),
        "filter": ",".join(filters),
        # Crossref's query.bibliographic intermittently returns 500 with
        # prefix filters; the broad query field is more reliable in practice.
        "query": _chemrxiv_search_query(query),
    }
    response = await _crossref_get_with_retry(client, params)
    data = response.json()
    message = data.get("message") if isinstance(data, dict) else None
    items = message.get("items") if isinstance(message, dict) else None
    if not isinstance(items, list):
        return []

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        record = _crossref_item_to_record(item)
        if record is None:
            continue
        key = _crossref_record_key(record)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


async def _crossref_get_with_retry(
    client: httpx.AsyncClient,
    params: dict[str, Any],
) -> httpx.Response:
    headers = {"Accept": "application/json", "User-Agent": _CROSSREF_USER_AGENT}

    for attempt in range(1, _CROSSREF_MAX_ATTEMPTS + 1):
        try:
            response = await _crossref_get_once(client, params, headers=headers)
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
            retry_after = None
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                if status not in _CHEMRXIV_RETRYABLE_STATUS_CODES:
                    raise
                retry_after = _retry_after_seconds(exc.response.headers.get("Retry-After"))
            if attempt >= _CROSSREF_MAX_ATTEMPTS:
                raise
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                delay = (
                    retry_after
                    if retry_after is not None
                    else _CROSSREF_DEFAULT_429_RETRY_AFTER_SECONDS
                )
            else:
                delay = retry_after if retry_after is not None else min(8.0, float(2 ** (attempt - 1)))
            if delay > 0:
                await asyncio.sleep(delay)

    raise RuntimeError("unreachable Crossref retry loop")


async def _crossref_get_once(
    client: httpx.AsyncClient,
    params: dict[str, Any],
    *,
    headers: dict[str, str],
) -> httpx.Response:
    interval = max(0.0, _CROSSREF_MIN_REQUEST_INTERVAL_SECONDS)
    if interval <= 0:
        return await client.get(CROSSREF_WORKS_URL, params=params, headers=headers)

    lock = _crossref_request_lock()
    async with lock:
        limiter = await asyncio.to_thread(_acquire_crossref_rate_limit, interval)
        try:
            return await client.get(CROSSREF_WORKS_URL, params=params, headers=headers)
        finally:
            await asyncio.to_thread(_release_crossref_rate_limit, limiter)


def _acquire_crossref_rate_limit(interval: float):
    path = _crossref_rate_limit_path()
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

        f.seek(0)
        f.truncate()
        f.write(f"{now:.6f}\n")
        f.flush()
        os.fsync(f.fileno())
        return f
    except Exception:
        f.close()
        raise


def _release_crossref_rate_limit(f) -> None:
    try:
        release_file_lock(f)
    finally:
        f.close()


def _crossref_rate_limit_path() -> Path:
    override = os.environ.get(_CROSSREF_RATE_LIMIT_FILE_ENV)
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "hypothesis_engine_crossref_rate_limit.txt"


def _crossref_request_lock() -> asyncio.Lock:
    global _CROSSREF_REQUEST_LOCK

    if _CROSSREF_REQUEST_LOCK is None:
        _CROSSREF_REQUEST_LOCK = asyncio.Lock()
    return _CROSSREF_REQUEST_LOCK


async def _enrich_with_cambridge_doi_items(
    client: httpx.AsyncClient,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched_records: list[dict[str, Any]] = []
    headers = {"Accept": "application/json", "User-Agent": _CHEMRXIV_USER_AGENT}
    for record in records:
        doi = str(record.get("doi") or "").strip()
        if not doi:
            enriched_records.append(record)
            continue
        data: dict[str, Any] | None = None
        for doi_candidate in _cambridge_doi_candidates(doi):
            try:
                response = await client.get(
                    f"{CHEMRXIV_CAMBRIDGE_DOI_URL}/{doi_candidate}",
                    headers=headers,
                )
                response.raise_for_status()
                maybe_data = response.json()
            except (httpx.HTTPError, ValueError):
                continue
            if isinstance(maybe_data, dict):
                data = maybe_data
                break
        if data is None:
            enriched_records.append(record)
            continue

        item = data.get("item") if isinstance(data, dict) else None
        if not isinstance(item, dict) and isinstance(data, dict):
            item = data
        if not isinstance(item, dict):
            enriched_records.append(record)
            continue

        parsed = _parse_chemrxiv_items({"itemHits": [{"item": item}]})
        if not parsed:
            enriched_records.append(record)
            continue
        enriched = parsed[0]
        for key, value in record.items():
            if not enriched.get(key) and value:
                enriched[key] = value
        enriched["metadata_provider"] = "cambridge_doi"
        enriched_records.append(enriched)
    return enriched_records


def _cambridge_doi_candidates(doi: str) -> list[str]:
    cleaned = doi.strip()
    normalized = re.sub(r"/v\d+$", "", cleaned, flags=re.IGNORECASE)
    normalized = re.sub(r"\.v\d+$", "", normalized, flags=re.IGNORECASE)
    return list(dict.fromkeys(candidate for candidate in (cleaned, normalized) if candidate))


def _crossref_item_to_record(item: dict[str, Any]) -> dict[str, Any] | None:
    doi = _clean_text(item.get("DOI"))
    titles = item.get("title")
    title = _clean_text(titles[0]) if isinstance(titles, list) and titles else ""
    if not doi or not title:
        return None

    published = _crossref_date(item.get("posted")) or _crossref_date(item.get("issued"))
    resource = item.get("resource") if isinstance(item.get("resource"), dict) else {}
    primary = resource.get("primary") if isinstance(resource.get("primary"), dict) else {}
    abs_url = _clean_text(primary.get("URL")) or f"https://doi.org/{doi}"
    licenses = item.get("license") if isinstance(item.get("license"), list) else []
    license_url = ""
    if licenses and isinstance(licenses[0], dict):
        license_url = _clean_text(licenses[0].get("URL"))

    return {
        "arxiv_id": doi,
        "chemrxiv_id": doi,
        "source_id": doi,
        "source": "chemrxiv",
        "metadata_provider": "crossref",
        "title": title,
        "summary": "",
        "authors": _crossref_authors(item.get("author")),
        "year": published[:4] if published else None,
        "published": published,
        "updated": "",
        "pdf_url": None,
        "abs_url": abs_url,
        "categories": ["chemrxiv"],
        "keywords": [],
        "doi": doi,
        "license": license_url or None,
        "views_count": None,
        "read_count": None,
        "citation_count": item.get("is-referenced-by-count"),
    }


def _crossref_authors(raw_authors: Any) -> list[str]:
    if not isinstance(raw_authors, list):
        return []
    authors: list[str] = []
    for author in raw_authors:
        if not isinstance(author, dict):
            continue
        given = _clean_text(author.get("given"))
        family = _clean_text(author.get("family"))
        name = " ".join(part for part in (given, family) if part)
        if name:
            authors.append(name)
    return authors


def _crossref_date(raw_date: Any) -> str:
    if not isinstance(raw_date, dict):
        return ""
    date_parts = raw_date.get("date-parts")
    if not isinstance(date_parts, list) or not date_parts or not isinstance(date_parts[0], list):
        return ""
    parts = date_parts[0]
    if not parts:
        return ""
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def _crossref_record_key(record: dict[str, Any]) -> str:
    title_key = _chemrxiv_title_key(str(record.get("title") or ""))
    if title_key:
        return f"title:{title_key}"
    doi = str(record.get("doi") or record.get("chemrxiv_id") or "").casefold()
    doi = re.sub(r"/v\d+$", "", doi)
    doi = re.sub(r"\.v\d+$", "", doi)
    return f"doi:{doi}"


def _chemrxiv_title_key(title: str) -> str | None:
    value = title.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = " ".join(value.split())
    return value if len(value) >= 12 else None


async def _review_search_payload(ctx: ToolCtx, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from ..rag import review_search_payload

        return await review_search_payload(ctx, payload, provider="chemrxiv")
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


def _chemrxiv_search_query(query: str) -> str:
    """Build the ChemRxiv API term query.

    ChemRxiv/Open Engage exposes a simpler ``term`` search than arXiv's native
    fielded query syntax. Preserve quoted phrases by removing only quote marks,
    and otherwise pass the human-readable query through.
    """
    q = " ".join(query.strip().split())
    if not q:
        return q

    phrases = [phrase.strip() for phrase in _PHRASE_RE.findall(q) if phrase.strip()]
    if len(phrases) == 1 and _PHRASE_RE.fullmatch(q):
        return phrases[0]

    return q.replace('"', "")


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

    text = (
        f"{record.get('title') or ''} "
        f"{record.get('summary') or ''} "
        f"{' '.join(record.get('categories') or [])} "
        f"{' '.join(record.get('keywords') or [])}"
    ).lower()

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


def _parse_chemrxiv_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    item_hits = data.get("itemHits", [])
    if not isinstance(item_hits, list):
        raise ValueError("ChemRxiv response missing itemHits list")

    out: list[dict[str, Any]] = []

    for hit in item_hits:
        item = hit.get("item") if isinstance(hit, dict) else None
        if not isinstance(item, dict):
            continue

        chemrxiv_id = str(item.get("id") or "").strip()
        if not chemrxiv_id:
            continue

        title = _clean_text(item.get("title"))
        summary = _clean_text(item.get("abstract"))
        published = _clean_text(item.get("publishedDate"))
        updated = _clean_text(item.get("updatedDate"))
        authors = _parse_authors(item.get("authors"))
        categories = _parse_categories(item.get("categories"))
        keywords = _parse_keywords(item.get("keywords"))
        pdf_url = _extract_pdf_url(item)
        abs_url = _extract_abs_url(item, chemrxiv_id)
        license_info = _parse_license(item.get("license"))

        out.append(
            {
                # Compatibility with arxiv.py-shaped downstream code.
                "arxiv_id": chemrxiv_id,
                "chemrxiv_id": chemrxiv_id,
                "source_id": chemrxiv_id,
                "source": "chemrxiv",
                "title": title,
                "summary": summary,
                "authors": authors,
                "year": published[:4] if published else None,
                "published": published[:10] if published else "",
                "updated": updated[:10] if updated else "",
                "pdf_url": pdf_url,
                "abs_url": abs_url,
                "categories": categories,
                "keywords": keywords,
                "doi": _clean_text(item.get("doi")),
                "license": license_info,
                "views_count": item.get("viewsCount"),
                "read_count": item.get("readCount"),
                "citation_count": item.get("citationCount"),
            }
        )

    return out


def _parse_authors(raw_authors: Any) -> list[str]:
    if not isinstance(raw_authors, list):
        return []

    authors: list[str] = []
    for author in raw_authors:
        if isinstance(author, str):
            name = author.strip()
        elif isinstance(author, dict):
            name = _clean_text(author.get("name"))
            if not name:
                first = _clean_text(author.get("firstName"))
                last = _clean_text(author.get("lastName"))
                name = " ".join(part for part in [first, last] if part)
        else:
            name = ""

        if name:
            authors.append(name)

    return authors


def _parse_categories(raw_categories: Any) -> list[str]:
    if not isinstance(raw_categories, list):
        return []

    categories: list[str] = []
    for category in raw_categories:
        if isinstance(category, str):
            name = category.strip()
        elif isinstance(category, dict):
            name = _clean_text(category.get("name")) or _clean_text(category.get("id"))
        else:
            name = ""

        if name:
            categories.append(name)

    return categories


def _parse_keywords(raw_keywords: Any) -> list[str]:
    if not isinstance(raw_keywords, list):
        return []

    keywords: list[str] = []
    for keyword in raw_keywords:
        if isinstance(keyword, str):
            value = keyword.strip()
        elif isinstance(keyword, dict):
            value = _clean_text(keyword.get("name")) or _clean_text(keyword.get("value"))
        else:
            value = ""

        if value:
            keywords.append(value)

    return keywords


def _parse_license(raw_license: Any) -> str | None:
    if isinstance(raw_license, str):
        return raw_license.strip() or None
    if isinstance(raw_license, dict):
        return (
            _clean_text(raw_license.get("name"))
            or _clean_text(raw_license.get("id"))
            or _clean_text(raw_license.get("url"))
        )
    return None


def _extract_pdf_url(item: dict[str, Any]) -> str | None:
    pdf_url = _clean_text(item.get("pdfUrl"))
    if pdf_url:
        return pdf_url

    asset = item.get("asset")
    if isinstance(asset, dict):
        original = asset.get("original")
        if isinstance(original, dict):
            pdf_url = _clean_text(original.get("url"))
            if pdf_url:
                return pdf_url

    return None


def _extract_abs_url(item: dict[str, Any], chemrxiv_id: str) -> str:
    for key in ("htmlUrl", "landingPageUrl", "url"):
        value = _clean_text(item.get(key))
        if value and value.startswith("http"):
            return value

    return f"{CHEMRXIV_ITEM_URL}/{chemrxiv_id}"


def _as_open_engage_date(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return cleaned
    if "T" in cleaned:
        return cleaned
    return f"{cleaned}T00:00:00.000Z"


def _clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _clean_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
