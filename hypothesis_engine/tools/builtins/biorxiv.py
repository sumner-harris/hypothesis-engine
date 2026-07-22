"""bioRxiv search via Europe PMC metadata plus canonical bioRxiv PDF URLs.

No API key required. Returns records compatible with the arXiv search tool:
{arxiv_id, biorxiv_id, title, summary, authors, year, pdf_url, abs_url, categories}.

The arxiv_id field is intentionally populated with the bioRxiv DOI or Europe
PMC preprint ID as a compatibility shim for downstream code that expects an
arXiv-style identifier. Prefer biorxiv_id or source_id for provider-aware code.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any

import httpx

from ...file_lock import acquire_exclusive_file_lock, release_file_lock
from ..base import ToolCtx, ToolResult
from .search_cache import cached_payload, normalized_query, read_search_cache, write_search_cache

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
DEFAULT_MAX_RESULTS = 30
BIORXIV_API_MAX_RESULTS = 50
BIORXIV_FILTER_VERSION = "europe_pmc_biorxiv_v1"

# Keep bioRxiv metadata requests at the same conservative cadence as arXiv. The
# limiter is file-backed so parallel workers and duplicate processes share it.
_BIORXIV_MIN_REQUEST_INTERVAL_SECONDS = 3.2
_BIORXIV_MAX_ATTEMPTS = 3
_BIORXIV_DEFAULT_429_RETRY_AFTER_SECONDS = 60.0
_BIORXIV_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_BIORXIV_RATE_LIMIT_FILE_ENV = "HYPOTHESIS_ENGINE_BIORXIV_RATE_LIMIT_FILE"
_BIORXIV_REQUEST_LOCK: asyncio.Lock | None = None
_BIORXIV_USER_AGENT = "hypothesis-engine biorxiv-search/0.1"


class BiorxivSearchTool:
    name = "biorxiv_search"
    description = (
        "Search bioRxiv life-sciences preprints. Returns up to N records with "
        "arxiv-compatible metadata fields: arxiv_id, biorxiv_id, title, summary, "
        "authors, year, pdf_url, abs_url, categories, doi."
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

        if not query:
            return ToolResult(is_error=True, error_message="empty query")

        biorxiv_query = _biorxiv_search_query(query)
        epmc_query = _europe_pmc_biorxiv_query(
            biorxiv_query,
            date_from=date_from,
            date_to=date_to,
        )
        cache_params = {
            "query": normalized_query(query),
            "biorxiv_query": normalized_query(biorxiv_query),
            "epmc_query": normalized_query(epmc_query),
            "max_results": n,
            "sort": sort,
            "date_from": date_from or "",
            "date_to": date_to or "",
            "filter": BIORXIV_FILTER_VERSION,
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
                api_n = min(BIORXIV_API_MAX_RESULTS, max(n, n * 3))
                r = await _biorxiv_get_with_retry(
                    client,
                    EUROPE_PMC_URL,
                    _europe_pmc_params(epmc_query, limit=api_n, sort=sort),
                )
                raw_records = await asyncio.to_thread(_parse_europe_pmc_response, r.json())
                records, filtered_out = _filter_records_by_query(query, raw_records, limit=n)
                if sort in {"submitted", "lastUpdated"}:
                    records = sorted(
                        records,
                        key=lambda record: str(record.get("published") or ""),
                        reverse=True,
                    )
        except httpx.HTTPError as e:
            return ToolResult(
                is_error=True,
                error_message=f"biorxiv failed: {_format_httpx_error(e)}",
            )
        except ValueError as e:
            return ToolResult(is_error=True, error_message=f"biorxiv parse failed: {e}")

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


def _europe_pmc_params(query: str, *, limit: int, sort: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "query": query,
        "format": "json",
        "pageSize": min(max(1, limit), BIORXIV_API_MAX_RESULTS),
        "resultType": "core",
    }
    if sort in {"submitted", "lastUpdated"}:
        params["sort"] = "FIRST_PDATE_D desc"
    return params


async def _biorxiv_get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
) -> httpx.Response:
    headers = {"Accept": "application/json", "User-Agent": _BIORXIV_USER_AGENT}

    for attempt in range(1, _BIORXIV_MAX_ATTEMPTS + 1):
        try:
            response = await _biorxiv_get_once(client, url, params, headers=headers)
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
            retry_after = None
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                if status not in _BIORXIV_RETRYABLE_STATUS_CODES:
                    raise
                retry_after = _retry_after_seconds(exc.response.headers.get("Retry-After"))
            if attempt >= _BIORXIV_MAX_ATTEMPTS:
                raise
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                delay = (
                    retry_after
                    if retry_after is not None
                    else _BIORXIV_DEFAULT_429_RETRY_AFTER_SECONDS
                )
            else:
                delay = (
                    retry_after
                    if retry_after is not None
                    else min(8.0, float(2 ** (attempt - 1)))
                )
            if delay > 0:
                await asyncio.sleep(delay)

    raise RuntimeError("unreachable bioRxiv retry loop")


async def _biorxiv_get_once(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    *,
    headers: dict[str, str],
) -> httpx.Response:
    interval = max(0.0, _BIORXIV_MIN_REQUEST_INTERVAL_SECONDS)
    if interval <= 0:
        return await client.get(url, params=params, headers=headers)

    lock = _biorxiv_request_lock()
    async with lock:
        limiter = await asyncio.to_thread(_acquire_biorxiv_rate_limit, interval)
        try:
            return await client.get(url, params=params, headers=headers)
        finally:
            await asyncio.to_thread(_release_biorxiv_rate_limit, limiter)


def _acquire_biorxiv_rate_limit(interval: float):
    path = _biorxiv_rate_limit_path()
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


def _release_biorxiv_rate_limit(f) -> None:
    try:
        release_file_lock(f)
    finally:
        f.close()


def _biorxiv_rate_limit_path() -> Path:
    override = os.environ.get(_BIORXIV_RATE_LIMIT_FILE_ENV)
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "hypothesis_engine_biorxiv_rate_limit.txt"


def _biorxiv_request_lock() -> asyncio.Lock:
    global _BIORXIV_REQUEST_LOCK

    if _BIORXIV_REQUEST_LOCK is None:
        _BIORXIV_REQUEST_LOCK = asyncio.Lock()
    return _BIORXIV_REQUEST_LOCK


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

        return await review_search_payload(ctx, payload, provider="biorxiv")
    except Exception as exc:
        out = dict(payload)
        out["rag_ingest"] = {"enabled": True, "scheduled": False, "error": str(exc)[:300]}
        return out


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.-]*")
_PHRASE_RE = re.compile(r'"([^"\n]+)"')
_HTML_TAG_RE = re.compile(r"<[^>]+>")
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


def _biorxiv_search_query(query: str) -> str:
    """Build the human-readable query passed to Europe PMC."""
    q = " ".join(query.strip().split())
    if not q:
        return q

    phrases = [phrase.strip() for phrase in _PHRASE_RE.findall(q) if phrase.strip()]
    if len(phrases) == 1 and _PHRASE_RE.fullmatch(q):
        return phrases[0]

    return q.replace('"', "")


def _europe_pmc_biorxiv_query(
    query: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    parts = [
        f"({query})",
        "SRC:PPR",
        '(PUBLISHER:"bioRxiv" OR JOURNAL:"bioRxiv")',
    ]
    if date_from or date_to:
        start = (date_from or "1900-01-01")[:10]
        end = (date_to or "3000-12-31")[:10]
        parts.append(f"FIRST_PDATE:[{start} TO {end}]")
    return " AND ".join(parts)


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


def _parse_europe_pmc_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    result_list = data.get("resultList") if isinstance(data, dict) else None
    results = result_list.get("result") if isinstance(result_list, dict) else None
    if not isinstance(results, list):
        raise ValueError("Europe PMC response missing resultList.result list")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in results:
        if not isinstance(hit, dict) or not _is_biorxiv_hit(hit):
            continue
        record = _europe_pmc_hit_to_record(hit)
        if record is None:
            continue
        key = _record_key(record)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


def _is_biorxiv_hit(hit: dict[str, Any]) -> bool:
    if str(hit.get("source") or "").upper() != "PPR":
        return False
    details = hit.get("bookOrReportDetails")
    publisher = ""
    if isinstance(details, dict):
        publisher = _clean_text(details.get("publisher"))
    journal = _clean_text(hit.get("journalTitle"))
    return publisher.casefold() == "biorxiv" or journal.casefold() == "biorxiv"


def _europe_pmc_hit_to_record(hit: dict[str, Any]) -> dict[str, Any] | None:
    epmc_id = _clean_text(hit.get("id"))
    doi = _clean_text(hit.get("doi"))
    title = _clean_text(hit.get("title"))
    if not epmc_id and not doi:
        return None
    if not title:
        return None

    published = _clean_text(hit.get("firstPublicationDate")) or _clean_text(
        hit.get("dateOfCreation")
    )
    year = _clean_text(hit.get("pubYear")) or (published[:4] if published else "")
    source_id = doi or epmc_id
    abs_url = _biorxiv_abs_url(doi) if doi else f"https://europepmc.org/article/PPR/{epmc_id}"
    pdf_url = _biorxiv_pdf_url(doi) if doi else None

    return {
        "arxiv_id": source_id,
        "biorxiv_id": source_id,
        "source_id": source_id,
        "source": "biorxiv",
        "metadata_provider": "europe_pmc",
        "title": title,
        "summary": _clean_text(hit.get("abstractText")),
        "authors": _parse_authors(hit),
        "year": year or None,
        "published": published,
        "updated": "",
        "pdf_url": pdf_url,
        "abs_url": abs_url,
        "categories": ["biorxiv"],
        "keywords": [],
        "doi": doi or None,
        "license": None,
        "epmc_id": epmc_id or None,
        "is_open_access": hit.get("isOpenAccess") == "Y",
        "citation_count": hit.get("citedByCount"),
    }


def _biorxiv_abs_url(doi: str) -> str:
    return f"https://www.biorxiv.org/content/{doi.strip()}"


def _biorxiv_pdf_url(doi: str) -> str:
    return f"https://www.biorxiv.org/content/{doi.strip()}.full.pdf"


def _parse_authors(hit: dict[str, Any]) -> list[str]:
    author_list = hit.get("authorList")
    authors = author_list.get("author") if isinstance(author_list, dict) else None
    if isinstance(authors, list):
        out = []
        for author in authors:
            if not isinstance(author, dict):
                continue
            name = _clean_text(author.get("fullName"))
            if name:
                out.append(name)
        if out:
            return out

    raw = _clean_text(hit.get("authorString")).rstrip(".")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _record_key(record: dict[str, Any]) -> str:
    doi = str(record.get("doi") or record.get("biorxiv_id") or "").casefold()
    if doi:
        return f"doi:{doi}"
    title = str(record.get("title") or "").casefold()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    title = " ".join(title.split())
    return f"title:{title}"


def _clean_optional_str(value: Any) -> str | None:
    cleaned = _clean_text(value)
    return cleaned or None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = _HTML_TAG_RE.sub(" ", text)
    text = unescape(text)
    return " ".join(text.split())
