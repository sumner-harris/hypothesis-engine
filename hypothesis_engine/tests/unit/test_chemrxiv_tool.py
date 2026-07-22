from __future__ import annotations

from typing import ClassVar

import httpx
import pytest

from hypothesis_engine.config import Config, StorageCfg
from hypothesis_engine.tools.base import ToolCtx
from hypothesis_engine.tools.builtins import chemrxiv as chemrxiv_mod
from hypothesis_engine.tools.builtins.chemrxiv import (
    CHEMRXIV_FILTER_VERSION,
    ChemrxivSearchTool,
    _cambridge_doi_candidates,
    _chemrxiv_params,
    _chemrxiv_search_query,
    _crossref_item_to_record,
    _crossref_record_key,
    _filter_records_by_query,
    _parse_chemrxiv_items,
)


def test_chemrxiv_filter_version_invalidates_old_cache() -> None:
    assert CHEMRXIV_FILTER_VERSION == "lexical_dynamic_v2"


def test_chemrxiv_search_query_preserves_human_readable_terms() -> None:
    assert _chemrxiv_search_query("  palladium   selenide   superatom  ") == "palladium selenide superatom"
    assert _chemrxiv_search_query('"pulsed laser deposition"') == "pulsed laser deposition"
    assert (
        _chemrxiv_search_query('solid electrolyte "lithium metal" dendrite')
        == "solid electrolyte lithium metal dendrite"
    )


def test_chemrxiv_params_maps_optional_filters() -> None:
    params = _chemrxiv_params(
        term="solid electrolyte",
        limit=99,
        skip=-10,
        sort="PUBLISHED_DATE_DESC",
        date_from="2024-01-01",
        date_to="2024-12-31T12:00:00.000Z",
        license_filter="CC BY 4.0",
        category_ids=["cat-a", "cat-b"],
    )

    assert params == {
        "term": "solid electrolyte",
        "limit": 50,
        "skip": 0,
        "sort": "PUBLISHED_DATE_DESC",
        "searchDateFrom": "2024-01-01T00:00:00.000Z",
        "searchDateTo": "2024-12-31T12:00:00.000Z",
        "searchLicense": "CC BY 4.0",
        "categoryIds": "cat-a,cat-b",
    }


def test_parse_chemrxiv_items_outputs_arxiv_compatible_records() -> None:
    records = _parse_chemrxiv_items(
        {
            "itemHits": [
                {
                    "item": {
                        "id": "abc123",
                        "title": "Solid electrolyte interphase chemistry",
                        "abstract": "Lithium metal battery interphase.",
                        "publishedDate": "2025-02-03T10:11:12.000Z",
                        "updatedDate": "2025-02-04T10:11:12.000Z",
                        "authors": [{"firstName": "Ada", "lastName": "Lovelace"}],
                        "categories": [{"name": "Materials Chemistry"}],
                        "keywords": [{"name": "lithium metal"}],
                        "pdfUrl": "https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/abc123/original/paper.pdf",
                        "htmlUrl": "https://chemrxiv.org/engage/chemrxiv/article-details/abc123",
                        "doi": "10.26434/chemrxiv-abc123",
                        "license": {"name": "CC BY 4.0"},
                        "viewsCount": 12,
                    }
                }
            ]
        }
    )

    assert records == [
        {
            "arxiv_id": "abc123",
            "chemrxiv_id": "abc123",
            "source_id": "abc123",
            "source": "chemrxiv",
            "title": "Solid electrolyte interphase chemistry",
            "summary": "Lithium metal battery interphase.",
            "authors": ["Ada Lovelace"],
            "year": "2025",
            "published": "2025-02-03",
            "updated": "2025-02-04",
            "pdf_url": "https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/abc123/original/paper.pdf",
            "abs_url": "https://chemrxiv.org/engage/chemrxiv/article-details/abc123",
            "categories": ["Materials Chemistry"],
            "keywords": ["lithium metal"],
            "doi": "10.26434/chemrxiv-abc123",
            "license": "CC BY 4.0",
            "views_count": 12,
            "read_count": None,
            "citation_count": None,
        }
    ]


def test_chemrxiv_filter_scores_title_summary_categories_and_keywords() -> None:
    records = [
        {
            "title": "Unrelated organic synthesis",
            "summary": "A route to a small molecule.",
            "categories": ["Organic Chemistry"],
            "keywords": [],
        },
        {
            "title": "Lithium metal batteries with robust solid electrolyte interphase",
            "summary": "Electrolyte design improves dendrite suppression.",
            "categories": ["Materials Chemistry"],
            "keywords": ["solid electrolyte"],
        },
    ]

    filtered, filtered_out = _filter_records_by_query(
        "lithium metal solid electrolyte interphase",
        records,
        limit=30,
    )

    assert [record["title"] for record in filtered] == [
        "Lithium metal batteries with robust solid electrolyte interphase"
    ]
    assert filtered[0]["relevance_score"] > 0
    assert filtered_out == 1


def test_crossref_item_to_record_outputs_chemrxiv_record_and_dedupes_versions() -> None:
    item = {
        "DOI": "10.26434/chemrxiv-2024-solid/v2",
        "title": ["Lithium metal solid electrolyte interphase"],
        "posted": {"date-parts": [[2024, 2, 3]]},
        "author": [{"given": "Ada", "family": "Lovelace"}],
        "resource": {"primary": {"URL": "https://doi.org/10.26434/chemrxiv-2024-solid"}},
        "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
        "is-referenced-by-count": 7,
    }

    record = _crossref_item_to_record(item)

    assert record is not None
    assert record["source"] == "chemrxiv"
    assert record["metadata_provider"] == "crossref"
    assert record["doi"] == "10.26434/chemrxiv-2024-solid/v2"
    assert record["title"] == "Lithium metal solid electrolyte interphase"
    assert record["authors"] == ["Ada Lovelace"]
    assert record["published"] == "2024-02-03"
    assert record["pdf_url"] is None
    assert record["citation_count"] == 7
    assert _crossref_record_key(record) == "title:lithium metal solid electrolyte interphase"
    assert _cambridge_doi_candidates("10.26434/chemrxiv-2025-27573/v2") == [
        "10.26434/chemrxiv-2025-27573/v2",
        "10.26434/chemrxiv-2025-27573",
    ]


@pytest.mark.asyncio
async def test_chemrxiv_403_falls_back_to_crossref_and_cambridge_pdf(monkeypatch, tmp_path) -> None:
    _FakeChemrxivClient.requests = []
    _FakeChemrxivClient.responses = [
        _FakeChemrxivResponse(status_code=403, text="cloudflare challenge"),
        _FakeChemrxivResponse(
            {
                "message": {
                    "items": [
                        {
                            "DOI": "10.26434/chemrxiv-2024-solid",
                            "title": ["Lithium metal solid electrolyte interphase"],
                            "posted": {"date-parts": [[2024, 2, 3]]},
                            "author": [{"given": "Ada", "family": "Lovelace"}],
                            "resource": {
                                "primary": {
                                    "URL": "https://doi.org/10.26434/chemrxiv-2024-solid"
                                }
                            },
                        }
                    ]
                }
            }
        ),
        _FakeChemrxivResponse(
            {
                "item": {
                    "id": "abc123",
                    "doi": "10.26434/chemrxiv-2024-solid",
                    "title": "Lithium metal solid electrolyte interphase",
                    "abstract": "Battery materials chemistry",
                    "publishedDate": "2024-02-03T00:00:00.000Z",
                    "authors": [{"firstName": "Ada", "lastName": "Lovelace"}],
                    "asset": {
                        "original": {
                            "url": "https://www.cambridge.org/engage/api-gateway/coe/assets/paper.pdf"
                        }
                    },
                    "htmlUrl": "https://chemrxiv.org/engage/chemrxiv/article-details/abc123",
                }
            }
        ),
    ]
    read_params = []
    write_params = []
    ingest_calls = []

    async def fake_read_cache(_cfg, _session_id, _tool_name, params):
        read_params.append(dict(params))
        return None

    async def fake_write_cache(_cfg, _session_id, _tool_name, params, _payload):
        write_params.append(dict(params))

    def fake_ingest(ctx, records):
        ingest_calls.append((ctx.session_id, records))
        return {"enabled": True, "scheduled": True, "candidate_records": len(records)}

    monkeypatch.setattr(chemrxiv_mod.httpx, "AsyncClient", _FakeChemrxivClient)
    monkeypatch.setattr(chemrxiv_mod, "_CHEMRXIV_MIN_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(chemrxiv_mod, "read_search_cache", fake_read_cache)
    monkeypatch.setattr(chemrxiv_mod, "write_search_cache", fake_write_cache)
    from hypothesis_engine.tools import rag as rag_mod

    monkeypatch.setattr(rag_mod, "schedule_chemrxiv_ingest", fake_ingest)

    cfg = Config(storage=StorageCfg(data_dir=str(tmp_path)))
    cfg.rag.enabled = True
    result = await ChemrxivSearchTool().call(
        {"query": "lithium metal solid electrolyte", "max_results": 1},
        ToolCtx(cfg=cfg, session_id="ses"),
    )

    assert not result.is_error
    assert result.content["fallback"] == "crossref_cambridge_doi"
    assert result.content["n"] == 1
    record = result.content["results"][0]
    assert record["metadata_provider"] == "cambridge_doi"
    assert record["pdf_url"] == "https://www.cambridge.org/engage/api-gateway/coe/assets/paper.pdf"
    assert record["summary"] == "Battery materials chemistry"
    assert result.content["rag_ingest"]["scheduled"] is True
    assert ingest_calls[0][1][0]["pdf_url"] == record["pdf_url"]
    assert read_params[0]["filter"] == "lexical_dynamic_v2"
    assert write_params[0] == read_params[0]
    assert len(_FakeChemrxivClient.requests) == 3
    assert _FakeChemrxivClient.requests[1]["url"] == chemrxiv_mod.CROSSREF_WORKS_URL
    assert _FakeChemrxivClient.requests[1]["params"]["query"] == "lithium metal solid electrolyte"


class _FakeChemrxivResponse:
    def __init__(
        self,
        payload: dict | None = None,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self._payload = payload if payload is not None else {"itemHits": []}
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}
        self.request = httpx.Request("GET", "https://chemrxiv.org")

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return None
        response = httpx.Response(
            self.status_code,
            headers=self.headers,
            content=self.text.encode("utf-8"),
            request=self.request,
        )
        raise httpx.HTTPStatusError(
            f"status {self.status_code}",
            request=self.request,
            response=response,
        )


def test_chemrxiv_rate_limit_file_waits_for_two_requests_per_second(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(
        chemrxiv_mod._CHEMRXIV_RATE_LIMIT_FILE_ENV,
        str(tmp_path / "chemrxiv_rate_limit.txt"),
    )
    now = [1000.0]
    sleeps: list[float] = []

    def fake_time() -> float:
        return now[0]

    def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(chemrxiv_mod.time, "time", fake_time)
    monkeypatch.setattr(chemrxiv_mod.time, "sleep", fake_sleep)

    limiter = chemrxiv_mod._acquire_chemrxiv_rate_limit(0.5)
    chemrxiv_mod._release_chemrxiv_rate_limit(limiter)

    now[0] = 1000.25
    limiter = chemrxiv_mod._acquire_chemrxiv_rate_limit(0.5)
    chemrxiv_mod._release_chemrxiv_rate_limit(limiter)

    assert sleeps == [pytest.approx(0.25)]


def test_crossref_rate_limit_file_waits_for_one_request_per_second(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(
        chemrxiv_mod._CROSSREF_RATE_LIMIT_FILE_ENV,
        str(tmp_path / "crossref_rate_limit.txt"),
    )
    now = [1000.0]
    sleeps: list[float] = []

    def fake_time() -> float:
        return now[0]

    def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(chemrxiv_mod.time, "time", fake_time)
    monkeypatch.setattr(chemrxiv_mod.time, "sleep", fake_sleep)

    limiter = chemrxiv_mod._acquire_crossref_rate_limit(1.0)
    chemrxiv_mod._release_crossref_rate_limit(limiter)

    now[0] = 1000.25
    limiter = chemrxiv_mod._acquire_crossref_rate_limit(1.0)
    chemrxiv_mod._release_crossref_rate_limit(limiter)

    assert sleeps == [pytest.approx(0.75)]


def test_chemrxiv_http_error_format_includes_429_details() -> None:
    request = httpx.Request("GET", "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items")
    response = httpx.Response(
        429,
        headers={"Retry-After": "3"},
        content=b"rate limit exceeded",
        request=request,
    )
    exc = httpx.HTTPStatusError("too many", request=request, response=response)

    message = chemrxiv_mod._format_httpx_error(exc)

    assert "HTTP 429" in message
    assert "Retry-After=3" in message
    assert "rate limit exceeded" in message
    assert "chemrxiv.org" in message


@pytest.mark.asyncio
async def test_crossref_search_retries_rate_limited_requests(monkeypatch) -> None:
    _FakeChemrxivClient.requests = []
    _FakeChemrxivClient.responses = [
        _FakeChemrxivResponse(status_code=429, headers={"Retry-After": "0.25"}),
        _FakeChemrxivResponse({"message": {"items": []}}),
    ]
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(chemrxiv_mod, "_CROSSREF_MIN_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(chemrxiv_mod.asyncio, "sleep", fake_sleep)

    response = await chemrxiv_mod._crossref_get_with_retry(
        _FakeChemrxivClient(),
        {"query": "solid electrolyte", "rows": 1},
    )

    assert response.status_code == 200
    assert sleeps == [0.25]
    assert len(_FakeChemrxivClient.requests) == 2
    assert _FakeChemrxivClient.requests[0]["url"] == chemrxiv_mod.CROSSREF_WORKS_URL


@pytest.mark.asyncio
async def test_chemrxiv_429_without_retry_after_uses_conservative_cooldown(monkeypatch) -> None:
    _FakeChemrxivClient.requests = []
    _FakeChemrxivClient.responses = [
        _FakeChemrxivResponse(status_code=429),
        _FakeChemrxivResponse(),
    ]
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(chemrxiv_mod, "_CHEMRXIV_MIN_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(chemrxiv_mod, "_CHEMRXIV_DEFAULT_429_RETRY_AFTER_SECONDS", 4.0)
    monkeypatch.setattr(chemrxiv_mod.asyncio, "sleep", fake_sleep)

    response = await chemrxiv_mod._chemrxiv_get_with_retry(
        _FakeChemrxivClient(),
        {"term": "solid electrolyte", "limit": 1},
    )

    assert response.status_code == 200
    assert sleeps == [4.0]
    assert len(_FakeChemrxivClient.requests) == 2


class _FakeChemrxivClient:
    requests: ClassVar[list[dict]] = []
    responses: ClassVar[list[_FakeChemrxivResponse]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url, params=None, headers=None):
        self.requests.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        if self.responses:
            return self.responses.pop(0)
        return _FakeChemrxivResponse()


async def _run_chemrxiv_call(monkeypatch, tmp_path, raw_records: list[dict], *, max_results: int = 2):
    _FakeChemrxivClient.requests = []
    if not _FakeChemrxivClient.responses:
        _FakeChemrxivClient.responses = []
    read_params = []
    write_params = []
    ingest_calls = []

    async def fake_read_cache(_cfg, _session_id, _tool_name, params):
        read_params.append(dict(params))
        return None

    async def fake_write_cache(_cfg, _session_id, _tool_name, params, _payload):
        write_params.append(dict(params))

    def fake_ingest(ctx, records):
        ingest_calls.append((ctx.session_id, records))
        return {"enabled": True, "scheduled": True, "candidate_records": len(records)}

    monkeypatch.setattr(chemrxiv_mod.httpx, "AsyncClient", _FakeChemrxivClient)
    monkeypatch.setattr(chemrxiv_mod, "_CHEMRXIV_MIN_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(chemrxiv_mod, "_CHEMRXIV_LAST_REQUEST_MONOTONIC", 0.0)
    monkeypatch.setattr(chemrxiv_mod, "read_search_cache", fake_read_cache)
    monkeypatch.setattr(chemrxiv_mod, "write_search_cache", fake_write_cache)
    monkeypatch.setattr(chemrxiv_mod, "_parse_chemrxiv_items", lambda _json: raw_records)

    from hypothesis_engine.tools import rag as rag_mod

    monkeypatch.setattr(rag_mod, "schedule_chemrxiv_ingest", fake_ingest)

    cfg = Config(storage=StorageCfg(data_dir=str(tmp_path)))
    cfg.rag.enabled = True
    result = await ChemrxivSearchTool().call(
        {
            "query": 'solid electrolyte "lithium metal"',
            "max_results": max_results,
            "date_from": "2024-01-01",
            "category_ids": ["materials"],
        },
        ToolCtx(cfg=cfg, session_id="ses"),
    )
    assert not result.is_error
    return result, _FakeChemrxivClient.requests, read_params, write_params, ingest_calls


@pytest.mark.asyncio
async def test_chemrxiv_call_filters_records_caches_query_and_schedules_rag(monkeypatch, tmp_path) -> None:
    raw_records = [
        {
            "title": "Lithium metal solid electrolyte interphase",
            "summary": "Battery materials chemistry",
            "pdf_url": "https://chemrxiv.org/paper.pdf",
        },
        {"title": "Unrelated synthesis", "summary": "Small molecule route"},
    ]

    result, requests, read_params, write_params, ingest_calls = await _run_chemrxiv_call(
        monkeypatch, tmp_path, raw_records
    )

    assert requests[0]["params"]["term"] == "solid electrolyte lithium metal"
    assert requests[0]["params"]["limit"] == 6
    assert requests[0]["params"]["searchDateFrom"] == "2024-01-01T00:00:00.000Z"
    assert requests[0]["params"]["categoryIds"] == "materials"
    assert result.content["n"] == 1
    assert result.content["filtered_out"] == 1
    assert result.content["rag_ingest"]["scheduled"] is True
    assert len(ingest_calls) == 1
    assert ingest_calls[0][0] == "ses"
    assert read_params[0]["chemrxiv_query"] == "solid electrolyte lithium metal"
    assert read_params[0]["filter"] == CHEMRXIV_FILTER_VERSION
    assert write_params[0] == read_params[0]


@pytest.mark.asyncio
async def test_chemrxiv_search_retries_rate_limited_requests(monkeypatch, tmp_path) -> None:
    _FakeChemrxivClient.responses = [
        _FakeChemrxivResponse(status_code=429, headers={"Retry-After": "0"}),
        _FakeChemrxivResponse(),
    ]

    result, requests, _read_params, _write_params, _ingest_calls = await _run_chemrxiv_call(
        monkeypatch,
        tmp_path,
        [{"title": "Lithium metal solid electrolyte", "summary": "interphase"}],
    )

    assert len(requests) == 2
    assert result.content["n"] == 1
    assert not _FakeChemrxivClient.responses
