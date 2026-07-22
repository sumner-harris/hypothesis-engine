from __future__ import annotations

from typing import ClassVar

import httpx
import pytest

from hypothesis_engine.config import Config, StorageCfg
from hypothesis_engine.tools.base import ToolCtx
from hypothesis_engine.tools.builtins import arxiv as arxiv_mod
from hypothesis_engine.tools.builtins.arxiv import (
    ARXIV_FILTER_VERSION,
    ArxivSearchTool,
    _arxiv_search_query,
    _filter_records_by_query,
    _is_structured_arxiv_query,
)
from hypothesis_engine.tools.builtins.biorxiv import BiorxivSearchTool
from hypothesis_engine.tools.builtins.chemrxiv import ChemrxivSearchTool
from hypothesis_engine.tools.builtins.europe_pmc import EuropePMCSearchTool
from hypothesis_engine.tools.builtins.pubmed import PubmedSearchTool
from hypothesis_engine.tools.builtins.search_cache import (
    cached_payload,
    normalized_query,
    read_search_cache,
    write_search_cache,
)


def test_arxiv_filter_version_bumped_for_query_behavior_change() -> None:
    assert ARXIV_FILTER_VERSION == "lexical_dynamic_v3"


def test_is_structured_arxiv_query_detects_native_syntax() -> None:
    assert not _is_structured_arxiv_query("palladium selenide superatom")
    assert not _is_structured_arxiv_query("graphene and moire")
    assert _is_structured_arxiv_query('cat:cond-mat.mtrl-sci AND all:"Bayesian optimization"')
    assert _is_structured_arxiv_query('abs:"pulsed laser deposition" AND cat:cond-mat.mtrl-sci')
    assert _is_structured_arxiv_query("ti:graphene OR abs:moire")
    assert _is_structured_arxiv_query("(all:mos2 AND cat:cond-mat.mtrl-sci)")


def test_arxiv_search_query_converts_plain_text_and_passes_structured_queries() -> None:
    assert (
        _arxiv_search_query("palladium selenide superatom")
        == "all:palladium OR all:selenide OR all:superatom"
    )
    assert (
        _arxiv_search_query('cat:cond-mat.mtrl-sci AND all:"Bayesian optimization"')
        == 'cat:cond-mat.mtrl-sci AND all:"Bayesian optimization"'
    )
    assert (
        _arxiv_search_query('abs:"pulsed laser deposition" AND cat:cond-mat.mtrl-sci')
        == 'abs:"pulsed laser deposition" AND cat:cond-mat.mtrl-sci'
    )
    assert _arxiv_search_query("and or") == 'all:"and or"'
    assert (
        _arxiv_search_query('"ion implantation" monolayer TMD defect formation')
        == 'all:"ion implantation" OR all:monolayer OR all:tmd OR all:defect OR all:formation'
    )


def test_literature_search_tools_default_to_thirty_results() -> None:
    assert ArxivSearchTool.input_schema["properties"]["max_results"]["default"] == 30
    assert BiorxivSearchTool.input_schema["properties"]["max_results"]["default"] == 30
    assert ChemrxivSearchTool.input_schema["properties"]["max_results"]["default"] == 30
    assert EuropePMCSearchTool.input_schema["properties"]["max_results"]["default"] == 30
    assert PubmedSearchTool(Config()).input_schema["properties"]["max_results"]["default"] == 30


def test_arxiv_filter_drops_broad_unrelated_energy_hits() -> None:
    records = [
        {
            "title": "The Dark Energy Survey",
            "summary": "A cosmology survey of dark energy.",
            "pdf_url": "https://arxiv.org/pdf/astro-ph/0510346v1",
        },
        {
            "title": "Defects in graphite engineered by ion implantation",
            "summary": "Ion implantation creates point defects and enables nanoparticle assembly.",
            "pdf_url": "https://arxiv.org/pdf/2411.02204v1",
        },
        {
            "title": "Controlled ion implantation defect production in monolayer MoS2",
            "summary": "Low energy ion implantation creates TMD chalcogen defects.",
            "pdf_url": "https://arxiv.org/pdf/2210.04662v2",
        },
    ]

    filtered, filtered_out = _filter_records_by_query(
        "low energy ion implantation monolayer TMD defect mechanisms",
        records,
        limit=30,
    )

    assert [r["title"] for r in filtered] == [
        "Controlled ion implantation defect production in monolayer MoS2",
        "Defects in graphite engineered by ion implantation",
    ]
    assert filtered[0]["relevance_score"] > filtered[1]["relevance_score"]
    assert filtered_out == 1


def test_arxiv_filter_does_not_hard_require_domain_anchors() -> None:
    records = [
        {
            "title": "The Dark Energy Survey",
            "summary": "A cosmology survey of dark energy.",
            "pdf_url": "https://arxiv.org/pdf/astro-ph/0510346v1",
        },
        {
            "title": "Defects in graphite engineered by ion implantation",
            "summary": "Ion implantation creates point defects and enables nanoparticle assembly.",
            "pdf_url": "https://arxiv.org/pdf/2411.02204v1",
        },
    ]

    filtered, filtered_out = _filter_records_by_query(
        "low energy ion implantation monolayer TMD defect mechanisms",
        records,
        limit=30,
    )

    assert [r["title"] for r in filtered] == [
        "Defects in graphite engineered by ion implantation"
    ]
    assert filtered_out == 1


def test_arxiv_filter_keeps_exact_phrase_match() -> None:
    records = [
        {
            "title": "Optical properties of MoSe2 monolayer implanted with ultra-low energy Cr ions",
            "summary": "Ultra-low energy Cr ion irradiation produces defects in MoSe2.",
        }
    ]

    filtered, filtered_out = _filter_records_by_query(
        '"Optical properties of MoSe2 monolayer implanted with ultra-low energy Cr ions"',
        records,
        limit=30,
    )

    assert len(filtered) == 1
    assert filtered[0]["relevance_score"] >= 4
    assert filtered_out == 0


class _FakeArxivResponse:
    def __init__(
        self,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        text: str = "<feed></feed>",
    ) -> None:
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}
        self.request = httpx.Request("GET", "https://export.arxiv.org/api/query")

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


def test_arxiv_rate_limit_file_waits_between_request_starts(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(
        arxiv_mod._ARXIV_RATE_LIMIT_FILE_ENV,
        str(tmp_path / "arxiv_rate_limit.txt"),
    )
    now = [1000.0]
    sleeps: list[float] = []

    def fake_time() -> float:
        return now[0]

    def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(arxiv_mod.time, "time", fake_time)
    monkeypatch.setattr(arxiv_mod.time, "sleep", fake_sleep)

    limiter = arxiv_mod._acquire_arxiv_rate_limit(3.2)
    arxiv_mod._release_arxiv_rate_limit(limiter)

    now[0] = 1001.0
    limiter = arxiv_mod._acquire_arxiv_rate_limit(3.2)
    arxiv_mod._release_arxiv_rate_limit(limiter)

    assert sleeps == [pytest.approx(2.2)]


def test_arxiv_http_error_format_includes_429_details() -> None:
    request = httpx.Request("GET", "https://export.arxiv.org/api/query?search_query=all%3Atmd")
    response = httpx.Response(
        429,
        headers={"Retry-After": "12"},
        content=b"rate limit exceeded",
        request=request,
    )
    exc = httpx.HTTPStatusError("too many", request=request, response=response)

    message = arxiv_mod._format_httpx_error(exc)

    assert "HTTP 429" in message
    assert "Retry-After=12" in message
    assert "rate limit exceeded" in message
    assert "export.arxiv.org/api/query" in message


@pytest.mark.asyncio
async def test_arxiv_429_without_retry_after_uses_conservative_cooldown(monkeypatch) -> None:
    _FakeArxivClient.requests = []
    _FakeArxivClient.responses = [
        _FakeArxivResponse(status_code=429),
        _FakeArxivResponse(),
    ]
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(arxiv_mod, "_ARXIV_MIN_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(arxiv_mod, "_ARXIV_DEFAULT_429_RETRY_AFTER_SECONDS", 7.0)
    monkeypatch.setattr(arxiv_mod.asyncio, "sleep", fake_sleep)

    response = await arxiv_mod._arxiv_get_with_retry(
        _FakeArxivClient(),
        {"search_query": "all:tmd", "max_results": 1},
    )

    assert response.status_code == 200
    assert sleeps == [7.0]
    assert len(_FakeArxivClient.requests) == 2


class _FakeArxivClient:
    requests: ClassVar[list[dict]] = []
    responses: ClassVar[list[_FakeArxivResponse]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url, params):
        self.requests.append({"url": url, "params": dict(params)})
        if self.responses:
            return self.responses.pop(0)
        return _FakeArxivResponse()


async def _run_arxiv_call(monkeypatch, tmp_path, query: str, raw_records: list[dict], *, max_results: int = 2):
    _FakeArxivClient.requests = []
    if not _FakeArxivClient.responses:
        _FakeArxivClient.responses = []
    read_params = []
    write_params = []

    async def fake_read_cache(_cfg, _session_id, _tool_name, params):
        read_params.append(dict(params))
        return None

    async def fake_write_cache(_cfg, _session_id, _tool_name, params, _payload):
        write_params.append(dict(params))

    monkeypatch.setattr(arxiv_mod.httpx, "AsyncClient", _FakeArxivClient)
    monkeypatch.setattr(arxiv_mod, "_ARXIV_MIN_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(arxiv_mod, "_ARXIV_LAST_REQUEST_MONOTONIC", 0.0)
    monkeypatch.setattr(arxiv_mod, "read_search_cache", fake_read_cache)
    monkeypatch.setattr(arxiv_mod, "write_search_cache", fake_write_cache)
    monkeypatch.setattr(arxiv_mod, "_parse_atom", lambda _xml: raw_records)

    cfg = Config(storage=StorageCfg(data_dir=str(tmp_path)))
    result = await ArxivSearchTool().call(
        {"query": query, "max_results": max_results},
        ToolCtx(cfg=cfg, session_id="ses"),
    )
    assert not result.is_error
    return result, _FakeArxivClient.requests, read_params, write_params


@pytest.mark.asyncio
async def test_arxiv_plain_query_uses_broad_or_query_and_local_filtering(monkeypatch, tmp_path) -> None:
    raw_records = [
        {"title": "Palladium selenide superatom cluster", "summary": "superatom behavior"},
        {"title": "Dark energy survey", "summary": "cosmology"},
    ]
    filter_calls = []

    def fake_filter(query, records, *, limit):
        filter_calls.append((query, records, limit))
        return [records[0]], len(records) - 1

    monkeypatch.setattr(arxiv_mod, "_filter_records_by_query", fake_filter)

    result, requests, read_params, write_params = await _run_arxiv_call(
        monkeypatch, tmp_path, "palladium selenide superatom", raw_records
    )

    assert requests[0]["params"]["search_query"] == "all:palladium OR all:selenide OR all:superatom"
    assert len(filter_calls) == 1
    assert filter_calls[0][0] == "palladium selenide superatom"
    assert result.content["n"] == 1
    assert result.content["filtered_out"] == 1
    assert read_params[0]["arxiv_query"] == "all:palladium or all:selenide or all:superatom"
    assert write_params[0] == read_params[0]


@pytest.mark.asyncio
async def test_arxiv_search_retries_rate_limited_requests(monkeypatch, tmp_path) -> None:
    _FakeArxivClient.responses = [
        _FakeArxivResponse(status_code=429, headers={"Retry-After": "0"}),
        _FakeArxivResponse(),
    ]
    raw_records = [
        {"title": "Monolayer MoS2 ion implantation", "summary": "TMD defect formation"},
    ]

    result, requests, _read_params, _write_params = await _run_arxiv_call(
        monkeypatch, tmp_path, "monolayer TMD", raw_records
    )

    assert len(requests) == 2
    assert result.content["n"] == 1
    assert not _FakeArxivClient.responses


@pytest.mark.asyncio
async def test_arxiv_structured_query_passes_through_and_skips_local_filtering(
    monkeypatch, tmp_path
) -> None:
    raw_records = [
        {"title": "A", "summary": "one"},
        {"title": "B", "summary": "two"},
        {"title": "C", "summary": "three"},
    ]

    def fail_filter(*_args, **_kwargs):
        raise AssertionError("structured arXiv queries should not use local filtering")

    monkeypatch.setattr(arxiv_mod, "_filter_records_by_query", fail_filter)
    query = 'cat:cond-mat.mtrl-sci AND all:"Bayesian optimization"'

    result, requests, read_params, write_params = await _run_arxiv_call(
        monkeypatch, tmp_path, query, raw_records, max_results=2
    )

    assert requests[0]["params"]["search_query"] == query
    assert [record["title"] for record in result.content["results"]] == ["A", "B"]
    assert result.content["filtered_out"] == 1
    assert read_params[0]["arxiv_query"] == 'cat:cond-mat.mtrl-sci and all:"bayesian optimization"'
    assert write_params[0] == read_params[0]


@pytest.mark.asyncio
async def test_session_search_cache_round_trips(tmp_path) -> None:
    cfg = Config(storage=StorageCfg(data_dir=str(tmp_path)))
    params = {"query": normalized_query("  MoS2   Defects "), "max_results": 30}
    payload = {"query": "MoS2 Defects", "n": 1, "results": [{"title": "x"}]}

    assert await read_search_cache(cfg, "ses", "arxiv_search", params) is None
    await write_search_cache(cfg, "ses", "arxiv_search", params, payload)

    cached = await read_search_cache(cfg, "ses", "arxiv_search", params)
    assert cached == payload
    assert cached_payload(cached)["cached"] is True
    assert normalized_query("  MoS2   Defects ") == "mos2 defects"
