from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import httpx
import pytest

from hypothesis_engine.config import Config, StorageCfg
from hypothesis_engine.tools import rag as rag_mod
from hypothesis_engine.tools.base import ToolCtx
from hypothesis_engine.tools.builtins import biorxiv as biorxiv_mod
from hypothesis_engine.tools.builtins.biorxiv import (
    BIORXIV_FILTER_VERSION,
    BiorxivSearchTool,
    _biorxiv_search_query,
    _europe_pmc_biorxiv_query,
)

REFERENCE_TITLE = (
    "Facility-Scale Workflows for Data Acquisition, Standardization, Machine Learning "
    "Analysis, and Reproducible Science"
)


def test_biorxiv_search_query_preserves_human_readable_terms() -> None:
    assert (
        _biorxiv_search_query("  microscopy   machine   learning  ")
        == "microscopy machine learning"
    )
    assert _biorxiv_search_query('"single cell imaging"') == "single cell imaging"
    assert (
        _europe_pmc_biorxiv_query("microscopy", date_from="2026-01-01", date_to="2026-12-31")
        == (
            '(microscopy) AND SRC:PPR AND (PUBLISHER:"bioRxiv" OR JOURNAL:"bioRxiv") '
            "AND FIRST_PDATE:[2026-01-01 TO 2026-12-31]"
        )
    )


def test_biorxiv_rate_limit_file_waits_between_request_starts(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(
        biorxiv_mod._BIORXIV_RATE_LIMIT_FILE_ENV,
        str(tmp_path / "biorxiv_rate_limit.txt"),
    )
    now = [1000.0]
    sleeps: list[float] = []

    def fake_time() -> float:
        return now[0]

    def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(biorxiv_mod.time, "time", fake_time)
    monkeypatch.setattr(biorxiv_mod.time, "sleep", fake_sleep)

    limiter = biorxiv_mod._acquire_biorxiv_rate_limit(3.2)
    biorxiv_mod._release_biorxiv_rate_limit(limiter)

    now[0] = 1001.0
    limiter = biorxiv_mod._acquire_biorxiv_rate_limit(3.2)
    biorxiv_mod._release_biorxiv_rate_limit(limiter)

    assert sleeps == [pytest.approx(2.2)]


class _FakeBiorxivResponse:
    def __init__(
        self,
        data: dict[str, Any] | None = None,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self._data = data or {"resultList": {"result": []}}
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self.request = httpx.Request("GET", biorxiv_mod.EUROPE_PMC_URL)

    def json(self) -> dict[str, Any]:
        return self._data

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


class _FakeBiorxivClient:
    requests: ClassVar[list[dict[str, Any]]] = []
    responses: ClassVar[list[_FakeBiorxivResponse]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url, params, headers):
        self.requests.append({"url": url, "params": dict(params), "headers": dict(headers)})
        if self.responses:
            return self.responses.pop(0)
        return _FakeBiorxivResponse()


def _reference_europe_pmc_payload() -> dict[str, Any]:
    return {
        "resultList": {
            "result": [
                {
                    "id": "PPR1210831",
                    "source": "PPR",
                    "doi": "10.64898/2026.05.06.723241",
                    "title": REFERENCE_TITLE,
                    "authorString": (
                        "Madugula SS, Brown SR, Bible AN, Solsona RM, Checa M, "
                        "Harris SB, Vasudevan RK."
                    ),
                    "authorList": {
                        "author": [
                            {"fullName": "Madugula SS"},
                            {"fullName": "Brown SR"},
                            {"fullName": "Harris SB"},
                        ]
                    },
                    "pubYear": "2026",
                    "firstPublicationDate": "2026-05-11",
                    "abstractText": (
                        "Scientific user facilities generate microscopy datasets for "
                        "standardized machine learning workflows."
                    ),
                    "bookOrReportDetails": {
                        "publisher": "bioRxiv",
                        "yearOfPublication": 2026,
                    },
                    "isOpenAccess": "N",
                    "citedByCount": 0,
                },
                {
                    "id": "MED123",
                    "source": "MED",
                    "title": "A journal article that should not be treated as bioRxiv",
                },
            ]
        }
    }


async def _run_biorxiv_call(monkeypatch, tmp_path, *, max_results: int = 2):
    _FakeBiorxivClient.requests = []
    _FakeBiorxivClient.responses = [_FakeBiorxivResponse(_reference_europe_pmc_payload())]
    read_params = []
    write_params = []
    ingest_calls = []

    async def fake_read_cache(_cfg, _session_id, _tool_name, params):
        read_params.append(dict(params))
        return None

    async def fake_write_cache(_cfg, _session_id, _tool_name, params, _payload):
        write_params.append(dict(params))

    def fake_schedule_biorxiv_ingest(ctx, records):
        ingest_calls.append((ctx.session_id, records))
        return {"enabled": True, "scheduled": True, "provider": "biorxiv"}

    monkeypatch.setattr(biorxiv_mod.httpx, "AsyncClient", _FakeBiorxivClient)
    monkeypatch.setattr(biorxiv_mod, "_BIORXIV_MIN_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(biorxiv_mod, "read_search_cache", fake_read_cache)
    monkeypatch.setattr(biorxiv_mod, "write_search_cache", fake_write_cache)
    monkeypatch.setattr(rag_mod, "schedule_biorxiv_ingest", fake_schedule_biorxiv_ingest)

    cfg = Config(storage=StorageCfg(data_dir=str(tmp_path)))
    cfg.rag.enabled = True
    cfg.rag.auto_ingest_arxiv_pdfs = True
    cfg.rag.package_path = str(tmp_path)
    result = await BiorxivSearchTool().call(
        {"query": "facility scale workflows machine learning", "max_results": max_results},
        ToolCtx(cfg=cfg, session_id="ses"),
    )
    assert not result.is_error
    return result, _FakeBiorxivClient.requests, read_params, write_params, ingest_calls


@pytest.mark.asyncio
async def test_biorxiv_search_returns_arxiv_compatible_pdf_records(monkeypatch, tmp_path) -> None:
    result, requests, read_params, write_params, ingest_calls = await _run_biorxiv_call(
        monkeypatch,
        tmp_path,
    )

    assert len(requests) == 1
    assert requests[0]["url"] == biorxiv_mod.EUROPE_PMC_URL
    assert "SRC:PPR" in requests[0]["params"]["query"]
    assert 'PUBLISHER:"bioRxiv"' in requests[0]["params"]["query"]
    assert requests[0]["params"]["pageSize"] == 6
    assert read_params[0]["filter"] == BIORXIV_FILTER_VERSION
    assert write_params[0] == read_params[0]

    assert result.content["n"] == 1
    record = result.content["results"][0]
    assert record["title"] == REFERENCE_TITLE
    assert record["source"] == "biorxiv"
    assert record["arxiv_id"] == "10.64898/2026.05.06.723241"
    assert record["biorxiv_id"] == "10.64898/2026.05.06.723241"
    assert record["authors"] == ["Madugula SS", "Brown SR", "Harris SB"]
    assert record["published"] == "2026-05-11"
    assert record["pdf_url"] == (
        "https://www.biorxiv.org/content/10.64898/2026.05.06.723241.full.pdf"
    )
    assert record["abs_url"] == "https://www.biorxiv.org/content/10.64898/2026.05.06.723241"
    assert result.content["rag_ingest"]["scheduled"] is True
    assert ingest_calls[0][0] == "ses"
    assert ingest_calls[0][1][0]["pdf_url"] == record["pdf_url"]


@pytest.mark.asyncio
async def test_biorxiv_search_retries_rate_limited_requests(monkeypatch, tmp_path) -> None:
    _FakeBiorxivClient.requests = []
    _FakeBiorxivClient.responses = [
        _FakeBiorxivResponse(status_code=429, headers={"Retry-After": "0"}),
        _FakeBiorxivResponse(_reference_europe_pmc_payload()),
    ]

    async def fake_read_cache(_cfg, _session_id, _tool_name, _params):
        return None

    async def fake_write_cache(_cfg, _session_id, _tool_name, _params, _payload):
        return None

    monkeypatch.setattr(biorxiv_mod.httpx, "AsyncClient", _FakeBiorxivClient)
    monkeypatch.setattr(biorxiv_mod, "_BIORXIV_MIN_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(biorxiv_mod, "read_search_cache", fake_read_cache)
    monkeypatch.setattr(biorxiv_mod, "write_search_cache", fake_write_cache)

    cfg = Config(storage=StorageCfg(data_dir=str(tmp_path)))
    result = await BiorxivSearchTool().call(
        {"query": "facility scale workflows machine learning", "max_results": 2},
        ToolCtx(cfg=cfg, session_id="ses"),
    )

    assert not result.is_error
    assert len(_FakeBiorxivClient.requests) == 2
    assert result.content["n"] == 1


@pytest.mark.asyncio
async def test_biorxiv_pdf_downloads_use_shared_global_limiter(
    tmp_cfg, monkeypatch, tmp_path
) -> None:
    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.auto_ingest_arxiv_pdfs = True
    tmp_cfg.rag.auto_ingest_max_pdfs_per_search = 1
    tmp_cfg.rag.package_path = str(tmp_path)
    tmp_cfg.web_fetch.max_bytes = 1_000_000

    acquire_calls: list[float] = []
    release_calls: list[object] = []
    landing_requests: list[dict[str, Any]] = []
    stream_requests: list[dict[str, Any]] = []

    def fake_acquire(interval: float):
        token = object()
        acquire_calls.append(interval)
        return token

    def fake_release(token) -> None:
        release_calls.append(token)

    async def fake_ingest_pdf_payloads(_cfg, _session_id, payloads):
        assert payloads[0]["pdf_bytes"].startswith(b"%PDF")
        return {"enabled": True, "ingested": len(payloads), "fake": True}

    class FakeLandingResponse:
        status_code = 200
        url = "https://www.biorxiv.org/content/10.64898/2026.05.06.723241v1"

    class FakeResponse:
        status_code = 200

        async def aiter_bytes(self):
            await asyncio.sleep(0)
            yield b"%PDF fake biorxiv"

    class FakeStream:
        def __init__(self, url: str, headers: dict[str, str]) -> None:
            self.url = url
            self.headers = headers

        async def __aenter__(self):
            stream_requests.append({"url": self.url, "headers": self.headers})
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, *, headers: dict[str, str]):
            landing_requests.append({"url": url, "headers": headers})
            return FakeLandingResponse()

        def stream(self, method: str, url: str, *, headers: dict[str, str]):
            assert method == "GET"
            assert headers
            return FakeStream(url, headers)

    monkeypatch.setattr(biorxiv_mod, "_BIORXIV_MIN_REQUEST_INTERVAL_SECONDS", 3.2)
    monkeypatch.setattr(biorxiv_mod, "_acquire_biorxiv_rate_limit", fake_acquire)
    monkeypatch.setattr(biorxiv_mod, "_release_biorxiv_rate_limit", fake_release)
    monkeypatch.setattr(rag_mod.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(rag_mod, "ingest_pdf_payloads", fake_ingest_pdf_payloads)

    result = await rag_mod.ingest_arxiv_records(
        ToolCtx(cfg=tmp_cfg, session_id="ses_biorxiv_pdf_limit"),
        [
            {
                "pdf_url": "https://www.biorxiv.org/content/10.64898/2026.05.06.723241.full.pdf",
                "title": REFERENCE_TITLE,
            }
        ],
    )

    assert result["downloaded"] == 1
    assert result["ingested"] == 1
    assert acquire_calls == [3.2]
    assert len(release_calls) == 1
    assert landing_requests[0]["url"] == (
        "https://www.biorxiv.org/content/10.64898/2026.05.06.723241"
    )
    assert landing_requests[0]["headers"]["User-Agent"].startswith("Mozilla/5.0")
    assert "text/html" in landing_requests[0]["headers"]["Accept"]
    assert stream_requests[0]["url"] == (
        "https://www.biorxiv.org/content/10.64898/2026.05.06.723241.full.pdf"
    )
    assert stream_requests[0]["headers"]["Referer"] == (
        "https://www.biorxiv.org/content/10.64898/2026.05.06.723241v1"
    )
    assert stream_requests[0]["headers"]["User-Agent"].startswith("Mozilla/5.0")
    assert stream_requests[0]["headers"]["Accept"] == "application/pdf,*/*"
