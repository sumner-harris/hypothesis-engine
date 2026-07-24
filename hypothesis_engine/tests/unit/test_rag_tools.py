"""Tests for optional RAG tool configuration and source tracking."""

from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from hypothesis_engine.config import load_config
from hypothesis_engine.llm.tool_loop import _recordable_seen_urls
from hypothesis_engine.tools import rag as rag_mod
from hypothesis_engine.tools.registry import ToolRegistry


def test_default_config_keeps_rag_disabled() -> None:
    cfg = load_config(Path("config/default.toml"))
    assert cfg.rag.enabled is False
    assert cfg.rag.seed_kb_path == ""


def test_a100_remote_rag_config_uses_nvidiaspark_endpoints() -> None:
    cfg = load_config(Path("config/a100_remote.toml"))
    assert cfg.rag.enabled is True
    assert cfg.rag.llm_base_url == "http://nvidiaspark:8000/v1"
    assert cfg.rag.llm_model == "gemma-4-26b-a4b-nvfp4"
    assert cfg.rag.embedding_base_url == "http://nvidiaspark:8001/v1"
    assert cfg.rag.embedding_model == "sfr-embedding-mistral"
    assert cfg.rag.embedding_profile == "sfr"
    assert cfg.rag.rerank_base_url == "http://nvidiaspark:8002/v1"
    assert cfg.rag.rerank_model == "nemotron-rerank-1b-v2"
    assert cfg.rag.retrieval_method == "hybrid"
    assert cfg.rag.generation_discovery_max_pdfs_per_round == 100
    assert cfg.rag.generation_wait_min_indexed_papers == 50
    assert cfg.rag.generation_wait_timeout_seconds == 300
    assert cfg.models.literature_review == "gemma-4-31b-it-nvfp4"


def test_registry_exposes_rag_tools_only_when_enabled(tmp_cfg, tmp_path) -> None:
    reg = ToolRegistry(tmp_cfg).discover()
    assert "rag_retrieve_context" not in {t.name for t in reg.all()}

    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.package_path = str(tmp_path)
    reg = ToolRegistry(tmp_cfg).discover()
    names = {t.name for t in reg.all()}
    assert "rag_kb_status" in names
    assert "rag_retrieve_context" in names
    assert "rag_retrieve_context" in {t.name for t in reg.tools_for("generation")}
    assert "rag_retrieve_context" in {t.name for t in reg.tools_for("reflection")}
    assert "rag_retrieve_context" in {t.name for t in reg.tools_for("evolution")}


def test_rag_retrieval_method_is_not_exposed_to_agents(tmp_cfg) -> None:
    tool = rag_mod.RAGRetrieveContextTool(tmp_cfg)

    assert "retrieval_method" not in tool.input_schema["properties"]
    assert tool.input_schema["additionalProperties"] is False


def test_rag_retrieval_method_always_comes_from_config(tmp_cfg, monkeypatch) -> None:
    session_id = "ses_configured_retrieval"
    tmp_cfg.rag.retrieval_method = "hybrid"
    paths = rag_mod._paths(tmp_cfg, session_id)
    paths.root.mkdir(parents=True)
    paths.index_path.write_bytes(b"index")
    paths.meta_path.write_bytes(b"meta")
    captured: dict[str, object] = {}

    def fake_retrieve_context(**kwargs):
        captured.update(kwargs)
        return {
            "context_text": "configured retrieval result",
            "sources": [],
            "source_documents": [],
            "rerank_chunks": [],
            "embedding_model": "test-embedding",
            "embedding_profile": "test",
        }

    monkeypatch.setattr(rag_mod, "_kb_status_sync", lambda _cfg, _session_id: {"status": "ok"})
    monkeypatch.setattr(rag_mod, "_import_retrieval_service", lambda _cfg: fake_retrieve_context)

    result = rag_mod._retrieve_context_unlocked(
        tmp_cfg,
        session_id,
        {
            "query": "focused query",
            # Simulate a stale or malformed model call from a permissive provider.
            "retrieval_method": "hybrid_multi_query_compression",
        },
    )

    assert captured["retrieval_method"] == "hybrid"
    assert result["retrieval_method"] == "hybrid"


def test_seen_urls_record_rag_retrieval_sources() -> None:
    result = {
        "is_error": False,
        "content": {
            "sources": [{"url": "https://arxiv.org/pdf/1234.5678v1", "chunk_id": 2}],
            "rerank_chunks": [{"url": "https://example.test/paper.pdf", "text": "chunk"}],
        },
    }
    assert _recordable_seen_urls("rag_retrieve_context", {"query": "x"}, result) == [
        "https://arxiv.org/pdf/1234.5678v1",
        "https://example.test/paper.pdf",
    ]


def test_rag_build_marks_only_files_that_exist_as_indexed(tmp_cfg, monkeypatch) -> None:
    session_id = "ses_manifest"
    tmp_cfg.rag.enabled = True
    paths = rag_mod._paths(tmp_cfg, session_id)
    paths.root.mkdir(parents=True)
    paths.pdf_dir.mkdir(parents=True)
    paths.manifest_path.write_text(
        json.dumps(
            {
                "papers": {
                    "reserved_missing": {
                        "url": "https://arxiv.org/pdf/9999.99999v1",
                        "title": "Reserved but not downloaded",
                        "file": "missing.pdf",
                        "indexed": False,
                        "reserved": True,
                    }
                }
            }
        )
    )

    def fake_create_embedding_function(**_kwargs):
        return object()

    def fake_build_kb(**kwargs):
        Path(kwargs["index_path"]).write_bytes(b"index")
        Path(kwargs["meta_path"]).write_bytes(b"meta")
        return {"warnings": []}

    def fake_append_kb(**_kwargs):  # pragma: no cover - should not run for a new KB
        raise AssertionError("append_kb should not be called for first build")

    monkeypatch.setattr(
        rag_mod,
        "_import_kb_services",
        lambda _cfg: (
            fake_create_embedding_function,
            fake_build_kb,
            fake_append_kb,
            None,
            object(),
        ),
    )

    url = "https://arxiv.org/pdf/1234.56789v1"
    result = rag_mod._ingest_pdf_payloads_sync(
        tmp_cfg,
        session_id,
        [{"url": url, "title": "Downloaded TMD paper", "pdf_bytes": b"%PDF test"}],
    )

    manifest = json.loads(paths.manifest_path.read_text())
    papers = manifest["papers"]
    downloaded = papers[rag_mod._source_key(url)]
    missing = papers["reserved_missing"]

    assert result["ingested"] == 1
    assert downloaded["indexed"] is True
    assert downloaded["reserved"] is False
    assert (paths.pdf_dir / downloaded["file"]).is_file()
    assert missing["indexed"] is False
    assert missing["reserved"] is True


def test_rag_ingest_preserves_generation_discovery_round_marker(tmp_cfg, monkeypatch) -> None:
    session_id = "ses_round_marker_preserved"
    tmp_cfg.rag.enabled = True
    paths = rag_mod._paths(tmp_cfg, session_id)
    paths.root.mkdir(parents=True)
    paths.pdf_dir.mkdir(parents=True)
    url = "https://arxiv.org/pdf/2401.12345v1"
    round_key = "generation:literature_discovery"
    paths.manifest_path.write_text(
        json.dumps(
            {
                "papers": {
                    rag_mod._source_key(url): {
                        "url": url,
                        "title": "Reserved TMD paper",
                        "title_key": rag_mod._title_key("Reserved TMD paper"),
                        "file": "reserved.pdf",
                        "indexed": False,
                        "reserved": True,
                        "ingest_round": round_key,
                        "reserved_at": "2026-01-01T00:00:00+00:00",
                    }
                }
            }
        )
    )

    def fake_create_embedding_function(**_kwargs):
        return object()

    def fake_build_kb(**kwargs):
        Path(kwargs["index_path"]).write_bytes(b"index")
        Path(kwargs["meta_path"]).write_bytes(b"meta")
        return {"warnings": []}

    def fake_append_kb(**_kwargs):  # pragma: no cover - should not run for a new KB
        raise AssertionError("append_kb should not be called for first build")

    monkeypatch.setattr(
        rag_mod,
        "_import_kb_services",
        lambda _cfg: (
            fake_create_embedding_function,
            fake_build_kb,
            fake_append_kb,
            None,
            object(),
        ),
    )

    result = rag_mod._ingest_pdf_payloads_sync(
        tmp_cfg,
        session_id,
        [{"url": url, "title": "Reserved TMD paper", "pdf_bytes": b"%PDF test"}],
    )

    paper = json.loads(paths.manifest_path.read_text())["papers"][rag_mod._source_key(url)]

    assert result["ingested"] == 1
    assert paper["indexed"] is True
    assert paper["reserved"] is False
    assert paper["ingest_round"] == round_key
    assert paper["reserved_at"] == "2026-01-01T00:00:00+00:00"


def test_rag_reservation_dedupes_pdf_candidates_by_normalized_title(tmp_cfg) -> None:
    session_id = "ses_title_reserve"
    first_url = "https://arxiv.org/pdf/2304.10992v2"
    duplicate_url = "https://chemrxiv.org/engage/chemrxiv/assets/duplicate-paper.pdf"
    title = "Optical properties of MoSe2 monolayer implanted with ultra-low energy Cr ions"

    result = rag_mod._reserve_pdf_candidates_sync(
        tmp_cfg,
        session_id,
        [
            {"url": first_url, "title": title},
            {"url": duplicate_url, "title": title.replace("ultra-low", "ultra low")},
        ],
    )

    manifest = json.loads(rag_mod._paths(tmp_cfg, session_id).manifest_path.read_text())
    papers = manifest["papers"]

    assert result["reserved"] == 1
    assert result["skipped_existing"] == 0
    assert result["skipped_existing_title"] == 1
    assert len(papers) == 1
    stored = papers[rag_mod._source_key(first_url)]
    assert stored["title_key"] == rag_mod._title_key(title)


def test_rag_reservation_caps_generation_discovery_round(tmp_cfg) -> None:
    session_id = "ses_round_cap"
    tmp_cfg.rag.generation_discovery_max_pdfs_per_round = 2
    round_key = "generation:literature_discovery"

    first = rag_mod._reserve_pdf_candidates_sync(
        tmp_cfg,
        session_id,
        [
            {"url": "https://arxiv.org/pdf/2401.00001v1", "title": "paper one"},
            {"url": "https://arxiv.org/pdf/2401.00002v1", "title": "paper two"},
            {"url": "https://arxiv.org/pdf/2401.00003v1", "title": "paper three"},
        ],
        round_key=round_key,
        round_max_papers=tmp_cfg.rag.generation_discovery_max_pdfs_per_round,
    )
    second = rag_mod._reserve_pdf_candidates_sync(
        tmp_cfg,
        session_id,
        [{"url": "https://arxiv.org/pdf/2401.00004v1", "title": "paper four"}],
        round_key=round_key,
        round_max_papers=tmp_cfg.rag.generation_discovery_max_pdfs_per_round,
    )

    manifest = json.loads(rag_mod._paths(tmp_cfg, session_id).manifest_path.read_text())
    papers = manifest["papers"]

    assert first["reserved"] == 2
    assert first["skipped_round_cap"] == 1
    assert first["round_paper_count"] == 2
    assert second["reserved"] == 0
    assert second["skipped_round_cap"] == 1
    assert len(papers) == 2
    assert {item.get("ingest_round") for item in papers.values()} == {round_key}


def test_rag_direct_pdf_ingest_dedupes_by_normalized_title(tmp_cfg) -> None:
    session_id = "ses_title_ingest"
    paths = rag_mod._paths(tmp_cfg, session_id)
    paths.root.mkdir(parents=True)
    paths.pdf_dir.mkdir(parents=True)
    existing_url = "https://arxiv.org/pdf/2304.10992v2"
    duplicate_url = "https://chemrxiv.org/engage/chemrxiv/assets/duplicate-paper.pdf"
    title = "Optical properties of MoSe2 monolayer implanted with ultra-low energy Cr ions"
    paths.manifest_path.write_text(
        json.dumps(
            {
                "papers": {
                    rag_mod._source_key(existing_url): {
                        "url": existing_url,
                        "title": title,
                        "file": "existing.pdf",
                        "indexed": True,
                        "reserved": False,
                    }
                }
            }
        )
    )

    result = rag_mod._ingest_pdf_payloads_sync(
        tmp_cfg,
        session_id,
        [
            {
                "url": duplicate_url,
                "title": title.replace("ultra-low", "ultra low"),
                "pdf_bytes": b"%PDF duplicate",
            }
        ],
    )

    manifest = json.loads(paths.manifest_path.read_text())

    assert result["ingested"] == 0
    assert result["skipped_duplicate_titles"] == 1
    assert result["skipped_duplicates"] == 0
    assert len(manifest["papers"]) == 1
    assert not any(paths.pdf_dir.iterdir())


@pytest.mark.asyncio
async def test_background_ingest_status_tracks_scheduled_session_tasks(
    tmp_cfg, monkeypatch, tmp_path
) -> None:
    from hypothesis_engine.tools.base import ToolCtx

    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.auto_ingest_arxiv_pdfs = True
    tmp_cfg.rag.package_path = str(tmp_path)
    session_id = "ses_background_ingest"
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_ingest(_ctx, _records):
        started.set()
        await release.wait()
        return {"enabled": True, "ingested": 1}

    monkeypatch.setattr(rag_mod, "ingest_arxiv_records", fake_ingest)

    result = rag_mod.schedule_arxiv_ingest(
        ToolCtx(cfg=tmp_cfg, session_id=session_id),
        [{"pdf_url": "https://arxiv.org/pdf/2401.00001v1", "title": "paper"}],
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert result["scheduled"] is True
        assert rag_mod.background_ingest_pending(tmp_cfg, session_id) is True
        assert rag_mod.background_ingest_pending(tmp_cfg, "other_session") is False

        status = rag_mod.background_ingest_status(tmp_cfg, session_id)
        assert status["pending_background_ingest"] is True
        assert status["active_background_tasks"] == 1

        still_pending = await rag_mod.wait_for_background_ingest_step(
            tmp_cfg, session_id, timeout_seconds=0
        )
        assert still_pending["pending_background_ingest"] is True

        release.set()
        done = await rag_mod.wait_for_background_ingest_step(
            tmp_cfg, session_id, timeout_seconds=1.0
        )
        assert done["pending_background_ingest"] is False
        assert done["active_background_tasks"] == 0
    finally:
        release.set()
        await rag_mod.wait_for_background_ingest_step(tmp_cfg, session_id, timeout_seconds=1.0)


def test_background_ingest_status_reports_manifest_pending_counts(tmp_cfg) -> None:
    session_id = "ses_manifest_pending"
    paths = rag_mod._paths(tmp_cfg, session_id)
    paths.root.mkdir(parents=True)
    paths.pdf_dir.mkdir(parents=True)
    (paths.pdf_dir / "downloaded.pdf").write_bytes(b"%PDF")
    paths.manifest_path.write_text(
        json.dumps(
            {
                "papers": {
                    "reserved": {
                        "title": "Reserved",
                        "file": "reserved.pdf",
                        "indexed": False,
                        "reserved": True,
                    },
                    "downloaded": {
                        "title": "Downloaded",
                        "file": "downloaded.pdf",
                        "indexed": False,
                        "reserved": False,
                    },
                    "indexed": {
                        "title": "Indexed",
                        "file": "indexed.pdf",
                        "indexed": True,
                        "reserved": False,
                    },
                }
            }
        )
    )

    status = rag_mod.background_ingest_status(tmp_cfg, session_id)

    assert status["paper_count"] == 3
    assert status["indexed_paper_count"] == 1
    assert status["seed_chunk_count"] == 0
    assert status["seed_kb_ready"] is False
    assert status["reserved_paper_count"] == 1
    assert status["unindexed_paper_count"] == 2
    assert status["downloaded_unindexed_paper_count"] == 1


@pytest.mark.asyncio
async def test_arxiv_pdf_downloads_use_shared_global_limiter(
    tmp_cfg, monkeypatch, tmp_path
) -> None:
    from hypothesis_engine.tools.base import ToolCtx
    from hypothesis_engine.tools.builtins import arxiv as arxiv_mod

    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.auto_ingest_arxiv_pdfs = True
    tmp_cfg.rag.auto_ingest_max_pdfs_per_search = 1
    tmp_cfg.rag.package_path = str(tmp_path)
    tmp_cfg.web_fetch.max_bytes = 1_000_000

    acquire_calls: list[float] = []
    release_calls: list[object] = []
    active_streams = 0
    max_active_streams = 0
    streamed_urls: list[str] = []

    def fake_acquire(_interval: float):
        token = object()
        acquire_calls.append(_interval)
        return token

    def fake_release(token) -> None:
        release_calls.append(token)

    async def fake_ingest_pdf_payloads(_cfg, _session_id, payloads):
        return {"enabled": True, "ingested": len(payloads), "fake": True}

    class FakeResponse:
        status_code = 200

        async def aiter_bytes(self):
            await asyncio.sleep(0)
            yield b"%PDF fake"

    class FakeStream:
        def __init__(self, url: str) -> None:
            self.url = url

        async def __aenter__(self):
            nonlocal active_streams, max_active_streams
            active_streams += 1
            max_active_streams = max(max_active_streams, active_streams)
            streamed_urls.append(self.url)
            await asyncio.sleep(0.01)
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            nonlocal active_streams
            await asyncio.sleep(0)
            active_streams -= 1
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def stream(self, method: str, url: str, *, headers: dict[str, str]):
            assert method == "GET"
            assert headers
            return FakeStream(url)

    monkeypatch.setattr(arxiv_mod, "_ARXIV_MIN_REQUEST_INTERVAL_SECONDS", 3.2)
    monkeypatch.setattr(arxiv_mod, "_acquire_arxiv_rate_limit", fake_acquire)
    monkeypatch.setattr(arxiv_mod, "_release_arxiv_rate_limit", fake_release)
    monkeypatch.setattr(rag_mod.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(rag_mod, "ingest_pdf_payloads", fake_ingest_pdf_payloads)

    async def run_one(i: int):
        return await rag_mod.ingest_arxiv_records(
            ToolCtx(cfg=tmp_cfg, session_id=f"ses_pdf_limit_{i}"),
            [
                {
                    "pdf_url": f"https://arxiv.org/pdf/2401.0000{i}v1",
                    "title": f"paper {i}",
                }
            ],
        )

    results = await asyncio.gather(*(run_one(i) for i in range(5)))

    assert [result["downloaded"] for result in results] == [1, 1, 1, 1, 1]
    assert len(acquire_calls) == 5
    assert acquire_calls == [3.2] * 5
    assert len(release_calls) == 5
    assert len(streamed_urls) == 5
    assert max_active_streams == 1


class _FakeLiteratureReviewLLM:
    def __init__(self, selected_results):
        self.selected_results = selected_results
        self.calls = []

    async def call(self, spec, ctx):
        self.calls.append((spec, ctx))
        block = SimpleNamespace(
            type="tool_use",
            name="record_literature_selection",
            input={
                "selected_results": self.selected_results,
                "rejected_count": 1,
                "notes": "selected by fake reviewer",
            },
            text="",
            id="toolu_select",
        )
        return SimpleNamespace(raw=SimpleNamespace(stop_reason="tool_use", content=[block]))


class _FakeFailedLiteratureReviewLLM:
    def __init__(self):
        self.calls = []

    async def call(self, spec, ctx):
        self.calls.append((spec, ctx))
        return SimpleNamespace(raw=SimpleNamespace(stop_reason="max_tokens", content=[]))


@pytest.mark.asyncio
async def test_review_and_schedule_pdf_ingest_uses_literature_review_selection(
    tmp_cfg, monkeypatch, tmp_path
) -> None:
    from hypothesis_engine.tools.base import ToolCtx

    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.auto_ingest_arxiv_pdfs = True
    tmp_cfg.rag.auto_ingest_max_pdfs_per_search = 2
    tmp_cfg.rag.literature_review_max_output_tokens = 1234
    tmp_cfg.rag.package_path = str(tmp_path)
    records = [
        {
            "title": "Relevant enzymatic nylon hydrolysis",
            "summary": "A detailed abstract about nylon-6,6 hydrolases.",
            "pdf_url": "https://arxiv.org/pdf/2401.00001v1",
        },
        {
            "title": "Unrelated cosmology",
            "summary": "A detailed abstract about galaxy formation.",
            "pdf_url": "https://arxiv.org/pdf/2401.00002v1",
        },
        {
            "title": "Exploratory polymer depolymerization",
            "summary": "A detailed abstract about plastic waste depolymerization.",
            "pdf_url": "https://arxiv.org/pdf/2401.00003v1",
        },
    ]
    llm = _FakeLiteratureReviewLLM(
        [
            {
                "index": 1,
                "title": records[0]["title"],
                "priority": "core",
                "reason": "directly matches enzyme nylon hydrolysis",
                "download_pdf": True,
            },
            {
                "index": 3,
                "title": records[2]["title"],
                "priority": "exploratory",
                "reason": "adjacent polymer recycling mechanism",
                "download_pdf": True,
            },
        ]
    )
    scheduled_records = []

    def fake_schedule(_ctx, selected):
        scheduled_records.extend(selected)
        return {"enabled": True, "scheduled": True, "candidate_records": len(selected)}

    monkeypatch.setattr(rag_mod, "schedule_arxiv_ingest", fake_schedule)

    result = await rag_mod.review_and_schedule_pdf_ingest(
        ToolCtx(
            cfg=tmp_cfg,
            session_id="ses_lit_review",
            llm_client=llm,
            extra={"agent": "reflection", "action": "ReviewHypothesis"},
        ),
        records,
        provider="arxiv",
        query="nylon hydrolase polymer recycling",
    )

    assert result["scheduled"] is True
    assert [item["pdf_url"] for item in scheduled_records] == [
        records[0]["pdf_url"],
        records[2]["pdf_url"],
    ]
    assert result["literature_review"]["reviewed"] is True
    assert result["literature_review"]["selected"] == 2
    assert llm.calls[0][0].max_output_tokens == 1234
    assert llm.calls[0][1].agent == "literature_review"
    prompt = llm.calls[0][0].user_blocks[0].text
    assert "nylon hydrolase polymer recycling" in prompt
    assert "A detailed abstract about nylon-6,6 hydrolases." in prompt


@pytest.mark.asyncio
async def test_review_and_schedule_pdf_ingest_defaults_selected_exploratory_pdfs(
    tmp_cfg, monkeypatch, tmp_path
) -> None:
    from hypothesis_engine.tools.base import ToolCtx

    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.auto_ingest_arxiv_pdfs = True
    tmp_cfg.rag.auto_ingest_max_pdfs_per_search = 2
    tmp_cfg.rag.package_path = str(tmp_path)
    records = [
        {
            "title": "Impact of Au Ion Implantation on 2D Cr2Ge2Te6 for Spintronics",
            "summary": "Ion implantation in a 2D layered material for spintronic modification.",
            "pdf_url": "https://arxiv.org/pdf/2401.00008v1",
        },
        {
            "title": "Defects in graphite engineered by ion implantation",
            "summary": "DFT and microscopy identify defect structures from ion implantation in graphite.",
            "pdf_url": "https://arxiv.org/pdf/2401.00009v1",
        },
    ]
    llm = _FakeLiteratureReviewLLM(
        [
            {
                "index": 1,
                "title": records[0]["title"],
                "priority": "exploratory",
                "reason": "2D ion implantation method context",
                "download_pdf": True,
            },
            {
                "index": 2,
                "title": records[1]["title"],
                "priority": "exploratory",
                "reason": "2D defect mechanism context with DFT evidence",
                "download_pdf": False,
            },
        ]
    )
    scheduled_records = []

    def fake_schedule(_ctx, selected):
        scheduled_records.extend(selected)
        return {"enabled": True, "scheduled": True, "candidate_records": len(selected)}

    monkeypatch.setattr(rag_mod, "schedule_arxiv_ingest", fake_schedule)

    result = await rag_mod.review_and_schedule_pdf_ingest(
        ToolCtx(cfg=tmp_cfg, session_id="ses_lit_review_default_pdf", llm_client=llm),
        records,
        provider="arxiv",
        query="ion implantation 2D defect mechanisms",
    )

    assert result["scheduled"] is True
    assert [item["pdf_url"] for item in scheduled_records] == [
        records[0]["pdf_url"],
        records[1]["pdf_url"],
    ]
    assert result["literature_review"]["selected_pdfs"] == 2
    assert [item["download_pdf"] for item in result["literature_review"]["selected_results"]] == [
        True,
        True,
    ]
    assert result["literature_review"]["selected_pdf_records"][1]["download_inferred"] is True


@pytest.mark.asyncio
async def test_review_and_schedule_pdf_ingest_allows_zero_selected(
    tmp_cfg, monkeypatch, tmp_path
) -> None:
    from hypothesis_engine.tools.base import ToolCtx

    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.auto_ingest_arxiv_pdfs = True
    tmp_cfg.rag.package_path = str(tmp_path)
    llm = _FakeLiteratureReviewLLM([])

    def fail_schedule(_ctx, _records):  # pragma: no cover - should not be called
        raise AssertionError("no PDFs should be scheduled")

    monkeypatch.setattr(rag_mod, "schedule_arxiv_ingest", fail_schedule)

    result = await rag_mod.review_and_schedule_pdf_ingest(
        ToolCtx(cfg=tmp_cfg, session_id="ses_lit_review_zero", llm_client=llm),
        [
            {
                "title": "Weak result",
                "summary": "Abstract is unrelated to the search intent.",
                "pdf_url": "https://arxiv.org/pdf/2401.00004v1",
            }
        ],
        provider="arxiv",
        query="specific enzyme hydrolysis",
    )

    assert result["scheduled"] is False
    assert result["literature_review"]["reviewed"] is True
    assert result["literature_review"]["selected"] == 0


@pytest.mark.asyncio
async def test_review_and_schedule_pdf_ingest_fails_closed_when_review_has_no_tool_call(
    tmp_cfg, monkeypatch, tmp_path
) -> None:
    from hypothesis_engine.tools.base import ToolCtx

    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.auto_ingest_arxiv_pdfs = True
    tmp_cfg.rag.package_path = str(tmp_path)
    llm = _FakeFailedLiteratureReviewLLM()

    def fail_schedule(_ctx, _records):  # pragma: no cover - should not be called
        raise AssertionError("failed literature review must not schedule PDFs")

    monkeypatch.setattr(rag_mod, "schedule_arxiv_ingest", fail_schedule)

    result = await rag_mod.review_and_schedule_pdf_ingest(
        ToolCtx(cfg=tmp_cfg, session_id="ses_lit_review_failed", llm_client=llm),
        [
            {
                "title": "Noisy top result",
                "summary": "A broad but potentially irrelevant abstract.",
                "pdf_url": "https://arxiv.org/pdf/2401.00007v1",
            }
        ],
        provider="arxiv",
        query="specific target mechanism",
    )

    assert result["scheduled"] is False
    assert result["literature_review"]["reviewed"] is False
    assert result["literature_review"]["selected"] == 0
    assert result["literature_review"]["selected_pdfs"] == 0
    assert result["literature_review"]["fallback"] == "missing_selection_tool_call"
    assert result["literature_review"]["stop_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_review_and_schedule_pdf_ingest_falls_back_without_llm(
    tmp_cfg, monkeypatch, tmp_path
) -> None:
    from hypothesis_engine.tools.base import ToolCtx

    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.auto_ingest_arxiv_pdfs = True
    tmp_cfg.rag.auto_ingest_max_pdfs_per_search = 1
    tmp_cfg.rag.package_path = str(tmp_path)
    records = [
        {"title": "first", "pdf_url": "https://arxiv.org/pdf/2401.00005v1"},
        {"title": "second", "pdf_url": "https://arxiv.org/pdf/2401.00006v1"},
    ]
    scheduled_records = []

    def fake_schedule(_ctx, selected):
        scheduled_records.extend(selected)
        return {"enabled": True, "scheduled": True, "candidate_records": len(selected)}

    monkeypatch.setattr(rag_mod, "schedule_arxiv_ingest", fake_schedule)

    result = await rag_mod.review_and_schedule_pdf_ingest(
        ToolCtx(cfg=tmp_cfg, session_id="ses_lit_review_fallback"),
        records,
        provider="arxiv",
        query="fallback query",
    )

    assert result["scheduled"] is True
    assert [item["pdf_url"] for item in scheduled_records] == [records[0]["pdf_url"]]
    assert result["literature_review"]["reviewed"] is False
    assert result["literature_review"]["fallback"] == "missing_llm_client_or_session"


@pytest.mark.asyncio
async def test_review_search_payload_filters_model_context_to_selected_records(
    tmp_cfg, tmp_path
) -> None:
    from hypothesis_engine.tools.base import ToolCtx

    tmp_cfg.rag.enabled = False
    tmp_cfg.rag.package_path = str(tmp_path)
    records = [
        {
            "title": "Weakly related overview",
            "summary": "Broad background but not the search intent.",
            "url": "https://example.test/overview",
        },
        {
            "title": "Specific hydrolase specificity study",
            "summary": "Reports nylon-6,6 selectivity and oligomer products.",
            "url": "https://example.test/specific",
        },
        {
            "title": "Unrelated astronomy",
            "summary": "Not relevant.",
            "url": "https://example.test/astronomy",
        },
    ]
    llm = _FakeLiteratureReviewLLM(
        [
            {
                "index": 2,
                "title": records[1]["title"],
                "priority": "core",
                "reason": "directly matches the search intent",
                "download_pdf": False,
            }
        ]
    )

    reviewed = await rag_mod.review_search_payload(
        ToolCtx(cfg=tmp_cfg, session_id="ses_context_filter", llm_client=llm),
        {"query": "nylon hydrolase selectivity", "n": 3, "results": records},
        provider="pubmed",
    )

    assert reviewed["n"] == 1
    assert reviewed["total_results"] == 3
    assert [item["title"] for item in reviewed["results"]] == [records[1]["title"]]
    assert "Weakly related overview" not in str(reviewed["results"])
    assert reviewed["literature_review"]["reviewed"] is True
    assert reviewed["literature_review"]["selected"] == 1
    assert reviewed["rag_ingest"]["scheduled"] is False


@pytest.mark.asyncio
async def test_review_search_payload_dedupes_records_already_sent_to_reviewer(
    tmp_cfg, tmp_path
) -> None:
    from hypothesis_engine.tools.base import ToolCtx

    tmp_cfg.rag.enabled = False
    tmp_cfg.rag.package_path = str(tmp_path)
    session_id = "ses_review_dedupe"
    first_records = [
        {
            "title": "Repeated ion implantation defect paper",
            "summary": "Mechanistic ion implantation damage study.",
            "pdf_url": "https://arxiv.org/pdf/2401.11111v1",
        },
        {
            "title": "Previously seen graphite implantation paper",
            "summary": "DFT identifies graphite defect structures.",
            "pdf_url": "https://arxiv.org/pdf/2401.22222v1",
        },
    ]
    llm = _FakeLiteratureReviewLLM(
        [
            {
                "index": 1,
                "title": first_records[0]["title"],
                "priority": "core",
                "reason": "selected by fake reviewer",
                "download_pdf": False,
            }
        ]
    )
    ctx = ToolCtx(cfg=tmp_cfg, session_id=session_id, llm_client=llm)

    first = await rag_mod.review_search_payload(
        ctx,
        {"query": "first search", "n": 2, "results": first_records},
        provider="arxiv",
    )

    second_records = [
        {
            "title": "Previously seen graphite implantation paper",
            "summary": "Same paper from another overlapping search.",
            "pdf_url": "https://arxiv.org/pdf/2401.22222v2",
        },
        {
            "title": "New Janus TMD implantation mechanism paper",
            "summary": "A new paper that has not yet been sent to the reviewer.",
            "pdf_url": "https://arxiv.org/pdf/2401.33333v1",
        },
    ]
    second = await rag_mod.review_search_payload(
        ctx,
        {"query": "overlapping search", "n": 2, "results": second_records},
        provider="arxiv",
    )

    assert first["literature_review"]["new_candidate_records"] == 2
    assert first["literature_review"]["deduped_records"] == 0
    assert second["total_results"] == 2
    assert second["literature_review"]["new_candidate_records"] == 1
    assert second["literature_review"]["deduped_records"] == 1
    assert len(llm.calls) == 2
    second_prompt = llm.calls[1][0].user_blocks[0].text
    assert "New Janus TMD implantation mechanism paper" in second_prompt
    assert "Previously seen graphite implantation paper" not in second_prompt

    third = await rag_mod.review_search_payload(
        ctx,
        {"query": "same overlapping search", "n": 2, "results": second_records},
        provider="arxiv",
    )

    assert len(llm.calls) == 2
    assert third["n"] == 0
    assert third["literature_review"]["skipped"] == "all_candidates_previously_reviewed"
    assert third["literature_review"]["deduped_records"] == 2


@pytest.mark.asyncio
async def test_review_search_payload_retries_records_after_failed_review(tmp_cfg, tmp_path) -> None:
    from hypothesis_engine.tools.base import ToolCtx

    tmp_cfg.rag.enabled = False
    tmp_cfg.rag.package_path = str(tmp_path)
    session_id = "ses_review_failed_retry"
    records = [
        {
            "title": "Retryable ion implantation paper",
            "summary": "Mechanistic ion implantation damage study.",
            "pdf_url": "https://arxiv.org/pdf/2401.44444v1",
        }
    ]
    llm = _FakeFailedLiteratureReviewLLM()
    ctx = ToolCtx(cfg=tmp_cfg, session_id=session_id, llm_client=llm)

    first = await rag_mod.review_search_payload(
        ctx,
        {"query": "first failed search", "n": 1, "results": records},
        provider="arxiv",
    )
    second = await rag_mod.review_search_payload(
        ctx,
        {"query": "same failed search", "n": 1, "results": records},
        provider="arxiv",
    )

    assert len(llm.calls) == 2
    assert first["literature_review"]["fallback"] == "missing_selection_tool_call"
    assert second["literature_review"]["fallback"] == "missing_selection_tool_call"
    assert second["literature_review"]["new_candidate_records"] == 1
    assert second["literature_review"]["deduped_records"] == 0
    memory_path = (
        tmp_cfg.session_artifact_dir(session_id) / "literature_review" / "reviewed_records.json"
    )
    assert not memory_path.exists()


def _configure_seed_kb(tmp_cfg, tmp_path: Path, monkeypatch) -> Path:
    vendor = tmp_path / "vendor-rag"
    vendor.mkdir()
    seed = tmp_path / "private-seed-kb"
    seed.mkdir()
    (seed / "kb.index").write_bytes(b"private-index")
    (seed / "kb.pkl").write_bytes(
        pickle.dumps(
            [
                {"source": "publisher-paper.pdf", "text": "licensed chunk one"},
                {"source": "publisher-paper.pdf", "text": "licensed chunk two"},
            ]
        )
    )
    (seed / "pdfs").mkdir()
    (seed / "pdfs" / "publisher-paper.pdf").write_bytes(b"paywalled source")
    (seed / "manifest.json").write_text(
        json.dumps(
            {
                "papers": {
                    "publisher-paper": {
                        "title": "Licensed publisher paper",
                        "url": "https://doi.org/10.1000/private-paper",
                        "file": "publisher-paper.pdf",
                        "indexed": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.package_path = str(vendor)
    tmp_cfg.rag.seed_kb_path = str(seed)

    def fake_load_kb(**kwargs):
        assert Path(kwargs["index_path"]).name == "kb.index"
        assert Path(kwargs["meta_path"]).name == "kb.pkl"
        return {
            "status": "ok",
            "embedding_model": "licensed-embedding-model",
            "embedding_profile": "",
            "dimension": 4,
            "chunk_count": 12,
        }

    monkeypatch.setattr(
        rag_mod,
        "_import_kb_services",
        lambda _cfg: (None, None, None, fake_load_kb, None),
    )
    return seed


def test_seed_kb_initialization_is_isolated_idempotent_and_provenanced(
    tmp_cfg, tmp_path, monkeypatch
) -> None:
    seed = _configure_seed_kb(tmp_cfg, tmp_path, monkeypatch)
    session_id = "ses_seed_private"

    result = rag_mod._initialize_session_rag_sync(tmp_cfg, session_id)

    paths = rag_mod._paths(tmp_cfg, session_id)
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads((seed / "manifest.json").read_text(encoding="utf-8"))
    assert result["status"] == "seeded"
    assert result["seeded"] is True
    assert paths.index_path.read_bytes() == b"private-index"
    assert paths.meta_path.read_bytes() == (seed / "kb.pkl").read_bytes()
    assert not paths.pdf_dir.exists()
    assert "seed" not in source_manifest
    assert "seeded" not in source_manifest["papers"]["publisher-paper"]
    assert manifest["papers"]["publisher-paper"]["seeded"] is True
    assert manifest["seed"]["kb"]["chunk_count"] == 12
    assert manifest["kb"]["chunk_count"] == 12
    assert manifest["seed"]["pdfs_copied"] is False
    assert len(manifest["seed"]["source_fingerprint"]) == 64

    tmp_cfg.rag.seed_kb_path = str(tmp_path / "seed-no-longer-mounted")
    repeated = rag_mod._initialize_session_rag_sync(tmp_cfg, session_id)
    assert repeated["status"] == "existing"
    assert repeated["seeded"] is True

    status = rag_mod._kb_status_sync(tmp_cfg, session_id)
    assert status["seeded"] is True
    assert status["seeded_paper_count"] == 1
    assert status["session_paper_count"] == 0
    assert status["seed_chunk_count"] == 12
    assert status["seed_kb_ready"] is True
    assert status["kb"]["chunk_count"] == 12


def test_seed_kb_requires_both_vendor_artifacts(tmp_cfg, tmp_path) -> None:
    vendor = tmp_path / "vendor-rag"
    vendor.mkdir()
    seed = tmp_path / "incomplete-seed"
    seed.mkdir()
    (seed / "kb.index").write_bytes(b"index")
    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.package_path = str(vendor)
    tmp_cfg.rag.seed_kb_path = str(seed)

    with pytest.raises(rag_mod.RAGSeedError, match=r"kb\.pkl"):
        rag_mod._initialize_session_rag_sync(tmp_cfg, "ses_incomplete_seed")


def test_seeded_papers_do_not_consume_session_ingest_cap(tmp_cfg, tmp_path, monkeypatch) -> None:
    _configure_seed_kb(tmp_cfg, tmp_path, monkeypatch)
    session_id = "ses_seed_cap"
    rag_mod._initialize_session_rag_sync(tmp_cfg, session_id)
    tmp_cfg.rag.max_session_papers = 1

    result = rag_mod._reserve_pdf_candidates_sync(
        tmp_cfg,
        session_id,
        [
            {"url": "https://example.test/new-1.pdf", "title": "New session paper one"},
            {"url": "https://example.test/new-2.pdf", "title": "New session paper two"},
        ],
    )

    assert result["reserved"] == 1
    assert result["skipped_cap"] == 1
    assert result["paper_count"] == 2
    assert result["seeded_paper_count"] == 1
    assert result["session_paper_count"] == 1


def test_new_pdf_appends_to_seeded_index(tmp_cfg, tmp_path, monkeypatch) -> None:
    _configure_seed_kb(tmp_cfg, tmp_path, monkeypatch)
    session_id = "ses_seed_append"
    rag_mod._initialize_session_rag_sync(tmp_cfg, session_id)
    calls: list[dict] = []

    def fail_build_kb(**_kwargs):
        raise AssertionError("seeded KB must be appended, not rebuilt")

    def fake_append_kb(**kwargs):
        calls.append(kwargs)
        return {"new_chunks": 3, "warnings": []}

    monkeypatch.setattr(
        rag_mod,
        "_import_kb_services",
        lambda _cfg: (None, fail_build_kb, fake_append_kb, None, object()),
    )
    result = rag_mod._ingest_pdf_payloads_sync(
        tmp_cfg,
        session_id,
        [
            {
                "url": "https://arxiv.org/pdf/2601.00001",
                "title": "New open paper",
                "pdf_bytes": b"%PDF new paper",
            }
        ],
    )

    assert result["ingested"] == 1
    assert result["built_new_index"] is False
    assert len(calls) == 1
    assert Path(calls[0]["index_path"]) == rag_mod._paths(tmp_cfg, session_id).index_path
    assert result["seeded_paper_count"] == 1
    assert result["session_paper_count"] == 1
    manifest = json.loads(rag_mod._paths(tmp_cfg, session_id).manifest_path.read_text())
    assert manifest["seed"]["kb"]["chunk_count"] == 12
    assert manifest["kb"]["chunk_count"] == 15
