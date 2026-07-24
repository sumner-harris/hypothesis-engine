"""Optional bridge to the cloned RAGAgent knowledge-base tools.

This module keeps the upstream RAG package as a vendored dependency and exposes
small hypothesis-engine-native tools around its build/append/retrieve services.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import sys
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .. import ids
from ..config import PROJECT_ROOT, Config
from .base import ToolCtx, ToolResult

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class RAGPaths:
    root: Path
    pdf_dir: Path
    index_path: Path
    meta_path: Path
    manifest_path: Path
    graphrag_dir: Path


def rag_is_enabled(cfg: Config) -> bool:
    return bool(cfg.rag.enabled and _package_root(cfg).is_dir())


def _package_root(cfg: Config) -> Path:
    root = Path(cfg.rag.package_path)
    return root if root.is_absolute() else PROJECT_ROOT / root


def _ensure_vendor_imports(cfg: Config) -> None:
    root = _package_root(cfg).resolve()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _paths(cfg: Config, session_id: str) -> RAGPaths:
    root = cfg.session_rag_dir(session_id)
    return RAGPaths(
        root=root,
        pdf_dir=root / "pdfs",
        index_path=root / "kb.index",
        meta_path=root / "kb.pkl",
        manifest_path=root / "manifest.json",
        graphrag_dir=root / "graphrag",
    )


class RAGSeedError(ValueError):
    """Raised when a configured prebuilt RAG knowledge base cannot be seeded."""


async def initialize_session_rag(cfg: Config, session_id: str) -> dict[str, Any]:
    """Install a configured prebuilt KB before agents begin using a session."""
    return await asyncio.to_thread(_initialize_session_rag_sync, cfg, session_id)


def _initialize_session_rag_sync(cfg: Config, session_id: str) -> dict[str, Any]:
    if not cfg.rag.enabled:
        return {"enabled": False, "status": "disabled", "seeded": False}

    paths = _paths(cfg, session_id)
    with _session_lock(session_id):
        has_index = paths.index_path.is_file()
        has_meta = paths.meta_path.is_file()
        if has_index and has_meta:
            manifest = _read_manifest(paths.manifest_path)
            seed = manifest.get("seed")
            return {
                "enabled": rag_is_enabled(cfg),
                "status": "existing",
                "seeded": isinstance(seed, dict),
                "seed": seed if isinstance(seed, dict) else None,
            }

        seed_root = cfg.rag_seed_kb_path
        if seed_root is None:
            return {
                "enabled": rag_is_enabled(cfg),
                "status": "not_configured",
                "seeded": False,
            }
        if has_index != has_meta:
            raise RAGSeedError(
                f"session RAG directory is incomplete at {paths.root}; "
                "refusing to overwrite it with the configured seed"
            )
        if paths.root.exists() and any(paths.root.iterdir()):
            raise RAGSeedError(
                f"session RAG directory is not empty at {paths.root}; "
                "refusing to overwrite it with the configured seed"
            )
        if not _package_root(cfg).is_dir():
            raise RAGSeedError(
                "RAG seed_kb_path requires an installed "
                "RAGAgent_AutonomousMaterialsSynthesis package"
            )

        source_root = seed_root.resolve()
        destination_root = paths.root.resolve()
        if source_root == destination_root:
            raise RAGSeedError("RAG seed_kb_path cannot be the session RAG directory")
        if not source_root.is_dir():
            raise RAGSeedError(f"RAG seed_kb_path is not a directory: {source_root}")
        source_index = source_root / "kb.index"
        source_meta = source_root / "kb.pkl"
        missing = [path.name for path in (source_index, source_meta) if not path.is_file()]
        if missing:
            missing_text = ", ".join(missing)
            raise RAGSeedError(f"RAG seed KB is missing required file(s): {missing_text}")

        source_manifest_path = source_root / "manifest.json"
        source_manifest = _read_seed_manifest(source_manifest_path)
        source_graphrag = source_root / "graphrag"
        if cfg.rag.use_graphrag and not source_graphrag.is_dir():
            raise RAGSeedError(
                "[rag].use_graphrag is true but the seed KB has no graphrag directory"
            )
        try:
            _create_embeddings, _build_kb, _append_kb, load_kb, _enc = _import_kb_services(cfg)
            kb_summary = load_kb(
                index_path=str(source_index),
                meta_path=str(source_meta),
                graphrag_dir=(str(source_graphrag) if cfg.rag.use_graphrag else ""),
            )
        except Exception as exc:
            raise RAGSeedError(f"RAG seed KB validation failed: {exc}") from exc

        paths.root.parent.mkdir(parents=True, exist_ok=True)
        staged_root = Path(tempfile.mkdtemp(prefix=f".{session_id}-seed-", dir=paths.root.parent))
        initialized_at = datetime.now(UTC).isoformat()
        try:
            index_sha256 = _copy_file_with_sha256(source_index, staged_root / "kb.index")
            metadata_sha256 = _copy_file_with_sha256(source_meta, staged_root / "kb.pkl")
            fingerprint = hashlib.sha256(
                f"{index_sha256}:{metadata_sha256}".encode("ascii")
            ).hexdigest()

            papers = source_manifest.setdefault("papers", {})
            for item in papers.values():
                if isinstance(item, dict):
                    item["seeded"] = True
                    item["seed_fingerprint"] = fingerprint

            graphrag_copied = False
            if source_graphrag.is_dir() and (cfg.rag.use_graphrag or cfg.rag.run_graphrag):
                shutil.copytree(source_graphrag, staged_root / "graphrag")
                graphrag_copied = True

            seed_record = {
                "schema_version": 1,
                "source_path": str(source_root),
                "source_fingerprint": fingerprint,
                "initialized_at": initialized_at,
                "index_sha256": index_sha256,
                "metadata_sha256": metadata_sha256,
                "index_bytes": source_index.stat().st_size,
                "metadata_bytes": source_meta.stat().st_size,
                "source_manifest_present": source_manifest_path.is_file(),
                "source_paper_count": len(papers),
                "pdfs_copied": False,
                "graphrag_copied": graphrag_copied,
                "kb": kb_summary,
            }
            source_manifest["kb"] = dict(kb_summary)
            source_manifest["seed"] = seed_record
            _write_manifest(staged_root / "manifest.json", source_manifest)

            if paths.root.exists():
                paths.root.rmdir()
            staged_root.replace(paths.root)
        except Exception:
            shutil.rmtree(staged_root, ignore_errors=True)
            raise

        return {
            "enabled": True,
            "status": "seeded",
            "seeded": True,
            "seed": seed_record,
        }


def _read_seed_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"papers": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RAGSeedError(f"invalid RAG seed manifest at {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("papers", {}), dict):
        raise RAGSeedError(
            f"RAG seed manifest must be an object with an optional papers object: {path}"
        )
    data.setdefault("papers", {})
    return data


def _copy_file_with_sha256(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
            digest.update(chunk)
    shutil.copystat(source, destination)
    return digest.hexdigest()


def _session_lock(session_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[session_id] = lock
        return lock


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    papers = data.setdefault("papers", {})
    if not isinstance(papers, dict):
        data["papers"] = {}
    return data


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    tmp.replace(path)


def _source_key(url: str) -> str:
    return ids.url_hash(url)


def _title_key(title: str | None) -> str | None:
    value = (title or "").casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = " ".join(value.split())
    if len(value) < 12:
        return None
    return value


def _existing_title_keys(papers: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for item in papers.values():
        if not isinstance(item, dict):
            continue
        key = item.get("title_key")
        if not isinstance(key, str) or not key:
            key = _title_key(str(item.get("title") or ""))
        if key:
            out.add(key)
    return out


def _seeded_paper_count(papers: dict[str, Any]) -> int:
    return sum(
        1 for item in papers.values() if isinstance(item, dict) and item.get("seeded") is True
    )


def _session_added_paper_count(papers: dict[str, Any]) -> int:
    return sum(
        1 for item in papers.values() if isinstance(item, dict) and item.get("seeded") is not True
    )


def _paper_count_fields(papers: dict[str, Any]) -> dict[str, int]:
    seeded = _seeded_paper_count(papers)
    return {
        "paper_count": len(papers),
        "seeded_paper_count": seeded,
        "session_paper_count": _session_added_paper_count(papers),
    }


def _slug(text: str | None) -> str:
    value = (text or "paper").lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:60] or "paper"


def _pdf_filename(url: str, title: str | None) -> str:
    return f"{_source_key(url)[:16]}-{_slug(title)}.pdf"


def _rag_api_key(cfg: Config) -> str:
    return cfg.secrets.OPENAI_API_KEY or ""


def _embedding_model(cfg: Config) -> str:
    return (cfg.rag.embedding_model or cfg.embeddings.model).strip()


def _embedding_profile(cfg: Config) -> str:
    explicit = cfg.rag.embedding_profile.strip()
    if explicit:
        return explicit
    model = _embedding_model(cfg).casefold()
    return "sfr" if "sfr" in model else ""


def _embedding_base_url(cfg: Config) -> str:
    return (cfg.rag.embedding_base_url or cfg.embeddings.base_url or "").strip()


def _llm_model(cfg: Config) -> str:
    return (cfg.rag.llm_model or cfg.models.reflection).strip()


def _llm_base_url(cfg: Config) -> str:
    return (cfg.rag.llm_base_url or cfg.llm.openai.base_url or "").strip()


def _rerank_base_url(cfg: Config) -> str:
    return (cfg.rag.rerank_base_url or "").strip() if cfg.rag.rerank_enabled else ""


def _rerank_model(cfg: Config) -> str:
    return cfg.rag.rerank_model.strip() if cfg.rag.rerank_enabled else ""


def _import_kb_services(cfg: Config):
    _ensure_vendor_imports(cfg)
    from ingestion.embedding_profiles import create_embedding_function
    from RAG.services.kb_service import append_kb, build_kb, load_kb
    from state.config import ENC

    return create_embedding_function, build_kb, append_kb, load_kb, ENC


def _import_retrieval_service(cfg: Config):
    _ensure_vendor_imports(cfg)
    from RAG.services.retrieval_service import retrieve_context

    return retrieve_context


def _save_tool_artifact(
    cfg: Config,
    session_id: str,
    tool_name: str,
    run_id: str | None,
    payload: dict[str, Any],
) -> str:
    artifact_id = run_id or ids.tool_run_id()
    rel = Path("artifacts") / session_id / "tool_runs" / tool_name / f"{artifact_id}.json"
    path = cfg.data_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return str(rel)


async def ingest_pdf_bytes_from_web_fetch(
    ctx: ToolCtx,
    *,
    url: str,
    pdf_bytes: bytes,
    title: str | None = None,
) -> dict[str, Any]:
    if not _can_auto_ingest_fetched_pdf(ctx):
        return {"enabled": False, "ingested": 0}
    return await ingest_pdf_payloads(
        ctx.cfg,
        ctx.session_id or "",
        [{"url": url, "title": title or url, "pdf_bytes": pdf_bytes}],
    )


_BACKGROUND_INGEST_TASKS: set[asyncio.Task] = set()
_BACKGROUND_INGEST_TASK_SESSIONS: dict[asyncio.Task, str] = {}


def _finish_background_ingest(task: asyncio.Task) -> None:
    _BACKGROUND_INGEST_TASKS.discard(task)
    _BACKGROUND_INGEST_TASK_SESSIONS.pop(task, None)
    with suppress(BaseException):
        task.result()


def _background_ingest_tasks_for_session(session_id: str) -> set[asyncio.Task]:
    for task in list(_BACKGROUND_INGEST_TASKS):
        if task.done():
            _finish_background_ingest(task)
    return {
        task
        for task in _BACKGROUND_INGEST_TASKS
        if _BACKGROUND_INGEST_TASK_SESSIONS.get(task) == session_id
    }


def background_ingest_pending(cfg: Config, session_id: str | None) -> bool:
    if not session_id or not rag_is_enabled(cfg):
        return False
    return bool(_background_ingest_tasks_for_session(session_id))


def background_ingest_status(cfg: Config, session_id: str | None) -> dict[str, Any]:
    if not session_id:
        return {"enabled": rag_is_enabled(cfg), "active_background_tasks": 0}
    active_tasks = (
        len(_background_ingest_tasks_for_session(session_id)) if rag_is_enabled(cfg) else 0
    )
    counts = _manifest_ingest_counts_sync(cfg, session_id)
    counts.update(
        {
            "enabled": rag_is_enabled(cfg),
            "active_background_tasks": active_tasks,
            "pending_background_ingest": active_tasks > 0,
        }
    )
    return counts


async def wait_for_background_ingest_step(
    cfg: Config,
    session_id: str,
    *,
    timeout_seconds: float = 1.0,
) -> dict[str, Any]:
    if not rag_is_enabled(cfg):
        return background_ingest_status(cfg, session_id)
    tasks = _background_ingest_tasks_for_session(session_id)
    if tasks:
        await asyncio.wait(
            tasks,
            timeout=max(0.0, float(timeout_seconds)),
            return_when=asyncio.FIRST_COMPLETED,
        )
    return background_ingest_status(cfg, session_id)


def schedule_arxiv_ingest(ctx: ToolCtx, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not (
        ctx.session_id
        and ctx.cfg.rag.enabled
        and ctx.cfg.rag.auto_ingest_arxiv_pdfs
        and _package_root(ctx.cfg).is_dir()
    ):
        return {"enabled": False, "scheduled": False}
    loop = asyncio.get_running_loop()
    task = loop.create_task(ingest_arxiv_records(ctx, records))
    _BACKGROUND_INGEST_TASKS.add(task)
    _BACKGROUND_INGEST_TASK_SESSIONS[task] = ctx.session_id
    task.add_done_callback(_finish_background_ingest)
    return {
        "enabled": True,
        "scheduled": True,
        "candidate_records": len(records),
        "max_pdfs_per_search": int(ctx.cfg.rag.auto_ingest_max_pdfs_per_search),
        "max_session_papers": int(ctx.cfg.rag.max_session_papers),
    }


def schedule_chemrxiv_ingest(ctx: ToolCtx, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Schedule ChemRxiv PDF ingestion using the shared preprint ingestion settings."""
    result = schedule_arxiv_ingest(ctx, records)
    if isinstance(result, dict):
        result["provider"] = "chemrxiv"
    return result


def schedule_biorxiv_ingest(ctx: ToolCtx, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Schedule bioRxiv PDF ingestion using the shared preprint ingestion settings."""
    result = schedule_arxiv_ingest(ctx, records)
    if isinstance(result, dict):
        result["provider"] = "biorxiv"
    return result


async def _dedupe_literature_review_records(
    ctx: ToolCtx,
    records: list[dict[str, Any]],
    *,
    provider: str,
    query: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dedupe_enabled = bool(
        ctx.session_id
        and ctx.cfg.rag.literature_review_enabled
        and ctx.cfg.rag.literature_review_dedupe_reviewed_results
    )
    if not dedupe_enabled:
        return records, {
            "dedupe_enabled": dedupe_enabled,
            "original_candidate_records": len(records),
            "new_candidate_records": len(records),
            "deduped_records": 0,
        }
    return await asyncio.to_thread(
        _dedupe_literature_review_records_sync,
        ctx.cfg,
        ctx.session_id,
        records,
        provider,
        query,
    )


def _dedupe_literature_review_records_sync(
    cfg: Config,
    session_id: str,
    records: list[dict[str, Any]],
    provider: str,
    query: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _literature_review_memory_path(cfg, session_id)
    with _session_lock(session_id):
        memory = _read_literature_review_memory(path)
        known_keys = set(memory["keys"])
        new_records: list[dict[str, Any]] = []
        deduped: list[dict[str, Any]] = []

        for record in records:
            record_keys = _literature_review_record_keys(record)
            duplicate_keys = [key for key in record_keys if key in known_keys]
            if duplicate_keys:
                first_key = duplicate_keys[0]
                deduped.append(
                    {
                        "title": str(record.get("title") or "")[:200],
                        "matched_key": first_key,
                        "first_seen_as": memory["keys"].get(first_key, first_key),
                    }
                )
                continue

            new_records.append(record)

    return new_records, {
        "dedupe_enabled": True,
        "original_candidate_records": len(records),
        "new_candidate_records": len(new_records),
        "deduped_records": len(deduped),
        "deduped_record_examples": deduped[:5],
    }


async def _remember_literature_review_records(
    ctx: ToolCtx,
    records: list[dict[str, Any]],
    *,
    provider: str,
    query: str,
    review_meta: dict[str, Any],
) -> None:
    if not (
        ctx.session_id
        and records
        and ctx.cfg.rag.literature_review_enabled
        and ctx.cfg.rag.literature_review_dedupe_reviewed_results
        and _should_advance_literature_review_memory(review_meta)
    ):
        return
    await asyncio.to_thread(
        _remember_literature_review_records_sync,
        ctx.cfg,
        ctx.session_id,
        records,
        provider,
        query,
    )


def _should_advance_literature_review_memory(review_meta: dict[str, Any]) -> bool:
    return review_meta.get("reviewed") is True


def _remember_literature_review_records_sync(
    cfg: Config,
    session_id: str,
    records: list[dict[str, Any]],
    provider: str,
    query: str,
) -> None:
    path = _literature_review_memory_path(cfg, session_id)
    with _session_lock(session_id):
        memory = _read_literature_review_memory(path)
        known_keys = set(memory["keys"])
        now = datetime.now(UTC).isoformat()
        changed = False

        for record in records:
            record_keys = _literature_review_record_keys(record)
            if not record_keys:
                continue
            primary_key = next((key for key in record_keys if key in known_keys), record_keys[0])
            for key in record_keys:
                if memory["keys"].get(key) != primary_key:
                    memory["keys"][key] = primary_key
                    changed = True
                known_keys.add(key)
            if primary_key not in memory["records"]:
                memory["records"][primary_key] = {
                    "title": str(record.get("title") or "")[:500],
                    "provider": provider,
                    "query": query,
                    "first_seen_at": now,
                    "keys": record_keys[:12],
                    "pdf_url": str(record.get("pdf_url") or ""),
                    "url": str(record.get("abs_url") or record.get("url") or ""),
                }
                changed = True

        if changed:
            _write_literature_review_memory(path, memory)


def _literature_review_memory_path(cfg: Config, session_id: str) -> Path:
    return cfg.session_artifact_dir(session_id) / "literature_review" / "reviewed_records.json"


def _read_literature_review_memory(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    keys = data.setdefault("keys", {})
    records = data.setdefault("records", {})
    if not isinstance(keys, dict):
        data["keys"] = {}
    if not isinstance(records, dict):
        data["records"] = {}
    data["version"] = 1
    return data


def _write_literature_review_memory(path: Path, memory: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(memory, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    tmp.replace(path)


def _literature_review_record_keys(record: dict[str, Any]) -> list[str]:
    keys: list[str] = []

    def add(key: str | None) -> None:
        if key and key not in keys:
            keys.append(key)

    for field in ("pdf_url", "abs_url", "url", "pubmed_url"):
        add(_review_url_key(str(record.get(field) or "")))

    add(_identifier_key("doi", record.get("doi")))
    add(_identifier_key("pmid", record.get("pmid")))
    add(_identifier_key("pmcid", record.get("pmcid")))
    add(_identifier_key("arxiv", record.get("arxiv_id")))
    add(_identifier_key("source", record.get("source_id") or record.get("id")))

    title_key = _title_key(str(record.get("title") or ""))
    add(f"title:{title_key}" if title_key else None)
    return keys


def _identifier_key(prefix: str, value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    if prefix == "doi":
        text = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", text).strip()
    if prefix == "arxiv":
        text = re.sub(r"^(arxiv:|https?://arxiv\.org/(abs|pdf)/)", "", text).strip()
        text = re.sub(r"\.pdf$", "", text).strip()
        text = re.sub(r"v\d+$", "", text).strip()
    text = re.sub(r"\s+", "", text)
    return f"{prefix}:{text}" if text else None


def _review_url_key(url: str) -> str | None:
    value = url.strip()
    if not value.startswith(("http://", "https://")):
        return None
    try:
        parts = urlsplit(value)
    except ValueError:
        return None
    netloc = parts.netloc.casefold()
    path = re.sub(r"/+", "/", parts.path).rstrip("/")
    if netloc == "arxiv.org":
        match = re.search(r"/(abs|pdf)/([^/?#]+)", path)
        if match:
            return _identifier_key("arxiv", match.group(2))
    return "url:" + urlunsplit((parts.scheme.casefold(), netloc, path, "", ""))


async def review_search_payload(
    ctx: ToolCtx,
    payload: dict[str, Any],
    *,
    provider: str,
) -> dict[str, Any]:
    """Filter search results through LiteratureReviewAgent before model context."""
    results = payload.get("results")
    if not isinstance(results, list):
        return payload

    records = [item for item in results if isinstance(item, dict)]
    if not records:
        return payload

    should_schedule = bool(ctx.session_id and ctx.cfg.rag.enabled)
    should_review_context = bool(ctx.llm_client is not None and ctx.session_id)
    if not should_review_context and not should_schedule:
        return payload

    query = str(payload.get("query") or "")
    review_records = records
    dedupe_meta = {
        "dedupe_enabled": False,
        "original_candidate_records": len(records),
        "new_candidate_records": len(records),
        "deduped_records": 0,
    }
    if should_review_context:
        review_records, dedupe_meta = await _dedupe_literature_review_records(
            ctx,
            records,
            provider=provider,
            query=query,
        )

    if should_review_context and not review_records:
        selection = None
        review_meta = {
            "enabled": bool(getattr(ctx.cfg.rag, "literature_review_enabled", False)),
            "provider": provider,
            "query": query,
            "candidate_records": 0,
            "reviewed_candidates": 0,
            "reviewed": False,
            "selected": 0,
            "selected_pdfs": 0,
            "skipped": "all_candidates_previously_reviewed",
            **dedupe_meta,
        }
    else:
        try:
            from ..agents.literature_review import select_search_records

            selection = await select_search_records(
                cfg=ctx.cfg,
                llm_client=ctx.llm_client,
                session_id=ctx.session_id,
                task_id=ctx.task_id,
                query=query,
                provider=provider,
                records=review_records,
                max_pdfs=int(ctx.cfg.rag.auto_ingest_max_pdfs_per_search),
                caller=ctx.extra,
            )
        except Exception as exc:
            selection = None
            review_meta = {
                "enabled": bool(getattr(ctx.cfg.rag, "literature_review_enabled", False)),
                "provider": provider,
                "query": query,
                "candidate_records": len(review_records),
                "reviewed": False,
                "selected": 0,
                "selected_pdfs": 0,
                "error": str(exc)[:300],
                **dedupe_meta,
            }
        else:
            review_meta = {**selection.metadata, **dedupe_meta}
            await _remember_literature_review_records(
                ctx,
                review_records,
                provider=provider,
                query=query,
                review_meta=review_meta,
            )

    selected_records = selection.selected_records if selection is not None else []
    selected_pdf_records = selection.selected_pdf_records if selection is not None else []

    out = dict(payload)
    out["results"] = selected_records
    out["n"] = len(selected_records)
    out["total_results"] = len(records)
    out["literature_review"] = review_meta
    out["model_context_note"] = (
        "Search results shown here were filtered by LiteratureReviewAgent. "
        "Full raw search results remain stored in the search cache/artifacts."
    )

    if should_schedule:
        out["rag_ingest"] = _schedule_reviewed_pdf_ingest(
            ctx,
            selected_pdf_records,
            provider=provider,
            candidate_count=len(records),
            review_meta=review_meta,
        )
    else:
        out["rag_ingest"] = {
            "enabled": bool(ctx.cfg.rag.enabled and ctx.cfg.rag.auto_ingest_arxiv_pdfs),
            "scheduled": False,
            "candidate_records": len(records),
            "provider": provider,
            "literature_review": review_meta,
        }
    return out


async def review_and_schedule_pdf_ingest(
    ctx: ToolCtx,
    records: list[dict[str, Any]],
    *,
    provider: str,
    query: str = "",
) -> dict[str, Any]:
    """Review search records and return only the PDF-ingest scheduling status."""
    payload = {"query": query, "n": len(records), "results": records}
    reviewed = await review_search_payload(ctx, payload, provider=provider)
    ingest = reviewed.get("rag_ingest")
    if isinstance(ingest, dict):
        return ingest
    return {"enabled": False, "scheduled": False}


def _schedule_reviewed_pdf_ingest(
    ctx: ToolCtx,
    records: list[dict[str, Any]],
    *,
    provider: str,
    candidate_count: int,
    review_meta: dict[str, Any],
) -> dict[str, Any]:
    if not records:
        return {
            "enabled": True,
            "scheduled": False,
            "candidate_records": candidate_count,
            "max_pdfs_per_search": int(ctx.cfg.rag.auto_ingest_max_pdfs_per_search),
            "max_session_papers": int(ctx.cfg.rag.max_session_papers),
            "provider": provider,
            "literature_review": review_meta,
        }
    result = _schedule_pdf_ingest_for_provider(ctx, records, provider=provider)
    if isinstance(result, dict):
        result["literature_review"] = review_meta
        result["review_candidate_records"] = candidate_count
        result["review_selected_records"] = len(records)
    return result


def _schedule_pdf_ingest_for_provider(
    ctx: ToolCtx,
    records: list[dict[str, Any]],
    *,
    provider: str,
) -> dict[str, Any]:
    normalized = provider.lower().strip()
    if normalized == "chemrxiv":
        return schedule_chemrxiv_ingest(ctx, records)
    if normalized == "biorxiv":
        return schedule_biorxiv_ingest(ctx, records)
    result = schedule_arxiv_ingest(ctx, records)
    if isinstance(result, dict) and normalized:
        result["provider"] = normalized
    return result


def _can_auto_ingest_fetched_pdf(ctx: ToolCtx) -> bool:
    return bool(
        ctx.session_id
        and ctx.cfg.rag.enabled
        and ctx.cfg.rag.auto_ingest_fetched_pdfs
        and _package_root(ctx.cfg).is_dir()
    )


async def ingest_arxiv_records(ctx: ToolCtx, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not (
        ctx.session_id
        and ctx.cfg.rag.enabled
        and ctx.cfg.rag.auto_ingest_arxiv_pdfs
        and _package_root(ctx.cfg).is_dir()
    ):
        return {"enabled": False, "downloaded": 0, "ingested": 0}

    limit = max(0, int(ctx.cfg.rag.auto_ingest_max_pdfs_per_search))
    if limit <= 0:
        return {"enabled": True, "downloaded": 0, "ingested": 0, "skipped": "limit_zero"}

    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        url = str(record.get("pdf_url") or "").strip()
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        candidates.append({"url": url, "title": str(record.get("title") or url)})
        if len(candidates) >= limit:
            break

    if not candidates:
        return {"enabled": True, "downloaded": 0, "ingested": 0}

    round_key = _ingest_round_key(ctx)
    round_max_papers = _ingest_round_max_papers(ctx.cfg, round_key)
    reservation = await asyncio.to_thread(
        _reserve_pdf_candidates_sync,
        ctx.cfg,
        ctx.session_id,
        candidates,
        round_key=round_key,
        round_max_papers=round_max_papers,
    )
    to_download = reservation["candidates"]
    if not to_download:
        return {
            "enabled": True,
            "downloaded": 0,
            "ingested": 0,
            "skipped": "already_reserved_or_capped",
            "reservation": {k: v for k, v in reservation.items() if k != "candidates"},
        }

    payloads: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    timeout = max(10.0, float(ctx.cfg.web_fetch.timeout_seconds))
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for candidate in to_download:
            try:
                async with _rate_limited_pdf_stream(
                    client,
                    candidate["url"],
                    headers={"User-Agent": ctx.cfg.web_fetch.user_agent},
                ) as response:
                    if response.status_code >= 400:
                        failures.append(
                            {"url": candidate["url"], "error": f"HTTP {response.status_code}"}
                        )
                        continue
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > ctx.cfg.web_fetch.max_bytes:
                            failures.append(
                                {"url": candidate["url"], "error": "response too large"}
                            )
                            chunks = []
                            break
                        chunks.append(chunk)
                    if chunks:
                        payloads.append({**candidate, "pdf_bytes": b"".join(chunks)})
            except httpx.HTTPError as exc:
                failures.append({"url": candidate["url"], "error": str(exc)[:200]})

    if failures:
        await asyncio.to_thread(
            _release_reserved_candidates_sync,
            ctx.cfg,
            ctx.session_id,
            [item["url"] for item in failures if "url" in item],
        )
    result = await ingest_pdf_payloads(ctx.cfg, ctx.session_id, payloads)
    result.update(
        {
            "downloaded": len(payloads),
            "download_failures": failures[:10],
            "reservation": {k: v for k, v in reservation.items() if k != "candidates"},
        }
    )
    return result


def _is_arxiv_pdf_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.startswith(
        (
            "https://arxiv.org/pdf/",
            "http://arxiv.org/pdf/",
            "https://www.arxiv.org/pdf/",
            "http://www.arxiv.org/pdf/",
            "https://export.arxiv.org/pdf/",
            "http://export.arxiv.org/pdf/",
        )
    )


def _is_biorxiv_pdf_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.startswith(
        (
            "https://www.biorxiv.org/content/",
            "http://www.biorxiv.org/content/",
            "https://biorxiv.org/content/",
            "http://biorxiv.org/content/",
        )
    ) and lowered.endswith(".full.pdf")


def _rate_limited_pdf_stream(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
):
    if _is_arxiv_pdf_url(url):
        return _ArxivRateLimitedStream(client, url, headers=headers)
    if _is_biorxiv_pdf_url(url):
        return _BiorxivRateLimitedStream(client, url, headers=headers)
    return client.stream("GET", url, headers=headers)


_BIORXIV_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)


def _biorxiv_landing_url(pdf_url: str) -> str:
    lowered = pdf_url.lower()
    if lowered.endswith(".full.pdf"):
        return pdf_url[: -len(".full.pdf")]
    return pdf_url


def _biorxiv_landing_headers(base_headers: dict[str, str]) -> dict[str, str]:
    headers = dict(base_headers)
    headers["User-Agent"] = _BIORXIV_BROWSER_USER_AGENT
    headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    headers.setdefault("Accept-Language", "en-US,en;q=0.9")
    return headers


def _biorxiv_pdf_headers(base_headers: dict[str, str], *, referer: str) -> dict[str, str]:
    headers = dict(base_headers)
    headers["User-Agent"] = _BIORXIV_BROWSER_USER_AGENT
    headers["Accept"] = "application/pdf,*/*"
    headers["Referer"] = referer
    headers.setdefault("Accept-Language", "en-US,en;q=0.9")
    return headers


class _ArxivRateLimitedStream:
    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        headers: dict[str, str],
    ) -> None:
        self._client = client
        self._url = url
        self._headers = headers
        self._stream_cm = None
        self._limiter = None
        self._lock = None

    async def __aenter__(self):
        from .builtins import arxiv as arxiv_mod

        interval = max(0.0, arxiv_mod._ARXIV_MIN_REQUEST_INTERVAL_SECONDS)
        if interval > 0:
            self._lock = arxiv_mod._arxiv_request_lock()
            await self._lock.acquire()
            self._limiter = await asyncio.to_thread(arxiv_mod._acquire_arxiv_rate_limit, interval)
        self._stream_cm = self._client.stream("GET", self._url, headers=self._headers)
        try:
            return await self._stream_cm.__aenter__()
        except Exception:
            await self._release()
            raise

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._stream_cm is not None:
                return await self._stream_cm.__aexit__(exc_type, exc, tb)
            return None
        finally:
            await self._release()

    async def _release(self) -> None:
        from .builtins import arxiv as arxiv_mod

        limiter = self._limiter
        self._limiter = None
        if limiter is not None:
            await asyncio.to_thread(arxiv_mod._release_arxiv_rate_limit, limiter)
        lock = self._lock
        self._lock = None
        if lock is not None:
            lock.release()


class _BiorxivRateLimitedStream:
    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        headers: dict[str, str],
    ) -> None:
        self._client = client
        self._url = url
        self._headers = headers
        self._stream_cm = None
        self._limiter = None
        self._lock = None

    async def __aenter__(self):
        from .builtins import biorxiv as biorxiv_mod

        interval = max(0.0, biorxiv_mod._BIORXIV_MIN_REQUEST_INTERVAL_SECONDS)
        if interval > 0:
            self._lock = biorxiv_mod._biorxiv_request_lock()
            await self._lock.acquire()
            self._limiter = await asyncio.to_thread(
                biorxiv_mod._acquire_biorxiv_rate_limit, interval
            )
        referer = _biorxiv_landing_url(self._url)
        try:
            landing = await self._client.get(
                referer,
                headers=_biorxiv_landing_headers(self._headers),
            )
            if landing.status_code < 400:
                referer = str(landing.url)
        except httpx.HTTPError:
            pass

        self._stream_cm = self._client.stream(
            "GET",
            self._url,
            headers=_biorxiv_pdf_headers(self._headers, referer=referer),
        )
        try:
            return await self._stream_cm.__aenter__()
        except Exception:
            await self._release()
            raise

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._stream_cm is not None:
                return await self._stream_cm.__aexit__(exc_type, exc, tb)
            return None
        finally:
            await self._release()

    async def _release(self) -> None:
        from .builtins import biorxiv as biorxiv_mod

        limiter = self._limiter
        self._limiter = None
        if limiter is not None:
            await asyncio.to_thread(biorxiv_mod._release_biorxiv_rate_limit, limiter)
        lock = self._lock
        self._lock = None
        if lock is not None:
            lock.release()


def _reserve_pdf_candidates_sync(
    cfg: Config,
    session_id: str,
    candidates: list[dict[str, str]],
    *,
    round_key: str | None = None,
    round_max_papers: int | None = None,
) -> dict[str, Any]:
    paths = _paths(cfg, session_id)
    with _session_lock(session_id):
        paths.pdf_dir.mkdir(parents=True, exist_ok=True)
        manifest = _read_manifest(paths.manifest_path)
        papers: dict[str, Any] = manifest.setdefault("papers", {})
        current_count = max(
            _session_added_paper_count(papers),
            len(list(paths.pdf_dir.glob("*.pdf"))),
        )
        max_papers = max(1, int(cfg.rag.max_session_papers))
        round_count = _ingest_round_paper_count(papers, round_key)
        selected: list[dict[str, str]] = []
        skipped_existing = 0
        skipped_existing_title = 0
        skipped_cap = 0
        skipped_round_cap = 0
        existing_title_keys = _existing_title_keys(papers)
        now = datetime.now(UTC).isoformat()
        for candidate in candidates:
            url = candidate["url"]
            key = _source_key(url)
            existing = papers.get(key)
            if isinstance(existing, dict):
                skipped_existing += 1
                continue
            title = candidate.get("title") or url
            title_key = _title_key(title)
            if title_key and title_key in existing_title_keys:
                skipped_existing_title += 1
                continue
            if current_count >= max_papers:
                skipped_cap += 1
                continue
            if round_max_papers is not None and round_count >= round_max_papers:
                skipped_round_cap += 1
                continue
            filename = _pdf_filename(url, title)
            papers[key] = {
                "url": url,
                "title": title,
                "title_key": title_key,
                "file": filename,
                "indexed": False,
                "reserved": True,
                "reserved_at": now,
            }
            if round_key:
                papers[key]["ingest_round"] = round_key
            selected.append(candidate)
            if title_key:
                existing_title_keys.add(title_key)
            current_count += 1
            if round_key:
                round_count += 1
        _write_manifest(paths.manifest_path, manifest)
        return {
            "candidates": selected,
            "reserved": len(selected),
            "skipped_existing": skipped_existing,
            "skipped_existing_title": skipped_existing_title,
            "skipped_cap": skipped_cap,
            "skipped_round_cap": skipped_round_cap,
            "round_key": round_key or "",
            "round_max_papers": round_max_papers or 0,
            "round_paper_count": round_count if round_key else 0,
            **_paper_count_fields(papers),
            "pdf_file_count": len(list(paths.pdf_dir.glob("*.pdf"))),
        }


def _ingest_round_key(ctx: ToolCtx) -> str | None:
    if not isinstance(ctx.extra, dict):
        return None
    if ctx.extra.get("agent") == "generation" and ctx.extra.get("mode") == "literature_discovery":
        return "generation:literature_discovery"
    return None


def _ingest_round_max_papers(cfg: Config, round_key: str | None) -> int | None:
    if round_key != "generation:literature_discovery":
        return None
    raw = int(getattr(cfg.rag, "generation_discovery_max_pdfs_per_round", 0) or 0)
    return raw if raw > 0 else None


def _ingest_round_paper_count(papers: dict[str, Any], round_key: str | None) -> int:
    if not round_key:
        return 0
    return sum(
        1
        for item in papers.values()
        if isinstance(item, dict) and item.get("ingest_round") == round_key
    )


def _release_reserved_candidates_sync(cfg: Config, session_id: str, urls: list[str]) -> None:
    if not urls:
        return
    paths = _paths(cfg, session_id)
    with _session_lock(session_id):
        manifest = _read_manifest(paths.manifest_path)
        papers = manifest.get("papers", {})
        if not isinstance(papers, dict):
            return
        for url in urls:
            key = _source_key(url)
            item = papers.get(key)
            if isinstance(item, dict) and item.get("reserved") and not item.get("indexed"):
                file_name = item.get("file")
                if isinstance(file_name, str):
                    with suppress(OSError):
                        (paths.pdf_dir / file_name).unlink(missing_ok=True)
                papers.pop(key, None)
        _write_manifest(paths.manifest_path, manifest)


async def ingest_pdf_payloads(
    cfg: Config,
    session_id: str,
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    if not payloads:
        return {"enabled": bool(cfg.rag.enabled), "ingested": 0}
    if not (cfg.rag.enabled and session_id and _package_root(cfg).is_dir()):
        return {"enabled": False, "ingested": 0}
    return await asyncio.to_thread(_ingest_pdf_payloads_sync, cfg, session_id, payloads)


def _ingest_pdf_payloads_sync(
    cfg: Config,
    session_id: str,
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    paths = _paths(cfg, session_id)
    started = time.monotonic()
    with _session_lock(session_id):
        paths.pdf_dir.mkdir(parents=True, exist_ok=True)
        manifest = _read_manifest(paths.manifest_path)
        papers: dict[str, Any] = manifest.setdefault("papers", {})
        max_papers = max(1, int(cfg.rag.max_session_papers))
        session_paper_count = _session_added_paper_count(papers)

        new_entries: list[tuple[str, Path]] = []
        skipped_duplicates = 0
        skipped_duplicate_titles = 0
        skipped_cap = 0
        existing_title_keys = _existing_title_keys(papers)
        for payload in payloads:
            url = str(payload.get("url") or "").strip()
            pdf_bytes = payload.get("pdf_bytes")
            if not url or not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
                continue
            key = _source_key(url)
            title = str(payload.get("title") or url)
            title_key = _title_key(title)
            existing = papers.get(key)
            if isinstance(existing, dict) and existing.get("indexed"):
                skipped_duplicates += 1
                continue
            if not isinstance(existing, dict) and title_key and title_key in existing_title_keys:
                skipped_duplicate_titles += 1
                continue
            is_new_paper = key not in papers
            if is_new_paper and session_paper_count >= max_papers:
                skipped_cap += 1
                continue
            filename = _pdf_filename(url, title)
            pdf_path = paths.pdf_dir / filename
            if not pdf_path.exists():
                pdf_path.write_bytes(bytes(pdf_bytes))
            entry = {
                "url": url,
                "title": title,
                "title_key": title_key,
                "file": filename,
                "bytes": len(pdf_bytes),
                "indexed": False,
                "reserved": False,
                "stored_at": datetime.now(UTC).isoformat(),
            }
            if isinstance(existing, dict):
                for field in ("ingest_round", "reserved_at"):
                    if existing.get(field):
                        entry[field] = existing[field]
            papers[key] = entry
            if is_new_paper:
                session_paper_count += 1
            new_entries.append((key, pdf_path))
            if title_key:
                existing_title_keys.add(title_key)

        if not new_entries:
            _write_manifest(paths.manifest_path, manifest)
            return {
                "enabled": True,
                "ingested": 0,
                "skipped_duplicates": skipped_duplicates,
                "skipped_duplicate_titles": skipped_duplicate_titles,
                "skipped_cap": skipped_cap,
                **_paper_count_fields(papers),
            }

        create_embedding_function, build_kb, append_kb, _load_kb, enc = _import_kb_services(cfg)
        embedding_model = _embedding_model(cfg)
        embedding_profile = _embedding_profile(cfg)
        embedding_base_url = _embedding_base_url(cfg)
        llm_base_url = _llm_base_url(cfg)
        llm_model = _llm_model(cfg)
        api_key = _rag_api_key(cfg)

        warnings: list[str] = []
        built_new = not (paths.index_path.is_file() and paths.meta_path.is_file())
        try:
            if built_new:
                embeddings = create_embedding_function(
                    model=embedding_model,
                    profile_name=embedding_profile or None,
                    api_key=api_key,
                    base_url=embedding_base_url,
                    fallback_to_env=False,
                )
                result = build_kb(
                    client=None,
                    embeddings=embeddings,
                    enc=enc,
                    pdf_dir=str(paths.pdf_dir),
                    index_path=str(paths.index_path),
                    meta_path=str(paths.meta_path),
                    graphrag_dir=str(paths.graphrag_dir) if cfg.rag.run_graphrag else "",
                    embedding_model=embedding_model,
                    embedding_profile=embedding_profile or None,
                    run_graphrag=bool(cfg.rag.run_graphrag),
                    api_key=api_key,
                    base_url=llm_base_url,
                    embedding_base_url=embedding_base_url,
                    chat_model=llm_model,
                )
                indexed_at = datetime.now(UTC).isoformat()
                for item in papers.values():
                    if not isinstance(item, dict):
                        continue
                    file_name = item.get("file")
                    if not isinstance(file_name, str) or not (paths.pdf_dir / file_name).is_file():
                        continue
                    item["indexed"] = True
                    item["reserved"] = False
                    item["indexed_at"] = indexed_at
            else:
                append_dir = paths.root / "append" / ids.tool_run_id()
                append_dir.mkdir(parents=True, exist_ok=True)
                try:
                    for _key, pdf_path in new_entries:
                        shutil.copy2(pdf_path, append_dir / pdf_path.name)
                    result = append_kb(
                        client=None,
                        enc=enc,
                        index_path=str(paths.index_path),
                        meta_path=str(paths.meta_path),
                        append_folder=str(append_dir),
                        graphrag_dir=str(paths.graphrag_dir) if cfg.rag.run_graphrag else "",
                        run_graphrag=bool(cfg.rag.run_graphrag),
                        api_key=api_key,
                        base_url=llm_base_url,
                        embedding_base_url=embedding_base_url,
                        chat_model=llm_model,
                    )
                finally:
                    shutil.rmtree(append_dir, ignore_errors=True)
                for key, _pdf_path in new_entries:
                    papers[key]["indexed"] = True
                    papers[key]["reserved"] = False
                    papers[key]["indexed_at"] = datetime.now(UTC).isoformat()
            warnings = list(result.get("warnings") or [])
        except Exception as exc:
            _write_manifest(paths.manifest_path, manifest)
            return {
                "enabled": True,
                "ingested": 0,
                "error": str(exc)[:500],
                "skipped_duplicates": skipped_duplicates,
                "skipped_duplicate_titles": skipped_duplicate_titles,
                "skipped_cap": skipped_cap,
                **_paper_count_fields(papers),
            }

        _update_manifest_kb_summary(manifest, result=result, built_new=built_new)
        _write_manifest(paths.manifest_path, manifest)
        return {
            "enabled": True,
            "ingested": len(new_entries),
            "built_new_index": built_new,
            **_paper_count_fields(papers),
            "skipped_duplicates": skipped_duplicates,
            "skipped_duplicate_titles": skipped_duplicate_titles,
            "skipped_cap": skipped_cap,
            "index_path": str(paths.index_path),
            "meta_path": str(paths.meta_path),
            "warnings": warnings[:10],
            "duration_ms": int((time.monotonic() - started) * 1000),
        }


def _update_manifest_kb_summary(
    manifest: dict[str, Any],
    *,
    result: dict[str, Any],
    built_new: bool,
) -> None:
    current = manifest.get("kb") if isinstance(manifest.get("kb"), dict) else {}
    seed = manifest.get("seed") if isinstance(manifest.get("seed"), dict) else {}
    seed_kb = seed.get("kb") if isinstance(seed.get("kb"), dict) else {}
    base_raw = current.get("chunk_count", seed_kb.get("chunk_count", 0))
    base_count = base_raw if isinstance(base_raw, int) else 0
    if built_new:
        total_raw = result.get("total_chunks")
        chunk_count = total_raw if isinstance(total_raw, int) else 0
    else:
        added_raw = result.get("new_chunks")
        added_count = added_raw if isinstance(added_raw, int) else 0
        chunk_count = base_count + added_count
    if chunk_count <= 0:
        return
    summary = dict(seed_kb)
    summary.update(current)
    summary["status"] = "ok"
    summary["chunk_count"] = chunk_count
    manifest["kb"] = summary


class RAGRetrieveContextTool:
    name = "rag_retrieve_context"
    description = (
        "Retrieve concise, reranked context from the session RAG knowledge base of fetched "
        "full-text PDFs. Use this before reviewing or evolving hypotheses when literature "
        "context is needed without reading whole papers into the prompt. The retrieval "
        "strategy is fixed by the session configuration."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Focused retrieval query."},
            "top_k_faiss": {"type": "integer", "minimum": 1, "maximum": 100},
            "diversity": {"type": "number", "minimum": 0, "maximum": 1},
            "max_chars": {"type": "integer", "minimum": 1000, "maximum": 50000},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        if not ctx.session_id:
            return ToolResult(
                is_error=True, error_message="rag_retrieve_context requires a session"
            )
        if not rag_is_enabled(ctx.cfg):
            return ToolResult(is_error=True, error_message="RAG is not enabled/configured")
        t0 = time.monotonic()
        try:
            payload = await asyncio.to_thread(_retrieve_context_sync, ctx.cfg, ctx.session_id, args)
        except Exception as exc:
            return ToolResult(
                is_error=True,
                error_message=f"RAG retrieval failed: {str(exc)[:500]}",
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        artifact_path = _save_tool_artifact(ctx.cfg, ctx.session_id, self.name, ctx.run_id, payload)
        payload["artifact_path"] = artifact_path
        return ToolResult(
            content=payload,
            artifact_path=artifact_path,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(json.dumps(payload, ensure_ascii=False, default=str)),
        )


class RAGStatusTool:
    name = "rag_kb_status"
    description = "Inspect the session RAG knowledge base status and indexed paper count."
    input_schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        if not ctx.session_id:
            return ToolResult(is_error=True, error_message="rag_kb_status requires a session")
        payload = await asyncio.to_thread(_kb_status_sync, ctx.cfg, ctx.session_id)
        return ToolResult(content=payload, result_bytes=len(json.dumps(payload, default=str)))


def _kb_status_sync(cfg: Config, session_id: str) -> dict[str, Any]:
    paths = _paths(cfg, session_id)
    counts = _manifest_ingest_counts_sync(cfg, session_id)
    manifest = _read_manifest(paths.manifest_path)
    seed = manifest.get("seed")
    out: dict[str, Any] = {
        "enabled": rag_is_enabled(cfg),
        "index_path": str(paths.index_path),
        "meta_path": str(paths.meta_path),
        "index_exists": paths.index_path.is_file(),
        "meta_exists": paths.meta_path.is_file(),
        "active_background_tasks": len(_background_ingest_tasks_for_session(session_id)),
        "seeded": isinstance(seed, dict),
        "seed": seed if isinstance(seed, dict) else None,
    }
    out.update(counts)
    out["pending_background_ingest"] = out["active_background_tasks"] > 0
    if paths.index_path.is_file() and paths.meta_path.is_file():
        try:
            _create_embedding_function, _build_kb, _append_kb, load_kb, _enc = _import_kb_services(
                cfg
            )
            out["kb"] = load_kb(
                index_path=str(paths.index_path),
                meta_path=str(paths.meta_path),
                graphrag_dir=str(paths.graphrag_dir) if cfg.rag.use_graphrag else "",
            )
        except Exception as exc:
            out["load_error"] = str(exc)[:300]
    return out


def _manifest_ingest_counts_sync(cfg: Config, session_id: str) -> dict[str, Any]:
    paths = _paths(cfg, session_id)
    manifest = _read_manifest(paths.manifest_path)
    papers = manifest.get("papers", {}) if isinstance(manifest.get("papers"), dict) else {}
    indexed = 0
    reserved = 0
    unindexed = 0
    downloaded_unindexed = 0
    for item in papers.values():
        if not isinstance(item, dict):
            continue
        is_indexed = bool(item.get("indexed"))
        if is_indexed:
            indexed += 1
            continue
        unindexed += 1
        if item.get("reserved"):
            reserved += 1
        file_name = item.get("file")
        if isinstance(file_name, str) and (paths.pdf_dir / file_name).is_file():
            downloaded_unindexed += 1
    seed = manifest.get("seed") if isinstance(manifest.get("seed"), dict) else {}
    seed_kb = seed.get("kb") if isinstance(seed.get("kb"), dict) else {}
    seed_chunk_count = int(seed_kb.get("chunk_count") or 0)
    seed_kb_ready = bool(
        seed_kb.get("status") == "ok"
        and seed_chunk_count > 0
        and paths.index_path.is_file()
        and paths.meta_path.is_file()
    )
    return {
        **_paper_count_fields(papers),
        "indexed_paper_count": indexed,
        "seed_chunk_count": seed_chunk_count,
        "seed_kb_ready": seed_kb_ready,
        "reserved_paper_count": reserved,
        "unindexed_paper_count": unindexed,
        "downloaded_unindexed_paper_count": downloaded_unindexed,
    }


def _retrieve_context_sync(cfg: Config, session_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with _session_lock(session_id):
        return _retrieve_context_unlocked(cfg, session_id, args)


def _retrieve_context_unlocked(
    cfg: Config, session_id: str, args: dict[str, Any]
) -> dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    paths = _paths(cfg, session_id)
    status = _kb_status_sync(cfg, session_id)
    if not (paths.index_path.is_file() and paths.meta_path.is_file()):
        return {
            "context_text": "",
            "sources": [],
            "message": "No RAG knowledge base has been built for this session yet.",
            "kb_status": status,
        }

    retrieve_context = _import_retrieval_service(cfg)
    # Retrieval strategy is a session-level policy. Never allow an agent-emitted
    # argument (including one from a stale prompt or permissive provider) to
    # override the frozen configuration.
    method = str(cfg.rag.retrieval_method or "faiss_mmr")
    top_k = int(args.get("top_k_faiss") or cfg.rag.top_k_faiss)
    diversity = float(
        args.get("diversity") if args.get("diversity") is not None else cfg.rag.diversity
    )
    max_chars = int(args.get("max_chars") or cfg.rag.context_max_chars)

    raw = retrieve_context(
        query=query,
        index_path=str(paths.index_path),
        meta_path=str(paths.meta_path),
        graphrag_dir=str(paths.graphrag_dir) if cfg.rag.use_graphrag else "",
        api_key=_rag_api_key(cfg),
        base_url=_llm_base_url(cfg),
        embedding_base_url=_embedding_base_url(cfg),
        rerank_base_url=_rerank_base_url(cfg),
        rerank_model=_rerank_model(cfg),
        rerank_top_k=int(cfg.rag.rerank_top_k),
        diversity=diversity,
        top_k_faiss=top_k,
        retrieval_method=method,
        use_graphrag=bool(cfg.rag.use_graphrag),
        retriever_model=_llm_model(cfg),
        compressor_model=_llm_model(cfg),
        chat_model=_llm_model(cfg),
    )
    source_map = _source_url_map(paths)
    sources = _enrich_sources(raw.get("sources") or [], source_map)
    source_documents = [
        _enrich_source_document(str(source), source_map)
        for source in raw.get("source_documents") or []
    ]
    context_text = str(raw.get("context_text") or "")
    truncated = len(context_text) > max_chars
    compact_context = context_text[:max_chars]
    return {
        "query": query,
        "retrieval_method": method,
        "context_text": compact_context,
        "context_truncated": truncated,
        "context_chars": len(compact_context),
        "full_context_chars": len(context_text),
        "sources": sources,
        "source_documents": source_documents,
        "rerank_chunks": _compact_rerank_chunks(raw.get("rerank_chunks") or [], source_map),
        "embedding_model": raw.get("embedding_model"),
        "embedding_profile": raw.get("embedding_profile"),
        "kb_status": status,
    }


def _source_url_map(paths: RAGPaths) -> dict[str, dict[str, Any]]:
    manifest = _read_manifest(paths.manifest_path)
    papers = manifest.get("papers", {})
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(papers, dict):
        return out
    for item in papers.values():
        if not isinstance(item, dict):
            continue
        file_name = item.get("file")
        if isinstance(file_name, str):
            out[file_name] = item
            out[Path(file_name).name] = item
    return out


def _lookup_source(source: str, source_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    return source_map.get(source) or source_map.get(Path(source).name)


def _enrich_sources(
    sources: list[Any], source_map: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        enriched = dict(source)
        item = _lookup_source(str(source.get("source") or ""), source_map)
        if item:
            enriched["url"] = item.get("url")
            enriched["title"] = item.get("title")
        out.append(enriched)
    return out


def _enrich_source_document(source: str, source_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item = _lookup_source(source, source_map) or {}
    return {"source": source, "url": item.get("url"), "title": item.get("title")}


def _compact_rerank_chunks(
    chunks: list[Any], source_map: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for chunk in chunks[:20]:
        if not isinstance(chunk, dict):
            continue
        source = str(chunk.get("source") or "")
        item = _lookup_source(source, source_map) or {}
        text = str(chunk.get("text") or "")
        out.append(
            {
                "rank": chunk.get("rank"),
                "source": source,
                "url": item.get("url"),
                "title": item.get("title"),
                "chunk_id": chunk.get("chunk_id"),
                "score": chunk.get("score"),
                "text": text[:1200],
            }
        )
    return out


def recordable_rag_urls(content: Any) -> list[str]:
    if not isinstance(content, dict):
        return []
    urls: list[str] = []
    for key in ("sources", "source_documents", "rerank_chunks"):
        value = content.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                urls.append(url)
    return list(dict.fromkeys(urls))
