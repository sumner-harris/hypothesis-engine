"""LiteratureReviewAgent for selecting search hits and PDF downloads."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import Config
from ..llm.anthropic_client import AgentCallSpec, CachedBlock, CallContext
from ..llm.routing import route
from .schemas import RECORD_LITERATURE_SELECTION_TOOL


@dataclass(frozen=True)
class LiteratureReviewResult:
    selected_records: list[dict[str, Any]]
    selected_pdf_records: list[dict[str, Any]]
    metadata: dict[str, Any]


async def select_search_records(
    *,
    cfg: Config,
    llm_client: Any | None,
    session_id: str | None,
    task_id: str | None,
    query: str,
    provider: str,
    records: list[dict[str, Any]],
    max_pdfs: int,
    caller: dict[str, Any] | None = None,
) -> LiteratureReviewResult:
    """Choose search records for agent context and PDF-capable records for download."""
    max_context = max(0, int(cfg.rag.literature_review_max_context_results))
    max_pdf_select = max(0, int(max_pdfs))
    max_candidates = max(1, int(cfg.rag.literature_review_max_candidates))
    candidates = _candidate_records(records, limit=max_candidates)
    base_meta: dict[str, Any] = {
        "enabled": bool(cfg.rag.literature_review_enabled),
        "provider": provider,
        "query": query,
        "candidate_records": len(records),
        "reviewed_candidates": len(candidates),
        "max_context_results": max_context,
        "max_pdfs": max_pdf_select,
    }
    if max_context <= 0:
        return LiteratureReviewResult(
            [],
            [],
            {**base_meta, "selected": 0, "selected_pdfs": 0, "skipped": "context_limit_zero"},
        )
    if not candidates:
        return LiteratureReviewResult(
            [],
            [],
            {**base_meta, "selected": 0, "selected_pdfs": 0, "skipped": "no_candidates"},
        )
    if not cfg.rag.literature_review_enabled:
        return _fallback_selection(
            candidates,
            base_meta,
            max_context,
            max_pdf_select,
            reason="disabled",
            cfg=cfg,
        )
    if llm_client is None or not session_id:
        return _fallback_selection(
            candidates,
            base_meta,
            max_context,
            max_pdf_select,
            reason="missing_llm_client_or_session",
            cfg=cfg,
        )

    prompt = _selection_prompt(
        query=query,
        provider=provider,
        candidates=candidates,
        max_context=max_context,
        max_pdfs=max_pdf_select,
        caller=caller or {},
    )
    spec = AgentCallSpec(
        route=route(cfg, "literature_review", "selection"),
        system_blocks=[CachedBlock(_SYSTEM_PROMPT, cache=False)],
        user_blocks=[CachedBlock(prompt, cache=False)],
        tools=[RECORD_LITERATURE_SELECTION_TOOL],
        tool_choice={"type": "tool", "name": "record_literature_selection"},
        max_output_tokens=max(1024, int(cfg.rag.literature_review_max_output_tokens)),
    )
    call_ctx = CallContext(
        session_id=session_id,
        task_id=task_id,
        agent="literature_review",
        action="SelectSearchContext",
        mode=provider or "selection",
    )
    marker_path = _write_active_marker(cfg, session_id, task_id, provider, query, caller or {})
    try:
        timeout = max(1.0, float(cfg.rag.literature_review_timeout_seconds))
        response = await asyncio.wait_for(llm_client.call(spec, call_ctx), timeout=timeout)
    except Exception as exc:  # pragma: no cover - provider exceptions vary.
        return _failed_review_selection(
            {**base_meta, "error": str(exc)[:300]},
            reason="review_failed",
        )
    finally:
        _clear_active_marker(marker_path)

    record = _final_tool_use(response, "record_literature_selection")
    if record is None:
        stop_reason = str(getattr(getattr(response, "raw", None), "stop_reason", "") or "")
        metadata = base_meta if not stop_reason else {**base_meta, "stop_reason": stop_reason}
        return _failed_review_selection(
            metadata,
            reason="missing_selection_tool_call",
        )

    selected_records, selected_notes = _selected_records_from_tool(record, candidates, max_context)
    if not selected_records and (record.get("selected_results") or record.get("selected_pdfs")):
        return _failed_review_selection(
            base_meta,
            reason="invalid_selected_results",
        )

    pdf_records, pdf_notes = _selected_pdf_records_from_tool(
        record,
        candidates,
        selected_records,
        max_pdf_select,
    )
    downloaded_indexes = {
        int(note["index"])
        for note in pdf_notes
        if isinstance(note, dict) and str(note.get("index") or "").isdigit()
    }
    if downloaded_indexes:
        selected_notes = [
            {
                **note,
                "download_pdf": bool(note.get("download_pdf"))
                or int(note.get("index") or -1) in downloaded_indexes,
            }
            for note in selected_notes
        ]
    metadata = {
        **base_meta,
        "reviewed": True,
        "selected": len(selected_records),
        "selected_results": selected_notes,
        "selected_pdfs": len(pdf_records),
        "selected_pdf_records": pdf_notes,
        "rejected_count": record.get("rejected_count"),
        "notes": str(record.get("notes") or "")[:1000],
    }
    return LiteratureReviewResult(selected_records, pdf_records, metadata)


async def select_pdf_records(**kwargs: Any) -> LiteratureReviewResult:
    """Backward-compatible wrapper; selected_records are still context records."""
    return await select_search_records(**kwargs)


_SYSTEM_PROMPT = """You are LiteratureReviewAgent. Your job is to decide which search results are relevant enough to show to the calling scientific agent, and which of those should be downloaded as full PDFs for the session RAG library.

Use only the titles, abstracts/snippets, source metadata, and search query provided. Select records that are likely to provide useful evidence for the current search intent. Prefer papers with specific mechanisms, methods, datasets, experiments, quantitative results, benchmarks, or negative evidence over generic reviews or weakly related hits.

Allow exploration: when there is room under the context limit, you may include adjacent, cross-field, or negative-evidence records if they could broaden hypothesis diversity or prevent a false assumption. Do not select off-topic records just to fill the limit. Do not invent indexes or URLs.

For selected records with a direct pdf_url, set download_pdf=true by default, including exploratory records, when the paper provides mechanisms, methods, datasets, experiments, quantitative results, benchmarks, or negative evidence. Use download_pdf=false only when the record has no direct pdf_url, is a duplicate of another selected paper, is weak context-only background, or the PDF cap is already consumed. If you use download_pdf=false for a selected record with a direct pdf_url, state the exception in the reason or notes.
"""


def _selection_prompt(
    *,
    query: str,
    provider: str,
    candidates: list[dict[str, Any]],
    max_context: int,
    max_pdfs: int,
    caller: dict[str, Any],
) -> str:
    compact = [_candidate_for_prompt(item) for item in candidates]
    caller_bits = {k: v for k, v in caller.items() if v not in (None, "")}
    return (
        f"# Search query\n{query or '(not provided)'}\n\n"
        f"# Search provider\n{provider or '(unknown)'}\n\n"
        "# Calling workflow\n"
        f"{json.dumps(caller_bits, ensure_ascii=True, sort_keys=True)}\n\n"
        "# Selection limits\n"
        f"Select 0 to {max_context} search results for the calling agent to see. "
        f"At most {max_pdfs} selected records may set download_pdf=true. "
        "Selected direct-PDF records should normally set download_pdf=true unless they match an explicit exception. "
        "If no record is relevant, select none.\n\n"
        "# Candidate records\n"
        f"{json.dumps(compact, ensure_ascii=True, indent=2)}\n\n"
        "Call record_literature_selection exactly once."
    )


def _candidate_for_prompt(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": record["_review_index"],
        "source": record.get("source") or record.get("metadata_provider") or "",
        "title": record.get("title") or "",
        "abstract": _record_abstract(record),
        "year": record.get("year") or record.get("published") or record.get("published_at") or "",
        "doi": record.get("doi") or "",
        "categories": record.get("categories") or [],
        "pdf_url": record.get("pdf_url") or "",
        "url": record.get("abs_url") or record.get("url") or record.get("pubmed_url") or "",
    }


def _candidate_records(records: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        key = _record_key(record)
        if key in seen:
            continue
        seen.add(key)
        candidate = dict(record)
        candidate["_review_index"] = idx
        pdf_url = _record_pdf_url(record)
        if pdf_url:
            candidate["pdf_url"] = pdf_url
        out.append(candidate)
        if len(out) >= limit:
            break
    return out


def _write_active_marker(
    cfg: Config,
    session_id: str,
    task_id: str | None,
    provider: str,
    query: str,
    caller: dict[str, Any],
) -> Path | None:
    try:
        root = cfg.session_artifact_dir(session_id) / "literature_review" / "active_calls"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{uuid4().hex}.json"
        trigger_agent = str(caller.get("agent") or "").strip() if isinstance(caller, dict) else ""
        trigger_action = str(caller.get("action") or "").strip() if isinstance(caller, dict) else ""
        payload = {
            "session_id": session_id,
            "task_id": task_id,
            "provider": provider,
            "query": query,
            "trigger_agent": trigger_agent,
            "trigger_action": trigger_action,
            "started_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except OSError:
        return None


def _clear_active_marker(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _record_key(record: dict[str, Any]) -> str:
    for field in ("pdf_url", "url", "abs_url", "pubmed_url", "doi", "pmid", "id", "arxiv_id", "source_id"):
        value = str(record.get(field) or "").strip().casefold()
        if value:
            return f"{field}:{value}"
    title = " ".join(str(record.get("title") or "").casefold().split())
    return f"title:{title}" if title else f"record:{id(record)}"


def _record_pdf_url(record: dict[str, Any]) -> str:
    pdf_url = str(record.get("pdf_url") or "").strip()
    if pdf_url.startswith(("http://", "https://")):
        return pdf_url
    url = str(record.get("url") or "").strip()
    if url.startswith(("http://", "https://")) and _looks_like_pdf_url(url):
        return url
    return ""


def _looks_like_pdf_url(url: str) -> bool:
    lowered = url.lower().split("?", 1)[0]
    return lowered.endswith(".pdf") or "/pdf/" in lowered


def _record_abstract(record: dict[str, Any]) -> str:
    text = record.get("summary") or record.get("abstract") or record.get("snippet") or ""
    return " ".join(str(text).split())


def _selected_records_from_tool(
    record: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_context: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_index = {int(item["_review_index"]): item for item in candidates}
    selected: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw_item in record.get("selected_results") or []:
        if not isinstance(raw_item, dict):
            continue
        try:
            index = int(raw_item.get("index"))
        except (TypeError, ValueError):
            continue
        if index not in by_index or index in seen:
            continue
        seen.add(index)
        selected.append(_public_record(by_index[index]))
        notes.append(
            {
                "index": index,
                "title": str(raw_item.get("title") or by_index[index].get("title") or ""),
                "priority": str(raw_item.get("priority") or "core"),
                "reason": str(raw_item.get("reason") or "")[:500],
                "download_pdf": bool(raw_item.get("download_pdf")),
            }
        )
        if len(selected) >= max_context:
            break
    if selected:
        return selected, notes

    # Compatibility path for older fake/tool outputs that selected only PDFs.
    return _selected_records_from_pdf_urls(record, candidates, max_context)


def _selected_records_from_pdf_urls(
    record: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_context: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_url = {str(item.get("pdf_url") or ""): item for item in candidates if item.get("pdf_url")}
    selected: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_item in record.get("selected_pdfs") or []:
        if not isinstance(raw_item, dict):
            continue
        url = str(raw_item.get("pdf_url") or "").strip()
        if url not in by_url or url in seen:
            continue
        seen.add(url)
        candidate = by_url[url]
        selected.append(_public_record(candidate))
        notes.append(
            {
                "index": int(candidate["_review_index"]),
                "title": str(raw_item.get("title") or candidate.get("title") or ""),
                "priority": str(raw_item.get("priority") or "core"),
                "reason": str(raw_item.get("reason") or "")[:500],
                "download_pdf": True,
            }
        )
        if len(selected) >= max_context:
            break
    return selected, notes


def _selected_pdf_records_from_tool(
    record: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    max_pdfs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_pdfs <= 0:
        return [], []
    by_index = {int(item["_review_index"]): item for item in candidates}
    selected_keys = {_record_key(item) for item in selected_records}
    pdf_records: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    selected_items: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def add_pdf(raw_item: dict[str, Any], candidate: dict[str, Any], *, inferred: bool) -> bool:
        pdf_url = _record_pdf_url(candidate)
        if not pdf_url or pdf_url in seen_urls:
            return False
        seen_urls.add(pdf_url)
        pdf_records.append(_public_record(candidate))
        notes.append(
            {
                "index": int(candidate["_review_index"]),
                "pdf_url": pdf_url,
                "title": str(raw_item.get("title") or candidate.get("title") or ""),
                "priority": str(raw_item.get("priority") or "core"),
                "reason": str(raw_item.get("reason") or "")[:500],
                "download_inferred": inferred,
            }
        )
        return len(pdf_records) >= max_pdfs

    for raw_item in record.get("selected_results") or []:
        if not isinstance(raw_item, dict):
            continue
        try:
            index = int(raw_item.get("index"))
        except (TypeError, ValueError):
            continue
        candidate = by_index.get(index)
        if not candidate or _record_key(candidate) not in selected_keys:
            continue
        selected_items.append((raw_item, candidate))
        if raw_item.get("download_pdf") and add_pdf(raw_item, candidate, inferred=False):
            return pdf_records, notes

    for raw_item, candidate in selected_items:
        if _selected_item_defaults_to_pdf(raw_item, candidate) and add_pdf(raw_item, candidate, inferred=True):
            return pdf_records, notes

    # Compatibility path for older fake/tool outputs that selected only PDFs.
    by_url = {str(item.get("pdf_url") or ""): item for item in candidates if item.get("pdf_url")}
    for raw_item in record.get("selected_pdfs") or []:
        if not isinstance(raw_item, dict):
            continue
        url = str(raw_item.get("pdf_url") or "").strip()
        candidate = by_url.get(url)
        if not candidate or url in seen_urls:
            continue
        seen_urls.add(url)
        pdf_records.append(_public_record(candidate))
        notes.append(
            {
                "index": int(candidate["_review_index"]),
                "pdf_url": url,
                "title": str(raw_item.get("title") or candidate.get("title") or ""),
                "priority": str(raw_item.get("priority") or "core"),
                "reason": str(raw_item.get("reason") or "")[:500],
                "download_inferred": False,
            }
        )
        if len(pdf_records) >= max_pdfs:
            break
    return pdf_records, notes


def _selected_item_defaults_to_pdf(raw_item: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if not _record_pdf_url(candidate):
        return False
    priority = str(raw_item.get("priority") or "core").strip().casefold()
    return priority in {"core", "exploratory", "negative_evidence"}


def _fallback_selection(
    candidates: list[dict[str, Any]],
    base_meta: dict[str, Any],
    max_context: int,
    max_pdfs: int,
    *,
    reason: str,
    cfg: Config,
) -> LiteratureReviewResult:
    if not cfg.rag.literature_review_fallback_to_top_results:
        return LiteratureReviewResult(
            [],
            [],
            {**base_meta, "reviewed": False, "selected": 0, "selected_pdfs": 0, "fallback": reason},
        )
    selected = [_public_record(item) for item in candidates[:max_context]]
    selected_keys = {_record_key(item) for item in selected}
    pdf_records: list[dict[str, Any]] = []
    for item in candidates:
        if _record_key(item) not in selected_keys:
            continue
        if item.get("pdf_url"):
            pdf_records.append(_public_record(item))
        if len(pdf_records) >= max_pdfs:
            break
    return LiteratureReviewResult(
        selected,
        pdf_records,
        {
            **base_meta,
            "reviewed": False,
            "selected": len(selected),
            "selected_pdfs": len(pdf_records),
            "fallback": reason,
        },
    )


def _failed_review_selection(
    base_meta: dict[str, Any],
    *,
    reason: str,
) -> LiteratureReviewResult:
    """Fail closed after an attempted live review.

    A configured fallback is useful when the reviewer is disabled or no LLM is
    available, but once a live LiteratureReviewAgent call was attempted, raw
    top hits should not be treated as approved evidence or PDF-download input.
    """
    return LiteratureReviewResult(
        [],
        [],
        {
            **base_meta,
            "reviewed": False,
            "selected": 0,
            "selected_pdfs": 0,
            "fallback": reason,
        },
    )


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in record.items() if not k.startswith("_review_")}
    return out


def _final_tool_use(response: Any, name: str) -> dict[str, Any] | None:
    for block in getattr(getattr(response, "raw", None), "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == name:
            payload = getattr(block, "input", None)
            return payload if isinstance(payload, dict) else None
    return None
