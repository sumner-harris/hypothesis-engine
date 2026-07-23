"""Post-hoc analysis report for a hypothesis-engine session.

The report is intentionally artifact-first: it reads the local SQLite database,
stored search/paper/transcript artifacts, hypothesis FAISS index, and optional
RAG knowledge-base index. It does not call models, tools, or network services.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import re
import sqlite3
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from ..config import Config
from .rag_projection import (
    load_or_build_rag_projection,
    transform_rag_vectors,
)

try:  # pragma: no cover - exercised in environments with FAISS installed.
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


try:  # pragma: no cover - optional nonlinear reduction dependency.
    import umap
except ImportError:  # pragma: no cover
    umap = None


_URL_RE = re.compile(r"https?://[^\s)>\]}\"']+")
_ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s)>\]}\"']+", re.IGNORECASE)
_CHEMRXIV_ID_RE = re.compile(r"\b[0-9a-f]{24}\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.-]*")
_ANALYSIS_STOPWORDS = {
    "about", "above", "after", "also", "among", "and", "are", "based", "been",
    "between", "both", "can", "could", "during", "each", "effect", "effects",
    "fig", "fig.", "figure", "figures", "for", "from", "have", "into", "its",
    "may", "more", "most", "not", "one", "only", "other", "our", "paper",
    "papers", "pdf", "results", "show", "shown", "study", "such", "than",
    "that", "the", "their", "these", "this", "through", "using", "via",
    "was", "were", "where", "with", "within", "would", "which", "while",
    "all", "case", "data", "eq", "eq.", "has", "value", "values",
}


@dataclass(frozen=True)
class _Paths:
    output: Path
    figures: Path
    tables: Path


def analyze_session(
    cfg: Config,
    session_id: str,
    *,
    output_dir: Path | None = None,
    snapshot_every: int = 25,
    max_kb_points: int = 5000,
    n_clusters: int = 8,
) -> Path:
    """Analyze a completed or in-progress session and write a report.

    Returns the path to ``report.md``. The output directory defaults to
    ``data/artifacts/<session_id>/analysis``.
    """

    out = output_dir or (cfg.session_artifact_dir(session_id) / "analysis")
    paths = _Paths(output=out, figures=out / "figures", tables=out / "tables")
    paths.figures.mkdir(parents=True, exist_ok=True)
    paths.tables.mkdir(parents=True, exist_ok=True)

    conn = _connect(cfg.db_path)
    try:
        session = _one(conn, "SELECT * FROM sessions WHERE id = ?", (session_id,))
        if session is None:
            raise ValueError(f"No session found in database: {session_id}")

        hypotheses = _rows(
            conn,
            "SELECT * FROM hypotheses WHERE session_id = ? ORDER BY created_at, id",
            (session_id,),
        )
        matches = _rows(
            conn,
            "SELECT * FROM tournament_matches WHERE session_id = ? ORDER BY created_at, id",
            (session_id,),
        )
        reviews = _rows(
            conn,
            "SELECT * FROM reviews WHERE session_id = ? ORDER BY created_at, id",
            (session_id,),
        )
        transcripts = _rows(
            conn,
            "SELECT * FROM transcripts WHERE session_id = ? ORDER BY started_at, id",
            (session_id,),
        )
        feedback = _rows(
            conn,
            "SELECT * FROM system_feedback WHERE session_id = ? ORDER BY created_at, id",
            (session_id,),
        )
        embeddings_meta = _rows(
            conn,
            "SELECT * FROM embeddings_meta WHERE session_id = ? ORDER BY created_at, id",
            (session_id,),
        )
    finally:
        conn.close()

    artifact_dir = cfg.session_artifact_dir(session_id)
    sources = _collect_sources(artifact_dir, cfg.session_rag_dir(session_id))
    source_use = _analyze_source_use(hypotheses, sources)
    tool_usage = _scan_transcript_artifacts(artifact_dir)
    transcript_summary = _summarize_transcripts(transcripts)
    review_summary = _summarize_reviews(reviews)
    lineage_summary = _summarize_lineage(hypotheses)
    elo_logistic_scale = _session_elo_logistic_scale(session)
    debate_summary = _summarize_debates(
        matches,
        hypotheses=hypotheses,
        reviews=reviews,
        elo_logistic_scale=elo_logistic_scale,
    )
    elo_summary = _analyze_elo(
        matches,
        hypotheses,
        snapshot_every=snapshot_every,
        elo_logistic_scale=elo_logistic_scale,
    )
    research_goal = str(session.get("research_goal") or "")

    hyp_vectors = _load_hypothesis_vectors(cfg, session_id)
    hyp_vector_analysis = _analyze_hypothesis_vectors(
        hypotheses, hyp_vectors, paths, n_clusters=n_clusters
    )
    kb_projection = load_or_build_rag_projection(cfg, session_id)
    kb = _load_kb_vectors(cfg, session_id, max_points=max_kb_points)
    kb_analysis = _analyze_kb_vectors(
        cfg,
        kb,
        paths,
        n_clusters=n_clusters,
        research_goal=research_goal,
        projection=kb_projection,
    )
    joint_pca = _analyze_joint_pca(
        hyp_vectors,
        hypotheses,
        kb,
        kb_analysis,
        paths,
        projection=kb_projection,
    )

    _write_tables(
        paths,
        hypotheses=hypotheses,
        reviews=reviews,
        matches=matches,
        source_use=source_use,
        sources=sources,
        transcript_summary=transcript_summary,
        hyp_vector_analysis=hyp_vector_analysis,
        debate_summary=debate_summary,
        elo_summary=elo_summary,
        kb_analysis=kb_analysis,
        joint_pca=joint_pca,
    )
    _write_figures(
        paths,
        hypotheses=hypotheses,
        matches=matches,
        debate_summary=debate_summary,
        elo_summary=elo_summary,
        source_use=source_use,
        transcript_summary=transcript_summary,
        hyp_vector_analysis=hyp_vector_analysis,
        kb_analysis=kb_analysis,
        joint_pca=joint_pca,
    )

    metrics = {
        "session_id": session_id,
        "session_status": session.get("status"),
        "research_goal": session.get("research_goal"),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "hypotheses": {
            "count": len(hypotheses),
            "embedding_rows": len(embeddings_meta),
            "faiss_vectors": hyp_vectors.get("count", 0),
            "missing_faiss_vectors": hyp_vectors.get("missing_ids", []),
            **lineage_summary,
        },
        "hypothesis_vectors": hyp_vector_analysis.get("metrics", {}),
        "matches": {
            "count": len(matches),
            **debate_summary.get("metrics", {}),
            **elo_summary.get("metrics", {}),
        },
        "reviews": review_summary,
        "sources": source_use["metrics"],
        "rag_kb": kb_analysis["metrics"],
        "joint_pca": joint_pca.get("metrics", {}),
        "tool_usage": tool_usage,
        "transcripts": transcript_summary,
    }
    (paths.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    report = _render_report(
        session=session,
        hypotheses=hypotheses,
        matches=matches,
        feedback=feedback,
        transcript_summary=transcript_summary,
        review_summary=review_summary,
        lineage_summary=lineage_summary,
        debate_summary=debate_summary,
        elo_summary=elo_summary,
        source_use=source_use,
        sources=sources,
        tool_usage=tool_usage,
        hyp_vector_analysis=hyp_vector_analysis,
        kb_analysis=kb_analysis,
        joint_pca=joint_pca,
        metrics=metrics,
        paths=paths,
    )
    report_path = paths.output / "report.md"
    report_path.write_text(report, encoding="utf-8")
    _write_html_report(report, paths, session_id=session_id)
    _write_bundle(paths, session_id=session_id)
    return report_path


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, args).fetchall()]


def _one(conn: sqlite3.Connection, sql: str, args: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    row = conn.execute(sql, args).fetchone()
    return dict(row) if row else None


def _collect_sources(artifact_dir: Path, rag_dir: Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    search_counts: Counter[str] = Counter()
    search_queries: Counter[str] = Counter()

    for file in sorted((artifact_dir / "searches").glob("*/*.json")):
        tool = file.parent.name
        payload = _read_json(file)
        if not isinstance(payload, dict):
            continue
        search_counts[tool] += 1
        query = str(payload.get("query") or "").strip()
        if query:
            search_queries[f"{tool}: {query}"] += 1
        for result in _extract_results(payload):
            key = _source_key(result)
            if not key:
                continue
            record = records.setdefault(key, _source_record(result, "search", tool))
            record["seen_in_search"] = True
            record["search_tools"].add(tool)
            record["search_queries"].add(query)

    paper_artifact_count = 0
    for file in sorted((artifact_dir / "papers").glob("*.json")):
        payload = _read_json(file)
        if not isinstance(payload, dict):
            continue
        paper_artifact_count += 1
        key = _source_key(payload)
        if not key:
            continue
        record = records.setdefault(key, _source_record(payload, "paper_artifact", "web_fetch"))
        record["paper_artifact"] = True
        record["paper_artifact_path"] = str(file)
        record["text_chars"] = max(record.get("text_chars", 0), len(str(payload.get("text") or "")))

    manifest = _read_json(rag_dir / "manifest.json")
    rag_papers = {}
    if isinstance(manifest, dict) and isinstance(manifest.get("papers"), dict):
        rag_papers = manifest["papers"]
    for digest, item in rag_papers.items():
        if not isinstance(item, dict):
            continue
        key = _source_key(item) or str(digest)
        record = records.setdefault(key, _source_record(item, "rag_manifest", "rag"))
        record["rag_manifest"] = True
        record["rag_digest"] = digest
        record["rag_indexed"] = bool(item.get("indexed"))
        record["rag_reserved"] = bool(item.get("reserved"))
        record["bytes"] = item.get("bytes") or record.get("bytes")
        record["file"] = item.get("file") or record.get("file")
        record["title_key"] = item.get("title_key") or record.get("title_key")

    for record in records.values():
        record["search_tools"] = sorted(record.get("search_tools", set()))
        record["search_queries"] = sorted(q for q in record.get("search_queries", set()) if q)
        record["aliases"] = sorted(_source_aliases(record))

    return {
        "records": records,
        "search_counts": dict(search_counts),
        "search_queries": dict(search_queries),
        "paper_artifact_count": paper_artifact_count,
        "rag_manifest_count": len(rag_papers),
        "rag_indexed_count": sum(
            1 for item in rag_papers.values()
            if isinstance(item, dict) and bool(item.get("indexed"))
        ),
    }


def _extract_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if isinstance(results, list):
        return [r for r in results if isinstance(r, dict)]
    content = payload.get("content")
    if isinstance(content, dict) and isinstance(content.get("results"), list):
        return [r for r in content["results"] if isinstance(r, dict)]
    return []


def _source_key(data: dict[str, Any]) -> str:
    for field in ("pdf_url", "url", "requested_url", "abs_url", "doi", "arxiv_id", "chemrxiv_id", "source_id"):
        value = data.get(field)
        if value:
            return _normalize_source_id(str(value))
    title = str(data.get("title") or data.get("title_key") or "").strip()
    if title:
        return f"title:{_title_key(title)}"
    file = str(data.get("file") or data.get("source") or "").strip()
    if file:
        return f"file:{file.lower()}"
    return ""


def _source_record(data: dict[str, Any], origin: str, tool: str) -> dict[str, Any]:
    return {
        "key": _source_key(data),
        "origin": origin,
        "title": str(data.get("title") or data.get("title_key") or "").strip(),
        "url": str(data.get("url") or data.get("pdf_url") or data.get("requested_url") or "").strip(),
        "pdf_url": str(data.get("pdf_url") or data.get("url") or "").strip(),
        "abs_url": str(data.get("abs_url") or "").strip(),
        "doi": str(data.get("doi") or "").strip(),
        "arxiv_id": str(data.get("arxiv_id") or "").strip(),
        "chemrxiv_id": str(data.get("chemrxiv_id") or "").strip(),
        "source_id": str(data.get("source_id") or "").strip(),
        "source_provider": str(data.get("source") or "").strip(),
        "file": str(data.get("file") or data.get("source") or "").strip(),
        "year": data.get("year"),
        "categories": data.get("categories") if isinstance(data.get("categories"), list) else [],
        "seen_in_search": False,
        "paper_artifact": False,
        "rag_manifest": False,
        "rag_indexed": False,
        "search_tools": {tool} if tool else set(),
        "search_queries": set(),
        "bytes": data.get("bytes"),
        "text_chars": len(str(data.get("text") or "")),
    }


def _source_aliases(record: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for field in ("url", "pdf_url", "abs_url", "doi", "arxiv_id", "chemrxiv_id", "source_id"):
        value = str(record.get(field) or "").strip()
        if value:
            aliases.add(value.lower())
            if field == "arxiv_id":
                aliases.add(re.sub(r"v\d+$", "", value.lower()))
    title = str(record.get("title") or "").strip()
    if len(title) >= 20:
        aliases.add(f"title:{_title_key(title)}")
    title_key = str(record.get("title_key") or "").strip()
    if title_key:
        aliases.add(f"title:{_title_key(title_key)}")
    file = str(record.get("file") or "").strip()
    if file:
        aliases.add(file.lower())
    return {a for a in aliases if a}


def _analyze_source_use(
    hypotheses: list[dict[str, Any]], sources: dict[str, Any]
) -> dict[str, Any]:
    source_records: dict[str, dict[str, Any]] = sources["records"]
    alias_to_key: dict[str, str] = {}
    for key, record in source_records.items():
        for alias in record.get("aliases", []):
            alias_to_key.setdefault(str(alias).lower(), key)

    per_hyp: list[dict[str, Any]] = []
    source_hits: Counter[str] = Counter()
    explicit_tokens: Counter[str] = Counter()

    for hyp in hypotheses:
        text = f"{hyp.get('title') or ''}\n{hyp.get('summary') or ''}\n{hyp.get('full_text') or ''}"
        lower = text.lower()
        matched_keys: set[str] = set()
        tokens = set()
        tokens.update(t.lower().rstrip(".,;") for t in _URL_RE.findall(text))
        tokens.update(t.lower().rstrip(".,;") for t in _DOI_RE.findall(text))
        tokens.update(t.lower() for t in _ARXIV_ID_RE.findall(text))
        tokens.update(t.lower() for t in _CHEMRXIV_ID_RE.findall(text))

        for token in tokens:
            explicit_tokens[token] += 1
            normalized = _normalize_source_id(token)
            if normalized in alias_to_key:
                matched_keys.add(alias_to_key[normalized])
            else:
                bare_arxiv = re.sub(r"v\d+$", "", token)
                if bare_arxiv in alias_to_key:
                    matched_keys.add(alias_to_key[bare_arxiv])

        # Exact normalized title matches catch sources cited by title rather than URL.
        for alias, key in alias_to_key.items():
            if alias.startswith("title:"):
                needle = alias.removeprefix("title:")
                if needle and needle in _title_key(lower):
                    matched_keys.add(key)

        for key in matched_keys:
            source_hits[key] += 1
        per_hyp.append(
            {
                "hypothesis_id": hyp.get("id"),
                "title": hyp.get("title"),
                "explicit_source_tokens": len(tokens),
                "matched_known_sources": len(matched_keys),
                "matched_source_keys": sorted(matched_keys),
            }
        )

    used_known = set(source_hits)
    metrics = {
        "known_sources_total": len(source_records),
        "known_sources_in_searches": sum(1 for r in source_records.values() if r.get("seen_in_search")),
        "known_sources_in_rag_manifest": sum(1 for r in source_records.values() if r.get("rag_manifest")),
        "known_sources_indexed_in_rag": sum(1 for r in source_records.values() if r.get("rag_indexed")),
        "paper_text_artifacts": sources["paper_artifact_count"],
        "unique_explicit_source_tokens_in_hypotheses": len(explicit_tokens),
        "known_sources_explicitly_matched_in_hypotheses": len(used_known),
        "hypotheses_with_explicit_source_tokens": sum(1 for row in per_hyp if row["explicit_source_tokens"] > 0),
        "hypotheses_with_known_source_matches": sum(1 for row in per_hyp if row["matched_known_sources"] > 0),
        "source_reuse_mean_hypotheses_per_used_source": _safe_mean(list(source_hits.values())),
    }
    top_sources = []
    for key, count in source_hits.most_common(20):
        record = source_records.get(key, {})
        top_sources.append(
            {
                "source_key": key,
                "hypothesis_count": count,
                "title": record.get("title") or record.get("file") or key,
                "url": record.get("url") or record.get("pdf_url") or record.get("abs_url") or "",
                "rag_indexed": bool(record.get("rag_indexed")),
                "search_tools": ",".join(record.get("search_tools", [])),
            }
        )
    return {
        "metrics": metrics,
        "per_hypothesis": per_hyp,
        "source_hits": dict(source_hits),
        "top_sources": top_sources,
        "explicit_tokens": dict(explicit_tokens),
    }


def _scan_transcript_artifacts(artifact_dir: Path) -> dict[str, Any]:
    names = [
        "rag_retrieve_context",
        "retrieve_context",
        "retrieve_and_answer",
        "arxiv_search",
        "biorxiv_search",
        "chemrxiv_search",
        "web_fetch",
        "europe_pmc_search",
        "pubmed_search",
    ]
    by_agent: dict[str, Counter[str]] = defaultdict(Counter)
    total = Counter()
    transcripts_scanned = 0
    for file in sorted((artifact_dir / "transcripts").glob("*/*.json")):
        agent = file.parent.name
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        transcripts_scanned += 1
        for name in names:
            count = text.count(name)
            if count:
                by_agent[agent][name] += count
                total[name] += count
    return {
        "transcripts_scanned": transcripts_scanned,
        "tool_name_mentions": dict(total),
        "tool_name_mentions_by_agent": {
            agent: dict(counter) for agent, counter in sorted(by_agent.items())
        },
    }


def _summarize_transcripts(transcripts: list[dict[str, Any]]) -> dict[str, Any]:
    by_agent: dict[str, dict[str, Any]] = {}
    for row in transcripts:
        agent = str(row.get("agent") or "unknown")
        summary = by_agent.setdefault(
            agent,
            {
                "count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "first_started_at": None,
                "last_finished_at": None,
            },
        )
        summary["count"] += 1
        summary["input_tokens"] += int(row.get("input_tokens") or 0)
        summary["output_tokens"] += int(row.get("output_tokens") or 0)
        summary["cost_usd"] += float(row.get("cost_usd") or 0.0)
        summary["first_started_at"] = _min_dt_str(summary["first_started_at"], row.get("started_at"))
        summary["last_finished_at"] = _max_dt_str(summary["last_finished_at"], row.get("finished_at"))
    for summary in by_agent.values():
        count = summary["count"] or 1
        summary["tokens_per_call_mean"] = (
            summary["input_tokens"] + summary["output_tokens"]
        ) / count
        summary["cost_usd"] = round(summary["cost_usd"], 6)
    total = {
        "count": sum(s["count"] for s in by_agent.values()),
        "input_tokens": sum(s["input_tokens"] for s in by_agent.values()),
        "output_tokens": sum(s["output_tokens"] for s in by_agent.values()),
        "cost_usd": round(sum(s["cost_usd"] for s in by_agent.values()), 6),
    }
    return {"by_agent": by_agent, "total": total}


def _summarize_reviews(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["novelty", "correctness", "testability", "feasibility"]
    out: dict[str, Any] = {"count": len(reviews)}
    for field in fields:
        values = [_to_float(r.get(field)) for r in reviews if _to_float(r.get(field)) is not None]
        out[field] = _summary_stats(values)
    verdicts = Counter(str(r.get("verdict") or "unknown") for r in reviews)
    kinds = Counter(str(r.get("kind") or "unknown") for r in reviews)
    out["verdict_counts"] = dict(verdicts)
    out["kind_counts"] = dict(kinds)
    return out


def _summarize_lineage(hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    created_by = Counter(str(h.get("created_by") or "unknown") for h in hypotheses)
    strategies = Counter(str(h.get("strategy") or "unknown") for h in hypotheses)
    states = Counter(str(h.get("state") or "unknown") for h in hypotheses)
    parent_counts = []
    roots = 0
    evolved = 0
    for hyp in hypotheses:
        parents = _parse_parent_ids(hyp.get("parent_ids"))
        parent_counts.append(len(parents))
        if parents:
            evolved += 1
        else:
            roots += 1
    return {
        "created_by_counts": dict(created_by),
        "strategy_counts": dict(strategies),
        "state_counts": dict(states),
        "root_hypotheses": roots,
        "hypotheses_with_parents": evolved,
        "mean_parent_count": _safe_mean(parent_counts),
    }


def _summarize_debates(
    matches: list[dict[str, Any]],
    *,
    hypotheses: list[dict[str, Any]] | None = None,
    reviews: list[dict[str, Any]] | None = None,
    elo_logistic_scale: float = 400.0,
) -> dict[str, Any]:
    mode_counts = Counter(str(m.get("mode") or "unknown") for m in matches)
    upset_count = 0
    expected_winner_prob = []
    winner_expected_probs = []
    elo_deltas = []
    elo_gaps = []
    rationale_lengths = []
    invalid = 0

    hyp_by_id = {str(h.get("id")): h for h in (hypotheses or []) if h.get("id")}
    review_by_hyp = _best_review_body_by_hypothesis(reviews or [])
    bias_rows: list[dict[str, Any]] = []
    side_a_scores: list[float] = []
    prompt1_scores: list[float] = []
    prompt1_expected_probs: list[float] = []
    side_a_expected_residuals: list[float] = []
    prompt1_expected_residuals: list[float] = []
    longer_input_scores: list[float] = []
    more_discussed_scores: list[float] = []
    input_delta_values: list[float] = []
    attention_delta_values: list[float] = []
    prompt1_longer_flags: list[float] = []
    prompt1_more_discussed_flags: list[float] = []

    for match_index, match in enumerate(matches, 1):
        a0 = _to_float(match.get("elo_a_before"))
        b0 = _to_float(match.get("elo_b_before"))
        a1 = _to_float(match.get("elo_a_after"))
        b1 = _to_float(match.get("elo_b_after"))
        winner = str(match.get("winner") or "").lower()
        if winner not in {"a", "b", "draw"}:
            invalid += 1
        if a0 is None or b0 is None:
            continue
        scale = elo_logistic_scale if elo_logistic_scale > 0 else 400.0
        p_a = 1.0 / (1.0 + 10.0 ** ((b0 - a0) / scale))
        p_b = 1.0 - p_a
        if winner == "a":
            winner_expected_probs.append(p_a)
            if a0 < b0:
                upset_count += 1
        elif winner == "b":
            winner_expected_probs.append(1.0 - p_a)
            if b0 < a0:
                upset_count += 1
        expected_winner_prob.append(max(p_a, 1.0 - p_a))
        elo_gaps.append(abs(a0 - b0))
        if a1 is not None:
            elo_deltas.append(abs(a1 - a0))
        elif b1 is not None:
            elo_deltas.append(abs(b1 - b0))
        rationale = str(match.get("rationale") or "")
        rationale_lengths.append(len(rationale))

        prompt = _prompt_position_info(match)
        prompt1_id = prompt.get("prompt1_id") or ""
        prompt2_id = prompt.get("prompt2_id") or ""
        winner_prompt_position = prompt.get("winner_prompt_position")
        prompt1_expected = p_a if prompt.get("prompt1_side") == "a" else p_b
        side_a_score = 1.0 if winner == "a" else 0.0 if winner == "b" else 0.5 if winner == "draw" else None
        prompt1_score = (
            1.0 if winner_prompt_position == 1 else 0.0 if winner_prompt_position == 2 else 0.5 if winner == "draw" else None
        )

        stored_prompt1_chars = _to_float(match.get("prompt1_chars"))
        stored_prompt2_chars = _to_float(match.get("prompt2_chars"))
        prompt1_text_chars = (
            int(stored_prompt1_chars)
            if stored_prompt1_chars is not None
            else _hypothesis_prompt_chars(hyp_by_id.get(prompt1_id), review_by_hyp.get(prompt1_id))
        )
        prompt2_text_chars = (
            int(stored_prompt2_chars)
            if stored_prompt2_chars is not None
            else _hypothesis_prompt_chars(hyp_by_id.get(prompt2_id), review_by_hyp.get(prompt2_id))
        )
        input_delta = prompt1_text_chars - prompt2_text_chars
        longer_input_position = 1 if input_delta > 0 else 2 if input_delta < 0 else None
        attention = _rationale_position_attention(rationale)
        attention_delta = attention["position1_chars"] - attention["position2_chars"]
        more_discussed_position = 1 if attention_delta > 0 else 2 if attention_delta < 0 else None

        if side_a_score is not None:
            side_a_scores.append(side_a_score)
            side_a_expected_residuals.append(side_a_score - p_a)
        if prompt1_score is not None:
            prompt1_scores.append(prompt1_score)
            prompt1_expected_probs.append(prompt1_expected)
            prompt1_expected_residuals.append(prompt1_score - prompt1_expected)
            input_delta_values.append(float(input_delta))
            attention_delta_values.append(float(attention_delta))
            if longer_input_position in {1, 2}:
                longer_input_scores.append(1.0 if winner_prompt_position == longer_input_position else 0.0)
                prompt1_longer_flags.append(1.0 if longer_input_position == 1 else 0.0)
            if more_discussed_position in {1, 2}:
                more_discussed_scores.append(1.0 if winner_prompt_position == more_discussed_position else 0.0)
                prompt1_more_discussed_flags.append(1.0 if more_discussed_position == 1 else 0.0)

        bias_rows.append(
            {
                "match_index": match_index,
                "match_id": match.get("id"),
                "mode": match.get("mode"),
                "hyp_a": match.get("hyp_a"),
                "hyp_b": match.get("hyp_b"),
                "winner_side": winner if winner in {"a", "b", "draw"} else "",
                "prompt1_id": prompt1_id,
                "prompt2_id": prompt2_id,
                "prompt1_side": prompt.get("prompt1_side"),
                "prompt_order_key": match.get("prompt_order_key"),
                "winner_prompt_position": winner_prompt_position,
                "side_a_expected_probability": p_a,
                "prompt1_expected_probability": prompt1_expected,
                "side_a_score": side_a_score,
                "prompt1_score": prompt1_score,
                "prompt1_input_chars": prompt1_text_chars,
                "prompt2_input_chars": prompt2_text_chars,
                "prompt1_minus_prompt2_input_chars": input_delta,
                "longer_input_position": longer_input_position,
                "longer_input_won": (
                    winner_prompt_position == longer_input_position if longer_input_position in {1, 2} else None
                ),
                "rationale_prompt1_chars": attention["position1_chars"],
                "rationale_prompt2_chars": attention["position2_chars"],
                "rationale_prompt1_minus_prompt2_chars": attention_delta,
                "rationale_prompt1_mentions": attention["position1_mentions"],
                "rationale_prompt2_mentions": attention["position2_mentions"],
                "more_discussed_position": more_discussed_position,
                "more_discussed_won": (
                    winner_prompt_position == more_discussed_position if more_discussed_position in {1, 2} else None
                ),
                "rationale_chars": len(rationale),
            }
        )

    n_decided = len(winner_expected_probs)
    prompt1_win_count = sum(1 for score in prompt1_scores if score == 1.0)
    side_a_win_count = sum(1 for score in side_a_scores if score == 1.0)
    longer_input_win_count = sum(1 for score in longer_input_scores if score == 1.0)
    more_discussed_win_count = sum(1 for score in more_discussed_scores if score == 1.0)
    metrics = {
        "mode_counts": dict(mode_counts),
        "invalid_winner_count": invalid,
        "upset_count": upset_count,
        "upset_rate": upset_count / n_decided if n_decided else None,
        "mean_pre_match_elo_gap": _safe_mean(elo_gaps),
        "median_pre_match_elo_gap": _safe_median(elo_gaps),
        "elo_logistic_scale": elo_logistic_scale,
        "mean_winner_expected_probability": _safe_mean(winner_expected_probs),
        "mean_expected_favorite_probability": _safe_mean(expected_winner_prob),
        "mean_abs_elo_update": _safe_mean(elo_deltas),
        "median_rationale_chars": _safe_median(rationale_lengths),
        "position_bias_decided_matches": len(prompt1_scores),
        "storage_side_a_win_rate": _safe_mean(side_a_scores),
        "storage_side_a_win_count": side_a_win_count,
        "storage_side_a_expected_residual_mean": _safe_mean(side_a_expected_residuals),
        "storage_side_a_bias_p_value_normal": _binomial_two_sided_normal_p(side_a_win_count, len(side_a_scores)),
        "prompt_position_1_win_rate": _safe_mean(prompt1_scores),
        "prompt_position_1_win_count": prompt1_win_count,
        "prompt_position_2_win_rate": 1.0 - _safe_mean(prompt1_scores) if _safe_mean(prompt1_scores) is not None else None,
        "prompt_position_1_expected_probability_mean": _safe_mean(prompt1_expected_probs),
        "prompt_position_1_expected_residual_mean": _safe_mean(prompt1_expected_residuals),
        "prompt_position_1_bias_p_value_normal": _binomial_two_sided_normal_p(prompt1_win_count, len(prompt1_scores)),
        "longer_input_win_rate": _safe_mean(longer_input_scores),
        "longer_input_win_count": longer_input_win_count,
        "longer_input_match_count": len(longer_input_scores),
        "prompt_position_1_longer_input_rate": _safe_mean(prompt1_longer_flags),
        "input_verbosity_delta_vs_prompt1_win_pearson": _pearson_corr(input_delta_values, prompt1_scores),
        "more_discussed_win_rate": _safe_mean(more_discussed_scores),
        "more_discussed_win_count": more_discussed_win_count,
        "more_discussed_match_count": len(more_discussed_scores),
        "prompt_position_1_more_discussed_rate": _safe_mean(prompt1_more_discussed_flags),
        "rationale_attention_delta_vs_prompt1_win_pearson": _pearson_corr(attention_delta_values, prompt1_scores),
    }
    bias_summary_rows = _debate_bias_summary_rows(metrics)
    return {
        "metrics": metrics,
        "elo_gaps": elo_gaps,
        "winner_expected_probs": winner_expected_probs,
        "elo_deltas": elo_deltas,
        "rationale_lengths": rationale_lengths,
        "bias_rows": bias_rows,
        "bias_summary_rows": bias_summary_rows,
    }


def _best_review_body_by_hypothesis(reviews: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        hid = str(review.get("hypothesis_id") or "")
        if hid:
            grouped[hid].append(review)
    out = {}
    for hid, rows in grouped.items():
        rows.sort(key=lambda row: (str(row.get("kind") or "") != "full", -float(row.get("novelty") or 0.0)))
        out[hid] = str(rows[0].get("body") or "") if rows else ""
    return out


def _hypothesis_prompt_chars(hypothesis: dict[str, Any] | None, review_body: str | None) -> int:
    if not hypothesis:
        return len(str(review_body or ""))
    text = "\n".join(
        str(hypothesis.get(field) or "")
        for field in ("title", "summary", "full_text")
    )
    return len(text) + len(str(review_body or ""))


def _prompt_position_info(match: dict[str, Any]) -> dict[str, Any]:
    hyp_a = str(match.get("hyp_a") or "")
    hyp_b = str(match.get("hyp_b") or "")
    stored_prompt1 = str(match.get("prompt1_hyp_id") or "")
    stored_prompt2 = str(match.get("prompt2_hyp_id") or "")
    stored_prompt1_side = str(match.get("prompt1_side") or "").lower()

    if stored_prompt1 and stored_prompt2:
        prompt1_id, prompt2_id = stored_prompt1, stored_prompt2
        if stored_prompt1_side in {"a", "b"}:
            prompt1_side = stored_prompt1_side
        elif stored_prompt1 == hyp_a:
            prompt1_side = "a"
        elif stored_prompt1 == hyp_b:
            prompt1_side = "b"
        else:
            prompt1_side = ""
    elif hyp_a and hyp_b and hyp_a <= hyp_b:
        prompt1_id, prompt2_id = hyp_a, hyp_b
        prompt1_side = "a"
    else:
        prompt1_id, prompt2_id = hyp_b, hyp_a
        prompt1_side = "b"

    winner = str(match.get("winner") or "").lower()
    stored_winner_position = _to_float(match.get("winner_prompt_position"))
    winner_prompt_position = int(stored_winner_position) if stored_winner_position in {1.0, 2.0} else None
    if winner_prompt_position is None and winner in {"a", "b"} and prompt1_side in {"a", "b"}:
        winner_prompt_position = 1 if winner == prompt1_side else 2
    return {
        "prompt1_id": prompt1_id,
        "prompt2_id": prompt2_id,
        "prompt1_side": prompt1_side,
        "winner_prompt_position": winner_prompt_position,
    }


_POSITION_1_RE = re.compile(r"\b(?:hypothesis|hyp|idea)\s*1\b|\bH1\b", re.IGNORECASE)
_POSITION_2_RE = re.compile(r"\b(?:hypothesis|hyp|idea)\s*2\b|\bH2\b", re.IGNORECASE)


def _rationale_position_attention(text: str) -> dict[str, int]:
    matches: list[tuple[int, int, int]] = []
    for m in _POSITION_1_RE.finditer(text):
        matches.append((m.start(), m.end(), 1))
    for m in _POSITION_2_RE.finditer(text):
        matches.append((m.start(), m.end(), 2))
    matches.sort(key=lambda item: item[0])
    mentions_1 = sum(1 for _start, _end, pos in matches if pos == 1)
    mentions_2 = sum(1 for _start, _end, pos in matches if pos == 2)
    chars_1 = 0
    chars_2 = 0
    for idx, (_start, end, pos) in enumerate(matches):
        next_start = matches[idx + 1][0] if idx + 1 < len(matches) else len(text)
        span = max(0, next_start - end)
        if pos == 1:
            chars_1 += span
        elif pos == 2:
            chars_2 += span
    return {
        "position1_chars": chars_1,
        "position2_chars": chars_2,
        "position1_mentions": mentions_1,
        "position2_mentions": mentions_2,
    }


def _binomial_two_sided_normal_p(successes: int, n: int, p0: float = 0.5) -> float | None:
    if n <= 0:
        return None
    variance = n * p0 * (1.0 - p0)
    if variance <= 0:
        return None
    z = (successes - n * p0) / math.sqrt(variance)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _debate_bias_summary_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("storage_side_a_win_rate", "Storage side A win rate", "A/B side-position check before prompt anchor remapping"),
        ("storage_side_a_expected_residual_mean", "Storage side A expected residual", "Actual A score minus Elo-expected A score"),
        ("prompt_position_1_win_rate", "Prompt position 1 win rate", "Whether the first presented hypothesis tends to win"),
        ("prompt_position_1_expected_probability_mean", "Prompt position 1 expected probability", "Mean Elo-expected score for prompt position 1"),
        ("prompt_position_1_expected_residual_mean", "Prompt position 1 expected residual", "Actual prompt-1 score minus Elo-expected prompt-1 score"),
        ("longer_input_win_rate", "Longer input wins", "Whether the hypothesis plus review with more prompt text wins"),
        ("prompt_position_1_longer_input_rate", "Prompt position 1 longer input rate", "Whether the first-presented hypothesis is also the more verbose prompt input"),
        ("input_verbosity_delta_vs_prompt1_win_pearson", "Input verbosity delta correlation", "Pearson correlation between prompt1-prompt2 input chars and prompt1 win"),
        ("more_discussed_win_rate", "More discussed wins", "Whether the side receiving more H1/H2-labeled rationale text wins"),
        ("prompt_position_1_more_discussed_rate", "Prompt position 1 more discussed rate", "Whether final rationale gives more labeled text to prompt position 1"),
        ("rationale_attention_delta_vs_prompt1_win_pearson", "Rationale attention delta correlation", "Pearson correlation between prompt1-prompt2 rationale attention and prompt1 win"),
    ]
    return [
        {"metric": key, "label": label, "value": metrics.get(key), "interpretation": interpretation}
        for key, label, interpretation in specs
    ]


def _analyze_elo(
    matches: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    *,
    snapshot_every: int,
    elo_logistic_scale: float = 400.0,
) -> dict[str, Any]:
    final_elo = {str(h["id"]): float(h.get("elo") or 0.0) for h in hypotheses if h.get("id")}
    ranked_final = sorted(final_elo.items(), key=lambda item: item[1], reverse=True)
    final_rank = {hid: rank for rank, (hid, _elo) in enumerate(ranked_final, 1)}
    final_top10 = [hid for hid, _elo in ranked_final[:10]]
    final_rating_values = [elo for _hid, elo in ranked_final]
    top5_percentiles = [_percentile_rank(final_rating_values, elo) for _hid, elo in ranked_final[:5]]
    top10_percentiles = [_percentile_rank(final_rating_values, elo) for _hid, elo in ranked_final[:10]]

    ratings: dict[str, float] = {}
    snapshots: list[dict[str, Any]] = []
    snapshot_states: list[dict[str, Any]] = []
    match_counts: Counter[str] = Counter()
    change_events: list[dict[str, Any]] = []
    earliest_seen: dict[str, int] = {}

    for i, match in enumerate(matches, 1):
        a = str(match.get("hyp_a") or "")
        b = str(match.get("hyp_b") or "")
        signed_changes = []
        abs_changes = []

        if a:
            earliest_seen.setdefault(a, i)
            before = _to_float(match.get("elo_a_before"))
            ratings.setdefault(a, before if before is not None else 1200.0)
            after = _to_float(match.get("elo_a_after"))
            if before is not None and after is not None:
                delta = after - before
                signed_changes.append(delta)
                abs_changes.append(abs(delta))
            if after is not None:
                ratings[a] = after
            match_counts[a] += 1
        if b:
            earliest_seen.setdefault(b, i)
            before = _to_float(match.get("elo_b_before"))
            ratings.setdefault(b, before if before is not None else 1200.0)
            after = _to_float(match.get("elo_b_after"))
            if before is not None and after is not None:
                delta = after - before
                signed_changes.append(delta)
                abs_changes.append(abs(delta))
            if after is not None:
                ratings[b] = after
            match_counts[b] += 1

        change_events.append(
            {
                "match_index": i,
                "signed_changes": signed_changes,
                "abs_changes": abs_changes,
            }
        )

        if i == 1 or i % snapshot_every == 0 or i == len(matches):
            ranked = sorted(ratings.items(), key=lambda item: item[1], reverse=True)
            snapshot_states.append(
                {
                    "match_index": i,
                    "ratings": dict(ratings),
                    "ranked_ids": [hid for hid, _elo in ranked],
                }
            )
            snapshots.append(
                {
                    "match_index": i,
                    "active_hypotheses": len(ratings),
                    "top1_id": ranked[0][0] if ranked else "",
                    "top1_elo": ranked[0][1] if ranked else None,
                    "top5_ids": ";".join(hid for hid, _elo in ranked[:5]),
                    "top10_ids": ";".join(hid for hid, _elo in ranked[:10]),
                    "top5_mean_elo": _safe_mean([elo for _hid, elo in ranked[:5]]),
                    "top10_mean_elo": _safe_mean([elo for _hid, elo in ranked[:10]]),
                }
            )

    volatility_window = max(10, min(200, max(10, len(matches) // 20))) if matches else 10
    volatility_rows = _elo_volatility_rows(change_events, window=volatility_window)
    rank_stability_rows = _elo_rank_stability_rows(snapshot_states, final_rank)
    bootstrap_tier_rows = _elo_bootstrap_tier_rows(snapshot_states, ranked_final)
    stable_rank_match = _first_stable_threshold(
        rank_stability_rows,
        key="kendall_tau_final",
        threshold=0.90,
        min_remaining=3,
    )
    calibration_start_match = stable_rank_match if stable_rank_match is not None else _fallback_calibration_start(matches)
    calibration_rows = _elo_calibration_rows(
        matches,
        start_match=calibration_start_match,
        logistic_scale=elo_logistic_scale,
    )

    initial_volatility = volatility_rows[0].get("rating_change_std") if volatility_rows else None
    final_volatility = volatility_rows[-1].get("rating_change_std") if volatility_rows else None
    final_rank_row = rank_stability_rows[-1] if rank_stability_rows else {}
    calibration_errors = [
        row.get("abs_calibration_error") for row in calibration_rows for _ in range(int(row.get("match_count") or 0))
    ]

    metrics = {
        "snapshot_every": snapshot_every,
        "snapshots": len(snapshots),
        "final_top_elo": ranked_final[0][1] if ranked_final else None,
        "final_top10_ids": final_top10,
        "final_top5_mean_elo": _safe_mean([elo for _hid, elo in ranked_final[:5]]),
        "final_top5_min_elo": min([elo for _hid, elo in ranked_final[:5]], default=None),
        "final_top5_max_elo": max([elo for _hid, elo in ranked_final[:5]], default=None),
        "final_top10_mean_elo": _safe_mean([elo for _hid, elo in ranked_final[:10]]),
        "final_top10_min_elo": min([elo for _hid, elo in ranked_final[:10]], default=None),
        "final_top10_max_elo": max([elo for _hid, elo in ranked_final[:10]], default=None),
        "final_top5_percentile_mean": _safe_mean(top5_percentiles),
        "final_top5_percentile_min": min(top5_percentiles, default=None),
        "final_top10_percentile_mean": _safe_mean(top10_percentiles),
        "final_top10_percentile_min": min(top10_percentiles, default=None),
        "match_count_mean": _safe_mean(list(match_counts.values())),
        "match_count_min": min(match_counts.values(), default=0),
        "match_count_max": max(match_counts.values(), default=0),
        "match_count_p90": _percentile(list(match_counts.values()), 90),
        "volatility_window_matches": volatility_window,
        "volatility_initial_std": initial_volatility,
        "volatility_final_std": final_volatility,
        "volatility_decay_ratio": (
            final_volatility / initial_volatility
            if _is_finite(final_volatility) and _is_finite(initial_volatility) and float(initial_volatility) != 0.0
            else None
        ),
        "rank_stability_tau_0_90_match": stable_rank_match,
        "rank_stability_final_kendall_tau": final_rank_row.get("kendall_tau_final"),
        "rank_stability_final_spearman": final_rank_row.get("spearman_final"),
        "rank_stability_final_top10_jaccard": final_rank_row.get("top10_jaccard_final"),
        "calibration_start_match": calibration_start_match,
        "calibration_elo_logistic_scale": elo_logistic_scale,
        "calibration_bin_count": len(calibration_rows),
        "calibration_matches": sum(int(row.get("match_count") or 0) for row in calibration_rows),
        "calibration_mean_abs_error": _safe_mean(calibration_errors),
    }
    return {
        "metrics": metrics,
        "snapshots": snapshots,
        "volatility_rows": volatility_rows,
        "bootstrap_tier_rows": bootstrap_tier_rows,
        "rank_stability_rows": rank_stability_rows,
        "calibration_rows": calibration_rows,
        "match_counts": dict(match_counts),
        "earliest_seen_match": earliest_seen,
    }


def _elo_volatility_rows(change_events: list[dict[str, Any]], *, window: int) -> list[dict[str, Any]]:
    rows = []
    for idx, event in enumerate(change_events):
        start = max(0, idx - window + 1)
        window_events = change_events[start : idx + 1]
        signed = [float(value) for item in window_events for value in item.get("signed_changes", []) if _is_finite(value)]
        abs_values = [float(value) for item in window_events for value in item.get("abs_changes", []) if _is_finite(value)]
        rows.append(
            {
                "match_index": event.get("match_index"),
                "window_matches": len(window_events),
                "rating_change_events": len(signed),
                "rating_change_std": float(np.std(signed, ddof=1)) if len(signed) > 1 else 0.0,
                "mean_abs_rating_change": _safe_mean(abs_values),
                "median_abs_rating_change": _safe_median(abs_values),
            }
        )
    return rows


def _elo_rank_stability_rows(
    snapshot_states: list[dict[str, Any]],
    final_rank: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    final_top5 = {hid for hid, rank in final_rank.items() if rank <= 5}
    final_top10 = {hid for hid, rank in final_rank.items() if rank <= 10}
    for state in snapshot_states:
        ranked_ids = [str(hid) for hid in state.get("ranked_ids", [])]
        current_rank = {hid: rank for rank, hid in enumerate(ranked_ids, 1)}
        common = [hid for hid in ranked_ids if hid in final_rank]
        current_top5 = set(ranked_ids[:5])
        current_top10 = set(ranked_ids[:10])
        rows.append(
            {
                "match_index": state.get("match_index"),
                "active_hypotheses": len(ranked_ids),
                "common_with_final": len(common),
                "kendall_tau_final": _kendall_tau_from_rank_maps(current_rank, final_rank, common),
                "spearman_final": _spearman_corr(
                    [current_rank[hid] for hid in common],
                    [final_rank[hid] for hid in common],
                ),
                "top5_jaccard_final": _jaccard(current_top5, final_top5),
                "top10_jaccard_final": _jaccard(current_top10, final_top10),
            }
        )
    return rows


def _elo_bootstrap_tier_rows(
    snapshot_states: list[dict[str, Any]],
    ranked_final: list[tuple[str, float]],
) -> list[dict[str, Any]]:
    tiers = _final_rank_tiers(ranked_final)
    rng = np.random.default_rng(13)
    rows = []
    for state in snapshot_states:
        ratings = state.get("ratings") or {}
        for tier_name, ids in tiers:
            values = [float(ratings[hid]) for hid in ids if hid in ratings and _is_finite(ratings[hid])]
            mean, low, high = _bootstrap_mean_ci(values, rng=rng, n_boot=500)
            rows.append(
                {
                    "match_index": state.get("match_index"),
                    "tier": tier_name,
                    "tier_size": len(ids),
                    "active_in_tier": len(values),
                    "mean_elo": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return rows


def _final_rank_tiers(ranked_final: list[tuple[str, float]]) -> list[tuple[str, list[str]]]:
    specs = [
        ("final rank 1-5", 0, 5),
        ("final rank 6-10", 5, 10),
        ("final rank 11-25", 10, 25),
        ("final rank 26+", 25, len(ranked_final)),
    ]
    tiers = []
    for name, start, end in specs:
        ids = [hid for hid, _elo in ranked_final[start:end]]
        if ids:
            tiers.append((name, ids))
    return tiers


def _bootstrap_mean_ci(
    values: list[float],
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float | None, float | None, float | None]:
    vals = np.asarray([float(value) for value in values if _is_finite(value)], dtype="float64")
    if vals.size == 0:
        return None, None, None
    mean = float(np.mean(vals))
    if vals.size == 1:
        return mean, mean, mean
    samples = rng.choice(vals, size=(n_boot, vals.size), replace=True)
    means = np.mean(samples, axis=1)
    return mean, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _first_stable_threshold(
    rows: list[dict[str, Any]],
    *,
    key: str,
    threshold: float,
    min_remaining: int,
) -> int | None:
    clean = [row for row in rows if _is_finite(row.get(key)) and _is_finite(row.get("match_index"))]
    for idx, row in enumerate(clean):
        tail = clean[idx:]
        if len(tail) < min_remaining:
            continue
        if all(float(item[key]) >= threshold for item in tail):
            return int(float(row["match_index"]))
    return None


def _fallback_calibration_start(matches: list[dict[str, Any]]) -> int | None:
    if not matches:
        return None
    return max(1, math.ceil(len(matches) * 2 / 3))


def _elo_calibration_rows(
    matches: list[dict[str, Any]],
    *,
    start_match: int | None,
    bin_width: int = 50,
    logistic_scale: float = 400.0,
) -> list[dict[str, Any]]:
    bins: dict[int, list[dict[str, float]]] = defaultdict(list)
    for i, match in enumerate(matches, 1):
        if start_match is not None and i < start_match:
            continue
        a0 = _to_float(match.get("elo_a_before"))
        b0 = _to_float(match.get("elo_b_before"))
        winner = str(match.get("winner") or "").lower()
        if a0 is None or b0 is None or winner not in {"a", "b", "draw"}:
            continue
        gap = abs(a0 - b0)
        if a0 >= b0:
            favorite_score = 1.0 if winner == "a" else 0.0 if winner == "b" else 0.5
        else:
            favorite_score = 1.0 if winner == "b" else 0.0 if winner == "a" else 0.5
        bucket = int(gap // bin_width) * bin_width
        bins[bucket].append(
            {
                "gap": gap,
                "favorite_score": favorite_score,
                "theoretical": _elo_expected_favorite_probability(
                    gap, logistic_scale=logistic_scale
                ),
            }
        )

    rows = []
    for bucket in sorted(bins):
        items = bins[bucket]
        empirical = _safe_mean([item["favorite_score"] for item in items])
        theoretical = _safe_mean([item["theoretical"] for item in items])
        rows.append(
            {
                "elo_gap_bin_low": bucket,
                "elo_gap_bin_high": bucket + bin_width,
                "mean_elo_gap": _safe_mean([item["gap"] for item in items]),
                "match_count": len(items),
                "empirical_favorite_score": empirical,
                "theoretical_favorite_win_probability": theoretical,
                "abs_calibration_error": (
                    abs(empirical - theoretical)
                    if empirical is not None and theoretical is not None
                    else None
                ),
            }
        )
    return rows


def _elo_expected_favorite_probability(gap: float, *, logistic_scale: float = 400.0) -> float:
    scale = float(logistic_scale)
    if scale <= 0:
        scale = 400.0
    return 1.0 / (1.0 + 10.0 ** (-float(gap) / scale))


def _session_elo_logistic_scale(session: dict[str, Any]) -> float:
    raw = session.get("config_snapshot")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    if not isinstance(raw, dict):
        return 400.0
    ranking = raw.get("ranking")
    if not isinstance(ranking, dict):
        return 400.0
    try:
        scale = float(ranking.get("elo_logistic_scale", 400.0))
    except (TypeError, ValueError):
        return 400.0
    return scale if scale > 0 else 400.0


def _kendall_tau_from_rank_maps(
    rank_a: dict[str, int],
    rank_b: dict[str, int],
    ids: list[str],
) -> float | None:
    common = [hid for hid in ids if hid in rank_a and hid in rank_b]
    if len(common) < 2:
        return None
    concordant = 0
    discordant = 0
    for i, left in enumerate(common[:-1]):
        for right in common[i + 1 :]:
            da = rank_a[left] - rank_a[right]
            db = rank_b[left] - rank_b[right]
            product = da * db
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return None
    return (concordant - discordant) / total


def _jaccard(left: set[str], right: set[str]) -> float | None:
    denom = len(left | right)
    if denom == 0:
        return None
    return len(left & right) / denom


def _load_hypothesis_vectors(cfg: Config, session_id: str) -> dict[str, Any]:
    vector_dir = cfg.session_vector_dir(session_id)
    index_path = vector_dir / "index.faiss"
    meta_path = vector_dir / "index.meta.json"
    if faiss is None or not index_path.exists() or not meta_path.exists():
        return {"available": False, "count": 0, "ids": [], "vectors": None, "error": "missing FAISS index"}
    try:
        idx = faiss.read_index(str(index_path))
        meta = _read_json(meta_path)
        ids = list(meta.get("ordered_ids", [])) if isinstance(meta, dict) else []
        n = min(idx.ntotal, len(ids))
        vectors = idx.reconstruct_n(0, n).astype("float32") if n else np.zeros((0, 0), dtype="float32")
        vectors = _normalize_rows(vectors)
        return {
            "available": True,
            "count": n,
            "index_total": idx.ntotal,
            "ids": ids[:n],
            "vectors": vectors,
            "dim": int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else None,
            "missing_ids": [],
        }
    except Exception as exc:
        return {"available": False, "count": 0, "ids": [], "vectors": None, "error": str(exc)}


def _analyze_hypothesis_vectors(
    hypotheses: list[dict[str, Any]],
    hyp_vectors: dict[str, Any],
    paths: _Paths,
    *,
    n_clusters: int,
) -> dict[str, Any]:
    all_ids = [str(h.get("id")) for h in hypotheses if h.get("id")]
    ids = list(hyp_vectors.get("ids") or [])
    vectors = hyp_vectors.get("vectors")
    missing_ids = sorted(set(all_ids) - set(ids))
    hyp_vectors["missing_ids"] = missing_ids

    if vectors is None or len(ids) < 2:
        return {
            "available": False,
            "metrics": {"embedding_coverage": len(ids) / len(all_ids) if all_ids else None},
            "per_hypothesis": [],
            "cluster_rows": [],
            "pca_rows": [],
            "umap_rows": [],
        }

    sims = np.clip(vectors @ vectors.T, -1.0, 1.0)
    upper = sims[np.triu_indices_from(sims, k=1)]
    elo_by_id = {str(h.get("id")): _to_float(h.get("elo")) for h in hypotheses if h.get("id")}
    ranked_ids = [
        str(h.get("id"))
        for h in sorted(hypotheses, key=lambda item: float(item.get("elo") or 0.0), reverse=True)
        if h.get("id")
    ]
    rank_by_id = {hid: rank for rank, hid in enumerate(ranked_ids, 1)}
    index_by_id = {hid: idx for idx, hid in enumerate(ids)}
    top5_ids = [hid for hid in ranked_ids[:5] if hid in index_by_id]
    top10_ids = [hid for hid in ranked_ids[:10] if hid in index_by_id]
    bottom10_ids = [hid for hid in ranked_ids[-10:] if hid in index_by_id]
    nearest: dict[str, dict[str, Any]] = {}
    for i, hid in enumerate(ids):
        row = sims[i].copy()
        row[i] = -np.inf
        j = int(np.argmax(row))
        nearest[hid] = {"nearest_id": ids[j], "nearest_similarity": float(row[j])}

    k = min(max(2, n_clusters), len(ids))
    labels = _kmeans_labels(vectors, k)
    silhouette = None
    if len(set(labels)) > 1 and len(set(labels)) < len(labels):
        try:
            silhouette = float(silhouette_score(vectors, labels, metric="cosine"))
        except Exception:
            silhouette = None

    pca_rows: list[dict[str, Any]] = []
    umap_rows: list[dict[str, Any]] = []
    matches_by_id = {str(h.get("id")): int(h.get("matches_played") or 0) for h in hypotheses}
    title_by_id = {str(h.get("id")): str(h.get("title") or "") for h in hypotheses}
    if len(ids) >= 2:
        coords = PCA(n_components=2, svd_solver="randomized", random_state=13).fit_transform(vectors)
        for hid, (x, y), label in zip(ids, coords, labels, strict=False):
            pca_rows.append(
                {
                    "hypothesis_id": hid,
                    "x": float(x),
                    "y": float(y),
                    "cluster": int(label),
                    "elo": elo_by_id.get(hid),
                    "rank": rank_by_id.get(hid),
                    "matches_played": matches_by_id.get(hid),
                    "title": title_by_id.get(hid, ""),
                }
            )

        if umap is not None:
            try:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(30, max(2, len(ids) - 1)),
                    min_dist=0.05,
                    metric="cosine",
                    random_state=13,
                )
                umap_coords = reducer.fit_transform(vectors).astype("float32")
                for hid, (x, y), label in zip(ids, umap_coords, labels, strict=False):
                    umap_rows.append(
                        {
                            "hypothesis_id": hid,
                            "x": float(x),
                            "y": float(y),
                            "cluster": int(label),
                            "elo": elo_by_id.get(hid),
                            "rank": rank_by_id.get(hid),
                            "matches_played": matches_by_id.get(hid),
                            "title": title_by_id.get(hid, ""),
                        }
                    )
            except Exception:
                umap_rows = []

    cluster_rows = []
    for label in sorted(set(labels)):
        members = [ids[i] for i, value in enumerate(labels) if value == label]
        member_idx = [i for i, value in enumerate(labels) if value == label]
        within = []
        if len(member_idx) > 1:
            sub = sims[np.ix_(member_idx, member_idx)]
            within = list(sub[np.triu_indices_from(sub, k=1)])
        cluster_rows.append(
            {
                "cluster": int(label),
                "size": len(members),
                "mean_within_similarity": _safe_mean([float(x) for x in within]),
                "members": ";".join(members),
            }
        )

    def subset_pairwise_similarity(member_ids: list[str]) -> list[float]:
        idx = [index_by_id[hid] for hid in member_ids if hid in index_by_id]
        if len(idx) < 2:
            return []
        sub = sims[np.ix_(idx, idx)]
        return [float(x) for x in sub[np.triu_indices_from(sub, k=1)]]

    def mean_similarity_to(i: int, member_ids: list[str]) -> float | None:
        idx = [index_by_id[hid] for hid in member_ids if hid in index_by_id and index_by_id[hid] != i]
        if not idx:
            return None
        return _safe_mean([float(sims[i, j]) for j in idx])

    def max_similarity_to(i: int, member_ids: list[str]) -> float | None:
        idx = [index_by_id[hid] for hid in member_ids if hid in index_by_id and index_by_id[hid] != i]
        if not idx:
            return None
        return max(float(sims[i, j]) for j in idx)

    top5_pairwise = subset_pairwise_similarity(top5_ids)
    top10_pairwise = subset_pairwise_similarity(top10_ids)
    bottom10_pairwise = subset_pairwise_similarity(bottom10_ids)
    top10_idx = [index_by_id[hid] for hid in top10_ids if hid in index_by_id]
    bottom10_idx = [index_by_id[hid] for hid in bottom10_ids if hid in index_by_id]
    top10_to_bottom10 = [
        float(sims[i, j])
        for i in top10_idx
        for j in bottom10_idx
        if i != j
    ]

    per_hypothesis = []
    rank_similarity_rows = []
    labels_by_id = {hid: int(label) for hid, label in zip(ids, labels, strict=False)}
    for hyp in hypotheses:
        hid = str(hyp.get("id"))
        idx = index_by_id.get(hid)
        nearest_similarity = nearest.get(hid, {}).get("nearest_similarity")
        embedding_novelty = 1.0 - nearest_similarity if nearest_similarity is not None else None
        sim_to_top5 = mean_similarity_to(idx, top5_ids) if idx is not None else None
        sim_to_top10 = mean_similarity_to(idx, top10_ids) if idx is not None else None
        max_sim_to_top10 = max_similarity_to(idx, top10_ids) if idx is not None else None
        row = {
            "hypothesis_id": hid,
            "rank": rank_by_id.get(hid),
            "elo": elo_by_id.get(hid),
            "cluster": labels_by_id.get(hid),
            "nearest_id": nearest.get(hid, {}).get("nearest_id"),
            "nearest_similarity": nearest_similarity,
            "embedding_novelty": embedding_novelty,
            "mean_similarity_to_top5": sim_to_top5,
            "mean_similarity_to_top10": sim_to_top10,
            "max_similarity_to_top10": max_sim_to_top10,
        }
        per_hypothesis.append(row)
        if rank_by_id.get(hid) is not None:
            rank_similarity_rows.append(row)

    rank_similarity_rows.sort(key=lambda row: int(row.get("rank") or 10_000))
    rank_bucket_rows = []
    if rank_similarity_rows:
        bucket_count = min(5, len(rank_similarity_rows))
        bucket_size = math.ceil(len(rank_similarity_rows) / bucket_count)
        for bucket_idx in range(bucket_count):
            rows = rank_similarity_rows[bucket_idx * bucket_size : (bucket_idx + 1) * bucket_size]
            if not rows:
                continue
            rank_bucket_rows.append(
                {
                    "rank_bucket": f"{rows[0].get('rank')}-{rows[-1].get('rank')}",
                    "count": len(rows),
                    "mean_elo": _safe_mean([row.get("elo") for row in rows]),
                    "mean_nearest_similarity": _safe_mean([row.get("nearest_similarity") for row in rows]),
                    "mean_embedding_novelty": _safe_mean([row.get("embedding_novelty") for row in rows]),
                    "mean_similarity_to_top10": _safe_mean([row.get("mean_similarity_to_top10") for row in rows]),
                    "max_similarity_to_top10_mean": _safe_mean([row.get("max_similarity_to_top10") for row in rows]),
                }
            )

    elo_values_for_ranked = [row.get("elo") for row in rank_similarity_rows]
    novelty_values_for_ranked = [row.get("embedding_novelty") for row in rank_similarity_rows]
    nearest_values_for_ranked = [row.get("nearest_similarity") for row in rank_similarity_rows]
    sim_top10_values_for_ranked = [row.get("mean_similarity_to_top10") for row in rank_similarity_rows]

    metrics = {
        "available": True,
        "vectors": len(ids),
        "embedding_coverage": len(ids) / len(all_ids) if all_ids else None,
        "missing_vector_count": len(missing_ids),
        "missing_vector_ids": missing_ids,
        "pairwise_similarity_mean": _safe_mean([float(x) for x in upper]),
        "pairwise_similarity_median": _safe_median([float(x) for x in upper]),
        "pairwise_similarity_p90": _percentile([float(x) for x in upper], 90),
        "pairwise_similarity_p95": _percentile([float(x) for x in upper], 95),
        "pairwise_similarity_max": float(np.max(upper)) if upper.size else None,
        "near_duplicate_pairs_0_90": int(np.sum(upper >= 0.90)),
        "near_duplicate_pairs_0_95": int(np.sum(upper >= 0.95)),
        "near_duplicate_pairs_0_98": int(np.sum(upper >= 0.98)),
        "embedding_novelty_mean": _safe_mean([row.get("embedding_novelty") for row in rank_similarity_rows]),
        "embedding_novelty_median": _safe_median([row.get("embedding_novelty") for row in rank_similarity_rows]),
        "embedding_novelty_p10": _percentile([row.get("embedding_novelty") for row in rank_similarity_rows], 10),
        "embedding_novelty_p90": _percentile([row.get("embedding_novelty") for row in rank_similarity_rows], 90),
        "top5_pairwise_similarity_mean": _safe_mean(top5_pairwise),
        "top5_pairwise_similarity_min": min(top5_pairwise, default=None),
        "top5_pairwise_similarity_max": max(top5_pairwise, default=None),
        "top10_pairwise_similarity_mean": _safe_mean(top10_pairwise),
        "top10_pairwise_similarity_min": min(top10_pairwise, default=None),
        "top10_pairwise_similarity_max": max(top10_pairwise, default=None),
        "bottom10_pairwise_similarity_mean": _safe_mean(bottom10_pairwise),
        "top10_to_bottom10_similarity_mean": _safe_mean(top10_to_bottom10),
        "top10_to_bottom10_similarity_max": max(top10_to_bottom10, default=None),
        "elo_vs_embedding_novelty_pearson": _pearson_corr(elo_values_for_ranked, novelty_values_for_ranked),
        "elo_vs_embedding_novelty_spearman": _spearman_corr(elo_values_for_ranked, novelty_values_for_ranked),
        "elo_vs_nearest_similarity_spearman": _spearman_corr(elo_values_for_ranked, nearest_values_for_ranked),
        "elo_vs_similarity_to_top10_spearman": _spearman_corr(elo_values_for_ranked, sim_top10_values_for_ranked),
        "cluster_count": len(set(labels)),
        "silhouette_cosine": silhouette,
        "umap_available": bool(umap_rows),
        "umap_points": len(umap_rows),
        "umap_status": "ok" if umap_rows else ("umap-learn not installed" if umap is None else "unavailable"),
    }

    return {
        "available": True,
        "metrics": metrics,
        "pairwise_similarities": [float(x) for x in upper],
        "per_hypothesis": per_hypothesis,
        "rank_similarity_rows": rank_similarity_rows,
        "rank_bucket_rows": rank_bucket_rows,
        "cluster_rows": cluster_rows,
        "pca_rows": pca_rows,
        "umap_rows": umap_rows,
    }



def _analyze_density_reductions(
    vectors: np.ndarray,
    base_rows: list[dict[str, Any]],
    *,
    texts: list[str] | None = None,
    sources: list[str] | None = None,
    label_key: str,
) -> dict[str, Any]:
    if vectors is None or len(vectors) < 5:
        return {
            "methods": {
                "umap": {"available": False, "status": "not_enough_points", "rows": [], "cluster_rows": [], "metrics": {}},
            },
            "metrics": {},
        }

    texts = texts or ["" for _ in range(len(vectors))]
    sources = sources or ["" for _ in range(len(vectors))]
    methods = {
        "umap": _density_reduction_method(
            "umap", vectors, base_rows, texts=texts, sources=sources, label_key=label_key
        ),
    }
    return {
        "methods": methods,
        "metrics": {
            f"{name}_{key}": value
            for name, method in methods.items()
            for key, value in (method.get("metrics") or {}).items()
        },
    }


def _density_reduction_method(
    method: str,
    vectors: np.ndarray,
    base_rows: list[dict[str, Any]],
    *,
    texts: list[str],
    sources: list[str],
    label_key: str,
) -> dict[str, Any]:
    n = len(vectors)
    try:
        if method == "umap":
            if umap is None:
                return {
                    "available": False,
                    "status": "umap-learn not installed",
                    "rows": [],
                    "cluster_rows": [],
                    "metrics": {},
                }
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(30, max(2, n - 1)),
                min_dist=0.05,
                metric="cosine",
                random_state=13,
            )
            coords = reducer.fit_transform(vectors).astype("float32")
        else:
            return {"available": False, "status": f"unknown method: {method}", "rows": [], "cluster_rows": [], "metrics": {}}

        min_cluster_size = _density_min_cluster_size(n)
        min_samples = max(2, min_cluster_size // 2)
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            allow_single_cluster=False,
            copy=False,
        ).fit_predict(coords)

        rows = []
        for i, ((x, y), cluster) in enumerate(zip(coords, labels, strict=False)):
            row = dict(base_rows[i]) if i < len(base_rows) else {}
            row.update(
                {
                    "reduction": method,
                    "x": float(x),
                    "y": float(y),
                    "density_cluster": int(cluster),
                }
            )
            rows.append(row)

        cluster_rows = _density_cluster_rows(
            method, labels, rows, texts=texts, sources=sources, label_key=label_key, vectors=vectors
        )
        cluster_values = sorted({int(label) for label in labels if int(label) >= 0})
        noise_count = int(sum(1 for label in labels if int(label) == -1))
        silhouette = None
        clustered_idx = [idx for idx, label in enumerate(labels) if int(label) >= 0]
        if len(cluster_values) > 1 and len(clustered_idx) > len(cluster_values):
            try:
                silhouette = float(
                    silhouette_score(
                        coords[clustered_idx],
                        [int(labels[idx]) for idx in clustered_idx],
                        metric="euclidean",
                    )
                )
            except Exception:
                silhouette = None

        return {
            "available": True,
            "status": "ok",
            "rows": rows,
            "cluster_rows": cluster_rows,
            "metrics": {
                "available": True,
                "points": n,
                "cluster_count": len(cluster_values),
                "noise_count": noise_count,
                "noise_fraction": noise_count / n if n else None,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "silhouette_euclidean_clustered": silhouette,
            },
        }
    except Exception as exc:
        return {
            "available": False,
            "status": f"error: {_truncate(str(exc), 220)}",
            "rows": [],
            "cluster_rows": [],
            "metrics": {},
        }



def _density_min_cluster_size(n: int) -> int:
    if n < 20:
        return max(2, n // 3)
    if n < 200:
        return max(5, n // 12)
    return max(10, min(80, n // 80))


def _density_cluster_rows(
    method: str,
    labels: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    texts: list[str],
    sources: list[str],
    label_key: str,
    vectors: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    out = []
    for label in sorted({int(x) for x in labels}):
        member_idx = [i for i, value in enumerate(labels) if int(value) == label]
        member_rows = [rows[i] for i in member_idx]
        member_texts = [texts[i] for i in member_idx if i < len(texts)]
        member_sources = [sources[i] for i in member_idx if i < len(sources)]
        top_terms = _top_terms(member_texts, limit=12) if member_texts else []
        examples = _density_cluster_examples(
            vectors, member_idx, rows, texts=texts, sources=sources, limit=6
        )
        top_items = []
        for row in sorted(member_rows, key=lambda item: float(item.get("elo") or 0), reverse=True)[:6]:
            value = str(row.get(label_key) or row.get("source") or row.get("hypothesis_id") or "")
            if value:
                top_items.append(_truncate(value, 90))
        out.append(
            {
                "reduction": method,
                "density_cluster": label,
                "is_noise": label == -1,
                "size": len(member_idx),
                "fraction": len(member_idx) / len(labels) if len(labels) else None,
                "unique_documents": len(set(member_sources)) if member_sources else None,
                "mean_elo": _safe_mean([row.get("elo") for row in member_rows]),
                "max_elo": max([row.get("elo") for row in member_rows if row.get("elo") is not None], default=None),
                "top_terms": ", ".join(top_terms),
                "deterministic_label": (
                    "HDBSCAN noise / sparse chunks" if label == -1 else _deterministic_cluster_label(top_terms)
                ),
                "example_snippets": " || ".join(item["snippet"] for item in examples[:3]),
                "top_documents": ";".join(doc for doc, _count in Counter(member_sources).most_common(8)),
                "top_items": "; ".join(top_items),
                "examples": examples,
            }
        )
    return out


def _density_cluster_examples(
    vectors: np.ndarray | None,
    member_indices: list[int],
    rows: list[dict[str, Any]],
    *,
    texts: list[str],
    sources: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if not member_indices:
        return []

    scores: dict[int, float] = {}
    if vectors is not None and len(vectors) > max(member_indices):
        sub = vectors[member_indices]
        centroid = np.mean(sub, axis=0)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        scores = {idx: float(np.dot(vectors[idx], centroid)) for idx in member_indices}

    examples = []
    for idx in member_indices:
        row = rows[idx] if idx < len(rows) else {}
        source = sources[idx] if idx < len(sources) else str(row.get("source") or "")
        text = " ".join(str(texts[idx] if idx < len(texts) else "").split())
        if not text:
            continue
        examples.append(
            (
                scores.get(idx, float(len(text))),
                {
                    "source": str(source or row.get("source") or row.get("file") or ""),
                    "chunk_id": row.get("chunk_id"),
                    "snippet": _truncate(text, 700),
                },
            )
        )
    examples.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _score, item in examples[:limit]]


def _top_terms(texts: list[str], *, limit: int) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for token in _TOKEN_RE.findall(text.lower()):
            if len(token) < 3 or token in _ANALYSIS_STOPWORDS:
                continue
            if token.isdigit():
                continue
            counts[_stem_analysis_token(token)] += 1
    return [term for term, _count in counts.most_common(limit)]


def _stem_analysis_token(token: str) -> str:
    if token in {"implanted", "implanting", "implantation", "implants"}:
        return "implant"
    if token in {"irradiated", "irradiating", "irradiation"}:
        return "irradiat"
    if token in {"defects", "defective"}:
        return "defect"
    if token in {"monolayers"}:
        return "monolayer"
    if token in {"materials"}:
        return "material"
    if token in {"substrates"}:
        return "substrate"
    if token in {"ions"}:
        return "ion"
    if token in {"tmds"}:
        return "tmd"
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _deterministic_cluster_label(top_terms: list[str]) -> str:
    return ", ".join(top_terms[:4]) if top_terms else "unlabeled cluster"


def _representative_cluster_examples(
    vectors: np.ndarray,
    member_indices: list[int],
    meta: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not member_indices:
        return []
    sub = vectors[member_indices]
    centroid = np.mean(sub, axis=0)
    norm = np.linalg.norm(centroid)
    if norm:
        centroid = centroid / norm
    scored = []
    for idx in member_indices:
        score = float(np.dot(vectors[idx], centroid))
        item = meta[idx]
        text = " ".join(str(item.get("text") or "").split())
        scored.append(
            (
                score,
                {
                    "source": str(item.get("source") or item.get("file") or ""),
                    "chunk_id": item.get("chunk_id"),
                    "snippet": _truncate(text, 700),
                },
            )
        )
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _score, item in scored[:limit]]


def _summarize_kb_clusters_with_llm(
    cfg: Config,
    research_goal: str,
    cluster_rows: list[dict[str, Any]],
    *,
    cluster_key: str = "cluster",
    chunks_key: str = "chunks",
    cluster_kind: str = "cluster",
    skip_negative_ids: bool = False,
) -> dict[int, dict[str, str]]:
    fallback: dict[int, dict[str, str]] = {}
    for row in cluster_rows:
        cluster_id = _cluster_row_id(row, cluster_key)
        if cluster_id is None:
            continue
        if skip_negative_ids and cluster_id < 0:
            fallback[cluster_id] = {
                "label": "HDBSCAN noise / sparse chunks",
                "summary": (
                    "Heterogeneous sampled chunks that HDBSCAN did not assign to any dense local neighborhood; "
                    "this row should not be interpreted as one semantic topic."
                ),
                "relevance": (
                    "May contain important, off-topic, or boundary material. Inspect examples and source documents "
                    "rather than treating the aggregate noise row as a cluster."
                ),
                "status": "hdbscan_noise_not_labeled",
            }
            continue
        fallback[cluster_id] = {
            "label": row.get("deterministic_label") or _label_from_top_terms(row.get("top_terms")) or "unlabeled cluster",
            "summary": "",
            "relevance": "",
            "status": "fallback_no_llm",
        }
    base_url = (
        getattr(getattr(cfg.llm, "openai", None), "base_url", None)
        or os.environ.get("OPENAI_BASE_URL")
    )
    provider = str(getattr(cfg.llm, "provider", "") or "").lower()
    if not base_url and provider not in {"openai", "openai_compatible"}:
        return fallback
    model = str(
        getattr(cfg.models, "metareview_final", "")
        or getattr(cfg.models, "judge", "")
        or getattr(cfg.models, "reflection", "")
    ).strip()
    if not model:
        return fallback

    try:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY") or "compat-no-key"
        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 90.0}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
    except Exception as exc:
        for item in fallback.values():
            item["status"] = f"llm_error:{_truncate(str(exc), 160)}"
        return fallback

    out = dict(fallback)
    for row in cluster_rows:
        cluster_id = _cluster_row_id(row, cluster_key)
        if cluster_id is None or cluster_id not in fallback:
            continue
        if skip_negative_ids and cluster_id < 0:
            continue
        examples = row.get("examples") if isinstance(row.get("examples"), list) else []
        snippets = []
        for ex in examples[:4]:
            if not isinstance(ex, dict):
                continue
            snippets.append(
                {
                    "source": ex.get("source"),
                    "chunk_id": ex.get("chunk_id"),
                    "snippet": _truncate(str(ex.get("snippet") or ""), 520),
                }
            )
        prompt = (
            "Label one semantic cluster from a RAG knowledge base used in a scientific hypothesis-generation session.\n"
            "Return exactly three lines and no markdown:\n"
            "LABEL: <short label, <=8 words>\n"
            "SUMMARY: <what this cluster generally represents, <=45 words>\n"
            "RELEVANCE: <how it relates to the research goal, <=45 words>\n\n"
            f"Research goal: {research_goal}\n"
            f"{cluster_kind} id: {cluster_id}\n"
            f"Chunk count: {row.get(chunks_key) or row.get('chunks') or row.get('size')} "
            f"from {row.get('unique_documents')} documents\n"
            f"Top terms: {row.get('top_terms')}\n"
            f"Top documents: {row.get('top_documents')}\n"
            f"Representative snippets: {json.dumps(snippets, ensure_ascii=False)}"
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You label scientific literature clusters concisely."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=900,
            )
            message = response.choices[0].message
            text = (message.content or getattr(message, "reasoning", None) or "")
            parsed = _parse_labeled_cluster_summary(text)
            out[cluster_id] = {
                "label": parsed.get("label") or fallback[cluster_id]["label"],
                "summary": parsed.get("summary") or "",
                "relevance": parsed.get("relevance") or "",
                "status": "llm_ok" if parsed else "llm_unparsed",
            }
        except Exception as exc:
            out[cluster_id]["status"] = f"llm_error:{_truncate(str(exc), 160)}"
    return out


def _cluster_row_id(row: dict[str, Any], key: str) -> int | None:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _label_from_top_terms(value: Any) -> str:
    terms = [term.strip() for term in str(value or "").split(",") if term.strip()]
    return _deterministic_cluster_label(terms) if terms else ""


def _parse_labeled_cluster_summary(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().lstrip("*-• ").strip()
        value = value.strip().strip("-* ")
        if key == "label" and value:
            value = _clean_cluster_label(value)
        elif key in {"summary", "relevance"} and value:
            value = _clean_cluster_text(value)
        if key in {"label", "summary", "relevance"} and value:
            out[key] = value
    if out:
        return out
    # Last resort for models that still return JSON despite instructions.
    try:
        decoded = _loads_llm_json(text)
    except Exception:
        return {}
    if isinstance(decoded, dict):
        return {
            "label": str(decoded.get("label") or ""),
            "summary": str(decoded.get("summary") or ""),
            "relevance": str(decoded.get("relevance") or ""),
        }
    return {}


def _clean_cluster_label(value: str) -> str:
    return _clean_cluster_text(value)


def _clean_cluster_text(value: str) -> str:
    cleaned = re.sub(r"\s*\([^)]*words?[^)]*\)\s*", " ", value, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*-\s*OK\.?$", "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split()).strip()


def _loads_llm_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_candidates = [idx for idx in (cleaned.find("["), cleaned.find("{")) if idx >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end <= start:
            raise
        return json.loads(cleaned[start:end + 1])

def _load_kb_vectors(cfg: Config, session_id: str, *, max_points: int) -> dict[str, Any]:
    rag_dir = cfg.session_rag_dir(session_id)
    index_path = rag_dir / "kb.index"
    meta_path = rag_dir / "kb.pkl"
    if faiss is None or not index_path.exists() or not meta_path.exists():
        return {"available": False, "count": 0, "sample_vectors": None, "sample_meta": []}
    try:
        with meta_path.open("rb") as handle:
            metadata = pickle.load(handle)
        if not isinstance(metadata, list):
            metadata = []
        idx = faiss.read_index(str(index_path))
        n = min(idx.ntotal, len(metadata))
        if n == 0:
            return {"available": False, "count": 0, "sample_vectors": None, "sample_meta": []}
        sample_n = min(max_points, n)
        rng = np.random.default_rng(13)
        sample_idx = np.sort(rng.choice(n, size=sample_n, replace=False)) if sample_n < n else np.arange(n)
        vectors = np.vstack([idx.reconstruct(int(i)) for i in sample_idx]).astype("float32")
        vectors = _normalize_rows(vectors)
        sample_meta = [metadata[int(i)] if isinstance(metadata[int(i)], dict) else {} for i in sample_idx]
        return {
            "available": True,
            "count": n,
            "sample_count": sample_n,
            "sample_indices": [int(i) for i in sample_idx],
            "sample_vectors": vectors,
            "sample_meta": sample_meta,
            "dim": int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else None,
            "metadata_count": len(metadata),
            "index_total": idx.ntotal,
        }
    except Exception as exc:
        return {"available": False, "count": 0, "sample_vectors": None, "sample_meta": [], "error": str(exc)}


def _analyze_kb_vectors(
    cfg: Config,
    kb: dict[str, Any],
    paths: _Paths,
    *,
    n_clusters: int,
    research_goal: str,
    projection: dict[str, Any],
) -> dict[str, Any]:
    vectors = kb.get("sample_vectors")
    meta = kb.get("sample_meta") or []
    if vectors is None or len(meta) < 2:
        return {
            "available": False,
            "metrics": {
                "available": bool(kb.get("available")),
                "chunks": kb.get("count", 0),
                "sample_chunks": kb.get("sample_count", 0),
                "error": kb.get("error"),
            },
            "cluster_rows": [],
            "pca_rows": [],
            "density": {"methods": {}, "metrics": {}},
        }

    sources = [str(m.get("source") or m.get("file") or m.get("url") or "unknown") for m in meta]
    source_counts = Counter(sources)
    sample_indices = [
        int(index) for index in (kb.get("sample_indices") or range(len(meta)))
    ]
    if (
        not projection.get("available")
        or len(projection.get("coordinates", [])) < max(sample_indices, default=-1) + 1
    ):
        return {
            "available": False,
            "metrics": {
                "available": bool(kb.get("available")),
                "chunks": kb.get("count", 0),
                "sample_chunks": kb.get("sample_count", 0),
                "error": projection.get("error") or "missing shared RAG projection",
            },
            "cluster_rows": [],
            "pca_rows": [],
            "density": {"methods": {}, "metrics": {}},
        }
    labels = [int(projection["labels"][index]) for index in sample_indices]
    silhouette = None
    if len(set(labels)) > 1 and len(set(labels)) < len(labels):
        try:
            silhouette = float(silhouette_score(vectors, labels, metric="cosine"))
        except Exception:
            silhouette = None

    pca_rows = []
    coords = projection["coordinates"][sample_indices]
    for source_index, (x, y), label, item in zip(
        sample_indices, coords, labels, meta, strict=False
    ):
        source = str(item.get("source") or item.get("file") or "")
        pca_rows.append(
            {
                "kb_sample_index": source_index,
                "x": float(x),
                "y": float(y),
                "cluster": int(label),
                "source": source,
                "chunk_id": item.get("chunk_id"),
                "text_chars": len(str(item.get("text") or "")),
            }
        )

    density_base_rows = []
    density_texts = []
    for i, item in enumerate(meta):
        source = str(item.get("source") or item.get("file") or item.get("url") or "unknown")
        density_base_rows.append(
            {
                "kb_sample_index": i,
                "source": source,
                "chunk_id": item.get("chunk_id"),
                "text_chars": len(str(item.get("text") or "")),
                "kmeans_cluster": int(labels[i]),
            }
        )
        density_texts.append(str(item.get("text") or ""))
    density_analysis = _analyze_density_reductions(
        vectors,
        density_base_rows,
        texts=density_texts,
        sources=sources,
        label_key="source",
    )

    cluster_rows = []
    for label in sorted(set(labels)):
        members = [i for i, value in enumerate(labels) if value == label]
        docs = {sources[i] for i in members}
        texts = [str(meta[i].get("text") or "") for i in members]
        top_terms = _top_terms(texts, limit=14)
        examples = _representative_cluster_examples(vectors, members, meta, limit=6)
        cluster_rows.append(
            {
                "cluster": int(label),
                "chunks": len(members),
                "unique_documents": len(docs),
                "top_terms": ", ".join(top_terms),
                "deterministic_label": _deterministic_cluster_label(top_terms),
                "example_snippets": " || ".join(item["snippet"] for item in examples[:3]),
                "top_documents": ";".join(doc for doc, _count in Counter(sources[i] for i in members).most_common(8)),
                "examples": examples,
            }
        )

    llm_cluster_summaries = _summarize_kb_clusters_with_llm(cfg, research_goal, cluster_rows)
    for row in cluster_rows:
        summary = llm_cluster_summaries.get(int(row["cluster"]), {})
        row["llm_label"] = summary.get("label") or row["deterministic_label"]
        row["llm_summary"] = summary.get("summary") or ""
        row["llm_relevance"] = summary.get("relevance") or ""
        row["llm_summary_status"] = summary.get("status") or ("fallback" if not summary else "ok")

    density_summary_statuses: list[str] = []
    for method_name, method in ((density_analysis.get("methods") or {}).items()):
        density_rows = method.get("cluster_rows") if isinstance(method, dict) else None
        if not isinstance(density_rows, list) or not density_rows:
            continue
        density_summaries = _summarize_kb_clusters_with_llm(
            cfg,
            research_goal,
            density_rows,
            cluster_key="density_cluster",
            chunks_key="size",
            cluster_kind=f"{method_name.upper()} HDBSCAN cluster",
            skip_negative_ids=True,
        )
        for row in density_rows:
            cluster_id = _cluster_row_id(row, "density_cluster")
            if cluster_id is None:
                continue
            summary = density_summaries.get(cluster_id, {})
            row["llm_label"] = summary.get("label") or row.get("deterministic_label") or _label_from_top_terms(row.get("top_terms"))
            row["llm_summary"] = summary.get("summary") or ""
            row["llm_relevance"] = summary.get("relevance") or ""
            row["llm_summary_status"] = summary.get("status") or ("fallback" if not summary else "ok")
            density_summary_statuses.append(str(row.get("llm_summary_status") or ""))

    sample_pairwise = _sample_pairwise_cosine(vectors, max_pairs=250_000)
    metrics = {
        "available": True,
        "chunks": kb.get("count", 0),
        "sample_chunks": len(meta),
        "metadata_count": kb.get("metadata_count"),
        "index_total": kb.get("index_total"),
        "unique_documents_in_sample": len(source_counts),
        "sample_document_entropy": _entropy(list(source_counts.values())),
        "cluster_count": len(set(labels)),
        "silhouette_cosine": silhouette,
        "projection_silhouette": projection.get("silhouette"),
        "projection_stability": projection.get("stability"),
        "projection_selected_k": projection.get("selected_k"),
        "projection_source_sha256": (projection.get("meta") or {}).get("source_sha256"),
        "projection_pca_components": len(projection.get("components", [])),
        "cluster_summary_status_counts": dict(Counter(str(row.get("llm_summary_status") or "") for row in cluster_rows)),
        "density_cluster_summary_status_counts": dict(Counter(density_summary_statuses)),
        "sample_pairwise_similarity_mean": _safe_mean(sample_pairwise),
        "sample_pairwise_similarity_median": _safe_median(sample_pairwise),
        "sample_pairwise_similarity_p95": _percentile(sample_pairwise, 95),
        **density_analysis.get("metrics", {}),
    }
    return {
        "available": True,
        "metrics": metrics,
        "cluster_rows": cluster_rows,
        "pca_rows": pca_rows,
        "density": density_analysis,
        "sample_pairwise_similarities": sample_pairwise,
    }


def _analyze_joint_pca(
    hyp_vectors: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    kb: dict[str, Any],
    kb_analysis: dict[str, Any],
    paths: _Paths,
    *,
    projection: dict[str, Any],
) -> dict[str, Any]:
    hvec = hyp_vectors.get("vectors")
    kvec = kb.get("sample_vectors")
    if hvec is None or kvec is None or hvec.size == 0 or kvec.size == 0:
        return {"available": False, "reason": "missing hypothesis or KB vectors", "rows": []}
    if hvec.shape[1] != kvec.shape[1]:
        return {
            "available": False,
            "reason": f"dimension mismatch: hypothesis={hvec.shape[1]} kb={kvec.shape[1]}",
            "rows": [],
        }
    if not projection.get("available"):
        return {
            "available": False,
            "reason": projection.get("error") or "missing shared RAG projection",
            "rows": [],
        }

    hyp_ids = list(hyp_vectors.get("ids") or [])
    elo_by_id = {str(h.get("id")): _to_float(h.get("elo")) for h in hypotheses}
    rank_by_id = {
        str(h.get("id")): rank
        for rank, h in enumerate(sorted(hypotheses, key=lambda x: float(x.get("elo") or 0), reverse=True), 1)
        if h.get("id")
    }
    kb_meta = kb.get("sample_meta") or []
    kb_pca_rows = kb_analysis.get("pca_rows") or []
    kb_labels = [row.get("cluster") for row in kb_pca_rows]
    if len(kb_labels) != len(kb_meta):
        kb_labels = [None] * len(kb_meta)
    cluster_info = {int(row["cluster"]): row for row in kb_analysis.get("cluster_rows", []) if row.get("cluster") is not None}

    try:
        transformed_hypotheses = transform_rag_vectors(projection, hvec)
    except ValueError as exc:
        return {"available": False, "reason": str(exc), "rows": []}
    hcoords = transformed_hypotheses["coordinates"]
    hypothesis_cluster_labels = transformed_hypotheses["labels"]
    sample_indices = [
        int(index) for index in (kb.get("sample_indices") or range(len(kb_meta)))
    ]
    if len(projection["coordinates"]) < max(sample_indices, default=-1) + 1:
        return {
            "available": False,
            "reason": "shared RAG projection does not cover sampled KB chunks",
            "rows": [],
        }
    kcoords = projection["coordinates"][sample_indices]
    rows = []
    for i, hid in enumerate(hyp_ids):
        rows.append(
            {
                "kind": "hypothesis",
                "id": hid,
                "x": float(hcoords[i, 0]),
                "y": float(hcoords[i, 1]),
                "elo": elo_by_id.get(hid),
                "rank": rank_by_id.get(hid),
                "source": "",
                "cluster": int(hypothesis_cluster_labels[i]),
            }
        )
    for j, item in enumerate(kb_meta):
        rows.append(
            {
                "kind": "kb_chunk",
                "id": f"kb_{sample_indices[j]}",
                "x": float(kcoords[j, 0]),
                "y": float(kcoords[j, 1]),
                "elo": None,
                "rank": None,
                "source": str(item.get("source") or ""),
                "cluster": kb_labels[j] if j < len(kb_labels) else None,
            }
        )

    kb_cluster_values = [int(label) if label is not None else None for label in kb_labels]
    pca_cluster_centroids: dict[int, np.ndarray] = {}
    for cluster in sorted({label for label in kb_cluster_values if label is not None}):
        mask = np.asarray([label == cluster for label in kb_cluster_values], dtype=bool)
        if mask.any():
            pca_cluster_centroids[cluster] = kcoords[mask].mean(axis=0)

    def kb_document_key(idx: int) -> str:
        item = kb_meta[idx] if idx < len(kb_meta) and isinstance(kb_meta[idx], dict) else {}
        for field in ("source", "file", "url", "pdf_url", "abs_url", "doi", "arxiv_id", "chemrxiv_id", "source_id"):
            value = str(item.get(field) or "").strip()
            if value:
                return _normalize_source_id(value)
        title = str(item.get("title") or item.get("title_key") or "").strip()
        if title:
            return f"title:{_title_key(title)}"
        return f"kb_sample:{idx}"

    alignment_rows = []
    shared_cluster_counts: Counter[int] = Counter()
    top10_shared_cluster_counts: Counter[int] = Counter()
    top_cluster_counts: Counter[int] = Counter()
    top10_cluster_counts: Counter[int] = Counter()
    pca_centroid_counts: Counter[int] = Counter()
    top10_pca_centroid_counts: Counter[int] = Counter()
    pca_point_counts: Counter[int] = Counter()
    top10_pca_point_counts: Counter[int] = Counter()
    for i, hid in enumerate(hyp_ids):
        sims = kvec @ hvec[i]
        if len(sims) == 0:
            continue
        shared_cluster = int(hypothesis_cluster_labels[i])
        shared_cluster_counts[shared_cluster] += 1
        rank = rank_by_id.get(hid)
        if rank is not None and rank <= 10:
            top10_shared_cluster_counts[shared_cluster] += 1

        best_by_document: dict[str, tuple[float, int]] = {}
        for idx, sim in enumerate(sims):
            doc_key = kb_document_key(int(idx))
            current = best_by_document.get(doc_key)
            if current is None or float(sim) > current[0]:
                best_by_document[doc_key] = (float(sim), int(idx))

        top_document_hits = sorted(
            (sim, idx, doc_key) for doc_key, (sim, idx) in best_by_document.items()
        )[-min(25, len(best_by_document)):][::-1]
        cluster_counts: Counter[int] = Counter()
        for _sim, idx, _doc_key in top_document_hits:
            label = kb_labels[int(idx)] if int(idx) < len(kb_labels) else None
            if label is not None:
                cluster_counts[int(label)] += 1
        nearest_cluster = cluster_counts.most_common(1)[0][0] if cluster_counts else None
        if nearest_cluster is not None:
            top_cluster_counts[nearest_cluster] += 1
            if rank is not None and rank <= 10:
                top10_cluster_counts[nearest_cluster] += 1
        pca_centroid_cluster = None
        pca_centroid_distance = None
        if pca_cluster_centroids:
            pca_distances = {
                cluster: float(np.linalg.norm(hcoords[i] - centroid))
                for cluster, centroid in pca_cluster_centroids.items()
            }
            if pca_distances:
                pca_centroid_cluster = min(pca_distances, key=pca_distances.get)
                pca_centroid_distance = pca_distances[pca_centroid_cluster]
                pca_centroid_counts[pca_centroid_cluster] += 1
                if rank is not None and rank <= 10:
                    top10_pca_centroid_counts[pca_centroid_cluster] += 1

        pca_point_cluster = None
        pca_point_distance = None
        if len(kcoords):
            point_distances = np.linalg.norm(kcoords - hcoords[i], axis=1)
            pca_point_idx = int(np.argmin(point_distances))
            pca_point_distance = float(point_distances[pca_point_idx])
            label = kb_cluster_values[pca_point_idx] if pca_point_idx < len(kb_cluster_values) else None
            pca_point_cluster = int(label) if label is not None else None
            if pca_point_cluster is not None:
                pca_point_counts[pca_point_cluster] += 1
                if rank is not None and rank <= 10:
                    top10_pca_point_counts[pca_point_cluster] += 1

        info = cluster_info.get(nearest_cluster if nearest_cluster is not None else -1, {})
        shared_info = cluster_info.get(shared_cluster, {})
        pca_centroid_info = cluster_info.get(
            pca_centroid_cluster if pca_centroid_cluster is not None else -1, {}
        )
        pca_point_info = cluster_info.get(pca_point_cluster if pca_point_cluster is not None else -1, {})
        alignment_rows.append(
            {
                "hypothesis_id": hid,
                "rank": rank,
                "elo": elo_by_id.get(hid),
                "shared_kb_cluster": shared_cluster,
                "shared_kb_cluster_label": shared_info.get("llm_label")
                or shared_info.get("deterministic_label"),
                "shared_kb_cluster_summary": shared_info.get("llm_summary"),
                "nearest_kb_cluster": nearest_cluster,
                "nearest_kb_cluster_label": info.get("llm_label") or info.get("deterministic_label"),
                "nearest_kb_cluster_summary": info.get("llm_summary"),
                "top25_cluster_counts": ";".join(f"{cluster}:{count}" for cluster, count in cluster_counts.most_common()),
                "top25_document_cluster_counts": ";".join(f"{cluster}:{count}" for cluster, count in cluster_counts.most_common()),
                "top25_document_count": len(top_document_hits),
                "top25_document_keys": ";".join(doc_key for _sim, _idx, doc_key in top_document_hits),
                "top1_kb_similarity": float(top_document_hits[0][0]) if top_document_hits else None,
                "mean_top25_kb_similarity": _safe_mean([sim for sim, _idx, _doc_key in top_document_hits]),
                "mean_top25_kb_document_similarity": _safe_mean([sim for sim, _idx, _doc_key in top_document_hits]),
                "joint_pca_nearest_centroid_cluster": pca_centroid_cluster,
                "joint_pca_nearest_centroid_label": pca_centroid_info.get("llm_label") or pca_centroid_info.get("deterministic_label"),
                "joint_pca_nearest_centroid_distance": pca_centroid_distance,
                "joint_pca_nearest_point_cluster": pca_point_cluster,
                "joint_pca_nearest_point_label": pca_point_info.get("llm_label") or pca_point_info.get("deterministic_label"),
                "joint_pca_nearest_point_distance": pca_point_distance,
            }
        )

    return {
        "available": True,
        "rows": rows,
        "alignment_rows": sorted(alignment_rows, key=lambda row: int(row.get("rank") or 10_000)),
        "metrics": {
            "hypothesis_points": len(hyp_ids),
            "kb_points": len(kb_meta),
            "embedding_dim": int(hvec.shape[1]),
            "shared_kb_cluster_counts": dict(shared_cluster_counts),
            "top10_shared_kb_cluster_counts": dict(top10_shared_cluster_counts),
            "hypothesis_nearest_kb_cluster_counts": dict(top_cluster_counts),
            "top10_nearest_kb_cluster_counts": dict(top10_cluster_counts),
            "hypothesis_nearest_kb_document_cluster_counts": dict(top_cluster_counts),
            "top10_nearest_kb_document_cluster_counts": dict(top10_cluster_counts),
            "joint_pca_centroid_cluster_counts": dict(pca_centroid_counts),
            "top10_joint_pca_centroid_cluster_counts": dict(top10_pca_centroid_counts),
            "joint_pca_nearest_point_cluster_counts": dict(pca_point_counts),
            "top10_joint_pca_nearest_point_cluster_counts": dict(top10_pca_point_counts),
            "mean_top1_kb_similarity": _safe_mean([row.get("top1_kb_similarity") for row in alignment_rows]),
            "mean_top25_kb_similarity": _safe_mean([row.get("mean_top25_kb_similarity") for row in alignment_rows]),
            "mean_top25_kb_document_similarity": _safe_mean([row.get("mean_top25_kb_document_similarity") for row in alignment_rows]),
        },
    }


def _write_tables(
    paths: _Paths,
    *,
    hypotheses: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    source_use: dict[str, Any],
    sources: dict[str, Any],
    transcript_summary: dict[str, Any],
    hyp_vector_analysis: dict[str, Any],
    debate_summary: dict[str, Any],
    elo_summary: dict[str, Any],
    kb_analysis: dict[str, Any],
    joint_pca: dict[str, Any],
) -> None:
    review_by_hyp: dict[str, dict[str, Any]] = {}
    for review in reviews:
        hid = str(review.get("hypothesis_id") or "")
        if not hid:
            continue
        review_by_hyp[hid] = review
    vector_by_hyp = {
        row["hypothesis_id"]: row for row in hyp_vector_analysis.get("per_hypothesis", [])
    }
    source_by_hyp = {
        row["hypothesis_id"]: row for row in source_use.get("per_hypothesis", [])
    }
    ranked = sorted(hypotheses, key=lambda h: float(h.get("elo") or 0.0), reverse=True)
    _write_csv(
        paths.tables / "top_hypotheses.csv",
        [
            {
                "rank": i,
                "hypothesis_id": h.get("id"),
                "title": h.get("title"),
                "elo": h.get("elo"),
                "matches_played": h.get("matches_played"),
                "state": h.get("state"),
                "created_by": h.get("created_by"),
                "strategy": h.get("strategy"),
                "parent_count": len(_parse_parent_ids(h.get("parent_ids"))),
                "novelty": review_by_hyp.get(str(h.get("id")), {}).get("novelty"),
                "correctness": review_by_hyp.get(str(h.get("id")), {}).get("correctness"),
                "testability": review_by_hyp.get(str(h.get("id")), {}).get("testability"),
                "feasibility": review_by_hyp.get(str(h.get("id")), {}).get("feasibility"),
                "cluster": vector_by_hyp.get(str(h.get("id")), {}).get("cluster"),
                "nearest_id": vector_by_hyp.get(str(h.get("id")), {}).get("nearest_id"),
                "nearest_similarity": vector_by_hyp.get(str(h.get("id")), {}).get("nearest_similarity"),
                "embedding_novelty": vector_by_hyp.get(str(h.get("id")), {}).get("embedding_novelty"),
                "mean_similarity_to_top10": vector_by_hyp.get(str(h.get("id")), {}).get("mean_similarity_to_top10"),
                "max_similarity_to_top10": vector_by_hyp.get(str(h.get("id")), {}).get("max_similarity_to_top10"),
                "explicit_source_tokens": source_by_hyp.get(str(h.get("id")), {}).get("explicit_source_tokens"),
                "matched_known_sources": source_by_hyp.get(str(h.get("id")), {}).get("matched_known_sources"),
            }
            for i, h in enumerate(ranked, 1)
        ],
    )
    _write_csv(paths.tables / "hypothesis_source_use.csv", source_use.get("per_hypothesis", []))
    _write_csv(paths.tables / "hypothesis_rank_similarity.csv", hyp_vector_analysis.get("rank_similarity_rows", []))
    _write_csv(paths.tables / "hypothesis_similarity_rank_buckets.csv", hyp_vector_analysis.get("rank_bucket_rows", []))
    _write_csv(paths.tables / "top_sources.csv", source_use.get("top_sources", []))
    _write_csv(paths.tables / "debate_bias_by_match.csv", debate_summary.get("bias_rows", []))
    _write_csv(paths.tables / "debate_bias_summary.csv", debate_summary.get("bias_summary_rows", []))
    _write_csv(paths.tables / "hypothesis_clusters.csv", hyp_vector_analysis.get("cluster_rows", []))
    _write_csv(paths.tables / "hypothesis_pca.csv", hyp_vector_analysis.get("pca_rows", []))
    _write_csv(paths.tables / "hypothesis_umap.csv", hyp_vector_analysis.get("umap_rows", []))
    for deprecated in (
        "hypothesis_pca_clusters.csv",
        "hypothesis_umap_hdbscan.csv",
        "hypothesis_umap_hdbscan_clusters.csv",
    ):
        (paths.tables / deprecated).unlink(missing_ok=True)
    _write_csv(paths.tables / "kb_clusters.csv", kb_analysis.get("cluster_rows", []))
    _write_csv(paths.tables / "kb_pca.csv", kb_analysis.get("pca_rows", []))
    kb_umap = (((kb_analysis.get("density") or {}).get("methods") or {}).get("umap") or {})
    _write_csv(paths.tables / "kb_umap_hdbscan.csv", kb_umap.get("rows", []))
    _write_csv(paths.tables / "kb_umap_hdbscan_clusters.csv", kb_umap.get("cluster_rows", []))
    _write_csv(paths.tables / "joint_pca_alignment.csv", joint_pca.get("alignment_rows", []))
    _write_csv(
        paths.tables / "joint_pca_visual_alignment.csv",
        [
            {
                "hypothesis_id": row.get("hypothesis_id"),
                "rank": row.get("rank"),
                "elo": row.get("elo"),
                "shared_pca50_kb_cluster": row.get("shared_kb_cluster"),
                "shared_pca50_kb_label": row.get("shared_kb_cluster_label"),
                "joint_pca_nearest_centroid_cluster": row.get("joint_pca_nearest_centroid_cluster"),
                "joint_pca_nearest_centroid_label": row.get("joint_pca_nearest_centroid_label"),
                "joint_pca_nearest_centroid_distance": row.get("joint_pca_nearest_centroid_distance"),
                "joint_pca_nearest_point_cluster": row.get("joint_pca_nearest_point_cluster"),
                "joint_pca_nearest_point_label": row.get("joint_pca_nearest_point_label"),
                "joint_pca_nearest_point_distance": row.get("joint_pca_nearest_point_distance"),
                "embedding_top25_document_kb_cluster": row.get("nearest_kb_cluster"),
                "embedding_top25_document_kb_label": row.get("nearest_kb_cluster_label"),
                "embedding_top25_document_cluster_counts": row.get("top25_document_cluster_counts"),
                "embedding_top25_document_count": row.get("top25_document_count"),
                "embedding_top25_document_keys": row.get("top25_document_keys"),
                "embedding_top1_document_similarity": row.get("top1_kb_similarity"),
                "embedding_mean_top25_document_similarity": row.get("mean_top25_kb_document_similarity"),
            }
            for row in joint_pca.get("alignment_rows", [])
        ],
    )
    _write_csv(paths.tables / "joint_pca_points.csv", joint_pca.get("rows", []))
    for deprecated in (
        "joint_umap_points.csv",
        "joint_umap_hdbscan_clusters.csv",
        "joint_umap_hypothesis_assignments.csv",
    ):
        (paths.tables / deprecated).unlink(missing_ok=True)
    _write_csv(paths.tables / "elo_snapshots.csv", elo_summary.get("snapshots", []))
    _write_csv(paths.tables / "elo_volatility.csv", elo_summary.get("volatility_rows", []))
    _write_csv(paths.tables / "elo_bootstrap_tier_ci.csv", elo_summary.get("bootstrap_tier_rows", []))
    _write_csv(paths.tables / "elo_rank_stability.csv", elo_summary.get("rank_stability_rows", []))
    _write_csv(paths.tables / "elo_calibration.csv", elo_summary.get("calibration_rows", []))
    (paths.tables / "elo_distribution_snapshots.csv").unlink(missing_ok=True)
    _write_csv(
        paths.tables / "matches.csv",
        [
            {
                "match_index": i,
                "id": m.get("id"),
                "created_at": m.get("created_at"),
                "hyp_a": m.get("hyp_a"),
                "hyp_b": m.get("hyp_b"),
                "winner": m.get("winner"),
                "mode": m.get("mode"),
                "similarity": m.get("similarity"),
                "prompt1_hyp_id": m.get("prompt1_hyp_id"),
                "prompt2_hyp_id": m.get("prompt2_hyp_id"),
                "prompt1_side": m.get("prompt1_side"),
                "prompt2_side": m.get("prompt2_side"),
                "winner_prompt_position": m.get("winner_prompt_position"),
                "prompt1_chars": m.get("prompt1_chars"),
                "prompt2_chars": m.get("prompt2_chars"),
                "prompt_order_key": m.get("prompt_order_key"),
                "elo_a_before": m.get("elo_a_before"),
                "elo_b_before": m.get("elo_b_before"),
                "elo_a_after": m.get("elo_a_after"),
                "elo_b_after": m.get("elo_b_after"),
                "pre_match_elo_gap": abs(
                    float(m.get("elo_a_before") or 0.0) - float(m.get("elo_b_before") or 0.0)
                ),
            }
            for i, m in enumerate(matches, 1)
        ],
    )
    _write_csv(
        paths.tables / "source_inventory.csv",
        [
            {
                "source_key": key,
                "title": r.get("title"),
                "url": r.get("url") or r.get("pdf_url") or r.get("abs_url"),
                "rag_indexed": r.get("rag_indexed"),
                "seen_in_search": r.get("seen_in_search"),
                "paper_artifact": r.get("paper_artifact"),
                "search_tools": ",".join(r.get("search_tools", [])),
                "hypothesis_mentions": source_use.get("source_hits", {}).get(key, 0),
            }
            for key, r in sorted(sources.get("records", {}).items())
        ],
    )
    _write_csv(
        paths.tables / "transcript_agent_usage.csv",
        [
            {"agent": agent, **summary}
            for agent, summary in sorted(transcript_summary.get("by_agent", {}).items())
        ],
    )


def _write_figures(
    paths: _Paths,
    *,
    hypotheses: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    debate_summary: dict[str, Any],
    elo_summary: dict[str, Any],
    source_use: dict[str, Any],
    transcript_summary: dict[str, Any],
    hyp_vector_analysis: dict[str, Any],
    kb_analysis: dict[str, Any],
    joint_pca: dict[str, Any],
) -> None:
    rank_similarity_rows = hyp_vector_analysis.get("rank_similarity_rows", [])
    rank_values = [int(row.get("rank") or 0) for row in rank_similarity_rows if row.get("rank")]
    rank_max = max(rank_values, default=1)
    rank_ticks = sorted(
        {
            1,
            max(1, rank_max // 4),
            max(1, rank_max // 2),
            max(1, (3 * rank_max) // 4),
            rank_max,
        }
    )

    for deprecated in ("elo_trajectory.svg", "top5_stability.svg", "elo_distribution.svg"):
        (paths.figures / deprecated).unlink(missing_ok=True)

    volatility_rows = elo_summary.get("volatility_rows", [])
    _write_line_svg(
        paths.figures / "elo_volatility.svg",
        {
            "rolling std Delta Elo": [
                (row.get("match_index"), row.get("rating_change_std"))
                for row in volatility_rows
            ],
            "mean abs Delta Elo": [
                (row.get("match_index"), row.get("mean_abs_rating_change"))
                for row in volatility_rows
            ],
        },
        title="Rolling Elo Update Volatility",
        xlabel="Match index",
        ylabel="Elo change per match",
        y_min=0.0,
    )
    _write_elo_bootstrap_ci_svg(
        paths.figures / "elo_bootstrap_tier_ci.svg",
        elo_summary.get("bootstrap_tier_rows", []),
        title="Bootstrap Confidence Bands by Final Elo Tier",
    )
    _write_line_svg(
        paths.figures / "elo_rank_stability.svg",
        {
            "Kendall tau vs final": [
                (row.get("match_index"), row.get("kendall_tau_final"))
                for row in elo_summary.get("rank_stability_rows", [])
            ],
            "Spearman vs final": [
                (row.get("match_index"), row.get("spearman_final"))
                for row in elo_summary.get("rank_stability_rows", [])
            ],
            "top-10 Jaccard": [
                (row.get("match_index"), row.get("top10_jaccard_final"))
                for row in elo_summary.get("rank_stability_rows", [])
            ],
        },
        title="Dynamic Rank-Order Stability Against Final Ranking",
        xlabel="Match index",
        ylabel="agreement with final ranking",
        y_min=-1.0,
        y_max=1.0,
    )
    _write_elo_calibration_svg(
        paths.figures / "elo_calibration.svg",
        elo_summary.get("calibration_rows", []),
        title="Elo Win-Rate Calibration After Convergence",
        logistic_scale=float(
            elo_summary.get("metrics", {}).get("calibration_elo_logistic_scale") or 400.0
        ),
    )
    _write_hist_svg(
        paths.figures / "hypothesis_similarity_hist.svg",
        hyp_vector_analysis.get("pairwise_similarities", []),
        title="Final Hypothesis Pairwise Cosine Similarity",
        xlabel="Cosine similarity",
    )
    _write_scatter_svg(
        paths.figures / "hypothesis_pca.svg",
        hyp_vector_analysis.get("pca_rows", []),
        title="PCA of Final Hypothesis Embeddings",
        xlabel="PC1",
        ylabel="PC2",
        color_key="elo",
        label_key="hypothesis_id",
    )
    for deprecated in ("hypothesis_pca_clusters.svg", "hypothesis_umap_hdbscan.svg"):
        (paths.figures / deprecated).unlink(missing_ok=True)
    _write_scatter_svg(
        paths.figures / "hypothesis_umap.svg",
        hyp_vector_analysis.get("umap_rows", []),
        title="UMAP of Final Hypothesis Embeddings by Elo",
        xlabel="UMAP-1",
        ylabel="UMAP-2",
        color_key="elo",
        label_key="hypothesis_id",
    )
    _write_scatter_svg(
        paths.figures / "hypothesis_novelty_vs_rank.svg",
        [
            {
                "x": row.get("rank"),
                "y": row.get("embedding_novelty"),
                "elo": row.get("elo"),
                "hypothesis_id": row.get("hypothesis_id"),
            }
            for row in rank_similarity_rows
        ],
        title="Embedding Novelty by Final Elo Rank",
        xlabel="Final Elo rank",
        ylabel="1 - nearest-neighbor similarity",
        color_key="elo",
        label_key="hypothesis_id",
        x_min=1.0,
        x_max=float(rank_max),
        x_tick_values=[float(value) for value in rank_ticks],
    )
    _write_scatter_svg(
        paths.figures / "hypothesis_similarity_to_top10_vs_rank.svg",
        [
            {
                "x": row.get("rank"),
                "y": row.get("mean_similarity_to_top10"),
                "elo": row.get("elo"),
                "hypothesis_id": row.get("hypothesis_id"),
            }
            for row in rank_similarity_rows
        ],
        title="Similarity to Final Top 10 by Elo Rank",
        xlabel="Final Elo rank",
        ylabel="Mean cosine similarity to top 10",
        color_key="elo",
        label_key="hypothesis_id",
        x_min=1.0,
        x_max=float(rank_max),
        x_tick_values=[float(value) for value in rank_ticks],
    )
    _write_hist_svg(
        paths.figures / "debate_expected_probability.svg",
        debate_summary.get("winner_expected_probs", []),
        title="Debate Outcomes by Pre-Match Elo Expectation",
        xlabel="Expected probability of observed winner",
    )
    debate_metrics = debate_summary.get("metrics", {})
    _write_bar_svg(
        paths.figures / "debate_position_bias.svg",
        [
            ("side A wins", debate_metrics.get("storage_side_a_win_rate")),
            ("prompt 1 wins", debate_metrics.get("prompt_position_1_win_rate")),
            ("prompt 1 expected", debate_metrics.get("prompt_position_1_expected_probability_mean")),
        ],
        title="Tournament Position Bias Checks",
        ylabel="rate",
    )
    _write_bar_svg(
        paths.figures / "debate_verbosity_bias.svg",
        [
            ("longer input wins", debate_metrics.get("longer_input_win_rate")),
            ("more discussed wins", debate_metrics.get("more_discussed_win_rate")),
            ("prompt 1 longer", debate_metrics.get("prompt_position_1_longer_input_rate")),
            ("prompt 1 discussed", debate_metrics.get("prompt_position_1_more_discussed_rate")),
        ],
        title="Tournament Verbosity Bias Checks",
        ylabel="rate",
    )
    _write_bar_svg(
        paths.figures / "agent_token_usage.svg",
        [
            (agent, vals.get("input_tokens", 0) + vals.get("output_tokens", 0))
            for agent, vals in sorted(transcript_summary.get("by_agent", {}).items())
        ],
        title="Token Use by Agent",
        ylabel="Tokens",
    )
    _write_bar_svg(
        paths.figures / "source_use.svg",
        [
            ("known sources", source_use["metrics"].get("known_sources_total", 0)),
            ("RAG indexed", source_use["metrics"].get("known_sources_indexed_in_rag", 0)),
            ("explicit tokens", source_use["metrics"].get("unique_explicit_source_tokens_in_hypotheses", 0)),
            ("matched in hyps", source_use["metrics"].get("known_sources_explicitly_matched_in_hypotheses", 0)),
        ],
        title="Source Inventory and Explicit Hypothesis Use",
        ylabel="Count",
    )
    _write_scatter_svg(
        paths.figures / "kb_pca.svg",
        kb_analysis.get("pca_rows", []),
        title="PCA of Sampled RAG KB Chunks",
        xlabel="PC1",
        ylabel="PC2",
        color_key="cluster",
        label_key="source",
    )
    kb_umap = (((kb_analysis.get("density") or {}).get("methods") or {}).get("umap") or {})
    _write_scatter_svg(
        paths.figures / "kb_umap_hdbscan.svg",
        kb_umap.get("rows", []),
        title="UMAP + HDBSCAN of Sampled RAG KB Chunks",
        xlabel="UMAP-1",
        ylabel="UMAP-2",
        color_key="density_cluster",
        label_key="source",
    )
    _write_joint_pca_svg(
        paths.figures / "kb_hypothesis_joint_pca.svg",
        joint_pca.get("rows", []),
        title="Hypotheses in the Shared RAG KB PCA Frame",
    )
    (paths.figures / "kb_hypothesis_joint_umap_hdbscan.svg").unlink(missing_ok=True)


def _render_report(
    *,
    session: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    feedback: list[dict[str, Any]],
    transcript_summary: dict[str, Any],
    review_summary: dict[str, Any],
    lineage_summary: dict[str, Any],
    debate_summary: dict[str, Any],
    elo_summary: dict[str, Any],
    source_use: dict[str, Any],
    sources: dict[str, Any],
    tool_usage: dict[str, Any],
    hyp_vector_analysis: dict[str, Any],
    kb_analysis: dict[str, Any],
    joint_pca: dict[str, Any],
    metrics: dict[str, Any],
    paths: _Paths,
) -> str:
    top = sorted(hypotheses, key=lambda h: float(h.get("elo") or 0.0), reverse=True)[:10]
    start = _parse_dt(session.get("created_at"))
    end = _parse_dt(session.get("updated_at"))
    duration = _human_duration((end - start).total_seconds()) if start and end else "unknown"
    src = source_use["metrics"]
    hvec = hyp_vector_analysis.get("metrics", {})
    kb = kb_analysis.get("metrics", {})
    debate = debate_summary.get("metrics", {})
    elo = elo_summary.get("metrics", {})
    kb_clusters = kb_analysis.get("cluster_rows", [])
    kb_density_umap = (((kb_analysis.get("density") or {}).get("methods") or {}).get("umap") or {})
    hyp_rank_buckets = hyp_vector_analysis.get("rank_bucket_rows", [])
    joint_metrics = joint_pca.get("metrics", {})
    joint_alignment_rows = joint_pca.get("alignment_rows", [])
    kb_cluster_by_id = {int(row["cluster"]): row for row in kb_clusters if row.get("cluster") is not None}
    tool_mentions = tool_usage.get("tool_name_mentions", {})
    total_tokens = transcript_summary["total"]["input_tokens"] + transcript_summary["total"]["output_tokens"]

    lines = [
        f"# Session Analysis: {session.get('id')}",
        "",
        "## Abstract",
        "",
        (
            f"This report analyzes a hypothesis-engine session for the goal: "
            f"**{_md_escape(str(session.get('research_goal') or ''))}**. "
            f"The run produced {len(hypotheses)} hypotheses, {len(matches)} tournament matches, "
            f"{review_summary.get('count', 0)} reviews, and {len(feedback)} metareview/system-feedback records "
            f"over {duration}. Stored transcript accounting reports {total_tokens:,} tokens "
            f"and ${float(session.get('budget_used_usd') or 0.0):.2f} of configured budget use."
        ),
        "",
        "## Methods",
        "",
        (
            "The analysis reads the session SQLite tables, cached search artifacts, paper text artifacts, "
            "the hypothesis FAISS index, and the RAG knowledge-base index. It does not perform web requests, "
            "PDF downloads, or embeddings; the only optional model call is a compact local-LLM labeling pass "
            "over representative KB cluster snippets."
        ),
        "",
        (
            "Source-use attribution is conservative: a source is counted as meaningfully used only when "
            "the final hypothesis text contains an explicit URL, DOI, arXiv/ChemRxiv identifier, or exact "
            "title match against known search/RAG records. This likely undercounts paraphrased influence."
        ),
        "",
        (
            "The RAG KB uses one persisted all-chunk projection shared with the live UI: normalized source "
            "embeddings are reduced to 50 PCA components without whitening, normalized again, and clustered "
            "there. The first two components are used only for visualization, and hypothesis overlays are "
            "transformed through the same KB-fitted PCA model. KB cluster labels and summaries, including the "
            "separate UMAP/HDBSCAN density view, are generated from representative chunks using the configured "
            "local LLM when available, with deterministic top-term labels as a fallback."
        ),
        "",
        "## Generation Trajectory",
        "",
        (
            f"The hypothesis pool contains {len(hypotheses)} entries: "
            f"{_counter_sentence(lineage_summary.get('created_by_counts', {}))}. "
            f"{lineage_summary.get('root_hypotheses', 0)} were root hypotheses and "
            f"{lineage_summary.get('hypotheses_with_parents', 0)} had recorded parents."
        ),
        "",
        "![Token use by agent](figures/agent_token_usage.svg)",
        "",
        (
            "Interpretation: token allocation is a proxy for where the workflow spent cognitive effort. "
            "Heavy ranking usage indicates that many pairwise comparisons were debated; heavy reflection "
            "or evolution usage indicates deeper review or candidate refinement. This figure should be read "
            "alongside source-use and tournament plots because high token use is not automatically high scientific yield."
        ),
        "",
        _markdown_table(
            ["agent", "calls", "input tokens", "output tokens", "cost"],
            [
                [
                    agent,
                    str(vals.get("count", 0)),
                    f"{int(vals.get('input_tokens', 0)):,}",
                    f"{int(vals.get('output_tokens', 0)):,}",
                    f"${float(vals.get('cost_usd', 0.0)):.2f}",
                ]
                for agent, vals in sorted(transcript_summary.get("by_agent", {}).items())
            ],
        ),
        "",
        "## Source and RAG Utilization",
        "",
        (
            f"Search artifacts contain {sum(sources.get('search_counts', {}).values())} cached searches "
            f"across {_counter_sentence(sources.get('search_counts', {}))}. The RAG manifest contains "
            f"{sources.get('rag_manifest_count', 0)} unique paper records, of which "
            f"{sources.get('rag_indexed_count', 0)} are marked indexed. The stored KB contains "
            f"{kb.get('chunks', 0):,} chunks; the shared PCA and KMeans models were fitted on all chunks, "
            f"and this report retained {kb.get('sample_chunks', 0):,} chunks for rendered and sampled "
            f"diversity diagnostics."
        ),
        "",
        (
            f"Final hypotheses contain {src.get('unique_explicit_source_tokens_in_hypotheses', 0)} unique "
            f"explicit source tokens. These matched {src.get('known_sources_explicitly_matched_in_hypotheses', 0)} "
            f"known search/RAG records across {src.get('hypotheses_with_known_source_matches', 0)} hypotheses."
        ),
        "",
        "![Source inventory and explicit hypothesis use](figures/source_use.svg)",
        "",
        (
            "Interpretation: the gap between indexed sources and explicitly cited sources estimates how much "
            "of the KB remained latent background context rather than visible citation support. A large gap can be "
            "healthy if RAG was used for broad grounding, but it can also indicate that hypotheses converged on a "
            "small subset of repeatedly useful papers."
        ),
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["known source records", src.get("known_sources_total", 0)],
                ["RAG-indexed source records", src.get("known_sources_indexed_in_rag", 0)],
                ["paper text artifacts", src.get("paper_text_artifacts", 0)],
                ["hypotheses with explicit source tokens", src.get("hypotheses_with_explicit_source_tokens", 0)],
                ["hypotheses with matched known sources", src.get("hypotheses_with_known_source_matches", 0)],
                ["RAG retrieval mentions in transcripts", tool_mentions.get("rag_retrieve_context", 0) + tool_mentions.get("retrieve_context", 0)],
                ["arXiv search mentions in transcripts", tool_mentions.get("arxiv_search", 0)],
                ["bioRxiv search mentions in transcripts", tool_mentions.get("biorxiv_search", 0)],
                ["ChemRxiv search mentions in transcripts", tool_mentions.get("chemrxiv_search", 0)],
            ],
        ),
        "",
        "## Tournament and Ranking Dynamics",
        "",
        (
            f"The tournament recorded {len(matches)} matches. The final top Elo was "
            f"{_fmt_float(elo.get('final_top_elo'), 1)}, and the final top-10 mean Elo was "
            f"{_fmt_float(elo.get('final_top10_mean_elo'), 1)}. Per-hypothesis match counts ranged "
            f"from {elo.get('match_count_min', 0)} to {elo.get('match_count_max', 0)} "
            f"(mean {_fmt_float(elo.get('match_count_mean'), 1)}). Elo is analyzed here as a convergence "
            "and calibration signal: the rated objects are hypotheses, not separate LLM models."
        ),
        "",
        "![Rolling Elo update volatility](figures/elo_volatility.svg)",
        "",
        (
            "Volatility interpretation: the rolling standard deviation of per-match rating changes measures whether "
            "the judge/tournament is still making large updates. A healthy tournament should generally decay toward "
            "smaller updates as the ranking approaches equilibrium. Persistent high volatility would suggest noisy "
            "pairwise judgments, non-transitive comparisons, or hypotheses that are too similar for the judge to separate. "
            f"This run used a {elo.get('volatility_window_matches')} match rolling window; volatility moved from "
            f"{_fmt_float(elo.get('volatility_initial_std'), 2)} to {_fmt_float(elo.get('volatility_final_std'), 2)} "
            f"Elo points, a final/initial ratio of {_fmt_float(elo.get('volatility_decay_ratio'), 3)}."
        ),
        "",
        "![Bootstrap confidence bands by final Elo tier](figures/elo_bootstrap_tier_ci.svg)",
        "",
        (
            "Confidence-band interpretation: individual hypothesis bands would be unreadable for a 100-hypothesis pool, "
            "so the plot groups candidates by their final Elo tier and bootstraps the tier mean at each match milestone. "
            "Wide, overlapping bands mean the tournament has not cleanly separated tiers; narrow, separated bands mean "
            "the match density was sufficient to distinguish elite, middle, and lower-ranked hypothesis families."
        ),
        "",
        "![Dynamic rank-order stability](figures/elo_rank_stability.svg)",
        "",
        (
            "Rank-stability interpretation: each milestone leaderboard is compared with the final leaderboard using "
            "Kendall's tau, Spearman rank correlation, and top-10 Jaccard overlap. The practical threshold is the first "
            "match where Kendall's tau reaches at least 0.90 and stays there through the remaining snapshots. In this run "
            f"that threshold was {elo.get('rank_stability_tau_0_90_match') if elo.get('rank_stability_tau_0_90_match') is not None else 'not reached'}. "
            f"The final snapshot had Kendall tau {_fmt_float(elo.get('rank_stability_final_kendall_tau'), 3)}, "
            f"Spearman {_fmt_float(elo.get('rank_stability_final_spearman'), 3)}, and top-10 Jaccard "
            f"{_fmt_float(elo.get('rank_stability_final_top10_jaccard'), 3)} against the final ranking."
        ),
        "",
        "![Elo win-rate calibration](figures/elo_calibration.svg)",
        "",
        (
            "Calibration interpretation: after the detected convergence threshold, or the final-third fallback if no "
            "stable threshold is reached, matches are binned by pre-match Elo gap. The empirical favorite score is "
            "compared with the theoretical Elo logistic curve. A well-calibrated tournament "
            "should place empirical points near the curve; systematic deviations suggest the K-factor is mismatched, the "
            "judge is noisy, or the preference relation is non-transitive. Calibration used matches starting at "
            f"{elo.get('calibration_start_match')} and covered {elo.get('calibration_matches', 0)} matches across "
            f"{elo.get('calibration_bin_count', 0)} bins; the weighted mean absolute calibration error was "
            f"{_fmt_float(elo.get('calibration_mean_abs_error'), 3)}."
        ),
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["rolling volatility window", elo.get("volatility_window_matches")],
                ["initial volatility std", _fmt_float(elo.get("volatility_initial_std"), 2)],
                ["final volatility std", _fmt_float(elo.get("volatility_final_std"), 2)],
                ["final/initial volatility ratio", _fmt_float(elo.get("volatility_decay_ratio"), 3)],
                ["Kendall tau >= 0.90 stable match", elo.get("rank_stability_tau_0_90_match") or "not reached"],
                ["final Kendall tau", _fmt_float(elo.get("rank_stability_final_kendall_tau"), 3)],
                ["final Spearman", _fmt_float(elo.get("rank_stability_final_spearman"), 3)],
                ["calibration logistic scale", _fmt_float(elo.get("calibration_elo_logistic_scale"), 1)],
                ["calibration mean abs error", _fmt_float(elo.get("calibration_mean_abs_error"), 3)],
            ],
        ),
        "",
        _markdown_table(
            ["rank", "hypothesis", "Elo", "matches"],
            [
                [
                    str(i),
                    _truncate(str(h.get("title") or h.get("id")), 70),
                    _fmt_float(h.get("elo"), 1),
                    str(h.get("matches_played") or 0),
                ]
                for i, h in enumerate(top, 1)
            ],
        ),
        "",
        "## Debate Effectiveness",
        "",
        (
            f"Ranking mode counts were {_counter_sentence(debate.get('mode_counts', {}))}. "
            f"The upset rate was {_fmt_percent(debate.get('upset_rate'))}, where an upset means "
            f"the lower pre-match Elo hypothesis won. The average pre-match Elo gap was "
            f"{_fmt_float(debate.get('mean_pre_match_elo_gap'), 1)}, and the observed winner's "
            f"mean expected probability was {_fmt_percent(debate.get('mean_winner_expected_probability'))}."
        ),
        "",
        "![Debate winner expected probability](figures/debate_expected_probability.svg)",
        "",
        (
            "Interpretation: values near 0.5 mean debates often overturned or barely favored the pre-match Elo "
            "expectation; values near 1.0 mean debates mostly confirmed the favorite. A meaningful upset rate is "
            "useful because it indicates the debate agent was not simply rubber-stamping current rank."
        ),
        "",
        "![Tournament position bias checks](figures/debate_position_bias.svg)",
        "",
        "![Tournament verbosity bias checks](figures/debate_verbosity_bias.svg)",
        "",
        (
            "Bias interpretation: side A/B checks test storage-order bias, while prompt-position checks test whether "
            "the first-presented hypothesis (`Hypothesis 1`) or second-presented hypothesis (`Hypothesis 2`) wins more "
            "often than expected. In the current ranking implementation, prompt position 1 is the lower-ID anchor, so "
            "this diagnostic can detect a position/order issue but cannot fully separate it from ID-order effects unless "
            "future tournaments randomize presentation order."
        ),
        "",
        (
            f"Prompt position 1 won {_fmt_percent(debate.get('prompt_position_1_win_rate'))} of decided matches "
            f"against a mean Elo-expected probability of {_fmt_percent(debate.get('prompt_position_1_expected_probability_mean'))}; "
            f"its mean expected residual was {_fmt_float(debate.get('prompt_position_1_expected_residual_mean'), 3)}. "
            f"Storage side A won {_fmt_percent(debate.get('storage_side_a_win_rate'))}, with residual "
            f"{_fmt_float(debate.get('storage_side_a_expected_residual_mean'), 3)}."
        ),
        "",
        (
            "Verbosity interpretation: input verbosity compares the character count of each hypothesis plus its selected "
            "review in the ranking prompt. Rationale attention is a rough output-side proxy that partitions text following "
            "H1/H2 or Hypothesis 1/2 labels in the stored debate rationale. These are diagnostics, not causal estimates: a "
            "better hypothesis may naturally receive more discussion."
        ),
        "",
        (
            f"The longer prompt input won {_fmt_percent(debate.get('longer_input_win_rate'))} of matches where one side "
            f"was longer. The side receiving more labeled rationale attention won "
            f"{_fmt_percent(debate.get('more_discussed_win_rate'))}. Input verbosity delta vs prompt-1 win correlation was "
            f"{_fmt_float(debate.get('input_verbosity_delta_vs_prompt1_win_pearson'), 3)}, and rationale-attention delta "
            f"vs prompt-1 win correlation was {_fmt_float(debate.get('rationale_attention_delta_vs_prompt1_win_pearson'), 3)}."
        ),
        "",
        _markdown_table(
            ["bias metric", "value", "interpretation"],
            [
                [
                    row.get("label"),
                    _fmt_float(row.get("value"), 3) if _is_finite(row.get("value")) else row.get("value"),
                    row.get("interpretation"),
                ]
                for row in debate_summary.get("bias_summary_rows", [])
            ],
        ),
        "",
        "## Hypothesis Similarity and Novelty",
        "",
        (
            f"Hypothesis vector coverage was {_fmt_percent(hvec.get('embedding_coverage'))}; "
            f"{hvec.get('missing_vector_count', 0)} hypotheses were missing from the stored FAISS metadata. "
            f"Among embedded hypotheses, mean pairwise cosine similarity was "
            f"{_fmt_float(hvec.get('pairwise_similarity_mean'), 3)} with p95 "
            f"{_fmt_float(hvec.get('pairwise_similarity_p95'), 3)}. "
            f"Near-duplicate counts were {hvec.get('near_duplicate_pairs_0_95', 0)} pairs at >=0.95 "
            f"and {hvec.get('near_duplicate_pairs_0_98', 0)} pairs at >=0.98."
        ),
        "",
        "![Hypothesis similarity histogram](figures/hypothesis_similarity_hist.svg)",
        "",
        (
            "Interpretation: the similarity histogram estimates redundancy in the final hypothesis pool. Novelty is "
            "assessed in two ways: the review agent's novelty score is a model judgment made during reflection, while "
            "the embedding novelty proxy used here is `1 - nearest-neighbor cosine similarity` within the final pool. "
            "The proxy is session-relative: it identifies hypotheses that are semantically isolated from other final "
            "hypotheses, not necessarily novel against the full literature. In this run, embedding novelty had median "
            f"{_fmt_float(hvec.get('embedding_novelty_median'), 3)} and p90 {_fmt_float(hvec.get('embedding_novelty_p90'), 3)}. "
            f"The final top 5 had mean internal similarity {_fmt_float(hvec.get('top5_pairwise_similarity_mean'), 3)} "
            f"(range {_fmt_float(hvec.get('top5_pairwise_similarity_min'), 3)}-{_fmt_float(hvec.get('top5_pairwise_similarity_max'), 3)}), "
            f"and the final top 10 had mean internal similarity {_fmt_float(hvec.get('top10_pairwise_similarity_mean'), 3)}. "
            f"The bottom 10 had mean internal similarity {_fmt_float(hvec.get('bottom10_pairwise_similarity_mean'), 3)}, "
            f"while top-10 to bottom-10 similarity averaged {_fmt_float(hvec.get('top10_to_bottom10_similarity_mean'), 3)}. "
            f"Elo-vs-novelty Spearman correlation was {_fmt_float(hvec.get('elo_vs_embedding_novelty_spearman'), 3)}; "
            f"Elo-vs-similarity-to-top-10 Spearman correlation was {_fmt_float(hvec.get('elo_vs_similarity_to_top10_spearman'), 3)}."
        ),
        "",
        "![Embedding novelty by final Elo rank](figures/hypothesis_novelty_vs_rank.svg)",
        "",
        "![Similarity to final top 10 by Elo rank](figures/hypothesis_similarity_to_top10_vs_rank.svg)",
        "",
        _markdown_table(
            ["rank bucket", "count", "mean Elo", "nearest sim", "novelty", "sim to top 10"],
            [
                [
                    row.get("rank_bucket"),
                    row.get("count"),
                    _fmt_float(row.get("mean_elo"), 1),
                    _fmt_float(row.get("mean_nearest_similarity"), 3),
                    _fmt_float(row.get("mean_embedding_novelty"), 3),
                    _fmt_float(row.get("mean_similarity_to_top10"), 3),
                ]
                for row in hyp_rank_buckets
            ],
        ),
        "",
        "![Hypothesis PCA](figures/hypothesis_pca.svg)",
        "",
        (
            "Interpretation: this PCA places final hypotheses in a linear two-dimensional projection of embedding space "
            "and colors points by final Elo. It is useful as a stable global overview, but it can compress local semantic "
            "neighborhoods."
        ),
        "",
        "![Hypothesis UMAP](figures/hypothesis_umap.svg)",
        "",
        (
            "Interpretation: this UMAP places final hypotheses in a nonlinear local-neighborhood projection and colors "
            "points by final Elo. It replaces the earlier PCA-cluster and final-hypothesis HDBSCAN views: the intent here "
            "is to inspect whether high-ranking hypotheses localize in a semantic neighborhood without imposing cluster "
            "classes on the final hypothesis set."
        ),
        "",
        "## Knowledge-Base Diversity",
        "",
        (
            f"The sampled KB had {kb.get('unique_documents_in_sample', 0)} unique source documents "
            f"and document entropy {_fmt_float(kb.get('sample_document_entropy'), 2)}. "
            f"Stability-aware KMeans in the normalized PCA-{kb.get('projection_pca_components', 0)} space "
            f"selected {kb.get('projection_selected_k', kb.get('cluster_count', 0))} clusters with sampled "
            f"silhouette {_fmt_float(kb.get('projection_silhouette'), 3)} and stability "
            f"{_fmt_float(kb.get('projection_stability'), 3)}. The independent full-embedding cosine "
            f"silhouette diagnostic was {_fmt_float(kb.get('silhouette_cosine'), 3)}."
        ),
        "",
        "![KB PCA](figures/kb_pca.svg)",
        "",
        (
            "Interpretation: KB clusters are semantic neighborhoods in the embedded literature sample. They may "
            "correspond to topical regimes such as ion irradiation damage, Janus/TMD synthesis, substrate effects, "
            "characterization methods, or less relevant background materials. Mixed clusters can occur when papers "
            "share vocabulary despite different experimental contexts."
        ),
        "",
        _markdown_table(
            ["KB cluster", "chunks", "documents", "label", "summary", "relevance"],
            [
                [
                    row.get("cluster"),
                    row.get("chunks"),
                    row.get("unique_documents"),
                    row.get("llm_label") or row.get("deterministic_label"),
                    row.get("llm_summary") or row.get("top_terms"),
                    row.get("llm_relevance") or "",
                ]
                for row in kb_clusters
            ],
        ),
        "",
        "![KB UMAP HDBSCAN](figures/kb_umap_hdbscan.svg)",
        "",
        (
            "UMAP/HDBSCAN interpretation: this view is intended to expose local literature neighborhoods that may be "
            "washed out by linear PCA. HDBSCAN noise points are chunks that do not sit in a sufficiently dense sampled "
            "neighborhood; they can be off-topic material, rare but relevant mechanisms, or simply boundary chunks between "
            "topics. The `-1` row is intentionally reported as noise rather than labeled as one semantic topic. "
            f"The UMAP pass status was `{kb_density_umap.get('status', 'missing')}`; it found "
            f"{kb.get('umap_cluster_count', 0)} dense clusters with "
            f"{_fmt_percent(kb.get('umap_noise_fraction'))} of sampled chunks marked noise."
        ),
        "",
        _markdown_table(
            ["HDBSCAN cluster", "noise", "chunks", "documents", "label", "summary", "relevance", "top documents"],
            [
                [
                    row.get("density_cluster"),
                    row.get("is_noise"),
                    row.get("size"),
                    row.get("unique_documents"),
                    row.get("llm_label") or row.get("deterministic_label") or _label_from_top_terms(row.get("top_terms")),
                    _truncate(str(row.get("llm_summary") or ""), 220),
                    _truncate(str(row.get("llm_relevance") or ""), 180),
                    _truncate(str(row.get("top_documents") or ""), 160),
                ]
                for row in kb_density_umap.get("cluster_rows", [])
            ],
        ),
        "",
        (
            "The shared-frame PCA below uses the KB-fitted projection unchanged and transforms hypothesis "
            "embeddings into it. KB chunks are colored by the persisted KB clusters above; hypotheses are colored "
            "by Elo with an explicit colorbar. Hypothesis markers are semi-transparent and drawn from lower to "
            "higher Elo so higher-scoring hypotheses remain visible. The table's primary assignment is the same "
            "normalized PCA-50 KMeans prediction used for hypothesis overlays in the UI. Nearest cluster centroid "
            "in the fixed 2D KB frame and high-dimensional top-25-document neighbors are retained as separate "
            "diagnostics because both can legitimately disagree with the PCA-50 cluster model."
        ),
        "",
        "![Hypotheses in the shared RAG KB PCA frame](figures/kb_hypothesis_joint_pca.svg)",
        "",
        (
            f"Shared PCA-50 cluster assignments across all hypotheses were "
            f"{_cluster_count_sentence(joint_metrics.get('shared_kb_cluster_counts'), kb_cluster_by_id)}. "
            f"For the final top 10, shared PCA-50 assignments were "
            f"{_cluster_count_sentence(joint_metrics.get('top10_shared_kb_cluster_counts'), kb_cluster_by_id)}. "
            f"Across all hypotheses, 2D nearest-centroid counts were "
            f"{_cluster_count_sentence(joint_metrics.get('joint_pca_centroid_cluster_counts'), kb_cluster_by_id)}. "
            f"For the final top 10, 2D joint-PCA nearest-centroid counts were "
            f"{_cluster_count_sentence(joint_metrics.get('top10_joint_pca_centroid_cluster_counts'), kb_cluster_by_id)}. "
            f"The high-dimensional embedding-space top-25-document assignment gives final top-10 counts of "
            f"{_cluster_count_sentence(joint_metrics.get('top10_nearest_kb_document_cluster_counts'), kb_cluster_by_id)}. "
            f"Embedding-space mean top-1 hypothesis-to-KB similarity was "
            f"{_fmt_float(joint_metrics.get('mean_top1_kb_similarity'), 3)}, and mean top-25-document similarity was "
            f"{_fmt_float(joint_metrics.get('mean_top25_kb_document_similarity'), 3)}."
        ),
        "",
        _markdown_table(
            [
                "rank",
                "hypothesis",
                "Elo",
                "shared PCA-50 cluster",
                "shared label",
                "2D nearest cluster",
                "embedding top-25 docs cluster",
                "embedding label",
                "top25 doc sim",
            ],
            [
                [
                    row.get("rank"),
                    _truncate(str(row.get("hypothesis_id") or ""), 22),
                    _fmt_float(row.get("elo"), 1),
                    row.get("shared_kb_cluster"),
                    row.get("shared_kb_cluster_label"),
                    row.get("joint_pca_nearest_centroid_cluster"),
                    row.get("nearest_kb_cluster"),
                    row.get("nearest_kb_cluster_label"),
                    _fmt_float(row.get("mean_top25_kb_document_similarity"), 3),
                ]
                for row in joint_alignment_rows[:15]
            ],
        ),
        "",
        "## Limitations",
        "",
        "- Explicit source-use counts undercount paraphrased or latent use of retrieved context.",
        "- PCA is a lossy 2D projection of high-dimensional embeddings; local distances are more meaningful than global geometry.",
        "- Debate effectiveness is measured from internal Elo dynamics and rationales, not from external scientific ground truth.",
        "- Missing hypothesis vectors reduce similarity/PCA coverage; see `metrics.json` for exact missing IDs.",
        "",
        "## Artifacts",
        "",
        "- `report.html`: self-contained HTML report with figures embedded for remote viewing/download.",
        "- `analysis_report.zip`: complete downloadable bundle containing report, figures, tables, and metrics.",
        "- `metrics.json`: machine-readable metrics.",
        "- `tables/top_hypotheses.csv`: final ranking with review, source, and similarity columns.",
        "- `tables/debate_bias_by_match.csv`: per-match position, prompt-order, input-verbosity, and rationale-attention diagnostics.",
        "- `tables/debate_bias_summary.csv`: aggregate position and verbosity bias metrics.",
        "- `tables/source_inventory.csv`: discovered/RAG-indexed source records and explicit hypothesis mentions.",
        "- `tables/elo_snapshots.csv`: top-rank trajectory snapshots.",
        "- `tables/elo_volatility.csv`: rolling standard deviation and mean absolute Elo update size.",
        "- `tables/elo_bootstrap_tier_ci.csv`: bootstrap confidence intervals for final-rank tier mean Elo.",
        "- `tables/elo_rank_stability.csv`: Kendall/Spearman/Jaccard agreement against the final ranking by milestone.",
        "- `tables/elo_calibration.csv`: empirical favorite score vs theoretical Elo expectation by Elo-gap bin.",
        "- `tables/kb_clusters.csv`: KB cluster labels, summaries, and representative text snippets.",
        "- `tables/kb_pca.csv`: sampled KB chunks in the persisted PCA frame shared with the live UI.",
        "- `tables/kb_umap_hdbscan.csv`: UMAP coordinates and HDBSCAN labels for sampled KB chunks.",
        "- `tables/kb_umap_hdbscan_clusters.csv`: density-cluster diagnostics and LLM summaries for sampled KB chunks.",
        "- `tables/hypothesis_umap.csv`: UMAP coordinates for final hypotheses colored by Elo in the report.",
        "- `tables/hypothesis_rank_similarity.csv`: per-hypothesis rank, novelty proxy, and similarity-to-leaders metrics.",
        "- `tables/hypothesis_similarity_rank_buckets.csv`: rank-bucket averages for novelty and similarity trends.",
        "- `tables/joint_pca_alignment.csv`: full hypothesis-to-KB alignment diagnostics.",
        "- `tables/joint_pca_visual_alignment.csv`: joint-PCA visual cluster assignment first, embedding top-25-document diagnostic second.",
        "- `figures/*.svg`: reusable static figures.",
    ]
    return "\n".join(str(line) for line in lines) + "\n"



def _write_html_report(report_md: str, paths: _Paths, *, session_id: str) -> Path:
    try:
        from markdown import markdown as markdown_to_html

        body = markdown_to_html(report_md, extensions=["tables", "fenced_code"])
    except Exception:
        body = f"<pre>{escape(report_md)}</pre>"

    body = _embed_local_svgs(body, paths)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hypothesis Engine Analysis {escape(session_id)}</title>
  <style>
    :root {{ color-scheme: light; --ink: #0f172a; --muted: #475569; --line: #cbd5e1; --bg: #ffffff; --soft: #f8fafc; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ font-size: 2rem; line-height: 1.15; margin: 0 0 1rem; }}
    h2 {{ border-top: 1px solid var(--line); padding-top: 1.5rem; margin-top: 2rem; }}
    h3 {{ margin-top: 1.5rem; }}
    p, li {{ color: var(--ink); }}
    a {{ color: #2563eb; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 1.4rem; font-size: 0.92rem; }}
    th, td {{ border: 1px solid var(--line); padding: 0.45rem 0.55rem; vertical-align: top; }}
    th {{ background: var(--soft); text-align: left; }}
    code {{ background: var(--soft); padding: 0.1rem 0.25rem; border-radius: 4px; }}
    pre {{ overflow-x: auto; background: var(--soft); padding: 1rem; border: 1px solid var(--line); }}
    figure {{ margin: 1.25rem 0 1.75rem; padding: 0; }}
    figure svg {{ width: 100%; height: auto; border: 1px solid var(--line); background: #fff; }}
    figcaption {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.4rem; }}
    .download-note {{ background: #eff6ff; border: 1px solid #bfdbfe; padding: 0.8rem 1rem; border-radius: 6px; color: #1e3a8a; }}
    @media print {{ main {{ max-width: none; }} figure svg {{ page-break-inside: avoid; }} }}
  </style>
</head>
<body>
<main>
<p class="download-note">This is a self-contained HTML report. Figures are embedded inline; CSV tables and metrics are included in the accompanying analysis_report.zip bundle.</p>
{body}
</main>
</body>
</html>
"""
    path = paths.output / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def _embed_local_svgs(html: str, paths: _Paths) -> str:
    img_re = re.compile(
        r'(?:<p>)?<img alt="([^"]*)" src="(figures/[^"]+\.svg)"\s*/?>(?:</p>)?'
    )

    def repl(match: re.Match[str]) -> str:
        alt = match.group(1)
        rel = match.group(2)
        svg_path = (paths.output / rel).resolve()
        try:
            svg_path.relative_to(paths.output.resolve())
        except ValueError:
            return match.group(0)
        if not svg_path.is_file():
            return match.group(0)
        svg = svg_path.read_text(encoding="utf-8")
        return f"<figure>{svg}<figcaption>{escape(alt)}</figcaption></figure>"

    return img_re.sub(repl, html)


def _write_bundle(paths: _Paths, *, session_id: str) -> Path:
    bundle = paths.output / "analysis_report.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(paths.output.rglob("*")):
            if not file.is_file() or file == bundle:
                continue
            arcname = Path(f"hypothesis_engine_analysis_{session_id}") / file.relative_to(paths.output)
            zf.write(file, arcname.as_posix())
    return bundle

def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _write_hist_svg(path: Path, values: list[float], *, title: str, xlabel: str) -> None:
    width, height = 900, 420
    margin = {"left": 70, "right": 30, "top": 52, "bottom": 58}
    values = [float(v) for v in values if _is_finite(v)]
    if not values:
        _write_empty_svg(path, title, "No data")
        return
    bins = min(35, max(8, int(math.sqrt(len(values)))))
    counts, edges = np.histogram(values, bins=bins)
    x0, x1 = float(edges[0]), float(edges[-1])
    y1 = max(int(np.max(counts)), 1)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    parts = [_svg_header(width, height, title)]
    parts.append(_axis_svg(width, height, margin, xlabel, "count"))
    for count, lo, hi in zip(counts, edges[:-1], edges[1:], strict=True):
        x = margin["left"] + ((float(lo) - x0) / (x1 - x0 or 1.0)) * plot_w
        w = max(1.0, ((float(hi) - float(lo)) / (x1 - x0 or 1.0)) * plot_w - 1)
        h = (float(count) / y1) * plot_h
        y = margin["top"] + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#3b82f6" opacity="0.78"/>')
    parts.append(_ticks_svg(values_min=x0, values_max=x1, y_min=0, y_max=y1, margin=margin, width=width, height=height))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_line_svg(
    path: Path,
    series: dict[str, list[tuple[int | float, int | float]]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    width, height = 900, 420
    margin = {"left": 78, "right": 150, "top": 52, "bottom": 58}
    clean: dict[str, list[tuple[float, float]]] = {}
    for name, points in series.items():
        vals = [(float(x), float(y)) for x, y in points if _is_finite(x) and _is_finite(y)]
        if vals:
            clean[name] = vals
    if not clean:
        _write_empty_svg(path, title, "No data")
        return
    xs = [x for points in clean.values() for x, _y in points]
    ys = [y for points in clean.values() for _x, y in points]
    x0, x1 = min(xs), max(xs)
    yy0 = min(ys) if y_min is None else y_min
    yy1 = max(ys) if y_max is None else y_max
    if yy0 == yy1:
        yy0 -= 1.0
        yy1 += 1.0
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    def sx(x: float) -> float:
        return margin["left"] + ((x - x0) / (x1 - x0 or 1.0)) * plot_w

    def sy(y: float) -> float:
        return margin["top"] + plot_h - ((y - yy0) / (yy1 - yy0 or 1.0)) * plot_h

    parts = [_svg_header(width, height, title), _axis_svg(width, height, margin, xlabel, ylabel)]
    colors = _palette()
    for i, (name, points) in enumerate(clean.items()):
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        color = colors[i % len(colors)]
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        legend_y = margin["top"] + 20 + i * 22
        parts.append(f'<line x1="{width - 130}" y1="{legend_y}" x2="{width - 106}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width - 98}" y="{legend_y + 4}" font-size="12" fill="#334155">{escape(name)}</text>')
    parts.append(_ticks_svg(values_min=x0, values_max=x1, y_min=yy0, y_max=yy1, margin=margin, width=width, height=height))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")



def _write_elo_bootstrap_ci_svg(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    width, height = 900, 460
    margin = {"left": 78, "right": 170, "top": 52, "bottom": 58}
    clean = [
        row for row in rows
        if _is_finite(row.get("match_index")) and _is_finite(row.get("mean_elo"))
    ]
    if not clean:
        _write_empty_svg(path, title, "No data")
        return

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        groups[str(row.get("tier") or "unknown")].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda item: float(item.get("match_index") or 0.0))

    xs = [float(row["match_index"]) for row in clean]
    ys = []
    for row in clean:
        for key in ("mean_elo", "ci95_low", "ci95_high"):
            value = _to_float(row.get(key))
            if value is not None:
                ys.append(value)
    x0, x1 = min(xs), max(xs)
    y0, y1 = _padded_range(ys)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    def sx(x: float) -> float:
        return margin["left"] + ((x - x0) / (x1 - x0 or 1.0)) * plot_w

    def sy(y: float) -> float:
        return margin["top"] + plot_h - ((y - y0) / (y1 - y0 or 1.0)) * plot_h

    parts = [_svg_header(width, height, title), _axis_svg(width, height, margin, "Match index", "tier mean Elo")]
    colors = _palette()
    for i, (tier, group_rows) in enumerate(groups.items()):
        color = colors[i % len(colors)]
        band_upper = []
        band_lower = []
        line = []
        for row in group_rows:
            x = _to_float(row.get("match_index"))
            mean = _to_float(row.get("mean_elo"))
            low = _to_float(row.get("ci95_low"))
            high = _to_float(row.get("ci95_high"))
            if x is None or mean is None:
                continue
            line.append((sx(x), sy(mean)))
            if low is not None and high is not None:
                band_upper.append((sx(x), sy(high)))
                band_lower.append((sx(x), sy(low)))
        if len(band_upper) >= 2 and len(band_lower) >= 2:
            band = band_upper + list(reversed(band_lower))
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in band)
            parts.append(f'<polygon points="{points}" fill="{color}" opacity="0.16"><title>{escape(tier)} 95% bootstrap CI</title></polygon>')
        if len(line) >= 2:
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in line)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"><title>{escape(tier)} mean Elo</title></polyline>')
        elif line:
            x, y = line[0]
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"><title>{escape(tier)} mean Elo</title></circle>')
        legend_y = margin["top"] + 20 + i * 22
        parts.append(f'<line x1="{width - 145}" y1="{legend_y}" x2="{width - 121}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width - 113}" y="{legend_y + 4}" font-size="12" fill="#334155">{escape(tier)}</text>')

    parts.append(_ticks_svg(values_min=x0, values_max=x1, y_min=y0, y_max=y1, margin=margin, width=width, height=height))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_elo_calibration_svg(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    logistic_scale: float = 400.0,
) -> None:
    width, height = 900, 460
    margin = {"left": 78, "right": 150, "top": 52, "bottom": 58}
    clean = [
        row for row in rows
        if _is_finite(row.get("mean_elo_gap")) and _is_finite(row.get("empirical_favorite_score"))
    ]
    if not clean:
        _write_empty_svg(path, title, "No data")
        return

    x_max = max(50.0, max(float(row["mean_elo_gap"]) for row in clean) * 1.08)
    y0, y1 = 0.45, 1.0
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    def sx(x: float) -> float:
        return margin["left"] + (x / (x_max or 1.0)) * plot_w

    def sy(y: float) -> float:
        return margin["top"] + plot_h - ((y - y0) / (y1 - y0 or 1.0)) * plot_h

    parts = [_svg_header(width, height, title), _axis_svg(width, height, margin, "pre-match Elo gap", "favorite score / win probability")]
    theory = []
    for step in range(101):
        gap = x_max * step / 100.0
        theory.append((
            sx(gap),
            sy(_elo_expected_favorite_probability(gap, logistic_scale=logistic_scale)),
        ))
    theory_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in theory)
    parts.append(f'<polyline points="{theory_points}" fill="none" stroke="#0f172a" stroke-width="2.2"><title>Theoretical Elo logistic curve</title></polyline>')

    empirical = []
    max_count = max(int(row.get("match_count") or 1) for row in clean)
    for row in sorted(clean, key=lambda item: float(item.get("mean_elo_gap") or 0.0)):
        x = float(row["mean_elo_gap"])
        y = float(row["empirical_favorite_score"])
        count = int(row.get("match_count") or 1)
        r = 4.0 + 7.0 * math.sqrt(count / max_count)
        empirical.append((sx(x), sy(y)))
        tooltip = (
            f"gap {x:.1f}; empirical {y:.3f}; "
            f"theory {_fmt_float(row.get('theoretical_favorite_win_probability'), 3)}; n={count}"
        )
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r:.1f}" fill="#2563eb" opacity="0.76"><title>{escape(tooltip)}</title></circle>')
    if len(empirical) >= 2:
        empirical_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in empirical)
        parts.append(f'<polyline points="{empirical_points}" fill="none" stroke="#2563eb" stroke-width="1.6" stroke-dasharray="4 4" opacity="0.7"/>')

    legend_x = width - 130
    legend_y = margin["top"] + 22
    parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="#0f172a" stroke-width="2.2"/>')
    parts.append(f'<text x="{legend_x + 32}" y="{legend_y + 4}" font-size="12" fill="#334155">theory</text>')
    parts.append(f'<circle cx="{legend_x + 12}" cy="{legend_y + 24}" r="6" fill="#2563eb" opacity="0.76"/>')
    parts.append(f'<text x="{legend_x + 32}" y="{legend_y + 28}" font-size="12" fill="#334155">empirical</text>')
    parts.append(_ticks_svg(values_min=0, values_max=x_max, y_min=y0, y_max=y1, margin=margin, width=width, height=height))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")



def _write_elo_distribution_svg(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    width, height = 900, 420
    margin = {"left": 78, "right": 150, "top": 52, "bottom": 58}
    clean = [
        row for row in rows
        if _is_finite(row.get("match_index")) and _is_finite(row.get("elo_median"))
    ]
    if not clean:
        _write_empty_svg(path, title, "No data")
        return

    xs = [float(row["match_index"]) for row in clean]
    y_values: list[float] = []
    for row in clean:
        for key in ("elo_p10", "elo_p25", "elo_median", "elo_mean", "elo_p75", "elo_p90"):
            value = _to_float(row.get(key))
            if value is not None:
                y_values.append(value)
    x0, x1 = min(xs), max(xs)
    y0, y1 = _padded_range(y_values)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    def sx(x: float) -> float:
        return margin["left"] + ((x - x0) / (x1 - x0 or 1.0)) * plot_w

    def sy(y: float) -> float:
        return margin["top"] + plot_h - ((y - y0) / (y1 - y0 or 1.0)) * plot_h

    def band_points(lower_key: str, upper_key: str) -> str:
        upper = []
        lower = []
        for row in clean:
            x = _to_float(row.get("match_index"))
            lo = _to_float(row.get(lower_key))
            hi = _to_float(row.get(upper_key))
            if x is None or lo is None or hi is None:
                continue
            upper.append((sx(x), sy(hi)))
            lower.append((sx(x), sy(lo)))
        points = upper + list(reversed(lower))
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    def line_points(key: str) -> str:
        points = []
        for row in clean:
            x = _to_float(row.get("match_index"))
            y = _to_float(row.get(key))
            if x is None or y is None:
                continue
            points.append((sx(x), sy(y)))
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    parts = [_svg_header(width, height, title), _axis_svg(width, height, margin, "Match index", "Elo")]
    p10_p90 = band_points("elo_p10", "elo_p90")
    if p10_p90:
        parts.append(f'<polygon points="{p10_p90}" fill="#bfdbfe" opacity="0.50"><title>p10-p90 Elo band</title></polygon>')
    p25_p75 = band_points("elo_p25", "elo_p75")
    if p25_p75:
        parts.append(f'<polygon points="{p25_p75}" fill="#60a5fa" opacity="0.45"><title>p25-p75 Elo band</title></polygon>')
    median = line_points("elo_median")
    if median:
        parts.append(f'<polyline points="{median}" fill="none" stroke="#0f172a" stroke-width="2.4"><title>median Elo</title></polyline>')
    mean = line_points("elo_mean")
    if mean:
        parts.append(f'<polyline points="{mean}" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="5 4"><title>mean Elo</title></polyline>')
    top10_mean = line_points("elo_top10_mean")
    if top10_mean:
        parts.append(f'<polyline points="{top10_mean}" fill="none" stroke="#7c3aed" stroke-width="2.1"><title>mean top-10 Elo</title></polyline>')
    top5_mean = line_points("elo_top5_mean")
    if top5_mean:
        parts.append(f'<polyline points="{top5_mean}" fill="none" stroke="#059669" stroke-width="2.1"><title>mean top-5 Elo</title></polyline>')
    legend_x = width - 130
    legend_y = margin["top"] + 20
    parts.append(f'<rect x="{legend_x}" y="{legend_y - 8}" width="22" height="10" fill="#bfdbfe" opacity="0.50"/>')
    parts.append(f'<text x="{legend_x + 30}" y="{legend_y + 1}" font-size="12" fill="#334155">p10-p90</text>')
    parts.append(f'<rect x="{legend_x}" y="{legend_y + 14}" width="22" height="10" fill="#60a5fa" opacity="0.45"/>')
    parts.append(f'<text x="{legend_x + 30}" y="{legend_y + 23}" font-size="12" fill="#334155">p25-p75</text>')
    parts.append(f'<line x1="{legend_x}" y1="{legend_y + 44}" x2="{legend_x + 22}" y2="{legend_y + 44}" stroke="#0f172a" stroke-width="2.4"/>')
    parts.append(f'<text x="{legend_x + 30}" y="{legend_y + 48}" font-size="12" fill="#334155">median</text>')
    parts.append(f'<line x1="{legend_x}" y1="{legend_y + 66}" x2="{legend_x + 22}" y2="{legend_y + 66}" stroke="#dc2626" stroke-width="2" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{legend_x + 30}" y="{legend_y + 70}" font-size="12" fill="#334155">mean</text>')
    parts.append(f'<line x1="{legend_x}" y1="{legend_y + 88}" x2="{legend_x + 22}" y2="{legend_y + 88}" stroke="#7c3aed" stroke-width="2.1"/>')
    parts.append(f'<text x="{legend_x + 30}" y="{legend_y + 92}" font-size="12" fill="#334155">top-10 mean</text>')
    parts.append(f'<line x1="{legend_x}" y1="{legend_y + 110}" x2="{legend_x + 22}" y2="{legend_y + 110}" stroke="#059669" stroke-width="2.1"/>')
    parts.append(f'<text x="{legend_x + 30}" y="{legend_y + 114}" font-size="12" fill="#334155">top-5 mean</text>')
    parts.append(_ticks_svg(values_min=x0, values_max=x1, y_min=y0, y_max=y1, margin=margin, width=width, height=height))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")

def _write_bar_svg(path: Path, values: list[tuple[str, int | float]], *, title: str, ylabel: str) -> None:
    width, height = 900, 420
    margin = {"left": 78, "right": 30, "top": 52, "bottom": 100}
    values = [(label, float(value)) for label, value in values if _is_finite(value)]
    if not values:
        _write_empty_svg(path, title, "No data")
        return
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_v = max(v for _label, v in values) or 1.0
    bar_w = plot_w / len(values)
    parts = [_svg_header(width, height, title), _axis_svg(width, height, margin, "", ylabel)]
    colors = _palette()
    for i, (label, value) in enumerate(values):
        h = (value / max_v) * plot_h
        x = margin["left"] + i * bar_w + bar_w * 0.12
        y = margin["top"] + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.76:.1f}" height="{h:.1f}" fill="{colors[i % len(colors)]}" opacity="0.82"/>')
        parts.append(f'<text transform="translate({x + bar_w * 0.38:.1f},{height - margin["bottom"] + 12}) rotate(45)" font-size="11" fill="#334155">{escape(_truncate(label, 24))}</text>')
    parts.append(_ticks_svg(values_min=0, values_max=len(values), y_min=0, y_max=max_v, margin=margin, width=width, height=height, x_labels=False))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_scatter_svg(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color_key: str,
    label_key: str,
    x_min: float | None = None,
    x_max: float | None = None,
    x_tick_values: list[float] | None = None,
) -> None:
    width, height = 900, 560
    categorical = color_key in {"cluster", "pca_cluster", "nearest_kb_cluster", "density_cluster"}
    margin = {"left": 78, "right": 170 if categorical else 120, "top": 52, "bottom": 58}
    points = [
        row for row in rows
        if _is_finite(row.get("x")) and _is_finite(row.get("y"))
    ]
    if not points:
        _write_empty_svg(path, title, "No data")
        return
    xs = [float(row["x"]) for row in points]
    ys = [float(row["y"]) for row in points]
    x0, x1 = _padded_range(xs)
    if x_min is not None:
        x0 = float(x_min)
    if x_max is not None:
        x1 = float(x_max)
    if x0 == x1:
        x0 -= 1.0
        x1 += 1.0
    y0, y1 = _padded_range(ys)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    color_values = [_to_float(row.get(color_key)) for row in points]
    numeric_values = [v for v in color_values if v is not None]
    c0 = min(numeric_values) if numeric_values else 0.0
    c1 = max(numeric_values) if numeric_values else 1.0
    category_values = sorted({int(v) for v in numeric_values}) if categorical else []
    colors = _palette()

    def sx(x: float) -> float:
        return margin["left"] + ((x - x0) / (x1 - x0 or 1.0)) * plot_w

    def sy(y: float) -> float:
        return margin["top"] + plot_h - ((y - y0) / (y1 - y0 or 1.0)) * plot_h

    def point_color(value: float | None) -> str:
        if value is None:
            return "#64748b"
        if categorical:
            category = int(value)
            if color_key == "density_cluster" and category < 0:
                return "#94a3b8"
            return colors[category % len(colors)]
        return _gradient_color(value, c0, c1)

    parts = [_svg_header(width, height, title), _axis_svg(width, height, margin, xlabel, ylabel)]
    for row in points:
        value = _to_float(row.get(color_key))
        color = point_color(value)
        radius = 4.5 if row.get("kind") != "kb_chunk" else 2.2
        opacity = 0.86 if row.get("kind") != "kb_chunk" else 0.38
        tooltip = escape(str(row.get(label_key) or row.get("hypothesis_id") or "point"))
        parts.append(
            f'<circle cx="{sx(float(row["x"])):.1f}" cy="{sy(float(row["y"])):.1f}" '
            f'r="{radius}" fill="{color}" opacity="{opacity}"><title>{tooltip}</title></circle>'
        )
    parts.append(
        _ticks_svg(
            values_min=x0,
            values_max=x1,
            y_min=y0,
            y_max=y1,
            margin=margin,
            width=width,
            height=height,
            x_tick_values=x_tick_values,
        )
    )

    legend_x = width - margin["right"] + 22
    legend_y = margin["top"] + 18
    if categorical:
        legend_title = "HDBSCAN cluster" if color_key == "density_cluster" else ("PCA cluster" if color_key == "pca_cluster" else "KB cluster")
        parts.append(f'<text x="{legend_x}" y="{legend_y}" font-size="12" font-weight="700" fill="#334155">{escape(legend_title)}</text>')
        for i, value in enumerate(category_values[:18]):
            y = legend_y + 22 + i * 18
            is_noise = color_key == "density_cluster" and value < 0
            color = "#94a3b8" if is_noise else colors[value % len(colors)]
            label = "noise" if is_noise else f"cluster {value}"
            parts.append(f'<circle cx="{legend_x + 6}" cy="{y - 4}" r="5" fill="{color}" opacity="0.86"/>')
            parts.append(f'<text x="{legend_x + 18}" y="{y}" font-size="12" fill="#334155">{label}</text>')
        if len(category_values) > 18:
            parts.append(f'<text x="{legend_x}" y="{legend_y + 22 + 18 * 18}" font-size="11" fill="#64748b">+{len(category_values) - 18} more</text>')
    else:
        parts.append(f'<text x="{width - 100}" y="{margin["top"] + 18}" font-size="12" fill="#334155">{escape(color_key)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_joint_pca_svg(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    width, height = 900, 560
    margin = {"left": 78, "right": 150, "top": 52, "bottom": 58}
    points = [row for row in rows if _is_finite(row.get("x")) and _is_finite(row.get("y"))]
    if not points:
        _write_empty_svg(path, title, "No data")
        return
    xs = [float(row["x"]) for row in points]
    ys = [float(row["y"]) for row in points]
    x0, x1 = _padded_range(xs)
    y0, y1 = _padded_range(ys)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    elos = [_to_float(row.get("elo")) for row in points if row.get("kind") == "hypothesis"]
    elos = [v for v in elos if v is not None]
    e0, e1 = (min(elos), max(elos)) if elos else (0.0, 1.0)

    def sx(x: float) -> float:
        return margin["left"] + ((x - x0) / (x1 - x0 or 1.0)) * plot_w

    def sy(y: float) -> float:
        return margin["top"] + plot_h - ((y - y0) / (y1 - y0 or 1.0)) * plot_h

    parts = [_svg_header(width, height, title), _axis_svg(width, height, margin, "PC1", "PC2")]
    colors = _palette()
    # Draw KB first so hypotheses remain visible. KB color encodes the KB cluster.
    for row in points:
        if row.get("kind") == "hypothesis":
            continue
        cluster = _to_float(row.get("cluster"))
        color = colors[int(cluster) % len(colors)] if cluster is not None else "#94a3b8"
        label = f"cluster={row.get('cluster')} source={row.get('source') or ''}"
        parts.append(
            f'<circle cx="{sx(float(row["x"])):.1f}" cy="{sy(float(row["y"])):.1f}" '
            f'r="2.0" fill="{color}" opacity="0.24"><title>{escape(label)}</title></circle>'
        )
    hypothesis_points = [row for row in points if row.get("kind") == "hypothesis"]
    hypothesis_points.sort(key=lambda row: _to_float(row.get("elo")) or float("-inf"))
    for row in hypothesis_points:
        color = _gradient_color(_to_float(row.get("elo")), e0, e1)
        label = f"{row.get('id')} rank={row.get('rank')} elo={_fmt_float(row.get('elo'), 1)}"
        parts.append(
            f'<circle cx="{sx(float(row["x"])):.1f}" cy="{sy(float(row["y"])):.1f}" '
            f'r="5.0" fill="{color}" stroke="#0f172a" stroke-width="0.4" opacity="0.72">'
            f'<title>{escape(label)}</title></circle>'
        )
    parts.append(_ticks_svg(values_min=x0, values_max=x1, y_min=y0, y_max=y1, margin=margin, width=width, height=height))
    parts.append(f'<circle cx="{width - 130}" cy="{margin["top"] + 20}" r="4" fill="#ef4444"/>')
    parts.append(f'<text x="{width - 119}" y="{margin["top"] + 24}" font-size="12" fill="#334155">hypothesis Elo</text>')
    parts.append(f'<circle cx="{width - 130}" cy="{margin["top"] + 42}" r="4" fill="#2563eb" opacity="0.45"/>')
    parts.append(f'<text x="{width - 119}" y="{margin["top"] + 46}" font-size="12" fill="#334155">KB cluster</text>')
    bar_x = width - 75
    bar_y = margin["top"] + 82
    bar_h = 170
    bar_w = 14
    for i in range(40):
        frac = i / 39
        value = e1 - frac * (e1 - e0)
        color = _gradient_color(value, e0, e1)
        y = bar_y + frac * bar_h
        parts.append(f'<rect x="{bar_x}" y="{y:.1f}" width="{bar_w}" height="{bar_h / 39 + 0.6:.1f}" fill="{color}"/>')
    parts.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" fill="none" stroke="#334155" stroke-width="0.7"/>')
    parts.append(f'<text x="{bar_x - 5}" y="{bar_y + 4}" text-anchor="end" font-size="11" fill="#334155">{_compact_num(e1)}</text>')
    parts.append(f'<text x="{bar_x - 5}" y="{bar_y + bar_h + 4}" text-anchor="end" font-size="11" fill="#334155">{_compact_num(e0)}</text>')
    parts.append(f'<text transform="translate({bar_x + 34},{bar_y + bar_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="12" fill="#334155">Elo</text>')
    kb_clusters = sorted({int(row.get("cluster")) for row in points if row.get("kind") == "kb_chunk" and _to_float(row.get("cluster")) is not None})
    cluster_x = width - 135
    cluster_y = bar_y + bar_h + 42
    parts.append(f'<text x="{cluster_x}" y="{cluster_y}" font-size="12" font-weight="700" fill="#334155">KB cluster colors</text>')
    for i, cluster in enumerate(kb_clusters[:10]):
        y = cluster_y + 22 + i * 17
        color = colors[cluster % len(colors)]
        parts.append(f'<circle cx="{cluster_x + 6}" cy="{y - 4}" r="4.5" fill="{color}" opacity="0.75"/>')
        parts.append(f'<text x="{cluster_x + 18}" y="{y}" font-size="11" fill="#334155">cluster {cluster}</text>')
    if len(kb_clusters) > 10:
        parts.append(f'<text x="{cluster_x}" y="{cluster_y + 22 + 10 * 17}" font-size="11" fill="#64748b">+{len(kb_clusters) - 10} more</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_empty_svg(path: Path, title: str, message: str) -> None:
    path.write_text(
        "\n".join(
            [
                _svg_header(900, 300, title),
                f'<text x="450" y="160" text-anchor="middle" font-size="16" fill="#64748b">{escape(message)}</text>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def _svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#0f172a">'
        f'{escape(title)}</text>'
    )


def _axis_svg(width: int, height: int, margin: dict[str, int], xlabel: str, ylabel: str) -> str:
    x0, y0 = margin["left"], height - margin["bottom"]
    x1, y1 = width - margin["right"], margin["top"]
    return (
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#334155" stroke-width="1"/>'
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#334155" stroke-width="1"/>'
        f'<text x="{(x0 + x1) / 2:.0f}" y="{height - 14}" text-anchor="middle" '
        f'font-size="13" fill="#334155">{escape(xlabel)}</text>'
        f'<text transform="translate(18,{(y0 + y1) / 2:.0f}) rotate(-90)" text-anchor="middle" '
        f'font-size="13" fill="#334155">{escape(ylabel)}</text>'
    )


def _ticks_svg(
    *,
    values_min: float,
    values_max: float,
    y_min: float,
    y_max: float,
    margin: dict[str, int],
    width: int,
    height: int,
    x_labels: bool = True,
    x_tick_values: list[float] | None = None,
) -> str:
    parts = []
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    if x_tick_values is None:
        x_tick_values = [values_min + (i / 4) * (values_max - values_min) for i in range(5)]
    for val in x_tick_values:
        x = margin["left"] + ((float(val) - values_min) / (values_max - values_min or 1.0)) * plot_w
        y = height - margin["bottom"]
        parts.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + 5}" stroke="#64748b"/>')
        if x_labels:
            parts.append(f'<text x="{x:.1f}" y="{y + 20}" text-anchor="middle" font-size="11" fill="#64748b">{_compact_num(float(val))}</text>')
    for i in range(5):
        frac = i / 4
        y = height - margin["bottom"] - frac * plot_h
        x = margin["left"]
        val = y_min + frac * (y_max - y_min)
        parts.append(f'<line x1="{x - 5}" y1="{y:.1f}" x2="{x}" y2="{y:.1f}" stroke="#64748b"/>')
        parts.append(f'<text x="{x - 9}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{_compact_num(val)}</text>')
    return "\n".join(parts)


def _kmeans_labels(vectors: np.ndarray, k: int) -> list[int]:
    if len(vectors) <= 1:
        return [0] * len(vectors)
    k = min(max(1, k), len(vectors))
    if k == 1:
        return [0] * len(vectors)
    model = KMeans(n_clusters=k, n_init=10, random_state=13)
    return [int(x) for x in model.fit_predict(vectors)]


def _sample_pairwise_cosine(vectors: np.ndarray, *, max_pairs: int) -> list[float]:
    n = len(vectors)
    if n < 2:
        return []
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        sims = vectors @ vectors.T
        return [float(x) for x in sims[np.triu_indices_from(sims, k=1)]]
    rng = np.random.default_rng(13)
    out = []
    for _ in range(max_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        out.append(float(np.dot(vectors[int(i)], vectors[int(j)])))
    return out


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


def _summary_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": _safe_mean(values),
        "median": _safe_median(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _safe_mean(values: list[float] | tuple[float, ...]) -> float | None:
    vals = [float(v) for v in values if _is_finite(v)]
    return float(sum(vals) / len(vals)) if vals else None


def _safe_median(values: list[float] | tuple[float, ...]) -> float | None:
    vals = sorted(float(v) for v in values if _is_finite(v))
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _percentile(values: list[float], pct: float) -> float | None:
    vals = [float(v) for v in values if _is_finite(v)]
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype="float64"), pct))


def _percentile_rank(values: list[float], value: float) -> float | None:
    vals = [float(v) for v in values if _is_finite(v)]
    if not vals or not _is_finite(value):
        return None
    below_or_equal = sum(1 for v in vals if v <= float(value))
    return 100.0 * below_or_equal / len(vals)


def _pearson_corr(xs: list[Any], ys: list[Any]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys, strict=False) if _is_finite(x) and _is_finite(y)]
    if len(pairs) < 2:
        return None
    x = np.asarray([p[0] for p in pairs], dtype="float64")
    y = np.asarray([p[1] for p in pairs], dtype="float64")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rank_numeric(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype="float64")
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype="float64")
    ranks[order] = np.arange(1, len(arr) + 1, dtype="float64")
    return [float(x) for x in ranks]


def _spearman_corr(xs: list[Any], ys: list[Any]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys, strict=False) if _is_finite(x) and _is_finite(y)]
    if len(pairs) < 2:
        return None
    return _pearson_corr(_rank_numeric([p[0] for p in pairs]), _rank_numeric([p[1] for p in pairs]))


def _entropy(counts: list[int]) -> float | None:
    total = sum(counts)
    if total <= 0:
        return None
    out = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        out -= p * math.log(p, 2)
    return out


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _parse_parent_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if isinstance(decoded, list):
        return [str(v) for v in decoded]
    return []


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _min_dt_str(a: Any, b: Any) -> str | None:
    if not a:
        return str(b) if b else None
    if not b:
        return str(a)
    da, db = _parse_dt(a), _parse_dt(b)
    if da and db:
        return str(a if da <= db else b)
    return str(a)


def _max_dt_str(a: Any, b: Any) -> str | None:
    if not a:
        return str(b) if b else None
    if not b:
        return str(a)
    da, db = _parse_dt(a), _parse_dt(b)
    if da and db:
        return str(a if da >= db else b)
    return str(a)


def _normalize_source_id(value: str) -> str:
    v = value.strip().lower().rstrip(".,;")
    if v.startswith("http://arxiv.org/"):
        v = "https://" + v.removeprefix("http://")
    return v


def _title_key(value: str) -> str:
    tokens = _TOKEN_RE.findall(value.lower())
    return " ".join(tokens)


def _padded_range(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    if lo == hi:
        return lo - 1.0, hi + 1.0
    pad = (hi - lo) * 0.06
    return lo - pad, hi + pad


def _gradient_color(value: float | None, lo: float, hi: float) -> str:
    if value is None or not math.isfinite(value):
        return "#64748b"
    t = (value - lo) / (hi - lo or 1.0)
    t = max(0.0, min(1.0, t))
    # Blue -> amber -> red.
    if t < 0.5:
        u = t / 0.5
        c0, c1 = (37, 99, 235), (245, 158, 11)
    else:
        u = (t - 0.5) / 0.5
        c0, c1 = (245, 158, 11), (220, 38, 38)
    rgb = tuple(round(c0[i] + u * (c1[i] - c0[i])) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _palette() -> list[str]:
    return ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f59e0b", "#0891b2", "#db2777", "#4f46e5"]


def _compact_num(value: float) -> str:
    if math.isfinite(float(value)) and float(value).is_integer() and abs(value) < 1000:
        return str(int(value))
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"{value / 1_000:.0f}k"
    if abs(value) >= 1000:
        return f"{value / 1_000:.1f}k"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _fmt_float(value: Any, digits: int = 2) -> str:
    v = _to_float(value)
    return "n/a" if v is None else f"{v:.{digits}f}"


def _fmt_percent(value: Any) -> str:
    v = _to_float(value)
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _human_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _counter_sentence(counter: dict[str, Any]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def _cluster_count_sentence(counts: Any, cluster_info: dict[int, dict[str, Any]]) -> str:
    if not isinstance(counts, dict) or not counts:
        return "none"
    parts = []
    for key, value in sorted(counts.items(), key=lambda item: int(item[0])):
        cluster = int(key)
        label = cluster_info.get(cluster, {}).get("llm_label") or cluster_info.get(cluster, {}).get("deterministic_label") or "cluster"
        parts.append(f"{cluster} ({label})={value}")
    return ", ".join(parts)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["" for _ in headers]]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_md_escape(str(cell)) for cell in row) + " |")
    return "\n".join(out)


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|")


def _truncate(value: str, length: int) -> str:
    return value if len(value) <= length else value[: length - 1] + "..."
