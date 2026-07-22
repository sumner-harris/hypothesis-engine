"""Citation formatting helpers for stored hypothesis sources."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import CitedPaper, Hypothesis


def merge_citation_candidates(
    citations: Iterable[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
    *,
    seen_urls: Iterable[str] | None = None,
    max_total: int = 8,
) -> list[dict[str, Any]]:
    """Merge explicit LLM citations with source metadata from retrieval tools.

    The merged list is only stored/reported; it is not fed back into future LLM
    prompts. Explicit citations keep priority, then ordered RAG/web source
    candidates fill the remaining slots.
    """
    allowed = set(seen_urls or []) or None
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        if len(out) >= max(0, int(max_total)):
            return
        if not isinstance(item, dict):
            return
        url = str(item.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return
        if allowed is not None and url not in allowed:
            return
        key = url.casefold()
        if key in seen:
            return
        title = str(item.get("title") or item.get("source") or url).strip()
        citation: dict[str, Any] = {"url": url, "title": title or url}
        for field in ("excerpt", "doi", "year"):
            value = item.get(field)
            if value not in (None, ""):
                citation[field] = value
        out.append(citation)
        seen.add(key)

    for citation in citations or []:
        if isinstance(citation, dict):
            add(citation)
    for candidate in candidates or []:
        if isinstance(candidate, dict):
            add(candidate)
    return out


def render_citations_md(
    citations: Iterable[CitedPaper],
    *,
    heading: str = "## Stored citations",
    include_excerpts: bool = True,
) -> str:
    items = list(citations)
    if not items:
        return ""
    parts: list[str] = [heading]
    for citation in items:
        parts.append(_citation_bullet(citation, include_excerpts=include_excerpts))
    return "\n".join(parts)


def render_hypothesis_citation_appendix(
    hypotheses: Iterable[Hypothesis],
    *,
    heading: str = "# Hypothesis citation sources",
) -> str:
    sections: list[str] = []
    for hypothesis in hypotheses:
        if not hypothesis.citations:
            continue
        title = hypothesis.title or hypothesis.id
        lines = [f"## {title} (`{hypothesis.id}`)"]
        for citation in hypothesis.citations:
            lines.append(_citation_bullet(citation, include_excerpts=False))
        sections.append("\n".join(lines))
    if not sections:
        return ""
    return heading + "\n\n" + "\n\n".join(sections)


def _citation_bullet(citation: CitedPaper, *, include_excerpts: bool) -> str:
    title = (citation.title or "(untitled source)").strip()
    year = f" ({citation.year})" if citation.year else ""
    suffixes: list[str] = []
    if citation.doi:
        suffixes.append(f"DOI: {citation.doi}")
    suffix = f"; {'; '.join(suffixes)}" if suffixes else ""
    line = f"- {title}{year} - {citation.url}{suffix}"
    if include_excerpts and citation.excerpt:
        excerpt = " ".join(str(citation.excerpt).split())
        if excerpt:
            line += f"\n  - Excerpt: {excerpt}"
    return line
