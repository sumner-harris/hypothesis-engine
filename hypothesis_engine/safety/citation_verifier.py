# Modified from the original work.
"""Citation verifier.

For every Evidence{claim, url, excerpt} in a Reflection record:
1. Fetch the URL (cached on disk by web_fetch).
2. Check whether the excerpt actually appears on the page.
3. If not, mark the review with a `citation_unverified` flag (in artifact JSON)
   and emit a `citation_unverified` event.

The fuzzy-match step uses simple normalized substring search. We deliberately
do NOT make a second LLM call by default; if the verifier is going to be a
hot path we can add Haiku-as-judge later.
"""

from __future__ import annotations

import re
from typing import Any

from ..config import Config
from ..logging import get_logger
from ..models import Review
from ..storage.artifacts import read_json
from ..storage.repos import sessions as sess_repo
from ..tools.base import ToolCtx
from ..tools.web_fetch import WebFetchTool

log = get_logger("safety.citation_verifier")

_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WS_RE.sub(" ", s.lower()).strip()


class CitationVerifier:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._fetcher = WebFetchTool(cfg)

    async def verify_review(self, session_id: str, review: Review, db) -> dict[str, Any]:
        """Returns {url: {status: 'ok'|'unverified'|'fetch_failed'}}."""
        if not self._cfg.safety.enable_citation_verifier:
            return {}
        # Pull the evidence array from the stored artifact (the row drops it).
        try:
            payload = await read_json(self._cfg, review.artifact_path)
        except Exception as e:
            log.warning("verify_review_no_artifact", review_id=review.id, err=str(e))
            return {}
        record = payload.get("record") or {}
        evidence: list[dict[str, Any]] = record.get("evidence") or []
        if not evidence:
            return {}

        # Ensure session exists so artifacts path resolution works
        session = await sess_repo.fetch(db, session_id)
        if session is None:
            return {}

        ctx = ToolCtx(cfg=self._cfg, session_id=session_id, task_id=None, run_id=review.id)
        out: dict[str, Any] = {}
        for ev in evidence:
            url = ev.get("url")
            excerpt = ev.get("excerpt")
            if not url or not excerpt:
                continue
            result = await self._fetcher.call(
                {"url": url, "max_chars": self._cfg.web_fetch.max_chars}, ctx
            )
            if result.is_error or not isinstance(result.content, dict):
                out[url] = {"status": "fetch_failed", "error": result.error_message}
                continue
            text = result.content.get("text") or ""
            verified = _normalize(excerpt[:200]) in _normalize(text)
            out[url] = {
                "status": "ok" if verified else "unverified",
                "excerpt_len": len(excerpt),
            }
        return out
