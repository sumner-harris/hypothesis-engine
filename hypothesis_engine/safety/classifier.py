"""Abuse / dual-use classifier.

A small Haiku-backed classifier with a structured-output tool. Placement:
- goal-parse time (mandatory): block / warn / allow
- hypothesis-save time (mandatory): ok / quarantine / block
- final-report time (optional): ok / redact_quarantined / block_publish

Defensive use only. When the API key is missing, we return `ok` and log a
warning rather than crashing the session — agents stay functional in dev.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from anthropic import AsyncAnthropic

from ..config import Config
from ..logging import get_logger

log = get_logger("safety.classifier")


Action = Literal["allow", "warn", "block", "quarantine"]


CLASSIFY_TOOL: dict[str, Any] = {
    "name": "record_safety_assessment",
    "description": "Record a structured safety assessment of input text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "categories": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "none",
                        "dual_use_bio",
                        "cbrn",
                        "weapons",
                        "illicit_synthesis",
                        "csam",
                    ],
                },
                "description": "All categories that apply. Use ['none'] if benign.",
            },
            "confidence": {
                "type": "number", "minimum": 0, "maximum": 1,
                "description": "0..1 confidence in the worst-case categorization.",
            },
            "rationale": {"type": "string"},
        },
        "required": ["categories", "confidence", "rationale"],
    },
}


_SYSTEM_PROMPT = (
    "You are a content-safety classifier for an AI research-hypothesis system. "
    "You will be shown a passage of text (a research goal, a hypothesis, or a "
    "synthesized report). Classify the content into one or more categories and "
    "report your confidence.\n\n"
    "Categories:\n"
    " - none: benign scientific content.\n"
    " - dual_use_bio: research that could plausibly be misused for biological "
    "harm (gain-of-function, enhancement of pathogen transmissibility, etc.).\n"
    " - cbrn: chemical, biological, radiological, or nuclear weapons.\n"
    " - weapons: conventional weapons synthesis or improvement.\n"
    " - illicit_synthesis: drug or precursor synthesis routes for unlawful use.\n"
    " - csam: child sexual abuse material.\n\n"
    "Be precise. Benign biology research is NOT dual_use_bio; only flag when the "
    "stated outcome could plausibly increase harm capability. When in doubt, "
    "use 'none' with low confidence. Always call the record_safety_assessment tool."
)


class ClassifierResult:
    def __init__(self, categories: list[str], confidence: float, rationale: str) -> None:
        self.categories = categories or ["none"]
        self.confidence = confidence
        self.rationale = rationale

    @property
    def is_benign(self) -> bool:
        return self.categories == ["none"] or ("none" in self.categories and len(self.categories) == 1)

    def action(self, cfg: Config) -> Action:
        if self.is_benign:
            return "allow"
        block = set(cfg.safety.classifier_block_categories)
        warn = set(cfg.safety.classifier_warn_categories)
        flagged = set(self.categories) - {"none"}
        if flagged & block:
            return "block"
        if flagged & warn and self.confidence >= 0.6:
            return "quarantine"
        if flagged & warn:
            return "warn"
        return "allow"


class SafetyClassifier:
    """One classifier per Config; reuses the Anthropic client."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._client: AsyncAnthropic | None = None
        api_key = cfg.secrets.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY") or ""
        if api_key:
            self._client = AsyncAnthropic(api_key=api_key)

    async def classify(self, text: str, *, label: str = "input") -> ClassifierResult:
        """Always returns a result; degrades to benign + warning log on failure."""
        if not self._cfg.safety.enable_classifier or self._client is None:
            return ClassifierResult(categories=["none"], confidence=0.0,
                                    rationale="classifier disabled or no key")
        text = text.strip()
        if not text:
            return ClassifierResult(categories=["none"], confidence=1.0,
                                    rationale="empty input")
        try:
            resp = await self._client.messages.create(
                model=self._cfg.models.classifier,
                system=_SYSTEM_PROMPT,
                max_tokens=512,
                tools=[CLASSIFY_TOOL],
                tool_choice={"type": "tool", "name": "record_safety_assessment"},
                messages=[
                    {"role": "user", "content": f"<TEXT label=\"{label}\">\n{text[:8000]}\n</TEXT>"},
                ],
            )
        except Exception as e:
            log.warning("classifier_call_failed", err=str(e))
            return ClassifierResult(categories=["none"], confidence=0.0,
                                    rationale=f"classifier_error: {e!s}")
        for b in resp.content:
            if getattr(b, "type", None) == "tool_use" and getattr(b, "name", "") == "record_safety_assessment":
                inp = getattr(b, "input", None)
                if isinstance(inp, dict):
                    return ClassifierResult(
                        categories=list(inp.get("categories", ["none"])),
                        confidence=float(inp.get("confidence", 0.0)),
                        rationale=str(inp.get("rationale", "")),
                    )
        return ClassifierResult(categories=["none"], confidence=0.0,
                                rationale="no tool_use block in response")
