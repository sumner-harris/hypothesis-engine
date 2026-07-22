"""Review models — produced by the Reflection agent."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ReviewKind = Literal["full", "verification", "observation", "simulation"]
ReviewVerdict = Literal[
    "already_explained", "other_more_likely", "missing_piece", "neutral", "disproved"
]


class Evidence(BaseModel):
    """Required structure for any factual claim a review makes.

    The Reflection agent's `record_review` tool requires url + excerpt,
    so the citation_verifier can fetch the URL and check the excerpt
    actually appears on the page.
    """

    claim: str
    url: str
    excerpt: str


class AssumptionCheck(BaseModel):
    """One row of the deep-verification decomposition."""

    assumption: str
    plausibility: Literal["plausible", "uncertain", "implausible"]
    rationale: str


class ReviewScores(BaseModel):
    novelty: float | None = None         # 0..1
    correctness: float | None = None
    testability: float | None = None
    feasibility: float | None = None


class Review(BaseModel):
    id: str
    hypothesis_id: str
    session_id: str
    created_at: datetime
    kind: ReviewKind
    verdict: ReviewVerdict | None = None
    scores: ReviewScores = Field(default_factory=ReviewScores)
    assumptions: list[AssumptionCheck] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    body: str                              # markdown
    artifact_path: str
