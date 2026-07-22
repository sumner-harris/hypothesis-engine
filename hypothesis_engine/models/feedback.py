"""System / human feedback models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

FeedbackSource = Literal["human", "meta_review"]
FeedbackKind = Literal["directive", "preference", "rejection", "pin", "system_feedback"]


class SystemFeedback(BaseModel):
    id: str
    session_id: str
    created_at: datetime
    source: FeedbackSource
    kind: FeedbackKind
    target_id: str | None = None
    text: str
    artifact_path: str | None = None
    active: bool = True


class MetaReviewReport(BaseModel):
    common_weaknesses: list[str] = Field(default_factory=list)
    common_strengths: list[str] = Field(default_factory=list)
    suggested_focus_areas: list[str] = Field(default_factory=list)
    narrative: str = ""
