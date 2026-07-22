# Modified from the original work.
"""Pre-flight cost estimator.

Given a parsed ResearchPlan and config, predict the session's expected USD
spend. Used by `hypothesis-engine run` to surface a warning before launching when
the estimate is more than ~1.2x the budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from .routing import estimate_cost_usd


@dataclass(frozen=True)
class CallEstimate:
    label: str
    n_calls: int
    model: str
    input_tokens_each: int
    output_tokens_each: int
    cache_ratio: float       # 0..1, fraction of inputs that come from cache

    @property
    def cached_in(self) -> int:
        return int(self.input_tokens_each * self.cache_ratio)

    @property
    def uncached_in(self) -> int:
        return self.input_tokens_each - self.cached_in

    def usd(self) -> float:
        per = estimate_cost_usd(
            model=self.model,
            input_tokens=self.uncached_in,
            output_tokens=self.output_tokens_each,
            cache_read=self.cached_in,
            cache_write=0,
        )
        return per * self.n_calls


@dataclass(frozen=True)
class SessionEstimate:
    rows: list[CallEstimate]
    total_usd: float
    warning: str | None = None

    def to_dict(self) -> dict:
        return {
            "rows": [
                {
                    "label": r.label, "n_calls": r.n_calls, "model": r.model,
                    "input_tokens_each": r.input_tokens_each,
                    "output_tokens_each": r.output_tokens_each,
                    "cache_ratio": r.cache_ratio, "usd": r.usd(),
                }
                for r in self.rows
            ],
            "total_usd": self.total_usd,
            "warning": self.warning,
        }


def estimate(cfg: Config, *, max_ideas: int | None = None,
             max_matches_per_idea: int | None = None) -> SessionEstimate:
    """Coarse pre-flight estimate. Heuristics live in this one place."""
    n_ideas = max_ideas or cfg.run.max_ideas
    matches_per = max_matches_per_idea or cfg.run.max_matches_per_idea
    n_matches = n_ideas * matches_per // 2  # each match counts for both sides
    n_evolutions = int(n_ideas * 0.4)
    # n_reviews is implicit in the rows below (full + verification per hypothesis).
    n_meta_periodic = max(1, n_matches // 50)

    rows = [
        CallEstimate(
            label="parse_goal", n_calls=1, model=cfg.models.parse_goal,
            input_tokens_each=600, output_tokens_each=300, cache_ratio=0.0,
        ),
        CallEstimate(
            label="generation", n_calls=n_ideas, model=cfg.models.generation,
            input_tokens_each=18_000, output_tokens_each=2_500, cache_ratio=0.55,
        ),
        CallEstimate(
            label="reflection.full", n_calls=n_ideas, model=cfg.models.reflection,
            input_tokens_each=16_000, output_tokens_each=2_000, cache_ratio=0.55,
        ),
        CallEstimate(
            label="reflection.verification", n_calls=n_ideas,
            model=cfg.models.reflection,
            input_tokens_each=14_000, output_tokens_each=3_000, cache_ratio=0.55,
        ),
        CallEstimate(
            label="ranking_pairwise", n_calls=n_matches,
            model=cfg.models.ranking_pairwise,
            input_tokens_each=8_000, output_tokens_each=800, cache_ratio=0.70,
        ),
        CallEstimate(
            label="evolution", n_calls=n_evolutions, model=cfg.models.evolution,
            input_tokens_each=16_000, output_tokens_each=2_500, cache_ratio=0.50,
        ),
        CallEstimate(
            label="metareview.system", n_calls=n_meta_periodic,
            model=cfg.models.metareview_feedback,
            input_tokens_each=14_000, output_tokens_each=2_000, cache_ratio=0.50,
        ),
        CallEstimate(
            label="metareview.final", n_calls=1,
            model=cfg.models.metareview_final,
            input_tokens_each=24_000, output_tokens_each=6_000, cache_ratio=0.40,
        ),
    ]
    total = sum(r.usd() for r in rows)
    warn = None
    if total > cfg.run.budget_usd * 1.2:
        warn = (
            f"Estimated cost ${total:.2f} exceeds 120% of budget "
            f"${cfg.run.budget_usd:.2f}. Consider reducing max_ideas, lowering "
            f"max_matches_per_idea, or raising --budget-usd."
        )
    return SessionEstimate(rows=rows, total_usd=total, warning=warn)
