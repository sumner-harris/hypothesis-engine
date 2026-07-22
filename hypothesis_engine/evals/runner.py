# Modified from the original work.
"""Eval runner. Reads JSONL fixtures and scores them with rubrics.

Layout:
    hypothesis_engine/evals/fixtures/<agent>.jsonl   — one fixture per line
    each fixture: {"id": "...", "candidate": "...", "expected": {...}}

`expected` is a dict of structural assertions:
- must_contain: list[str]
- must_cite_at_least: int (reflection only)

Run:
    hypothesis-engine eval <agent>             — score all fixtures with the judge
    hypothesis-engine eval <agent> --offline   — only check structural assertions
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from typing import Any

from ..config import PROJECT_ROOT, Config
from ..logging import get_logger
from .rubrics import (
    GENERATION_RUBRIC,
    OVERVIEW_RUBRIC,
    RANKING_RUBRIC,
    REFLECTION_RUBRIC,
    RubricCriterion,
    judge,
)

log = get_logger("evals")
FIXTURES_DIR = PROJECT_ROOT / "hypothesis_engine" / "evals" / "fixtures"

RUBRICS: dict[str, list[RubricCriterion]] = {
    "generation": GENERATION_RUBRIC,
    "reflection": REFLECTION_RUBRIC,
    "ranking":    RANKING_RUBRIC,
    "overview":   OVERVIEW_RUBRIC,
}


@dataclass
class FixtureResult:
    id: str
    structural_ok: bool
    structural_errors: list[str]
    weighted: float | None = None
    scores: list[dict[str, Any]] | None = None
    notes: str | None = None


def _check_structure(agent: str, candidate: str, expected: dict[str, Any]) -> list[str]:
    """Cheap structural checks that don't need an LLM."""
    errors: list[str] = []
    for needle in expected.get("must_contain", []):
        if needle not in candidate:
            errors.append(f"missing required substring: {needle!r}")
    if agent == "ranking" and "better idea:" not in candidate.lower():
        errors.append("ranking output must end with 'better idea: 1|2'")
    if agent == "reflection":
        hits = expected.get("must_cite_at_least", 0)
        if hits and candidate.lower().count("http") < hits:
            errors.append(f"expected at least {hits} URLs, found fewer")
    if agent == "generation":
        # Heuristic: insist on the four canonical sections appearing.
        for h in ("mechanism", "entit", "anticipated", "novelty"):
            if h not in candidate.lower():
                errors.append(f"hypothesis is missing {h}-style section")
    return errors


def _iter_fixtures(agent: str) -> list[dict[str, Any]]:
    p = FIXTURES_DIR / f"{agent}.jsonl"
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for lineno, raw in enumerate(p.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as e:
            # A single malformed fixture must not abort the whole run.
            log.warning(
                "eval_fixture_parse_failed",
                path=str(p), line=lineno, err=str(e),
            )
            continue
        if not isinstance(parsed, dict):
            log.warning("eval_fixture_not_dict", path=str(p), line=lineno)
            continue
        out.append(parsed)
    return out


async def run_agent(
    cfg: Config, agent: str, *, offline: bool = False, judge_call: bool = True
) -> dict[str, Any]:
    rubric = RUBRICS.get(agent)
    if rubric is None:
        return {"agent": agent, "error": f"no rubric for agent {agent!r}"}
    fixtures = _iter_fixtures(agent)
    if not fixtures:
        return {"agent": agent, "n_fixtures": 0, "note": f"no fixtures at {FIXTURES_DIR / (agent + '.jsonl')}"}

    results: list[FixtureResult] = []
    for f in fixtures:
        cand = f.get("candidate", "")
        errs = _check_structure(agent, cand, f.get("expected", {}))
        ok = not errs
        weighted = None
        scores: list[dict[str, Any]] | None = None
        notes = None
        if ok and judge_call and not offline:
            j = await judge(cfg, rubric=rubric, candidate=cand, label=f.get("id", "candidate"))
            weighted = j.get("weighted")
            scores = j.get("scores")
            notes = j.get("notes")
        results.append(FixtureResult(
            id=f.get("id", "?"),
            structural_ok=ok,
            structural_errors=errs,
            weighted=weighted,
            scores=scores,
            notes=notes,
        ))

    weighted_scores = [r.weighted for r in results if r.weighted is not None]
    return {
        "agent": agent,
        "n_fixtures": len(results),
        "structural_pass_rate": sum(1 for r in results if r.structural_ok) / max(1, len(results)),
        "mean_weighted": sum(weighted_scores) / len(weighted_scores) if weighted_scores else None,
        "results": [asdict(r) for r in results],
    }


async def run_all(cfg: Config, *, offline: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"agents": {}}
    for agent in RUBRICS:
        out["agents"][agent] = await run_agent(cfg, agent, offline=offline)
    return out


def _main() -> None:  # pragma: no cover
    import argparse

    from ..config import load_config

    parser = argparse.ArgumentParser(prog="hypothesis-engine-evals")
    parser.add_argument("agent", nargs="?", help="generation|reflection|ranking|overview, or omit for all")
    parser.add_argument("--offline", action="store_true", help="Skip judge calls; structural only.")
    args = parser.parse_args()
    cfg = load_config()
    if args.agent:
        result = asyncio.run(run_agent(cfg, args.agent, offline=args.offline))
    else:
        result = asyncio.run(run_all(cfg, offline=args.offline))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    _main()
