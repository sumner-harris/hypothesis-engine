# Modified from the original work.
"""Built-in bench candidate presets.

Curated comparison setups so users can reproduce known benchmarks with one
flag instead of typing N `--candidate` lines.

Substitutions in the "paper" preset
-----------------------------------
Google's original multi-agent scientific-discovery paper compared its system against:

    Gemini 2.0 Flash Thinking Experimental 12-19
    Gemini 2.0 Pro Experimental
    OpenAI o1

across 15 expert-curated research goals. Of those, OpenRouter currently
serves only `openai/o1`; the experimental Gemini 2.0 Thinking and Pro
Experimental branches were retired. We substitute the closest current
analogues and document the swap so reported numbers are interpretable:

    paper-baseline                            current substitute
    ----------------------------------------  --------------------------------
    Gemini 2.0 Flash Thinking Experimental    google/gemini-2.0-flash-001
       (Thinking branch deprecated; production 2.0 Flash is the closest)
    Gemini 2.0 Pro Experimental               google/gemini-2.5-pro
       (2.0 Pro Experimental removed; 2.5 Pro is the current Pro-tier Gemini)
    OpenAI o1                                 openai/o1               (exact)
    Claude Haiku (added by user request)      anthropic/claude-haiku-4.5

Judge model is left configurable via --judge so users can pick a different
referee. The recommended default for this preset is
`openrouter:google/gemini-3-flash-preview`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .goldset import AML_REPURPOSING_PAPER_TOP3, GoldSet
from .runner import BenchCandidate


@dataclass(frozen=True)
class BenchPreset:
    name: str
    description: str
    candidates: tuple[BenchCandidate, ...]
    suggested_judge: str        # "provider:model"
    # Optional preset defaults — let the CLI invoke a preset without forcing
    # the user to retype the goal or attach a gold set.
    default_goal: str | None = None
    goldset: GoldSet | None = None


_PAPER_CANDIDATES: tuple[BenchCandidate, ...] = (
    BenchCandidate(
        label="gemini-2-flash-thinking",
        provider="openrouter",
        model="google/gemini-2.0-flash-001",
    ),
    BenchCandidate(
        label="gemini-2-pro",
        provider="openrouter",
        model="google/gemini-2.5-pro",
    ),
    BenchCandidate(
        label="openai-o1",
        provider="openrouter",
        model="openai/o1",
    ),
    BenchCandidate(
        label="claude-haiku-4.5",
        provider="openrouter",
        model="anthropic/claude-haiku-4.5",
    ),
)


# The AML drug-repurposing goal mirrors the exact methodology in the
# source paper:
#   - Ranked list of repurposing candidates for AML.
#   - Candidates must NOT have been previously repurposed for AML.
#   - There must be NO prior preclinical evidence supporting the candidate
#     in AML.
#   - The system uses only its internal knowledge — no DepMap dependency
#     scores, no genomic priors, no human expert feedback.
#
# We additionally tell models to NAME the specific drug (INN, brand, or
# research code) rather than a drug class so the gold-set matcher can
# score recall against named entities.
_PAPER_AML_GOAL = (
    "Produce a ranked list of drug repurposing candidates for acute "
    "myeloid leukemia (AML), strictly under the following constraints:\n\n"
    "(1) Each candidate must NOT have prior published evidence of being "
    "repurposed for AML, and there must be no preclinical studies in AML "
    "for the proposed compound at the time of writing.\n"
    "(2) Use only your internal knowledge. Do NOT assume access to DepMap "
    "dependency scores, gene-essentiality datasets, transcriptomic "
    "screens, or human expert curation. No external inputs.\n"
    "(3) Name the specific compound (INN, brand name, or research-code "
    "alias) — do not propose generic drug classes (e.g. \"MEK inhibitors\")."
    " Cover diverse mechanisms across the ranked list (avoid 5 ideas all "
    "hitting the same pathway).\n\n"
    "For each candidate hypothesis, give: the named compound, the "
    "molecular mechanism by which it would act against AML blasts or "
    "leukemic stem cells (with the specific target/pathway), the "
    "scientific reasoning that licenses the AML hypothesis even though "
    "no AML-specific evidence currently exists, and one concrete in vitro "
    "or in vivo experiment that would falsify the hypothesis."
)


def _vs_raw(candidates: tuple[BenchCandidate, ...]) -> tuple[BenchCandidate, ...]:
    """Double every candidate: once in pipeline mode, once in direct mode.

    This is the apples-to-apples comparison of hypothesis-engine's multi-agent
    Generation harness against a raw single-shot LM call on the same goal.
    The label gets a `[pipe]` / `[raw]` suffix so the result table makes
    the distinction obvious.
    """
    out: list[BenchCandidate] = []
    for c in candidates:
        out.append(BenchCandidate(
            label=f"{c.label}[pipe]", provider=c.provider, model=c.model,
            mode="pipeline",
        ))
        out.append(BenchCandidate(
            label=f"{c.label}[raw]", provider=c.provider, model=c.model,
            mode="direct",
        ))
    return tuple(out)


# Current frontier set — what you'd actually want to use today. Picked to
# span pricing tiers so the bench surfaces $/quality tradeoffs, not just
# raw quality.
_FRONTIER_CANDIDATES: tuple[BenchCandidate, ...] = (
    BenchCandidate(
        label="claude-opus-4.7",
        provider="openrouter",
        # OpenRouter uses dots in the version suffix.
        model="anthropic/claude-opus-4.7",
    ),
    BenchCandidate(
        label="gpt-5",
        provider="openrouter",
        model="openai/gpt-5",
    ),
    BenchCandidate(
        label="gemini-3-pro",
        provider="openrouter",
        model="google/gemini-3.1-pro-preview",
    ),
    BenchCandidate(
        label="gemini-3-flash",
        provider="openrouter",
        model="google/gemini-3-flash-preview",
    ),
)


PRESETS: dict[str, BenchPreset] = {
    "paper": BenchPreset(
        name="paper",
        description=(
            "Reproduce the source paper's preference-ranking comparison "
            "(plus Haiku) using current OpenRouter models. See module docstring "
            "for the substitutions we had to make for retired experimental models."
        ),
        candidates=_PAPER_CANDIDATES,
        suggested_judge="openrouter:google/gemini-3-flash-preview",
    ),
    "paper-aml": BenchPreset(
        name="paper-aml",
        description=(
            "Reproduce the paper's AML repurposing benchmark under its "
            "strict methodology: candidates with NO prior repurposing "
            "evidence and NO preclinical evidence in AML, no external "
            "inputs (DepMap, expert feedback). Recall is scored against "
            "the top-3 candidates the paper surfaced: Nanvuranlat, KIRA6, "
            "and Leflunomide. Uses the same candidate set + judge as the "
            "`paper` preset."
        ),
        candidates=_PAPER_CANDIDATES,
        suggested_judge="openrouter:google/gemini-3-flash-preview",
        default_goal=_PAPER_AML_GOAL,
        goldset=AML_REPURPOSING_PAPER_TOP3,
    ),
    "paper-aml-vs-raw": BenchPreset(
        name="paper-aml-vs-raw",
        description=(
            "AML repurposing benchmark for the paper's baseline models, "
            "under the strict no-prior-evidence methodology — each model "
            "runs TWICE: once through the full hypothesis-engine Generation "
            "pipeline (literature tools + tool loop + dedup) and once as a "
            "single raw LM call. Isolates how much of the system's "
            "performance comes from the multi-agent harness vs the model "
            "itself. Same top-3 gold set as `paper-aml`."
        ),
        candidates=_vs_raw(_PAPER_CANDIDATES),
        suggested_judge="openrouter:google/gemini-3-flash-preview",
        default_goal=_PAPER_AML_GOAL,
        goldset=AML_REPURPOSING_PAPER_TOP3,
    ),
    "frontier-aml-vs-raw": BenchPreset(
        name="frontier-aml-vs-raw",
        description=(
            "Same setup as `paper-aml-vs-raw` but with current frontier "
            "models (Claude Opus 4.7, GPT-5, Gemini 3 Pro, Gemini 3 Flash). "
            "Lets you see whether the multi-agent harness still adds value "
            "with stronger base models."
        ),
        candidates=_vs_raw(_FRONTIER_CANDIDATES),
        suggested_judge="openrouter:google/gemini-3-flash-preview",
        default_goal=_PAPER_AML_GOAL,
        goldset=AML_REPURPOSING_PAPER_TOP3,
    ),
}


def get_preset(name: str) -> BenchPreset:
    try:
        return PRESETS[name]
    except KeyError as e:
        names = ", ".join(sorted(PRESETS))
        raise KeyError(f"unknown bench preset {name!r}; available: {names}") from e
