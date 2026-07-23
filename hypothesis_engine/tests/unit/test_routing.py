# Modified from the original work.
"""Tests for model routing + downgrade chain + cost estimation."""

from __future__ import annotations

from hypothesis_engine.config import Config
from hypothesis_engine.llm.routing import (
    NEVER_DEGRADE,
    estimate_cost_usd,
    route,
    thinking_budget_for,
)


def test_default_routes_use_opus_for_heavy_modes() -> None:
    cfg = Config()
    assert route(cfg, "generation", "literature").model == cfg.models.generation
    assert route(cfg, "reflection", "verification").model == cfg.models.reflection
    assert route(cfg, "metareview", "final").model == cfg.models.metareview_final


def test_thinking_only_on_opus() -> None:
    cfg = Config()
    r = route(cfg, "reflection", "verification")
    if r.model.startswith("claude-opus"):
        assert r.thinking_tokens == cfg.thinking.reflection_verification
    else:
        assert r.thinking_tokens == 0


def test_openai_compatible_route_retains_thinking_budget() -> None:
    cfg = Config()
    cfg.llm.provider = "openai_compatible"
    cfg.models.reflection = "gemma-4-31b-it-nvfp4"
    r = route(cfg, "reflection", "verification")
    assert r.thinking_tokens == cfg.thinking.reflection_verification


def test_degrade_walks_chain_once() -> None:
    cfg = Config()
    r1 = route(cfg, "generation", "literature", degraded=False)
    r2 = route(cfg, "generation", "literature", degraded=True)
    # Should be different unless never-degrade
    assert "generation.literature" not in NEVER_DEGRADE
    assert r1.model != r2.model


def test_never_degrade_modes_stay_put() -> None:
    cfg = Config()
    r1 = route(cfg, "reflection", "verification", degraded=False)
    r2 = route(cfg, "reflection", "verification", degraded=True)
    assert r1.model == r2.model


def test_thinking_budget_lookup() -> None:
    cfg = Config()
    assert (
        thinking_budget_for(cfg, "reflection.verification") == cfg.thinking.reflection_verification
    )
    assert thinking_budget_for(cfg, "ranking.pairwise") == cfg.thinking.ranking_pairwise
    assert thinking_budget_for(cfg, "made_up.mode") == 0


def test_cache_reads_are_cheaper_than_uncached_input() -> None:
    """Same total 10k-token context: all-uncached is more expensive than
    mostly-cached (2k uncached + 8k cache reads)."""
    all_uncached = estimate_cost_usd(
        model="claude-opus-4-7", input_tokens=10_000, output_tokens=1_000
    )
    mostly_cached = estimate_cost_usd(
        model="claude-opus-4-7",
        input_tokens=2_000,
        output_tokens=1_000,
        cache_read=8_000,
        cache_write=0,
    )
    assert mostly_cached < all_uncached


def test_unknown_flash_model_prices_as_flash_not_sonnet() -> None:
    """Brand-new gemini-3-flash-preview must NOT be priced at the conservative
    sonnet-class fallback (otherwise the pre-flight estimator over-reports by
    10x and the budget admission rejects work that should be affordable).
    """
    flash_cost = estimate_cost_usd(
        model="google/gemini-3-flash-preview",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    # flash-tier output is ~$2.5/M, so 1M+1M tokens ≈ $2.80.
    # sonnet fallback would be ~$18.
    assert flash_cost < 5.0


def test_unknown_mini_model_uses_mini_tier_pricing() -> None:
    cost = estimate_cost_usd(
        model="some-provider/gpt-7-mini-preview",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    # mini tier: 1.1 input + 4.4 output per million ≈ $5.5
    # gpt-5 tier would be ~$25, sonnet ~$18
    assert cost < 10.0


def test_unknown_opus_class_model_uses_opus_pricing() -> None:
    """Family hints map upward as well: an unknown opus-named model should
    NOT silently price as a sonnet (and risk underbudgeting)."""
    cost = estimate_cost_usd(
        model="anthropic/claude-99-opus-experimental",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    # opus tier: 15 + 75 = $90/M+M
    assert cost > 50.0


def test_known_model_takes_precedence_over_family_hint() -> None:
    """`gemini-3-flash-preview` is in the table; the family hint should not
    override the explicit entry."""
    flash_known = estimate_cost_usd(
        model="gemini-3-flash-preview",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert 2.5 < flash_known < 3.5


def test_completely_unknown_model_uses_conservative_fallback() -> None:
    """No family hint match → fall back to sonnet-class so a misconfigured
    route doesn't accidentally run unbounded on a cheap fallback."""
    cost = estimate_cost_usd(
        model="totally-novel-vendor/unrecognized-model-7",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    # Sonnet-class fallback: 3 + 15 = $18/M+M
    assert 15.0 < cost < 25.0


def test_local_vllm_model_prices_as_zero() -> None:
    cost = estimate_cost_usd(
        model="gemma-4-26b-a4b-nvfp4",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == 0.0
