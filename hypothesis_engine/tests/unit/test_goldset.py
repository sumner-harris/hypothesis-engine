# Modified from the original work.
"""Tests for the bench gold-set matcher.

The matcher is the substantive part — it has to handle aliases, hyphens,
multi-word names, research-code lookups (JPH-203 → Nanvuranlat), and not
over-match (a generic "DHODH inhibitor" mention without naming Leflunomide
must not count).
"""

from __future__ import annotations

from hypothesis_engine.bench.goldset import (
    AML_REPURPOSING_PAPER_5,
    AML_REPURPOSING_PAPER_TOP3,
    GOLDSETS,
    GoldEntity,
    GoldSet,
    _contains_subseq,
    _tokens,
    score_candidate_against_goldset,
    score_hypothesis_against_goldset,
)


def _hyp(**kwargs) -> dict:
    """Build a hypothesis-record dict with the bench-relevant fields."""
    return {
        "id": "hyp_t",
        "title": kwargs.pop("title", ""),
        "summary": kwargs.pop("summary", ""),
        "full_text": kwargs.pop("full_text", ""),
        "entities": kwargs.pop("entities", []),
        "citations": kwargs.pop("citations", []),
    }


# ----------------------------- tokenization ----------------------------- #


def test_tokens_lowercases_and_splits_on_punctuation() -> None:
    assert _tokens("JPH-203 (Nanvuranlat)") == ["jph", "203", "nanvuranlat"]


def test_tokens_handles_unicode_normalization() -> None:
    # NFKD normalization — for accented Latin chars, decompose.
    assert _tokens("Café") == ["cafe"]


def test_contains_subseq_requires_contiguous_match() -> None:
    h = ["the", "gut", "microbiome", "drives", "inflammation"]
    assert _contains_subseq(h, ["gut", "microbiome"])
    assert not _contains_subseq(h, ["microbiome", "gut"])
    assert not _contains_subseq(h, ["gut", "drives"])
    assert not _contains_subseq(h, [])


# ----------------------------- canonical hits ----------------------------- #


def test_canonical_name_in_title_hits() -> None:
    h = _hyp(title="Repurposing Leflunomide for AML",
             summary="DHODH inhibition impairs blast metabolism.")
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_TOP3)
    assert any(r.entity == "Leflunomide" for r in hits)


def test_alias_in_text_hits_to_canonical_name() -> None:
    """JPH-203 should resolve to Nanvuranlat."""
    h = _hyp(full_text="JPH-203 selectively inhibits LAT1/SLC7A5.")
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_TOP3)
    assert len(hits) == 1
    assert hits[0].entity == "Nanvuranlat"
    assert hits[0].matched_alias == "JPH-203"


def test_research_compound_kira6_matches_only_at_word_boundary() -> None:
    """KIRA6 has no aliases — only the literal name. Must match standalone."""
    h_yes = _hyp(full_text="KIRA6 attenuates the IRE1-alpha RNase activity.")
    hits = score_hypothesis_against_goldset(h_yes, AML_REPURPOSING_PAPER_TOP3)
    assert any(r.entity == "KIRA6" for r in hits)


def test_alias_in_entities_array_hits() -> None:
    h = _hyp(entities=["SLC7A5", "JPH203", "LAT1"])
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_TOP3)
    assert any(r.entity == "Nanvuranlat" for r in hits)


def test_alias_in_citation_excerpt_hits() -> None:
    h = _hyp(
        title="A repurposed DMARD",
        summary="An old DHODH inhibitor shows activity.",
        citations=[
            {"title": "Arava metabolism and pharmacokinetics",
             "url": "https://example.com/x",
             "excerpt": "Arava (leflunomide) reduces pyrimidine ..."},
        ],
    )
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_TOP3)
    assert any(r.entity == "Leflunomide" for r in hits)


def test_teriflunomide_counts_as_leflunomide() -> None:
    """Teriflunomide is the active metabolite of leflunomide — same DHODH
    mechanism, so we accept it as a hit."""
    h = _hyp(full_text="Teriflunomide (Aubagio) blocks DHODH in AML blasts.")
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_TOP3)
    assert any(r.entity == "Leflunomide" for r in hits)


# ----------------------------- non-matches ----------------------------- #


def test_class_label_alone_does_not_hit() -> None:
    """The matcher must NOT count generic class mentions like "DHODH
    inhibitors" as a hit for Leflunomide; the candidate has to name the
    actual compound."""
    h = _hyp(full_text="DHODH inhibitors block pyrimidine synthesis in AML.")
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_TOP3)
    assert hits == []


def test_partial_compound_name_does_not_hit() -> None:
    """'nanvuran' alone must not match 'Nanvuranlat'."""
    h = _hyp(full_text="The compound nanvuran-related analog ...")
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_TOP3)
    assert hits == []


def test_lat1_inhibitor_class_does_not_hit() -> None:
    """Same rule: naming the target (LAT1/SLC7A5) without naming
    Nanvuranlat or its aliases is NOT a hit."""
    h = _hyp(full_text="LAT1 transporters are essential for AML blasts.")
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_TOP3)
    assert hits == []


def test_token_boundary_prevents_false_positives() -> None:
    """'Leflunomide-resistant' must still match Leflunomide (whole-token
    `leflunomide` is present); but 'pseudoleflunomide' must not."""
    h_yes = _hyp(full_text="Leflunomide-resistant phenotype")
    h_no = _hyp(full_text="pseudoleflunomideX is a fictitious compound")
    assert any(r.entity == "Leflunomide"
               for r in score_hypothesis_against_goldset(h_yes, AML_REPURPOSING_PAPER_TOP3))
    assert score_hypothesis_against_goldset(h_no, AML_REPURPOSING_PAPER_TOP3) == []


def test_short_research_code_only_at_word_boundary() -> None:
    """`HWA-486` should match standalone but not embedded."""
    h_yes = _hyp(full_text="HWA-486 (the leflunomide development code)")
    h_no  = _hyp(full_text="some unrelated XHWA-486Y identifier")
    # Both will tokenize HWA / 486 — let's verify the positive case at least.
    hits_yes = score_hypothesis_against_goldset(h_yes, AML_REPURPOSING_PAPER_TOP3)
    assert any(r.entity == "Leflunomide" for r in hits_yes)
    # The negative is a tokenization edge case — "xhwa" / "486y" both
    # tokenize as separate runs that DON'T match `hwa 486`. Confirm.
    hits_no = score_hypothesis_against_goldset(h_no, AML_REPURPOSING_PAPER_TOP3)
    assert all(r.entity != "Leflunomide" for r in hits_no)


# ----------------------------- candidate aggregation ----------------------------- #


def test_score_candidate_dedups_across_hypotheses() -> None:
    """If two hypotheses both mention Leflunomide, the aggregate has it
    once — but with both per-hypothesis records."""
    hyps = [
        _hyp(title="h1", summary="Leflunomide for AML"),
        _hyp(title="h2", full_text="Leflunomide + venetoclax combo"),
    ]
    agg = score_candidate_against_goldset(hyps, AML_REPURPOSING_PAPER_TOP3)
    assert list(agg) == ["Leflunomide"]
    assert len(agg["Leflunomide"]) == 2


def test_score_candidate_full_recall_top3() -> None:
    """A single hypothesis-list that mentions all 3 picks scores 3/3."""
    hyps = [
        _hyp(full_text="Nanvuranlat targets LAT1."),
        _hyp(full_text="KIRA6 attenuates IRE1-alpha."),
        _hyp(full_text="Leflunomide / teriflunomide inhibits DHODH."),
    ]
    agg = score_candidate_against_goldset(hyps, AML_REPURPOSING_PAPER_TOP3)
    assert set(agg.keys()) == {"Nanvuranlat", "KIRA6", "Leflunomide"}


def test_empty_candidate_returns_empty() -> None:
    agg = score_candidate_against_goldset([], AML_REPURPOSING_PAPER_TOP3)
    assert agg == {}


# ----------------------------- gold-set metadata ----------------------------- #


def test_top3_goldset_has_exactly_three_entities() -> None:
    names = [e.name for e in AML_REPURPOSING_PAPER_TOP3.entities]
    assert names == ["Nanvuranlat", "KIRA6", "Leflunomide"]


def test_nanvuranlat_aliases_include_research_code() -> None:
    """JPH-203 was the development code in early SLC7A5 literature."""
    nan = next(e for e in AML_REPURPOSING_PAPER_TOP3.entities
               if e.name == "Nanvuranlat")
    assert "JPH-203" in nan.aliases or "JPH203" in nan.aliases


# ----------------------------- registry + broader 5-drug set ----------------------------- #


def test_goldsets_registry_contains_both_aml_sets() -> None:
    """We keep both vintages of the paper's AML result so historical bench
    artifacts that scored against the broader 5-drug set stay interpretable."""
    assert "aml-repurposing-paper-5" in GOLDSETS
    assert "aml-repurposing-paper-top3" in GOLDSETS
    assert GOLDSETS["aml-repurposing-paper-5"] is AML_REPURPOSING_PAPER_5
    assert GOLDSETS["aml-repurposing-paper-top3"] is AML_REPURPOSING_PAPER_TOP3


def test_paper_5_contains_original_five_drugs() -> None:
    names = {e.name for e in AML_REPURPOSING_PAPER_5.entities}
    assert names == {
        "Binimetinib", "Pacritinib", "Cerivastatin", "Pravastatin",
        "Dimethyl fumarate",
    }


def test_paper_5_dmf_alias_resolves() -> None:
    """The broader set should still match DMF/BG-12/Tecfidera."""
    h = _hyp(full_text="BG-12 has been studied for relapsing MS.")
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_5)
    assert any(r.entity == "Dimethyl fumarate" for r in hits)


def test_paper_5_and_top3_have_disjoint_entities() -> None:
    """The two sets correspond to different methodologies in the paper, so
    none of the entries should overlap."""
    five = {e.name for e in AML_REPURPOSING_PAPER_5.entities}
    top3 = {e.name for e in AML_REPURPOSING_PAPER_TOP3.entities}
    assert five.isdisjoint(top3)


# ----------------------------- custom gold sets ----------------------------- #


def test_custom_gold_set_with_aliases() -> None:
    gs = GoldSet(
        label="custom",
        description="test",
        entities=[
            GoldEntity(name="Quercetin", aliases=("3,3',4',5,7-pentahydroxyflavone",)),
        ],
    )
    h = _hyp(full_text="dietary quercetin shows anti-inflammatory activity")
    hits = score_hypothesis_against_goldset(h, gs)
    assert len(hits) == 1
    assert hits[0].entity == "Quercetin"
