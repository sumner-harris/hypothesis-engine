# Modified from the original work.
"""Tests for deterministic ID helpers and ULID prefixes."""

from __future__ import annotations

from hypothesis_engine import ids


def test_normalize_text_collapses_whitespace_and_lowercases() -> None:
    assert ids.normalize_text("  Hello\tWorld\n\n") == "hello world"


def test_hypothesis_id_is_deterministic_per_normalized_statement() -> None:
    a = ids.hypothesis_id("ses_1", "generation/literature", "  Hypothesis X involves Y. ")
    b = ids.hypothesis_id("ses_1", "generation/literature", "hypothesis x involves y.")
    assert a == b


def test_hypothesis_id_changes_when_session_or_origin_changes() -> None:
    a = ids.hypothesis_id("ses_1", "generation/literature", "Hypothesis X.")
    b = ids.hypothesis_id("ses_2", "generation/literature", "Hypothesis X.")
    c = ids.hypothesis_id("ses_1", "evolution/combine", "Hypothesis X.")
    assert a != b
    assert a != c


def test_match_id_is_order_independent() -> None:
    assert ids.match_id("hyp_a", "hyp_b", "r1") == ids.match_id("hyp_b", "hyp_a", "r1")


def test_match_id_changes_with_round() -> None:
    assert ids.match_id("a", "b", "r1") != ids.match_id("a", "b", "r2")


def test_ulid_ids_have_expected_prefixes() -> None:
    assert ids.session_id().startswith("ses_")
    assert ids.task_id().startswith("tsk_")
    assert ids.transcript_id().startswith("trn_")
    assert ids.feedback_id().startswith("fb_")
