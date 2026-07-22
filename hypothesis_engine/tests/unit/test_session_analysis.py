from __future__ import annotations

import numpy as np

from hypothesis_engine.analysis.session_report import (
    _analyze_elo,
    _analyze_source_use,
    _density_cluster_rows,
    _source_aliases,
    _source_record,
    _summarize_debates,
    _summarize_kb_clusters_with_llm,
)
from hypothesis_engine.config import Config


def test_source_use_matches_urls_ids_and_titles() -> None:
    record = _source_record(
        {
            "title": "Substrate dependent pore formation in molybdenum disulfide monolayers",
            "pdf_url": "https://arxiv.org/pdf/2603.24416v1",
            "arxiv_id": "2603.24416v1",
        },
        "search",
        "arxiv_search",
    )
    record["aliases"] = sorted(_source_aliases(record))
    sources = {
        "records": {record["key"]: record},
        "paper_artifact_count": 0,
    }
    hypotheses = [
        {
            "id": "hyp_url",
            "title": "URL cite",
            "summary": "",
            "full_text": "This mechanism follows https://arxiv.org/pdf/2603.24416v1.",
        },
        {
            "id": "hyp_title",
            "title": "Title cite",
            "summary": "",
            "full_text": "Consistent with substrate dependent pore formation in molybdenum disulfide monolayers.",
        },
    ]

    out = _analyze_source_use(hypotheses, sources)

    assert out["metrics"]["hypotheses_with_known_source_matches"] == 2
    assert out["metrics"]["known_sources_explicitly_matched_in_hypotheses"] == 1


def test_density_cluster_rows_include_llm_ready_examples() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    labels = np.asarray([0, 0, -1])
    rows = [
        {"source": "paper-a", "chunk_id": "a0"},
        {"source": "paper-a", "chunk_id": "a1"},
        {"source": "paper-b", "chunk_id": "b0"},
    ]
    texts = [
        "ion implantation defect formation in monolayer TMD materials",
        "implantation induced defect migration in monolayer TMDs",
        "unrelated spectroscopy background",
    ]

    out = _density_cluster_rows(
        "umap", labels, rows, texts=texts, sources=["paper-a", "paper-a", "paper-b"], label_key="source", vectors=vectors
    )
    by_cluster = {row["density_cluster"]: row for row in out}

    assert by_cluster[0]["deterministic_label"]
    assert by_cluster[0]["examples"][0]["source"] == "paper-a"
    assert "implant" in by_cluster[0]["top_terms"]
    assert by_cluster[-1]["deterministic_label"] == "HDBSCAN noise / sparse chunks"


def test_hdbscan_noise_row_is_not_labeled_as_topic() -> None:
    summaries = _summarize_kb_clusters_with_llm(
        Config(),
        "ion implantation in monolayer Janus TMDs",
        [
            {
                "density_cluster": -1,
                "is_noise": True,
                "size": 12,
                "unique_documents": 8,
                "top_terms": "implant, defect, tmd",
            }
        ],
        cluster_key="density_cluster",
        chunks_key="size",
        cluster_kind="UMAP HDBSCAN cluster",
        skip_negative_ids=True,
    )

    assert summaries[-1]["label"] == "HDBSCAN noise / sparse chunks"
    assert summaries[-1]["status"] == "hdbscan_noise_not_labeled"
    assert "not be interpreted as one semantic topic" in summaries[-1]["summary"]


def test_debate_summary_counts_upsets_from_pre_match_elo() -> None:
    matches = [
        {
            "mode": "debate",
            "winner": "a",
            "elo_a_before": 1100,
            "elo_b_before": 1250,
            "elo_a_after": 1120,
            "elo_b_after": 1230,
            "rationale": "lower rated won",
        },
        {
            "mode": "debate",
            "winner": "b",
            "elo_a_before": 1200,
            "elo_b_before": 1220,
            "elo_a_after": 1190,
            "elo_b_after": 1230,
            "rationale": "favorite won",
        },
    ]

    out = _summarize_debates(matches)

    assert out["metrics"]["mode_counts"] == {"debate": 2}
    assert out["metrics"]["upset_count"] == 1
    assert out["metrics"]["upset_rate"] == 0.5


def test_debate_summary_reports_position_and_verbosity_bias() -> None:
    hypotheses = [
        {"id": "h1", "title": "Long anchor", "summary": "", "full_text": "alpha " * 40},
        {"id": "h2", "title": "Short challenger", "summary": "", "full_text": "beta"},
        {"id": "h3", "title": "Other challenger", "summary": "", "full_text": "gamma"},
    ]
    matches = [
        {
            "id": "m1",
            "hyp_a": "h1",
            "hyp_b": "h2",
            "mode": "debate",
            "winner": "a",
            "elo_a_before": 1200,
            "elo_b_before": 1200,
            "elo_a_after": 1216,
            "elo_b_after": 1184,
            "rationale": "Hypothesis 1 has stronger support and a clearer mechanism. Hypothesis 2 is thinner.",
        },
        {
            "id": "m2",
            "hyp_a": "h3",
            "hyp_b": "h1",
            "mode": "debate",
            "winner": "a",
            "elo_a_before": 1200,
            "elo_b_before": 1200,
            "elo_a_after": 1216,
            "elo_b_after": 1184,
            "rationale": "Hypothesis 1 is verbose but indirect. Hypothesis 2 is better. H2 has the decisive test.",
        },
    ]

    out = _summarize_debates(matches, hypotheses=hypotheses, reviews=[])
    metrics = out["metrics"]

    assert metrics["storage_side_a_win_rate"] == 1.0
    assert metrics["prompt_position_1_win_rate"] == 0.5
    assert metrics["prompt_position_1_expected_probability_mean"] == 0.5
    assert metrics["longer_input_win_rate"] == 0.5
    assert metrics["more_discussed_win_rate"] == 1.0
    assert len(out["bias_rows"]) == 2
    assert out["bias_rows"][1]["prompt1_id"] == "h1"
    assert out["bias_rows"][1]["prompt1_side"] == "b"
    assert {row["metric"] for row in out["bias_summary_rows"]} >= {
        "prompt_position_1_win_rate",
        "longer_input_win_rate",
        "more_discussed_win_rate",
    }


def test_elo_snapshots_track_top_rank_and_match_counts() -> None:
    matches = [
        {
            "hyp_a": "h1",
            "hyp_b": "h2",
            "winner": "a",
            "elo_a_before": 1200,
            "elo_b_before": 1200,
            "elo_a_after": 1216,
            "elo_b_after": 1184,
        },
        {
            "hyp_a": "h3",
            "hyp_b": "h1",
            "winner": "b",
            "elo_a_before": 1200,
            "elo_b_before": 1216,
            "elo_a_after": 1185,
            "elo_b_after": 1231,
        },
    ]
    hypotheses = [
        {"id": "h1", "elo": 1231},
        {"id": "h2", "elo": 1184},
        {"id": "h3", "elo": 1185},
    ]

    out = _analyze_elo(matches, hypotheses, snapshot_every=1)

    assert out["metrics"]["final_top_elo"] == 1231
    assert out["metrics"]["match_count_max"] == 2
    assert out["snapshots"][-1]["top1_id"] == "h1"



def test_html_report_embeds_svg_and_bundle_contains_outputs(tmp_path) -> None:
    from zipfile import ZipFile

    from hypothesis_engine.analysis.session_report import _Paths, _write_bundle, _write_html_report

    paths = _Paths(output=tmp_path, figures=tmp_path / "figures", tables=tmp_path / "tables")
    paths.figures.mkdir()
    paths.tables.mkdir()
    (paths.figures / "plot.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="1" cy="1" r="1"/></svg>',
        encoding="utf-8",
    )
    (paths.tables / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (paths.output / "report.md").write_text("![Plot](figures/plot.svg)\n", encoding="utf-8")

    html_path = _write_html_report("![Plot](figures/plot.svg)\n", paths, session_id="ses_test")
    bundle_path = _write_bundle(paths, session_id="ses_test")

    html = html_path.read_text(encoding="utf-8")
    assert "<figure><svg" in html
    assert 'src="figures/plot.svg"' not in html
    with ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
    assert "hypothesis_engine_analysis_ses_test/report.html" in names
    assert "hypothesis_engine_analysis_ses_test/report.md" in names
    assert "hypothesis_engine_analysis_ses_test/figures/plot.svg" in names
    assert "hypothesis_engine_analysis_ses_test/tables/table.csv" in names
