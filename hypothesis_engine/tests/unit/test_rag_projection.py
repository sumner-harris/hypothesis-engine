from __future__ import annotations

import pickle

import faiss
import numpy as np

from hypothesis_engine.analysis.rag_projection import (
    load_or_build_rag_projection,
    transform_rag_vectors,
)
from hypothesis_engine.analysis.session_report import (
    _analyze_kb_vectors,
    _load_kb_vectors,
    _Paths,
)
from hypothesis_engine.web.app import _rag_cluster_plot_payload


def _write_kb(tmp_cfg, session_id: str, vectors: np.ndarray) -> np.ndarray:
    normalized = vectors.astype("float32")
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    rag_dir = tmp_cfg.session_rag_dir(session_id)
    rag_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    faiss.write_index(index, str(rag_dir / "kb.index"))
    metadata = [
        {
            "source": f"paper-{index // 3}",
            "chunk_id": f"chunk-{index}",
            "text": f"topic material chunk {index}",
        }
        for index in range(len(normalized))
    ]
    with (rag_dir / "kb.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)
    return normalized


def _clustered_vectors(*, count: int = 36, dim: int = 12) -> np.ndarray:
    rng = np.random.default_rng(7)
    centers = np.eye(3, dim, dtype="float32")
    rows = []
    for index in range(count):
        rows.append(centers[index % 3] + rng.normal(0, 0.04, size=dim))
    return np.asarray(rows, dtype="float32")


def test_shared_projection_is_persisted_and_transforms_identically(tmp_cfg) -> None:
    session_id = "ses_projection"
    vectors = _write_kb(tmp_cfg, session_id, _clustered_vectors())

    first = load_or_build_rag_projection(tmp_cfg, session_id)
    second = load_or_build_rag_projection(tmp_cfg, session_id)
    transformed = transform_rag_vectors(first, vectors)

    assert first["available"] is True
    assert (tmp_cfg.session_rag_dir(session_id) / "kb.projection.npz").is_file()
    assert (tmp_cfg.session_rag_dir(session_id) / "kb.projection.json").is_file()
    np.testing.assert_allclose(second["coordinates"], first["coordinates"])
    np.testing.assert_allclose(transformed["coordinates"], first["coordinates"], atol=1e-5)
    np.testing.assert_array_equal(transformed["labels"], first["labels"])
    for component in first["components"]:
        anchor = int(np.argmax(np.abs(component)))
        assert component[anchor] >= 0


def test_projection_cache_invalidates_when_kb_changes(tmp_cfg) -> None:
    session_id = "ses_projection_refresh"
    vectors = _write_kb(tmp_cfg, session_id, _clustered_vectors(count=24))
    first = load_or_build_rag_projection(tmp_cfg, session_id)

    changed = np.vstack([vectors, np.eye(1, vectors.shape[1], dtype="float32")])
    _write_kb(tmp_cfg, session_id, changed)
    refreshed = load_or_build_rag_projection(tmp_cfg, session_id)

    assert len(first["coordinates"]) == 24
    assert len(refreshed["coordinates"]) == 25
    assert first["meta"]["source_sha256"] != refreshed["meta"]["source_sha256"]


def test_web_and_report_use_the_same_saved_kb_coordinates(
    tmp_cfg, tmp_path, monkeypatch
) -> None:
    session_id = "ses_projection_shared"
    _write_kb(tmp_cfg, session_id, _clustered_vectors())
    projection = load_or_build_rag_projection(tmp_cfg, session_id)

    monkeypatch.setattr(
        "hypothesis_engine.analysis.session_report._summarize_kb_clusters_with_llm",
        lambda *args, **kwargs: {},
    )
    kb = _load_kb_vectors(tmp_cfg, session_id, max_points=5_000)
    report = _analyze_kb_vectors(
        tmp_cfg,
        kb,
        _Paths(output=tmp_path, figures=tmp_path, tables=tmp_path),
        n_clusters=8,
        research_goal="test",
        projection=projection,
    )
    web = _rag_cluster_plot_payload(tmp_cfg, session_id, [])

    report_points = {int(row["kb_sample_index"]): row for row in report["pca_rows"]}
    for point in web["points"]:
        index = int(str(point["id"]).removeprefix("kb_"))
        row = report_points[index]
        assert point["x"] == row["x"]
        assert point["y"] == row["y"]
        assert point["cluster"] == row["cluster"]
