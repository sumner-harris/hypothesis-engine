"""Shared, persisted projection and clustering for a session RAG knowledge base."""

from __future__ import annotations

import hashlib
import json
import pickle
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ..config import Config

try:  # pragma: no cover - exercised where FAISS is installed.
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


PROJECTION_VERSION = 1
PCA_COMPONENTS = 50
CLUSTER_MIN = 4
CLUSTER_MAX = 12
CLUSTER_STABILITY_SEEDS = (29, 47)
SILHOUETTE_SAMPLE_SIZE = 2_000
RANDOM_STATE = 13

_ARRAYS_FILE = "kb.projection.npz"
_META_FILE = "kb.projection.json"


def load_or_build_rag_projection(cfg: Config, session_id: str) -> dict[str, Any]:
    """Return the canonical all-chunk KB projection, rebuilding stale artifacts."""

    rag_dir = cfg.session_rag_dir(session_id)
    index_path = rag_dir / "kb.index"
    metadata_path = rag_dir / "kb.pkl"
    arrays_path = rag_dir / _ARRAYS_FILE
    projection_meta_path = rag_dir / _META_FILE
    if faiss is None:
        return _unavailable("FAISS is not installed")
    if not index_path.is_file() or not metadata_path.is_file():
        return _unavailable("missing RAG KB index")

    source_files = _source_file_stats(index_path, metadata_path)
    cached_meta = _read_json(projection_meta_path)
    if _cache_matches(cached_meta, source_files) and arrays_path.is_file():
        try:
            return _load_projection(arrays_path, cached_meta)
        except Exception:
            pass

    try:
        with metadata_path.open("rb") as handle:
            metadata = pickle.load(handle)
        if not isinstance(metadata, list):
            metadata = []
        index = faiss.read_index(str(index_path))
        count = min(int(index.ntotal), len(metadata))
        if count < 2:
            return _unavailable("need at least two RAG KB chunks")
        vectors = np.vstack([index.reconstruct(i) for i in range(count)]).astype("float32")
        vectors = normalize_rows(vectors)
        fitted = fit_rag_projection(vectors)
        source_sha256 = _combined_sha256(index_path, metadata_path)
        meta = {
            "version": PROJECTION_VERSION,
            "source_files": source_files,
            "source_sha256": source_sha256,
            "count": count,
            "dim": int(vectors.shape[1]),
            "pca_components": int(fitted["components"].shape[0]),
            "explained_variance_ratio": [
                float(value) for value in fitted["explained_variance_ratio"]
            ],
            "cluster_count": len(set(fitted["labels"].tolist())),
            "selected_k": int(fitted["selected_k"]),
            "silhouette": _finite_or_none(fitted["silhouette"]),
            "stability": _finite_or_none(fitted["stability"]),
            "selection_score": _finite_or_none(fitted["selection_score"]),
            "cluster_candidates": fitted["cluster_candidates"],
            "random_state": RANDOM_STATE,
            "cluster_space": "l2_normalized_pca",
            "pca_whiten": False,
            "sign_canonicalization": "largest_absolute_loading_positive",
        }
        _write_projection(arrays_path, projection_meta_path, fitted, meta)
        return {**fitted, "available": True, "meta": meta}
    except Exception as exc:
        return _unavailable(str(exc))


def fit_rag_projection(vectors: np.ndarray) -> dict[str, Any]:
    """Fit canonical PCA and stable KMeans assignments to normalized KB vectors."""

    normalized = normalize_rows(vectors)
    if normalized.ndim != 2 or len(normalized) < 2:
        raise ValueError("need at least two two-dimensional vectors")
    count, dim = normalized.shape
    component_count = min(PCA_COMPONENTS, dim, count - 1)
    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=RANDOM_STATE,
        whiten=False,
    )
    reduced = pca.fit_transform(normalized).astype("float32")
    components = np.asarray(pca.components_, dtype="float32").copy()
    _canonicalize_component_signs(components, reduced)
    cluster_vectors = normalize_rows(reduced)
    clustering = _select_clustering(cluster_vectors)
    labels, centers = _canonicalize_cluster_ids(
        np.asarray(clustering["labels"], dtype="int32"),
        np.asarray(clustering["centers"], dtype="float32"),
    )
    return {
        "coordinates": _pad_two_dimensions(reduced),
        "reduced": reduced,
        "cluster_vectors": cluster_vectors,
        "labels": labels,
        "centers": centers,
        "mean": np.asarray(pca.mean_, dtype="float32"),
        "components": components,
        "explained_variance_ratio": np.asarray(
            pca.explained_variance_ratio_, dtype="float32"
        ),
        "selected_k": int(clustering["selected_k"]),
        "silhouette": clustering["silhouette"],
        "stability": clustering["stability"],
        "selection_score": clustering["selection_score"],
        "cluster_candidates": clustering["cluster_candidates"],
    }


def transform_rag_vectors(projection: dict[str, Any], vectors: Any) -> dict[str, np.ndarray]:
    """Project new vectors and assign them to the persisted KB clusters."""

    normalized = normalize_rows(np.asarray(vectors, dtype="float32"))
    components = np.asarray(projection["components"], dtype="float32")
    mean = np.asarray(projection["mean"], dtype="float32")
    if normalized.ndim != 2 or normalized.shape[1] != components.shape[1]:
        raise ValueError(
            f"projection dimension mismatch: vectors={getattr(normalized, 'shape', None)} "
            f"components={components.shape}"
        )
    reduced = ((normalized - mean) @ components.T).astype("float32")
    cluster_vectors = normalize_rows(reduced)
    centers = np.asarray(projection["centers"], dtype="float32")
    if len(centers):
        distances = (
            np.sum(cluster_vectors * cluster_vectors, axis=1, keepdims=True)
            - 2.0 * (cluster_vectors @ centers.T)
            + np.sum(centers * centers, axis=1)[None, :]
        )
        labels = np.argmin(distances, axis=1).astype("int32")
    else:
        labels = np.zeros((len(cluster_vectors),), dtype="int32")
    return {
        "coordinates": _pad_two_dimensions(reduced),
        "reduced": reduced,
        "cluster_vectors": cluster_vectors,
        "labels": labels,
    }


def deterministic_sample_indices(count: int, max_points: int) -> np.ndarray:
    """Choose stable rendering/analysis rows without changing the fitted projection."""

    sample_count = min(max(0, int(max_points)), max(0, int(count)))
    if sample_count >= count:
        return np.arange(count, dtype="int64")
    if sample_count <= 0:
        return np.zeros((0,), dtype="int64")
    rng = np.random.default_rng(RANDOM_STATE)
    return np.sort(rng.choice(count, size=sample_count, replace=False)).astype("int64")


def load_rag_metadata(cfg: Config, session_id: str, *, count: int | None = None) -> list[Any]:
    """Load the local metadata rows corresponding to the FAISS vector order."""

    path = cfg.session_rag_dir(session_id) / "kb.pkl"
    try:
        with path.open("rb") as handle:
            metadata = pickle.load(handle)
    except Exception:
        return []
    if not isinstance(metadata, list):
        return []
    return metadata[:count] if count is not None else metadata


def normalize_rows(vectors: Any) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim != 2 or arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _select_clustering(vectors: np.ndarray) -> dict[str, Any]:
    count = len(vectors)
    if count < 3:
        return {
            "labels": np.zeros((count,), dtype="int32"),
            "centers": np.mean(vectors, axis=0, keepdims=True).astype("float32"),
            "selected_k": 1,
            "silhouette": None,
            "stability": 1.0,
            "selection_score": None,
            "cluster_candidates": [],
        }

    maximum = min(CLUSTER_MAX, count - 1)
    minimum = CLUSTER_MIN if maximum >= CLUSTER_MIN else 2
    candidates: list[dict[str, Any]] = []
    fitted: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(minimum, maximum + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(vectors).astype("int32")
        silhouette = float(
            silhouette_score(
                vectors,
                labels,
                sample_size=min(SILHOUETTE_SAMPLE_SIZE, count),
                random_state=RANDOM_STATE,
            )
        )
        stability_values = []
        for seed in CLUSTER_STABILITY_SEEDS:
            alternate = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(vectors)
            stability_values.append(float(adjusted_rand_score(labels, alternate)))
        stability = float(np.mean(stability_values)) if stability_values else 1.0
        selection_score = 0.7 * silhouette + 0.3 * stability
        candidates.append(
            {
                "k": k,
                "silhouette": silhouette,
                "stability": stability,
                "selection_score": selection_score,
            }
        )
        fitted[k] = (labels, np.asarray(model.cluster_centers_, dtype="float32"))

    best = max(
        candidates,
        key=lambda row: (float(row["selection_score"]), float(row["silhouette"]), -int(row["k"])),
    )
    labels, centers = fitted[int(best["k"])]
    return {
        "labels": labels,
        "centers": centers,
        "selected_k": int(best["k"]),
        "silhouette": float(best["silhouette"]),
        "stability": float(best["stability"]),
        "selection_score": float(best["selection_score"]),
        "cluster_candidates": candidates,
    }


def _canonicalize_component_signs(components: np.ndarray, scores: np.ndarray) -> None:
    for index, component in enumerate(components):
        anchor = int(np.argmax(np.abs(component)))
        if component[anchor] < 0:
            components[index] *= -1.0
            scores[:, index] *= -1.0


def _canonicalize_cluster_ids(
    labels: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(centers) <= 1:
        return labels, centers
    order = sorted(
        range(len(centers)),
        key=lambda index: tuple(float(value) for value in centers[index, : min(6, centers.shape[1])]),
    )
    remap = {old: new for new, old in enumerate(order)}
    canonical_labels = np.asarray([remap[int(label)] for label in labels], dtype="int32")
    canonical_centers = np.vstack([centers[index] for index in order]).astype("float32")
    return canonical_labels, canonical_centers


def _pad_two_dimensions(values: np.ndarray) -> np.ndarray:
    if values.shape[1] >= 2:
        return np.asarray(values[:, :2], dtype="float32")
    return np.column_stack(
        [values[:, 0], np.zeros((len(values),), dtype="float32")]
    ).astype("float32")


def _source_file_stats(index_path: Path, metadata_path: Path) -> dict[str, Any]:
    return {
        "index": _file_stat(index_path),
        "metadata": _file_stat(metadata_path),
    }


def _file_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _combined_sha256(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _cache_matches(meta: dict[str, Any] | None, source_files: dict[str, Any]) -> bool:
    return bool(
        isinstance(meta, dict)
        and meta.get("version") == PROJECTION_VERSION
        and meta.get("source_files") == source_files
        and meta.get("pca_components_requested", PCA_COMPONENTS) == PCA_COMPONENTS
    )


def _load_projection(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as arrays:
        result = {
            "coordinates": arrays["coordinates"].astype("float32"),
            "reduced": arrays["reduced"].astype("float32"),
            "cluster_vectors": arrays["cluster_vectors"].astype("float32"),
            "labels": arrays["labels"].astype("int32"),
            "centers": arrays["centers"].astype("float32"),
            "mean": arrays["mean"].astype("float32"),
            "components": arrays["components"].astype("float32"),
            "explained_variance_ratio": arrays["explained_variance_ratio"].astype("float32"),
        }
    count = int(meta.get("count") or 0)
    if len(result["coordinates"]) != count or len(result["labels"]) != count:
        raise ValueError("stale or malformed RAG projection artifact")
    return {
        **result,
        "available": True,
        "selected_k": int(meta.get("selected_k") or meta.get("cluster_count") or 1),
        "silhouette": meta.get("silhouette"),
        "stability": meta.get("stability"),
        "selection_score": meta.get("selection_score"),
        "cluster_candidates": list(meta.get("cluster_candidates") or []),
        "meta": meta,
    }


def _write_projection(
    arrays_path: Path,
    meta_path: Path,
    fitted: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    arrays_tmp = arrays_path.with_name(f".{arrays_path.name}.{token}.tmp")
    meta_tmp = meta_path.with_name(f".{meta_path.name}.{token}.tmp")
    try:
        with arrays_tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                coordinates=fitted["coordinates"],
                reduced=fitted["reduced"],
                cluster_vectors=fitted["cluster_vectors"],
                labels=fitted["labels"],
                centers=fitted["centers"],
                mean=fitted["mean"],
                components=fitted["components"],
                explained_variance_ratio=fitted["explained_variance_ratio"],
            )
        meta["pca_components_requested"] = PCA_COMPONENTS
        meta_tmp.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        arrays_tmp.replace(arrays_path)
        meta_tmp.replace(meta_path)
    finally:
        arrays_tmp.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return decoded if isinstance(decoded, dict) else None


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _unavailable(error: str) -> dict[str, Any]:
    return {
        "available": False,
        "error": error,
        "coordinates": np.zeros((0, 2), dtype="float32"),
        "labels": np.zeros((0,), dtype="int32"),
        "meta": {},
    }
