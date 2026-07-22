# Modified from the original work.
"""Tests for the FAISS store. Embedder is network-bound; we feed fake vectors."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from hypothesis_engine.vectors.store import FaissStore


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype("float32")
    return v / np.linalg.norm(v)


@pytest.mark.asyncio
async def test_faiss_store_add_search_persist(tmp_cfg) -> None:
    store = FaissStore(tmp_cfg, "ses_v", dim=8)
    await store.load_or_create()
    assert store.n == 0

    o1 = await store.add("hyp_1", _vec(1))
    o2 = await store.add("hyp_2", _vec(2))
    assert (o1, o2) == (0, 1)
    assert store.n == 2

    # k-NN should find itself first
    results = await store.search(_vec(1), k=2)
    assert results[0][0] == "hyp_1"
    assert results[0][1] == pytest.approx(1.0, abs=1e-3)

    # cosine matrix is 2x2 with 1s on diagonal
    m = await store.cosine_matrix()
    assert m.shape == (2, 2)
    assert m[0, 0] == pytest.approx(1.0, abs=1e-3)

    # Persist, then re-open
    await store.save()

    store2 = FaissStore(tmp_cfg, "ses_v", dim=8)
    await store2.load_or_create()
    assert store2.n == 2
    assert store2.hypothesis_at(0) == "hyp_1"
    assert store2.hypothesis_at(1) == "hyp_2"


@pytest.mark.asyncio
async def test_faiss_store_add_and_save_serializes_parallel_writers(tmp_cfg) -> None:
    async def add_one(i: int) -> int:
        store = FaissStore(tmp_cfg, "ses_parallel", dim=8)
        return await store.add_and_save(f"hyp_{i}", _vec(i + 1))

    offsets = await asyncio.gather(*(add_one(i) for i in range(12)))

    assert sorted(offsets) == list(range(12))
    reopened = FaissStore(tmp_cfg, "ses_parallel", dim=8)
    await reopened.load_or_create()
    assert reopened.n == 12
    assert {reopened.hypothesis_at(i) for i in range(12)} == {
        f"hyp_{i}" for i in range(12)
    }


@pytest.mark.asyncio
async def test_faiss_offset_lookup(tmp_cfg) -> None:
    store = FaissStore(tmp_cfg, "ses_v2", dim=4)
    await store.load_or_create()
    await store.add("a", _vec(1, 4))
    await store.add("b", _vec(2, 4))
    assert store.offset_of("a") == 0
    assert store.offset_of("b") == 1
    assert store.offset_of("missing") is None


# ----------------------------- embedder fallback ----------------------------- #


@pytest.mark.asyncio
async def test_make_embedder_falls_back_to_hash_when_no_keys() -> None:
    """Without VOYAGE_API_KEY or OPENAI_API_KEY, make_embedder should return
    HashEmbedder so dedup / proximity degrade rather than crash."""
    from hypothesis_engine.config import Config
    from hypothesis_engine.vectors.embedder import HashEmbedder, make_embedder

    cfg = Config()
    cfg.embeddings.provider = "voyage"
    cfg.secrets.VOYAGE_API_KEY = ""
    cfg.secrets.OPENAI_API_KEY = ""
    emb = make_embedder(cfg)
    assert isinstance(emb, HashEmbedder)


@pytest.mark.asyncio
async def test_hash_embedder_produces_normalized_unit_vectors() -> None:
    from hypothesis_engine.config import Config
    from hypothesis_engine.vectors.embedder import HashEmbedder

    cfg = Config()
    cfg.embeddings.dim = 128
    emb = HashEmbedder(cfg)
    vecs = await emb.embed(["microbiome inflammation hypothesis",
                            "tournament ranking hypothesis"])
    assert vecs.shape == (2, 128)
    # L2-normalized → ||v|| ≈ 1
    norms = np.linalg.norm(vecs, axis=1)
    assert all(abs(n - 1.0) < 1e-5 for n in norms)


@pytest.mark.asyncio
async def test_hash_embedder_similar_texts_have_higher_cosine() -> None:
    """The hash embedder is a bag-of-features stub, but near-duplicates of
    a text should still produce a higher cosine than unrelated text."""
    from hypothesis_engine.config import Config
    from hypothesis_engine.vectors.embedder import HashEmbedder

    cfg = Config()
    cfg.embeddings.dim = 1024
    emb = HashEmbedder(cfg)
    vecs = await emb.embed([
        "the gut microbiome drives chronic systemic inflammation",
        "the gut microbiome drives chronic systemic inflammation in humans",
        "quantum computing for solving prime factorization problems",
    ])
    sim_near = float(vecs[0] @ vecs[1])
    sim_far  = float(vecs[0] @ vecs[2])
    assert sim_near > sim_far


@pytest.mark.asyncio
async def test_make_embedder_prefers_openai_when_voyage_missing_but_openai_set() -> None:
    from hypothesis_engine.config import Config
    from hypothesis_engine.vectors.embedder import OpenAIEmbedder, make_embedder

    cfg = Config()
    cfg.embeddings.provider = "voyage"
    cfg.secrets.VOYAGE_API_KEY = ""
    cfg.secrets.OPENAI_API_KEY = "sk-fake"
    emb = make_embedder(cfg)
    assert isinstance(emb, OpenAIEmbedder)


@pytest.mark.asyncio
async def test_openai_compatible_embedder_uses_local_base_url_without_key(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    from hypothesis_engine.config import Config
    from hypothesis_engine.vectors.embedder import OpenAIEmbedder, make_embedder

    seen = {}

    class _Embeddings:
        async def create(self, **kwargs):
            seen["request"] = kwargs
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0, 0.0])])

    class _Client:
        def __init__(self, **kwargs):
            seen["client"] = kwargs
            self.embeddings = _Embeddings()

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDINGS_BASE_URL", raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=_Client))

    cfg = Config()
    cfg.embeddings.provider = "openai_compatible"
    cfg.embeddings.model = "local-embedding-model"
    cfg.embeddings.dim = 3
    cfg.embeddings.base_url = "http://localhost:8001/v1"
    cfg.secrets.OPENAI_API_KEY = ""

    emb = make_embedder(cfg)
    assert isinstance(emb, OpenAIEmbedder)
    vecs = await emb.embed(["hello"])

    assert vecs.shape == (1, 3)
    assert seen["client"] == {
        "api_key": "compat-no-key",
        "base_url": "http://localhost:8001/v1",
    }
    assert seen["request"] == {
        "model": "local-embedding-model",
        "input": ["hello"],
    }


def test_fallback_warning_emits_once_per_process() -> None:
    """Regression: ranking calls make_embedder() inside the pair-selection
    loop (potentially hundreds of times per session). The fallback warning
    must emit exactly once per process, not once per call.

    We probe the internal `_FALLBACK_WARNED` set rather than caplog because
    the project uses structlog, which doesn't always route through pytest's
    logging capture. The set is the source of truth for the once-per-process
    contract.
    """
    from hypothesis_engine.config import Config
    from hypothesis_engine.vectors import embedder as emb_mod

    emb_mod._reset_fallback_warned_for_tests()
    cfg = Config()
    cfg.embeddings.provider = "voyage"
    cfg.secrets.VOYAGE_API_KEY = ""
    cfg.secrets.OPENAI_API_KEY = ""

    for _ in range(50):
        emb_mod.make_embedder(cfg)

    # Exactly one warning marker recorded; subsequent calls hit the cache.
    assert {"no_embedding_key_using_hash_fallback"} == emb_mod._FALLBACK_WARNED
