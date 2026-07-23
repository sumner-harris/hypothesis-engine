# Modified from the original work.
"""Tests for the supported embedders and FAISS store."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from hypothesis_engine.vectors.store import FaissStore


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype("float32")
    return vector / np.linalg.norm(vector)


@pytest.mark.asyncio
async def test_faiss_store_add_search_persist(tmp_cfg) -> None:
    store = FaissStore(tmp_cfg, "ses_v", dim=8)
    await store.load_or_create()
    assert store.n == 0

    first = await store.add("hyp_1", _vec(1))
    second = await store.add("hyp_2", _vec(2))
    assert (first, second) == (0, 1)
    assert store.n == 2

    results = await store.search(_vec(1), k=2)
    assert results[0][0] == "hyp_1"
    assert results[0][1] == pytest.approx(1.0, abs=1e-3)

    matrix = await store.cosine_matrix()
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0, abs=1e-3)

    await store.save()
    reopened = FaissStore(tmp_cfg, "ses_v", dim=8)
    await reopened.load_or_create()
    assert reopened.n == 2
    assert reopened.hypothesis_at(0) == "hyp_1"
    assert reopened.hypothesis_at(1) == "hyp_2"


@pytest.mark.asyncio
async def test_faiss_store_add_and_save_serializes_parallel_writers(tmp_cfg) -> None:
    async def add_one(index: int) -> int:
        store = FaissStore(tmp_cfg, "ses_parallel", dim=8)
        return await store.add_and_save(f"hyp_{index}", _vec(index + 1))

    offsets = await asyncio.gather(*(add_one(index) for index in range(12)))

    assert sorted(offsets) == list(range(12))
    reopened = FaissStore(tmp_cfg, "ses_parallel", dim=8)
    await reopened.load_or_create()
    assert reopened.n == 12
    assert {reopened.hypothesis_at(index) for index in range(12)} == {
        f"hyp_{index}" for index in range(12)
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


def test_embedding_config_supports_only_openai_and_compatible() -> None:
    from pydantic import ValidationError

    from hypothesis_engine.config import EmbeddingsCfg

    with pytest.raises(ValidationError, match="openai_compatible"):
        EmbeddingsCfg(provider="voyage")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="text-embedding-3-large"):
        EmbeddingsCfg(
            provider="openai",
            model="text-embedding-3-small",
            dim=1536,
        )


def test_openai_config_discards_inherited_compatible_base_url() -> None:
    from hypothesis_engine.config import EmbeddingsCfg

    config = EmbeddingsCfg(
        provider="openai",
        model="text-embedding-3-large",
        dim=3072,
        base_url="http://localhost:8001/v1",
    )

    assert config.base_url is None


def test_openai_embedder_requires_api_key(monkeypatch) -> None:
    from hypothesis_engine.config import Config, EmbeddingsCfg
    from hypothesis_engine.vectors.embedder import (
        NoEmbeddingsAvailable,
        make_embedder,
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = Config(
        embeddings=EmbeddingsCfg(
            provider="openai",
            model="text-embedding-3-large",
            dim=3072,
        )
    )
    config.secrets.OPENAI_API_KEY = ""

    with pytest.raises(NoEmbeddingsAvailable, match="OPENAI_API_KEY"):
        make_embedder(config)


@pytest.mark.asyncio
async def test_openai_embedder_uses_only_text_embedding_3_large(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    from hypothesis_engine.config import Config, EmbeddingsCfg
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
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=_Client))

    config = Config(
        embeddings=EmbeddingsCfg(
            provider="openai",
            model="text-embedding-3-large",
            dim=3,
        )
    )
    config.secrets.OPENAI_API_KEY = "sk-fake"

    embedder = make_embedder(config)
    assert isinstance(embedder, OpenAIEmbedder)
    vectors = await embedder.embed(["hello"])

    assert vectors.shape == (1, 3)
    assert seen["client"] == {"api_key": "sk-fake"}
    assert seen["request"] == {
        "model": "text-embedding-3-large",
        "input": ["hello"],
        "dimensions": 3,
    }


@pytest.mark.asyncio
async def test_openai_compatible_private_endpoint_is_keyless(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    from hypothesis_engine.config import Config, EmbeddingsCfg
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

    config = Config(
        embeddings=EmbeddingsCfg(
            provider="openai_compatible",
            model="sfr-embedding-mistral",
            dim=3,
            base_url="http://192.0.2.10:8001/v1",
        )
    )
    config.secrets.OPENAI_API_KEY = ""

    embedder = make_embedder(config)
    assert isinstance(embedder, OpenAIEmbedder)
    vectors = await embedder.embed(["hello"])

    assert vectors.shape == (1, 3)
    assert seen["client"] == {
        "api_key": "compat-no-key",
        "base_url": "http://192.0.2.10:8001/v1",
    }
    assert seen["request"] == {
        "model": "sfr-embedding-mistral",
        "input": ["hello"],
    }
