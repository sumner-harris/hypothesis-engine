# Modified from the original work.
"""OpenAI and OpenAI-compatible embedding clients.

All clients return `np.ndarray` of shape (n, dim), L2-normalized so cosine
similarity equals inner product in the FAISS `IndexFlatIP` store.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np

from ..config import Config


class Embedder(Protocol):
    model: str
    dim: int

    async def embed(self, texts: list[str]) -> np.ndarray: ...


class NoEmbeddingsAvailable(RuntimeError):
    """Raised when the configured embedding endpoint cannot be used."""


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


class OpenAIEmbedder:
    """Client for direct OpenAI or a configured OpenAI-compatible endpoint."""

    def __init__(self, cfg: Config) -> None:
        provider = cfg.embeddings.provider.lower()
        if provider not in {"openai", "openai_compatible"}:
            raise ValueError(f"unknown embeddings provider: {provider}")
        if provider == "openai" and cfg.embeddings.model != "text-embedding-3-large":
            raise ValueError(
                'provider="openai" requires model="text-embedding-3-large"'
            )

        self.model = cfg.embeddings.model
        self.dim = cfg.embeddings.dim
        self._cfg = cfg
        self._compat_mode = provider == "openai_compatible"
        self._base_url = (
            cfg.embeddings.base_url or os.environ.get("OPENAI_EMBEDDINGS_BASE_URL")
            if self._compat_mode
            else None
        )

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")

        api_key = self._cfg.secrets.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        if self._compat_mode:
            if not self._base_url:
                raise NoEmbeddingsAvailable(
                    "openai_compatible embeddings require embeddings.base_url "
                    "or OPENAI_EMBEDDINGS_BASE_URL"
                )
            # The OpenAI SDK requires a non-empty value even when the selected
            # compatible endpoint does not authenticate.
            api_key = api_key or "compat-no-key"
        elif not api_key:
            raise NoEmbeddingsAvailable(
                "OPENAI_API_KEY is required for OpenAI embeddings"
            )

        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("the openai package is required for embeddings") from exc

        kwargs = {"api_key": api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        client = openai.AsyncOpenAI(**kwargs)

        out: list[list[float]] = []
        for start in range(0, len(texts), 256):
            batch = texts[start : start + 256]
            request: dict[str, object] = {"model": self.model, "input": batch}
            if not self._compat_mode:
                # OpenAI text-embedding-3 models support dimension truncation;
                # many compatible local servers do not accept this parameter.
                request["dimensions"] = self.dim
            response = await client.embeddings.create(**request)
            out.extend(item.embedding for item in response.data)

        vectors = np.asarray(out, dtype="float32")
        expected = (len(texts), self.dim)
        if vectors.shape != expected:
            raise RuntimeError(
                f"embedding endpoint returned shape {vectors.shape}; expected {expected}"
            )
        return _l2_normalize(vectors)


def make_embedder(cfg: Config) -> Embedder:
    """Construct one of the two supported embedding providers."""
    provider = cfg.embeddings.provider.lower()
    if provider == "openai":
        if not (cfg.secrets.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")):
            raise NoEmbeddingsAvailable(
                "OPENAI_API_KEY is required for OpenAI embeddings"
            )
        return OpenAIEmbedder(cfg)
    if provider == "openai_compatible":
        base_url = cfg.embeddings.base_url or os.environ.get(
            "OPENAI_EMBEDDINGS_BASE_URL"
        )
        if not base_url:
            raise NoEmbeddingsAvailable(
                "openai_compatible embeddings require embeddings.base_url "
                "or OPENAI_EMBEDDINGS_BASE_URL"
            )
        return OpenAIEmbedder(cfg)
    raise ValueError(f"unknown embeddings provider: {provider}")
