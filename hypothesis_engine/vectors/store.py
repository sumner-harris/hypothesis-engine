# Modified from the original work.
"""Per-session FAISS index wrapper.

Uses `IndexFlatIP` over L2-normalized vectors → exact cosine similarity.
Tradeoff: O(N) search, fine for N < ~10k hypotheses per session. If a session
ever grows past that, swap to `IndexHNSWFlat` (changes the persisted index
format and requires rebuilding existing indexes).
"""

from __future__ import annotations

import asyncio
import json
import os
from uuid import uuid4

import faiss
import numpy as np

from ..config import Config
from ..file_lock import acquire_exclusive_file_lock, release_file_lock

_STORE_LOCKS: dict[str, asyncio.Lock] = {}


def _store_lock(directory: object) -> asyncio.Lock:
    key = str(directory)
    lock = _STORE_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _STORE_LOCKS[key] = lock
    return lock


class FaissStore:
    def __init__(self, cfg: Config, session_id: str, dim: int) -> None:
        self.dim = dim
        self.index: faiss.IndexFlatIP | None = None
        self._dir = cfg.session_vector_dir(session_id)
        self._index_path = self._dir / "index.faiss"
        self._meta_path = self._dir / "index.meta.json"
        self._ordered_ids: list[str] = []   # hypothesis_id at each faiss offset
        self._offset_by_id: dict[str, int] = {}  # mirror of _ordered_ids for O(1) offset_of()
        # FAISS itself is not thread-safe, and multiple agent workers create
        # separate FaissStore instances for the same session. Use one in-process
        # lock per session directory; write operations also take a file lock so
        # duplicate server processes do not race on index.faiss.
        self._lock = _store_lock(self._dir)
        self._file_lock_path = self._dir / ".store.lock"

    # ------------------------- lifecycle -------------------------------- #

    async def load_or_create(self) -> None:
        def _do() -> tuple[faiss.IndexFlatIP, list[str]]:
            return self._with_file_lock_sync(self._load_unlocked)

        async with self._lock:
            self.index, self._ordered_ids = await asyncio.to_thread(_do)
            self._offset_by_id = {hid: i for i, hid in enumerate(self._ordered_ids)}

    async def save(self) -> None:
        assert self.index is not None

        def _do() -> None:
            self._with_file_lock_sync(
                lambda: self._write_unlocked(self.index, self._ordered_ids)
            )

        async with self._lock:
            await asyncio.to_thread(_do)

    async def add_and_save(self, hypothesis_id: str, vec: np.ndarray) -> int:
        """Atomically load the latest index, append one vector, and persist it."""
        if vec.ndim == 1:
            vec = vec[None, :]

        def _do() -> tuple[faiss.IndexFlatIP, list[str], int]:
            def _locked() -> tuple[faiss.IndexFlatIP, list[str], int]:
                idx, ordered_ids = self._load_unlocked()
                offset = idx.ntotal
                idx.add(vec.astype("float32"))
                ordered_ids.append(hypothesis_id)
                self._write_unlocked(idx, ordered_ids)
                return idx, ordered_ids, offset

            return self._with_file_lock_sync(_locked)

        async with self._lock:
            self.index, self._ordered_ids, offset = await asyncio.to_thread(_do)
            self._offset_by_id = {hid: i for i, hid in enumerate(self._ordered_ids)}
            return offset

    def _with_file_lock_sync(self, fn):
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._file_lock_path.open("a+", encoding="utf-8") as lock_file:
            acquire_exclusive_file_lock(lock_file)
            try:
                return fn()
            finally:
                release_file_lock(lock_file)

    def _load_unlocked(self) -> tuple[faiss.IndexFlatIP, list[str]]:
        if self._index_path.exists() and self._meta_path.exists():
            idx = faiss.read_index(str(self._index_path))
            meta = json.loads(self._meta_path.read_text())
            return idx, list(meta.get("ordered_ids", []))
        return faiss.IndexFlatIP(self.dim), []

    def _write_unlocked(self, index: faiss.IndexFlatIP, ordered_ids: list[str]) -> None:
        # Unique temp names avoid parallel writers deleting each other's fixed
        # index.faiss.tmp before os.replace runs. The caller holds the file lock.
        suffix = f".{os.getpid()}.{uuid4().hex}.tmp"
        idx_tmp = self._index_path.with_name(f"{self._index_path.name}{suffix}")
        meta_tmp = self._meta_path.with_name(f"{self._meta_path.name}{suffix}")
        faiss.write_index(index, str(idx_tmp))
        meta_tmp.write_text(json.dumps({"dim": self.dim, "ordered_ids": ordered_ids}))
        os.replace(idx_tmp, self._index_path)
        os.replace(meta_tmp, self._meta_path)

    # ------------------------- ops -------------------------------------- #

    @property
    def n(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    async def add(self, hypothesis_id: str, vec: np.ndarray) -> int:
        """Append one vector. Returns its FAISS offset."""
        assert self.index is not None
        if vec.ndim == 1:
            vec = vec[None, :]

        def _do() -> int:
            off = self.index.ntotal
            self.index.add(vec.astype("float32"))
            return off

        async with self._lock:
            offset = await asyncio.to_thread(_do)
            self._ordered_ids.append(hypothesis_id)
            self._offset_by_id[hypothesis_id] = offset
        return offset

    async def search(
        self, query: np.ndarray, k: int = 5
    ) -> list[tuple[str, float]]:
        """Return [(hypothesis_id, cosine_sim)] best matches.

        Vectors are L2-normalized so inner product == cosine.
        """
        assert self.index is not None
        if self.n == 0:
            return []
        if query.ndim == 1:
            query = query[None, :]

        def _do(qk: int) -> tuple[np.ndarray, np.ndarray]:
            dists, idxs = self.index.search(query.astype("float32"), qk)
            return dists, idxs

        async with self._lock:
            k = min(k, self.n)
            dists, idxs = await asyncio.to_thread(_do, k)
            ordered = list(self._ordered_ids)
        out: list[tuple[str, float]] = []
        for sim, idx in zip(dists[0], idxs[0], strict=True):
            if idx < 0 or idx >= len(ordered):
                continue
            out.append((ordered[int(idx)], float(sim)))
        return out

    async def cosine_matrix(self) -> np.ndarray:
        """Full N×N cosine similarity matrix. Used for clustering."""
        assert self.index is not None
        if self.n == 0:
            return np.zeros((0, 0), dtype="float32")

        def _do() -> np.ndarray:
            vecs = self.index.reconstruct_n(0, self.n)
            return vecs @ vecs.T

        async with self._lock:
            return await asyncio.to_thread(_do)

    def offset_of(self, hypothesis_id: str) -> int | None:
        return self._offset_by_id.get(hypothesis_id)

    def hypothesis_at(self, offset: int) -> str | None:
        if 0 <= offset < len(self._ordered_ids):
            return self._ordered_ids[offset]
        return None
