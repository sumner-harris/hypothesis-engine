# Modified from the original work.
"""JSON-on-disk helpers.

Artifacts live under `<data_dir>/artifacts/<session_id>/<kind>/<id>.json`.
The DB stores the path *relative* to `data_dir` for portability.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import Config

# Identifiers used as path components. We refuse anything outside this charset
# so a caller can't smuggle "../../etc/passwd" through `kind` or `id_`.
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
_WRITE_LOCKS = tuple(threading.Lock() for _ in range(64))


def _validate_component(value: str, *, label: str) -> str:
    if not value or value in ("..", ".") or value.startswith("/"):
        raise ValueError(f"{label}={value!r} is not a valid path component")
    if not _SAFE_COMPONENT_RE.fullmatch(value):
        raise ValueError(f"{label}={value!r} contains disallowed characters")
    parts = [p for p in value.split("/") if p]
    if any(p == ".." for p in parts):
        raise ValueError(f"{label}={value!r} contains '..' segment")
    return value


def _resolved_under(base: Path, candidate: Path) -> Path:
    """Resolve `candidate` and assert it stays under `base` (after both are
    resolved). Raises ValueError on escape.
    """
    base_resolved = base.resolve()
    # We resolve without strict=True so non-existent leaves are allowed; the
    # call still normalizes "../" segments and any symlink hops.
    candidate_resolved = candidate.resolve()
    try:
        base_comparison = _path_for_comparison(base_resolved)
        candidate_comparison = _path_for_comparison(candidate_resolved)
        if os.path.commonpath((base_comparison, candidate_comparison)) != base_comparison:
            raise ValueError
    except ValueError as e:
        raise ValueError(f"path {candidate} escapes {base}") from e
    return candidate_resolved


def _path_for_comparison(path: Path) -> str:
    """Normalize equivalent Windows device and drive path spellings."""
    value = os.path.normcase(str(path))
    if value.startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[8:]
    if value.startswith("\\\\?\\"):
        return value[4:]
    return value


def _rel(cfg: Config, p: Path) -> str:
    return p.relative_to(cfg.data_dir).as_posix()


def session_root(cfg: Config, session_id: str) -> Path:
    _validate_component(session_id, label="session_id")
    return cfg.data_dir / "artifacts" / session_id


def _tmp_path(p: Path) -> Path:
    return p.with_name(f".{p.name}.{uuid4().hex}.tmp")


def _write_text_atomic(p: Path, body: str) -> None:
    # Windows can reject concurrent os.replace calls to the same target. A
    # striped lock preserves parallelism across unrelated artifact paths.
    lock = _WRITE_LOCKS[hash(p) % len(_WRITE_LOCKS)]
    with lock:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp_path(p)
        try:
            tmp.write_text(body, encoding="utf-8")
            _replace_with_retry(tmp, p)
        finally:
            tmp.unlink(missing_ok=True)


def _replace_with_retry(source: Path, target: Path) -> None:
    """Replace ``target``, tolerating brief Windows sharing violations."""
    for attempt in range(7):
        try:
            source.replace(target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 6:
                raise
            time.sleep(0.01 * (2**attempt))


def _write(p: Path, payload: Any) -> None:
    body = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
    _write_text_atomic(p, body)


def _read(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


async def write_json(cfg: Config, session_id: str, kind: str, id_: str, payload: Any) -> str:
    """Persist a JSON artifact; return its relative path."""
    _validate_component(kind, label="kind")
    _validate_component(id_, label="id")
    root = session_root(cfg, session_id)
    p = root / kind / f"{id_}.json"
    # Defence-in-depth: even if the regex misses some pathological case,
    # confirm the final path is inside the session root.
    _resolved_under(root.parent, p)
    await asyncio.to_thread(_write, p, payload)
    return _rel(cfg, p)


async def read_json(cfg: Config, rel_path: str) -> Any:
    p = cfg.data_dir / rel_path
    _resolved_under(cfg.data_dir, p)
    return await asyncio.to_thread(_read, p)


async def write_text(cfg: Config, session_id: str, kind: str, id_: str, suffix: str, body: str) -> str:
    _validate_component(kind, label="kind")
    _validate_component(id_, label="id")
    if "/" in suffix or ".." in suffix:
        raise ValueError(f"suffix={suffix!r} is not a valid filename suffix")
    root = session_root(cfg, session_id)
    p = root / kind / f"{id_}{suffix}"
    _resolved_under(root.parent, p)

    await asyncio.to_thread(_write_text_atomic, p, body)
    return _rel(cfg, p)
