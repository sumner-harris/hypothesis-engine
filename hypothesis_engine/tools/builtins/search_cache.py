"""Session-scoped cache for deterministic search tool calls."""

from __future__ import annotations

import asyncio
import json
from hashlib import sha1
from pathlib import Path
from typing import Any
from uuid import uuid4

from ...config import Config


def normalized_query(query: str) -> str:
    return " ".join(query.split()).casefold()


def cache_key(params: dict[str, Any]) -> str:
    data = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return sha1(data.encode("utf-8")).hexdigest()


def _cache_path(cfg: Config, session_id: str, tool_name: str, key: str) -> Path:
    return cfg.session_artifact_dir(session_id) / "searches" / tool_name / f"{key}.json"


async def read_search_cache(
    cfg: Config,
    session_id: str | None,
    tool_name: str,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    if not session_id:
        return None
    path = _cache_path(cfg, session_id, tool_name, cache_key(params))
    return await asyncio.to_thread(_read_json_if_exists, path)


async def write_search_cache(
    cfg: Config,
    session_id: str | None,
    tool_name: str,
    params: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    if not session_id:
        return
    path = _cache_path(cfg, session_id, tool_name, cache_key(params))
    await asyncio.to_thread(_write_json, path, payload)


def cached_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["cached"] = True
    return out


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)
