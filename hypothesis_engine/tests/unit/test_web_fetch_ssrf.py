# Modified from the original work.
"""SSRF guard tests for web_fetch.

We can't easily exercise the live httpx path in a unit test, but we *can*
verify that the SSRF guard rejects private/loopback/metadata-service hosts
before the network call.
"""

from __future__ import annotations

import pytest

from hypothesis_engine.config import Config
from hypothesis_engine.tools.base import ToolCtx
from hypothesis_engine.tools.web_fetch import WebFetchTool, _is_private_ip


def test_private_ip_helper_blocks_loopback() -> None:
    assert _is_private_ip("127.0.0.1") is True
    assert _is_private_ip("localhost") is True


def test_private_ip_helper_blocks_link_local_metadata() -> None:
    # AWS / GCP metadata service
    assert _is_private_ip("169.254.169.254") is True


def test_private_ip_helper_blocks_rfc1918() -> None:
    assert _is_private_ip("10.0.0.1") is True
    assert _is_private_ip("192.168.1.1") is True
    assert _is_private_ip("172.16.0.1") is True


def test_private_ip_helper_allows_public() -> None:
    # 1.1.1.1 (Cloudflare DNS) is a stable public address
    assert _is_private_ip("1.1.1.1") is False


@pytest.mark.asyncio
async def test_web_fetch_rejects_loopback_url() -> None:
    tool = WebFetchTool(Config())
    res = await tool.call({"url": "http://127.0.0.1/admin"}, ToolCtx(cfg=Config(), db=None))
    assert res.is_error
    assert "private" in (res.error_message or "").lower()


@pytest.mark.asyncio
async def test_web_fetch_rejects_metadata_url() -> None:
    tool = WebFetchTool(Config())
    res = await tool.call(
        {"url": "http://169.254.169.254/latest/meta-data/"},
        ToolCtx(cfg=Config(), db=None),
    )
    assert res.is_error
    assert "private" in (res.error_message or "").lower()


@pytest.mark.asyncio
async def test_web_fetch_rejects_unsupported_scheme() -> None:
    tool = WebFetchTool(Config())
    res = await tool.call({"url": "file:///etc/passwd"}, ToolCtx(cfg=Config(), db=None))
    assert res.is_error


def test_web_fetch_max_chars_defaults_to_compact_source_preview() -> None:
    cfg = Config()
    tool = WebFetchTool(cfg)
    assert cfg.web_fetch.max_bytes == 50_010_000
    assert cfg.web_fetch.max_chars == 20_000
    assert tool.input_schema["properties"]["max_chars"]["maximum"] == 20_000
    assert tool.input_schema["properties"]["max_chars"]["default"] == 20_000
