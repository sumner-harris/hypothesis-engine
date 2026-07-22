# Modified from the original work.
"""Tests for the provider-aware API-key checker.

`has_llm_key(cfg)` decides whether the configured provider has the credentials
it needs. The CLI uses this to gate `run` / `resume` so users don't get
mysterious errors at first call.
"""

from __future__ import annotations

from hypothesis_engine.config import Config, has_llm_key, provider_key_env


def _cfg(provider: str) -> Config:
    cfg = Config()
    cfg.llm.provider = provider
    cfg.secrets.ANTHROPIC_API_KEY = ""
    cfg.secrets.OPENAI_API_KEY = ""
    cfg.secrets.OPENROUTER_API_KEY = ""
    cfg.secrets.GEMINI_API_KEY = ""
    cfg.secrets.GROQ_API_KEY = ""
    cfg.secrets.TOGETHER_API_KEY = ""
    cfg.secrets.MISTRAL_API_KEY = ""
    return cfg


def test_anthropic_provider_requires_anthropic_key() -> None:
    cfg = _cfg("anthropic")
    assert not has_llm_key(cfg)
    cfg.secrets.ANTHROPIC_API_KEY = "sk-fake"
    assert has_llm_key(cfg)


def test_openrouter_provider_uses_openrouter_key() -> None:
    cfg = _cfg("openrouter")
    assert not has_llm_key(cfg)
    cfg.secrets.OPENROUTER_API_KEY = "sk-or-fake"
    assert has_llm_key(cfg)


def test_openrouter_provider_accepts_openai_api_key_as_override() -> None:
    """OPENAI_API_KEY is the universal override for OpenAI-compat presets."""
    cfg = _cfg("openrouter")
    cfg.secrets.OPENAI_API_KEY = "sk-fake"
    assert has_llm_key(cfg)


def test_gemini_provider_uses_gemini_key() -> None:
    cfg = _cfg("gemini")
    assert not has_llm_key(cfg)
    cfg.secrets.GEMINI_API_KEY = "gemini-fake"
    assert has_llm_key(cfg)


def test_google_alias_uses_gemini_key() -> None:
    cfg = _cfg("google")
    cfg.secrets.GEMINI_API_KEY = "gemini-fake"
    assert has_llm_key(cfg)


def test_ollama_is_keyless() -> None:
    cfg = _cfg("ollama")
    assert has_llm_key(cfg)


def test_openai_compatible_local_base_url_is_keyless(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = _cfg("openai_compatible")
    cfg.llm.openai.base_url = "http://localhost:8000/v1"
    assert provider_key_env(cfg) == ""
    assert has_llm_key(cfg)


def test_openai_compatible_remote_base_url_requires_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = _cfg("openai_compatible")
    cfg.llm.openai.base_url = "https://example.test/v1"
    assert provider_key_env(cfg) == "OPENAI_API_KEY"
    assert not has_llm_key(cfg)
    cfg.secrets.OPENAI_API_KEY = "sk-fake"
    assert has_llm_key(cfg)


def test_provider_key_env_returns_expected_var_name() -> None:
    cfg = _cfg("openrouter")
    assert provider_key_env(cfg) == "OPENROUTER_API_KEY"
    cfg.llm.provider = "anthropic"
    assert provider_key_env(cfg) == "ANTHROPIC_API_KEY"
    cfg.llm.provider = "ollama"
    assert provider_key_env(cfg) == ""


def test_anthropic_provider_does_not_accept_openai_key() -> None:
    """OPENAI_API_KEY shouldn't satisfy the Anthropic provider check."""
    cfg = _cfg("anthropic")
    cfg.secrets.OPENAI_API_KEY = "sk-fake"
    assert not has_llm_key(cfg)
