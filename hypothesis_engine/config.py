# Modified from the original work.
"""Configuration loader.

Layered: config/default.toml → ~/.hypothesis-engine/config.toml → ./hypothesis-engine.toml → env.
Secrets come from environment variables only.
"""

from __future__ import annotations

import ipaddress
import os
import tomllib
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.toml"


class RunCfg(BaseModel):
    concurrency: int = 4
    initial_generations: int = 3
    max_ideas: int = 60
    # Legacy estimate/display knob. Tournament scheduling no longer caps
    # individual hypotheses; sessions stop by Elo stability after max_ideas.
    max_matches_per_idea: int = 0
    wall_clock_seconds: int = 7200
    budget_tokens: int = 5_000_000
    budget_usd: float = 25.0


class StorageCfg(BaseModel):
    data_dir: str = "./data"


class ScienceSkillsCfg(BaseModel):
    path: str = "./vendor/science-skills"
    pinned_commit: str = ""


class CapabilitiesCfg(BaseModel):
    """Local catalog used to ground study plans in available capabilities."""

    enabled: bool = False
    catalog_path: str = "./config/capabilities"
    grounding_policy: Literal["advisory", "required"] = "advisory"
    max_search_results: int = Field(default=8, ge=1, le=50)


class EmbeddingsCfg(BaseModel):
    provider: Literal["openai", "openai_compatible"] = "openai_compatible"
    model: str = "sfr-embedding-mistral"
    dim: int = Field(default=4096, ge=1)
    base_url: str | None = "http://localhost:8001/v1"

    @model_validator(mode="after")
    def validate_provider_contract(self) -> EmbeddingsCfg:
        if self.provider == "openai":
            if self.model != "text-embedding-3-large":
                raise ValueError('provider="openai" requires model="text-embedding-3-large"')
            if self.dim > 3072:
                raise ValueError(
                    'provider="openai" requires dim <= 3072 for text-embedding-3-large'
                )
            # Direct OpenAI must not inherit the local endpoint from the
            # layered default configuration.
            self.base_url = None
        return self


class VectorsCfg(BaseModel):
    dedup_cosine_threshold: float = 0.92
    cluster_threshold: float = 0.15
    full_recluster_every_matches: int = 20


class RankingCfg(BaseModel):
    k_factor_new: int = 32
    k_factor_warm: int = 16
    elo_initial: int = 1200
    elo_logistic_scale: float = 400.0
    debate_when_matches_lt: int = 2
    debate_when_elo_delta_lt: int = 50
    # Ranking completions are capped independently of the context window.
    # Debate mode needs room to deliberate and still emit the required verdict.
    pairwise_max_output_tokens: int = 8192
    debate_max_output_tokens: int = 12288
    verdict_retry_max_output_tokens: int = 1024
    # Prompt-side budgets used to reduce verbosity bias in ranking comparisons.
    # Set any value below 0 to keep the corresponding text untrimmed.
    prompt_hypothesis_max_chars: int = -1
    prompt_review_max_chars: int = -1
    prompt_side_max_chars: int = -1
    batch_below_decile: bool = True
    batch_submit_every_seconds: int = 1800
    # How many idle ranking batches to enqueue while the tournament pool is
    # still warming up before the evolution maturity gate.
    idle_parallel_tasks: int = 1
    # Avoid over-ranking a tiny closed pool while slower RAG-backed
    # reflection/evolution tasks are still feeding new candidates.
    small_pool_backlog_size: int = 6
    small_pool_backlog_match_cap: int = 5
    p_new: float = 0.4
    p_close: float = 0.4
    p_random: float = 0.2
    # Unordered hypothesis pairs are judged as bounded best-of-N contests.
    # Elo is applied only once the pair has a real winner.
    pair_max_matches: int = 3
    pair_wins_to_close: int = 2


class ReflectionCfg(BaseModel):
    # Reflection completions are capped independently of the context window.
    # Recovery is a compact forced-tool call used when the normal review loop
    # fails to emit a complete record_review terminal tool.
    review_max_output_tokens: int = 8192
    review_recovery_max_output_tokens: int = 2048
    review_recovery_max_attempts: int = 4
    review_recovery_token_multiplier: int = 2
    review_recovery_max_output_tokens_cap: int = 16384


class GenerationCfg(BaseModel):
    # Generation completions are capped independently of the context window.
    # Recovery is a compact forced-tool call used when the literature loop
    # produces prose or a malformed/truncated record_hypothesis payload.
    discovery_profiles: str = "./config/discovery_profiles/materials_chemistry.yaml"
    hypothesis_max_output_tokens: int = 8192
    hypothesis_recovery_max_output_tokens: int = 4096
    hypothesis_recovery_max_attempts: int = 3
    hypothesis_recovery_token_multiplier: int = 2
    hypothesis_recovery_max_output_tokens_cap: int = 16384
    # Extra record_hypothesis calls to make a mechanistically distinct
    # replacement when persistence rejects a proposal as a near duplicate.
    dedup_replacement_attempts: int = 0


class TerminationCfg(BaseModel):
    elo_stability_k: int = 5
    elo_stability_n: int = 3
    elo_stability_eps: float = 25.0
    match_snapshot_every: int = 10
    # Guards that prevent elo_stable from firing on a small pool.
    # Defaults of 0 preserve the original behaviour.
    min_ideas_before_stable: int = 0
    min_matches_before_stable: int = 0


class EvolutionCfg(BaseModel):
    """Controls when the idle-refinement loop triggers evolution."""

    # Minimum number of *mature* hypotheses required before the supervisor
    # schedules evolution tasks. A mature hypothesis has at least
    # `mature_matches` completed tournament matches.
    min_mature: int = 20
    mature_matches: int = 3
    # Do not start idle evolution before the tournament has this many matches.
    # 0 preserves the original permissive behavior.
    min_tournament_matches: int = 0
    # Minimum additional tournament matches required between idle evolution
    # batches. 1 preserves the original per-match-count idempotency cadence.
    min_matches_between_batches: int = 1
    # Require Elo rank stability before idle evolution, matching the paper's
    # "if scores are stable" trigger without reusing final-stop guards.
    require_rank_stability: bool = True
    rank_stability_n: int = 3
    rank_stability_eps: float = 25.0
    rank_stability_min_ideas: int = 0
    rank_stability_min_matches: int = 0
    # How many top hypotheses to consider in each evolution task.
    top_k: int = 5
    # How many evolution tasks the idle scheduler may enqueue at once once
    # enough hypotheses are mature. 1 preserves the original single-pass loop;
    # higher values keep run.concurrency better utilized during refinement.
    idle_parallel_tasks: int = 1
    # Evolution completions are capped independently of the context window.
    # Recovery is a compact forced-tool call used when a strategy produces prose
    # or hits max_tokens without a complete record_hypothesis payload.
    hypothesis_max_output_tokens: int = 8192
    hypothesis_recovery_max_output_tokens: int = 4096
    hypothesis_recovery_max_attempts: int = 3
    hypothesis_recovery_token_multiplier: int = 2
    hypothesis_recovery_max_output_tokens_cap: int = 16384
    dedup_replacement_attempts: int = 0


class BudgetSharesCfg(BaseModel):
    generation: float = 0.20
    reflection: float = 0.20
    ranking: float = 0.25
    evolution: float = 0.15
    metareview: float = 0.10
    literature_review: float = 0.03
    proximity: float = 0.02
    reserve: float = 0.05


class ModelsCfg(BaseModel):
    parse_goal: str = "claude-sonnet-4-6"
    generation: str = "claude-opus-4-7"
    reflection: str = "claude-opus-4-7"
    evolution: str = "claude-opus-4-7"
    ranking_pairwise: str = "claude-sonnet-4-6"
    ranking_debate: str = "claude-sonnet-4-6"
    ranking_priority: str = "claude-opus-4-7"
    metareview_feedback: str = "claude-sonnet-4-6"
    metareview_final: str = "claude-opus-4-7"
    literature_review: str = "claude-haiku-4-5-20251001"


class ThinkingCfg(BaseModel):
    generation_literature: int = 4000
    generation_debate: int = 8000
    reflection_full: int = 0
    reflection_verification: int = 12000
    reflection_observation: int = 6000
    ranking_pairwise: int = 4000
    ranking_debate: int = 8000
    evolution_combine: int = 6000
    evolution_out_of_box: int = 6000
    evolution_feasibility: int = 0
    evolution_simplify: int = 0
    metareview_feedback: int = 8000
    metareview_final: int = 16000


class ToolLoopCfg(BaseModel):
    generation_max_iters: int = 8
    reflection_max_iters: int = 8
    ranking_max_iters: int = 3
    evolution_max_iters: int = 6
    metareview_max_iters: int = 12
    parallel_cap: int = 4
    tool_timeout_seconds: int = 30


class RetryCfg(BaseModel):
    max_attempts_429: int = 6
    max_attempts_529: int = 8
    max_attempts_5xx: int = 5
    max_attempts_timeout: int = 3
    base_ms: int = 1000
    cap_ms: int = 60_000
    per_call_timeout_seconds: int = 120
    per_call_timeout_thinking_seconds: int = 300


class LeaseCfg(BaseModel):
    default_seconds: int = 300
    reflection_seconds: int = 600
    metareview_final_seconds: int = 1800
    heartbeat_seconds: int = 60
    max_attempts: int = 3


class WebSearchCfg(BaseModel):
    provider: str = "tavily"
    max_results: int = 30


class WebFetchCfg(BaseModel):
    max_bytes: int = 50_010_000
    # Return compact source previews to agents. The untruncated extracted text is
    # cached in artifacts and can later be used by a RAG retriever.
    max_chars: int = 20_000
    timeout_seconds: int = 30
    user_agent: str = "hypothesis-engine/0.1"


class RAGCfg(BaseModel):
    enabled: bool = False
    package_path: str = "./vendor/RAGAgent_AutonomousMaterialsSynthesis"
    seed_kb_path: str = ""
    auto_ingest_fetched_pdfs: bool = True
    auto_ingest_arxiv_pdfs: bool = False
    auto_ingest_max_pdfs_per_search: int = 5
    literature_review_enabled: bool = True
    literature_review_dedupe_reviewed_results: bool = True
    literature_review_max_candidates: int = 30
    literature_review_max_context_results: int = 30
    literature_review_max_output_tokens: int = 12_288
    literature_review_timeout_seconds: int = 300
    literature_review_fallback_to_top_results: bool = True
    max_session_papers: int = 120
    generation_discovery_max_pdfs_per_round: int = 100
    generation_wait_min_indexed_papers: int = 8
    generation_wait_timeout_seconds: int = 300
    embedding_model: str = ""
    embedding_profile: str = ""
    embedding_base_url: str | None = None
    llm_model: str = ""
    llm_base_url: str | None = None
    retrieval_method: str = "faiss_mmr"
    top_k_faiss: int = 30
    diversity: float = 0.7
    context_max_chars: int = 12_000
    rerank_enabled: bool = True
    rerank_base_url: str | None = None
    rerank_model: str = ""
    rerank_top_k: int = 8
    run_graphrag: bool = False
    use_graphrag: bool = False


class CodeExecCfg(BaseModel):
    provider: str = "anthropic"
    local_cpu_seconds: int = 30
    local_mem_mb: int = 512


class OpenAIProviderCfg(BaseModel):
    """OpenAI / OpenAI-compatible endpoint settings.

    `base_url` overrides the SDK default. Use it to point at any
    OpenAI-compatible provider (Groq, Together, OpenRouter, Mistral,
    Gemini OpenAI-compat, Ollama local, vLLM, ...). When a named preset
    such as `provider = "openrouter"` is used, this only needs to be set
    if you want to override the preset's base_url.

    `reasoning_effort`, when set, is sent on every OpenAI-backed chat
    completion request, including requests to compatible endpoints.
    """

    base_url: str | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None


class AnthropicProviderCfg(BaseModel):
    """Anthropic provider settings. `base_url` is rarely used; honored if set."""

    base_url: str | None = None


class OpenRouterProviderCfg(BaseModel):
    """OpenRouter attribution headers.

    OpenRouter ranks apps in its catalog by `HTTP-Referer` + `X-Title`.
    Setting these is optional but recommended for production traffic;
    leave blank for ad-hoc use.
    """

    referer: str = ""
    title: str = ""


class LLMCfg(BaseModel):
    """Choose which LLM vendor backs the agents.

    Supported values:
    - "anthropic" — Claude via the official Anthropic SDK (default). Cache
      breakpoints, extended thinking, and the Batch API are only available
      under this provider.
    - "openai" — OpenAI Chat Completions. An explicit
      `llm.openai.reasoning_effort` is sent with every request. Otherwise,
      thinking budgets are translated for recognized reasoning models.
      Cache breakpoints are stripped.
    - "openrouter" — OpenRouter (openrouter.ai). 200+ models from every
      major vendor in one place. Set OPENROUTER_API_KEY (or
      OPENAI_API_KEY). Optional attribution in [llm.openrouter].
    - "gemini" / "google" — Google Gemini via the official OpenAI-compat
      endpoint. Set GEMINI_API_KEY. Models: "gemini-2.5-pro",
      "gemini-2.5-flash", etc.
    - "groq", "together", "mistral", "ollama" — convenience presets for
      those endpoints; each reads its own API key env var
      (GROQ_API_KEY, TOGETHER_API_KEY, MISTRAL_API_KEY).
    - "openai_compatible" — same client and reasoning-effort behavior as
      `openai`, but allows `llm.openai.base_url` to point at any other
      OpenAI-compatible endpoint not yet covered by a preset.
    """

    provider: str = "anthropic"
    openai: OpenAIProviderCfg = Field(default_factory=OpenAIProviderCfg)
    anthropic: AnthropicProviderCfg = Field(default_factory=AnthropicProviderCfg)
    openrouter: OpenRouterProviderCfg = Field(default_factory=OpenRouterProviderCfg)


class WebUICfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7878


class Secrets(BaseSettings):
    """Secrets pulled from env only. Empty string means 'not configured'."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    TOGETHER_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""
    OLLAMA_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    BRAVE_API_KEY: str = ""
    NCBI_API_KEY: str = ""
    OPENALEX_API_KEY: str = ""


class Config(BaseModel):
    run: RunCfg = Field(default_factory=RunCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    science_skills: ScienceSkillsCfg = Field(default_factory=ScienceSkillsCfg)
    capabilities: CapabilitiesCfg = Field(default_factory=CapabilitiesCfg)
    embeddings: EmbeddingsCfg = Field(default_factory=EmbeddingsCfg)
    vectors: VectorsCfg = Field(default_factory=VectorsCfg)
    ranking: RankingCfg = Field(default_factory=RankingCfg)
    reflection: ReflectionCfg = Field(default_factory=ReflectionCfg)
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    termination: TerminationCfg = Field(default_factory=TerminationCfg)
    evolution: EvolutionCfg = Field(default_factory=EvolutionCfg)
    budget_shares: BudgetSharesCfg = Field(default_factory=BudgetSharesCfg)
    models: ModelsCfg = Field(default_factory=ModelsCfg)
    thinking: ThinkingCfg = Field(default_factory=ThinkingCfg)
    tool_loop: ToolLoopCfg = Field(default_factory=ToolLoopCfg)
    retry: RetryCfg = Field(default_factory=RetryCfg)
    lease: LeaseCfg = Field(default_factory=LeaseCfg)
    web_search: WebSearchCfg = Field(default_factory=WebSearchCfg)
    web_fetch: WebFetchCfg = Field(default_factory=WebFetchCfg)
    rag: RAGCfg = Field(default_factory=RAGCfg)
    code_exec: CodeExecCfg = Field(default_factory=CodeExecCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    web_ui: WebUICfg = Field(default_factory=WebUICfg)
    secrets: Secrets = Field(default_factory=Secrets)

    @property
    def data_dir(self) -> Path:
        p = Path(self.storage.data_dir)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "hypothesis_engine.db"

    @property
    def capability_catalog_path(self) -> Path:
        path = Path(self.capabilities.catalog_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def session_artifact_dir(self, session_id: str) -> Path:
        return self.data_dir / "artifacts" / session_id

    def session_vector_dir(self, session_id: str) -> Path:
        return self.data_dir / "vectors" / session_id

    def session_rag_dir(self, session_id: str) -> Path:
        return self.data_dir / "rag" / session_id

    @property
    def rag_seed_kb_path(self) -> Path | None:
        raw = self.rag.seed_kb_path.strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    def session_log_path(self, session_id: str) -> Path:
        return self.data_dir / "logs" / f"session-{session_id}.jsonl"


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(extra_path: Path | None = None) -> Config:
    """Layered load: default.toml → ~/.hypothesis-engine/config.toml → ./hypothesis-engine.toml → extra_path → env."""
    merged: dict[str, Any] = _read_toml(DEFAULT_CONFIG)

    for p in (
        Path.home() / ".hypothesis-engine" / "config.toml",
        Path.cwd() / "hypothesis-engine.toml",
        extra_path,
    ):
        if p is not None:
            merged = _deep_merge(merged, _read_toml(p))

    cfg = Config.model_validate(merged)
    # secrets pulled from env via Secrets() — already wired by default_factory above
    return cfg


def has_anthropic_key(cfg: Config) -> bool:
    return bool(cfg.secrets.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY"))


_LOCAL_OPENAI_COMPAT_HOSTS = {"localhost", "0.0.0.0", "::1"}


def _is_local_openai_compatible_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = urlparse(base_url).hostname
    if not host:
        return False
    host = host.lower()
    if host in _LOCAL_OPENAI_COMPAT_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# Env var names per provider preset (see llm/provider.py KNOWN_PROVIDERS).
_PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai_compatible": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "ollama": "",  # keyless
}


def provider_key_env(cfg: Config) -> str:
    """Env-var name the configured LLM provider expects, or '' if keyless."""
    name = (getattr(cfg.llm, "provider", "anthropic") or "anthropic").strip().lower()
    if name == "openai_compatible":
        base_url = getattr(getattr(cfg.llm, "openai", None), "base_url", None) or os.environ.get(
            "OPENAI_BASE_URL"
        )
        if _is_local_openai_compatible_base_url(base_url):
            return ""
    return _PROVIDER_ENV_VARS.get(name, "ANTHROPIC_API_KEY")


def has_llm_key(cfg: Config) -> bool:
    """True if the configured provider's API key is available, OR the provider
    is keyless (Ollama or local OpenAI-compatible endpoints)."""
    env_var = provider_key_env(cfg)
    if not env_var:
        return True  # keyless provider
    # Explicit OPENAI_API_KEY is always honored (lets users repurpose presets).
    openai_compat_envs = {
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "MISTRAL_API_KEY",
    }
    if env_var in openai_compat_envs and (
        cfg.secrets.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    ):
        return True
    return bool(getattr(cfg.secrets, env_var, "") or os.environ.get(env_var))
