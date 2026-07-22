# Modified from the original work.
"""ToolRegistry — discovers and indexes all available tools.

Tools available to each agent are decided by `tools_for(agent, mode)` so we can
restrict what the LLM sees per call (smaller tool list = better tool-use quality).
"""

from __future__ import annotations

import os
from typing import Any

from ..capabilities.catalog import CapabilityCatalog, require_valid_catalog
from ..config import Config
from .base import Tool, ToolCtx, ToolResult, to_anthropic_tool
from .builtins.arxiv import ArxivSearchTool
from .builtins.biorxiv import BiorxivSearchTool
from .builtins.chemrxiv import ChemrxivSearchTool
from .builtins.europe_pmc import EuropePMCSearchTool
from .builtins.pubmed import PubmedSearchTool
from .capabilities import (
    CapabilityGetTool,
    CapabilitySearchTool,
    CapabilityValidateWorkflowTool,
)
from .rag import RAGRetrieveContextTool, RAGStatusTool, rag_is_enabled
from .science_skills import ScienceSkillTool, discover_skills
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool

# Per-agent tool allowlists. Keys are tool names (or "literature_*" wildcards
# matched explicitly in the resolver).
AGENT_TOOLS: dict[str, set[str]] = {
    "generation": {
        "web_search",
        "web_fetch",
        "pubmed_search",
        "arxiv_search",
        "biorxiv_search",
        "chemrxiv_search",
        "europe_pmc_search",
        "rag_retrieve_context",
        "rag_kb_status",
        "capability_search",
        "capability_get",
        "capability_validate_workflow",
        "literature_*",  # any science-skills literature_* tools
    },
    "reflection": {
        "web_search",
        "web_fetch",
        "pubmed_search",
        "arxiv_search",
        "biorxiv_search",
        "chemrxiv_search",
        "europe_pmc_search",
        "rag_retrieve_context",
        "rag_kb_status",
        "capability_search",
        "capability_get",
        "literature_*",
        # code_exec wired in M2
    },
    "ranking": set(),  # no tools mid-debate
    "evolution": {
        "web_search",
        "web_fetch",
        "pubmed_search",
        "arxiv_search",
        "biorxiv_search",
        "chemrxiv_search",
        "europe_pmc_search",
        "rag_retrieve_context",
        "rag_kb_status",
        "capability_search",
        "capability_get",
        "capability_validate_workflow",
        "literature_*",
    },
    "proximity": set(),
    "metareview": set(),
}


class ToolRegistry:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._tools: dict[str, Tool] = {}
        self._capability_catalog: CapabilityCatalog | None = None

    def discover(self) -> ToolRegistry:
        disabled = _disabled_tool_names()
        # Built-ins
        for t in (
            WebFetchTool(self._cfg),
            PubmedSearchTool(self._cfg),
            ArxivSearchTool(),
            BiorxivSearchTool(),
            ChemrxivSearchTool(),
            EuropePMCSearchTool(),
        ):
            if t.name not in disabled:
                self._register(t)
        # web_search only registers if a backing search API key is set.
        # Otherwise the model would see a tool it can't actually use and
        # smaller models tend to abort the task instead of falling back to
        # PubMed / arxiv / Europe PMC.
        if (
            self._cfg.secrets.TAVILY_API_KEY
            or os.environ.get("TAVILY_API_KEY")
            or self._cfg.secrets.BRAVE_API_KEY
            or os.environ.get("BRAVE_API_KEY")
        ):
            web_search = WebSearchTool(self._cfg)
            if web_search.name not in disabled:
                self._register(web_search)
        if rag_is_enabled(self._cfg):
            for t in (RAGStatusTool(self._cfg), RAGRetrieveContextTool(self._cfg)):
                if t.name not in disabled:
                    self._register(t)
        if self._cfg.capabilities.enabled:
            catalog = CapabilityCatalog.from_config(self._cfg)
            self._capability_catalog = catalog
            for t in (
                CapabilitySearchTool(
                    catalog,
                    default_limit=self._cfg.capabilities.max_search_results,
                ),
                CapabilityGetTool(catalog),
                CapabilityValidateWorkflowTool(catalog),
            ):
                if t.name not in disabled:
                    self._register(t)
        # Science-skills
        for meta in discover_skills(self._cfg):
            t = ScienceSkillTool(self._cfg, meta)
            if t.name not in disabled:
                self._register(t)
        self.validate_capabilities()
        return self

    def validate_capabilities(self):
        if self._capability_catalog is None:
            return None
        return require_valid_catalog(
            self._capability_catalog,
            registered_tool_names=set(self._tools),
        )

    def _register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            # later registrations win; log a warning at use site if needed
            pass
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def tools_for(self, agent: str) -> list[Tool]:
        allow = AGENT_TOOLS.get(agent, set())
        out: list[Tool] = []
        for t in self._tools.values():
            if t.name in allow:
                out.append(t)
            else:
                for pattern in allow:
                    if pattern.endswith("*") and t.name.startswith(pattern[:-1]):
                        out.append(t)
                        break
        return out

    def anthropic_tools_for(self, agent: str) -> list[dict[str, Any]]:
        return [to_anthropic_tool(t) for t in self.tools_for(agent)]

    async def call(self, name: str, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(is_error=True, error_message=f"unknown tool: {name}")
        return await tool.call(args, ctx)

    def summary(self) -> list[dict[str, Any]]:
        """Used by `hypothesis-engine tools list` and the UI."""
        return [
            {"name": t.name, "description": t.description[:200]}
            for t in sorted(self._tools.values(), key=lambda x: x.name)
        ]


def _disabled_tool_names() -> set[str]:
    raw = os.environ.get("HYPOTHESIS_ENGINE_DISABLED_TOOLS", "")
    return {name.strip() for name in raw.split(",") if name.strip()}
