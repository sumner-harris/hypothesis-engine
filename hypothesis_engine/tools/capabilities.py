"""Agent tools backed by the structured capability catalog."""

from __future__ import annotations

import time
from typing import Any

from ..capabilities.catalog import CapabilityCatalog, capability_summary
from ..capabilities.grounding import validate_study_plan
from .base import ToolCtx, ToolResult


class CapabilitySearchTool:
    name = "capability_search"
    description = (
        "Search the configured local library of available experimental, simulation, "
        "AI, and data capabilities. Use this before designing study-plan work packages; "
        "catalog records describe actual availability and operating limits."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "kinds": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "experimental",
                        "simulation",
                        "ai",
                        "data",
                    ],
                },
            },
            "domains": {"type": "array", "items": {"type": "string"}},
            "inputs": {"type": "array", "items": {"type": "string"}},
            "outputs": {"type": "array", "items": {"type": "string"}},
            "availability": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["available", "limited", "planned", "unavailable", "unknown"],
                },
            },
            "max_results": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["query"],
    }

    def __init__(self, catalog: CapabilityCatalog, *, default_limit: int = 8) -> None:
        self._catalog = catalog
        self._default_limit = default_limit

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        del ctx
        started = time.monotonic()
        query = str(args.get("query") or "").strip()
        results = self._catalog.search(
            query,
            kinds=_string_list(args.get("kinds")),
            domains=_string_list(args.get("domains")),
            inputs=_string_list(args.get("inputs")),
            outputs=_string_list(args.get("outputs")),
            availability=_string_list(args.get("availability")),
            limit=int(args.get("max_results") or self._default_limit),
        )
        payload = {
            "catalog_revision": self._catalog.revision,
            "query": query,
            "n": len(results),
            "results": [
                {**capability_summary(item), "match_score": score} for item, score in results
            ],
        }
        return ToolResult(
            content=payload,
            duration_ms=int((time.monotonic() - started) * 1000),
            result_bytes=len(str(payload)),
        )


class CapabilityGetTool:
    name = "capability_get"
    description = (
        "Retrieve exact versioned capability specifications, including parameter ranges, "
        "dependencies, constraints, access, provenance, and last verification date."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "capability_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 20,
            }
        },
        "required": ["capability_ids"],
    }

    def __init__(self, catalog: CapabilityCatalog) -> None:
        self._catalog = catalog

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        del ctx
        ids = _string_list(args.get("capability_ids"))
        if not ids:
            return ToolResult(is_error=True, error_message="capability_ids must be non-empty")
        found, missing = self._catalog.get_many(ids)
        payload = {
            "catalog_revision": self._catalog.revision,
            "capabilities": [capability_summary(item, detailed=True) for item in found],
            "missing_ids": missing,
        }
        return ToolResult(content=payload, result_bytes=len(str(payload)))


class CapabilityValidateWorkflowTool:
    name = "capability_validate_workflow"
    description = (
        "Validate capability_refs in a proposed structured study_plan. Checks exact IDs and "
        "versions, availability, required parameters, operating ranges, dependencies, and "
        "incompatibilities. Repair every error before recording the hypothesis."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "study_plan": {
                "type": "array",
                "items": {"type": "object"},
                "minItems": 1,
            }
        },
        "required": ["study_plan"],
    }

    def __init__(self, catalog: CapabilityCatalog) -> None:
        self._catalog = catalog

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        del ctx
        report = validate_study_plan(self._catalog, args.get("study_plan"))
        payload = report.model_dump()
        return ToolResult(content=payload, result_bytes=len(str(payload)))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
