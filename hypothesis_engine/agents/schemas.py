# Modified from the original work.
"""Shared Anthropic tool-use schemas for structured outputs.

Each agent gets one or more of these as required tool calls. Using tool-use
schemas (rather than "respond in JSON") is the most reliable structured-output
mechanism on the Anthropic API.
"""

from __future__ import annotations

from typing import Any

RECORD_HYPOTHESIS_TOOL: dict[str, Any] = {
    "name": "record_hypothesis",
    "description": (
        "Record a structured hypothesis at the end of generation/evolution. Call this "
        "exactly once when your hypothesis is finalized. All citations must reference "
        "URLs that previously appeared in your tool_result outputs from search/fetch."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short noun-phrase title."},
            "statement": {"type": "string", "description": "One sentence: the hypothesis."},
            "mechanism": {
                "type": "string",
                "description": "Detailed causal, mechanistic, algorithmic, or design rationale for the complete study.",
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific named actors (proteins, materials, datasets, agents, etc.).",
            },
            "anticipated_outcomes": {
                "type": "string",
                "description": "What would be observed, measured, built, benchmarked, or falsified if the hypothesis is true.",
            },
            "novelty_argument": {
                "type": "string",
                "description": "What is new relative to the cited literature, including how multiple study components are integrated.",
            },
            "study_plan": {
                "type": "array",
                "description": (
                    "Domain-generic structured work packages required to test the "
                    "hypothesis. Fill one item for each work package requested by "
                    "the active discovery profile."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "component_id": {
                            "type": "string",
                            "description": "Stable id from the active profile, or a concise snake_case id.",
                        },
                        "component_label": {"type": "string"},
                        "role": {
                            "type": "string",
                            "description": "How this component contributes, e.g. primary_driver, support, validation, falsification.",
                        },
                        "objective": {"type": "string"},
                        "methods": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Concrete experiments, analyses, simulations, benchmarks, assays, or implementation steps.",
                        },
                        "capability_refs": {
                            "type": "array",
                            "description": (
                                "Exact versioned references returned by the configured "
                                "capability catalog. Leave empty when no catalog match exists."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "capability_id": {"type": "string"},
                                    "version": {"type": "string"},
                                    "purpose": {"type": "string"},
                                    "parameters": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "value": {},
                                                "unit": {"type": "string"},
                                            },
                                            "required": ["name", "value"],
                                        },
                                    },
                                },
                                "required": ["capability_id", "purpose", "parameters"],
                            },
                        },
                        "variables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Independent variables, settings, perturbations, datasets, strains, materials, parameters, or conditions to vary.",
                        },
                        "outputs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Measurements, calculated quantities, metrics, artifacts, observations, or deliverables.",
                        },
                        "quantitative_targets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific thresholds, parameter ranges, expected effect sizes, pass/fail values, or quantities to estimate.",
                        },
                        "controls_or_comparators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Controls, baselines, ablations, null cases, reference systems, or comparators.",
                        },
                        "failure_criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Results that would falsify, weaken, or bound this component of the hypothesis.",
                        },
                    },
                    "required": [
                        "component_id",
                        "component_label",
                        "role",
                        "objective",
                        "methods",
                        "outputs",
                        "controls_or_comparators",
                        "failure_criteria",
                    ],
                },
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                        "excerpt": {
                            "type": "string",
                            "description": "Verbatim short quote from the source.",
                        },
                        "doi": {"type": "string"},
                        "year": {"type": "integer"},
                    },
                    "required": ["url", "title"],
                },
            },
            "strategy": {
                "type": "string",
                "enum": [
                    "literature",
                    "debate",
                    "combine",
                    "simplify",
                    "out_of_box",
                    "feasibility",
                    "assumption",
                    "feedback_driven",
                ],
                "description": "Strategy that produced this hypothesis (set by the agent).",
            },
            "parent_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Hypothesis IDs this one descends from (Evolution only).",
            },
        },
        "required": [
            "title",
            "statement",
            "mechanism",
            "entities",
            "anticipated_outcomes",
            "novelty_argument",
            "study_plan",
            "citations",
        ],
    },
}


RECORD_REVIEW_TOOL: dict[str, Any] = {
    "name": "record_review",
    "description": (
        "Record a structured review of a hypothesis. Every claim in `evidence[]` "
        "must include a URL and a verbatim excerpt; the URL must have appeared in "
        "your tool_result outputs. Pick exactly one verdict."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [
                    "already_explained",
                    "other_more_likely",
                    "missing_piece",
                    "neutral",
                    "disproved",
                ],
            },
            "kind": {
                "type": "string",
                "enum": ["full", "verification", "observation", "simulation"],
                "description": "Which review mode you ran.",
            },
            "novelty": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Decimal score from 0 to 1; do not use 1-5, 1-10, percentages, or 0-100.",
            },
            "correctness": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Decimal score from 0 to 1; do not use 1-5, 1-10, percentages, or 0-100.",
            },
            "testability": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Decimal score from 0 to 1; do not use 1-5, 1-10, percentages, or 0-100.",
            },
            "feasibility": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Decimal score from 0 to 1; do not use 1-5, 1-10, percentages, or 0-100.",
            },
            "assumptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "assumption": {"type": "string"},
                        "plausibility": {
                            "type": "string",
                            "enum": ["plausible", "uncertain", "implausible"],
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["assumption", "plausibility", "rationale"],
                },
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "url": {"type": "string"},
                        "excerpt": {"type": "string"},
                    },
                    "required": ["claim", "url", "excerpt"],
                },
            },
            "notes": {
                "type": "string",
                "description": "Anything that didn't fit the structured fields.",
            },
        },
        "required": [
            "verdict",
            "kind",
            "novelty",
            "correctness",
            "testability",
            "feasibility",
            "evidence",
        ],
    },
}


RECORD_SYSTEM_FEEDBACK_TOOL: dict[str, Any] = {
    "name": "record_system_feedback",
    "description": "Record a structured meta-review of the session's reviews + debates.",
    "input_schema": {
        "type": "object",
        "properties": {
            "common_weaknesses": {"type": "array", "items": {"type": "string"}},
            "common_strengths": {"type": "array", "items": {"type": "string"}},
            "suggested_focus_areas": {"type": "array", "items": {"type": "string"}},
            "narrative": {"type": "string"},
        },
        "required": ["narrative"],
    },
}


RECORD_RESEARCH_PLAN_TOOL: dict[str, Any] = {
    "name": "record_research_plan",
    "description": "Record the parsed research plan derived from the scientist's goal.",
    "input_schema": {
        "type": "object",
        "properties": {
            "objective": {"type": "string"},
            "preferences": {"type": "array", "items": {"type": "string"}},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "idea_attributes": {"type": "array", "items": {"type": "string"}},
            "domain_hint": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["objective", "preferences", "idea_attributes"],
    },
}


RECORD_LITERATURE_SELECTION_TOOL: dict[str, Any] = {
    "name": "record_literature_selection",
    "description": (
        "Select which search results are relevant enough to show to the calling agent, "
        "and which selected direct-PDF records deserve full-text download."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "selected_results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "Candidate index from the provided search-result list.",
                        },
                        "title": {"type": "string"},
                        "priority": {
                            "type": "string",
                            "enum": ["core", "exploratory", "negative_evidence"],
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief relevance rationale based on title/abstract/snippet.",
                        },
                        "download_pdf": {
                            "type": "boolean",
                            "description": "True for selected direct-PDF records by default; false only for no direct PDF, duplicates, weak context-only background, or exhausted PDF caps.",
                        },
                    },
                    "required": ["index", "title", "priority", "reason", "download_pdf"],
                },
            },
            "rejected_count": {"type": "integer", "minimum": 0},
            "notes": {"type": "string"},
        },
        "required": ["selected_results"],
    },
}
