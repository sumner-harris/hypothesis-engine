"""Shared prompt guidance for capability-grounded agent work."""

from __future__ import annotations

from typing import Any

CAPABILITY_TOOL_NAMES = {
    "capability_search",
    "capability_get",
    "capability_validate_workflow",
}

CAPABILITY_DISCOVERY_TOOL_NAMES = {
    "capability_search",
    "capability_get",
}

LITERATURE_SEARCH_TOOL_NAMES = {
    "arxiv_search",
    "biorxiv_search",
    "chemrxiv_search",
    "europe_pmc_search",
    "pubmed_search",
    "web_search",
}


def capability_tools_present(tools: list[dict[str, Any]]) -> bool:
    names = {str(tool.get("name") or "") for tool in tools if isinstance(tool, dict)}
    return names >= CAPABILITY_TOOL_NAMES


def capability_grounding_requirement(
    tools: list[dict[str, Any]],
    *,
    activity: str,
) -> str:
    if not capability_tools_present(tools):
        return ""
    requirement = (
        "# Capability-grounding requirement\n"
        f"Use the configured capability catalog while {activity}. Search with "
        "`capability_search`, inspect exact records and limits with `capability_get`, "
        "and reference only capability IDs and versions returned by those tools. For "
        "each executable `study_plan` work package, populate `capability_refs` with "
        "the selected capability's purpose and concrete parameter values. Run "
        "`capability_validate_workflow` on the complete study_plan before the terminal "
        "tool call and repair reported errors. Treat catalog availability, operating "
        "ranges, dependencies, constraints, provenance, and last-verified dates as "
        "authoritative. Do not infer availability from literature and do not invent "
        "capability IDs. When the catalog has no suitable capability, leave that work "
        "package ungrounded and state the capability gap explicitly instead of claiming "
        "that the workflow is feasible."
    )
    application_evidence = capability_application_evidence_requirement(
        tools,
        activity=activity,
    )
    if application_evidence:
        requirement = f"{requirement}\n\n{application_evidence}"
    return requirement


def capability_application_evidence_requirement(
    tools: list[dict[str, Any]],
    *,
    activity: str,
) -> str:
    names = {str(tool.get("name") or "") for tool in tools if isinstance(tool, dict)}
    if not names >= CAPABILITY_DISCOVERY_TOOL_NAMES:
        return ""
    search_tools = sorted(names & LITERATURE_SEARCH_TOOL_NAMES)
    if not search_tools:
        return ""
    return (
        "# Capability-application evidence requirement\n"
        f"While {activity}, connect relevant local capabilities to domain evidence, "
        "not just to generic workflow labels. Start with `capability_search`, then use "
        "`capability_get` for the most relevant records. Prioritize capabilities whose "
        "catalog availability is `available` or `limited`; treat `unknown` as conditional "
        "and do not use literature to upgrade local availability. For up to three selected "
        "capabilities, run at least one focused literature query that combines the research "
        "system, material, or target with the capability method and intended observable. "
        "For example, if the system is graphene and micro-Raman is relevant, search for how "
        "micro-Raman spectroscopy is used to characterize graphene, its defects, strain, "
        "doping, and layer number. Use the results to specify suitable inputs or specimen "
        "preparation, observables, controls, interpretation limits, and failure modes. "
        f"Available literature search tools are: {', '.join(f'`{name}`' for name in search_tools)}. "
        "Record negative or missing application evidence as a gap instead of inventing a use."
    )
