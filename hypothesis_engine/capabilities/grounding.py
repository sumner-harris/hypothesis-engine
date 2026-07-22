"""Deterministic validation of capability references in hypothesis study plans."""

from __future__ import annotations

from typing import Any

from ..config import Config
from .catalog import CapabilityCatalog
from .models import (
    Capability,
    CapabilityComponentReport,
    CapabilityGroundingIssue,
    CapabilityGroundingReport,
    CapabilityReference,
)


def annotate_hypothesis_record(
    cfg: Config,
    record: dict[str, Any],
    *,
    catalog: CapabilityCatalog | None = None,
) -> CapabilityGroundingReport | None:
    """Attach an authoritative grounding report when the catalog is enabled."""
    if not cfg.capabilities.enabled:
        return None
    active_catalog = catalog or CapabilityCatalog.from_config(cfg)
    report = validate_study_plan(active_catalog, record.get("study_plan"))
    record["capability_grounding"] = report.model_dump()
    return report


def grounding_policy_error(
    cfg: Config,
    report: CapabilityGroundingReport | None,
) -> str | None:
    if not cfg.capabilities.enabled or cfg.capabilities.grounding_policy != "required":
        return None
    if report is None:
        return "required capability grounding was not evaluated"
    missing = [item.component_id for item in report.components if not item.capability_ids]
    if report.error_count or missing or not report.referenced_capability_ids:
        detail = "; ".join(issue.message for issue in report.issues[:5])
        if missing:
            detail = f"ungrounded components: {', '.join(missing)}; {detail}".strip("; ")
        return (
            f"capability grounding policy rejected status={report.status}: "
            f"{detail or 'no validated capability references'}"
        )
    return None


def render_capability_references_md(component: dict[str, Any]) -> str:
    refs = component.get("capability_refs")
    if not isinstance(refs, list) or not refs:
        return ""
    lines = ["**Capability references.**"]
    for raw in refs:
        if not isinstance(raw, dict):
            continue
        capability_id = str(raw.get("capability_id") or "(missing id)")
        version = str(raw.get("version") or "").strip()
        purpose = str(raw.get("purpose") or "").strip()
        label = f"`{capability_id}`"
        if version:
            label += f" version `{version}`"
        line = f"- {label}"
        if purpose:
            line += f": {purpose}"
        lines.append(line)
        parameters = raw.get("parameters")
        if isinstance(parameters, list):
            for parameter in parameters:
                if not isinstance(parameter, dict):
                    continue
                name = str(parameter.get("name") or "").strip()
                value = parameter.get("value")
                unit = str(parameter.get("unit") or "").strip()
                if name:
                    lines.append(f"  - {name}: {value}{f' {unit}' if unit else ''}")
    return "\n".join(lines) if len(lines) > 1 else ""


def render_capability_grounding_md(raw_report: Any) -> str:
    if not isinstance(raw_report, dict):
        return ""
    status = str(raw_report.get("status") or "unknown")
    revision = str(raw_report.get("catalog_revision") or "unknown")
    ids = raw_report.get("referenced_capability_ids")
    issues = raw_report.get("issues")
    lines = [
        "## Capability grounding",
        f"**Status.** {status}",
        f"**Catalog revision.** `{revision}`",
    ]
    if isinstance(ids, list) and ids:
        lines.append("**Referenced capabilities.** " + ", ".join(f"`{item}`" for item in ids))
    if isinstance(issues, list) and issues:
        lines.append("**Validation issues.**")
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            severity = str(issue.get("severity") or "warning")
            component = str(issue.get("component_id") or "").strip()
            capability = str(issue.get("capability_id") or "").strip()
            scope = "/".join(item for item in (component, capability) if item)
            prefix = f"{severity} ({scope})" if scope else severity
            lines.append(f"- {prefix}: {issue.get('message', '')}")
    return "\n\n".join(lines)


def validate_study_plan(
    catalog: CapabilityCatalog,
    study_plan: Any,
) -> CapabilityGroundingReport:
    issues: list[CapabilityGroundingIssue] = []
    components: list[CapabilityComponentReport] = []
    referenced_ids: list[str] = []
    referenced_set: set[str] = set()
    resolved: list[tuple[str, CapabilityReference, Capability]] = []

    if not isinstance(study_plan, list) or not study_plan:
        issues.append(_issue("error", "missing_study_plan", "study_plan must be a non-empty array"))
        return CapabilityGroundingReport(
            catalog_revision=catalog.revision,
            status="ungrounded",
            issues=issues,
        )

    for index, component in enumerate(study_plan, 1):
        if not isinstance(component, dict):
            component_id = f"component_{index}"
            issues.append(
                _issue(
                    "error",
                    "invalid_component",
                    "study_plan component must be an object",
                    component_id=component_id,
                )
            )
            components.append(
                CapabilityComponentReport(
                    component_id=component_id, capability_ids=[], status="ungrounded"
                )
            )
            continue

        component_id = str(component.get("component_id") or f"component_{index}").strip()
        raw_refs = component.get("capability_refs")
        if not isinstance(raw_refs, list) or not raw_refs:
            issues.append(
                _issue(
                    "warning",
                    "component_without_capability",
                    "work package has no catalog-backed capability reference",
                    component_id=component_id,
                )
            )
            components.append(
                CapabilityComponentReport(
                    component_id=component_id, capability_ids=[], status="ungrounded"
                )
            )
            continue

        component_capability_ids: list[str] = []
        component_error_count = 0
        for raw_ref in raw_refs:
            try:
                reference = CapabilityReference.model_validate(raw_ref)
            except Exception as exc:
                component_error_count += 1
                issues.append(
                    _issue(
                        "error",
                        "invalid_capability_reference",
                        f"invalid capability reference: {exc}",
                        component_id=component_id,
                    )
                )
                continue

            capability_id = reference.capability_id
            component_capability_ids.append(capability_id)
            if capability_id not in referenced_set:
                referenced_set.add(capability_id)
                referenced_ids.append(capability_id)
            capability = catalog.get(capability_id)
            if capability is None:
                component_error_count += 1
                issues.append(
                    _issue(
                        "error",
                        "unknown_capability",
                        f"capability {capability_id!r} is not present in catalog {catalog.revision}",
                        component_id=component_id,
                        capability_id=capability_id,
                    )
                )
                continue
            before = len([item for item in issues if item.severity == "error"])
            _validate_reference(component_id, reference, capability, issues)
            after = len([item for item in issues if item.severity == "error"])
            component_error_count += after - before
            resolved.append((component_id, reference, capability))

        components.append(
            CapabilityComponentReport(
                component_id=component_id,
                capability_ids=component_capability_ids,
                status="invalid" if component_error_count else "validated",
            )
        )

    _validate_dependencies(resolved, referenced_set, issues)
    errors = [item for item in issues if item.severity == "error"]
    ungrounded_components = [item for item in components if not item.capability_ids]
    if not referenced_ids:
        status = "ungrounded"
    elif errors:
        status = "invalid"
    elif ungrounded_components or issues:
        status = "partial"
    else:
        status = "validated"
    return CapabilityGroundingReport(
        catalog_revision=catalog.revision,
        status=status,
        referenced_capability_ids=referenced_ids,
        components=components,
        issues=issues,
    )


def _validate_reference(
    component_id: str,
    reference: CapabilityReference,
    capability: Capability,
    issues: list[CapabilityGroundingIssue],
) -> None:
    if reference.version and reference.version != capability.version:
        issues.append(
            _issue(
                "error",
                "version_mismatch",
                f"requested version {reference.version!r}; catalog provides {capability.version!r}",
                component_id=component_id,
                capability_id=capability.id,
            )
        )
    if capability.availability.status in {"unavailable", "planned"}:
        issues.append(
            _issue(
                "error",
                "capability_unavailable",
                f"capability availability is {capability.availability.status}",
                component_id=component_id,
                capability_id=capability.id,
            )
        )
    elif capability.availability.status in {"limited", "unknown"}:
        issues.append(
            _issue(
                "warning",
                "capability_availability_uncertain",
                f"capability availability is {capability.availability.status}",
                component_id=component_id,
                capability_id=capability.id,
            )
        )

    supplied = {item.name.casefold(): item for item in reference.parameters}
    known_parameters = {item.name.casefold() for item in capability.parameters}
    for value in reference.parameters:
        if value.name.casefold() not in known_parameters:
            issues.append(
                _issue(
                    "error",
                    "unknown_parameter",
                    f"parameter {value.name!r} is not defined by this capability",
                    component_id=component_id,
                    capability_id=capability.id,
                )
            )
    for spec in capability.parameters:
        value = supplied.get(spec.name.casefold())
        if value is None:
            if spec.required:
                issues.append(
                    _issue(
                        "error",
                        "missing_required_parameter",
                        f"required parameter {spec.name!r} was not supplied",
                        component_id=component_id,
                        capability_id=capability.id,
                    )
                )
            continue
        if spec.unit and value.unit and spec.unit.casefold() != value.unit.casefold():
            issues.append(
                _issue(
                    "error",
                    "parameter_unit_mismatch",
                    f"parameter {spec.name!r} uses {value.unit!r}; catalog requires {spec.unit!r}",
                    component_id=component_id,
                    capability_id=capability.id,
                )
            )
        elif spec.unit and not value.unit:
            issues.append(
                _issue(
                    "warning",
                    "parameter_unit_missing",
                    f"parameter {spec.name!r} should specify unit {spec.unit!r}",
                    component_id=component_id,
                    capability_id=capability.id,
                )
            )
        if spec.allowed_values and not _allowed_value(value.value, spec.allowed_values):
            issues.append(
                _issue(
                    "error",
                    "parameter_value_not_allowed",
                    f"parameter {spec.name!r} value {value.value!r} is not allowed",
                    component_id=component_id,
                    capability_id=capability.id,
                )
            )
        numeric = _numeric(value.value)
        if numeric is None:
            if spec.minimum is not None or spec.maximum is not None:
                issues.append(
                    _issue(
                        "error",
                        "parameter_not_numeric",
                        f"parameter {spec.name!r} must be numeric for range validation",
                        component_id=component_id,
                        capability_id=capability.id,
                    )
                )
            continue
        if spec.minimum is not None and numeric < spec.minimum:
            issues.append(
                _issue(
                    "error",
                    "parameter_below_minimum",
                    f"parameter {spec.name!r} value {numeric} is below {spec.minimum}",
                    component_id=component_id,
                    capability_id=capability.id,
                )
            )
        if spec.maximum is not None and numeric > spec.maximum:
            issues.append(
                _issue(
                    "error",
                    "parameter_above_maximum",
                    f"parameter {spec.name!r} value {numeric} exceeds {spec.maximum}",
                    component_id=component_id,
                    capability_id=capability.id,
                )
            )


def _validate_dependencies(
    resolved: list[tuple[str, CapabilityReference, Capability]],
    referenced_ids: set[str],
    issues: list[CapabilityGroundingIssue],
) -> None:
    for component_id, _reference, capability in resolved:
        for required_id in capability.requires_capabilities:
            if required_id not in referenced_ids:
                issues.append(
                    _issue(
                        "error",
                        "missing_capability_dependency",
                        f"capability requires {required_id!r}, which is not referenced",
                        component_id=component_id,
                        capability_id=capability.id,
                    )
                )
        for incompatible_id in capability.incompatible_with:
            if incompatible_id in referenced_ids:
                issues.append(
                    _issue(
                        "error",
                        "incompatible_capabilities",
                        f"capability is incompatible with referenced capability {incompatible_id!r}",
                        component_id=component_id,
                        capability_id=capability.id,
                    )
                )


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    component_id: str | None = None,
    capability_id: str | None = None,
) -> CapabilityGroundingIssue:
    return CapabilityGroundingIssue(
        severity=severity,  # type: ignore[arg-type]
        code=code,
        message=message,
        component_id=component_id,
        capability_id=capability_id,
    )


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _allowed_value(value: Any, allowed: list[Any]) -> bool:
    normalized = str(value).strip().casefold()
    return any(normalized == str(candidate).strip().casefold() for candidate in allowed)
