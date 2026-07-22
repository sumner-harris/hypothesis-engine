"""Catalog loading, exact lookup, and compact filtered search."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from ..config import Config
from .models import (
    Capability,
    CapabilityCatalogDocument,
    CapabilityCatalogIssue,
    CapabilityCatalogValidationReport,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.-]*")


class CapabilityCatalog:
    def __init__(
        self,
        document: CapabilityCatalogDocument,
        *,
        source_path: Path,
        record_sources: dict[str, Path] | None = None,
    ) -> None:
        self.document = document
        self.source_path = source_path
        self._by_id = {item.id: item for item in document.capabilities}
        self._record_sources = record_sources or {
            item.id: source_path for item in document.capabilities
        }

    @property
    def revision(self) -> str:
        return self.document.revision

    @property
    def size(self) -> int:
        return len(self._by_id)

    def source_for(self, capability_id: str) -> Path | None:
        return self._record_sources.get(capability_id)

    @classmethod
    def load(cls, path: Path) -> CapabilityCatalog:
        if path.is_dir():
            return cls._load_directory(path)
        return cls._load_file(path)

    @classmethod
    def _load_file(cls, path: Path) -> CapabilityCatalog:
        raw = _read_serialized(path)
        if raw is None:
            raw = {}
        try:
            document = CapabilityCatalogDocument.model_validate(raw)
        except Exception as exc:
            raise ValueError(f"invalid capability catalog {path}: {exc}") from exc
        return cls(document, source_path=path)

    @classmethod
    def _load_directory(cls, path: Path) -> CapabilityCatalog:
        manifest = _find_manifest(path)
        manifest_raw = _read_serialized(manifest)
        if manifest_raw is None:
            manifest_raw = {}
        try:
            manifest_document = CapabilityCatalogDocument.model_validate(manifest_raw)
        except Exception as exc:
            raise ValueError(f"invalid capability catalog manifest {manifest}: {exc}") from exc

        capabilities = list(manifest_document.capabilities)
        record_sources = {item.id: manifest for item in capabilities}
        errors: list[str] = []
        fragment_paths = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and candidate != manifest
            and candidate.suffix.casefold() in {".yaml", ".yml", ".json"}
            and not any(part.startswith(".") for part in candidate.relative_to(path).parts)
        )
        for fragment in fragment_paths:
            try:
                records = _fragment_capabilities(fragment)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            for capability in records:
                previous = record_sources.get(capability.id)
                if previous is not None:
                    errors.append(
                        f"{fragment}: duplicate capability id {capability.id!r}; "
                        f"already defined in {previous}"
                    )
                    continue
                capabilities.append(capability)
                record_sources[capability.id] = fragment
        if errors:
            joined = "\n- ".join(errors)
            raise ValueError(f"invalid capability catalog directory {path}:\n- {joined}")
        document = CapabilityCatalogDocument(
            revision=manifest_document.revision,
            capabilities=capabilities,
        )
        return cls(
            document,
            source_path=path,
            record_sources=record_sources,
        )

    @classmethod
    def from_config(cls, cfg: Config) -> CapabilityCatalog:
        return cls.load(cfg.capability_catalog_path)

    def get(self, capability_id: str) -> Capability | None:
        return self._by_id.get(capability_id)

    def get_many(self, capability_ids: list[str]) -> tuple[list[Capability], list[str]]:
        found: list[Capability] = []
        missing: list[str] = []
        for capability_id in dict.fromkeys(capability_ids):
            item = self.get(capability_id)
            if item is None:
                missing.append(capability_id)
            else:
                found.append(item)
        return found, missing

    def search(
        self,
        query: str,
        *,
        kinds: list[str] | None = None,
        domains: list[str] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        availability: list[str] | None = None,
        limit: int = 8,
    ) -> list[tuple[Capability, float]]:
        query_tokens = _tokens(query)
        query_phrase = _normalize(query)
        kind_filter = {item.casefold() for item in kinds or []}
        availability_filter = {item.casefold() for item in availability or []}
        results: list[tuple[Capability, float]] = []
        for capability in self.document.capabilities:
            if kind_filter and capability.kind.casefold() not in kind_filter:
                continue
            if (
                availability_filter
                and capability.availability.status.casefold() not in availability_filter
            ):
                continue
            if not _matches_any(domains, capability.domains):
                continue
            if not _matches_any(inputs, capability.inputs):
                continue
            if not _matches_any(outputs, capability.outputs):
                continue
            score = _search_score(capability, query_tokens, query_phrase)
            if query_tokens and score <= 0:
                continue
            results.append((capability, score))
        results.sort(key=lambda item: (-item[1], item[0].name.casefold(), item[0].id))
        return results[: max(1, min(int(limit), 50))]


def _read_serialized(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read capability catalog {path}: {exc}") from exc
    try:
        raw = json.loads(text) if path.suffix.casefold() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot parse capability catalog {path}: {exc}") from exc
    return raw


def _find_manifest(path: Path) -> Path:
    candidates = [
        candidate
        for name in ("catalog.yaml", "catalog.yml", "catalog.json")
        if (candidate := path / name).is_file()
    ]
    if not candidates:
        raise ValueError(
            f"capability catalog directory {path} must contain catalog.yaml, "
            "catalog.yml, or catalog.json"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"capability catalog directory {path} contains multiple manifests: "
            + ", ".join(str(candidate.name) for candidate in candidates)
        )
    return candidates[0]


def _fragment_capabilities(path: Path) -> list[Capability]:
    raw = _read_serialized(path)
    if isinstance(raw, dict) and set(raw) == {"capabilities"}:
        raw_records = raw["capabilities"]
    elif isinstance(raw, list):
        raw_records = raw
    else:
        raw_records = [raw]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"{path}: capability fragment must contain at least one record")
    records: list[Capability] = []
    errors: list[str] = []
    for index, raw_record in enumerate(raw_records):
        try:
            records.append(Capability.model_validate(raw_record))
        except Exception as exc:
            errors.append(f"record {index + 1}: {exc}")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return records


class CapabilityCatalogValidationError(ValueError):
    """Raised when a configured catalog fails aggregate integrity checks."""

    def __init__(self, report: CapabilityCatalogValidationReport) -> None:
        self.report = report
        details = "; ".join(
            (f"{issue.capability_id or 'catalog'}.{issue.field or issue.code}: {issue.message}")
            for issue in report.issues
            if issue.severity == "error"
        )
        super().__init__(
            f"capability catalog {report.source_path} failed validation with "
            f"{report.error_count} error(s): {details}"
        )


def validate_catalog_integrity(
    catalog: CapabilityCatalog,
    *,
    registered_tool_names: set[str] | None = None,
) -> CapabilityCatalogValidationReport:
    """Collect cross-record and runtime integration issues without failing fast."""
    issues: list[CapabilityCatalogIssue] = []
    known_ids = {capability.id for capability in catalog.document.capabilities}
    if not known_ids:
        issues.append(
            _catalog_issue(
                "warning",
                "empty_catalog",
                "catalog contains no capabilities",
            )
        )

    for capability in catalog.document.capabilities:
        if capability.availability.status == "unknown":
            issues.append(
                _catalog_issue(
                    "warning",
                    "unknown_availability",
                    "availability status is unknown",
                    capability_id=capability.id,
                    field="availability.status",
                )
            )
        if not capability.last_verified:
            issues.append(
                _catalog_issue(
                    "warning",
                    "missing_last_verified",
                    "last_verified is not set",
                    capability_id=capability.id,
                    field="last_verified",
                )
            )
        for required_id in capability.requires_capabilities:
            if required_id == capability.id:
                issues.append(
                    _catalog_issue(
                        "error",
                        "self_dependency",
                        "capability cannot require itself",
                        capability_id=capability.id,
                        field="requires_capabilities",
                    )
                )
            elif required_id not in known_ids:
                issues.append(
                    _catalog_issue(
                        "error",
                        "unknown_dependency",
                        f"referenced capability {required_id!r} does not exist",
                        capability_id=capability.id,
                        field="requires_capabilities",
                    )
                )
        for incompatible_id in capability.incompatible_with:
            if incompatible_id == capability.id:
                issues.append(
                    _catalog_issue(
                        "error",
                        "self_incompatibility",
                        "capability cannot be incompatible with itself",
                        capability_id=capability.id,
                        field="incompatible_with",
                    )
                )
            elif incompatible_id not in known_ids:
                issues.append(
                    _catalog_issue(
                        "error",
                        "unknown_incompatible_capability",
                        f"referenced capability {incompatible_id!r} does not exist",
                        capability_id=capability.id,
                        field="incompatible_with",
                    )
                )
        executable_tool = str(capability.executable_tool or "").strip()
        if (
            executable_tool
            and registered_tool_names is not None
            and executable_tool not in registered_tool_names
        ):
            issues.append(
                _catalog_issue(
                    "error",
                    "unknown_executable_tool",
                    f"registered tool {executable_tool!r} was not found",
                    capability_id=capability.id,
                    field="executable_tool",
                )
            )

    return CapabilityCatalogValidationReport(
        catalog_revision=catalog.revision,
        source_path=str(catalog.source_path),
        capability_count=catalog.size,
        issues=issues,
    )


def require_valid_catalog(
    catalog: CapabilityCatalog,
    *,
    registered_tool_names: set[str] | None = None,
) -> CapabilityCatalogValidationReport:
    report = validate_catalog_integrity(
        catalog,
        registered_tool_names=registered_tool_names,
    )
    if not report.valid:
        raise CapabilityCatalogValidationError(report)
    return report


def _catalog_issue(
    severity: str,
    code: str,
    message: str,
    *,
    capability_id: str | None = None,
    field: str | None = None,
) -> CapabilityCatalogIssue:
    return CapabilityCatalogIssue(
        severity=severity,  # type: ignore[arg-type]
        code=code,
        message=message,
        capability_id=capability_id,
        field=field,
    )


def capability_summary(capability: Capability, *, detailed: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": capability.id,
        "version": capability.version,
        "name": capability.name,
        "kind": capability.kind,
        "description": capability.description,
        "domains": capability.domains,
        "methods": capability.methods,
        "inputs": capability.inputs,
        "outputs": capability.outputs,
        "availability": capability.availability.model_dump(exclude_none=True),
        "last_verified": capability.last_verified,
    }
    if detailed:
        out.update(
            {
                "parameters": [
                    item.model_dump(exclude_none=True) for item in capability.parameters
                ],
                "requires_capabilities": capability.requires_capabilities,
                "requirements": capability.requirements,
                "incompatible_with": capability.incompatible_with,
                "constraints": capability.constraints,
                "safety_notes": capability.safety_notes,
                "executable_tool": capability.executable_tool,
                "owner": capability.owner,
                "provenance": capability.provenance,
            }
        )
    return out


def _matches_any(needles: list[str] | None, values: list[str]) -> bool:
    if not needles:
        return True
    haystack = " ".join(values).casefold()
    return any(str(needle).casefold() in haystack for needle in needles if str(needle).strip())


def _tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN_RE.finditer(text or "")}


def _normalize(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _search_score(capability: Capability, query_tokens: set[str], query_phrase: str) -> float:
    if not query_tokens:
        return 1.0
    weighted_fields = (
        (capability.name, 6.0),
        (capability.id, 5.0),
        (" ".join(capability.methods), 4.0),
        (" ".join(capability.tags), 4.0),
        (" ".join(capability.domains), 3.0),
        (" ".join(capability.inputs), 2.0),
        (" ".join(capability.outputs), 2.0),
        (capability.description, 1.0),
    )
    score = 0.0
    for text, weight in weighted_fields:
        normalized = _normalize(text)
        tokens = _tokens(text)
        score += weight * len(query_tokens & tokens)
        if query_phrase and query_phrase in normalized:
            score += weight * 2.0
    return score
