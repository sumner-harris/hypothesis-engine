"""Pydantic contracts for the capability catalog and grounding reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CapabilityKind = Literal[
    "experimental",
    "simulation",
    "ai",
    "data",
]
AvailabilityStatus = Literal["available", "limited", "planned", "unavailable", "unknown"]
GroundingStatus = Literal["validated", "partial", "invalid", "ungrounded"]
IssueSeverity = Literal["warning", "error"]
Scalar = str | int | float | bool


class CapabilityParameter(BaseModel):
    """Supported parameter or operating range for a capability."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: list[Scalar] = Field(default_factory=list)
    required: bool = False

    @model_validator(mode="after")
    def validate_range(self) -> CapabilityParameter:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"parameter {self.name!r} minimum exceeds maximum")
        return self


class CapabilityAvailability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AvailabilityStatus = "unknown"
    location: str | None = None
    access: str | None = None
    lead_time: str | None = None
    cost: str | None = None
    notes: str | None = None


class Capability(BaseModel):
    """One versioned experimental, simulation, data, or AI capability."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=3, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: CapabilityKind
    description: str = Field(min_length=1)
    domains: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    parameters: list[CapabilityParameter] = Field(default_factory=list)
    requires_capabilities: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    incompatible_with: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    availability: CapabilityAvailability = Field(default_factory=CapabilityAvailability)
    executable_tool: str | None = None
    owner: str | None = None
    provenance: str = Field(min_length=1)
    last_verified: str | None = None

    @model_validator(mode="after")
    def unique_parameters(self) -> Capability:
        names = [item.name.casefold() for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"capability {self.id!r} has duplicate parameter names")
        return self


class CapabilityCatalogDocument(BaseModel):
    """Canonical, versioned catalog file."""

    model_config = ConfigDict(extra="forbid")

    revision: str = Field(min_length=1)
    capabilities: list[Capability] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_capabilities(self) -> CapabilityCatalogDocument:
        ids = [item.id for item in self.capabilities]
        if len(ids) != len(set(ids)):
            raise ValueError("capability ids must be unique within a catalog revision")
        return self


class CapabilityParameterValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: Any
    unit: str | None = None


class CapabilityReference(BaseModel):
    """A study-plan reference to one exact catalog capability."""

    model_config = ConfigDict(extra="forbid")

    capability_id: str = Field(min_length=1)
    version: str | None = None
    purpose: str = Field(min_length=1)
    parameters: list[CapabilityParameterValue] = Field(default_factory=list)


class CapabilityGroundingIssue(BaseModel):
    severity: IssueSeverity
    code: str
    message: str
    component_id: str | None = None
    capability_id: str | None = None


class CapabilityComponentReport(BaseModel):
    component_id: str
    capability_ids: list[str] = Field(default_factory=list)
    status: GroundingStatus


class CapabilityGroundingReport(BaseModel):
    catalog_revision: str
    status: GroundingStatus
    referenced_capability_ids: list[str] = Field(default_factory=list)
    components: list[CapabilityComponentReport] = Field(default_factory=list)
    issues: list[CapabilityGroundingIssue] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)


class CapabilityCatalogIssue(BaseModel):
    severity: IssueSeverity
    code: str
    message: str
    capability_id: str | None = None
    field: str | None = None


class CapabilityCatalogValidationReport(BaseModel):
    catalog_revision: str
    source_path: str
    capability_count: int
    issues: list[CapabilityCatalogIssue] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def valid(self) -> bool:
        return self.error_count == 0
