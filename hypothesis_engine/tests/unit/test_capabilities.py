"""Capability catalog, retrieval tools, and deterministic grounding tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from typer.testing import CliRunner

import hypothesis_engine.agents.generation as generation_mod
from hypothesis_engine.agents.supervisor import Supervisor
from hypothesis_engine.capabilities.catalog import (
    CapabilityCatalog,
    CapabilityCatalogValidationError,
    validate_catalog_integrity,
)
from hypothesis_engine.capabilities.grounding import (
    annotate_hypothesis_record,
    grounding_policy_error,
    render_capability_grounding_md,
    render_capability_references_md,
    validate_study_plan,
)
from hypothesis_engine.capabilities.models import Capability
from hypothesis_engine.capabilities.prompting import capability_grounding_requirement
from hypothesis_engine.cli import app
from hypothesis_engine.config import load_config
from hypothesis_engine.logging import setup_logging
from hypothesis_engine.tools.base import ToolCtx
from hypothesis_engine.tools.capabilities import (
    CapabilityGetTool,
    CapabilitySearchTool,
    CapabilityValidateWorkflowTool,
)
from hypothesis_engine.tools.registry import ToolRegistry


def _catalog_path(tmp_path: Path) -> Path:
    path = tmp_path / "capabilities.yaml"
    path.write_text(
        dedent(
            """\
            revision: test-lab-2026-07
            capabilities:
              - id: exp:prep:fib-01
                version: "1.0"
                name: Focused ion beam specimen preparation
                kind: experimental
                description: Prepare electron-transparent lamellae.
                domains: [materials science]
                methods: [FIB lift-out]
                inputs: [bulk solid specimen]
                outputs: [electron-transparent lamella]
                availability:
                  status: available
                provenance: Test facility specification
                last_verified: "2026-07-01"
              - id: exp:microscopy:tem-01
                version: "2.1"
                name: Atomic resolution electron microscopy
                kind: experimental
                description: Atomic-resolution HAADF-STEM imaging and EELS.
                domains: [materials science]
                methods: [HAADF-STEM, EELS]
                inputs: [electron-transparent lamella]
                outputs: [atomic-resolution image, elemental spectrum]
                tags: [microscopy, defects]
                parameters:
                  - name: accelerating_voltage
                    unit: kV
                    minimum: 60
                    maximum: 300
                    required: true
                requires_capabilities: [exp:prep:fib-01]
                availability:
                  status: available
                  location: Test microscopy facility
                provenance: Test facility specification
                last_verified: "2026-07-01"
              - id: ai:vision:segment-01
                version: "2026.1"
                name: Defect image segmentation pipeline
                kind: ai
                description: Segment and quantify defects in microscopy images.
                domains: [materials science, computer vision]
                methods: [semantic segmentation]
                inputs: [microscopy image]
                outputs: [defect mask, defect density]
                availability:
                  status: limited
                provenance: Internal model card
                last_verified: "2026-06-15"
            """
        ),
        encoding="utf-8",
    )
    return path


def _valid_study_plan() -> list[dict]:
    return [
        {
            "component_id": "sample_prep",
            "capability_refs": [
                {
                    "capability_id": "exp:prep:fib-01",
                    "version": "1.0",
                    "purpose": "Prepare an electron-transparent lamella.",
                    "parameters": [],
                }
            ],
        },
        {
            "component_id": "characterization",
            "capability_refs": [
                {
                    "capability_id": "exp:microscopy:tem-01",
                    "version": "2.1",
                    "purpose": "Resolve atomic defects and elemental contrast.",
                    "parameters": [
                        {
                            "name": "accelerating_voltage",
                            "value": 80,
                            "unit": "kV",
                        }
                    ],
                }
            ],
        },
    ]


def _invalid_integrity_catalog_path(tmp_path: Path) -> Path:
    path = tmp_path / "invalid-capabilities.yaml"
    path.write_text(
        dedent(
            """\
            revision: invalid-test
            capabilities:
              - id: sim:model:broken
                version: "1"
                name: Broken simulation
                kind: simulation
                description: A structurally valid record with invalid references.
                requires_capabilities: [sim:model:missing]
                incompatible_with: [sim:model:broken]
                executable_tool: missing_simulation_tool
                provenance: Test fixture
            """
        ),
        encoding="utf-8",
    )
    return path


def _catalog_directory(tmp_path: Path) -> Path:
    root = tmp_path / "capability-inventory"
    (root / "experimental").mkdir(parents=True)
    (root / "simulation").mkdir()
    (root / "ai").mkdir()
    (root / "catalog.yaml").write_text(
        "revision: directory-test-v1\ncapabilities: []\n",
        encoding="utf-8",
    )
    (root / "experimental" / "microscopy.yaml").write_text(
        dedent(
            """\
            id: exp:microscopy:test
            version: "1"
            name: Test microscope
            kind: experimental
            description: Test microscopy capability.
            availability:
              status: available
            provenance: Test fixture
            last_verified: "2026-07-10"
            """
        ),
        encoding="utf-8",
    )
    (root / "simulation" / "models.yml").write_text(
        dedent(
            """\
            capabilities:
              - id: sim:model:test
                version: "2"
                name: Test simulation
                kind: simulation
                description: Test numerical calculation capability.
                availability:
                  status: available
                provenance: Test fixture
                last_verified: "2026-07-10"
            """
        ),
        encoding="utf-8",
    )
    (root / "ai" / "analysis.json").write_text(
        """{
          "id": "ai:analysis:test",
          "version": "3",
          "name": "Test AI analysis",
          "kind": "ai",
          "description": "Test AI analysis capability.",
          "availability": {"status": "available"},
          "provenance": "Test fixture",
          "last_verified": "2026-07-10"
        }""",
        encoding="utf-8",
    )
    return root


def test_catalog_search_and_exact_lookup(tmp_path: Path) -> None:
    catalog = CapabilityCatalog.load(_catalog_path(tmp_path))

    results = catalog.search(
        "atomic microscopy",
        kinds=["experimental"],
        outputs=["elemental"],
    )
    found, missing = catalog.get_many(["exp:microscopy:tem-01", "missing:capability"])

    assert catalog.revision == "test-lab-2026-07"
    assert [item.id for item, _score in results] == ["exp:microscopy:tem-01"]
    assert [item.id for item in found] == ["exp:microscopy:tem-01"]
    assert missing == ["missing:capability"]


@pytest.mark.parametrize("removed_kind", ["theory", "facility", "service"])
def test_removed_capability_kinds_are_rejected(removed_kind: str) -> None:
    with pytest.raises(ValueError):
        Capability.model_validate(
            {
                "id": f"removed:{removed_kind}:test",
                "version": "1",
                "name": "Removed kind",
                "kind": removed_kind,
                "description": "This kind is intentionally unsupported.",
                "provenance": "Test fixture",
            }
        )


def test_catalog_directory_recursively_merges_yaml_and_json_fragments(
    tmp_path: Path,
) -> None:
    root = _catalog_directory(tmp_path)

    catalog = CapabilityCatalog.load(root)

    assert catalog.revision == "directory-test-v1"
    assert catalog.size == 3
    assert {capability.id for capability in catalog.document.capabilities} == {
        "exp:microscopy:test",
        "sim:model:test",
        "ai:analysis:test",
    }
    assert catalog.source_for("sim:model:test") == root / "simulation" / "models.yml"


def test_catalog_directory_reports_duplicate_ids_with_both_sources(
    tmp_path: Path,
) -> None:
    root = _catalog_directory(tmp_path)
    duplicate_path = root / "simulation" / "duplicate.yaml"
    duplicate_path.write_text(
        dedent(
            """\
            id: exp:microscopy:test
            version: "duplicate"
            name: Duplicate microscope
            kind: experimental
            description: Duplicate test record.
            provenance: Test fixture
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        CapabilityCatalog.load(root)

    message = str(exc_info.value)
    assert "duplicate capability id 'exp:microscopy:test'" in message
    assert "duplicate.yaml" in message
    assert "microscopy.yaml" in message


def test_validate_study_plan_checks_versions_ranges_and_dependencies(
    tmp_path: Path,
) -> None:
    catalog = CapabilityCatalog.load(_catalog_path(tmp_path))

    valid = validate_study_plan(catalog, _valid_study_plan())
    invalid_plan = [_valid_study_plan()[1]]
    invalid_plan[0]["capability_refs"][0]["version"] = "1.0"
    invalid_plan[0]["capability_refs"][0]["parameters"][0]["value"] = 500
    invalid_plan[0]["capability_refs"][0]["parameters"].append(
        {"name": "invented_setting", "value": "unsupported"}
    )
    invalid = validate_study_plan(catalog, invalid_plan)

    assert valid.status == "validated"
    assert valid.error_count == 0
    assert invalid.status == "invalid"
    assert {issue.code for issue in invalid.issues if issue.severity == "error"} == {
        "version_mismatch",
        "parameter_above_maximum",
        "unknown_parameter",
        "missing_capability_dependency",
    }


def test_ungrounded_work_package_is_explicit(tmp_path: Path) -> None:
    catalog = CapabilityCatalog.load(_catalog_path(tmp_path))

    report = validate_study_plan(
        catalog,
        [{"component_id": "simulation", "capability_refs": []}],
    )

    assert report.status == "ungrounded"
    assert report.issues[0].code == "component_without_capability"


def test_catalog_integrity_aggregates_cross_record_and_runtime_errors(
    tmp_path: Path,
) -> None:
    catalog = CapabilityCatalog.load(_invalid_integrity_catalog_path(tmp_path))

    report = validate_catalog_integrity(
        catalog,
        registered_tool_names={"web_fetch"},
    )

    assert report.valid is False
    assert report.error_count == 3
    assert report.warning_count == 2
    assert {issue.code for issue in report.issues} == {
        "unknown_availability",
        "missing_last_verified",
        "unknown_dependency",
        "self_incompatibility",
        "unknown_executable_tool",
    }


def test_annotation_rendering_and_required_policy(tmp_path: Path, tmp_cfg) -> None:
    tmp_cfg.capabilities.enabled = True
    tmp_cfg.capabilities.catalog_path = str(_catalog_path(tmp_path))
    tmp_cfg.capabilities.grounding_policy = "required"
    record = {"study_plan": _valid_study_plan()}

    report = annotate_hypothesis_record(tmp_cfg, record)

    assert report is not None
    assert grounding_policy_error(tmp_cfg, report) is None
    assert record["capability_grounding"]["status"] == "validated"
    assert "exp:microscopy:tem-01" in render_capability_grounding_md(record["capability_grounding"])
    refs = render_capability_references_md(record["study_plan"][1])
    assert "accelerating_voltage: 80 kV" in refs

    ungrounded = validate_study_plan(
        CapabilityCatalog.from_config(tmp_cfg),
        [{"component_id": "missing", "capability_refs": []}],
    )
    assert "ungrounded components: missing" in (grounding_policy_error(tmp_cfg, ungrounded) or "")


@pytest.mark.asyncio
async def test_capability_tools_return_structured_results(tmp_path: Path, tmp_cfg) -> None:
    catalog = CapabilityCatalog.load(_catalog_path(tmp_path))
    ctx = ToolCtx(cfg=tmp_cfg)

    search = await CapabilitySearchTool(catalog).call(
        {"query": "defect segmentation", "kinds": ["ai"]},
        ctx,
    )
    get = await CapabilityGetTool(catalog).call(
        {"capability_ids": ["exp:microscopy:tem-01"]},
        ctx,
    )
    validation = await CapabilityValidateWorkflowTool(catalog).call(
        {"study_plan": _valid_study_plan()},
        ctx,
    )

    assert search.content["results"][0]["id"] == "ai:vision:segment-01"
    assert get.content["capabilities"][0]["parameters"][0]["maximum"] == 300
    assert validation.content["status"] == "validated"


def test_registry_exposes_capabilities_only_when_enabled(
    tmp_path: Path,
    tmp_cfg,
) -> None:
    disabled = ToolRegistry(tmp_cfg).discover()
    assert "capability_search" not in {tool.name for tool in disabled.all()}

    tmp_cfg.capabilities.enabled = True
    tmp_cfg.capabilities.catalog_path = str(_catalog_path(tmp_path))
    enabled = ToolRegistry(tmp_cfg).discover()
    names = {tool.name for tool in enabled.all()}

    assert {
        "capability_search",
        "capability_get",
        "capability_validate_workflow",
    } <= names
    for agent in ("generation", "reflection", "evolution"):
        assert "capability_search" in {tool.name for tool in enabled.tools_for(agent)}
    assert "capability_validate_workflow" not in {
        tool.name for tool in enabled.tools_for("reflection")
    }
    for agent in ("generation", "evolution"):
        assert "capability_validate_workflow" in {tool.name for tool in enabled.tools_for(agent)}
    assert "capability_search" not in {tool.name for tool in enabled.tools_for("ranking")}


def test_registry_rejects_integrity_errors_when_capabilities_enabled(
    tmp_path: Path,
    tmp_cfg,
) -> None:
    tmp_cfg.capabilities.enabled = True
    tmp_cfg.capabilities.catalog_path = str(_invalid_integrity_catalog_path(tmp_path))

    with pytest.raises(CapabilityCatalogValidationError) as exc_info:
        ToolRegistry(tmp_cfg).discover()

    assert exc_info.value.report.error_count == 3


@pytest.mark.asyncio
async def test_supervisor_preflight_does_not_create_session_for_invalid_catalog(
    tmp_path: Path,
    tmp_cfg,
    conn,
) -> None:
    tmp_cfg.capabilities.enabled = True
    tmp_cfg.capabilities.catalog_path = str(_invalid_integrity_catalog_path(tmp_path))

    with pytest.raises(CapabilityCatalogValidationError):
        await Supervisor(tmp_cfg).run_session("This session must not be created.")

    async with conn.execute("SELECT COUNT(*) AS n FROM sessions") as cursor:
        row = await cursor.fetchone()
    assert row["n"] == 0


def test_default_config_keeps_capabilities_disabled() -> None:
    cfg = load_config(Path("config/default.toml"))

    assert cfg.capabilities.enabled is False
    assert cfg.capabilities.grounding_policy == "advisory"
    assert cfg.capability_catalog_path.name == "capabilities"
    catalog = CapabilityCatalog.from_config(cfg)
    assert catalog.revision


def test_capabilities_validate_cli_reports_valid_catalog(tmp_path: Path) -> None:
    catalog_path = _catalog_path(tmp_path)
    config_path = tmp_path / "hypothesis-engine.toml"
    config_path.write_text(
        dedent(
            f"""\
            [capabilities]
            enabled = false
            catalog_path = {str(catalog_path)!r}
            """
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["--config", str(config_path), "capabilities", "validate"],
    )
    setup_logging("INFO")

    assert result.exit_code == 0
    assert "Capability catalog is valid" in result.output
    assert "capabilities=3" in result.output


def test_capabilities_validate_cli_reports_all_integrity_errors(
    tmp_path: Path,
) -> None:
    catalog_path = _invalid_integrity_catalog_path(tmp_path)
    config_path = tmp_path / "hypothesis-engine.toml"
    config_path.write_text(
        dedent(
            f"""\
            [capabilities]
            enabled = false
            catalog_path = {str(catalog_path)!r}
            """
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["--config", str(config_path), "capabilities", "validate", "--json"],
    )
    setup_logging("INFO")

    assert result.exit_code == 1
    assert result.output.count('"severity": "error"') == 3
    assert "missing_simulation_tool" in result.output


def test_rag_generation_uses_capability_discovery_before_synthesis() -> None:
    tools = [
        {"name": "arxiv_search"},
        {"name": "rag_retrieve_context"},
        {"name": "rag_kb_status"},
        {"name": "capability_search"},
        {"name": "capability_get"},
        {"name": "capability_validate_workflow"},
    ]

    discovery_names = {tool["name"] for tool in generation_mod._without_rag_tools(tools)}
    synthesis_names = {tool["name"] for tool in generation_mod._only_rag_tools(tools)}

    assert discovery_names == {
        "arxiv_search",
        "capability_search",
        "capability_get",
    }
    assert synthesis_names == {
        "rag_retrieve_context",
        "rag_kb_status",
        "capability_search",
        "capability_get",
        "capability_validate_workflow",
    }


def test_capability_grounding_connects_catalog_records_to_literature() -> None:
    tools = [
        {"name": "arxiv_search"},
        {"name": "capability_search"},
        {"name": "capability_get"},
        {"name": "capability_validate_workflow"},
    ]

    requirement = capability_grounding_requirement(
        tools,
        activity="reviewing a hypothesis",
    )

    assert "# Capability-grounding requirement" in requirement
    assert "# Capability-application evidence requirement" in requirement
    assert "graphene" in requirement
    assert "micro-Raman spectroscopy" in requirement
    assert "do not use literature to upgrade local availability" in requirement
