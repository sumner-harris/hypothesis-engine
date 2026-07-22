# Capability catalog

`config/capabilities/` is the versioned source of truth for experimental,
simulation, AI, and data capabilities available to the
hypothesis-engine. Capability tools are disabled by default; populate the inventory and
set `[capabilities] enabled = true` in an override config to expose them to
Generation, Reflection, and Evolution.

Validate the configured catalog before running a session:

```bash
hypothesis-engine capabilities validate
# Machine-readable report:
hypothesis-engine capabilities validate --json
```

The command validates the catalog even when capability tools are disabled. When
capabilities are enabled, the Supervisor performs the same validation before
opening the database or creating a session row.

Each record must use a stable `id` and explicit `version`. Operational limits,
availability, provenance, and `last_verified` should describe the actual local
resource rather than a capability inferred from literature.

## How agents use the catalog

Generation first searches the catalog for capabilities relevant to the research
objective, inspects exact records, and then runs focused literature searches on
how selected methods apply to the target system. It records the actual queries,
application evidence, observables, limitations, and capability gaps before
hypothesis synthesis. Reflection and Evolution receive the same instruction
when their catalog and literature tools are available, so later stages can
audit or improve the proposed use.

The catalog remains authoritative for local availability, operating ranges,
dependencies, and access constraints. Literature can support how a capability
should be applied, but it cannot establish that the capability is locally
available or expand its validated parameter limits.

## Required fields

Directory mode requires a root `catalog.yaml`, `catalog.yml`, or
`catalog.json` manifest. The manifest owns the required `revision`. Its
`capabilities` array is optional and normally remains empty because records
live in fragments:

```yaml
revision: lab-2026-07-10
capabilities: []
```

Every capability record requires exactly these fields:

```yaml
- id: exp:microscopy:tem-01
  version: "2026.1"
  name: Aberration-corrected TEM
  kind: experimental
  description: Atomic-resolution imaging with local spectroscopy.
  provenance: Internal facility specification, SOP TEM-04
```

`kind` must be one of `experimental`, `simulation`, `ai`, or `data`.
Use `simulation` for analytical or numerical
calculation capabilities as well as computational simulations. All other fields shown below
are optional. Unknown or misspelled fields are rejected rather than silently
ignored.

## Catalog field reference

Required fields must appear in the YAML record. Optional fields use the stated
default when omitted. Strings marked nonempty must contain at least one
character.

### Catalog manifest

| Field | Required | Type, allowed values, and default |
| --- | --- | --- |
| `revision` | Yes | Nonempty string identifying the catalog revision. |
| `capabilities` | No | List of capability records; defaults to `[]` and normally remains empty in directory mode. |

### Capability record

| Field | Required | Type, allowed values, and default |
| --- | --- | --- |
| `id` | Yes | Unique string of at least three characters. It must start with a letter or number and may then contain letters, numbers, `_`, `.`, `:`, or `-`. |
| `version` | Yes | Nonempty string. Quote numeric-looking versions, for example `"2026.1"`. |
| `name` | Yes | Nonempty human-readable string. It does not have to be unique. |
| `kind` | Yes | One of `experimental`, `simulation`, `ai`, or `data`. |
| `description` | Yes | Nonempty string describing the capability. |
| `domains` | No | List of strings; defaults to `[]`. |
| `methods` | No | List of strings; defaults to `[]`. |
| `inputs` | No | List of consumed materials, specimens, files, or data; defaults to `[]`. |
| `outputs` | No | List of produced materials, measurements, files, or data; defaults to `[]`. |
| `tags` | No | List of search terms; defaults to `[]`. |
| `parameters` | No | List of parameter records; defaults to `[]`. Parameter names must be unique within the capability, case-insensitively. |
| `requires_capabilities` | No | List of capability IDs that must also be referenced; defaults to `[]`. Each ID must exist and cannot be the record's own ID. |
| `requirements` | No | List of prerequisites such as training, sample preparation, or software access; defaults to `[]`. |
| `incompatible_with` | No | List of capability IDs that cannot be used in the same workflow; defaults to `[]`. Each ID must exist and cannot be the record's own ID. |
| `constraints` | No | List of free-text limitations not represented by structured parameters; defaults to `[]`. |
| `safety_notes` | No | List of safety statements; defaults to `[]`. |
| `availability` | No | Availability record; defaults to `{status: unknown}`. |
| `executable_tool` | No | Registered tool name or `null`; defaults to `null`. A nonempty name must resolve to a registered tool. |
| `owner` | No | Responsible person, group, or facility name, or `null`; defaults to `null`. |
| `provenance` | Yes | Nonempty authoritative source string, such as an SOP, instrument specification, dataset DOI, or internal record. |
| `last_verified` | No | Verification date or revision string, or `null`; defaults to `null`. An omitted value produces a catalog warning. |

Capability IDs must be unique across the manifest and all fragments. ID
matching is exact, so use a consistent lowercase convention.

### Parameter record

| Field | Required | Type, allowed values, and default |
| --- | --- | --- |
| `name` | Yes | Nonempty parameter name. |
| `description` | No | String; defaults to an empty string. |
| `unit` | No | Unit string or `null`; defaults to `null`. |
| `minimum` | No | Inclusive number or `null`; defaults to `null`. |
| `maximum` | No | Inclusive number or `null`; defaults to `null` and cannot be less than `minimum`. |
| `allowed_values` | No | List containing string, integer, floating-point, or Boolean values; defaults to `[]`. |
| `required` | No | Boolean; defaults to `false`. When `true`, an agent reference must supply the parameter. |

### Availability record

| Field | Required | Type, allowed values, and default |
| --- | --- | --- |
| `status` | No | One of `available`, `limited`, `planned`, `unavailable`, or `unknown`; defaults to `unknown`. |
| `location` | No | String or `null`; defaults to `null`. |
| `access` | No | String or `null`; defaults to `null`. |
| `lead_time` | No | String or `null`; defaults to `null`. |
| `cost` | No | String or `null`; defaults to `null`. |
| `notes` | No | String or `null`; defaults to `null`. |

An `unknown` status produces a catalog warning. During workflow grounding,
`planned` and `unavailable` are errors, while `limited` and `unknown` produce
warnings.

## Parameter limits

State continuous operating limits with numeric `minimum` and `maximum` values
and put the unit in the separate `unit` field:

```yaml
parameters:
  - name: accelerating_voltage
    description: Validated instrument operating range.
    unit: kV
    minimum: 60
    maximum: 300
    required: true
```

The bounds are inclusive. Do not encode a range or unit in a string such as
`"60-300 kV"`; range validation requires numeric bounds. A one-sided limit is
supported by omitting the unbounded field:

```yaml
parameters:
  - name: minimum_sample_size
    unit: mm
    minimum: 2
```

For a discrete option set, use `allowed_values` instead of a numeric range:

```yaml
parameters:
  - name: exchange_correlation_functional
    allowed_values: [PBE, PBEsol, SCAN]
```

Units are compared by name, case-insensitively; the validator does not perform
unit conversion. For example, `mTorr` and `Pa` are different units. A
hypothesis value below `minimum`, above `maximum`, outside `allowed_values`, or
expressed in a different unit is invalid. Omitting a catalog-specified unit
from a hypothesis value produces a warning. Put conditional limits in the
parameter `description` or capability `constraints`, and only add bounds that
come from an authoritative local specification.

## Agent reference field reference

The hypothesis-facing `capability_refs` contract is separate from catalog
records. Agents emit these values; catalog authors do not add them to a
capability fragment.

| Field | Required | Type, allowed values, and default |
| --- | --- | --- |
| `capability_id` | Yes | Nonempty string that exactly matches a catalog capability ID. |
| `version` | No | Catalog version string or `null`; defaults to `null`. Supplying it is strongly recommended, and a mismatch is invalid. |
| `purpose` | Yes | Nonempty string explaining the role of the capability in the work package. |
| `parameters` | No | List of parameter values; defaults to `[]`. |

Each referenced parameter has a required nonempty `name`, a required `value`
of any YAML-compatible type, and an optional `unit` string. The `name` must
match a parameter declared by the referenced capability, case-insensitively;
undeclared parameter names are invalid. Generated grounding and validation
report models are output contracts and are not fields that catalog authors
populate.

## Directory layout

The loader recursively reads `*.yaml`, `*.yml`, and `*.json` below the
catalog directory, excluding the root manifest. The category directories are
organizational; capability `kind` is still determined by the record.

```text
config/capabilities/
  catalog.yaml
  experimental/
    tem-01.yaml
  simulation/
    dft-vasp.yaml
  ai/
    defect-segmentation.yaml
  data/
```

The recommended format is one capability object per fragment. A fragment may
also contain a YAML list or a `capabilities: [...]` wrapper when several
records must be maintained together. Capability IDs must be unique across the
manifest and every fragment. Validation errors include the source filename.

## Complete fragment example

```yaml
id: exp:microscopy:tem-01
version: "2026.1"
name: Aberration-corrected TEM
kind: experimental
description: Atomic-resolution imaging with local spectroscopy.
domains: [materials science]
methods: [HAADF-STEM, EELS]
inputs: [electron-transparent solid specimen]
outputs: [atomic-resolution image, elemental spectrum]
tags: [microscopy, characterization]
parameters:
  - name: accelerating_voltage
    unit: kV
    minimum: 60
    maximum: 300
    required: true
requirements: [Completed instrument training, approved specimen]
constraints: [Specimen thickness must support electron transmission]
safety_notes: []
availability:
  status: limited
  location: Shared microscopy facility
  access: Operator-assisted reservation
  lead_time: 2-4 weeks
owner: Facility manager
provenance: Internal facility specification
last_verified: "2026-07-10"
```

For backward compatibility, `catalog_path` may still point to one YAML or JSON
file containing both `revision` and `capabilities`.

Study-plan work packages reference records through `capability_refs`:

```yaml
capability_refs:
  - capability_id: exp:microscopy:tem-01
    version: "2026.1"
    purpose: Resolve defect structure and elemental contrast.
    parameters:
      - name: accelerating_voltage
        value: 80
        unit: kV
```
