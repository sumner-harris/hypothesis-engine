# Hypothesis Engine

Hypothesis Engine is a Python package and CLI for running a multi-agent research-hypothesis workflow. It began as an independent implementation inspired by Google's published multi-agent scientific-discovery system, but has diverged substantially: the default path now targets local/OpenAI-compatible model servers, optional RAG-backed literature retrieval, debate-style hypothesis generation and ranking, cluster-aware evolution, live session monitoring, and post-hoc analysis reports.

The distribution and CLI command are named `hypothesis-engine`; the Python import package is `hypothesis_engine`.

This repository intentionally does not include run data, databases, generated analysis reports, downloaded papers, FAISS indexes, or benchmark result artifacts. Those are produced locally under `data/` when you run sessions.

## What It Does

A session starts from a natural-language research goal and runs a durable SQLite-backed agent loop. The system can be driven from the CLI or the FastAPI web UI. Sessions can be paused, resumed, inspected, analyzed, and benchmarked.

## Writing A Good Session Prompt

The web UI exposes two text fields: `Research goal` and `Preferences`. They are rendered as separate labeled sections inside one `parse_goal` prompt, then the Supervisor stores the parsed `ResearchPlan` for the downstream agents. Put the scientific problem definition in `Research goal`; put evaluation criteria and style preferences in `Preferences`.

The `parse_goal` step can infer missing preferences and idea attributes, but runs are usually better when the user supplies the important structure explicitly. A useful `Research goal` format is:

```text
Objective: What should the system investigate or explain?
System/domain: What organism, material, dataset, method, or application area is in scope?
Background/context: What facts, assumptions, or prior observations should the agents treat as important?
Mechanism or intervention space: What mechanisms, variables, designs, or causal routes should be considered?
Scope and constraints: What should be included, excluded, avoided, or held fixed?
Evidence/work plan expectations: What experiments, simulations, measurements, benchmarks, or analyses should a good hypothesis include?
Success and failure criteria: What outcomes would support, falsify, or bound the hypothesis?
```

A useful `Preferences` format is:

```text
Prioritize: Mechanistic specificity, novelty, feasibility, quantitative predictions, or other qualities.
Evidence standards: Required literature types, full-text support, negative evidence, controls, or validation methods.
Diversity: How hypotheses should differ from each other, such as distinct mechanisms, disciplines, scales, or methods.
Output emphasis: Any domain-specific work packages the final hypotheses should include.
```

All sections are optional. For short exploratory runs, a clear objective plus 2-4 preferences is enough. For serious literature-grounded runs, include constraints, desired evidence, and success/failure criteria so Generation, Reflection, Evolution, and Meta-review all optimize against the same target.

## Workflow And Agent Behavior

The workflow is task-driven: the Supervisor stores tasks in SQLite, leases runnable work to agents, records transcripts/artifacts, and schedules follow-on tasks as state changes. The normal flow is:

1. **Supervisor / Parse Goal** creates the session, parses the user goal into a structured `ResearchPlan`, enqueues the initial Generation tasks, leases tasks with heartbeats, observes budget/wall-clock/session controls, and finalizes the run.
2. **GenerationAgent** creates the initial hypothesis pool. In non-RAG mode it runs a compact literature/search tool loop and records one hypothesis. In RAG mode each parallel Generation task first performs lens-specific literature discovery, all discovery outputs share a PDF ingest barrier, and then each task runs a RAG-grounded debate across the combined discovery map before calling `record_hypothesis` once.
3. **LiteratureReviewAgent** is invoked by search tools before search results are returned to Generation, Reflection, or Evolution. It reviews titles/abstracts/snippets, dedupes candidates already reviewed in the session, selects the compact result set the calling agent is allowed to see, and marks selected direct-PDF records for background RAG ingestion. It is not a top-level scheduler task; it appears in transcripts and the web workflow when a search call triggers it.
4. **ReflectionAgent** reviews each hypothesis for novelty, correctness, testability, feasibility, evidence quality, assumptions, and missing work. If RAG is available it is prompted to retrieve context before scoring; it can also call literature/search/fetch tools, whose search results are filtered through LiteratureReviewAgent.
5. **RankingAgent** runs pairwise or debate-style tournament matches, records bounded best-of-N pair outcomes, and updates Elo only when a pair has a resolved winner. It compares each hypothesis together with its latest ReflectionAgent review, including the review verdict and novelty/correctness/testability/feasibility scores. It can apply configurable prompt-side character limits for ranking comparisons, but the defaults keep full hypothesis and review text. It does not receive literature tools during ranking matches.
6. **ProximityAgent** maintains the hypothesis FAISS index used for deduplication, cluster-aware parent selection, hypothesis cluster plots, and informative tournament pair selection. Generation and Evolution also use the vector index during persistence to reject near duplicates.
7. **EvolutionAgent** creates new hypotheses from mature/high-scoring hypotheses using `combine`, `simplify`, `feasibility`, and `out_of_box` strategies. Parent selection is cluster-balanced so evolution does not only recycle near-duplicates from the top Elo slice. Evolved hypotheses go back through Reflection, Ranking, and Proximity like generated hypotheses.
8. **MetaReviewAgent** periodically writes system feedback and produces the final research overview. System feedback is injected into future Generation and Evolution prompts so later ideas can react to run-level weaknesses.

Search/RAG interaction is deliberately indirect: Generation, Reflection, and Evolution may call arXiv, ChemRxiv, bioRxiv, PubMed, Europe PMC, web search, web fetch, and RAG tools when available. Search tools pass candidate records through LiteratureReviewAgent; only selected records enter the caller's chat context, while selected direct-PDF records are queued for RAG ingestion under the session PDF budget. Ranking, Proximity, and Meta-review do not call search tools in their core loops.

## Agent Configuration Map

Most agents are configured by a combination of model routing, output budgets, tool-loop limits, and agent-specific sections in `config/default.toml` or your local override file:

| Agent | Main behavior | Key config sections |
| --- | --- | --- |
| Supervisor / Parse Goal | session creation, task scheduling, leases, parse-goal plan, stop/finalize logic | `[run]`, `[lease]`, `[termination]`, `[models].parse_goal`, `[budget_shares]` |
| GenerationAgent | initial hypotheses, RAG discovery/debate, discovery lenses, required work packages, dedup replacements, malformed `record_hypothesis` recovery | `[generation]`, `[models].generation`, `[thinking].generation_*`, `[tool_loop].generation_max_iters`, `[rag].generation_*`, `config/discovery_profiles/*.yaml` |
| LiteratureReviewAgent | search-result filtering, candidate dedupe, selected context, selected PDFs for ingestion, fallback behavior | `[models].literature_review`, `[budget_shares].literature_review`, `[rag].literature_review_*`, `[rag].auto_ingest_max_pdfs_per_search` |
| ReflectionAgent | hypothesis reviews, RAG/search-supported critique, malformed `record_review` recovery | `[reflection]`, `[models].reflection`, `[thinking].reflection_*`, `[tool_loop].reflection_max_iters` |
| RankingAgent | pair/debate matches, best-of-N pair closure, Elo updates, ranking prompt text limits, ranking batch cadence | `[ranking]`, `[models].ranking_pairwise`, `[models].ranking_debate`, `[models].ranking_priority`, `[thinking].ranking_*`, `[tool_loop].ranking_max_iters` |
| EvolutionAgent | cluster-balanced parent selection, combine/simplify/feasibility/out-of-box strategies, dedup replacements, malformed evolved-hypothesis recovery | `[evolution]`, `[models].evolution`, `[thinking].evolution_*`, `[tool_loop].evolution_max_iters` |
| ProximityAgent | hypothesis embedding index, dedup threshold, clustering, vector-backed pair/evolution support | `[embeddings]`, `[vectors]`, `[budget_shares].proximity` |
| MetaReviewAgent | periodic system feedback and final overview | `[models].metareview_feedback`, `[models].metareview_final`, `[thinking].metareview_*`, `[tool_loop].metareview_max_iters`, `[budget_shares].metareview` |

Tool availability is configured globally but applied per agent. Literature/search/fetch/RAG tools are exposed to Generation, Reflection, and Evolution. Ranking, Proximity, and Meta-review are intentionally kept off literature tools for their core calls.

Ranking prompt text limits are configured in `[ranking]` with `prompt_hypothesis_max_chars`, `prompt_review_max_chars`, and `prompt_side_max_chars`. These three ranking-only character limits use `-1` to mean no clipping, `0` to mean an empty section, and a positive integer to clip to that many characters. This `-1` convention is not global: token budgets, RAG context caps, web-fetch previews, and tool-loop limits keep their own documented semantics.

Review verdicts are produced by ReflectionAgent and carried into the RankingAgent comparison context:

- `already_explained`: the hypothesis is mostly covered by existing literature, so correctness may be reasonable but novelty is weak.
- `other_more_likely`: the hypothesis is possible, but available evidence favors another mechanism or explanation.
- `missing_piece`: the hypothesis is promising but lacks a key assumption, method, control, mechanism, or supporting evidence.
- `neutral`: evidence is mixed or insufficient for a strong positive or negative call.
- `disproved`: strong literature evidence directly contradicts the hypothesis or a central assumption.

These verdicts guide ranking context but do not directly accept or reject hypotheses. Relative standing is still determined by tournament outcomes and Elo updates.


## Current Defaults

`config/default.toml` is tuned for a local model stack:

- LLM provider: `openai_compatible`
- LLM endpoint: `http://localhost:8000/v1`
- Default model ids: `gemma-4-26b-a4b-nvfp4`
- Embedding provider: `openai_compatible`
- Embedding endpoint: `http://localhost:8001/v1`
- Embedding model: `sfr-embedding-mistral`
- RAG: disabled by default
- Runtime data directory: `./data`

If you are not running local OpenAI-compatible servers at those addresses, override the provider, base URL, and model ids before launching a real session.

`config/a100_remote.toml` is an alternate high-concurrency remote config. It enables RAG and points LLM, embedding, and rerank endpoints at a host named `nvidiaspark`. Use it only if that infrastructure exists or after editing the endpoints.

## Install

Recommended Python: 3.11 to 3.13.

```bash
# Option A: uv
uv sync --extra dev
source .venv/bin/activate

# Option B: pip
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy the environment template and fill only the keys your chosen providers need:

```bash
cp .env.example .env
```

For the default local OpenAI-compatible endpoints, no LLM API key is required when the base URL is localhost. Hosted providers need the matching key, for example `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`, or `MISTRAL_API_KEY`.

## Configure Models

Configuration is layered in this order:

1. `config/default.toml`
2. `~/.hypothesis-engine/config.toml`
3. `./hypothesis-engine.toml`
4. `--config <path>`
5. secrets from environment or `.env`

For a hosted OpenAI run, a minimal `hypothesis-engine.toml` might look like this:

```toml
[llm]
provider = "openai"

[models]
parse_goal = "gpt-5"
generation = "gpt-5"
reflection = "gpt-5"
evolution = "gpt-5"
ranking_pairwise = "gpt-5-mini"
ranking_debate = "gpt-5"
ranking_priority = "gpt-5"
metareview_feedback = "gpt-5-mini"
metareview_final = "gpt-5"
classifier = "gpt-5-mini"
judge = "gpt-5-mini"

[embeddings]
provider = "openai"
model = "text-embedding-3-large"
dim = 3072

[run]
budget_usd = 25.0
wall_clock_seconds = 7200
concurrency = 4
```

For a local OpenAI-compatible server with a different model:

```toml
[llm]
provider = "openai_compatible"

[llm.openai]
base_url = "http://localhost:8000/v1"

[models]
generation = "your-local-model"
reflection = "your-local-model"
evolution = "your-local-model"
ranking_pairwise = "your-local-model"
ranking_debate = "your-local-model"
ranking_priority = "your-local-model"
metareview_feedback = "your-local-model"
metareview_final = "your-local-model"
parse_goal = "your-local-model"
classifier = "your-local-model"
judge = "your-local-model"
```

Every model entry should be valid for the selected provider. The config deep-merges, so omitted model keys keep the defaults from `config/default.toml`.

### Customize Initial Discovery Lenses

Initial RAG generation can run several Generation tasks in parallel. Each task receives a different discovery lens so the first search pass covers mechanisms, routes, validation, modeling, analogs, and negative evidence instead of repeating the same literature slice. The default materials/chemistry lenses live in `config/discovery_profiles/materials_chemistry.yaml`.

To use field-specific lenses, copy that YAML file, edit the profiles, and point your local config at it:

```toml
[generation]
discovery_profiles = "/absolute/path/to/my_discovery_profiles.yaml"
```

Relative paths are resolved from the repository root. A profile file can contain either `profiles: [...]` or a top-level list. Each profile needs `id`, `label`, and `objective`; optional fields include `search_guidance`, `avoid_overfocus`, `suggested_query_angles`, `primary_driver_guidance`, and `required_study_elements`. The selected profile is stored in the session discovery artifacts for reproducibility.

Discovery profiles can also define domain-specific `required_work_packages`. These may be top-level, applying to every profile in the file, or attached to individual profiles. Generation uses them to force the generic `record_hypothesis.study_plan` field: every new hypothesis should include structured work packages with concrete methods, variables or conditions, outputs, quantitative targets, controls or comparators, and failure criteria. This keeps the package multidisciplinary: materials profiles can require synthesis, characterization, and theory/modeling plans; CS/AI profiles can require implementation, data, evaluation, systems, and robustness plans; microbiology profiles can require biological system, assay, omics, intervention, and biosafety plans.

## Initialize

```bash
hypothesis-engine init
hypothesis-engine tools list
```

`init` creates `data/`, applies SQLite migrations, and reports whether the configured LLM provider has the required key or is keyless. Runtime outputs are local:

- `data/hypothesis_engine.db`: sessions, tasks, hypotheses, reviews, tournaments, transcripts, bench rows
- `data/artifacts/<session_id>/`: JSON artifacts, final overview, analysis outputs
- `data/vectors/<session_id>/`: hypothesis FAISS index and metadata
- `data/rag/<session_id>/`: optional RAG KB PDFs, manifest, FAISS index, metadata
- `data/logs/`: session logs

## Run a Session

```bash
hypothesis-engine run "Identify testable hypotheses about microbiome-driven inflammation" \
  --n 3 \
  --budget-usd 10 \
  --wall-clock 3600
```

Useful session commands:

```bash
hypothesis-engine list
hypothesis-engine status <session_id>
hypothesis-engine report <session_id>
hypothesis-engine pause <session_id>
hypothesis-engine resume <session_id>
hypothesis-engine abort <session_id>
hypothesis-engine feedback <session_id> "focus on falsifiable mechanisms" --kind directive
hypothesis-engine estimate
```

The final overview is stored under `data/artifacts/<session_id>/final/overview.md` and is printed by `hypothesis-engine report` once the session has finalized.

## Web UI

```bash
hypothesis-engine serve --host 127.0.0.1 --port 7878
```

The web UI provides:

- session creation and session listing
- live workflow state via SSE
- token and cost summaries
- leaderboard and recent tournament matches
- hypothesis detail pages with rendered reviews
- pause, resume, abort, and feedback actions
- hypothesis cluster plots from the session FAISS index
- RAG KB cluster plots when a session has an indexed RAG KB
- generated post-hoc analysis pages and downloadable analysis bundles

The UI is FastAPI plus server-rendered Jinja templates, htmx-style fragments, SSE, and small static JavaScript files. There is no frontend build step.

## RAG Mode

RAG is optional, but highly recommended to get the most out of this package. It is enabled only when both are true:

1. `[rag].enabled = true`
2. `[rag].package_path` points to an installed `RAGAgent_AutonomousMaterialsSynthesis` package directory

The RAG package itself is treated as a local/vendor dependency and is not committed in this repository. `vendor/` is ignored. To install the pinned vendor checkout used by this branch:

```bash
scripts/install_rag_vendor.sh
```

By default the script clones `https://github.com/sumner-harris/RAGAgent_AutonomousMaterialsSynthesis.git` into `vendor/RAGAgent_AutonomousMaterialsSynthesis` and checks out commit `738d83d2ce7069cc35be69088522af39e68311e0`. It is safe to rerun; an existing Git checkout is fetched and moved to the requested ref.

Useful variants:

```bash
# Install the vendor repo plus its own Python requirements into the active env.
scripts/install_rag_vendor.sh --install-requirements

# Track a different branch, tag, or commit.
scripts/install_rag_vendor.sh --ref main

# Use a fork or a different checkout path.
scripts/install_rag_vendor.sh \
  --repo https://github.com/<owner>/RAGAgent_AutonomousMaterialsSynthesis.git \
  --target /opt/rag-agent
```

Environment variables `RAG_VENDOR_REPO`, `RAG_VENDOR_REF`, and `RAG_VENDOR_TARGET` provide the same overrides for automation.

When RAG is enabled, the tool registry exposes:

- `rag_kb_status`
- `rag_retrieve_context`

Generation, Reflection, and Evolution can use these tools. Search and fetch tools can ingest PDFs into `data/rag/<session_id>/`; the RAG bridge builds or appends a FAISS knowledge base and records a manifest. Before search results are returned into an agent chat, the `LiteratureReviewAgent` reviews the returned titles and abstracts/snippets and selects the records that are relevant enough for the calling agent to see, with room for limited exploratory or negative-evidence papers. Only selected records appear in the model-facing tool result; the full raw search results remain in the search cache/artifacts. If a selected record has a direct PDF URL and merits full-text evidence, the same review marks it for download. This review applies wherever the workflow calls search tools such as arXiv, ChemRxiv, bioRxiv, PubMed, Europe PMC, or web search. PubMed and Europe PMC currently return metadata/landing links rather than direct PDFs, so they can be selected for context but do not automatically download full text unless a direct PDF URL is present. Once a session KB is ready, the tool loop blocks some terminal record/search actions until the agent has attempted `rag_retrieve_context`, so reviews and evolved hypotheses use indexed source context instead of only search snippets.

Important RAG config knobs:

```toml
[rag]
enabled = true
package_path = "./vendor/RAGAgent_AutonomousMaterialsSynthesis"
seed_kb_path = "./data/rag_libraries/private-materials-kb"
auto_ingest_fetched_pdfs = true
auto_ingest_arxiv_pdfs = true
auto_ingest_max_pdfs_per_search = 20
literature_review_enabled = true
literature_review_max_candidates = 30
literature_review_max_context_results = 30
literature_review_timeout_seconds = 300
literature_review_fallback_to_top_results = true
max_session_papers = 1000
retrieval_method = "hybrid_multi_query"
context_max_chars = 30000
rerank_enabled = true
```

### Starting sessions from a prebuilt KB

Set `seed_kb_path` to reuse a trusted knowledge base, including a locally
licensed collection of paywalled papers that search tools cannot download:

```toml
[rag]
enabled = true
package_path = "./vendor/RAGAgent_AutonomousMaterialsSynthesis"
seed_kb_path = "./data/rag_libraries/private-materials-kb"
```

The seed directory must use the pinned vendor KB format and contain:

```text
private-materials-kb/
  kb.index          # required FAISS index
  kb.pkl            # required chunk text and embedding metadata
  manifest.json     # optional but recommended source provenance
  graphrag/         # required only when use_graphrag = true
```

`kb.pkl` contains serialized Python data and full extracted chunk text. Only
configure seed KBs from trusted sources, and only include publications that
you are authorized to process. Never place publisher credentials, cookies, or
download tokens in the seed directory or manifest.

Before goal parsing, each new session validates the seed with the vendor
`load_kb` service and copies `kb.index`, `kb.pkl`, and any configured GraphRAG
workspace into `data/rag/<session_id>/`. The source KB is never modified. The
session records SHA-256 fingerprints, the resolved embedding metadata, and the
seed origin in its `manifest.json`; newly discovered PDFs are appended only to
the session copy. A validated, nonempty seed KB satisfies the generation
readiness barrier, so hypothesis synthesis can begin while newly selected PDFs
continue ingesting in the background. Seeded papers do not consume
`max_session_papers`, which continues to limit papers acquired during that
session. Original PDFs are not
copied because retrieval uses the indexed chunks already stored in `kb.pkl`.

A seed `manifest.json` is optional for retrieval, but strongly recommended for
source attribution. Its `file` values must match the chunk `source` filenames
in `kb.pkl`; `url` may be a DOI or publisher landing page even when the full
text itself requires licensed access:

```json
{
  "papers": {
    "stable-local-key": {
      "title": "Licensed paper title",
      "url": "https://doi.org/10.xxxx/example",
      "file": "publisher-paper.pdf",
      "indexed": true
    }
  }
}
```

The configured embedding endpoint must serve the embedding model recorded in the
seed metadata; retrieval and later appends use that recorded model to remain
compatible with the existing FAISS vectors.

The seed path may be absolute or relative to the project root. Resuming a
session uses its existing session copy, so the original seed does not need to
remain mounted after initialization. Each session receives a physical copy of
the index and metadata to prevent later appends from mutating the shared seed.


The `auto_ingest_max_pdfs_per_search` limit is applied after LiteratureReviewAgent selection. `literature_review_max_candidates` controls how many title/abstract records the review agent sees from each search result set, while `literature_review_max_context_results` caps how many selected records are returned to the calling agent; both default to 30. If the review call fails and `literature_review_fallback_to_top_results = true`, the system preserves continuity by exposing/scheduling the first eligible records up to the configured caps and marking the fallback in metadata.

If RAG is disabled or the vendor package is missing, the rest of the system still runs with normal search/fetch tools and hypothesis-vector proximity.

## Capability Grounding

The optional capability catalog grounds the existing structured hypothesis
`study_plan` in available experimental, simulation, AI, and data resources.
It is deliberately separate from literature RAG:
literature supports scientific claims, while the capability catalog records
local availability, operating ranges, dependencies, access constraints, and
verification provenance.

Populate `config/capabilities/` using the schema and example in
`config/capabilities/README.md`, then enable it in a local override:

```toml
[capabilities]
enabled = true
catalog_path = "./config/capabilities"
grounding_policy = "advisory"  # advisory | required
max_search_results = 8
```

Validate syntax, cross-record references, and executable tool names without
starting a session:

```bash
hypothesis-engine capabilities validate
```

When enabled, Generation, Reflection, and Evolution receive:

- `capability_search`: filtered local catalog search
- `capability_get`: exact versioned specifications and constraints
- `capability_validate_workflow`: deterministic validation of study-plan references

During initial Generation discovery, the agent searches the capability catalog
before finalizing its literature map. For up to three relevant catalog records,
it then runs focused literature queries that combine the target system or
material with the method and intended observable. For example, a graphene
project with micro-Raman available should trigger searches on how Raman is used
to measure graphene defects, strain, doping, or layer number. The resulting
queries, method-specific evidence, observables, and limitations are recorded in
the discovery result and passed into hypothesis synthesis. Reflection and
Evolution receive the same capability-application guidance when both catalog
and literature-search tools are available, allowing them to challenge or
refine the proposed use.

Generation and Evolution store exact capability IDs, versions, purposes, and
parameter values in each work package. Before persistence, the application
independently validates the final tool payload and adds a catalog-revisioned
grounding report to the hypothesis artifact and markdown. Reflection records a
separate capability audit by deterministically revalidating the exact persisted
`study_plan`; the reflection model inspects catalog records and literature but
does not rewrite the plan for validation. Ranking and Meta-review consume the
persisted workflow and audit without live capability tools. `advisory` preserves
partially grounded hypotheses with explicit issues; `required` rejects
hypotheses with unknown/unavailable capabilities, invalid parameters,
unsatisfied dependencies, or ungrounded work packages.

## Tools

Built-in tools include:

- `web_fetch`: fetch and extract web pages or PDFs, with SSRF protections
- `pubmed_search`
- `arxiv_search`
- `chemrxiv_search`
- `europe_pmc_search`
- `web_search`: registered only when `TAVILY_API_KEY` or `BRAVE_API_KEY` is available
- optional RAG tools when RAG is configured
- optional capability catalog tools when `[capabilities].enabled = true`
- optional science-skill tools discovered from `vendor/science-skills`

Tool availability is agent-specific. Ranking, Proximity, and Meta-review do not receive literature tools during their core calls; Generation, Reflection, and Evolution do.

## Post-Hoc Analysis

Run a local analysis report for a completed or partially completed session:

```bash
hypothesis-engine analyze <session_id>
```

By default this writes:

- `data/artifacts/<session_id>/analysis/report.md`
- `data/artifacts/<session_id>/analysis/report.html`
- `data/artifacts/<session_id>/analysis/analysis_report.zip`
- CSV tables and SVG figures under the same analysis directory

The analysis reads SQLite rows, transcript artifacts, search/fetch artifacts, hypothesis vectors, tournament history, and optional RAG KB vectors. It includes Elo trajectory diagnostics, volatility and rank-stability plots, debate position/verbosity checks, hypothesis clustering, RAG KB clustering, and joint hypothesis/KB projection summaries where data is available.

Analysis outputs are local run artifacts and should not be committed.

## Benchmarks

`hypothesis-engine bench` compares model candidates on a shared goal. Each candidate can run through the full Generation pipeline or as a direct single-call baseline. A fixed judge model runs the cross-candidate tournament.

Examples:

```bash
hypothesis-engine bench "Identify hypotheses about X" \
  -c local-a=openai_compatible:your-local-model \
  -c openai-a=openai:gpt-5 \
  --judge openai:gpt-5-mini \
  --n 2 \
  --matches 2

hypothesis-engine bench --preset paper-aml --n 3 --matches 2
hypothesis-engine bench --preset paper-aml-vs-raw --n 1
hypothesis-engine bench --preset frontier-aml-vs-raw --n 1
```

Bench rows are stored in `data/hypothesis_engine.db` and JSON artifacts are written under `data/artifacts/<bench_session_id>/bench/`. The helper script can render a local markdown report:

```bash
python scripts/build_bench_report.py --db data/hypothesis_engine.db --out docs/BENCH_RESULTS.md
```

That generated report is not part of the published branch.

## Testing

```bash
pytest
pytest hypothesis_engine/tests/unit/test_evolution.py hypothesis_engine/tests/unit/test_web_fragments.py
ruff check .
```

The GitHub workflow runs the unit test suite on pushes and pull requests.

## Repository Layout

```text
hypothesis_engine/agents/        supervisor, generation, literature_review, reflection, ranking, evolution, proximity, meta-review
hypothesis_engine/analysis/      post-hoc session analysis report builder
hypothesis_engine/bench/         model benchmark runner, presets, gold-set scoring
hypothesis_engine/capabilities/  versioned capability models, catalog search, workflow validation
hypothesis_engine/llm/           provider adapters, routing, retries, token/cost accounting
hypothesis_engine/orchestrator/  termination, events, feedback actions, Elo helpers
hypothesis_engine/storage/       SQLite schema, migrations, repositories, artifact helpers
hypothesis_engine/tools/         built-in literature/search/fetch/RAG/science-skill tools
hypothesis_engine/vectors/       embedding clients and FAISS store
hypothesis_engine/web/           FastAPI UI, templates, static JS/CSS
config/                     default and alternate configs plus agent prompts
scripts/                    local report/maintenance scripts
```

## Notes And Caveats

- This project is independent and is not affiliated with Google or the authors of the research that inspired it.
- The committed branch is source-only. Runtime artifacts live under `data/` and are ignored.
- Default budgets in `config/default.toml` are permissive for local model servers. Set realistic `budget_usd`, `wall_clock_seconds`, and `concurrency` values when using paid APIs.
- Function/tool calling is required for normal agent operation. Providers or local servers that do not support tool calls will fail during structured record steps.
- Embeddings fall back to a deterministic hash embedder if no configured embedding backend is available. That keeps runs alive but weakens semantic deduplication and clustering.
- `config/a100_remote.toml` is environment-specific. Treat it as a template for a remote high-throughput setup, not as a portable default.

## License

This project is a substantially modified derivative of [Kaimen-Inc/Co-Scientist](https://github.com/Kaimen-Inc/Co-Scientist) and remains licensed under the Apache License 2.0. See [LICENSE](LICENSE).
