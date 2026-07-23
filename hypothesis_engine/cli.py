# Modified from the original work.
"""Typer CLI entrypoint.

M0 surface: init, list, status, tools list, serve, version. Run/resume/report/feedback
land in M3+. All commands resolve config the same way; secrets come from env.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import has_llm_key, load_config, provider_key_env
from .logging import get_logger, setup_logging
from .storage import db as db_mod

VERSION = "0.1.0"

app = typer.Typer(
    help="Hypothesis Engine — multi-agent hypothesis generation, ranking, and synthesis.",
    no_args_is_help=True,
    add_completion=False,
)
tools_app = typer.Typer(help="Tool registry inspection and debug invocation.", no_args_is_help=True)
capabilities_app = typer.Typer(
    help="Capability catalog validation and inspection.",
    no_args_is_help=True,
)
app.add_typer(tools_app, name="tools")
app.add_typer(capabilities_app, name="capabilities")

console = Console()
log = get_logger("cli")


def _common_setup(config_file: Path | None = None, verbose: bool = False) -> tuple:
    setup_logging("DEBUG" if verbose else "INFO")
    cfg = load_config(config_file)
    return cfg, log


@app.callback()
def _main(
    ctx: typer.Context,
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to an extra TOML config to overlay."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
) -> None:
    ctx.obj = _common_setup(config_file, verbose)


@app.command()
def version() -> None:
    """Print the hypothesis-engine version."""
    console.print(f"hypothesis-engine {VERSION}")


@app.command()
def init(ctx: typer.Context) -> None:
    """Create the data directory and database schema; sanity-check env."""
    cfg, _ = ctx.obj
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "artifacts").mkdir(exist_ok=True)
    (cfg.data_dir / "vectors").mkdir(exist_ok=True)
    (cfg.data_dir / "logs").mkdir(exist_ok=True)

    asyncio.run(db_mod.init_db(cfg))

    # Report
    env_var = provider_key_env(cfg)
    tbl = Table(title="Init complete", show_header=False, box=None)
    tbl.add_row("data dir", str(cfg.data_dir))
    tbl.add_row("database", str(cfg.db_path))
    tbl.add_row("LLM provider", cfg.llm.provider)
    tbl.add_row(
        f"{env_var or '(keyless)'} set",
        "yes" if has_llm_key(cfg) else "[red]no[/red]",
    )
    tbl.add_row("science-skills", cfg.science_skills.path)
    console.print(tbl)

    if not has_llm_key(cfg):
        console.print(
            f"[yellow]{env_var} is not set; set it (or change [llm] provider in your "
            f"config) before running a session. See .env.example.[/yellow]"
        )


@app.command(name="list")
def list_sessions(ctx: typer.Context) -> None:
    """List all sessions in the local database."""
    cfg, _ = ctx.obj

    async def _run() -> list[dict]:
        conn = await db_mod.connect(cfg)
        try:
            async with conn.execute(
                """SELECT id, status, research_goal, created_at, updated_at,
                          budget_usd, budget_used_usd,
                          (SELECT COUNT(*) FROM hypotheses WHERE session_id = s.id) AS n_hyps,
                          (SELECT MAX(elo) FROM hypotheses WHERE session_id = s.id) AS top_elo
                     FROM sessions s
                     ORDER BY updated_at DESC"""
            ) as cur:
                rows = await cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    rows = asyncio.run(_run())
    if not rows:
        console.print("[dim]No sessions yet. Try:  hypothesis-engine run \"your research goal\"[/dim]")
        return

    tbl = Table(title="Sessions")
    tbl.add_column("id")
    tbl.add_column("status")
    tbl.add_column("goal", overflow="fold", max_width=60)
    tbl.add_column("hyps", justify="right")
    tbl.add_column("top Elo", justify="right")
    tbl.add_column("$ used / $ budget", justify="right")
    tbl.add_column("updated")
    for r in rows:
        tbl.add_row(
            r["id"],
            r["status"],
            (r["research_goal"] or "")[:80],
            str(r["n_hyps"]),
            f"{r['top_elo']:.0f}" if r["top_elo"] is not None else "—",
            f"${r['budget_used_usd']:.2f} / ${r['budget_usd']:.2f}",
            r["updated_at"],
        )
    console.print(tbl)


@app.command()
def status(ctx: typer.Context, session_id: str = typer.Argument(...)) -> None:
    """Show detailed status for one session."""
    cfg, _ = ctx.obj

    async def _run() -> dict:
        conn = await db_mod.connect(cfg)
        try:
            async with conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cur:
                row = await cur.fetchone()
            if row is None:
                return {}
            out = dict(row)
            for col in ("research_plan", "config_snapshot"):
                with contextlib.suppress(Exception):
                    out[col] = json.loads(out[col])
            async with conn.execute(
                """SELECT status, COUNT(*) AS n FROM tasks
                       WHERE session_id = ? GROUP BY status""",
                (session_id,),
            ) as cur:
                out["task_counts"] = {r["status"]: r["n"] for r in await cur.fetchall()}
            async with conn.execute(
                """SELECT state, COUNT(*) AS n FROM hypotheses
                       WHERE session_id = ? GROUP BY state""",
                (session_id,),
            ) as cur:
                out["hypothesis_states"] = {r["state"]: r["n"] for r in await cur.fetchall()}
            return out
        finally:
            await conn.close()

    out = asyncio.run(_run())
    if not out:
        console.print(f"[red]No session {session_id}[/red]")
        raise typer.Exit(1)
    console.print_json(data=out)


@app.command()
def run(
    ctx: typer.Context,
    goal: str = typer.Argument(..., help="Research goal in natural language."),
    preferences_file: Path | None = typer.Option(
        None, "--preferences-file", help="Path to a text file with extra preferences."
    ),
    n_initial: int | None = typer.Option(
        None, "--n", help="Number of initial Generation calls (parallel). Defaults to run.initial_generations."
    ),
    wall_clock: int | None = typer.Option(
        None, "--wall-clock", help="Override wall-clock cap in seconds."
    ),
    budget_usd: float | None = typer.Option(
        None, "--budget-usd", help="Override session USD budget."
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency", help="Override worker concurrency."
    ),
) -> None:
    """Start a fresh research session. Generation -> Reflection -> Ranking tournament -> Meta-review."""
    cfg, _ = ctx.obj
    if not has_llm_key(cfg):
        env_var = provider_key_env(cfg)
        console.print(f"[red]{env_var} is not set (LLM provider = {cfg.llm.provider}). See .env.example.[/red]")
        raise typer.Exit(1)

    if budget_usd is not None:
        cfg.run.budget_usd = budget_usd
    if concurrency is not None:
        cfg.run.concurrency = concurrency

    # Pre-flight cost estimate
    from .llm.estimator import estimate as _estimate

    est = _estimate(cfg)
    console.print(
        f"[dim]Pre-flight estimate: ${est.total_usd:.2f} "
        f"(budget ${cfg.run.budget_usd:.2f}, "
        f"max_ideas={cfg.run.max_ideas}, "
        f"max_matches_per_idea={cfg.run.max_matches_per_idea})[/dim]"
    )
    if est.warning:
        console.print(f"[yellow]{est.warning}[/yellow]")

    prefs = preferences_file.read_text(encoding="utf-8") if preferences_file else None
    from .agents.supervisor import Supervisor

    sup = Supervisor(cfg)
    session_id = asyncio.run(
        sup.run_session(goal, preferences_text=prefs, n_initial=n_initial,
                        wall_clock_seconds=wall_clock)
    )
    console.print(f"[green]Done.[/green] session={session_id}")
    console.print(f"View report:  hypothesis-engine report {session_id}")


@app.command()
def resume(
    ctx: typer.Context,
    session_id: str = typer.Argument(...),
) -> None:
    """Resume a paused or interrupted session."""
    cfg, _ = ctx.obj
    if not has_llm_key(cfg):
        env_var = provider_key_env(cfg)
        console.print(f"[red]{env_var} is not set (LLM provider = {cfg.llm.provider}). See .env.example.[/red]")
        raise typer.Exit(1)
    from .agents.supervisor import Supervisor

    sup = Supervisor(cfg)
    sid = asyncio.run(sup.run_session("", resume_session_id=session_id))
    console.print(f"[green]Done.[/green] session={sid}")


@app.command()
def pause(ctx: typer.Context, session_id: str = typer.Argument(...)) -> None:
    """Pause a running session. Workers drain; the loop sleeps until resume."""
    cfg, _ = ctx.obj

    async def _do() -> None:
        conn = await db_mod.connect(cfg)
        try:
            from .storage.repos import sessions as sess_repo

            await sess_repo.set_status(conn, session_id, "paused")
        finally:
            await conn.close()

    asyncio.run(_do())
    console.print(f"[yellow]Paused[/yellow] session={session_id}")


@app.command()
def abort(ctx: typer.Context, session_id: str = typer.Argument(...)) -> None:
    """Abort a running session. The main loop exits at the next check."""
    cfg, _ = ctx.obj

    async def _do() -> None:
        conn = await db_mod.connect(cfg)
        try:
            from .storage.repos import sessions as sess_repo

            await sess_repo.set_status(conn, session_id, "aborted")
        finally:
            await conn.close()

    asyncio.run(_do())
    console.print(f"[red]Aborted[/red] session={session_id}")


@app.command()
def feedback(
    ctx: typer.Context,
    session_id: str = typer.Argument(...),
    text: str = typer.Argument(..., help="Free-text feedback to inject."),
    kind: str = typer.Option(
        "directive", "--kind",
        help="directive | preference | rejection | pin",
    ),
    target: str | None = typer.Option(
        None, "--target", help="Hypothesis ID this feedback is about (optional)."
    ),
) -> None:
    """Inject researcher feedback into a running (or future) session."""
    cfg, _ = ctx.obj
    from . import ids as _ids
    from .models import SystemFeedback
    from .orchestrator.feedback_actions import apply_human_feedback_actions
    from .storage.repos import feedback as fb_repo

    async def _do() -> None:
        conn = await db_mod.connect(cfg)
        try:
            fb = SystemFeedback(
                id=_ids.feedback_id(), session_id=session_id,
                created_at=datetime.now(UTC),
                source="human", kind=kind,
                target_id=target, text=text, active=True,
            )
            await fb_repo.insert(conn, fb)
            actions = await apply_human_feedback_actions(
                conn,
                session_id=session_id,
                feedback_id=fb.id,
                kind=kind,
                target_id=target,
            )
            if actions.get("enqueued"):
                console.print(
                    f"[green]Scheduled[/green] {actions['enqueued']} follow-up task(s): "
                    + ", ".join(actions.get("tasks", []))
                )
        finally:
            await conn.close()

    asyncio.run(_do())
    console.print(f"[green]Feedback recorded[/green] for session={session_id}")


@app.command()
def report(
    ctx: typer.Context,
    session_id: str = typer.Argument(...),
    format: str = typer.Option("md", "--format", "-f", help="md or json"),
) -> None:
    """Print the final research overview for a session (when available)."""
    cfg, _ = ctx.obj

    async def _path() -> Path | None:
        conn = await db_mod.connect(cfg)
        try:
            async with conn.execute(
                "SELECT final_overview FROM sessions WHERE id = ?", (session_id,)
            ) as cur:
                row = await cur.fetchone()
            if not row or not row["final_overview"]:
                return None
            return cfg.data_dir / row["final_overview"]
        finally:
            await conn.close()

    p = asyncio.run(_path())
    if p is None:
        console.print(f"[red]No final overview yet for {session_id}[/red]")
        raise typer.Exit(1)
    text = p.read_text(encoding="utf-8")
    if format == "json":
        console.print_json(data={"session_id": session_id, "path": str(p), "content": text})
    else:
        console.print(text)


@app.command()
def analyze(
    ctx: typer.Context,
    session_id: str = typer.Argument(...),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory. Defaults to data/artifacts/<session_id>/analysis.",
    ),
    snapshot_every: int = typer.Option(
        25,
        "--snapshot-every",
        help="Tournament match interval for Elo trajectory snapshots.",
    ),
    max_kb_points: int = typer.Option(
        5000,
        "--max-kb-points",
        help=(
            "Maximum RAG KB chunks to render and use for sampled diagnostics. "
            "The shared PCA and clustering artifact is always fitted on the full KB."
        ),
    ),
    clusters: int = typer.Option(
        8,
        "--clusters",
        help=(
            "KMeans cluster count for hypothesis summaries. "
            "The shared KB projection selects its cluster count by silhouette and stability."
        ),
    ),
) -> None:
    """Write a post-hoc research-style analysis report for one session."""
    cfg, _ = ctx.obj
    from .analysis import analyze_session

    report_path = analyze_session(
        cfg,
        session_id,
        output_dir=output,
        snapshot_every=snapshot_every,
        max_kb_points=max_kb_points,
        n_clusters=clusters,
    )
    console.print(f"[green]Analysis written[/green] {report_path}")
    console.print(f"HTML report: {report_path.parent / 'report.html'}")
    console.print(f"Download bundle: {report_path.parent / 'analysis_report.zip'}")


@app.command()
def serve(
    ctx: typer.Context,
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
) -> None:
    """Launch the FastAPI + htmx + SSE web UI."""
    cfg, _ = ctx.obj
    host = host or cfg.web_ui.host
    port = port or cfg.web_ui.port
    import uvicorn

    from .web.app import create_app

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


@app.command()
def estimate(ctx: typer.Context) -> None:
    """Print the pre-flight cost estimate without launching a session."""
    cfg, _ = ctx.obj
    from .llm.estimator import estimate as _estimate

    est = _estimate(cfg)
    console.print_json(data=est.to_dict())


@tools_app.command("list")
def tools_list(ctx: typer.Context) -> None:
    """List registered tools (builtins + any discovered science-skills)."""
    cfg, _ = ctx.obj
    from .tools.registry import AGENT_TOOLS, ToolRegistry

    reg = ToolRegistry(cfg).discover()
    tbl = Table(title=f"Tools ({len(reg.all())})")
    tbl.add_column("name", style="bold")
    tbl.add_column("description", overflow="fold")
    for item in reg.summary():
        tbl.add_row(item["name"], item["description"])
    console.print(tbl)

    # Show per-agent allowlist resolution counts
    tbl2 = Table(title="Per-agent tool availability")
    tbl2.add_column("agent")
    tbl2.add_column("# tools", justify="right")
    tbl2.add_column("allowlist (patterns)")
    for agent, patterns in AGENT_TOOLS.items():
        ts = reg.tools_for(agent)
        tbl2.add_row(agent, str(len(ts)), ", ".join(sorted(patterns)) or "—")
    console.print(tbl2)


@capabilities_app.command("validate")
def capabilities_validate(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the complete validation report as JSON.",
    ),
) -> None:
    """Validate the configured capability catalog without starting a session."""
    cfg, _ = ctx.obj
    from .capabilities.catalog import (
        CapabilityCatalog,
        CapabilityCatalogValidationError,
        validate_catalog_integrity,
    )
    from .tools.registry import ToolRegistry

    try:
        catalog = CapabilityCatalog.from_config(cfg)
        try:
            registry = ToolRegistry(cfg).discover()
            registered_tool_names = {tool.name for tool in registry.all()}
            report = validate_catalog_integrity(
                catalog,
                registered_tool_names=registered_tool_names,
            )
        except CapabilityCatalogValidationError as exc:
            report = exc.report
    except ValueError as exc:
        console.print(f"[red]Invalid capability catalog:[/red] {exc}")
        raise typer.Exit(1) from exc

    if json_output:
        console.print_json(data=report.model_dump())
    else:
        tbl = Table(title=f"Capability catalog {report.catalog_revision}")
        tbl.add_column("severity")
        tbl.add_column("capability")
        tbl.add_column("field")
        tbl.add_column("message", overflow="fold")
        for issue in report.issues:
            style = "red" if issue.severity == "error" else "yellow"
            tbl.add_row(
                f"[{style}]{issue.severity}[/{style}]",
                issue.capability_id or "catalog",
                issue.field or issue.code,
                issue.message,
            )
        if report.issues:
            console.print(tbl)
        console.print(
            f"path={report.source_path} capabilities={report.capability_count} "
            f"errors={report.error_count} warnings={report.warning_count}"
        )
    if not report.valid:
        raise typer.Exit(1)
    if not json_output:
        console.print("[green]Capability catalog is valid.[/green]")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
