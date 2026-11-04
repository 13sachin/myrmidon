"""Reading back what went through the harness: summary, report, the
force-matching report, the live feed, the status line, the tool list, and one tool call."""

from __future__ import annotations

import typer

from myrmidon.cli._app import PANEL_READ, app, resolve_workspace, tool_context, workspace_path


@app.command(hidden=True)
def summary(
    workspace: str = typer.Argument("", help="workspace path (default: the active one)"),
    since: str = typer.Option("session", help="'session', 'all', '6h', '2d', or ISO time"),
    no_usage: bool = typer.Option(False, help="skip backend accounting queries (faster)"),
) -> None:
    """What went through the harness: tools, gates, jobs, core-hours, artifacts."""
    from myrmidon.orchestration.summary import render, summarize
    from myrmidon.server.mcp_server import build_context

    ws = resolve_workspace(workspace)
    if ws is None:
        typer.echo("no active study (open one through the tools, or pass a path)")
        raise typer.Exit(1)
    ctx = build_context()
    report = summarize(
        ws, since, backend=ctx.backend, compute=ctx.config.compute, poll_usage=not no_usage,
    )
    typer.echo(render(report))


@app.command(rich_help_panel=PANEL_READ)
def report(
    workspace: str = typer.Argument("", help="workspace path (default: the active one)"),
    no_usage: bool = typer.Option(False, help="skip backend accounting queries (faster)"),
) -> None:
    """Study report from the ledger alone: steps, findings, artifacts, jobs, core-h."""
    from myrmidon.orchestration.report import render_report
    from myrmidon.server.mcp_server import build_context

    ws = resolve_workspace(workspace)
    if ws is None:
        typer.echo("no active study (open one through the tools, or pass a path)")
        raise typer.Exit(1)
    ctx = build_context()
    typer.echo(render_report(ws, compute=ctx.config.compute,
                             backend=None if no_usage else ctx.backend))


@app.command(name="fm-report", hidden=True)
def fm_report_cmd(
    workspace: str = typer.Argument("", help="workspace path (default: the active one)"),
    stage: str = typer.Option("report", help="stage to write fm_report.md/.json into"),
    no_usage: bool = typer.Option(False, help="skip backend accounting queries (faster)"),
    print_md: bool = typer.Option(True, "--print/--no-print", help="echo the markdown"),
) -> None:
    """Force-matching campaign report from the ledger: datasets, fit settings and outcome,
    charge and bonded-parameter tables (original vs fitted), validation MAE, compute."""
    from myrmidon.core.tool import REGISTRY
    from myrmidon.server.mcp_server import build_context

    ws = resolve_workspace(workspace)
    if ws is None:
        typer.echo("no active study (open one through the tools, or pass a path)")
        raise typer.Exit(1)
    ctx = build_context()
    ctx.workspace = ws
    if no_usage:
        ctx.backend = None
    obs = REGISTRY.call("fm_report", ctx, {"stage": stage})
    if not obs.ok:
        typer.echo(obs.summary)
        raise typer.Exit(1)
    if print_md:
        typer.echo((ws.root / obs.report_path).read_text())
    typer.echo(obs.summary)


@app.command(hidden=True)
def watch(
    workspace: str = typer.Argument("", help="workspace path (default: the active one)"),
    poll_s: float = typer.Option(2.0, help="seconds between looks at the ledger"),
    once: bool = typer.Option(False, "--once", help="print what has happened and exit"),
) -> None:
    """Follow what is happening, one line at a time: tool calls, job states, the
    engine's own step counter, and every decision with the evidence behind it.

    This is a view, not the run. Ctrl-C leaves whatever is running alone, and starting
    it again picks the feed back up — everything it shows is read from the ledger, so
    it works from any terminal, on any machine, at any time after the fact.
    """
    from myrmidon.orchestration.watch import follow, header

    ws = resolve_workspace(workspace)
    if ws is None:
        typer.echo("no active study (open one through the tools, or pass a path)")
        raise typer.Exit(1)
    for line in header(ws):
        typer.echo(line)
    try:
        for line in follow(ws, poll_s=poll_s, once=once):
            typer.echo(line)
    except KeyboardInterrupt:
        typer.echo("")
        typer.echo("stopped watching — the run continues. "
                   "`myrmidon study open` for where it stands.")


@app.command(hidden=True)
def statusline(
    workspace: str = typer.Argument("", help="workspace path (default: the active one)"),
) -> None:
    """One line for a terminal status bar: active workspace, live jobs, core-hours.

    Always one line and exit 0: a status bar shows every line it is given, and a failing
    command sends Claude Code's to its fallback on every redraw. A workspace that cannot be
    opened is named here; `myrmidon summary` says why.
    """
    from myrmidon.core.workspace import Workspace
    from myrmidon.errors import WorkspaceError
    from myrmidon.orchestration.summary import statusline as _statusline

    path = workspace_path(workspace)
    if path is None:
        typer.echo("myrmidon: no active study")
        return
    try:
        ws = Workspace.open(path)
    except WorkspaceError:
        typer.echo(f"myrmidon: {path.name} cannot be opened (`myrmidon summary` says why)")
        return
    typer.echo(_statusline(ws))


def _tool_lines(ctx) -> list[str]:
    from myrmidon.core.tool import REGISTRY, books

    lines = []
    for spec in sorted(REGISTRY.available(ctx.engines), key=lambda s: s.name):
        gate = " [books]" if books(spec) else ""
        lines.append(f"{spec.name}{gate}: {spec.description.splitlines()[0]}")
    return lines


@app.command(hidden=True)
def tools() -> None:
    """List every tool the registry would expose on this machine."""
    from myrmidon.server.mcp_server import build_context

    for line in _tool_lines(build_context()):
        typer.echo(line)


#: `call` takes the tool's parameters as bare key=value words, so click must not try to
#: read them, and --help belongs to the tool rather than to the command
CALL_CONTEXT = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}


@app.command(context_settings=CALL_CONTEXT, hidden=True)
def call(
    typer_ctx: typer.Context,
    tool: str = typer.Argument("", help="tool name; with no tool, every tool is listed"),
    workspace: str = typer.Option("", "--study",
                                 help="study path (default: the "
                                  "active one)"),
    as_json: bool = typer.Option(False, "--json", help="print the whole observation as "
                                 "JSON — the same shape an agent is given"),
    params_file: str = typer.Option("", "--from", help="a YAML or JSON file holding the "
                                    "tool's parameters; "
                                    "key=value on the command line overrides it"),
) -> None:
    """Call any registered tool: `myrmidon call gmx_solvate structure=... topology=...`.

    The parameters are the tool's own, validated by the registry exactly as an agent's
    call is, and the call lands in the workspace's ledger the same way. `myrmidon call
    TOOL --help` prints the parameters. Lists take a repeated key or commas; dicts take
    JSON; `--from FILE` takes them all from a file. The exit code follows the observation:
    0 ok, 1 failed or refused, 2 a mistake in the command line itself.
    """
    import getpass
    import json as _json

    from myrmidon.cli.loop import inside_agent
    from myrmidon.cli_call import (
        UsageError,
        parse_assignments,
        render_observation,
        tool_help,
    )
    from myrmidon.core.tool import REGISTRY
    from myrmidon.errors import UnknownToolError
    from myrmidon.server.mcp_server import build_context
    from myrmidon.tools import register_all

    register_all()  # the tool list is the registry's; --help must not need the engines
    args = [arg for arg in typer_ctx.args if arg not in ("--help", "-h")]
    wants_help = len(args) != len(typer_ctx.args)
    if tool in ("--help", "-h"):
        # `call` reads --help as the tool's; the bare habit `myrmidon call --help` must
        # not land on "unknown tool '--help'", an error that reads like a typo in a name
        tool, wants_help = "", True

    if not tool:
        typer.echo(call.__doc__.splitlines()[0])
        typer.echo("")
        for line in _tool_lines(build_context()):
            typer.echo(line)
        typer.echo("")
        typer.echo("myrmidon call TOOL --help   — the parameters of one tool")
        raise typer.Exit(0)

    name = tool
    try:
        spec = REGISTRY.get(name)
    except UnknownToolError:
        import difflib

        near = difflib.get_close_matches(tool, [s.name for s in REGISTRY.specs()], n=3)
        typer.echo(f"unknown tool '{tool}'")
        if near:
            typer.echo("  did you mean: " + ", ".join(near))
        typer.echo("  myrmidon call   — every tool on this machine")
        raise typer.Exit(2) from None
    if wants_help:
        typer.echo(tool_help(spec))
        raise typer.Exit(0)

    try:
        params = {}
        if params_file:
            from pathlib import Path

            import yaml

            try:
                loaded = yaml.safe_load(Path(params_file).read_text())
            except (OSError, yaml.YAMLError) as exc:
                raise UsageError(f"cannot read {params_file}: {exc}") from None
            if not isinstance(loaded, dict):
                raise UsageError(f"{params_file} does not hold a mapping of parameters")
            stray = sorted(set(loaded) - set(spec.params_model.model_fields))
            if stray:
                raise UsageError(f"{spec.name} takes no {', '.join(stray)}; "
                                 f"`myrmidon call {spec.name} --help` lists its parameters")
            params = dict(loaded)
        params.update(parse_assignments(args, spec.params_model, spec.name))
    except UsageError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(2) from None

    ctx = tool_context(workspace)
    if not inside_agent():  # a person at a terminal: their call accepts what it raises
        ctx.person = getpass.getuser()
    obs = REGISTRY.call(spec.name, ctx, params)
    if as_json:
        typer.echo(_json.dumps(obs.model_dump(mode="json", exclude_none=True), indent=1))
    else:
        typer.echo(render_observation(obs))
    raise typer.Exit(0 if obs.ok else 1)
