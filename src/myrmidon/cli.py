"""Myrmidon CLI: serve (MCP), doctor, job inspection."""

from __future__ import annotations

import json

import typer

import myrmidon
from myrmidon.server.client_registry import REGISTRY

_CLIENT_KEYS = " | ".join(REGISTRY)
_CLIENT_NAMES = ", ".join(d.display for d in REGISTRY.values())
_LAUNCHABLE = tuple(k for k, d in REGISTRY.items() if d.launchable)
_CLIENT_HELP = (f"the agent client to start: {' | '.join(_LAUNCHABLE)} (default: the "
                "project's or the machine config's `client`, else claude)")

_MODEL_HELP = ("the model to use (default: the one you used last; the first time, you "
               "pick from what the endpoint serves)")
_ENDPOINT_HELP = "a named endpoint from endpoints.toml (default: the one you used last)"
_CONFIG_HELP = ("read this config file only: no project .myrmidon.toml, no machine "
                "config, no MYRMIDON_* variables; keys it leaves out take the built-in "
                "defaults")
_TIMEOUT_HELP = "seconds one model request may take before it is abandoned (default 180)"
_VERBOSE_HELP = "also show provenance ids, artifacts and accepted findings' detail"
_DEBUG_HELP = "show every tool call, result, retry and per-step token count"


app = typer.Typer(
    name="myrmidon",
    help="Myrmidon: an agent harness for molecular simulation. `myrmidon client` starts "
         "your coding agent (Claude Code, ...) wired to the harness; `myrmidon` on its own "
         "starts Myrmidon's own agent on a model endpoint of yours; the other commands set "
         "up the machine and read back what went through it.",
    no_args_is_help=False,
    invoke_without_command=True,
)


def _version(shown: bool) -> None:
    """`--version` is the habit; `myrmidon version` is the other spelling."""
    if shown:
        typer.echo(myrmidon.__version__)
        raise typer.Exit(0)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", callback=_version,
                                 is_eager=True, help="print the version and exit"),
    model: str = typer.Option("", "--model", "-m", help=_MODEL_HELP),
    endpoint: str = typer.Option("", "--endpoint", "-e", help=_ENDPOINT_HELP),
    continue_: bool = typer.Option(False, "--continue", "-c",
                                   help="continue the last session"),
    resume: str = typer.Option("", "--resume", "-r", help="continue a session by id "
                               "(`myrmidon sessions` lists them)"),
    call_timeout: float = typer.Option(0, "--call-timeout", help=_TIMEOUT_HELP,
                                      show_default=False),
    verbose: bool = typer.Option(False, "--verbose", "-v", help=_VERBOSE_HELP),
    debug: bool = typer.Option(False, "--debug", help=_DEBUG_HELP),
) -> None:
    """Myrmidon's own agent: with no command, start it here."""
    if ctx.invoked_subcommand is None:
        _native(model=model, endpoint=endpoint, continue_=continue_, resume=resume,
                call_timeout=call_timeout, verbose=verbose, debug=debug)


def _launch(prompt: str = "", model: str = "", continue_: bool = False,
            resume: str = "", client: str = "") -> None:
    """`myrmidon client`: an agent client, wired to this harness."""
    from myrmidon.server.launch import LaunchError, launch

    if client and client not in REGISTRY:
        typer.echo(f"error: unknown client {client!r}; one of {', '.join(REGISTRY)} "
                   f"(started by myrmidon: {', '.join(_LAUNCHABLE)}).")
        raise typer.Exit(2)
    try:
        launch(model=model, continue_=continue_, resume=resume, prompt=prompt, client=client)
    except LaunchError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(2) from None


@app.command("chat")
def chat(
    prompt: str = typer.Argument("", help="a first question (optional)"),
    model: str = typer.Option("", "--model", "-m", help=_MODEL_HELP),
    endpoint: str = typer.Option("", "--endpoint", "-e", help=_ENDPOINT_HELP),
    continue_: bool = typer.Option(False, "--continue", "-c",
                                   help="continue the last session"),
    resume: str = typer.Option("", "--resume", "-r", help="continue a session by id"),
    call_timeout: float = typer.Option(0, "--call-timeout", help=_TIMEOUT_HELP,
                                      show_default=False),
    verbose: bool = typer.Option(False, "--verbose", "-v", help=_VERBOSE_HELP),
    debug: bool = typer.Option(False, "--debug", help=_DEBUG_HELP),
) -> None:
    """Start the agent, with a first question — what bare `myrmidon` does, named.

    You are at the keyboard: when a call needs a person's decision (a compute budget), you
    are asked, and your yes approves that one call. The session is saved after every
    answer; `myrmidon -c` continues it, and leaving never cancels a scheduler job.
    """
    _native(prompt=prompt, model=model, endpoint=endpoint, continue_=continue_,
            resume=resume, call_timeout=call_timeout, verbose=verbose, debug=debug)


@app.command("client")
def client_command(
    prompt: str = typer.Argument("", help="what you want done; asked in the session "
                                 "when you leave it out"),
    model: str = typer.Option("", "--model", help="model for the client to use"),
    continue_: bool = typer.Option(False, "--continue", "-c",
                                   help="continue this directory's last conversation"),
    resume: str = typer.Option("", "--resume", help="resume a conversation by id"),
    client: str = typer.Option("", "--client", help=_CLIENT_HELP),
) -> None:
    """Start an agent client (Claude Code, Qwen Code, Goose) wired to this harness.

    The directory is made a myrmidon project first if it is not one already (the same
    thing `myrmidon init-project` does), so the client finds the server, the guidance,
    the status line, the session-start hook and the skills where it looks for them.
    """
    _launch(prompt=prompt, model=model, continue_=continue_, resume=resume, client=client)


@app.command("version")
def version_command() -> None:
    """Print the myrmidon version."""
    typer.echo(myrmidon.__version__)


@app.command()
def serve(
    check: bool = typer.Option(
        False, "--check", help="do not serve: source env_script, discover engines, exit "
        "non-zero naming the first missing engine"),
    workspace: str = typer.Option("", "--study",
                                 help="open this study before serving"),
    mock: bool = typer.Option(False, "--mock", help="serve fake engines and a fake "
                              "scheduler: nothing real runs, job records stay in the study"),
    unattended: bool = typer.Option(False, "--unattended", help="no human sees the calls: "
                                    "findings that need a human cannot be acknowledged"),
    config: str = typer.Option("", "--config", help=_CONFIG_HELP),
) -> None:
    """Run the MCP server on stdio: every tool this machine's engines support. Sources
    config.env_script first, so the client that launches it need not have (register it
    with `myrmidon mcp install`)."""
    if check:
        from myrmidon.core.envscript import EnvScriptStatus
        from myrmidon.server.mcp_server import build_context

        ctx = build_context(_config_file(config))
        typer.echo(EnvScriptStatus(**ctx.env_script).describe() if ctx.env_script
                   else "env_script: not set")
        missing = [(n, i.hint) for n, i in ctx.engines.items() if not i.available]
        for n, i in ctx.engines.items():
            if i.available:
                typer.echo(f"  ✓ {n} {i.version}")
        for n, hint in missing:
            typer.echo(f"  ✗ {n}: {hint}")
        if missing:
            typer.echo(f"FAIL: {missing[0][0]} missing")
            raise typer.Exit(1)
        typer.echo("OK: every engine found")
        return
    from pathlib import Path

    from myrmidon.server.mcp_server import build_context, build_server

    ctx = serve_context(workspace=workspace, mock=mock, unattended=unattended, config=config)
    build_server(ctx=ctx, workspace=Path(workspace) if workspace else None).run()


def serve_context(workspace: str = "", mock: bool = False, unattended: bool = False,
                  config: str = ""):
    """The context `serve` runs with (factored out so tests can build it)."""
    from pathlib import Path

    from myrmidon.server.mcp_server import build_context

    if mock:
        from myrmidon.engines import mock as mocks

        if not workspace:
            typer.echo("--mock needs --study (the mock scheduler keeps its state there)")
            raise typer.Exit(2)
        # before the context is built, so discovery finds the mock binaries
        mocks.enter(Path(workspace))
        ctx = mocks.mock_context(build_context(_config_file(config)))
    else:
        ctx = build_context(_config_file(config))
    ctx.unattended = unattended
    return ctx


def _config_file(path: str):
    """The config `--config` names, read alone; None without the flag (the usual layering).
    A file that is missing or does not parse is the person's mistake: said, not raised."""
    if not path:
        return None
    from myrmidon.config import MyrmidonConfig
    from myrmidon.errors import ConfigError

    try:
        return MyrmidonConfig.from_file(path)
    except ConfigError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(2) from None


@app.command()
def doctor() -> None:
    """Report discovered engines, versions, and what is missing."""
    from myrmidon.server.mcp_server import build_context

    ctx = build_context()
    _print_doctor(ctx)


def _describe_compute(ctx) -> str:
    """One line saying what the harness thinks this machine is and what it will refuse.

    Worth printing because both numbers drive gates that only fire later, at submission
    time, when the reason for them has left the screen.
    """
    c = ctx.config.compute
    if ctx.config.backend != "local":
        name, node = c.billing()
        shape = (f"{node.cores_per_node} cores per node"
                 + (", billed whole" if node.node_exclusive else "")
                 + (f" on partition {name}, as the scheduler states it ("
                    f"{len(c.partitions)} partition(s) priced)" if name else
                    " (no partition read from the scheduler: re-run `myrmidon init`)"))
        account = ctx.config.slurm.account
        shape += (f", to account {account} unless a submission names another"
                  if account else ", to the scheduler's default account")
    elif c.cores_per_node > 1:
        shape = f"{c.cores_per_node} cores here, and a bigger request is flagged"
    else:
        shape = "machine size not recorded — re-run `myrmidon init` to enable that check"
    return (f"{ctx.config.backend} backend: {shape}; "
            f"submissions above {c.warn_core_h} core-h need acknowledging")


def _compute_drift(ctx) -> list[str]:
    """Where the config's partitions and what the scheduler says now disagree: a price
    computed from a stale table is wrong without a word."""
    if ctx.config.backend != "slurm":
        return []
    from myrmidon.execution.billing import read_partitions

    live, _ = read_partitions()
    if not live:
        return []
    known = ctx.config.compute.partitions
    out = [f"partition {name}: config has {known[name].model_dump()}, the scheduler "
           f"says {node.model_dump()}"
           for name, node in sorted(live.items()) if name in known and known[name] != node]
    unpriced = sorted(set(live) - set(known))
    if unpriced:
        out.append("partition(s) the config does not price (machine-wide numbers are "
                   f"used): {', '.join(unpriced)} — re-run `myrmidon init`")
    return out


def _print_doctor(ctx) -> None:
    from myrmidon.core.tool import REGISTRY

    obs = REGISTRY.call("doctor", ctx, {})
    typer.echo(obs.summary)
    for name, version_ in obs.engines.items():
        typer.echo(f"  ✓ {name} {version_}")
    for name, hint in obs.missing.items():
        typer.echo(f"  ✗ {name}: {hint}")
    from myrmidon.server.mcp_server import approx_tokens, exposed_tools

    shown = exposed_tools(ctx)
    cost = sum(approx_tokens(s) for s in shown)
    typer.echo(f"  tools: {len(shown)} shown to the model (~{cost // 1000}k tokens of "
               "schema per turn)")
    typer.echo("  " + _describe_compute(ctx))
    for line in _compute_drift(ctx):
        typer.echo(f"  ! {line}")
    from myrmidon.core.envscript import EnvScriptStatus
    from myrmidon.server.clients import describe

    typer.echo("  " + (EnvScriptStatus(**obs.env_script).describe() if obs.env_script
                       else "env_script: not set"))
    typer.echo("  agent clients:")
    for line in describe(obs.clients):
        typer.echo(f"    {line}")
    from myrmidon.server.launch import resolve_client

    key, source = resolve_client()
    typer.echo(f"  `myrmidon client` starts: {key}"
               + ("" if source == "default" else f"  (client = {key!r} in {source})"))
    if obs.workspace_strays:
        typer.echo(
            f"  ! {len(obs.workspace_strays)} stray entr"
            f"{'y' if len(obs.workspace_strays) == 1 else 'ies'} under studies_root "
            f"(move tests/debug runs to _scratch/): {', '.join(obs.workspace_strays)}"
        )


@app.command()
def init(
    backend: str = typer.Option("local", help="local | slurm"),
    studies_root: str = typer.Option("", "--studies-root",
                                     help="where studies live (HPC: on scratch)"),
    env_script: str = typer.Option("", help="shell script that provides the engines"),
    partition: str = typer.Option("", help="SLURM partition (backend=slurm)"),
    account: str = typer.Option("", help="SLURM account (backend=slurm)"),
    force: bool = typer.Option(False, "--force", help="overwrite an existing config"),
    path: str = typer.Option("", help="config file (default: $MYRMIDON_CONFIG or "
                             "~/.config/myrmidon/config.toml)"),
) -> None:
    """Write the machine config from a commented template (engines discovered on PATH),
    then run doctor. An existing config is never overwritten without --force: the diff of
    what init would write is printed instead."""
    from pathlib import Path

    from myrmidon.bootstrap import discover_plan, init_config

    if backend not in ("local", "slurm"):
        typer.echo("backend must be local or slurm")
        raise typer.Exit(2)
    plan = discover_plan(backend=backend, studies_root=studies_root or None,
                         env_script=env_script or None, partition=partition, account=account)
    result = init_config(plan, Path(path).expanduser() if path else None, force=force)
    if not result.written:
        typer.echo(f"{result.path} exists — not overwritten (use --force). "
                   + ("It already matches what init would write." if not result.diff
                      else "Diff against what init would write:"))
        if result.diff:
            typer.echo(result.diff)
        raise typer.Exit(0)
    typer.echo(f"wrote {result.path}")
    for key, ok in plan.found.items():
        typer.echo(f"  {'✓' if ok else '✗'} {key} = {getattr(plan, key)}"
                   + ("" if ok else "  (not on PATH: edit, or name it in env_script)"))
    if result.secrets_created:
        typer.echo(f"  created {result.path.parent / 'secrets.env'} (chmod 600) "
                   "for ANTHROPIC_API_KEY")
    typer.echo("")
    from myrmidon.config import MyrmidonConfig
    from myrmidon.server.mcp_server import build_context

    _print_doctor(build_context(MyrmidonConfig.load(result.path)))


@app.command("init-project")
def init_project_cmd(
    directory: str = typer.Argument("", help="project directory (default: current)"),
    client: str = typer.Option("", "--client",
                               help=f"{_CLIENT_KEYS} | all (comma-separated works too); "
                               "default: the project's recorded client, else the machine "
                               "config's, else claude"),
    name: str = typer.Option("myrmidon", help="server name in the client config"),
    no_server: bool = typer.Option(False, "--no-server", help="guidance block only: skip the "
                                   "project-scoped MCP entry (the server is already registered "
                                   "at user scope by `mcp install`)"),
    studies_root: str = typer.Option("", "--studies-root",
                                     help="where this "
                                        "project's studies are written ('workspaces' puts "
                                        "them beside the inputs); default: the machine "
                                        "config's. Written to .myrmidon.toml, yours to edit"),
) -> None:
    """Make a research repo a myrmidon project: the project-scoped MCP entry plus the
    guidance block in the file the client reads (CLAUDE.md / AGENTS.md / .cursor/rules).
    Re-running updates the block in place; nothing else in those files is touched."""
    from pathlib import Path

    from myrmidon.server.install import CLIENTS, init_project

    if not client:
        # the default is what the project already says, so a project wired for one client
        # and re-run without --client does not get another client's files on top
        from myrmidon.server.launch import resolve_client

        client = resolve_client()[0]
    clients = tuple(CLIENTS) if client == "all" else tuple(c.strip() for c in client.split(","))
    try:
        result = init_project(Path(directory) if directory else None, clients, name=name,
                              server=not no_server, studies_root=studies_root or None)
    except (ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(1) from None
    typer.echo(f"project: {result.directory}")
    for path, what in result.files:
        typer.echo(f"  {what:10s} {path}")
    # where the work lands: it is not this directory, and nothing else says so
    from myrmidon.server.launch import destination

    typer.echo(destination())
    # what a person does next is the client's business, and differs: Claude Code and Qwen
    # Code gate a fresh project's server until a human approves it once, and a headless
    # session has no way to; goose has no such step. Myrmidon never accepts a
    # trust dialog on the user's behalf: a tool that marks its own output as trusted
    # disables the control that protects against tools like itself.
    for c in dict.fromkeys(clients):
        typer.echo(REGISTRY[c].next_hint)


mcp_app = typer.Typer(help=f"Wire agent clients ({_CLIENT_NAMES}) to this server.",
                      no_args_is_help=True)
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("install")
def mcp_install(
    client: str = typer.Option(..., "--client", help=_CLIENT_KEYS),
    scope: str = typer.Option("user", help="user (every project) | project (./.mcp.json)"),
    directory: str = typer.Option("", "--dir", help="project directory for --scope project"),
    name: str = typer.Option("myrmidon", help="server name in the client config"),
    no_cli: bool = typer.Option(False, "--no-cli", help="claude: edit the config file "
                                "directly instead of calling `claude mcp add`"),
) -> None:
    """Register this install's `myrmidon serve` with an agent client. The entry uses the
    absolute path of this environment's myrmidon, so it works from any shell."""
    from pathlib import Path

    from myrmidon.server.install import install

    try:
        result = install(client, scope=scope, directory=Path(directory) if directory else None,
                         name=name, use_cli=not no_cli)
    except (ValueError, NotImplementedError, RuntimeError, json.JSONDecodeError) as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(1) from None
    cmd = " ".join([result.command, *result.args])
    state = ("registered" if result.changed else
             "not written" if result.via == "manual" else "already registered") + (
        " via `claude mcp add`" if result.via == "claude-cli" else "")
    typer.echo(f"{result.client} ({result.scope}): {state}\n  {name} -> {cmd}\n  {result.file}")
    for n in result.notes:
        typer.echo(f"  {n}")
    typer.echo("Verify: `myrmidon serve --check`, then `myrmidon doctor` (agent clients), "
               "then /mcp inside a new session.")


@mcp_app.command("show")
def mcp_show(
    client: str = typer.Option("claude", "--client", help=_CLIENT_KEYS),
    name: str = typer.Option("myrmidon", help="server name in the client config"),
) -> None:
    """Print the client config snippet for this install's server (paste it yourself)."""
    from myrmidon.server.install import show

    try:
        typer.echo(show(client, name=name))
    except ValueError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(1) from None


def _agent_endpoint(endpoint: str) -> str | None:
    """The endpoint asked for, else the one used last (if still defined), else the only one
    defined; its key must be set now."""
    from myrmidon.agent.session import default_endpoint, load_prefs
    from myrmidon.endpoints import EndpointError, get_endpoint, load_endpoints

    last = load_prefs().endpoint
    name = endpoint or (last if last in load_endpoints() else None) or default_endpoint()
    if name:
        try:
            get_endpoint(name).key()
        except EndpointError as exc:
            typer.echo(str(exc))
            raise typer.Exit(2) from None
    return name


def _agent_context(study: str, workspace_hint: str | None = None, config=None):
    """The tool context, with the study to work in open: the one asked for, else the
    session's own (resume), else this project's active study, else none yet."""
    from pathlib import Path

    from myrmidon.server.mcp_server import build_context

    ctx = build_context(config)
    if study:
        ctx.workspace = _resolve_workspace(study)
    elif workspace_hint and Path(workspace_hint).is_dir():
        ctx.workspace = _resolve_workspace(workspace_hint)
    else:
        ctx.workspace = _resolve_workspace("", ctx.config)
    return ctx


@app.command("run")
def agent_run(
    prompt: str,
    model: str = typer.Option("", "--model", "-m", help=_MODEL_HELP),
    endpoint: str = typer.Option("", "--endpoint", "-e", help=_ENDPOINT_HELP),
    max_turns: int = typer.Option(40),
    call_timeout: float = typer.Option(0, "--call-timeout", help=_TIMEOUT_HELP,
                                      show_default=False),
    study: str = typer.Option("", "--study", help="study to work in (default: active)"),
    config: str = typer.Option("", "--config", help=_CONFIG_HELP),
    mock: bool = typer.Option(False, "--mock", help="fake engines and a fake scheduler: "
                              "nothing real runs"),
) -> None:
    """One question, unattended (for scripts).

    The agent works until it answers or stops; a finding that needs a human is refused,
    and the run reports it instead of accepting it.

    Exit code: 0 answered, with every number traced to a tool's output; 1 no such answer
    (out of turns, tokens or time, a reply cut off, or numbers that could not be traced);
    3 the model or its endpoint failed. 2 is a usage error, as for every command.

    Every call goes through the same registry a person types at, so the gates and the
    ledger apply unchanged — and `myrmidon watch` shows what it is doing.
    """
    from myrmidon.agent.session import build_loop, run_transcripts

    chosen = _agent_endpoint(endpoint)
    model = _agent_model(model, chosen, interactive=False)
    chosen_config = _config_file(config)
    if mock:
        from pathlib import Path

        from myrmidon.config import MyrmidonConfig
        from myrmidon.engines import mock as mocks

        # before the context is built, so discovery finds the mock binaries; the fake
        # scheduler keeps its records in the study, else under the studies root — an
        # absolute path, since a mock job runs in its own stage directory
        mocks.enter(Path(study).resolve() if study
                    else Path((chosen_config or MyrmidonConfig.load()).studies_root).resolve())
    ctx = _agent_context(study, config=chosen_config)
    if mock:
        ctx = mocks.mock_context(ctx)
    loop = build_loop(ctx, model, endpoint=chosen,
                      call_timeout_s=call_timeout or None, max_turns=max_turns,
                      transcript_dir=run_transcripts(ctx))
    episode = loop.run(prompt)
    typer.echo(f"episode {episode.episode_id}: {episode.stop_reason}")
    typer.echo(f"transcript: {loop.transcript.path}")
    if episode.error:
        typer.echo(episode.error)
    typer.echo(f"turns={episode.n_turns} tool_calls={episode.n_tool_calls} "
               f"tokens={episode.prompt_tokens}+{episode.completion_tokens}")
    if episode.grounding:
        typer.echo(
            f"grounding: {episode.grounding.checked} numbers checked, "
            f"{len(episode.grounding.ungrounded)} ungrounded"
        )
    typer.echo("")
    typer.echo(episode.final_answer or "(no final answer)")
    raise typer.Exit(run_exit_code(episode))


def run_exit_code(episode) -> int:
    """What `myrmidon run` exits with, so a script can tell an answer from a run that
    stopped short, and both from a model that could not be reached."""
    from myrmidon.agent.loop import StopReason

    if episode.stop_reason == StopReason.MODEL_ERROR:
        return 3
    if episode.stop_reason == StopReason.COMPLETED and (
            episode.grounding is None or episode.grounding.ok):
        return 0
    return 1


def _chat(state, ctx, resumed: bool, first: str | None, level: str = "research",
          call_timeout: float = 0) -> None:
    from rich.console import Console

    from myrmidon.agent.chat import HELP, Renderer, make_decider, repl
    from myrmidon.agent.prompts import chat_system_prompt
    from myrmidon.agent.session import (
        CHAT_LIMITS,
        ChatSession,
        build_loop,
        sessions_root,
    )

    console = Console()
    renderer = Renderer(console, level=level)
    loop = build_loop(ctx, state.model, endpoint=state.endpoint, on_event=renderer,
                      on_decision=make_decider(console, renderer, console.is_terminal),
                      call_timeout_s=call_timeout or None, **CHAT_LIMITS)
    root = sessions_root(ctx.config.studies_root)
    session = (ChatSession.resume(loop, state, root) if resumed
               else ChatSession.begin(loop, state, root, chat_system_prompt()))
    study = ctx.workspace.root if ctx.workspace is not None else "none yet"
    console.print(f"[bold]myrmidon[/] · {state.model}"
                  + (f" @ {state.endpoint}" if state.endpoint else "")
                  + f" · study: {study}")
    console.print(f"[dim]session {state.session_id}"
                  + (f" resumed ({state.asks} question(s) so far)" if resumed else "")
                  + " · /help for commands[/]")
    if not resumed and first is None:
        console.print(f"[dim]{HELP.splitlines()[0]}[/]")
    repl(session, console, renderer, first=first)


def _agent_model(model: str, endpoint: str | None, interactive: bool) -> str:
    """The model asked for, else the one used last on this endpoint, else — at a
    terminal — one picked from what the endpoint serves. Remembered for next time."""
    from myrmidon.agent.session import load_prefs, save_prefs, served_models

    prefs = load_prefs()
    chosen = model or (prefs.model if prefs.endpoint == endpoint else None)
    if not chosen and interactive:
        from rich.console import Console

        from myrmidon.agent.chat import pick

        served = served_models(endpoint)
        if not served:
            typer.echo(f"could not list the models of {endpoint or 'your endpoint'}: give "
                       "one with --model (`myrmidon endpoints` shows the endpoints)")
            raise typer.Exit(2)
        chosen = pick(Console(), f"which model? ({endpoint} serves these)", served, None)
    if not chosen:
        typer.echo("no model chosen: run `myrmidon` in a terminal to pick one, or pass "
                   "--model")
        raise typer.Exit(2)
    if chosen != prefs.model or endpoint != prefs.endpoint:
        save_prefs(endpoint, chosen)
    return chosen


def _native(prompt: str = "", model: str = "", endpoint: str = "", continue_: bool = False,
            resume: str = "", call_timeout: float = 0, verbose: bool = False,
            debug: bool = False) -> None:
    """Bare `myrmidon` and `myrmidon chat`: a new session, or the one to continue."""
    import sys

    from myrmidon.agent.session import SessionError, load, new_state, sessions_root

    level = "debug" if debug else "verbose" if verbose else "research"
    from myrmidon.config import MyrmidonConfig

    if continue_ or resume:
        try:
            state = load(sessions_root(MyrmidonConfig.load().studies_root), resume or None)
        except SessionError as exc:
            typer.echo(str(exc))
            raise typer.Exit(2) from None
        if state.endpoint:
            _agent_endpoint(state.endpoint)  # its key must be set in this shell too
        _chat(state, _agent_context("", state.workspace), resumed=True,
              first=prompt or None, level=level, call_timeout=call_timeout)
        return
    chosen = _agent_endpoint(endpoint)
    model = _agent_model(model, chosen, interactive=sys.stdin.isatty())
    state = new_state(model, "pydantic", chosen)
    _chat(state, _agent_context(""), resumed=False, first=prompt or None, level=level,
          call_timeout=call_timeout)


@app.command("sessions")
def agent_sessions() -> None:
    """Saved sessions, most recent first."""
    from myrmidon.agent.session import describe, list_sessions, sessions_root
    from myrmidon.config import MyrmidonConfig

    sessions = list_sessions(sessions_root(MyrmidonConfig.load().studies_root))
    if not sessions:
        typer.echo("no saved sessions — `myrmidon` starts one")
        return
    for state in sessions:
        typer.echo(describe(state))


_FINDINGS_ARG = typer.Argument(None, help="finding ids to accept for the study, e.g. "
                               "COMPUTE_BUDGET; none: list the approvals that hold")


@app.command("approve")
def approve(
    findings: list[str] | None = _FINDINGS_ARG,
    study: str = typer.Option("", "--study", help="the study (default: the active one)"),
    approver: str = typer.Option("", "--approver", help="who is approving (default: your "
                                 "user name)"),
    config: str = typer.Option("", "--config", help=_CONFIG_HELP),
) -> None:
    """Accept findings for a study, as the person responsible for it.

    An unattended session (`myrmidon run`, an agent client with nobody watching) may then
    acknowledge exactly these findings in this study, and no others; the ledger names this
    approval in every call that relies on it. It holds until the study ends: the ledger is
    append-only, so a grant is not withdrawn but superseded by a new study. Approving a
    plan instead (`plan_approve`) ties the approval to that plan's contents.
    """
    import getpass
    import re

    from myrmidon.core.approvals import grant, holding

    ws = _resolve_workspace(study, _config_file(config))
    if ws is None:
        typer.echo("no study: give --study PATH, or open one (`myrmidon study open PATH`)")
        raise typer.Exit(2)
    if not findings:
        current = holding(ws)
        if not current:
            typer.echo(f"no approvals hold in {ws.root}")
        for a in current:
            scope = ("plan" if a.plan_sha256 else "one call" if a.call else "study")
            typer.echo(f"{a.event_id}  {', '.join(a.findings)}  by {a.approver}  "
                       f"(for the {scope})")
        return
    malformed = [f for f in findings if not re.fullmatch(r"[A-Z][A-Z0-9_]+", f)]
    if malformed:
        typer.echo(f"not a finding id: {', '.join(malformed)} — ids are upper case, like "
                   "COMPUTE_BUDGET, as a refusal names them")
        raise typer.Exit(2)
    who = approver or getpass.getuser()
    event = grant(ws, sorted(set(findings)), approver=who)
    typer.echo(f"{event}: {who} accepts {', '.join(sorted(set(findings)))} for {ws.root}")


def _loop_context(study: str, config: str, mock: bool):
    """The tool context a loop command works in: the study named, else the active one, on
    the mock engines when asked (their scheduler state lives in the study)."""
    from pathlib import Path

    chosen = _config_file(config)
    if mock:
        from myrmidon.config import MyrmidonConfig
        from myrmidon.engines import mock as mocks

        # resolved: the mock scheduler's state directory must be the same absolute path
        # from every working directory a mock job runs in
        mocks.enter(Path(study).resolve() if study
                    else Path((chosen or MyrmidonConfig.load()).studies_root).resolve())
    ctx = _agent_context(study, config=chosen)
    if mock:
        from myrmidon.engines import mock as mocks

        ctx = mocks.mock_context(ctx)
    if ctx.workspace is None:
        typer.echo("no study: give --study PATH, or open one (`myrmidon study open PATH`)")
        raise typer.Exit(2)
    return ctx


def _wake_for(ctx, model: str, endpoint: str):
    """A wake model when one is named; None means every question goes to the inbox."""
    if not model and not endpoint:
        return None
    from myrmidon.agent.pydantic_adapter import PydanticCompletion
    from myrmidon.core.wake import ModelWake

    chosen = _agent_endpoint(endpoint)
    name = _agent_model(model, chosen, interactive=False)
    connection = {}
    if chosen:
        from myrmidon.endpoints import get_endpoint

        connection = get_endpoint(chosen).call_kwargs()
    return ModelWake(PydanticCompletion(), name, connection, name=name)


@app.command("tick")
def tick_cmd(
    study: str = typer.Option("", "--study", help="the study (default: the active one)"),
    follow: float = typer.Option(0, "--follow", help="seconds between ticks; 0 runs one "
                                 "tick and exits. The loop is crash-only: kill it at any "
                                 "instant and start it again"),
    until_done: bool = typer.Option(False, "--until-done", help="with --follow: stop when "
                                    "the plan is done or every branch waits on a person"),
    model: str = typer.Option("", "--model", "-m", help="a wake model: a parked question "
                              "is put to it before it goes to the inbox"),
    endpoint: str = typer.Option("", "--endpoint", "-e", help=_ENDPOINT_HELP),
    config: str = typer.Option("", "--config", help=_CONFIG_HELP),
    mock: bool = typer.Option(False, "--mock", help="fake engines and a fake scheduler"),
    cron_line: bool = typer.Option(False, "--cron-line", help="print a cron entry that "
                                   "ticks this study every ten minutes, and exit"),
) -> None:
    """Advance the study by one pass: collect what ended, check it, continue what is
    ready, park what needs a person. Nothing runs without a holding plan approval.

    Between ticks only the engines move: there is no resident process, and a tick killed
    at any instant is reconciled by the next one rather than repeated. Exit code: 0 when
    something was done or nothing is pending, 1 when the plan is not approved or the study
    has no plan, 3 when the study is locked by another tick.
    """
    import time

    from myrmidon.core.loop import render, tick

    ctx = _loop_context(study, config, mock)
    if cron_line:
        flags = f" --study {ctx.workspace.root}" + (f" --config {config}" if config else "")
        typer.echo(f"*/10 * * * * myrmidon tick{flags} >> "
                   f"{ctx.workspace.state_dir}/tick.log 2>&1")
        return
    wake = _wake_for(ctx, model, endpoint)
    while True:
        report = tick(ctx, wake=wake)
        typer.echo(render(report))
        if not follow:
            if report.blocked:
                raise typer.Exit(3 if "another tick" in report.blocked else 1)
            return
        if until_done and (report.done or (report.idle and report.parked
                                           and not report.running)):
            return
        if report.blocked and "another tick" not in report.blocked:
            raise typer.Exit(1)
        time.sleep(follow)


@app.command("inbox")
def inbox_cmd(
    study: str = typer.Option("", "--study", help="the study (default: the active one)"),
    config: str = typer.Option("", "--config", help=_CONFIG_HELP),
    all_entries: bool = typer.Option(False, "--all", help="answered questions and notices too"),
) -> None:
    """What waits on a person: each parked question with its evidence and its options.
    Answer one with `myrmidon answer ID OPTION`."""
    from myrmidon.core.spool import Inbox

    ws = _resolve_workspace(study, _config_file(config))
    if ws is None:
        typer.echo("no study: give --study PATH, or open one (`myrmidon study open PATH`)")
        raise typer.Exit(2)
    inbox = Inbox(ws.root)
    entries = inbox.all() if all_entries else inbox.open_questions()
    if not entries:
        typer.echo("nothing waits on you" if not all_entries else "the inbox is empty")
        return
    for e in entries:
        if e.answers:
            typer.echo(f"{e.ts}  answer to {e.answers}: {e.option} by {e.by}")
            continue
        typer.echo(f"{e.entry_id}  {e.kind}  {e.subject}  ({e.ts})")
        typer.echo(f"  {e.question}")
        for line in e.evidence:
            if line:
                typer.echo(f"    - {line}")
        for o in e.options:
            cost = f"  ({o.cost_core_h:g} core-h)" if o.cost_core_h else ""
            typer.echo(f"    {o.id:<10} {o.text}{cost}")
        if e.cites:
            typer.echo("    cites " + ", ".join(c for c in e.cites if c))


@app.command("answer")
def answer_cmd(
    entry_id: str = typer.Argument(..., help="the inbox question (in-…)"),
    option: str = typer.Argument(..., help="one of its option ids"),
    study: str = typer.Option("", "--study", help="the study (default: the active one)"),
    by: str = typer.Option("", "--by", help="who answers (default: your user name)"),
    config: str = typer.Option("", "--config", help=_CONFIG_HELP),
) -> None:
    """Answer a parked question by its option id. The next tick carries it out."""
    import getpass

    from myrmidon.core.spool import Inbox, Spool

    ws = _resolve_workspace(study, _config_file(config))
    if ws is None:
        typer.echo("no study: give --study PATH, or open one (`myrmidon study open PATH`)")
        raise typer.Exit(2)
    inbox = Inbox(ws.root)
    question = inbox.get(entry_id)
    if question is None:
        typer.echo(f"no question {entry_id} in {ws.root}; `myrmidon inbox` lists them")
        raise typer.Exit(2)
    who = by or getpass.getuser()
    try:
        inbox.answer(entry_id, option, by=who)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(2) from None
    chosen = next(o for o in question.options if o.id == option)
    from myrmidon.core.plan_binding import current_plan_id

    Spool(ws.root).raise_event("human", question.subject, plan_id=current_plan_id(ws),
                               inbox=entry_id, option=option, action=chosen.action,
                               args=chosen.args, by=who)
    typer.echo(f"{entry_id}: {option} — {chosen.text}. The next tick carries it out.")


@app.command("status")
def status_cmd(
    study: str = typer.Option("", "--study", help="the study (default: the active one)"),
    config: str = typer.Option("", "--config", help=_CONFIG_HELP),
) -> None:
    """Where the study stands, from its own files and no model: the plan's steps, the
    jobs, what waits on a person, the open journal, the last events."""
    from myrmidon.core.spool import Inbox, Journal, LoopState, Spool
    from myrmidon.core.tool import REGISTRY
    from myrmidon.engines.gromacs.results import pending_lines

    ctx = _loop_context(study, config, mock=False)
    ws = ctx.workspace
    typer.echo(f"study {ws.meta.get('name', ws.root.name)} · {ws.root}")
    obs = REGISTRY.call("plan_check", ctx, {})
    if obs.ok:
        state = LoopState.load(ws.state_dir)
        width = max((len(s["name"]) for s in obs.steps), default=4)
        for step in obs.steps:
            extra = ""
            runs = [k for k in state.runs if k.split("[")[0] == step["name"]]
            if len(runs) > 1 or any("[" in k for k in runs):
                extra = f"  arms: {', '.join(runs)}"
            typer.echo(f"  [{'x' if step['state'] != 'pending' else ' '}] "
                       f"{step['name'].ljust(width)}  {step['tool']}{extra}")
        typer.echo("  " + obs.summary.split(". ")[-1])
    else:
        typer.echo("  " + obs.summary)
    jobs = pending_lines(ctx)
    if jobs:
        typer.echo("jobs:")
        for line in jobs:
            typer.echo(f"  {line}")
    questions = Inbox(ws.root).open_questions()
    if questions:
        typer.echo("waiting on you:")
        for q in questions:
            typer.echo(f"  {q.entry_id}  {q.subject}: {q.question[:120]}")
    entries = Journal(ws.root).open_entries()
    if entries:
        typer.echo("open in the journal:")
        for e in entries:
            typer.echo(f"  [{e.kind}, {e.by}] {e.text[:140]}")
    events = Spool(ws.root).all()[-5:]
    if events:
        typer.echo("last events:")
        for ev in events:
            typer.echo(f"  {ev.ts}  {ev.kind:<9} {ev.subject}  "
                       + (ev.detail.get("summary") or ev.detail.get("detail") or "")[:100])


@app.command("endpoints")
def endpoints_cmd() -> None:
    """The named model endpoints (endpoints.toml beside the machine config), and whether
    each one's key variable is set in this shell. Never prints a key."""
    import os

    from myrmidon.endpoints import EndpointError, endpoints_path, load_endpoints

    path = endpoints_path()
    try:
        endpoints = load_endpoints(path)
    except EndpointError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from None
    if not endpoints:
        typer.echo(f"no endpoints defined ({path} absent or empty). One per server:\n\n"
                   "  [endpoints.NAME]\n  api_base = \"https://host/v1\"\n"
                   "  key_env = \"NAME_API_KEY\"   # the variable, not the key")
        return
    typer.echo(f"{path}:")
    for name, ep in sorted(endpoints.items()):
        state = "set" if os.environ.get(ep.key_env) else "NOT SET"
        typer.echo(f"  {name:<16} {ep.api_base}   ${ep.key_env} {state}")


def _workspace_path(workspace: str, config=None):
    """The path given, else the active workspace's, else None. Nothing is opened."""
    from pathlib import Path

    from myrmidon.basic_tools import active_workspace_path
    from myrmidon.config import MyrmidonConfig

    if workspace:
        return Path(workspace)
    root = Path((config or MyrmidonConfig.load()).studies_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    return active_workspace_path(root, Path.cwd())


def _resolve_workspace(workspace: str, config=None):
    """The workspace to work in: the path given, else the active one, else None.

    A directory that cannot be opened is a mistake a person makes at the terminal, not a
    bug: it is answered with the reason the workspace gave — no manifest, or a format this
    code does not read — and the commands that make or point at one, never a traceback.
    """
    from myrmidon.core.workspace import Workspace
    from myrmidon.errors import WorkspaceError

    path = _workspace_path(workspace, config)
    if path is None:
        return None
    try:
        return Workspace.open(path)
    except WorkspaceError as exc:
        typer.echo(str(exc))
        typer.echo("  myrmidon workspace create NAME   — make one")
        typer.echo("  myrmidon study open PATH         — point at an existing one")
        raise typer.Exit(2) from None


def _tool_context(workspace: str):
    """A ToolContext bound to a workspace when there is one.

    It does not refuse without one: half the registry (``study_create``,
    ``inputset_lookup``, ``doctor``, ``lookup``) needs no workspace, and for the other
    half ``REGISTRY.call`` says so itself.
    """
    from myrmidon.server.mcp_server import build_context

    ctx = build_context()
    ctx.workspace = _resolve_workspace(workspace)
    return ctx


@app.command()
def summary(
    workspace: str = typer.Argument("", help="workspace path (default: the active one)"),
    since: str = typer.Option("session", help="'session', 'all', '6h', '2d', or ISO time"),
    no_usage: bool = typer.Option(False, help="skip backend accounting queries (faster)"),
) -> None:
    """What went through the harness: tools, gates, jobs, core-hours, artifacts."""
    from myrmidon.core.summary import render, summarize
    from myrmidon.server.mcp_server import build_context

    ws = _resolve_workspace(workspace)
    if ws is None:
        typer.echo("no active study (open one through the tools, or pass a path)")
        raise typer.Exit(1)
    ctx = build_context()
    report = summarize(
        ws, since, backend=ctx.backend, compute=ctx.config.compute, poll_usage=not no_usage,
    )
    typer.echo(render(report))


@app.command()
def report(
    workspace: str = typer.Argument("", help="workspace path (default: the active one)"),
    no_usage: bool = typer.Option(False, help="skip backend accounting queries (faster)"),
) -> None:
    """Study report from the ledger alone: steps, acknowledgments, artifacts, jobs, core-h."""
    from myrmidon.core.report import render_report
    from myrmidon.server.mcp_server import build_context

    ws = _resolve_workspace(workspace)
    if ws is None:
        typer.echo("no active study (open one through the tools, or pass a path)")
        raise typer.Exit(1)
    ctx = build_context()
    typer.echo(render_report(ws, compute=ctx.config.compute,
                             backend=None if no_usage else ctx.backend))


@app.command()
def replay(
    workspace: str = typer.Argument(..., help="source workspace to reproduce"),
    into: str = typer.Option("", help="new workspace directory (required with --execute)"),
    execute: bool = typer.Option(False, help="re-run the tools; default prints the plan"),
    script: bool = typer.Option(False, help="print the shell replay script instead"),
    job_timeout_s: int = typer.Option(7200, help="max wait per job during --execute"),
) -> None:
    """Reproduce a study from its ledger: the plan, the shell script, or a real re-run
    into a fresh workspace through the same tools, gates and backend, with artifact hashes
    compared at the end."""
    from pathlib import Path

    from myrmidon.core.replay import plan, render
    from myrmidon.core.replay import replay as _replay
    from myrmidon.core.workspace import Workspace

    ws = Workspace.open(Path(workspace))
    if script:
        typer.echo(ws.ledger.replay_script())
        return
    if not execute:
        for ev, action in plan(list(ws.ledger.events())):
            typer.echo(f"{ev.event_id}  {ev.tool:<22} {action}")
        typer.echo("\n(--execute --into DIR to run it)")
        return
    if not into:
        typer.echo("--execute needs --into DIR")
        raise typer.Exit(2)
    from myrmidon.server.mcp_server import build_context

    ctx = build_context()
    report = _replay(ws, ctx, Path(into), job_timeout_s=job_timeout_s)
    typer.echo(render(report))
    (Path(into) / "replay-report.txt").write_text(render(report) + "\n")
    raise typer.Exit(1 if report.n_failed else 0)


@app.command(name="fm-report")
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

    ws = _resolve_workspace(workspace)
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


@app.command()
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
    from myrmidon.core.watch import follow, header

    ws = _resolve_workspace(workspace)
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


@app.command()
def statusline(
    workspace: str = typer.Argument("", help="workspace path (default: the active one)"),
) -> None:
    """One line for a terminal status bar: active workspace, live jobs, core-hours.

    Always one line and exit 0: a status bar shows every line it is given, and a failing
    command sends Claude Code's to its fallback on every redraw. A workspace that cannot be
    opened is named here; `myrmidon summary` says why.
    """
    from myrmidon.core.summary import statusline as _statusline
    from myrmidon.core.workspace import Workspace
    from myrmidon.errors import WorkspaceError

    path = _workspace_path(workspace)
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
    from myrmidon.core.tool import REGISTRY

    lines = []
    for spec in sorted(REGISTRY.available(ctx.engines), key=lambda s: s.name):
        gate = " [gated]" if spec.gate else ""
        lines.append(f"{spec.name}{gate}: {spec.description.splitlines()[0]}")
    return lines


@app.command()
def tools() -> None:
    """List every tool the registry would expose on this machine."""
    from myrmidon.server.mcp_server import build_context

    for line in _tool_lines(build_context()):
        typer.echo(line)


#: repeatable --acknowledge, defined once (a typer.Option call in a default is ruff B008)
ACKNOWLEDGE_OPTION = typer.Option(
    [], "--acknowledge", help="id of a warn-level gate finding you accept (repeatable)"
)

#: `call` takes the tool's parameters as bare key=value words, so click must not try to
#: read them, and --help belongs to the tool rather than to the command
CALL_CONTEXT = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}


@app.command(context_settings=CALL_CONTEXT)
def call(
    typer_ctx: typer.Context,
    tool: str = typer.Argument("", help="tool name; with no tool, every tool is listed"),
    workspace: str = typer.Option("", "--study",
                                 help="study path (default: the "
                                  "active one)"),
    acknowledge: list[str] = ACKNOWLEDGE_OPTION,
    as_json: bool = typer.Option(False, "--json", help="print the whole observation as "
                                 "JSON — the same shape an agent is given"),
    params_file: str = typer.Option("", "--from", help="a YAML or JSON file holding the "
                                    "tool's parameters (a plan file for plan_write); "
                                    "key=value on the command line overrides it"),
) -> None:
    """Call any registered tool: `myrmidon call gmx_solvate structure=... topology=...`.

    The parameters are the tool's own, validated by the registry exactly as an agent's
    call is, and the call lands in the workspace's ledger the same way. `myrmidon call
    TOOL --help` prints the parameters. Lists take a repeated key or commas; dicts take
    JSON; `--from FILE` takes them all from a file. The exit code follows the observation:
    0 ok, 1 failed or refused, 2 a mistake in the command line itself.
    """
    import json as _json

    from myrmidon.basic_tools import register_all
    from myrmidon.cli_call import (
        UsageError,
        parse_assignments,
        render_observation,
        tool_help,
    )
    from myrmidon.core.tool import REGISTRY
    from myrmidon.errors import UnknownToolError
    from myrmidon.server.mcp_server import build_context

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
        if acknowledge:
            if "acknowledge_warnings" not in spec.params_model.model_fields:
                raise UsageError(f"{spec.name} has no findings to accept: it is not gated")
            listed = list(params.get("acknowledge_warnings", [])) + list(acknowledge)
            params["acknowledge_warnings"] = list(dict.fromkeys(listed))
    except UsageError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(2) from None

    ctx = _tool_context(workspace)
    obs = REGISTRY.call(spec.name, ctx, params)
    if as_json:
        typer.echo(_json.dumps(obs.model_dump(mode="json", exclude_none=True), indent=1))
    else:
        typer.echo(render_observation(obs))
    raise typer.Exit(0 if obs.ok else 1)


study_app = typer.Typer(
    help="Studies: one question each, with its directory, its ledger, and which is active.",
    no_args_is_help=True,
)
app.add_typer(study_app, name="study")
# `workspace` is an alias for `study`, so existing scripts keep working; hidden so the
# help lists one word for it
app.add_typer(study_app, name="workspace", hidden=True)


@study_app.command("create")
def study_create(
    name: str = typer.Argument(..., help="short study name; it becomes the directory "
                               "under studies_root, and may name subdirectories "
                               "('_scratch/try-3' — where smoke runs belong)"),
    description: str = typer.Option(..., "--description", help="one line: what this "
                                    "study is for. It is the first thing in the ledger."),
) -> None:
    """Create a study under studies_root and make it the active one.

    NAME may contain slashes: `_scratch/tests/acetone` puts the workspace where
    `myrmidon doctor` expects a throwaway run to be.
    """
    _call_and_exit("study_create", {"name": name, "description": description})


@study_app.command("open")
def study_open(
    path: str = typer.Argument("", help="path to an existing workspace; empty: report "
                               "the active one"),
) -> None:
    """Make a workspace active — or, with no path, say where the active one stands."""
    _call_and_exit("study_open", {"path": path})


protocol_app = typer.Typer(
    help="Curated inputsets: validated, source-bearing parameter recipes, layered "
         "engine < site < project. The only sanctioned source of 'default' parameters.",
    no_args_is_help=True,
)
app.add_typer(protocol_app, name="inputset")


plan_app = typer.Typer(
    help="This study's plan: what it is doing, in what order, for how much. Writing one "
         "is `call plan_write`; a plan runs nothing by itself.",
    no_args_is_help=True,
)
app.add_typer(plan_app, name="plan")


@plan_app.command("show")
def plan_show(
    workspace: str = typer.Option("", "--study", "-s", help="study directory"),
) -> None:
    """The plan as a person reads it: the steps, what each costs, and where its values
    come from. This is the same text a client's plan mode shows before it asks for the
    approval — the harness renders the plan and never implements the approval itself."""
    from myrmidon.plan_tools import read_plan, render_plan, unresolved

    ctx = _tool_context(workspace)
    plan = read_plan(ctx)
    if plan is None:
        typer.echo("this study has no plan — write one with `myrmidon call plan_write`")
        raise typer.Exit(2)
    typer.echo(render_plan(plan, digest=plan.digest(), still_open=unresolved(ctx, plan),
                           ctx=ctx))


@plan_app.command("check")
def plan_check_cmd(
    workspace: str = typer.Option("", "--study", "-s", help="study directory"),
) -> None:
    """The plan against the ledger: what has run, what has not, what was done that the
    plan never mentioned, and whether the plan on disk is still the approved one."""
    from myrmidon.core.tool import REGISTRY

    ctx = _tool_context(workspace)
    obs = REGISTRY.call("plan_check", ctx, {})
    if not obs.ok:
        typer.echo(obs.summary)
        raise typer.Exit(2)

    width = max((len(s["name"]) for s in obs.steps), default=4)
    mark = {"done": "x", "pending": " "}
    for step in obs.steps:
        typer.echo(f"  [{mark[step['state']]}] {step['name'].ljust(width)}  {step['tool']}")
    typer.echo("")
    for line in obs.summary.split(". "):
        typer.echo(line)
    # a study that has drifted from its approval is the one thing worth an exit code
    raise typer.Exit(0 if obs.approval_holds else 1)


@protocol_app.command("list")
def protocol_list() -> None:
    """Every input set this machine can see, with the layer it was found in."""
    from myrmidon.cli_call import render_protocol_list
    from myrmidon.core.tool import REGISTRY

    ctx = _tool_context("")
    obs = REGISTRY.call("inputset_lookup", ctx, {"query": ""})
    typer.echo(render_protocol_list(obs) if obs.ok else obs.summary)
    raise typer.Exit(0 if obs.ok else 1)


@protocol_app.command("show")
def protocol_show(
    inputset: str = typer.Argument(..., help="inputset id, from `myrmidon inputset list`"),
) -> None:
    """One input set: every value, why it is that value, what it refuses to supply, and
    the sources behind it. Rendering it into a real file is `call inputset_apply`."""
    from myrmidon.cli_call import render_protocol
    from myrmidon.core.tool import REGISTRY

    ctx = _tool_context("")
    obs = REGISTRY.call("inputset_lookup", ctx, {"query": inputset})
    hit = next((p for p in obs.inputsets if p.id == inputset), None)
    if hit is None:
        typer.echo(f"no inputset '{inputset}' — myrmidon inputset list")
        raise typer.Exit(2)
    typer.echo(render_protocol(hit))


def _call_and_exit(tool: str, params: dict, workspace: str = "") -> None:
    """A named command that is one tool call: same registry, same rendering."""
    from myrmidon.cli_call import render_observation
    from myrmidon.core.tool import REGISTRY

    obs = REGISTRY.call(tool, _tool_context(workspace), params)
    typer.echo(render_observation(obs))
    raise typer.Exit(0 if obs.ok else 1)


if __name__ == "__main__":
    app()
