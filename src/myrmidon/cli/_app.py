"""The CLI's application object and what its commands share: the help texts, the context
a command runs a tool in, and the study it names."""

from __future__ import annotations

import click
import typer
from typer._click.exceptions import UsageError as VendoredUsageError
from typer.core import TyperGroup

import myrmidon
from myrmidon.client_registry import REGISTRY

CLIENT_KEYS = " | ".join(REGISTRY)


CLIENT_NAMES = ", ".join(d.display for d in REGISTRY.values())


LAUNCHABLE = tuple(k for k, d in REGISTRY.items() if d.launchable)


CLIENT_HELP = (f"the agent client to start: {' | '.join(LAUNCHABLE)} (default: the "
                "project's or the machine config's `client`, else the first of pi, hermes, "
                "claude, qwen, goose installed here)")


MODEL_HELP = "the model to use (default: the one used last on this endpoint)"


ENDPOINT_HELP = "a named endpoint from endpoints.toml (default: the one you used last)"


CONFIG_HELP = ("read this config file only: no project .myrmidon.toml, no machine "
                "config, no MYRMIDON_* variables; keys it leaves out take the built-in "
                "defaults")


TIMEOUT_HELP = "seconds one model request may take before it is abandoned (default 180)"


PANEL_WORK = "Work"
PANEL_READ = "Read back"
PANEL_SETUP = "Set up"


class Door(TyperGroup):
    """The command group with one more way in: words that name no command are the
    researcher's question, and start the session with it typed."""

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except (click.UsageError, VendoredUsageError):  # typer vendors its own click
            first = args[0] if args else ""
            # one unknown word is a typo; a sentence is a question
            if first and not first.startswith("-") and (len(args) > 1 or " " in first):
                return super().resolve_command(ctx, ["ask", *args])
            raise


app = typer.Typer(
    cls=Door,
    name="myrmidon",
    help="Myrmidon: an agent harness for molecular simulation. `myrmidon` in your project "
         "starts your agent (Pi, Claude Code, ...) wired to the harness; `myrmidon "
         "'QUESTION'` starts it with the question typed. Then `approve` the mandate it writes, "
         "and read back with `status`. Everything else is the workshop: `myrmidon workshop` "
         "lists it.",
    no_args_is_help=False,
    invoke_without_command=True,
)


def show_version(shown: bool) -> None:
    """`--version` is the habit; `myrmidon version` is the other spelling."""
    if shown:
        typer.echo(myrmidon.__version__)
        raise typer.Exit(0)


def serve_context(workspace: str = "", mock: bool = False, unattended: bool = False,
                  config: str = ""):
    """The context `serve` runs with (factored out so tests can build it)."""

    from myrmidon.server.mcp_server import build_context

    if mock:
        from myrmidon.engines import mock as mocks

        chosen = config_file(config)
        # before the context is built, so discovery finds the mock binaries
        mocks.enter(mock_state_dir(workspace, chosen))
        ctx = mocks.mock_context(build_context(chosen))
    else:
        ctx = build_context(config_file(config))
    ctx.unattended = unattended
    return ctx


def config_file(path: str):
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


def agent_endpoint(endpoint: str) -> str | None:
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


def agent_context(study: str, workspace_hint: str | None = None, config=None):
    """The tool context, with the study to work in open: the one asked for, else the
    session's own (resume), else this project's active study, else none yet."""
    from pathlib import Path

    from myrmidon.server.mcp_server import build_context

    ctx = build_context(config)
    if study:
        ctx.workspace = resolve_workspace(study)
    elif workspace_hint and Path(workspace_hint).is_dir():
        ctx.workspace = resolve_workspace(workspace_hint)
    else:
        ctx.workspace = resolve_workspace("", ctx.config)
    return ctx


def loop_context(study: str, config: str, mock: bool):
    """The tool context a loop command works in: the study named, else the active one, on
    the mock engines when asked (their scheduler state lives in the study)."""

    chosen = config_file(config)
    if mock:
        from myrmidon.engines import mock as mocks

        mocks.enter(mock_state_dir(study, chosen))
    ctx = agent_context(study, config=chosen)
    if mock:
        from myrmidon.engines import mock as mocks

        ctx = mocks.mock_context(ctx)
    if ctx.workspace is None:
        typer.echo("no study: give --study PATH, or open one (`myrmidon study open PATH`)")
        raise typer.Exit(2)
    return ctx


def mock_state_dir(study: str, config=None):
    """Where the mock scheduler keeps its records: the study named, else the studies root.
    Absolute, since a mock job runs in its own stage directory, and the same rule for the
    server, the loop commands and `run`, so a session and a tick find the same jobs."""
    from pathlib import Path

    return Path(study).resolve() if study else studies_root_path(config).resolve()


def studies_root_path(config=None):
    from pathlib import Path

    from myrmidon.config import MyrmidonConfig

    root = Path((config or MyrmidonConfig.load()).studies_root)
    return root if root.is_absolute() else Path.cwd() / root


def workspace_path(workspace: str, config=None):
    """The study named — a path, or a bare name under the studies root — else the active
    workspace's path, else None. Nothing is opened."""
    from pathlib import Path

    from myrmidon.core.studies import active_workspace_path, resolve_study_path

    root = studies_root_path(config)
    if workspace:
        return resolve_study_path(root, workspace)
    return active_workspace_path(root, Path.cwd())


def resolve_workspace(workspace: str, config=None):
    """The workspace to work in: the one named, else the active one, else None.

    A named directory that cannot be opened is a mistake a person makes at the terminal,
    not a bug: it is answered with the reason the workspace gave — no manifest, or a format
    this code does not read — and the commands that make or point at one, never a
    traceback. An active one that cannot be opened is no active study, said in a line, so
    the command that would replace it can run.
    """
    from pathlib import Path

    from myrmidon.core.studies import active_workspace, resolve_study_path
    from myrmidon.core.workspace import Workspace
    from myrmidon.errors import WorkspaceError

    root = studies_root_path(config)
    if not workspace:
        ws, note = active_workspace(root, Path.cwd())
        if note:
            typer.echo(note, err=True)
        return ws
    try:
        return Workspace.open(resolve_study_path(root, workspace))
    except WorkspaceError as exc:
        typer.echo(str(exc))
        typer.echo("  myrmidon study create NAME --description '…'   — make one")
        typer.echo("  myrmidon study open NAME|PATH                   — point at an existing one")
        raise typer.Exit(2) from None


def tool_context(workspace: str):
    """A ToolContext bound to a workspace when there is one.

    It does not refuse without one: half the registry (``study_create``,
    ``inputset_lookup``, ``doctor``, ``lookup``) needs no workspace, and for the other
    half ``REGISTRY.call`` says so itself.
    """
    from myrmidon.server.mcp_server import build_context

    ctx = build_context()
    ctx.workspace = resolve_workspace(workspace)
    return ctx


def call_and_exit(tool: str, params: dict, workspace: str = "") -> None:
    """A named command that is one tool call: same registry, same rendering."""
    from myrmidon.cli_call import render_observation
    from myrmidon.core.tool import REGISTRY

    obs = REGISTRY.call(tool, tool_context(workspace), params)
    typer.echo(render_observation(obs))
    raise typer.Exit(0 if obs.ok else 1)
