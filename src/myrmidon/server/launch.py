"""``myrmidon client``: an agent client, in this directory, wired to the harness.

The loop, the sessions and the terminal UI are the user's own agent client; what this
install owns is the domain — the tools, the gates, the ledger, the jobs and the input
sets. So the launcher is deliberately thin: make the directory a project if
it is not one (the same `init_project` a person can run by hand) and exec the client in
it.

Nothing is passed on the command line that a file could carry. `.mcp.json` names the
server, the guidance block in CLAUDE.md is the system prompt, `.claude/settings.json`
holds the status line and the session-start hook, and `.claude/skills/` holds the
skills. Two things follow: plain `claude` in that directory behaves exactly as
`myrmidon` does, so nobody is trapped in our wrapper; and the guidance is in the model's
context once rather than twice.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from myrmidon.server import client_registry as registry
from myrmidon.server.client_registry import REGISTRY, ClientDescriptor

NESTED_HINT = (
    "already inside {display}: the myrmidon tools are here in this session "
    "(`/mcp` lists them). Run `myrmidon client` from a plain terminal to start a new one."
)


class LaunchError(RuntimeError):
    """Something the user can fix, said in a sentence rather than a traceback."""


def client_binary(desc: ClientDescriptor | str = "claude") -> str:
    """This machine's copy of the client, by env var or PATH."""
    desc = registry.get(desc) if isinstance(desc, str) else desc
    if not desc.launchable:
        raise LaunchError(f"{desc.display} is wired by `init-project`, not started by "
                          "myrmidon: open the directory in it yourself.")
    path = (os.environ.get(desc.bin_env) if desc.bin_env else None) or shutil.which(desc.binary)
    if not path:
        raise LaunchError(desc.install_hint)
    return path


def claude_binary() -> str:
    return client_binary("claude")


def launch_argv(binary: str, *, model: str = "", continue_: bool = False,
                resume: str = "", prompt: str = "",
                client: ClientDescriptor | str = "claude") -> list[str]:
    """The client's own command line, and nothing of ours.

    No ``--mcp-config``, ``--append-system-prompt`` or ``--settings``: those would make
    a session started by `myrmidon` different from one started by `claude`, in a
    directory whose files already say all three. Each option is spelled the way the
    client spells it (``flag_map``); one the client has no spelling for is an error
    naming the client's own way, not a flag it will reject on its own terms.
    """
    desc = registry.get(client) if isinstance(client, str) else client
    argv = [binary, *desc.launch_subcommand]
    wanted = (("model", model), ("continue", continue_), ("resume", resume),
              ("prompt", prompt))
    for option, value in wanted:
        if not value:
            continue
        if option not in desc.flag_map:
            how = desc.unsupported.get(option)
            raise LaunchError(
                f"{desc.display} has no --{option.replace('_', '-')} to pass this to"
                + (f"; its own way is {how}." if how else "."))
        flags = desc.flag_map[option]
        argv += list(flags)
        if value is not True:
            argv.append(value)
    return argv


def resolve_client(flag: str = "") -> tuple[str, str]:
    """Which client `myrmidon client` starts, and where that came from: the ``--client``
    flag, else the ``client`` key of the nearest project `.myrmidon.toml`, else the
    machine config's, else ``claude``. The key is one flat line so a project can state
    it; the source is named so `doctor` can say which file it was."""
    import tomllib

    from myrmidon.config import MyrmidonConfig

    if flag:
        return flag, "--client"
    for path in MyrmidonConfig.candidates():
        if not path.is_file():
            continue
        try:
            data = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError:
            continue
        if isinstance(data.get("client"), str):
            return data["client"], str(path)
    return "claude", "default"


def wired_here() -> list[str]:
    """The launchable clients whose binary this machine has — the alternatives named
    when the resolved one is missing, so the user picks rather than myrmidon."""
    out = []
    for desc in REGISTRY.values():
        if not desc.launchable:
            continue
        if (desc.bin_env and os.environ.get(desc.bin_env)) or shutil.which(desc.binary):
            out.append(desc.key)
    return out


def ensure_project(directory: Path, client: str = "claude") -> list[tuple[str, str]]:
    """Make the directory a myrmidon project, quietly. Returns what actually changed —
    not the "not written" row a user-config-only client prints on every run."""
    from myrmidon.server.install import init_project

    result = init_project(directory, (client,))
    return [(path, what) for path, what in result.files
            if what not in ("unchanged", "already installed")
            and not what.startswith("not written")]


def _user_scope_wired(desc: ClientDescriptor) -> bool:
    from myrmidon.server.clients import find_client_entries

    return any(e["client"] == desc.key and e["scope"] == "user"
               for e in find_client_entries())


def launch(directory: Path | None = None, *, model: str = "", continue_: bool = False,
           resume: str = "", prompt: str = "", client: str = "",
           exec_fn=None) -> list[str]:
    """Ensure the project, then become the client. Returns the argv when ``exec_fn``
    does not replace the process (tests).

    ``client`` empty means: resolve it (``resolve_client``). A resolved client whose
    binary is missing is an error naming the alternatives on this machine, never a
    silent fallback to another — which model stack runs is the user's decision."""
    key, source = resolve_client(client)
    desc = registry.get(key)
    if desc.nested_env and os.environ.get(desc.nested_env):
        raise LaunchError(NESTED_HINT.format(display=desc.display))
    if not desc.launchable:
        raise LaunchError(f"{desc.display} is not started by myrmidon: the project is wired "
                          f"by `myrmidon init-project --client {key}`; open it there yourself.")
    try:
        binary = client_binary(desc)
    except LaunchError as exc:
        others = [k for k in wired_here() if k != key]
        where = "" if source in ("--client", "default") else f" (client = {key!r} in {source})"
        alt = (f" Wired here instead: {', '.join(others)} — `myrmidon --client <name>`."
               if others else "")
        raise LaunchError(f"{exc}{where}{alt}") from None
    cwd = Path(directory or Path.cwd()).resolve()
    argv = launch_argv(binary, model=model, continue_=continue_, resume=resume,
                       prompt=prompt, client=desc)
    if desc.project_config is None and not _user_scope_wired(desc):
        # nothing in the project can name the server for this client: a session started
        # now would simply have no myrmidon tools, and nothing would say so
        raise LaunchError(f"{desc.display} has no myrmidon extension in its user config — "
                          f"run `myrmidon mcp install --client {key}` first.")
    changed = ensure_project(cwd, key)
    if exec_fn is None:
        exec_fn = _exec
    exec_fn(binary, argv, cwd, changed)
    return argv


def destination() -> str:
    """Where studies will land — the first thing a new user looks for and the last place
    they think to look, because it is not in the directory they are standing in. On a
    cluster that is deliberate: computation belongs on a filesystem
    the compute nodes can see, and the project holds only what the client reads."""
    from myrmidon.config import MyrmidonConfig

    try:
        root = MyrmidonConfig.load().studies_root
    except Exception:  # no config yet, or one that will not parse: `init` says so
        return "  workspaces  (not configured — run `myrmidon init`)"
    return f"  workspaces  {root}  (studies_root in the machine config)"


def status_lines(cwd: Path) -> list[str]:
    """What this project's study is doing — for the person, in their terminal.

    Reopening a session shows an empty screen: the session-start hook's output goes into
    the model's context, not onto the terminal, and the model says nothing until it is
    spoken to. So the launcher says it. Each pending job is polled, because
    the answer that matters after a break is whether the thing finished — the job store
    alone would still say RUNNING.

    Best effort: a status line is not worth failing a launch over.
    """
    try:
        from myrmidon.basic_tools import active_workspace_path
        from myrmidon.config import MyrmidonConfig
        from myrmidon.core.tool import ToolContext
        from myrmidon.core.workspace import Workspace
        from myrmidon.engines.gromacs.results import pending_lines
        from myrmidon.server.mcp_server import build_backend

        config = MyrmidonConfig.load()
        path = active_workspace_path(Path(config.studies_root), cwd)
        if path is None:
            return []
        ws = Workspace.open(path)
        ctx = ToolContext(config=config, workspace=ws, backend=build_backend(config))
        jobs = pending_lines(ctx)
    except Exception:
        return []
    lines = [f"  study       {ws.meta.get('name', ws.root.name)}  {ws.root}"]
    lines += [f"  job         {line}" for line in jobs]
    if not jobs:
        lines.append("  jobs        none running or waiting to be collected")
    return lines


def _exec(binary: str, argv: list[str], cwd: Path, changed: list[tuple[str, str]]) -> None:
    import typer

    for path, what in changed:
        try:
            shown = Path(path).relative_to(cwd)
        except ValueError:
            shown = Path(path)
        typer.echo(f"  {what:10s} {shown}")
    typer.echo(destination())
    for line in status_lines(cwd):
        typer.echo(line)
    os.chdir(cwd)
    os.execv(binary, argv)
