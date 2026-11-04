"""Setting a machine up: the MCP server, doctor, the machine config, a project, and the
agent clients wired to the server."""

from __future__ import annotations

import json

import typer

from myrmidon.cli._app import (
    CLIENT_KEYS,
    CLIENT_NAMES,
    CONFIG_HELP,
    PANEL_SETUP,
    app,
    config_file,
    serve_context,
)
from myrmidon.client_registry import REGISTRY


@app.command(hidden=True)
def serve(
    check: bool = typer.Option(
        False, "--check", help="do not serve: source env_script, discover engines, exit "
        "non-zero naming the first missing engine"),
    workspace: str = typer.Option("", "--study",
                                 help="open this study before serving"),
    mock: bool = typer.Option(False, "--mock", help="serve fake engines and a fake "
                              "scheduler: nothing real runs, job records stay in the study"),
    unattended: bool = typer.Option(False, "--unattended", help="no human sees the calls: "
                                    "the researcher's findings are refused"),
    config: str = typer.Option("", "--config", help=CONFIG_HELP),
) -> None:
    """Run the MCP server on stdio: every tool this machine's engines support. Sources
    config.env_script first, so the client that launches it need not have (register it
    with `myrmidon mcp install`)."""
    if check:
        from myrmidon.core.envscript import EnvScriptStatus
        from myrmidon.server.mcp_server import build_context

        ctx = build_context(config_file(config))
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


@app.command(rich_help_panel=PANEL_SETUP)
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
    return (f"{ctx.config.backend} backend: {shape}; a booking runs within the ceiling of "
            "the mandate the researcher approved")


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
    from myrmidon.client_entries import describe
    from myrmidon.core.envscript import EnvScriptStatus

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


@app.command(hidden=True)
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


def _say(line: str) -> None:
    typer.echo(line)


def run_setup(*, yes: bool = False, client: str = "", backend: str = "", studies_root: str = "",
              env_script: str = "", partition: str = "", account: str = "", endpoint: str = "",
              api_base: str = "", key_env: str = "", path: str = "") -> None:
    """One conversation that sets the machine up: discover what can be discovered, ask
    what cannot, write the files, check them. Every answer has a default, and ``yes`` takes
    them all; flags answer in advance."""
    import shutil
    from pathlib import Path

    from myrmidon.bootstrap import (
        config_path,
        default_account,
        discover_plan,
        init_config,
        render_config,
        scratch_candidates,
    )
    from myrmidon.config import MyrmidonConfig
    from myrmidon.core.envscript import apply_env_script
    from myrmidon.endpoints import (
        EndpointError,
        apply_secrets,
        endpoints_path,
        ensure_secret_line,
        load_endpoints,
        secrets_path,
        write_endpoint,
    )
    from myrmidon.errors import ConfigError
    from myrmidon.server.launch import PREFERRED, wired_here

    def ask(prompt: str, default: str) -> str:
        if yes:
            return default
        return str(typer.prompt(f"  {prompt}", default=default or "", show_default=bool(default))
                   ).strip()

    def confirm(prompt: str, default: bool) -> bool:
        return default if yes else typer.confirm(f"  {prompt}", default=default)

    cfg_path = Path(path).expanduser() if path else config_path()
    existing = None
    if cfg_path.is_file():
        try:
            existing = MyrmidonConfig.load(cfg_path)
        except ConfigError as exc:
            _say(f"the config at {cfg_path} does not read: {exc}")
            raise typer.Exit(2) from None
    _say("Setting myrmidon up on this machine"
         + (f" — {cfg_path} exists; the answers update it." if existing else "."))

    # the scheduler, from the machine
    backend = backend or (existing.backend if existing else
                          ("slurm" if shutil.which("sbatch") else "local"))
    if backend not in ("local", "slurm"):
        _say("backend must be local or slurm")
        raise typer.Exit(2)
    _say("  scheduler  " + ("slurm (sbatch found); partitions and billing read from it"
                            if backend == "slurm" else "none found: jobs run here as processes"))

    # the engines, on PATH and after the script that loads them
    plan = discover_plan(backend=backend)
    engines = lambda pl: "   ".join(  # noqa: E731
        f"{'✓' if ok else '✗'} {key[:-4]}" for key, ok in pl.found.items())
    _say(f"  engines    {engines(plan)}")
    env = env_script or (str(existing.env_script) if existing and existing.env_script else "")
    if not env and (backend == "slurm" or not all(plan.found.values())):
        candidate = Path("~/.config/myrmidon/env_engines.sh").expanduser()
        env = ask("script that loads your engines (module loads, one per line; Enter for none)",
                  str(candidate) if candidate.is_file() else "")
    if env:
        status = apply_env_script(env)
        _say(f"             {status.describe()}")
        plan = discover_plan(backend=backend, env_script=env)
        _say(f"  engines    {engines(plan)}   (after sourcing it)")

    # where studies live
    scratch = scratch_candidates()
    if existing:
        root_default = str(existing.studies_root)
    elif scratch:
        root_default = str(Path(scratch[0]) / "myrmidon" / "studies")
    else:
        root_default = str(Path("~/myrmidon-studies").expanduser())
    if len(scratch) > 1 and not studies_root and not existing:
        _say("  scratch    " + "  ".join(scratch))
    root = studies_root or ask("where studies live (a filesystem the compute nodes see)",
                               root_default)

    # the scheduler's partition and account
    part = acct = ""
    if backend == "slurm":
        part = partition or (existing.slurm.partition if existing else "") \
            or plan.default_partition or ""
        part = ask("partition submissions go to", part)
        acct = account or (existing.slurm.account if existing else "") or default_account()
        acct = ask("account they are billed to", acct)

    # the agent
    found = [k for k in PREFERRED if k in wired_here()]
    chosen = client or (existing.client if existing and existing.client in found else "") \
        or (found[0] if found else "")
    if found:
        _say(f"  agent      found here: {', '.join(found)}")
    if len(found) > 1 and not client:
        chosen = ask("which drives your sessions", chosen)
    if not chosen:
        _say("  agent      none found here (pi, hermes, claude, qwen, goose). Install Pi "
             "(`npm install -g @earendil-works/pi-coding-agent`) and run `myrmidon setup` "
             "again; the tools and `myrmidon approve` work meanwhile.")
        chosen = "pi"
    elif chosen not in found:
        from myrmidon.client_registry import REGISTRY as CLIENTS

        desc = CLIENTS.get(chosen)
        _say(f"  agent      {chosen} is not on this shell's PATH"
             + (f": {desc.install_hint}" if desc and desc.install_hint else ""))

    # a model endpoint, for an agent on an open model
    try:
        eps = load_endpoints()
    except EndpointError as exc:
        _say(f"  model      {exc}")
        eps = {}
    if eps:
        _say(f"  model      endpoints defined: {', '.join(sorted(eps))}  ({endpoints_path()})")
    name = endpoint
    if not name and not yes and (not eps or confirm("add a model endpoint (an OpenAI-compatible "
                                                    "server)?", default=not eps)):
        name = ask("endpoint name (Enter to skip)", "")
    if name:
        base = api_base or ask("api_base (ends in /v1)", eps[name].api_base if name in eps else "")
        var = key_env or ask("the environment variable that holds its key",
                             eps[name].key_env if name in eps else
                             f"{name.upper().replace('-', '_')}_API_KEY")
        try:
            state = write_endpoint(name, base, var)
        except (EndpointError, ValueError) as exc:
            _say(f"  model      not written: {exc}")
            raise typer.Exit(2) from None
        _say(f"  model      {endpoints_path()} ({state}): {name} -> {base}, key in ${var}")
        sstate = ensure_secret_line(var)
        _say(f"             {secrets_path()} ({sstate}, chmod 600): put `export {var}=…` there")
    apply_secrets()
    import os

    from myrmidon.endpoints import cached_models

    for ep_name, ep in sorted(load_endpoints().items()):
        has_key = bool(os.environ.get(ep.key_env))
        models = cached_models(ep) if has_key else []
        _say(f"             {ep_name}: ${ep.key_env} "
             + (f"set ({len(models)} models cached)" if has_key
                else "NOT SET in the secrets file"))

    # the config, written from the answers
    plan.studies_root = root
    plan.env_script = env or None
    plan.partition, plan.account, plan.client = part, acct, chosen
    if existing is not None:
        current, new = cfg_path.read_text(), render_config(plan)
        if current == new:
            _say(f"  config     {cfg_path} unchanged")
        else:
            result = init_config(plan, cfg_path, force=False)
            _say("  config     the answers change it:")
            _say("\n".join("    " + ln for ln in result.diff.splitlines()[:40]))
            if confirm("write it?", default=True):
                init_config(plan, cfg_path, force=True)
                _say(f"  config     wrote {cfg_path}")
            else:
                _say(f"  config     kept {cfg_path}")
    else:
        result = init_config(plan, cfg_path)
        _say(f"  config     wrote {cfg_path}"
             + (f"; created {secrets_path()} (chmod 600)" if result.secrets_created else ""))

    # a client that reads only its user config gets its server entry now; the others get
    # a project's entry the first time `myrmidon` starts there
    if chosen in ("hermes", "goose"):
        from myrmidon.server.install import install

        try:
            res = install(chosen, scope="user")
            _say(f"  {chosen:9s}  {res.file}: "
                 + ("registered" if res.changed else "already there"))
        except (ValueError, NotImplementedError, RuntimeError) as exc:
            _say(f"  {chosen:9s}  not wired: {exc}")

    from myrmidon.server.mcp_server import build_context

    _say("")
    _print_doctor(build_context(MyrmidonConfig.load(cfg_path)))
    _say("")
    _say(f"Next: cd into your project and run `myrmidon` — it starts {chosen} there, wired.")


@app.command(rich_help_panel=PANEL_SETUP)
def setup(
    yes: bool = typer.Option(False, "--yes", "-y", help="take every default without asking"),
    client: str = typer.Option("", "--client", help=f"the agent to start: {CLIENT_KEYS}"),
    backend: str = typer.Option("", "--backend", help="local | slurm (default: slurm when "
                                "sbatch is found)"),
    studies_root: str = typer.Option("", "--studies-root", help="where studies live"),
    env_script: str = typer.Option("", "--env-script", help="shell script that provides the "
                                   "engines"),
    partition: str = typer.Option("", "--partition", help="SLURM partition"),
    account: str = typer.Option("", "--account", help="SLURM account"),
    endpoint: str = typer.Option("", "--endpoint", help="a model endpoint to add, by name"),
    api_base: str = typer.Option("", "--api-base", help="its OpenAI-compatible base URL"),
    key_env: str = typer.Option("", "--key-env", help="the variable that holds its key"),
    path: str = typer.Option("", "--path", help="config file (default: $MYRMIDON_CONFIG or "
                             "~/.config/myrmidon/config.toml)"),
) -> None:
    """Set this machine up, once, in one conversation: the scheduler, the engines and the
    script that loads them, where studies live, the agent that drives your sessions, a model
    endpoint and where its key goes. Discovers what it can, asks what it cannot, writes the
    config, and ends with `doctor`. Run it again any time; it updates what changed."""
    run_setup(yes=yes, client=client, backend=backend, studies_root=studies_root,
              env_script=env_script, partition=partition, account=account, endpoint=endpoint,
              api_base=api_base, key_env=key_env, path=path)


@app.command("init-project", hidden=True)
def init_project_cmd(
    directory: str = typer.Argument("", help="project directory (default: current)"),
    client: str = typer.Option("", "--client",
                               help=f"{CLIENT_KEYS} | all (comma-separated works too); "
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

    typer.echo(destination(Path(result.directory)))
    # what a person does next is the client's business, and differs: Claude Code and Qwen
    # Code gate a fresh project's server until a human approves it once, and a headless
    # session has no way to; goose has no such step. Myrmidon never accepts a
    # trust dialog on the user's behalf: a tool that marks its own output as trusted
    # disables the control that protects against tools like itself.
    for c in dict.fromkeys(clients):
        typer.echo(REGISTRY[c].next_hint)


mcp_app = typer.Typer(help=f"Wire agent clients ({CLIENT_NAMES}) to this server.",
                      no_args_is_help=True)


app.add_typer(mcp_app, name="mcp", hidden=True)


@mcp_app.command("install")
def mcp_install(
    client: str = typer.Option(..., "--client", help=CLIENT_KEYS),
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
    client: str = typer.Option("claude", "--client", help=CLIENT_KEYS),
    name: str = typer.Option("myrmidon", help="server name in the client config"),
) -> None:
    """Print the client config snippet for this install's server (paste it yourself)."""
    from myrmidon.server.install import show

    try:
        typer.echo(show(client, name=name))
    except ValueError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(1) from None
