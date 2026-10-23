"""``myrmidon mcp install`` / ``mcp show``: wire an agent client to this install's MCP
server.

Every client speaks the same stdio MCP; the entry is always
``{command: <absolute path of this environment's myrmidon>, args: ["serve"]}`` and only
the file it goes into differs — which file, and in what format, is the client's
``ClientDescriptor`` (``client_registry.py``). Claude Code: ``claude mcp add --scope user``
when the ``claude`` CLI is available (its own config handling), else the config file.
Files are merged — other servers stay — never clobbered. A format without a writer here
(Codex's TOML) is printed by ``show`` for the user to paste, the escape hatch for any
client whose format changes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from myrmidon.server import client_registry as registry
from myrmidon.server.client_registry import REGISTRY, ClientDescriptor, tilde
from myrmidon.server.clients import SERVER_NAME

CLIENTS = registry.keys()
MARK_BEGIN = "<!-- myrmidon:begin -->"
MARK_END = "<!-- myrmidon:end -->"


def _home() -> Path:
    return Path(os.environ.get("HOME") or Path.home())


def server_command() -> tuple[str, list[str]]:
    """The absolute command that starts this install's server, independent of PATH."""
    bindir = Path(sys.executable).resolve().parent
    binary = bindir / "myrmidon"
    if binary.exists():
        return str(binary), ["serve"]
    return str(Path(sys.executable).resolve()), ["-m", "myrmidon.cli", "serve"]


def server_entry() -> dict:
    command, args = server_command()
    return {"command": command, "args": args}


def goose_ext_entry(name: str) -> dict:
    """A goose stdio extension: ``cmd`` (not ``command``), ``name`` equal to the map key,
    ``timeout`` at goose's own default. ``enabled`` is what makes it run at once — goose
    has no approval step for an extension in its user config."""
    command, args = server_command()
    return {"name": name, "type": "stdio", "enabled": True, "cmd": command, "args": args,
            "timeout": 300}


def client_entry(desc: ClientDescriptor, name: str = SERVER_NAME,
                 scope: str = "project") -> dict:
    """The server entry as this client's config wants it: the common command plus
    whatever keys its format requires (``entry_extra``; ``user_entry_extra`` at user
    scope, the user's own file)."""
    if desc.config_format == "goose-yaml":
        return goose_ext_entry(name)
    extra = dict(desc.entry_extra)
    if scope == "user":
        extra.update(desc.user_entry_extra)
    return {**extra, **server_entry()}


def config_path(desc: ClientDescriptor, scope: str, cwd: Path,
                home: Path | None = None) -> Path | None:
    """Where this client keeps the server entry at ``scope``; None when it has no such
    scope (Codex and Goose read the user config only)."""
    fn = desc.user_config if scope == "user" else desc.project_config
    if fn is None:
        return None
    return fn(Path(home) if home else _home()) if scope == "user" else fn(cwd)


def merge_json_servers(path: Path, name: str, entry: dict) -> bool:
    """Add/update ``mcpServers[name]`` in a JSON config, keeping everything else — the
    other servers, and every key of this entry that is not ours. Myrmidon owns the command
    and its arguments; a key the user added by hand (Qwen Code's ``trust``, an ``env``)
    stays: `myrmidon client` re-runs this on every launch, and replacing the whole entry
    would silently undo a user's edit each time. Returns True when the file changed."""
    data: dict = {}
    if path.exists():
        text = path.read_text()
        if text.strip():
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError(f"{path} is not a JSON object")
    servers = data.setdefault("mcpServers", {})
    current = servers.get(name)
    merged = {**current, **entry} if isinstance(current, dict) else dict(entry)
    if current == merged:
        return False
    servers[name] = merged
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return True


def _has_comments(text: str) -> bool:
    return any(ln.lstrip().startswith("#") or " #" in ln for ln in text.splitlines())


def merge_goose_extension(path: Path, name: str, entry: dict) -> tuple[bool, list[str]]:
    """Add/replace ``extensions[name]`` in goose's YAML config, keeping the other
    extensions. Returns (changed, notes).

    PyYAML drops comments, and a hand-written goose config is full of them. So a file
    that carries comments is never rewritten: the entry is returned in ``notes`` for the
    user to paste (Codex's print-and-paste precedent), and ``changed`` is False. A file
    that already holds this exact entry is "unchanged" whatever it looks like."""
    import yaml

    data: dict = {}
    text = path.read_text() if path.exists() else ""
    if text.strip():
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a YAML mapping")
    extensions = data.setdefault("extensions", {}) or {}
    data["extensions"] = extensions
    if extensions.get(name) == entry:
        return False, []
    if _has_comments(text):
        snippet = yaml.safe_dump({"extensions": {name: entry}}, sort_keys=False).rstrip()
        return False, [f"not written: {path} has comments, which a rewrite would lose. "
                       f"Merge this yourself:\n{snippet}"]
    extensions[name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return True, []


#: config format -> writer(path, name, entry) -> changed, or (changed, notes). A format
#: absent here has no writer: ``install`` refuses and names ``show``, ``init-project``
#: prints the paste line.
WRITERS = {"mcp-json": merge_json_servers, "goose-yaml": merge_goose_extension}


def _write(desc: ClientDescriptor, path: Path, name: str, entry: dict) -> tuple[bool, list[str]]:
    out = WRITERS[desc.config_format](path, name, entry)
    return out if isinstance(out, tuple) else (out, [])


@dataclass
class InstallResult:
    client: str
    scope: str
    file: str
    command: str
    args: list[str] = field(default_factory=list)
    changed: bool = True
    via: str = "file"  # "file" | "claude-cli" | "manual" (not written; notes say what to paste)
    notes: list[str] = field(default_factory=list)


def _claude_cli(name: str, entry: dict, scope: str, cwd: Path) -> InstallResult | None:
    """Register through the claude CLI when it exists; None when it does not."""
    claude = shutil.which("claude")
    if not claude:
        return None
    # `claude mcp add` refuses an existing name: remove first (ignore a missing one)
    subprocess.run([claude, "mcp", "remove", "--scope", scope, name],
                   capture_output=True, text=True, cwd=str(cwd))
    proc = subprocess.run(
        [claude, "mcp", "add", "--scope", scope, name, "--", entry["command"], *entry["args"]],
        capture_output=True, text=True, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"claude mcp add failed: {(proc.stderr or proc.stdout).strip()[-300:]}")
    file = config_path(REGISTRY["claude"], scope, cwd)
    return InstallResult(client="claude", scope=scope, file=str(file), command=entry["command"],
                         args=entry["args"], via="claude-cli",
                         notes=[(proc.stdout or "").strip()] if proc.stdout.strip() else [])


def install(client: str, scope: str = "user", directory: Path | None = None,
            name: str = SERVER_NAME, use_cli: bool = True) -> InstallResult:
    desc = registry.get(client)
    if scope not in ("user", "project"):
        raise ValueError("scope must be 'user' or 'project'")
    cwd = Path(directory or Path.cwd()).resolve()
    entry = client_entry(desc, name=name, scope=scope)
    if desc.config_format not in WRITERS:
        raise NotImplementedError(
            f"writing {client} configs is not built yet — `myrmidon mcp show --client {client}` "
            "prints the entry to paste")
    if client == "claude" and use_cli:
        result = _claude_cli(name, entry, scope, cwd)
        if result is not None:
            return result
    path = config_path(desc, scope, cwd)
    if path is None:
        raise ValueError(f"{desc.display} has no {scope}-scoped config; use --scope "
                         f"{'user' if scope == 'project' else 'project'}")
    changed, notes = _write(desc, path, name, entry)
    command, args = server_command()
    return InstallResult(client=client, scope=scope, file=str(path), command=command,
                         args=args, changed=changed, notes=notes,
                         via="manual" if notes and not changed else "file")


def _where(desc: ClientDescriptor) -> str:
    """"~/.claude.json (user) or ./.mcp.json (project)" — the files, for a person."""
    home = _home()
    parts = []
    if desc.user_config:
        parts.append(f"{tilde(desc.user_config(home), home)} (user)")
    if desc.project_config:
        parts.append(f"./{desc.project_config(Path('.')).as_posix()} (project)")
    return " or ".join(parts)


def show(client: str, name: str = SERVER_NAME) -> str:
    desc = registry.get(client)
    entry = client_entry(desc, name=name)
    if desc.config_format == "goose-yaml":
        import yaml

        return f"# goose: merge into {_where(desc).removesuffix(' (user)')}\n" + yaml.safe_dump(
            {"extensions": {name: entry}}, sort_keys=False).rstrip()
    if desc.config_format == "mcp-json":
        return f"# {client}: merge into {_where(desc)}\n" + json.dumps(
            {"mcpServers": {name: entry}}, indent=2)
    if desc.config_format == "codex-toml":
        args = ", ".join(json.dumps(a) for a in entry["args"])
        return (f"# codex: append to {_where(desc).removesuffix(' (user)')}\n"
                f"[mcp_servers.{name}]\n"
                f"command = {json.dumps(entry['command'])}\nargs = [{args}]")
    raise NotImplementedError(f"no snippet for {desc.config_format!r}")


# -- init-project: project-scoped server entry + the guidance block ---------------------------

#: The client's own frame. The rules themselves are shared with the native loop, which
#: states a different one — see `agent/prompts.LOOP_PREAMBLE`.
CLIENT_PREAMBLE = (
    "This project uses the myrmidon MCP server, and a person is at the keyboard: put the "
    "researcher's decisions (rule 7) to them with their numbers, and acknowledge such a "
    "finding only on their yes. `myrmidon doctor` says what is wired; `myrmidon watch` "
    "follows a run live and `myrmidon report` reads a study back.\n")


def guidance_text(client: bool = False) -> str:
    """The session rules both drivers are given.

    ``client=True`` adds the sentence that is true only when an MCP client is driving.
    One file: two hand-written copies would drift apart unnoticed.
    """
    rules = (Path(__file__).parent / "guidance.md").read_text()
    if not client:
        return rules
    head, _, rest = rules.partition("\n")
    return f"{head}\n\n{CLIENT_PREAMBLE}{rest.lstrip()}"


def upsert_block(path: Path, block: str, prefix: str = "") -> str:
    """Write ``block`` between the myrmidon markers in ``path``: replaced in place when the
    markers exist, appended otherwise, the file created (with ``prefix``) when missing.
    Returns "created" | "updated" | "unchanged" | "appended"."""
    wrapped = f"{MARK_BEGIN}\n{block.rstrip()}\n{MARK_END}\n"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prefix + wrapped)
        return "created"
    text = path.read_text()
    if MARK_BEGIN in text and MARK_END in text:
        a = text.index(MARK_BEGIN)
        b = text.index(MARK_END) + len(MARK_END)
        new = text[:a] + wrapped.rstrip("\n") + text[b:]
        if new == text:
            return "unchanged"
        path.write_text(new)
        return "updated"
    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    path.write_text(text + sep + wrapped)
    return "appended"


def merge_claude_settings(path: Path, myrmidon_bin: str) -> str:
    """Add what a myrmidon project needs to a Claude Code settings file, keeping
    everything the user has: the status line, the session-start hook, and the permission
    that lets this server's tools run without a prompt.

    Returns "created" | "updated" | "unchanged". Never removes a key, never replaces a
    status line or a hook that is not ours: a project may already have both.

    The hook is what makes a second session know about the first. Its stdout is put in
    the model's context at the start of every session, and `study open` is the same
    look the agent's own `study_open` performs — one code path, one set of
    sentences. It ends in `|| true` because a non-zero hook exit is not injected, and
    there being no active study is a perfectly good answer.

    The permission is a decision: every myrmidon tool runs without Claude Code asking,
    because the harness has its own gates (a core-hour estimate over the threshold, a
    finding that needs a human) and two confirmations for one submission teach the user
    to dismiss both.
    """
    data: dict = {}
    if path.exists() and path.read_text().strip():
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a JSON object")
    before = json.dumps(data, sort_keys=True)

    status = data.get("statusLine")
    if not status or "myrmidon" in json.dumps(status):
        data["statusLine"] = {"type": "command",
                              "command": f"{myrmidon_bin} statusline",
                              "refreshInterval": 30}

    hook_cmd = f"{myrmidon_bin} study open || true"
    groups = data.setdefault("hooks", {}).setdefault("SessionStart", [])
    # `workspace open` is the same hook under the `study` group's alias: ours, so it is
    # replaced rather than left beside this one to print the same status twice
    ours_before = f"{myrmidon_bin} workspace open || true"
    for group in groups:
        group["hooks"] = [h for h in group.get("hooks", []) if h.get("command") != ours_before]
    groups[:] = [g for g in groups if g.get("hooks")]
    planted = any(h.get("command") == hook_cmd
                  for group in groups for h in group.get("hooks", []))
    if not planted:
        groups.append({"hooks": [{"type": "command", "command": hook_cmd}]})

    allow = data.setdefault("permissions", {}).setdefault("allow", [])
    if f"mcp__{SERVER_NAME}" not in allow:
        allow.append(f"mcp__{SERVER_NAME}")

    if json.dumps(data, sort_keys=True) == before:
        return "unchanged"
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return "updated" if existed else "created"


def merge_local_settings(path: Path, name: str = SERVER_NAME) -> str:
    """Enable this project's `.mcp.json` server without the client asking on first run.
    Local settings are the user's own file (git-ignored by convention), so this only
    ever appends a name."""
    data: dict = {}
    if path.exists() and path.read_text().strip():
        data = json.loads(path.read_text())
    enabled = data.setdefault("enabledMcpjsonServers", [])
    if name in enabled:
        return "unchanged"
    enabled.append(name)
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return "updated" if existed else "created"


def skills_dir() -> Path:
    """The skills this install ships (packaged with the module, like the input sets)."""
    return Path(__file__).resolve().parent.parent / "knowledge" / "skills"


def install_skills(directory: Path, into: str = ".claude/skills") -> list[tuple[str, str]]:
    """Copy each shipped skill into the project's `<into>/<name>/` — `.claude/skills/` for
    Claude Code, `.qwen/skills/` for Qwen Code; both read the same `SKILL.md` frontmatter.

    Copied, not linked: a project is rebuilt from a release snapshot into its own
    environment, and a link into the environment that installed it dies with that
    environment. A copy is also a file the user can read, edit and commit — which is
    the point of a skill.

    Only files this install ships are touched; a skill the user wrote is left alone, and
    a file whose bytes already match is reported "unchanged" rather than rewritten.
    """
    out: list[tuple[str, str]] = []
    source = skills_dir()
    if not source.is_dir():
        return out
    for skill in sorted(source.iterdir()):
        if not skill.is_dir():
            continue
        for item in sorted(skill.rglob("*")):
            if not item.is_file():
                continue
            target = directory / into / skill.name / item.relative_to(skill)
            data = item.read_bytes()
            if target.exists() and target.read_bytes() == data:
                out.append((str(target), "unchanged"))
                continue
            what = "updated" if target.exists() else "created"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            out.append((str(target), what))
    return out


def set_project_key(directory: Path, key: str, value: str) -> str:
    """Write one flat ``key = "value"`` into the project's `.myrmidon.toml`, keeping
    whatever else is in it: the file is edited line by line, so a table would be a trap.
    The machine config supplies everything a project does not restate."""
    path = directory / ".myrmidon.toml"
    header = ("# this project's own settings; everything else comes from "
              "~/.config/myrmidon/config.toml")
    lines = path.read_text().splitlines() if path.is_file() else []
    kept = [ln for ln in lines if not ln.strip().startswith(key) and ln.strip() != header]
    # the header once, at the top: this runs on every launch, and a header appended each
    # time would grow the file by a line per launch
    kept.insert(0, header)
    kept.insert(1, f'{key} = "{value}"')
    existed = path.is_file()
    path.write_text("\n".join(kept).rstrip() + "\n")
    return "updated" if existed else "created"


def set_project_studies_root(directory: Path, root: str) -> str:
    """A relative value is relative to the project, so `workspaces` puts the studies
    beside the inputs they came from — which is where a person looks for them."""
    return set_project_key(directory, "studies_root", root)


def set_project_client(directory: Path, client: str) -> str:
    """The client `myrmidon client` starts in this project — recorded when `init-project`
    named exactly one launchable client, so the project remembers what it was wired for."""
    return set_project_key(directory, "client", client)


@dataclass
class ProjectInitResult:
    directory: str
    files: list[tuple[str, str]] = field(default_factory=list)  # (path, what happened)


def claude_project_extras(cwd: Path, name: str, server: bool) -> list[tuple[str, str]]:
    """What makes the directory an agent's workplace rather than a config, for the one
    client that can be told all of it through files: the look at the top of every
    session, the status bar, the gates being ours, and the recipes the agent can follow.
    Tier 1 in `client_registry.py`."""
    out: list[tuple[str, str]] = []
    settings = cwd / ".claude" / "settings.json"
    out.append((str(settings), merge_claude_settings(settings, server_command()[0])))
    if server:
        local = cwd / ".claude" / "settings.local.json"
        out.append((str(local), merge_local_settings(local, name)))
    out.extend(install_skills(cwd))
    return out


def qwen_project_extras(cwd: Path, name: str, server: bool) -> list[tuple[str, str]]:
    """Qwen Code reads `.qwen/skills/<name>/SKILL.md` as Claude Code reads `.claude/skills/`,
    so the shipped recipes reach it the same way. Its hooks and status line are not wired:
    the guidance's `study_open` rule covers the session start."""
    return install_skills(cwd, ".qwen/skills")


def _not_written(desc: ClientDescriptor) -> tuple[str, str]:
    """The `init-project` row for a client whose server entry lives in the user config."""
    home = _home()
    fix = (f"run `myrmidon mcp install --client {desc.key}`" if desc.config_format in WRITERS
           else f"paste `myrmidon mcp show --client {desc.key}`")
    return (tilde(desc.user_config(home), home),
            f"not written — {desc.key} reads the user config: {fix}")


def default_studies_root(project: Path) -> Path:
    """Where a new project's studies go when nobody says: its own directory under the
    machine's root.

    Without a root of its own, every project's studies land in one machine-wide pile, and a
    second project asking for the name its task suggests is handed a stranger's study. The
    obvious alternative — `workspaces` beside the code — is wrong on the machines that
    matter: the machine config points its root at the filesystem meant for computation, and
    trajectories do not belong beside a repository. So a project gets a namespace *inside*
    that root, and the filesystem decision stays where it was made.

    Written only when a directory is made a project for the first time: re-running on an
    existing project must not move where its studies have been going.
    """
    from myrmidon.config import MyrmidonConfig

    try:
        # the machine's own root, not whatever project the command was typed in: the
        # loader walks up from the working directory, so initialising a project from
        # inside another one would nest its studies in that one's root
        machine = MyrmidonConfig.candidates()[-1]
        machine_root = Path(MyrmidonConfig.load(machine).studies_root)
    except Exception:  # an unreadable config is not a reason to refuse to init
        machine_root = Path("workspaces")
    if not machine_root.is_absolute():
        return Path(machine_root.name)
    return machine_root / project.name


def init_project(directory: Path | None = None, clients: tuple[str, ...] = ("claude",),
                 name: str = SERVER_NAME, server: bool = True,
                 studies_root: str | None = None) -> ProjectInitResult:
    """Make a research repo a myrmidon project for the given clients: the project-scoped MCP
    entry in the file each client reads (a client that reads only the user config gets a
    line saying so instead) and the guidance block in the file each client takes its
    instructions from. Re-running updates the block in place and leaves everything else
    alone. Two clients that read the same file (Claude Code and Qwen Code both read
    `.mcp.json`) get it written once.

    ``server=False`` installs only the guidance block — the case where the server is already
    registered at user scope (`mcp install`) and a project-scoped copy would duplicate it."""
    cwd = Path(directory or Path.cwd()).resolve()
    result = ProjectInitResult(directory=str(cwd))
    from myrmidon.basic_tools import is_project

    first_time = not is_project(cwd)
    if not studies_root and first_time:
        studies_root = str(default_studies_root(cwd))
    if studies_root:
        result.files.append((str(cwd / ".myrmidon.toml"),
                             set_project_studies_root(cwd, studies_root)))
    descs = [registry.get(client) for client in clients]
    launchables = [d for d in descs if d.launchable]
    if len(launchables) == 1 and launchables[0].key != "claude":
        # the default is claude; a project wired for one other client says so, and bare
        # `myrmidon client` there starts that client (launch.resolve_client)
        result.files.append((str(cwd / ".myrmidon.toml"),
                             set_project_client(cwd, launchables[0].key)))
    block = guidance_text()
    done: set[Path] = set()
    for desc in descs:
        if not server:
            pass
        elif desc.project_config is None:
            result.files.append(_not_written(desc))
        else:
            p = desc.project_config(cwd)
            if p not in done:
                done.add(p)
                changed, notes = _write(desc, p, name, client_entry(desc, name))
                result.files.append((str(p), "written" if changed else "unchanged"))
                result.files.extend((str(p), n) for n in notes)
        gpath = cwd / desc.guidance_file
        if gpath not in done:
            done.add(gpath)
            result.files.append((str(gpath), upsert_block(gpath, block, desc.guidance_prefix)))
        if desc.project_extras is not None:
            result.files.extend(desc.project_extras(cwd, name, server))
    return result
