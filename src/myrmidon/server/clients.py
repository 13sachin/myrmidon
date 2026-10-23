"""Which agent clients on this machine point at a myrmidon MCP server.

Read-only inspection for ``doctor``: every client in ``client_registry.REGISTRY``, at
every scope it has (user config, project config), read by its config format. For every
entry found: the command it launches and whether that command resolves — the common
breakage is an environment that moved. ``mcp install`` writes these same files.
"""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from pathlib import Path

from myrmidon.server.client_registry import REGISTRY

SERVER_NAME = "myrmidon"


def _resolves(command: str) -> bool:
    if not command:
        return False
    p = Path(command).expanduser()
    if p.is_absolute() or "/" in command:
        return p.exists()
    return shutil.which(command) is not None


def _json_servers(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("mcpServers", {}) if isinstance(data, dict) else {}


def _toml_servers(path: Path) -> dict:
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data.get("mcp_servers") or {}


def _yaml_extensions(path: Path) -> dict:
    """goose spells the command ``cmd``; ``_entry`` reads ``command``."""
    import yaml

    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    exts = (data.get("extensions") or {}) if isinstance(data, dict) else {}
    return {k: {**v, "command": v.get("cmd", "")} for k, v in exts.items()
            if isinstance(v, dict)}


#: config format -> reader(path) -> {server name: {command, args, ...}}
READERS = {"mcp-json": _json_servers, "codex-toml": _toml_servers,
           "goose-yaml": _yaml_extensions}


def _entry(client: str, scope: str, path: Path, server: dict) -> dict:
    command = str(server.get("command", ""))
    args = [str(a) for a in server.get("args", [])]
    return {"client": client, "scope": scope, "file": str(path), "command": command,
            "args": args, "command_found": _resolves(command)}


def find_client_entries(cwd: Path | None = None, home: Path | None = None,
                        name: str = SERVER_NAME) -> list[dict]:
    cwd = Path(cwd or Path.cwd())
    home = Path(home or os.environ.get("HOME") or Path.home())
    found: list[dict] = []
    for key, desc in REGISTRY.items():
        reader = READERS.get(desc.config_format)
        if reader is None:
            continue
        for scope, fn, base in (("user", desc.user_config, home),
                                ("project", desc.project_config, cwd)):
            if fn is None:
                continue
            path = fn(base)
            if path.is_file():
                server = reader(path).get(name)
                if isinstance(server, dict):
                    found.append(_entry(key, scope, path, server))
    return found


def describe(entries: list[dict]) -> list[str]:
    if not entries:
        return [f"no agent client has a myrmidon MCP entry ({'/'.join(REGISTRY)}) — "
                "run `myrmidon mcp install --client <name>`"]
    # one file, one row: `.mcp.json` is read by Claude Code and Qwen Code alike
    rows: dict[tuple[str, str], list[dict]] = {}
    for e in entries:
        rows.setdefault((e["file"], e["scope"]), []).append(e)
    out = []
    for (file, scope), group in rows.items():
        e = group[0]
        clients = ", ".join(g["client"] for g in group)
        cmd = " ".join([e["command"], *e["args"]])
        mark = "✓" if e["command_found"] else "✗"
        note = "" if e["command_found"] else "  (command not found — environment moved?)"
        out.append(f"{mark} {clients} ({scope}): {cmd}{note}  [{file}]")
    return out
