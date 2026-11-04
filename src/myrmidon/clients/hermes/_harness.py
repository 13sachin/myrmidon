"""Reaching the harness from inside Hermes.

Both Hermes plugins — the session plugin and the model-provider plugin — have to find the
same myrmidon install the session's tools come from, and run its CLI. That is all this
module does; a copy of it is installed beside each plugin, because Hermes discovers the two
through different roots and neither may import the other.

This runs inside Hermes's own interpreter, not myrmidon's: standard library only, and
nothing of myrmidon's is imported.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SERVER = "myrmidon"


#: written beside each plugin by `myrmidon mcp install` / `init-project`, which rewrite it
#: on every launch. The same command Hermes's own `mcp_servers` entry names, recorded in a
#: form the standard library reads: Hermes's config is YAML, and parsing it here would mean
#: either a YAML library its environment does not carry or one of its private modules.
SERVER_FILE = "server.json"


def server_entry() -> dict | None:
    """The myrmidon server Hermes starts: the same install the session's tools come from,
    so a view and a tool call cannot disagree."""
    try:
        data = json.loads((Path(__file__).parent / SERVER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("command"):
        return None
    args = data.get("args")
    return {"command": str(data["command"]),
            "args": [str(a) for a in args] if isinstance(args, list) else []}


def cli(entry: dict, command: list[str]) -> list[str]:
    """The myrmidon command line in the server's world: the program the server entry
    names, then the command. The entry names no study — the server finds the project from
    its working directory, and so does each command, run where the session is."""
    args = entry["args"]
    at = args.index("serve") if "serve" in args else len(args)
    return [entry["command"], *args[:at], *command]


def run(argv: list[str], cwd: Path, timeout: int = 120,
        env: dict[str, str] | None = None) -> tuple[int, str, str]:
    """``env`` adds to this process's environment; it does not replace it."""
    try:
        out = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True,
                             timeout=timeout, check=False,
                             env={**os.environ, **env} if env else None)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, "", str(exc)
    return out.returncode, out.stdout or "", out.stderr or ""


def session_cwd() -> Path:
    """The session's directory. A slash command runs in the gateway, whose own working
    directory is not the one the researcher started the session in."""
    try:
        from agent.runtime_cwd import resolve_agent_cwd

        return Path(resolve_agent_cwd())
    except Exception:
        return Path.cwd()
