"""Every agent client myrmidon can wire or start, described once.

The server, the gate inputset and the guidance are the same for every client; what differs
is where its config lives, what format the entry takes, which file it reads its
instructions from, and how it spells its command-line flags. A ``ClientDescriptor`` holds
exactly that, and ``install``, ``init-project``, ``doctor`` and the launcher read it
rather than branching on a name. Adding a client is an entry here — plus a config writer
in ``install.py`` when its format is new.

Two tiers, by what the client can be told through files. Tier 1 (Claude Code) also gets
the status line, the session-start hook, the permission and the skills, through
``project_extras``. Tier 2 gets the server and the guidance; the guidance's "call
`study_open` first" rule stands in for the hook. Qwen Code sits between: it reads
skills from `.qwen/skills/` in the same format, so it gets those too.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ConfigFormat = Literal["mcp-json", "codex-toml", "goose-yaml"]

#: A ``flag_map`` value meaning "passed as a positional argument".
POSITIONAL: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClientDescriptor:
    key: str                      # the `--client` value
    display: str                  # how a person names it
    config_format: ConfigFormat
    guidance_file: str            # the file the client reads its instructions from
    user_config: Callable[[Path], Path] | None = None     # home -> the user-scope config
    project_config: Callable[[Path], Path] | None = None  # cwd -> the project-scope config
    entry_extra: Mapping[str, object] = field(default_factory=dict)  # keys the entry needs
    user_entry_extra: Mapping[str, object] = field(default_factory=dict)  # user scope only
    guidance_prefix: str = ""     # written once when the guidance file is created
    binary: str | None = None     # None: the launcher cannot start it
    bin_env: str | None = None    # env var naming the binary, ahead of PATH
    install_hint: str = ""        # the LaunchError sentence when the binary is missing
    launch_subcommand: tuple[str, ...] = ()
    #: launch option -> the client's spelling; ``POSITIONAL`` for a bare argument; an
    #: absent option is unsupported and the launcher says so (``unsupported`` names the
    #: client's own way of doing it, when it has one).
    flag_map: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    unsupported: Mapping[str, str] = field(default_factory=dict)
    nested_env: str | None = None  # set in the shells this client spawns
    next_hint: str = ""            # what `init-project` prints last, true for this client
    #: tier-1 extras: (cwd, server_name, server: bool) -> [(path, what happened)]
    project_extras: Callable[[Path, str, bool], list[tuple[str, str]]] | None = None

    @property
    def launchable(self) -> bool:
        return self.binary is not None


def _claude_user_config(home: Path) -> Path:
    """Claude Code keeps its user config in $CLAUDE_CONFIG_DIR when that is set."""
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    return (Path(cfg) if cfg else home) / ".claude.json"


def _claude_project_extras(cwd: Path, name: str, server: bool) -> list[tuple[str, str]]:
    from myrmidon.server.install import claude_project_extras

    return claude_project_extras(cwd, name, server)


def _qwen_project_extras(cwd: Path, name: str, server: bool) -> list[tuple[str, str]]:
    from myrmidon.server.install import qwen_project_extras

    return qwen_project_extras(cwd, name, server)


CLAUDE_INSTALL_HINT = (
    "Claude Code is not installed, or not on PATH. Install it "
    "(https://claude.com/claude-code), or point MYRMIDON_CLAUDE_BIN at the binary."
)

REGISTRY: dict[str, ClientDescriptor] = {}


def register(desc: ClientDescriptor) -> ClientDescriptor:
    REGISTRY[desc.key] = desc
    return desc


register(ClientDescriptor(
    key="claude", display="Claude Code", config_format="mcp-json",
    guidance_file="CLAUDE.md",
    user_config=_claude_user_config,
    project_config=lambda cwd: cwd / ".mcp.json",
    binary="claude", bin_env="MYRMIDON_CLAUDE_BIN", install_hint=CLAUDE_INSTALL_HINT,
    flag_map={"model": ("--model",), "continue": ("--continue",), "resume": ("--resume",),
              "prompt": POSITIONAL},
    nested_env="CLAUDECODE",
    next_hint=("Next: `myrmidon client` here. Start it interactively the first time and accept "
               "the client's trust prompt once — until you do, it ignores this project's "
               "permissions, and `claude -p` cannot accept it for you."),
    project_extras=_claude_project_extras,
))

register(ClientDescriptor(
    key="cursor", display="Cursor", config_format="mcp-json",
    guidance_file=".cursor/rules/myrmidon.mdc",
    guidance_prefix="---\ndescription: myrmidon simulation harness\nalwaysApply: true\n---\n",
    user_config=lambda home: home / ".cursor" / "mcp.json",
    project_config=lambda cwd: cwd / ".cursor" / "mcp.json",
    # Cursor's docs list ``type`` as required for stdio servers; the others have no such key
    entry_extra={"type": "stdio"},
    next_hint="Next: open this directory in Cursor; the server and the rule are in place.",
))

register(ClientDescriptor(
    key="codex", display="Codex", config_format="codex-toml",
    guidance_file="AGENTS.md",
    user_config=lambda home: home / ".codex" / "config.toml",
    next_hint=("Next: paste `myrmidon mcp show --client codex` into ~/.codex/config.toml "
               "(codex has no project-scoped server config), then `codex` here."),
))

register(ClientDescriptor(
    key="qwen", display="Qwen Code", config_format="mcp-json",
    guidance_file="QWEN.md",
    user_config=lambda home: home / ".qwen" / "settings.json",
    # the same project file Claude Code reads (Qwen's scope "project"); a project wired
    # for both gets one entry, not two. Its own `.qwen/settings.json` is scope
    # "workspace" and is ignored until the folder is trusted when folder trust is on.
    project_config=lambda cwd: cwd / ".mcp.json",
    # `trust` skips Qwen's per-call tool confirmation, the counterpart of Claude Code's
    # `permissions.allow`: the harness has its own gates. It goes in the user's own file,
    # never in the shared `.mcp.json`, which Claude Code also reads and does not know the key.
    user_entry_extra={"trust": True},
    binary="qwen", bin_env="MYRMIDON_QWEN_BIN",
    install_hint=("Qwen Code is not installed, or not on PATH. Install it "
                  "(https://github.com/QwenLM/qwen-code), or point MYRMIDON_QWEN_BIN at "
                  "the binary."),
    # `-i` runs the prompt and stays interactive, which is what `myrmidon client "…"`
    # means; the bare positional prompt would answer and exit
    flag_map={"model": ("--model",), "continue": ("--continue",), "resume": ("--resume",),
              "prompt": ("-i",)},
    nested_env="QWEN_CODE",
    next_hint=("Next: `myrmidon --client qwen` here. The first time, approve the myrmidon "
               "server when Qwen Code asks (or run `qwen mcp approve myrmidon` in this "
               "directory) — until you do the server is pending and its tools are absent. "
               "Myrmidon does not approve it for you."),
    project_extras=_qwen_project_extras,
))

register(ClientDescriptor(
    key="goose", display="Goose", config_format="goose-yaml",
    # goose reads AGENTS.md (then .goosehints) at every directory level, while its
    # Developer extension is enabled; Codex reads the same file, and the block is
    # written once for both
    guidance_file="AGENTS.md",
    user_config=lambda home: home / ".config" / "goose" / "config.yaml",
    binary="goose", bin_env="MYRMIDON_GOOSE_BIN",
    install_hint=("Goose is not installed, or not on PATH. Install it "
                  "(https://block.github.io/goose), or point MYRMIDON_GOOSE_BIN at the "
                  "binary."),
    launch_subcommand=("session",),
    # `-r` alone resumes the most recent session; `-r --name` resumes one by its *name*
    flag_map={"model": ("--model",), "continue": ("-r",), "resume": ("-r", "--name")},
    unsupported={"prompt": '`goose run -t "…" -s` (run the text, then stay in the session)'},
    next_hint=("Next: `myrmidon mcp install --client goose` once (goose reads only its user "
               "config), then `myrmidon --client goose` here. Nothing to approve: an enabled "
               "extension runs at once, and in goose's default `auto` mode the harness's "
               "gates are the only gates. AGENTS.md is read while goose's Developer "
               "extension is on."),
))


def get(key: str) -> ClientDescriptor:
    if key not in REGISTRY:
        raise ValueError(f"unknown client {key!r}; one of {tuple(REGISTRY)}")
    return REGISTRY[key]


def keys() -> tuple[str, ...]:
    return tuple(REGISTRY)


def launchable() -> tuple[str, ...]:
    return tuple(k for k, d in REGISTRY.items() if d.launchable)


def tilde(path: Path, home: Path) -> str:
    """``~/.claude.json`` rather than the expanded home — for the person, not the OS."""
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)
