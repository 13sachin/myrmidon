"""The server owns its environment.

Engines on HPC come from ``module load`` lines in a shell script. Instead of requiring
the MCP client (or the user) to have sourced it before launching ``myrmidon serve``,
the machine config names it as ``env_script`` and myrmidon sources it itself: once in
the server/CLI process (captured through a child bash, applied to ``os.environ`` so
engine discovery and every subprocess see it) and in every sbatch script the SLURM
backend renders. A broken or missing script is reported, never silently ignored.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

APPLIED_MARKER = "MYRMIDON_ENV_SCRIPT_APPLIED"  # set in the environment once sourced
# A LOGIN shell (bash -lc), because `module` is a shell function that the cluster's
# /etc/profile.d defines: in the bare environment an MCP client hands the server, a
# non-login shell has no `module`, so every `module load` in the script silently loads
# nothing and only the script's own exports survive (the same failure mode as piping
# `module load` into a subshell).
#
# The environment is captured twice — after the login profile, then after sourcing the
# script — and only the difference is applied. Applying the whole login environment
# instead would drag in everything the login profile sets and resurrect variables this process
# deliberately unset. The login profile may also print, so each `env -0` payload is
# delimited by a sentinel rather than parsed from the first byte.
_SENTINEL = b"__MYRMIDON_ENV_BASE__"
_SENTINEL_AFTER = b"__MYRMIDON_ENV_AFTER__"
_CAPTURE = (
    'set -e\n'
    f'printf %s {_SENTINEL.decode()}\n'
    'env -0\n'
    f'printf %s {_SENTINEL_AFTER.decode()}\n'
    '. "$1" >/dev/null 2>&1 || exit 97\n'
    'env -0'
)
# Variables that belong to the child shell or to this process's identity, never to the
# engine stack. HOME/USER/LOGNAME matter: a login profile on some clusters resets HOME,
# and applying that would move where the config and the client files are read from — the
# script's job is to provide engines, not to relocate the user.
_SKIP = {"_", "SHLVL", "PWD", "OLDPWD", "BASH_FUNC", "HOME", "USER", "LOGNAME", "SHELL",
         "TERM", "TMPDIR"}


@dataclass
class EnvScriptStatus:
    path: str | None = None
    state: str = "none"  # none | applied | already | missing | failed
    changed: list[str] = field(default_factory=list)  # variable names set or changed
    error: str | None = None

    def as_dict(self) -> dict:
        return {"path": self.path, "state": self.state, "changed": sorted(self.changed),
                "error": self.error}

    def describe(self) -> str:
        if self.state == "none":
            return "env_script: not set"
        if self.state == "applied":
            return (f"env_script: {self.path} sourced ({len(self.changed)} variable(s) "
                    f"set: {', '.join(sorted(self.changed)[:6])}"
                    + (" ..." if len(self.changed) > 6 else "") + ")")
        if self.state == "already":
            return f"env_script: {self.path} (already sourced by the parent process)"
        if self.state == "missing":
            return f"env_script: {self.path} NOT FOUND"
        return f"env_script: {self.path} FAILED to source — {self.error}"


def _parse_env0(payload: bytes) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in payload.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, _, value = item.decode(errors="replace").partition("=")
        if key in _SKIP or key.startswith("BASH_FUNC"):
            continue
        env[key] = value
    return env


def capture_env(script: Path, timeout: int = 120) -> dict[str, str]:
    """Source ``script`` in a child login bash and return what it *contributed*: the
    variables it set or changed relative to the login environment, not the whole thing.

    Raises RuntimeError when the script cannot be sourced (exit 97 from the wrapper), bash
    is unavailable, or a sentinel is missing (a login profile that swallowed the output)."""
    proc = subprocess.run(
        ["bash", "-lc", _CAPTURE, "bash", str(script)],
        capture_output=True, timeout=timeout, env=os.environ.copy(),
    )
    if proc.returncode == 97:
        err = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"'. {script}' exited non-zero" + (f": {err[-300:]}" if err else ""))
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip()[-300:]
                           or f"bash exited {proc.returncode}")
    if _SENTINEL not in proc.stdout or _SENTINEL_AFTER not in proc.stdout:
        raise RuntimeError("no environment captured — the login shell produced no `env -0` "
                           "output (check the shell profile)")
    base_bytes, after_bytes = proc.stdout.rsplit(_SENTINEL, 1)[1].split(_SENTINEL_AFTER, 1)
    base, after = _parse_env0(base_bytes), _parse_env0(after_bytes)
    return {k: v for k, v in after.items() if base.get(k) != v}


def apply_env_script(script: Path | str | None) -> EnvScriptStatus:
    """Source the script once into this process's environment; idempotent across
    parent/child processes through APPLIED_MARKER."""
    if not script:
        return EnvScriptStatus()
    path = Path(str(script)).expanduser()
    status = EnvScriptStatus(path=str(path))
    if os.environ.get(APPLIED_MARKER) == str(path):
        status.state = "already"
        return status
    if not path.is_file():
        status.state = "missing"
        return status
    try:
        new_env = capture_env(path)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        status.state = "failed"
        status.error = str(exc)
        return status
    for key, value in new_env.items():
        if os.environ.get(key) != value:
            os.environ[key] = value
            status.changed.append(key)
    os.environ[APPLIED_MARKER] = str(path)
    status.state = "applied"
    return status


def ensure_own_bin_on_path() -> str | None:
    """Put the directory of the running interpreter first on PATH if it is not there.

    Console scripts installed alongside myrmidon (mimicpy, gmx wrappers in the same
    venv/conda env) must resolve even when a client launched ``myrmidon`` by absolute
    path without activating the environment. Returns the directory when
    it was added."""
    import sys

    bindir = str(Path(sys.executable).resolve().parent)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if bindir in parts:
        return None
    os.environ["PATH"] = os.pathsep.join([bindir, *[p for p in parts if p]])
    return bindir


def source_line(script: Path | str) -> str:
    """The line a batch script uses to source it (POSIX dot: the scripts start #!/bin/sh)."""
    return f". {Path(str(script)).expanduser()}"
