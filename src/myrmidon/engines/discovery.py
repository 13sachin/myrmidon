"""Engine discovery: find simulation binaries on this machine, never bundle them.

Missing engines are not errors — their tools are simply not exposed, and the
``doctor`` tool explains what is unavailable and why.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel

from myrmidon.config import MyrmidonConfig


def _which(binary: str) -> str | None:
    """shutil.which, resolved to an absolute path — tools run from other cwds."""
    found = shutil.which(binary)
    return str(Path(found).resolve()) if found else None


def _not_runnable_hint(config: MyrmidonConfig, path: str, out) -> str:
    """A binary on PATH that exits non-zero on --version is not available, and saying so
    is the point: reporting it as available makes every later engine call fail with the
    same message deep inside a study instead of here."""
    detail = (out.stderr or out.stdout or "").strip().splitlines()
    why = detail[-1][:200] if detail else f"exit {out.returncode}"
    tail = ""
    if "shared libraries" in why or "cannot open shared object" in why:
        tail = (" — a module-provided library is missing: the environment was not fully "
                "set up")
        if config.env_script:
            tail += f" (env_script {config.env_script} sourced without the module system?)"
        else:
            tail += " (name the module script as env_script in the myrmidon config)"
    return f"found at {path} but not runnable: {why}{tail}"


class EngineInfo(BaseModel):
    name: str
    path: str | None = None
    version: str = "unknown"
    available: bool = False
    hint: str = ""  # why unavailable / how to fix
    data_dir: str | None = None  # gromacs: <prefix>/share/gromacs/top (force fields)


def find_gromacs(config: MyrmidonConfig) -> EngineInfo:
    path = _which(config.gromacs_bin)
    if path is None:
        return EngineInfo(
            name="gromacs",
            hint=f"'{config.gromacs_bin}' not on PATH — set gromacs_bin in the "
            "myrmidon config or load the GROMACS module",
        )
    try:
        out = subprocess.run(  # noqa: S603
            [path, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return EngineInfo(name="gromacs", path=path, hint=f"found but not runnable: {exc}")
    if out.returncode != 0:
        # on PATH but unable to run: typically a module-provided shared library missing
        # because the environment was not fully set up (exit 127 + a loader message)
        return EngineInfo(name="gromacs", path=path,
                          hint=_not_runnable_hint(config, path, out))
    text = out.stdout + out.stderr
    m = re.search(r"GROMACS version:?\s*(\S+)", text)
    version = m.group(1) if m else "unknown"
    data_dir = None
    pm = re.search(r"Data prefix:\s*(\S+)", text)
    if pm:
        candidate = Path(pm.group(1)) / "share" / "gromacs" / "top"
        if candidate.is_dir():
            data_dir = str(candidate)
    return EngineInfo(
        name="gromacs", path=path, version=version, available=True, data_dir=data_dir
    )


def find_cpmd(config: MyrmidonConfig) -> EngineInfo:
    path = _which(config.cpmd_bin)
    if path is None:
        return EngineInfo(
            name="cpmd",
            hint=f"'{config.cpmd_bin}' not on PATH — set cpmd_bin in the myrmidon config",
        )
    # CPMD prints its version only on a real run; record presence, verify later.
    return EngineInfo(name="cpmd", path=path, version="unknown", available=True)


def find_mimicpy(config: MyrmidonConfig) -> EngineInfo:
    path = _which(config.mimicpy_bin)
    if path is None:
        return EngineInfo(
            name="mimicpy",
            hint=f"'{config.mimicpy_bin}' not on PATH — pip install mimicpy, or set "
            "mimicpy_bin in the myrmidon config",
        )
    return EngineInfo(name="mimicpy", path=path, version="unknown", available=True)


def find_chimerax(config: MyrmidonConfig) -> EngineInfo:
    path = _which(config.chimerax_bin)
    if path is None:
        # macOS app bundle (not on PATH by default); newest version wins.
        candidates = sorted(
            Path("/Applications").glob("ChimeraX*.app/Contents/bin/ChimeraX"),
            reverse=True,
        )
        path = str(candidates[0]) if candidates else None
    if path is None:
        return EngineInfo(
            name="chimerax",
            hint=f"'{config.chimerax_bin}' not on PATH and no ChimeraX*.app in "
            "/Applications — install from https://www.rbvi.ucsf.edu/chimerax/ or set "
            "chimerax_bin in the myrmidon config",
        )
    try:
        out = subprocess.run(  # noqa: S603
            [path, "--version"], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return EngineInfo(name="chimerax", path=path, hint=f"found but not runnable: {exc}")
    if out.returncode != 0:
        return EngineInfo(name="chimerax", path=path,
                          hint=_not_runnable_hint(config, path, out))
    m = re.search(r"ChimeraX version:?\s*(\S+)", out.stdout + out.stderr)
    return EngineInfo(
        name="chimerax",
        path=path,
        version=m.group(1) if m else "unknown",
        available=True,
    )


def discover(config: MyrmidonConfig) -> dict[str, EngineInfo]:
    return {
        "gromacs": find_gromacs(config),
        "cpmd": find_cpmd(config),
        "mimicpy": find_mimicpy(config),
        "chimerax": find_chimerax(config),
    }
