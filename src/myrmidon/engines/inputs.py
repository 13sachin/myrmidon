"""Input files a tool is given: where they are, and the finding when one is not there.

Every engine's gate checks its input files the same way, and a missing one is the same
blocking finding everywhere: ``INPUT_MISSING_<NAME>``, naming the parameter that resolves
it (``needs``).
"""

from __future__ import annotations

import os
from pathlib import Path

from myrmidon.core.tool import ToolContext
from myrmidon.core.validation import Finding, Precondition
from myrmidon.errors import MyrmidonError


def resolve_in_ws(ctx: ToolContext, path: str) -> Path:
    """``path`` as given if absolute, else under the study or its project directory
    (``Workspace.resolve``); raises if it is not there."""
    resolved = ctx.workspace.resolve(path)
    if not resolved.exists():
        raise MyrmidonError(f"input file not found: {path}" + _held(ctx, path))
    return resolved


def _held(ctx: ToolContext, path: str) -> str:
    held = same_name(ctx, path)
    return "; the study holds a file of that name at " + ", ".join(held) if held else ""


def same_name(ctx: ToolContext, path: str, limit: int = 3) -> list[str]:
    """The study's files named like ``path``, as study paths: those its calls wrote first,
    as the ledger records them, then any other under the study."""
    name = Path(path).name
    ws = ctx.workspace
    found: list[str] = []
    for event in ws.ledger.events():
        for art in event.outputs:
            if Path(art.path).name == name and art.path not in found \
                    and ws.resolve(art.path).exists():
                found.append(art.path)
    if not found:  # the name exactly, not as a pattern
        found = sorted(ws.relpath(Path(top) / name) for top, _, files in os.walk(ws.root)
                       if name in files)
    return found[:limit]


def require_input(ctx: ToolContext, param: str, path: str, finding_id: str | None = None,
                  message: str | None = None) -> tuple[Path | None, Finding | None]:
    """The file a parameter names, or the blocking finding that says it is missing."""
    resolved = ctx.workspace.resolve(path)
    if resolved.exists():
        return resolved, None
    return None, Precondition(
        id=finding_id or f"INPUT_MISSING_{param.upper()}",
        message=(message or f"{param} file not found: {path}") + _held(ctx, path),
        needs=[param])


def symlink_component(root: Path, relpath: str) -> Path | None:
    """The first path component under ``root`` that is a symlink, else None —
    the check every write must make before it follows a link somewhere shared."""
    current = root
    for part in Path(relpath).parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def materialize_symlink(link: Path) -> Path:
    """Replace ``link`` with a real copy of what it points to (a directory tree
    is copied with symlinks=False). Returns the former target."""
    import shutil

    real = link.resolve()
    link.unlink()
    if real.is_dir():
        shutil.copytree(real, link, symlinks=False)
    else:
        shutil.copy2(real, link)
    return real
