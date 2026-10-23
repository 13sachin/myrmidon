"""Input files a tool is given: where they are, and the finding when one is not there.

Every engine's gate checks its input files the same way, and a missing one is the same
blocking finding everywhere: ``INPUT_MISSING_<NAME>``, naming the parameter that resolves
it (``needs``), which the refusal's next action then names (core/tool.py).
"""

from __future__ import annotations

from pathlib import Path

from myrmidon.core.tool import ToolContext
from myrmidon.core.validation import ValidationWarning
from myrmidon.errors import MyrmidonError


def resolve_in_ws(ctx: ToolContext, path: str) -> Path:
    """``path`` as given if absolute, else under the study or its project directory
    (``Workspace.resolve``); raises if it is not there."""
    resolved = ctx.workspace.resolve(path)
    if not resolved.exists():
        raise MyrmidonError(f"input file not found: {path}")
    return resolved


def require_input(ctx: ToolContext, param: str, path: str, finding_id: str | None = None,
                  message: str | None = None) -> tuple[Path | None, ValidationWarning | None]:
    """The file a parameter names, or the blocking finding that says it is missing."""
    try:
        return resolve_in_ws(ctx, path), None
    except MyrmidonError:
        return None, ValidationWarning(
            id=finding_id or f"INPUT_MISSING_{param.upper()}", severity="block",
            message=message or f"{param} file not found: {path}", needs=[param])
