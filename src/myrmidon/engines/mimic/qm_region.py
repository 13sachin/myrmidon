"""MiMiCPy's prepqm as a program: its binary, and the commands it refused."""

from __future__ import annotations

import re

from myrmidon.core.tool import ToolContext
from myrmidon.errors import EngineNotFoundError


def mimicpy_bin(ctx: ToolContext) -> str:
    info = ctx.engines.get("mimicpy")
    if info is None or not info.available:
        raise EngineNotFoundError("mimicpy", info.hint if info else "")
    return info.path


# What prepqm prints when it skips a command: it prints the error and goes on to write the
# region from the commands it did apply, exit 0 (tests/fixtures/mimicpy/
# prepqm_rejected_commands.out). Each line follows the prompt of the command it rejects.
_REJECTION = re.compile(
    r"is an invalid command!|did not return any atoms|is not a valid (?:selection keyword"
    r"|logical operator|boolean operator)|bracket is missing in selection|selection cannot "
    r"be empty|were not found in the coordinates file"
)


def rejected_commands(stdout: str, selections: list[str]) -> list[tuple[str, str]]:
    """(command, prepqm's message) for each command prepqm printed an error for and skipped.

    After the line inviting a selection, every prompt '> ' is followed by what that command
    printed, so the n-th segment belongs to the n-th command."""
    _, _, prompts = stdout.partition("Please enter selection below")
    segments = prompts.split("\n", 1)[-1].split("> ")[1:]
    out = []
    for i, segment in enumerate(segments):
        message = segment.strip().splitlines()[0] if segment.strip() else ""
        if _REJECTION.search(message):
            out.append((selections[i] if i < len(selections) else "?", message))
    return out
