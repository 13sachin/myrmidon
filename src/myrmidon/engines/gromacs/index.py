"""make_ndx's account of its commands and the groups an index file holds."""

from __future__ import annotations

import re

_REJECTION = re.compile(r"Syntax error|Group is empty|does not exist|^Error", re.MULTILINE)


def ndx_groups(text: str) -> list[tuple[str, list[int]]]:
    """(name, atom numbers) for every group of a .ndx file, in order."""
    groups: list[tuple[str, list[int]]] = []
    for line in text.splitlines():
        s = line.split(";")[0].strip()
        if s.startswith("["):
            groups.append((s.strip("[] "), []))
        elif s and groups:
            groups[-1][1].extend(int(x) for x in s.split())
    return groups


def rejected_ndx_commands(stdout: str, commands: list[str]) -> list[tuple[str, str]]:
    """(command, make_ndx's message) for each command make_ndx printed an error for.
    The listing comes first; after it, each prompt '> ' is followed by its command's output."""
    segments = re.split(r"^> ", stdout, flags=re.MULTILINE)[1:]
    out = []
    for i, segment in enumerate(segments[: len(commands)]):
        m = _REJECTION.search(segment)
        if m:
            line = segment[m.start():].splitlines()[0].strip()
            out.append((commands[i], line))
    return out
