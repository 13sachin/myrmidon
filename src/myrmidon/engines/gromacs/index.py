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


_QUOTED = re.compile(r'"([^"]+)"')


def resolve_repeated_names(commands: list[str], groups: list[tuple[str, list[int]]]
                           ) -> tuple[list[str], list[str]]:
    """(the commands with each quoted group name that several groups share, all holding the
    same atoms, replaced by the first one's number; what was replaced). GROMACS's default
    groups repeat a name for one set of atoms (a ligand's residue and its molecule, an
    ion's), and make_ndx refuses the name as ambiguous; names whose groups differ are left
    for make_ndx to refuse. Only commands before the first that renumbers or renames groups
    (del, keep, name) are read against the listing; the rest go as written."""
    by_name: dict[str, list[int]] = {}
    for i, (name, _) in enumerate(groups):
        by_name.setdefault(name.lower(), []).append(i)
    said: list[str] = []

    def number(m: re.Match) -> str:
        found = by_name.get(m.group(1).lower(), [])
        if len(found) > 1 and len({tuple(groups[i][1]) for i in found}) == 1:
            said.append(f'"{m.group(1)}" -> group {found[0]} (groups {found} hold the same '
                        "atoms)")
            return str(found[0])
        return m.group(0)

    out: list[str] = []
    for i, c in enumerate(commands):
        if c.split()[:1] and c.split()[0] in ("del", "keep", "name"):
            return [*out, *commands[i:]], said
        out.append(_QUOTED.sub(number, c))
    return out, said
