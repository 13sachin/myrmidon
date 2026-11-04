"""A ligand's atom types split from its molecule, and its coordinates placed after the
protein's: the topology and coordinate edits a protein–ligand complex is assembled from."""

from __future__ import annotations

import re
from pathlib import Path

from myrmidon.core.validation import Finding, Precondition

_SECTION = re.compile(r"^\s*\[\s*(\w+)\s*\]")
_FORCEFIELD_INCLUDE = re.compile(r'^\s*#include\s+"[^"]*forcefield\.itp"')
_FF_INCLUDE = re.compile(r'^\s*#include\s+"[^"/]+\.ff/[^"]+"')


def split_atomtypes(text: str) -> tuple[str, str]:
    """(the ``[ atomtypes ]`` sections, everything else) of an .itp."""
    types, rest = [], []
    target = rest
    for line in text.splitlines():
        m = _SECTION.match(line.split(";")[0])
        if m:
            target = types if m.group(1).lower() == "atomtypes" else rest
        target.append(line)
    return "\n".join(types).strip() + "\n", "\n".join(rest).strip() + "\n"


def place_ligand(top: str, molecule: str, types_itp: str, mol_itp: str,
                 posre_itp: str | None, define: str) -> str | None:
    """The topology with the ligand's atom types after the force field, its molecule
    before the first force-field include that follows (water, ions) or ``[ system ]``,
    and one copy at the end of ``[ molecules ]``; None when it has no force field include."""
    lines = top.splitlines()
    ff = next((i for i, ln in enumerate(lines) if _FORCEFIELD_INCLUDE.match(ln)), None)
    if ff is None:
        return None
    lines[ff + 1:ff + 1] = ["", "; ligand atom types: before any moleculetype",
                            f'#include "{types_itp}"']
    at = next((i for i, ln in enumerate(lines) if i > ff + 3 and (
        _FF_INCLUDE.match(ln) or (_SECTION.match(ln) and "system" in ln.lower()))), len(lines))
    while at > 0 and lines[at - 1].lstrip().startswith(";"):
        at -= 1  # keep the next block's comment with it
    block = ["; Include ligand topology", f'#include "{mol_itp}"']
    if posre_itp:
        block += [f"#ifdef {define}", f'#include "{posre_itp}"', "#endif"]
    lines[at:at] = [*block, ""]
    section = None
    last = len(lines) - 1
    for i, ln in enumerate(lines):
        m = _SECTION.match(ln.split(";")[0])
        if m:
            section = m.group(1).lower()
        elif section == "molecules" and ln.split(";")[0].strip():
            last = i
    lines.insert(last + 1, f"{molecule:<20}1")
    return "\n".join(lines) + "\n"


def gro_line(line: str, resnr: int, atomnr: int) -> str:
    return f"{resnr % 100000:5d}{line[5:15]}{atomnr % 100000:5d}{line[20:]}"


def itp_block(fid: str, message: str, needs: list[str] | None = None) -> Finding:
    return Precondition(id=fid, message=message, needs=needs)


def read_gro(path: Path) -> tuple[str, list[str], str]:
    """(title, atom lines, box line) of a .gro file."""
    lines = path.read_text().splitlines()
    n = int(lines[1].strip())
    return lines[0], lines[2:2 + n], lines[2 + n]
