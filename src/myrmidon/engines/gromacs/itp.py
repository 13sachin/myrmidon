"""Reading a GROMACS topology (.itp/.top): one walk over its data rows, and what the
force-matching fit and its report read from it — charges, and bonded parameters.

The fit and the report both read ``[ atoms ]`` and the bonded sections through this module,
so there is one idea of where a section ends.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

_SECTION_RE = re.compile(r"^\s*\[\s*(\w+)\s*\]")
#: atoms per row of a bonded section
BONDED_N_ATOMS = {"bonds": 2, "angles": 3, "dihedrals": 4}


def itp_rows(text: str) -> Iterator[tuple[str | None, str | None, list[str]]]:
    """(moleculetype, section, fields) for every data row; comments and blanks skipped."""
    section = None
    mol = None
    for line in text.splitlines():
        body = line.split(";")[0].strip()
        if not body:
            continue
        m = _SECTION_RE.match(body)
        if m:
            section = m.group(1).lower()
            continue
        fields = body.split()
        if section == "moleculetype":
            mol = fields[0]
            continue
        yield mol, section, fields


def parse_itp_atoms(text: str) -> dict[str, list[dict]]:
    """moleculetype -> [{nr, type, name, charge}] from the [ atoms ] sections."""
    out: dict[str, list[dict]] = {}
    for mol, section, f in itp_rows(text):
        if section != "atoms" or len(f) < 7:
            continue
        try:
            out.setdefault(mol or "?", []).append(
                {"nr": int(f[0]), "type": f[1], "name": f[4], "charge": float(f[6])}
            )
        except ValueError:
            continue
    return out


def atom_charges(text: str) -> list[tuple[str, float]]:
    """(atom name, charge) for every [ atoms ] row, in file order."""
    return [(a["name"], a["charge"]) for atoms in parse_itp_atoms(text).values() for a in atoms]


def parse_itp_bonded(text: str) -> dict[str, dict[str, dict[tuple, list[float]]]]:
    """moleculetype -> section -> (atom indices..., funct, occurrence) -> parameters.

    Occurrence disambiguates repeated rows for the same atoms (funct 9 dihedrals
    carry one row per multiplicity)."""
    out: dict[str, dict[str, dict[tuple, list[float]]]] = {}
    seen: dict[tuple, int] = {}
    for mol, section, f in itp_rows(text):
        n = BONDED_N_ATOMS.get(section or "")
        if n is None or len(f) <= n:
            continue
        try:
            atoms = tuple(int(x) for x in f[:n])
            funct = int(f[n])
            params = [float(x) for x in f[n + 1:]]
        except ValueError:
            continue
        if not params:
            continue  # parameters come from the force field, nothing was fitted here
        base = (mol, section, atoms, funct)
        k = seen.get(base, 0)
        seen[base] = k + 1
        out.setdefault(mol or "?", {}).setdefault(section, {})[atoms + (funct, k)] = params
    return out


def bond_angle_rows(text: str) -> Iterator[tuple[str, tuple[str, ...], float, float]]:
    """(section, indices, equilibrium, force constant) for every bond and angle row.

    **Assumes GROMACS function type 1 column layout** — `b0 kb` for a bond, `th0 cth` for
    an angle. CHARMM Urey-Bradley angles (funct 5) parse correctly because their first two
    parameters are the same two; a Morse bond (funct 3) would have its `beta` read as a
    force constant. Nucleic acids and lipids bring other conventions, and on those the
    implausibility findings built on this would be reading the wrong column.
    """
    for _mol, section, fields in itp_rows(text):
        if section not in ("bonds", "angles"):
            continue
        n_idx = BONDED_N_ATOMS[section]
        try:
            yield section, tuple(fields[:n_idx]), float(fields[n_idx + 1]), \
                float(fields[n_idx + 2])
        except (IndexError, ValueError):
            continue


_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)


def expanded_topology(top: Path, search_dirs: tuple[Path, ...] = ()) -> tuple[str, list[str]]:
    """The topology's text with every ``#include`` it can find inlined, and the includes it
    could not. An include resolves as grompp's does: beside the file that includes it, then
    in each search directory (GROMACS's force-field directory). Conditionals are not
    evaluated; what they guard in a topology is restraints and flexible water, never a
    molecule's atoms."""
    missing: list[str] = []

    def walk(path: Path, depth: int) -> str:
        text = path.read_text(errors="replace")
        if depth > 20:
            return text

        def inline(m: re.Match) -> str:
            inc = m.group(1)
            for base in (path.parent, *search_dirs):
                if (base / inc).is_file():
                    return walk(base / inc, depth + 1)
            missing.append(inc)
            return ""

        return _INCLUDE_RE.sub(inline, text)

    return walk(top, 0), missing


def molecule_layout(text: str) -> list[tuple[str, int, int | None]]:
    """(moleculetype, copies, atoms per copy or None if no definition was read) for every
    ``[ molecules ]`` row, in order: how the topology lays out the structure's atoms."""
    sizes: dict[str, int] = {}
    rows: list[tuple[str, int]] = []
    for mol, section, f in itp_rows(text):
        if f[0].startswith("#"):
            continue
        if section == "atoms" and mol is not None and len(f) >= 5:
            sizes[mol] = sizes.get(mol, 0) + 1
        elif section == "molecules" and len(f) >= 2 and f[1].isdigit():
            rows.append((f[0], int(f[1])))
    return [(name, n, sizes.get(name)) for name, n in rows]


def molecule_of(layout: list[tuple[str, int, int | None]], atoms: list[int]):
    """Where atoms numbered over the whole structure sit in the topology's layout:
    ``(moleculetype, number of its first atom in the structure)`` when they all lie in one
    copy of one molecule, ``"spans"`` when they do not, None when a molecule before them
    has no definition and the layout cannot be followed that far."""
    lo, hi = min(atoms), max(atoms)
    start = 1
    for name, copies, size in layout:
        if size is None:
            return None
        end = start + copies * size  # first atom after this row
        if lo < end:
            if hi >= end:
                return "spans"
            first = start + (lo - start) // size * size
            return (name, first) if hi < first + size else "spans"
        start = end
    return "spans"
