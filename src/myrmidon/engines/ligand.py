"""acpype as a program: its run, and what a ligand's coordinates say about its residue and
hydrogens."""

from __future__ import annotations

import subprocess
from pathlib import Path

from myrmidon.engines.structure import parse_pdb_atoms


def residue_name(path: Path) -> str | None:
    if path.suffix.lower() == ".mol2":
        rows = path.read_text(errors="replace").split("@<TRIPOS>ATOM", 1)
        fields = rows[1].split("@<TRIPOS>", 1)[0].split() if len(rows) > 1 else []
        return fields[7] if len(fields) > 7 else None
    atoms, _, _ = parse_pdb_atoms(path.read_text(errors="replace"))
    return atoms[0].resname if atoms else None


def hydrogen_count(path: Path) -> int | None:
    if path.suffix.lower() != ".pdb":
        return None
    atoms, _, _ = parse_pdb_atoms(path.read_text(errors="replace"))
    return sum(a.element in ("H", "D") for a in atoms)


#: atomic numbers of the elements a small organic molecule carries
ATOMIC_NUMBER = {"H": 1, "D": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "SI": 14, "P": 15,
                 "S": 16, "CL": 17, "SE": 34, "BR": 35, "I": 53}


def nuclear_charge(path: Path) -> int | None:
    """The sum of a PDB molecule's atomic numbers: its electron count when neutral. None
    for another format, a record without its element column (a name such as CL1 does not
    say whether it is chlorine), or an element outside ATOMIC_NUMBER."""
    if path.suffix.lower() != ".pdb":
        return None
    atoms, _, _ = parse_pdb_atoms(path.read_text(errors="replace"))
    z = [ATOMIC_NUMBER.get(a.element) if a.line[76:78].strip() else None for a in atoms]
    return sum(z) if atoms and None not in z else None


def run_acpype(argv: list[str], cwd: Path, env: dict, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True,  # noqa: S603
                          timeout=timeout)
