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


def run_acpype(argv: list[str], cwd: Path, env: dict, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True,  # noqa: S603
                          timeout=timeout)
