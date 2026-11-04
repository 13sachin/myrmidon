"""What system preparation reads back: the solvent residues a structure holds, and an
output as an artifact of the study."""

from __future__ import annotations

from pathlib import Path

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.tool import ToolContext


def artifact(ctx: ToolContext, path: Path, role: str) -> ArtifactRef:
    return ArtifactRef.from_path(path, role, root=ctx.workspace.root)


#: residue names that mean "this is already solvent". GROMACS' own water models plus the
#: two spellings the PDB uses.
SOLVENT_RESIDUES = frozenset({"SOL", "WAT", "HOH", "TIP3", "TIP3P", "SPC", "SPCE", "T3P"})


def count_solvent_residues(structure: Path) -> tuple[int, str]:
    """(how many solvent molecules the coordinate file already holds, their name).

    Read off the .gro's residue column — the same place GROMACS reads it.
    """
    seen: set[str] = set()
    name = ""
    for line in structure.read_text(errors="replace").splitlines()[2:]:
        if len(line) < 10:
            continue
        residue = line[5:10].strip().upper()
        if residue in SOLVENT_RESIDUES:
            name = residue
            seen.add(line[:10])
    return len(seen), name
