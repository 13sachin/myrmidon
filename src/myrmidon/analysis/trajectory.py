"""What a structure file holds for an MDAnalysis reading of it: bonds, which making a
molecule whole needs, and how many atoms a selection selects."""

from __future__ import annotations


def has_bonds(u) -> bool:
    try:
        return len(u.bonds) > 0
    except Exception:  # noqa: BLE001 — MDAnalysis raises its own NoDataError
        return False


def selected_atoms(structure, selections: list[str]) -> list[tuple[int, int] | None]:
    """(atoms, residues) of ``structure`` each MDAnalysis selection selects; None for one
    MDAnalysis refuses. Empty when MDAnalysis is not installed or cannot read the file."""
    try:
        import MDAnalysis as mda

        u = mda.Universe(str(structure))
    except Exception:  # noqa: BLE001 — no reader for it: nothing is said about it
        return []
    out: list[tuple[int, int] | None] = []
    for selection in selections:
        try:
            atoms = u.select_atoms(selection)
            out.append((len(atoms), len(atoms.residues)))
        except Exception:  # noqa: BLE001 — a selection MDAnalysis does not parse
            out.append(None)
    return out
