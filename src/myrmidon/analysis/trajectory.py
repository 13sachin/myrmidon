"""What a structure file holds for an MDAnalysis reading of it: bonds, which making a
molecule whole needs, and how many atoms a selection selects."""

from __future__ import annotations

import numpy as np


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


def evenly_spaced(n_total: int, k: int, start: int = 0) -> list[int]:
    """``k`` indices evenly spaced over ``start`` to the last of ``n_total``."""
    k = min(k, n_total - start)
    return sorted({int(round(x)) for x in np.linspace(start, n_total - 1, k)}) if k > 0 else []


def settled_from(times: list[float], series: list[float]) -> int:
    """The index a series has settled from (Chodera 2016); the start for one too short to
    tell."""
    from myrmidon.analysis.convergence import describe

    if len(series) < 4:
        return 0
    found = describe("series", np.asarray(times, dtype=float), np.asarray(series, dtype=float),
                     "", None, None, 0.5)
    return next((i for i, t in enumerate(times) if t >= found.equilibrated_from_ps), 0)


def closest_settled(times: list[float], series: list[float], target: float) -> int:
    """The index in a series' settled part nearest ``target``."""
    start = settled_from(times, series)
    return min(range(start, len(series)), key=lambda i: abs(series[i] - target))


def displaced_structure(lines: list[str], anchor: int, mobile: int, target_nm: float,
                        rigid_group: list[int]) -> list[str]:
    """A .gro's lines with the mobile atom moved along the anchor->mobile axis to a target
    distance; with a rigid group (serials, holding the mobile atom), every one of its atoms moved
    by that same shift, so a displaced single point carries no strain inside the group."""
    def xyz(line: str) -> list[float]:
        return [float(line[20:28]), float(line[28:36]), float(line[36:44])]

    a, b = xyz(lines[1 + anchor]), xyz(lines[1 + mobile])
    vec = [b[k] - a[k] for k in range(3)]
    length = sum(v * v for v in vec) ** 0.5
    unit = [v / length for v in vec]
    shift = [(a[k] + unit[k] * target_nm) - b[k] for k in range(3)]
    out = list(lines)
    for serial in rigid_group or [mobile]:
        line = out[1 + serial]
        moved = [c + s for c, s in zip(xyz(line), shift, strict=True)]
        out[1 + serial] = line[:20] + "".join(f"{v:8.3f}" for v in moved) + line[44:]
    return out
