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


def picked_frames(ctx, request):
    """Frames of a trajectory by index or evenly spaced, as a .gro each or one trajectory in the
    order asked: an observation holding them, its ``frames`` the record of what was picked."""
    import MDAnalysis as mda

    from myrmidon.core.artifacts import ArtifactRef
    from myrmidon.core.observation import Observation
    from myrmidon.engines.inputs import resolve_in_ws

    u = mda.Universe(str(resolve_in_ws(ctx, request.structure)),
                     str(resolve_in_ws(ctx, request.trajectory)))
    n_total = len(u.trajectory)
    indices = list(request.frames) or evenly_spaced(n_total, request.n_frames)
    past = [i for i in indices if i < 0 or i >= n_total]
    if past:
        return Observation(ok=False, n_total=n_total, summary=f"frame index out of range {past}: "
                           f"the trajectory has {n_total} frames")
    stage = ctx.workspace.stage(request.output_stage)
    root = ctx.workspace.root
    frames, artifacts = [], []
    whole = stage / f"{request.prefix}.trr"
    if request.output == "trajectory":
        with mda.Writer(str(whole), n_atoms=len(u.atoms)) as writer:
            for i in indices:
                ts = u.trajectory[i]
                writer.write(u.atoms)
                frames.append({"index": i, "time_ps": float(ts.time),
                               "path": ctx.workspace.relpath(whole)})
        artifacts.append(ArtifactRef.from_path(whole, "trajectory", root=root))
    else:
        for i in indices:
            ts = u.trajectory[i]
            out = stage / f"{request.prefix}_{i}.gro"
            u.atoms.write(str(out))
            frames.append({"index": i, "time_ps": float(ts.time),
                           "path": ctx.workspace.relpath(out)})
            artifacts.append(ArtifactRef.from_path(out, "structure", root=root))
    return Observation(
        ok=True, frames=frames, artifacts=artifacts, n_total=n_total,
        summary=f"{len(frames)} of {n_total} frames: "
        + ", ".join(f"#{f['index']} at {f['time_ps']:g} ps" for f in frames),
        trajectory_path=ctx.workspace.relpath(whole) if request.output == "trajectory" else None)


def displaced_structures(ctx, request):
    """Structures with a pair at each target distance: the restraint-free way to sample
    off-equilibrium geometries."""
    from myrmidon.core.artifacts import ArtifactRef
    from myrmidon.core.observation import Observation
    from myrmidon.engines.inputs import resolve_in_ws

    lines = resolve_in_ws(ctx, request.coords).read_text().splitlines()
    n_atoms = int(lines[1].strip())
    if len(request.pair) != 2 or not all(1 <= s <= n_atoms for s in request.pair):
        return Observation(ok=False, summary=f"pair {request.pair} outside 1..{n_atoms}")
    anchor, mobile = request.pair
    group = sorted(set(request.rigid_group))
    if group and mobile not in group:
        return Observation(ok=False, summary=f"rigid_group must include the mobile atom {mobile}")
    if anchor in group:
        return Observation(ok=False, summary=f"rigid_group must not include the anchor {anchor}")
    outside = [s for s in group if not 1 <= s <= n_atoms]
    if outside:
        return Observation(ok=False, summary=f"rigid_group serials outside 1..{n_atoms}: "
                           f"{outside[:5]}")

    def at(serial: int) -> list[float]:
        line = lines[1 + serial]
        return [float(line[20:28]), float(line[28:36]), float(line[36:44])]

    original = sum((x - y) ** 2 for x, y in zip(at(anchor), at(mobile), strict=True)) ** 0.5
    if original < 1e-6:
        return Observation(ok=False, summary="pair atoms coincide")
    stage = ctx.workspace.stage(request.output_stage)
    windows, artifacts = [], []
    for target in request.targets_a:
        out = stage / f"disp_{target:g}A.gro"
        out.write_text("\n".join(displaced_structure(lines, anchor, mobile, target / 10.0, group))
                       + "\n")
        windows.append({"label": f"{request.pair_label}@{target:g}A", "target_a": target,
                        "coords": ctx.workspace.relpath(out)})
        artifacts.append(ArtifactRef.from_path(out, "structure", root=ctx.workspace.root))
    return Observation(
        ok=True, summary=f"{len(windows)} displaced structure(s) for {request.pair_label} (atoms "
        f"{anchor}->{mobile}, original {original * 10:.3f} Å) at "
        + ", ".join(f"{t:g}" for t in request.targets_a) + " Å",
        artifacts=artifacts, windows=windows, pair=request.pair,
        original_distance_a=round(original * 10, 4))
