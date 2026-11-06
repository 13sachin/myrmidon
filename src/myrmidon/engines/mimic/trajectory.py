"""What a coupled run leaves of its frames: CPMD's ENERGIES and TRAJECTORY read, a stored step
written as the GEOMETRY file a later stage starts from, and the steps it shares with the frames
of the GROMACS half (tests/fixtures/cpmd/acetone_coupled.*).

TRAJECTORY holds every atom, QM and MM, each line ``step x y z vx vy vz`` in atomic units, at
steps 1, 1 + N, 1 + 2N for a stride N; the GROMACS half numbers the same frames 0, N, 2N. GEOMETRY
is one such frame without the step.
"""

from __future__ import annotations

from pathlib import Path


def gromacs_step(cpmd_step: int) -> int:
    """The step the GROMACS half numbers a CPMD step's frame with."""
    return cpmd_step - 1


def parse_energies(text: str) -> dict[int, float]:
    """Step -> the ionic temperature in K, from ENERGIES (``step ekinc tempp ...``)."""
    out: dict[int, float] = {}
    for line in text.splitlines():
        cols = line.split()
        if len(cols) >= 3:
            try:
                out[int(cols[0])] = float(cols[2])
            except ValueError:
                continue
    return out


def trajectory_steps(path: Path) -> list[int]:
    """The steps TRAJECTORY stores, in order."""
    steps: list[int] = []
    with path.open() as fh:
        for line in fh:
            first = line.split(None, 1)[0] if line.strip() else ""
            if first.isdigit() and (not steps or steps[-1] != int(first)):
                steps.append(int(first))
    return steps


def frame_rows(path: Path, step: int) -> list[list[float]]:
    """Each atom's x, y, z, vx, vy, vz at a stored step; empty when it is not stored."""
    rows: list[list[float]] = []
    wanted = str(step)
    with path.open() as fh:
        for line in fh:
            cols = line.split()
            if len(cols) >= 7 and cols[0] == wanted:
                rows.append([float(v) for v in cols[1:7]])
    return rows


def write_geometry(rows: list[list[float]], path: Path) -> None:
    """A frame as CPMD's GEOMETRY file: positions then velocities, one atom a line."""
    path.write_text("".join(
        "".join(f"{v:20.12f}" for v in row[:3]) + "  " + "".join(f"{v:20.12f}" for v in row[3:])
        + "\n" for row in rows))


def cut_snapshots(ctx, request):
    """A coupled run's frames as matched pairs: GEOMETRY for CPMD and a .gro for GROMACS, cut at
    a step both engines stored, so neither starts from the other's past."""
    import MDAnalysis as mda
    from MDAnalysis.lib.formats.libmdaxdr import TRRFile

    from myrmidon.analysis.trajectory import closest_settled, evenly_spaced
    from myrmidon.core.artifacts import ArtifactRef
    from myrmidon.core.observation import Observation
    from myrmidon.engines.inputs import resolve_in_ws

    qm_path = resolve_in_ws(ctx, request.qm_trajectory)
    temps = parse_energies(resolve_in_ws(ctx, request.energies).read_text())
    trr = resolve_in_ws(ctx, request.trajectory)
    with TRRFile(str(trr)) as fh:
        mm_steps = [frame.step for frame in fh]
    common = [s for s in trajectory_steps(qm_path) if gromacs_step(s) in mm_steps and s in temps]
    if not common:
        return Observation(ok=False, summary="the two engines stored no step in common: a stage "
                           "writes both at one stride unless its input was edited")
    if request.target_temperature_k is not None:
        picks = [closest_settled([float(s) for s in common], [temps[s] for s in common],
                                 request.target_temperature_k)]
    elif request.frames:
        picks = list(request.frames)
        past = [i for i in picks if not 0 <= i < len(common)]
        if past:
            return Observation(ok=False, n_total=len(common), summary=f"frame index out of range "
                               f"{past}: the run has {len(common)} common frames")
    else:
        picks = evenly_spaced(len(common), request.n_frames)
    stage = ctx.workspace.stage(request.output_stage)
    root = ctx.workspace.root
    u = mda.Universe(str(resolve_in_ws(ctx, request.structure)), str(trr))
    frames, artifacts = [], []
    for pick in picks:
        step = common[pick]
        geometry = stage / f"{request.prefix}_{step}.GEOMETRY"
        write_geometry(frame_rows(qm_path, step), geometry)
        gro = stage / f"{request.prefix}_{step}.gro"
        u.trajectory[mm_steps.index(gromacs_step(step))]
        u.atoms.write(str(gro))
        frames.append({"step": step, "temperature_k": temps[step],
                       "geometry": ctx.workspace.relpath(geometry),
                       "structure": ctx.workspace.relpath(gro)})
        artifacts += [ArtifactRef.from_path(geometry, "geometry", root=root),
                      ArtifactRef.from_path(gro, "structure", root=root)]
    return Observation(
        ok=True, frames=frames, artifacts=artifacts,
        summary=f"{len(frames)} matched snapshot(s) of {len(common)} common steps: "
        + ", ".join(f"step {f['step']} at {f['temperature_k']:g} K" for f in frames)
        + ". stage_run takes the geometry as snapshot=<path>")
