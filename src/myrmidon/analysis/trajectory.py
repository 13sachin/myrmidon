"""Trajectory metrics via MDAnalysis (optional dependency, imported lazily).

Full series are saved as .npz artifacts; the observation carries the summary
statistics the harness computed from them.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation, TrajectoryObservation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.engines.inputs import resolve_in_ws


class TrajParams(BaseModel):
    structure: str = Field(description="topology-bearing file (.tpr/.gro)")
    trajectory: str = Field(description="trajectory file (.xtc/.trr)")
    metric: Literal["rmsd", "rmsf", "rg", "rdf", "mindist"]
    selection: str = Field(default="protein", description="MDAnalysis selection")
    selection2: str = Field(
        default="", description="second selection (rdf and mindist only)"
    )


@tool(
    "traj_analyze",
    description="Compute a trajectory metric (rmsd | rmsf | rg | rdf | mindist) with "
    "MDAnalysis. Summary stats are harness-computed; the full series is saved as an "
    ".npz artifact. Units: Å (MDAnalysis native). Prefer a .gro over a .tpr for "
    "'structure' — MDAnalysis lags behind new GROMACS tpr versions.",
    returns=TrajectoryObservation,
    tags={"analysis"},
)
def traj_analyze(ctx: ToolContext, params: TrajParams) -> TrajectoryObservation:
    try:
        import MDAnalysis as mda
    except ImportError:
        return TrajectoryObservation(
            ok=False,
            summary="MDAnalysis is not installed — install myrmidon[analysis]",
            metric=params.metric,
        )
    stage = ctx.workspace.stage("analysis")
    structure = resolve_in_ws(ctx, params.structure)
    trajectory = resolve_in_ws(ctx, params.trajectory)
    u = mda.Universe(str(structure), str(trajectory))

    if params.metric in ("rdf", "mindist") and not params.selection2:
        return TrajectoryObservation(
            ok=False,
            summary=f"{params.metric} needs selection2",
            metric=params.metric,
        )

    x: np.ndarray
    y: np.ndarray
    unit = "Å"
    x_label = "time_ps"

    if params.metric == "rmsd":
        from MDAnalysis.analysis import rms

        result = rms.RMSD(u, u, select=params.selection, ref_frame=0).run()
        arr = result.results.rmsd
        x, y = arr[:, 1], arr[:, 2]
    elif params.metric == "rmsf":
        from MDAnalysis.analysis import rms

        atoms = u.select_atoms(params.selection)
        result = rms.RMSF(atoms).run()
        x = np.arange(len(atoms), dtype=float)
        y = result.results.rmsf
        x_label = "atom_index"
    elif params.metric == "rg":
        atoms = u.select_atoms(params.selection)
        frames = [(ts.time, atoms.radius_of_gyration()) for ts in u.trajectory]
        x = np.array([f[0] for f in frames])
        y = np.array([f[1] for f in frames])
    elif params.metric == "rdf":
        from MDAnalysis.analysis.rdf import InterRDF

        g1 = u.select_atoms(params.selection)
        g2 = u.select_atoms(params.selection2)
        result = InterRDF(g1, g2).run()
        x, y = result.results.bins, result.results.rdf
        unit = "g(r)"
        x_label = "r_angstrom"
    else:  # mindist
        from MDAnalysis.analysis import distances

        g1 = u.select_atoms(params.selection)
        g2 = u.select_atoms(params.selection2)
        frames = []
        for ts in u.trajectory:
            d = distances.distance_array(g1.positions, g2.positions, box=ts.dimensions)
            frames.append((ts.time, float(d.min())))
        x = np.array([f[0] for f in frames])
        y = np.array([f[1] for f in frames])

    if y.size == 0:
        return TrajectoryObservation(
            ok=False,
            summary=f"selection matched no atoms or trajectory is empty "
            f"({params.selection!r})",
            metric=params.metric,
        )

    out = stage / f"{params.metric}.npz"
    np.savez(out, **{x_label: x, params.metric: y})
    return TrajectoryObservation(
        ok=True,
        summary=f"{params.metric} over {len(y)} points: mean {float(np.mean(y)):.3f}, "
        f"final {float(y[-1]):.3f} {unit}",
        artifacts=[ArtifactRef.from_path(out, "data", root=ctx.workspace.root)],
        metric=params.metric,
        unit=unit,
        mean=float(np.mean(y)),
        final=float(y[-1]),
        n_frames=int(len(y)),
    )


class TrajFramesParams(BaseModel):
    structure: str = Field(description="topology-bearing file (.gro preferred)")
    trajectory: str = Field(description="trajectory to pick frames from (.trr/.xtc)")
    frames: list[int] = Field(
        default_factory=list,
        description="frame INDICES (0-based) to write; empty = n_frames evenly spaced",
    )
    n_frames: int = Field(default=3, ge=1, description="used when frames is empty")
    stage: str = Field(default="frames")
    prefix: str = Field(default="frame")


class FrameRef(BaseModel):
    index: int
    time_ps: float
    path: str


class TrajFramesObservation(Observation):
    n_total: int = 0
    frames: list[FrameRef] = Field(default_factory=list)


@tool(
    "traj_frames",
    description="Pick snapshots from a trajectory and write each as a .gro (frame index "
    "and time recorded): explicit indices, or n_frames evenly spaced over the whole "
    "trajectory. The starting structures for QM/MM single points.",
    returns=TrajFramesObservation,
    # "analysis", not "fm": extracting frames is what any study does before single
    # points, and a tag says what a tool is
    tags={"analysis"},
)
def traj_frames(ctx: ToolContext, params: TrajFramesParams) -> TrajFramesObservation:
    try:
        import MDAnalysis as mda
    except ImportError:
        return TrajFramesObservation(
            ok=False, summary="MDAnalysis is not installed — install myrmidon[analysis]",
        )
    structure = resolve_in_ws(ctx, params.structure)
    trajectory = resolve_in_ws(ctx, params.trajectory)
    u = mda.Universe(str(structure), str(trajectory))
    n_total = len(u.trajectory)
    if params.frames:
        indices = list(params.frames)
        bad = [i for i in indices if i < 0 or i >= n_total]
        if bad:
            return TrajFramesObservation(
                ok=False, n_total=n_total,
                summary=f"frame index out of range {bad}: the trajectory has {n_total} frames",
            )
    else:
        k = min(params.n_frames, n_total)
        indices = sorted({int(round(x)) for x in np.linspace(0, n_total - 1, k)})
    stage = ctx.workspace.stage(params.stage)
    root = ctx.workspace.root
    refs, artifacts = [], []
    for i in indices:
        ts = u.trajectory[i]
        out = stage / f"{params.prefix}_{i:05d}.gro"
        with mda.Writer(str(out), n_atoms=len(u.atoms)) as w:
            w.write(u.atoms)
        refs.append(FrameRef(index=i, time_ps=float(ts.time), path=ctx.workspace.relpath(out)))
        artifacts.append(ArtifactRef.from_path(out, "structure", root=root))
    listed = ", ".join(f"#{r.index} (t={r.time_ps:g} ps)" for r in refs)
    return TrajFramesObservation(
        ok=True, n_total=n_total, frames=refs, artifacts=artifacts,
        summary=f"{len(refs)} frame(s) of {n_total} written to {ctx.workspace.relpath(stage)}: "
        f"{listed}",
    )
