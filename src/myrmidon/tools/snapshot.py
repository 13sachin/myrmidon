"""Starting configurations picked from what an earlier stage left, for a stage to start from: by
temperature in its settled part, evenly spaced, by index, or displaced along a pair."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from myrmidon.analysis.trajectory import closest_settled, displaced_structure, evenly_spaced
from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract
from myrmidon.core.lineage import stage_run
from myrmidon.core.mandate import read_mandate
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.mimic.trajectory import (
    frame_rows,
    gromacs_step,
    parse_energies,
    trajectory_steps,
    write_geometry,
)
from myrmidon.tools.inputsets import system_file


class SnapshotSelectParams(BaseModel):
    stage: str = Field(default="", description="the stage whose collected run the frames come "
                       "from; its trajectory is found in the ledger")
    structure: str | None = Field(default=None, description="the structure defining atom order "
                                  "(.gro); default: the system's")
    trajectory: str | None = Field(default=None, description="a trajectory in place of a "
                                   "stage's (.trr/.xtc)")
    target_temperature_k: float | None = Field(
        default=None, description="the settled frame of a coupled stage nearest this ionic "
        "temperature, cut as a matched GEOMETRY and .gro")
    frames: list[int] = Field(default_factory=list, description="frame indices (0-based)")
    n_frames: int = Field(default=3, ge=1, description="evenly spaced frames, when no target or "
                          "indices are given")
    output: Literal["structures", "trajectory"] = Field(
        default="structures", description="'structures': a file per frame; 'trajectory': the "
        "frames in order as one .trr, its k-th frame the k-th index (what a rerun re-evaluates)")
    coords: str | None = Field(default=None, description="displace: the structure to displace")
    pair: list[int] = Field(default_factory=list, description="displace: [anchor, mobile] .gro "
                            "serials; the mobile atom moves along their axis")
    pair_label: str = Field(default="", description="displace: a name for the pair")
    targets_a: list[float] = Field(default_factory=list, description="displace: target distances "
                                   "in Å, one structure each")
    rigid_group: list[int] = Field(default_factory=list, description="displace: serials moved "
                                   "rigidly with the mobile atom, so no strain appears inside "
                                   "them; must hold the mobile atom, not the anchor")
    output_stage: str = Field(default="frames", description="where the snapshots are written")
    prefix: str = Field(default="frame")


def _displaced(ctx: ToolContext, params: SnapshotSelectParams) -> Observation:
    """Structures with a pair at each target distance: the restraint-free way to sample
    off-equilibrium geometries."""
    lines = resolve_in_ws(ctx, params.coords).read_text().splitlines()
    n_atoms = int(lines[1].strip())
    if len(params.pair) != 2 or not all(1 <= s <= n_atoms for s in params.pair):
        return Observation(ok=False, summary=f"pair {params.pair} outside 1..{n_atoms}")
    anchor, mobile = params.pair
    group = sorted(set(params.rigid_group))
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
    stage = ctx.workspace.stage(params.output_stage)
    windows, artifacts = [], []
    for target in params.targets_a:
        out = stage / f"disp_{target:g}A.gro"
        out.write_text("\n".join(displaced_structure(lines, anchor, mobile, target / 10.0, group))
                       + "\n")
        windows.append({"label": f"{params.pair_label}@{target:g}A", "target_a": target,
                        "coords": ctx.workspace.relpath(out)})
        artifacts.append(ArtifactRef.from_path(out, "structure", root=ctx.workspace.root))
    return Observation(
        ok=True, summary=f"{len(windows)} displaced structure(s) for {params.pair_label} (atoms "
        f"{anchor}->{mobile}, original {original * 10:.3f} Å) at "
        + ", ".join(f"{t:g}" for t in params.targets_a) + " Å",
        artifacts=artifacts, windows=windows, pair=params.pair,
        original_distance_a=round(original * 10, 4))


def _indices(params: SnapshotSelectParams, n_total: int) -> list[int] | str:
    if not params.frames:
        return evenly_spaced(n_total, params.n_frames)
    bad = [i for i in params.frames if i < 0 or i >= n_total]
    return (f"frame index out of range {bad}: the trajectory has {n_total} frames" if bad
            else list(params.frames))


def _written(ctx: ToolContext, stage, frames: list[dict], artifacts: list, summary: str,
             **fields) -> Observation:
    """The call's result, with its record of what was picked: the file a report reads."""
    record = stage / "snapshots.json"
    record.write_text(json.dumps(frames, indent=1) + "\n")
    return Observation(
        ok=True, summary=summary, frames=frames, **fields,
        artifacts=[*artifacts, ArtifactRef.from_path(record, "snapshot",
                                                     root=ctx.workspace.root)])


def _classical(ctx: ToolContext, params: SnapshotSelectParams, structure: str, trajectory: str
               ) -> Observation:
    """Frames of a trajectory, as a .gro each or one trajectory."""
    import MDAnalysis as mda

    u = mda.Universe(str(resolve_in_ws(ctx, structure)), str(resolve_in_ws(ctx, trajectory)))
    n_total = len(u.trajectory)
    indices = _indices(params, n_total)
    if isinstance(indices, str):
        return Observation(ok=False, summary=indices, n_total=n_total)
    stage = ctx.workspace.stage(params.output_stage)
    root = ctx.workspace.root
    frames, artifacts = [], []
    if params.output == "trajectory":
        out = stage / f"{params.prefix}.trr"
        with mda.Writer(str(out), n_atoms=len(u.atoms)) as w:
            for i in indices:
                ts = u.trajectory[i]
                w.write(u.atoms)
                frames.append({"index": i, "time_ps": float(ts.time),
                               "path": ctx.workspace.relpath(out)})
        artifacts.append(ArtifactRef.from_path(out, "trajectory", root=root))
    else:
        for i in indices:
            ts = u.trajectory[i]
            out = stage / f"{params.prefix}_{i}.gro"
            u.atoms.write(str(out))
            frames.append({"index": i, "time_ps": float(ts.time),
                           "path": ctx.workspace.relpath(out)})
            artifacts.append(ArtifactRef.from_path(out, "structure", root=root))
    return _written(ctx, stage, frames, artifacts,
                    f"{len(frames)} of {n_total} frames: "
                    + ", ".join(f"#{f['index']} at {f['time_ps']:g} ps" for f in frames),
                    n_total=n_total, trajectory_path=(ctx.workspace.relpath(out)
                                                      if params.output == "trajectory" else None))


def _coupled(ctx: ToolContext, params: SnapshotSelectParams, structure: str, outputs: list
             ) -> Observation:
    """A coupled run's frames as matched pairs: GEOMETRY for CPMD and a .gro for GROMACS, cut at
    a step both engines stored, so neither starts from the other's past."""
    import MDAnalysis as mda
    from MDAnalysis.lib.formats.libmdaxdr import TRRFile

    roles = {a.role: a.path for a in outputs}
    missing = [r for r in ("qm_trajectory", "energies", "trajectory") if r not in roles]
    if missing:
        return Observation(ok=False, summary=f"the run wrote no {', '.join(missing)} to cut a "
                           "snapshot from")
    qm_path = ctx.workspace.resolve(roles["qm_trajectory"])
    temps = parse_energies(ctx.workspace.resolve(roles["energies"]).read_text())
    # the frame positions CPMD's own frames pair with, velocities included: the full-precision one
    trajectories = [a.path for a in outputs if a.role == "trajectory"]
    trr = ctx.workspace.resolve(next((p for p in trajectories if p.endswith(".trr")),
                                     trajectories[0]))
    with TRRFile(str(trr)) as fh:
        mm_steps = [frame.step for frame in fh]
    common = [s for s in trajectory_steps(qm_path) if gromacs_step(s) in mm_steps and s in temps]
    if not common:
        return Observation(ok=False, summary="the two engines stored no step in common: a stage "
                           "writes both at one stride unless its input was edited")
    if params.target_temperature_k is not None:
        series = [temps[s] for s in common]
        picks = [closest_settled([float(s) for s in common], series,
                                 params.target_temperature_k)]
    else:
        picks = _indices(params, len(common))
        if isinstance(picks, str):
            return Observation(ok=False, summary=picks, n_total=len(common))
    stage = ctx.workspace.stage(params.output_stage)
    root = ctx.workspace.root
    u = mda.Universe(str(resolve_in_ws(ctx, structure)), str(trr))
    frames, artifacts = [], []
    for pick in picks:
        step = common[pick]
        geometry = stage / f"{params.prefix}_{step}.GEOMETRY"
        write_geometry(frame_rows(qm_path, step), geometry)
        gro = stage / f"{params.prefix}_{step}.gro"
        u.trajectory[mm_steps.index(gromacs_step(step))]
        u.atoms.write(str(gro))
        frames.append({"step": step, "temperature_k": temps[step],
                       "geometry": ctx.workspace.relpath(geometry),
                       "structure": ctx.workspace.relpath(gro)})
        artifacts += [ArtifactRef.from_path(geometry, "geometry", root=root),
                      ArtifactRef.from_path(gro, "structure", root=root)]
    return _written(ctx, stage, frames, artifacts,
                    f"{len(frames)} matched snapshot(s) of {len(common)} common steps: "
                    + ", ".join(f"step {f['step']} at {f['temperature_k']:g} K" for f in frames)
                    + ". stage_run takes the geometry as snapshot=<path>")


@tool(
    "snapshot_select",
    description="Pick starting configurations from a stage's run or a trajectory: the settled "
    "frame nearest a temperature, evenly spaced frames or indices, or a pair displaced to target "
    "distances. A coupled stage's come as a matched GEOMETRY and .gro.",
    tags={"analysis"},
    contract=Contract(consumes={"structure": ("structure", "tpr"), "trajectory": "trajectory",
                                "coords": "structure"}),
)
def snapshot_select(ctx: ToolContext, params: SnapshotSelectParams) -> Observation:
    if params.targets_a:
        if not (params.coords and params.pair):
            return Observation(ok=False, summary="displacing needs coords, a pair and targets_a")
        return _displaced(ctx, params)
    mandate = read_mandate(ctx.workspace)
    structure = params.structure or (system_file(ctx, mandate, "structure") if mandate else None)
    if structure is None:
        return Observation(ok=False, summary="name the structure defining the atom order")
    if params.stage and not params.trajectory:
        run = stage_run(ctx.workspace, params.stage)
        if run is None:
            return Observation(ok=False, summary=f"no collected run of stage {params.stage}: run "
                               "and collect it first")
        outputs = run[0]
        if any(a.role == "qm_trajectory" for a in outputs):
            return _coupled(ctx, params, structure, outputs)
        found = next((a.path for a in outputs if a.role == "trajectory"), None)
        if found is None:
            return Observation(ok=False, summary=f"the run of stage {params.stage} wrote no "
                               "trajectory")
        return _classical(ctx, params, structure, found)
    if not params.trajectory:
        return Observation(ok=False, summary="name the stage whose run the frames come from, or "
                           "a trajectory")
    return _classical(ctx, params, structure, params.trajectory)
