"""Starting configurations picked from what an earlier stage left, for a stage to start from: by
temperature in its settled part, evenly spaced, by index, or displaced along a pair."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from myrmidon.analysis.trajectory import displaced_structures, picked_frames
from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract
from myrmidon.core.lineage import stage_run
from myrmidon.core.mandate import read_mandate
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.engines.registry import adapters
from myrmidon.tools.inputsets import system_file


class SnapshotSelectParams(BaseModel):
    stage: str = Field(default="", description="the stage whose collected run the frames come "
                       "from; its trajectory is found in the ledger")
    structure: str | None = Field(default=None, description="the structure defining atom order "
                                  "(.gro); default: the system's")
    trajectory: str | None = Field(default=None, description="a trajectory in place of a "
                                   "stage's (.trr/.xtc)")
    qm_trajectory: str | None = Field(default=None, description="filled from the stage's run: "
                                      "the QM engine's own frames")
    energies: str | None = Field(default=None, description="filled from the stage's run: the QM "
                                 "engine's temperatures by step")
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

    def sized(self, ctx: ToolContext) -> SnapshotSelectParams:
        """The call with what it read named, so the ledger holds the edge from each snapshot back
        to the run it was cut from: the stage's run's files, and the system's structure."""
        mandate = read_mandate(ctx.workspace)
        update: dict = {"structure": self.structure or (
            system_file(ctx, mandate, "structure") if mandate is not None else None)}
        run = stage_run(ctx.workspace, self.stage) if self.stage and not self.trajectory else None
        for artifact in run[0] if run else []:
            if artifact.role in ("qm_trajectory", "energies"):
                update[artifact.role] = artifact.path
        if run:
            trajectories = [a.path for a in run[0] if a.role == "trajectory"]
            update["trajectory"] = next((p for p in trajectories if p.endswith(".trr")),
                                        trajectories[0] if trajectories else None)
        return self.model_copy(update=update)


@tool(
    "snapshot_select",
    description="Pick starting configurations from a stage's run or a trajectory: the settled "
    "frame nearest a temperature, evenly spaced frames or indices, or a pair displaced to target "
    "distances. A coupled stage's come as a matched GEOMETRY and .gro.",
    tags={"analysis"},
    contract=Contract(consumes={"structure": ("structure", "tpr"), "trajectory": "trajectory",
                                "coords": "structure", "qm_trajectory": "qm_trajectory",
                                "energies": "energies"}),
)
def snapshot_select(ctx: ToolContext, params: SnapshotSelectParams) -> Observation:
    if params.targets_a:
        if not (params.coords and params.pair):
            return Observation(ok=False, summary="displacing needs coords, a pair and targets_a")
        return displaced_structures(ctx, params)
    if params.stage and not params.trajectory:
        return Observation(ok=False, summary=f"no collected run of stage {params.stage} wrote a "
                           "trajectory: run and collect it first")
    if not params.trajectory:
        return Observation(ok=False, summary="name the stage whose run the frames come from, or "
                           "a trajectory")
    if not params.structure:
        return Observation(ok=False, summary="name the structure defining the atom order")
    if params.qm_trajectory and params.energies:
        cutter = next(a.cut_snapshots for a in adapters() if a.cut_snapshots)
        obs = cutter(ctx, params)
    elif params.target_temperature_k is not None:
        return Observation(ok=False, summary="a target temperature picks from a coupled run's "
                           "own temperatures, and this run wrote none")
    else:
        obs = picked_frames(ctx, params)
    if obs.ok:  # the record a report reads of what was picked
        record = ctx.workspace.stage(params.output_stage) / "snapshots.json"
        record.write_text(json.dumps(obs.frames, indent=1) + "\n")
        obs.artifacts = [*obs.artifacts, ArtifactRef.from_path(record, "snapshot",
                                                               root=ctx.workspace.root)]
    return obs
