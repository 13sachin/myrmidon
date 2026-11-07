"""A stage's run as one trajectory: the segments a continued run wrote into directories of
their own, joined in order as a recorded call whose inputs are each segment's trajectory."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef, fresh_path
from myrmidon.core.contracts import Contract
from myrmidon.core.lineage import JOINS, stage_segments
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.engines.registry import adapter_with


def segment_trajectory(outputs) -> str | None:
    """The trajectory a segment wrote that a measurement reads: its compressed one first."""
    found = [a.path for a in outputs if a.role == "trajectory"]
    return next((p for p in found if p.endswith(".xtc")), found[0] if found else None)


class TrajectoryJoinParams(BaseModel):
    stage: str = Field(description="the stage whose latest run is joined, every segment of it "
                       "from its fresh start through each continuation")
    trajectories: list[str] = Field(default_factory=list, description="filled from the ledger: "
                                    "each segment's trajectory, oldest first")
    segments: list[str] = Field(default_factory=list, description="filled from the ledger: the "
                                "submission of each segment")

    def sized(self, ctx: ToolContext) -> TrajectoryJoinParams:
        """The call with the segments the ledger holds named, so it records each as an input."""
        found = [(sub.event_id, segment_trajectory(outputs))
                 for sub, outputs, _ in stage_segments(ctx.workspace, self.stage)]
        found = [(sub, path) for sub, path in found if path]
        return self.model_copy(update={"segments": [s for s, _ in found],
                                       "trajectories": [p for _, p in found]})


@tool(
    "trajectory_join",
    description="Join a stage's run into one trajectory: a continued run's segments one after "
    "another, every frame each wrote kept. observable_measure joins a stage it measures itself.",
    tags={"analysis"},
    contract=Contract(consumes={"trajectories": "trajectory"}),
)
def trajectory_join(ctx: ToolContext, params: TrajectoryJoinParams) -> Observation:
    refs = [f"stage:{params.stage}", *(f"{JOINS}{s}" for s in params.segments)]
    paths = list(dict.fromkeys(params.trajectories))
    if not paths:
        return Observation(ok=False, summary=f"no collected run of stage {params.stage} wrote a "
                           "trajectory")
    if len(paths) == 1:  # a run appended to one file holds every segment in it
        return Observation(ok=True, summary=f"{paths[0]} holds every segment of stage "
                           f"{params.stage}", trajectory=paths[0], segments=len(params.segments))
    joiner = adapter_with("join_trajectories")
    if joiner is None:
        return Observation(ok=False, summary="no engine here joins trajectories")
    files = [ctx.workspace.resolve(p) for p in paths]
    out = fresh_path(ctx.workspace.stage(params.stage) / f"{params.stage}_joined"
                     f"{Path(paths[-1]).suffix}")
    proc = joiner.join_trajectories(ctx, files, out)
    if proc.returncode != 0 or not out.is_file():
        return Observation(ok=False, summary=f"joining stage {params.stage}'s {len(paths)} "
                           f"segments failed (exit {proc.returncode})", command=list(proc.args),
                           log_tail=(proc.stderr or "")[-2000:])
    joined = ctx.workspace.relpath(out)
    return Observation(
        ok=True, summary=f"stage {params.stage}: {len(paths)} segments joined into {joined}",
        command=list(proc.args), trajectory=joined, segments=len(paths), refs=refs,
        artifacts=[ArtifactRef.from_path(out, "trajectory", root=ctx.workspace.root)])
