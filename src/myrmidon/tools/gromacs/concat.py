"""traj_concat: several trajectories as one, in the order given."""

from __future__ import annotations

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.engines.gromacs.common import run_gmx, tail
from myrmidon.engines.inputs import resolve_in_ws


class TrajConcatParams(BaseModel):
    trajectories: list[str] = Field(min_length=1, description="trajectories (.trr/.xtc), in "
                                    "the order they are joined")
    output: str = Field(default="concatenated.trr", description="the joined trajectory, in "
                        "the stage")
    stage: str = Field(default="trajectories")


@tool(
    "traj_concat",
    description="Join trajectories into one with gmx trjcat -cat, in the order given, every "
    "frame kept (segments of a chain, the windows of a sweep).",
    tags={"gromacs"},
    requires={"gromacs"},
    contract=Contract(consumes={"trajectories": "trajectory"},
                      produces=(Produces("trajectory"),)),
)
def traj_concat(ctx: ToolContext, params: TrajConcatParams) -> Observation:
    stage = ctx.workspace.stage(params.stage)
    sources = [str(resolve_in_ws(ctx, t)) for t in params.trajectories]
    out = stage / params.output
    proc = run_gmx(ctx, ["trjcat", "-f", *sources, "-o", str(out), "-cat"], cwd=stage)
    if proc.returncode != 0 or not out.exists():
        return Observation(ok=False, summary=f"gmx trjcat failed (exit {proc.returncode})",
                           command=list(proc.args), log_tail=tail(proc.stderr, 30))
    return Observation(
        ok=True,
        summary=f"{len(sources)} trajectorie(s) joined into {ctx.workspace.relpath(out)}",
        command=list(proc.args),
        artifacts=[ArtifactRef.from_path(out, "trajectory", root=ctx.workspace.root)],
    )
