"""gmx_minimize_frames: frames of a trajectory, each minimised under one topology.

A minimised frame says where a force field puts the nearest minimum of a geometry the
reference sampled: the cheap test of whether it holds a structure. One job minimises
every frame asked for and collects the minimised coordinates into one trajectory, which
``observable_measure`` then reads like any other; a comparison of topologies is the same call
on each, over the same frames.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from pydantic import Field

from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, precheck, tool
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.gromacs.common import gmx_bin, stage_topology
from myrmidon.engines.gromacs.mdp import parse_mdp_file
from myrmidon.engines.inputs import require_input, resolve_in_ws
from myrmidon.execution.backend import JobSpec
from myrmidon.execution.jobs import submit_job
from myrmidon.tools.gromacs.run import EM_INTEGRATORS, SubmitParams


class MinimizeFramesParams(SubmitParams):
    topology: str = Field(description="the topology to minimise under (.top)")
    mdp: str = Field(description="a minimisation mdp (integrator steep, cg or l-bfgs), e.g. "
                     "inputset_apply of a minimize stage")
    structure: str = Field(description="the structure that defines atom order (.gro)")
    trajectory: str = Field(description="the trajectory the frames are taken from")
    frames: list[int] = Field(
        default_factory=list,
        description="frame INDICES (0-based) to minimise; empty: n_frames evenly spaced over "
        "the whole trajectory")
    n_frames: int = Field(default=5, ge=1, description="used when frames is empty")
    index: str | None = Field(default=None, description="index file grompp reads (-n)")
    max_warn: int = Field(default=0, ge=0, description="grompp -maxwarn")
    output: str = Field(default="minimized.trr", description="the trajectory of minimised "
                        "frames, in the stage")
    stage: str = Field(default="minimize")

    def sized(self, ctx: ToolContext) -> MinimizeFramesParams:
        """One driver process, which runs each frame's mdrun with -ntomp cpus_per_task: on a
        whole-node partition the node's cores go to its threads, never to copies of it."""
        if self.ntasks is not None:
            return self
        cores = self.node_cores(ctx)
        threads = (cores if cores and "cpus_per_task" not in self.model_fields_set
                   else self.cpus_per_task)
        return self.model_copy(update={"ntasks": 1, "cpus_per_task": threads})




def minimize_gate(ctx: ToolContext, params: MinimizeFramesParams):
    warnings: list[Finding] = []
    checked = ["input_files"]
    for name in ("topology", "mdp", "structure", "trajectory"):
        _, missing = require_input(ctx, name, getattr(params, name))
        if missing:
            warnings.append(missing)
    if params.index:
        _, missing = require_input(ctx, "index", params.index)
        if missing:
            warnings.append(missing)
    mdp, _ = require_input(ctx, "mdp", params.mdp)
    if mdp is not None:
        checked.append("minimiser")
        integrator = parse_mdp_file(mdp).get("integrator", "md")
        if integrator not in EM_INTEGRATORS:
            warnings.append(Precondition(
                id="MINIMIZE_MDP_NOT_MINIMISER",
                message=f"{params.mdp} sets integrator = {integrator}; minimising frames "
                "needs steep, cg or l-bfgs",
                suggestion="render a minimisation mdp, e.g. inputset_apply of a minimize "
                "stage", needs=["mdp"]))
    if params.max_warn > 0:
        warnings.append(Finding(
            id="GROMPP_MAXWARN",
            message=f"max_warn={params.max_warn} suppresses grompp's own WARNINGs for every "
            "frame's run input. Each suppressed warning is a physics decision: fix the "
            "cause instead."))
    return warnings, checked


def _frames(params: MinimizeFramesParams, n_total: int) -> list[int]:
    if params.frames:
        return list(params.frames)
    k = min(params.n_frames, n_total)
    return sorted({int(round(x)) for x in np.linspace(0, n_total - 1, k)})


@tool(
    "gmx_minimize_frames",
    description="Minimise trajectory frames under one topology in one job: each frame built with "
                "grompp from the minimisation mdp and minimised. job_status collects the minimised"
                " trajectory with each frame's convergence and energies; compare force fields with"
                " observable_measure pair_distance.",
    tags={"gromacs", "run"},
    requires={"gromacs"},
    expensive=True,
    contract=Contract(consumes={"topology": "topology", "mdp": "mdp",
                                "structure": ("structure", "tpr"), "trajectory": "trajectory",
                                "index": "index"},
                      produces=(Produces("trajectory", collected=True),
                                Produces("data", collected=True))),
)
def gmx_minimize_frames(ctx: ToolContext, params: MinimizeFramesParams
                        ) -> Observation:
    precheck(ctx, minimize_gate(ctx, params)[0])
    import MDAnalysis as mda

    structure = resolve_in_ws(ctx, params.structure)
    trajectory = resolve_in_ws(ctx, params.trajectory)
    n_total = len(mda.Universe(str(structure), str(trajectory)).trajectory)
    frames = _frames(params, n_total)
    bad = [f for f in frames if f < 0 or f >= n_total]
    if bad:
        return Observation(
            ok=False, summary=f"frame index out of range {bad}: the trajectory has {n_total} "
            "frames")
    stage = ctx.workspace.stage(params.stage)
    (stage / "top").mkdir(exist_ok=True)
    top = stage_topology(stage / "top", resolve_in_ws(ctx, params.topology))
    driver = {
        "stage": str(stage), "gmx": gmx_bin(ctx), "topology": str(top),
        "mdp": str(resolve_in_ws(ctx, params.mdp)), "structure": str(structure),
        "trajectory": str(trajectory), "frames": frames, "max_warn": params.max_warn,
        "index": str(resolve_in_ws(ctx, params.index)) if params.index else None,
        "ntomp": params.cpus_per_task, "output": params.output,
    }
    params_file = stage / "minimize_params.json"
    params_file.write_text(json.dumps(driver, indent=1) + "\n")
    spec = JobSpec(argv=[sys.executable, "-m", "myrmidon.engines.gromacs.batch",
                         str(params_file)],
                   cwd=stage, stdout="minimize.stdout", stderr="minimize.stderr",
                   resources=params.resources())
    job_id, est = submit_job(ctx, spec, kind="minimize_frames",
                             label=f"minimize:{len(frames)} frames")
    return Observation(
        ok=True,
        summary=f"minimisation of {len(frames)} frame(s) of {params.trajectory} under "
        f"{params.topology} submitted as {job_id} (booked {est} core-h). Poll with "
        "job_status: once it has ended it returns the minimised frames' trajectory and each "
        "frame's convergence.",
        command=spec.argv, job_id=job_id, state="QUEUED", core_h_estimate=est,
        n_requested=len(frames),
    )
