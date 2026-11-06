"""grompp (validation-gated) and mdrun (async submission).

gmx_grompp is the last gate before dynamics: the MDP sanity rules run first, a
blocking finding refuses and an advisory is recorded. gmx_mdrun then
submits through the execution backend and returns immediately — poll with
job_status, which collects the results once the job has ended."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Self

from pydantic import BaseModel, Field, PrivateAttr

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import PREVIEW_HELP, ToolContext, precheck, tool
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.gromacs.common import run_gmx, stage_output, tail
from myrmidon.engines.gromacs.mdp import parse_mdp_file, validate_mdp
from myrmidon.engines.gromacs.run import grompp, minimisation_warning, throughput_warning
from myrmidon.engines.inputs import require_input, resolve_in_ws
from myrmidon.execution.backend import JobSpec, Resources
from myrmidon.execution.jobs import submit_job
from myrmidon.knowledge.catalog import for_engine


class GromppParams(BaseModel):
    mdp: str = Field(description="run-parameter file (.mdp)")
    structure: str = Field(description="input coordinates (.gro)")
    topology: str = Field(description="topology (.top)")
    index: str | None = Field(
        default=None, description="index file (.ndx) for group references, e.g. QMMM-grps"
    )
    restraint_reference: str | None = Field(
        default=None, description="coordinates position restraints hold atoms at (-r); "
        "when the mdp's define switches restraints on, the structure unless given"
    )
    output: str = Field(default="topol.tpr")
    stage: str = Field(default="run", description="workspace stage to write into")
    max_warn: int = Field(default=0, description="passed to grompp -maxwarn")


def grompp_gate(ctx: ToolContext, params: GromppParams):
    warnings: list[Finding] = []
    checked: list[str] = ["input_files"]
    inputs = [("mdp", params.mdp), ("structure", params.structure), ("topology", params.topology)]
    if params.index:
        inputs.append(("index", params.index))
    if params.restraint_reference:
        inputs.append(("restraint_reference", params.restraint_reference))
    for label, path in inputs:
        _, missing = require_input(ctx, label, path)
        if missing:
            warnings.append(missing)
    checked.append("parameter_source")
    if ctx.workspace is not None:
        from myrmidon.core.sourcing import parameter_source, sourced_finding, unsourced_finding

        sources = [parameter_source(ctx.workspace, params.mdp)]
        for finding in (unsourced_finding(sources), sourced_finding(sources)):
            if finding is not None:
                warnings.append(finding)
        # and what the user changed in it since it was rendered
        from myrmidon.engines.file_origin import edit_findings

        warnings += edit_findings(ctx, params.mdp)

    checked.append("maxwarn")
    if params.max_warn > 0:
        warnings.append(
            Finding(
                id="GROMPP_MAXWARN",
                message=f"max_warn={params.max_warn} suppresses grompp's own WARNINGs "
                "(net charge under PME, mismatched constraints, ...). Each suppressed "
                "warning is a physics decision: the only routine case is the scaffold .tpr "
                "built for gmx_genion, which neutralizes the system in the next step. "
                "Never carry it into the .tpr that runs dynamics — fix the cause instead.",
            )
        )
    if any(w.id == "INPUT_MISSING_MDP" for w in warnings):
        return warnings, checked
    mdp_params = parse_mdp_file(resolve_in_ws(ctx, params.mdp))
    mdp_warnings, mdp_checked = validate_mdp(mdp_params, for_engine(ctx.engines, "gromacs"))
    return warnings + mdp_warnings, checked + mdp_checked


#: a run is carried on from its checkpoint by checkpoint_continue, which appends to the
#: run's own files; how much further is `extend_ps`
#: what a run leaves under its deffnm when the job is collected: the log and the final
#: structure always, a checkpoint always (mdrun writes one at the end), energies and
#: frames only as the tpr's mdp asked
RUN_OUTPUTS = (
    Produces("log", collected=True),
    Produces("structure", collected=True),
    Produces("checkpoint", collected=True),
    Produces("energy", when="tpr.energies", collected=True),
    Produces("trajectory", when="tpr.frames", collected=True),
)


@tool(
    "gmx_grompp",
    description="Preprocess mdp+structure+topology into a portable run input (.tpr). "
    "Runs the MDP validation gate first: a blocking finding refuses, an advisory is "
    "recorded with the call. GROMACS' own notes are "
    "re-parsed into structured warnings.",
    tags={"gromacs", "run"},
    requires={"gromacs"},
    expensive=True,
    # the tpr carries what its mdp decided about the run's outputs
    contract=Contract(consumes={"mdp": "mdp", "structure": "structure",
                                "topology": "topology", "index": "index",
                                "restraint_reference": "structure"},
                      produces=(Produces("tpr"),)),
)
def gmx_grompp(ctx: ToolContext, params: GromppParams) -> Observation:
    precheck(ctx, grompp_gate(ctx, params)[0])
    stage = ctx.workspace.stage(params.stage)
    return grompp(
        ctx, resolve_in_ws(ctx, params.mdp), resolve_in_ws(ctx, params.structure),
        resolve_in_ws(ctx, params.topology),
        resolve_in_ws(ctx, params.index) if params.index else None,
        resolve_in_ws(ctx, params.restraint_reference) if params.restraint_reference else None,
        stage_output(stage, params.output), stage, params.max_warn,
        mdp_name=params.mdp, structure_name=params.structure)


class SubmitParams(BaseModel):
    """What every dynamics submission must state: the resources it books, so
    the core-hour estimate exists before the job does."""

    #: the fields this submission's cost is computed from
    RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("ntasks", "cpus_per_task", "walltime_min")

    preview: bool = Field(default=False, description=PREVIEW_HELP)

    ntasks: int | None = Field(
        default=None, ge=1, description="MPI ranks to book; unset, the ranks one node of the "
        "partition holds where it bills whole nodes (the cores the booking pays for), else 1")
    cpus_per_task: int = Field(default=1, ge=1)
    walltime_min: int = Field(default=30, ge=1, description="scheduler time limit")
    account: str | None = Field(
        default=None,
        description=
        "scheduler account to bill for THIS submission instead of the machine config's "
        "default — the user may hold several project allocations",
    )

    def resources(self) -> Resources:
        return Resources(
            ntasks=self.ntasks or 1, cpus_per_task=self.cpus_per_task,
            walltime_min=self.walltime_min, account=self.account,
        )

    def node_cores(self, ctx: ToolContext) -> int | None:
        """The cores one node of the partition holds when it bills whole nodes: what a
        booking pays for whatever the run uses; None elsewhere."""
        _, node = ctx.config.compute.billing(None)
        return node.cores_per_node if node.node_exclusive and node.cores_per_node > 1 else None

    def sized(self, ctx: ToolContext) -> Self:
        """The call with its ranks filled when unset: on a whole-node partition, the node's
        cores as MPI ranks, so the run uses what the booking pays for; else one."""
        if self.ntasks is not None:
            return self
        cores = self.node_cores(ctx)
        return self.model_copy(update={
            "ntasks": max(1, cores // self.cpus_per_task) if cores else 1})


class MdrunParams(SubmitParams):
    tpr: str = Field(description="run input (.tpr) from gmx_grompp; the run's outputs are "
                     "written beside it, so a run and what it ran on stay together")
    deffnm: str = Field(default="md", description="output file prefix")
    nthreads: int | None = Field(default=None, description="gmx mdrun -nt")
    walltime_min: int | None = Field(
        default=None, ge=1, description="scheduler time limit; unset, sized from the run "
        "in the .tpr: a minimisation's fixed walltime, a dynamics run's length at the pace "
        "measured on this machine")
    #: how the walltime was sized, when the call left it to the harness
    _sizing: str = PrivateAttr(default="")

    def resources(self) -> Resources:
        return Resources(ntasks=self.ntasks or 1, cpus_per_task=self.cpus_per_task,
                         walltime_min=self.walltime_min or FALLBACK_WALLTIME_MIN,
                         account=self.account)

    def sized(self, ctx: ToolContext) -> MdrunParams:
        """The call with its ranks filled (SubmitParams.sized) and its walltime sized from
        the run its .tpr carries, when unset."""
        ranked = super().sized(ctx)
        filled = ""
        if self.ntasks is None and ranked.ntasks != 1:
            try:
                tpr = resolve_in_ws(ctx, ranked.tpr)
                integrator = tpr_facts(ctx, tpr).get("integrator") if tpr.is_file() else None
            except Exception:  # noqa: BLE001 — read again, and said, by the walltime sizing
                integrator = None
            if integrator == "l-bfgs":
                # L-BFGS runs on one rank only: the node's cores go to threads
                ranked = ranked.model_copy(update={
                    "ntasks": 1, "cpus_per_task": ranked.ntasks * ranked.cpus_per_task})
                filled = f"one rank of {ranked.cpus_per_task} threads (L-BFGS runs on one)"
            else:
                filled = (f"{ranked.ntasks} ranks, the node the partition bills whole; "
                          "a small box may take fewer")
        if ranked.walltime_min is not None:
            ranked._sizing = f"ntasks filled: {filled}" if filled else ""
            return ranked
        try:
            tpr = resolve_in_ws(ctx, ranked.tpr)
            minutes, why = (mdrun_walltime(ctx, tpr, (ranked.ntasks or 1) * ranked.cpus_per_task)
                            if tpr.is_file() else (None, "no .tpr to read the run from"))
        except Exception as exc:  # noqa: BLE001 — the booking is the default, and says why
            minutes, why = None, f"the .tpr could not be read ({exc})"
        out = ranked.model_copy(update={"walltime_min": minutes or FALLBACK_WALLTIME_MIN})
        why = why or "the .tpr states no length"
        out._sizing = (f"walltime sized to {minutes} min: {why}" if minutes else
                       f"walltime {FALLBACK_WALLTIME_MIN} min, not sized: {why}") + (
                           f"; ntasks filled: {filled}" if filled else "")
        return out


from myrmidon.engines.gromacs.run import (  # noqa: E402
    EM_INTEGRATORS,
    FALLBACK_WALLTIME_MIN,
    beside,
    checkpoint_output_names,
    mdrun_walltime,
    tpr_facts,
    tpr_run_length_ns,
)


def mdrun_gate(ctx: ToolContext, params: MdrunParams):
    warnings: list[Finding] = []
    checked = ["input_files"]
    tpr, missing = require_input(ctx, "tpr", params.tpr, message=f"tpr not found: {params.tpr}")
    if missing:
        warnings.append(missing)
    if tpr is not None:
        fields = tpr_facts(ctx, tpr)
        if fields.get("integrator") in EM_INTEGRATORS:
            checked.append("walltime_covers_run")
            try:
                steps, atoms = int(fields["nsteps"]), int(fields["natoms"])
            except (KeyError, ValueError):
                steps = atoms = 0
            if steps > 0 and atoms > 0:
                warnings.append(
                    minimisation_warning(steps, atoms, params.resources().walltime_min, ctx)
                )
        else:
            length = tpr_run_length_ns(ctx, tpr)
            if length is not None:
                checked.append("walltime_covers_run")
                warnings.append(throughput_warning(length, params.resources().walltime_min,
                                                   ctx))
    return warnings, checked


def _submit(ctx: ToolContext, params: SubmitParams, argv: list[str], stage, label: str,
            stem: str) -> tuple[str, float]:
    spec = JobSpec(
        argv=argv, cwd=stage, stdout=f"{stem}.stdout", stderr=f"{stem}.stderr",
        resources=params.resources(),
    )
    return submit_job(ctx, spec, kind="gromacs", label=label)


@tool(
    "gmx_mdrun",
    description="Run dynamics through the execution backend; returns a job_id at once (job_status "
                "with wait_seconds waits and collects). Books ntasks x cpus_per_task x "
                "walltime_min under the approved mandate; checks the walltime covers the run "
                "length in the .tpr.",
    tags={"gromacs", "run"},
    requires={"gromacs"},
    expensive=True,
    contract=Contract(consumes={"tpr": "tpr"}, produces=RUN_OUTPUTS),
)
def gmx_mdrun(ctx: ToolContext, params: MdrunParams) -> Observation:
    precheck(ctx, mdrun_gate(ctx, params)[0])
    from myrmidon.engines.gromacs.common import gmx_bin

    tpr = resolve_in_ws(ctx, params.tpr)
    stage = beside(ctx, tpr)
    argv = [gmx_bin(ctx), "mdrun", "-s", str(tpr), "-deffnm", params.deffnm]
    if params.nthreads is not None:
        argv += ["-nt", str(params.nthreads)]
    job_id, est = _submit(ctx, params, argv, stage, f"mdrun:{params.deffnm}", params.deffnm)
    return Observation(
        ok=True,
        summary=f"mdrun submitted via {ctx.backend.name} backend as {job_id} "
        f"(outputs {params.deffnm}.* in {ctx.workspace.relpath(stage)}; "
        f"booked {est} core-h" + (f"; {params._sizing}" if params._sizing else "")
        + "). Poll with job_status.",
        command=argv,
        artifacts=[ArtifactRef.from_path(tpr, "tpr", root=ctx.workspace.root)],
        job_id=job_id,
        state="QUEUED",
        core_h_estimate=est,
    )


class CheckpointContinueParams(SubmitParams):
    tpr: str = Field(description="the .tpr of the finished run to extend")
    checkpoint: str = Field(description="its checkpoint file (.cpt)")
    extend_ps: float = Field(description="how many ps to extend the run by", gt=0)
    deffnm: str = Field(default="md", description="output prefix (match the parent run); "
                        "the continuation writes beside the checkpoint, as the parent did")
    nthreads: int | None = None


def continue_gate(ctx: ToolContext, params: CheckpointContinueParams):
    warnings: list[Finding] = []
    checked = ["input_files", "checkpoint_names"]
    _, missing_tpr = require_input(ctx, "tpr", params.tpr,
                                   message=f"tpr not found: {params.tpr}")
    cpt, missing_cpt = require_input(ctx, "checkpoint", params.checkpoint,
                                     message=f"checkpoint not found: {params.checkpoint}")
    warnings += [w for w in (missing_tpr, missing_cpt) if w]
    if cpt is not None:
        names = checkpoint_output_names(ctx, cpt)
        stage = beside(ctx, cpt)
        mismatched = [
            n for n in names
            if Path(n).stem != params.deffnm
            or (Path(n).is_absolute() and Path(n).parent != stage)
        ]
        if not names:
            warnings.append(Finding(
                id="CPT_UNREADABLE",
                message=f"could not read output names from {params.checkpoint} (gmx dump -cp); "
                "cannot verify that -deffnm matches the parent run",
            ))
        elif mismatched:
            warnings.append(Precondition(
                id="CPT_NAME_MISMATCH",
                message=f"the checkpoint records outputs {names[:4]} but this continuation "
                f"would write {params.deffnm}.* in {ctx.workspace.relpath(stage)}. mdrun "
                "-append compares these names verbatim (absolute stays absolute) and aborts; "
                "dropping -cpi instead would restart from the .tpr and silently NOT continue. "
                "Pass deffnm exactly as the parent run used it.",
            ))
    checked.append("walltime_covers_run")
    return warnings + [
        throughput_warning(params.extend_ps / 1000.0, params.walltime_min, ctx)
    ], checked


@tool(
    "checkpoint_continue",
    description="Extend a finished run from its checkpoint (convert-tpr adds time; mdrun -append "
                "continues from the .cpt beside which it writes, linked to the parent run). "
                "Checks the checkpoint's output names match deffnm. Books a job; returns its "
                "job_id.",
    tags={"gromacs", "run"},
    requires={"gromacs"},
    expensive=True,
    contract=Contract(consumes={"tpr": "tpr", "checkpoint": "checkpoint"},
                      produces=(Produces("tpr"), *RUN_OUTPUTS)),
)
def checkpoint_continue(ctx: ToolContext, params: CheckpointContinueParams) -> Observation:
    precheck(ctx, continue_gate(ctx, params)[0])
    from myrmidon.engines.gromacs.common import gmx_bin

    tpr = resolve_in_ws(ctx, params.tpr)
    cpt = resolve_in_ws(ctx, params.checkpoint)
    stage = beside(ctx, cpt)
    extended = stage / f"{params.deffnm}_ext.tpr"
    conv = run_gmx(
        ctx,
        ["convert-tpr", "-s", str(tpr), "-extend", str(params.extend_ps),
         "-o", str(extended)],
        cwd=stage,
    )
    if conv.returncode != 0 or not extended.exists():
        return Observation(
            ok=False,
            summary=f"convert-tpr failed (exit {conv.returncode})",
            command=list(conv.args),
            log_tail=tail(conv.stderr),
            job_id="",
            state="FAILED",
        )
    argv = [
        gmx_bin(ctx), "mdrun", "-s", str(extended), "-cpi", str(cpt),
        "-deffnm", params.deffnm, "-append",
    ]
    if params.nthreads is not None:
        argv += ["-nt", str(params.nthreads)]
    job_id, est = _submit(
        ctx, params, argv, stage, f"continue:{params.deffnm}", f"{params.deffnm}_ext",
    )
    return Observation(
        ok=True,
        summary=f"Continuation submitted as {job_id}: +{params.extend_ps} ps from "
        f"{ctx.workspace.relpath(cpt)} (booked {est} core-h). Poll with job_status.",
        command=argv,
        artifacts=[ArtifactRef.from_path(extended, "tpr", root=ctx.workspace.root)],
        job_id=job_id,
        state="QUEUED",
        core_h_estimate=est,
    )


class RerunParams(SubmitParams):
    tpr: str = Field(description="run input from gmx_grompp with the topology whose energies "
                     "and forces are wanted (nstfout = 1 in its mdp to write forces)")
    trajectory: str = Field(description="the frames to evaluate (.trr/.xtc), e.g. from "
                            "snapshot_select with output='trajectory'")
    deffnm: str = Field(default="rerun", description="output file prefix")
    stage: str = Field(default="rerun")
    nthreads: int | None = Field(default=None, description="gmx mdrun -nt")


def rerun_gate(ctx: ToolContext, params: RerunParams):
    warnings = [w for w in (require_input(ctx, "tpr", params.tpr)[1],
                            require_input(ctx, "trajectory", params.trajectory)[1]) if w]
    checked = ["input_files"]
    if not warnings:
        checked.append("classical_input")
        integrator = tpr_facts(ctx, resolve_in_ws(ctx, params.tpr)).get("integrator")
        if integrator == "mimic":
            warnings.append(Precondition(
                id="RERUN_TPR_COUPLED",
                message=f"{params.tpr} is a MiMiC run input: a classical rerun evaluates the "
                "MM side alone and drops the QM region's electrostatics",
                suggestion="build the rerun's tpr from a classical mdp (a single_point "
                "stage with regime mm, for a force comparison)", needs=["tpr"]))
    return warnings, checked


@tool(
    "gmx_rerun",
    description="Re-evaluate existing frames under a run input (mdrun -rerun): each frame's "
                "energies, and forces when the mdp sets nstfout, without integrating. Books a job;"
                " job_status collects the log, energies and forces. The basis of a force-field "
                "comparison.",
    tags={"gromacs", "run"},
    requires={"gromacs"},
    expensive=True,
    contract=Contract(consumes={"tpr": "tpr", "trajectory": "trajectory"},
                      produces=(Produces("log", collected=True),
                                Produces("energy", when="tpr.energies",
                                         collected=True),
                                Produces("trajectory", when="tpr.forces",
                                         collected=True))),
)
def gmx_rerun(ctx: ToolContext, params: RerunParams) -> Observation:
    precheck(ctx, rerun_gate(ctx, params)[0])
    from myrmidon.engines.gromacs.common import gmx_bin

    stage = ctx.workspace.stage(params.stage)
    tpr = resolve_in_ws(ctx, params.tpr)
    frames = resolve_in_ws(ctx, params.trajectory)
    d = params.deffnm
    argv = [gmx_bin(ctx), "mdrun", "-s", str(tpr), "-rerun", str(frames), "-deffnm", d,
            "-o", f"{d}.trr", "-e", f"{d}.edr", "-g", f"{d}.log"]
    if params.nthreads is not None:
        argv += ["-nt", str(params.nthreads)]
    job_id, est = _submit(ctx, params, argv, stage, f"rerun:{d}", d)
    root = ctx.workspace.root
    return Observation(
        ok=True,
        summary=f"rerun submitted via {ctx.backend.name} backend as {job_id} (outputs {d}.* "
        f"in {ctx.workspace.relpath(stage)}; booked {est} core-h). Poll with job_status.",
        command=argv,
        artifacts=[ArtifactRef.from_path(tpr, "tpr", root=root)],
        job_id=job_id,
        state="QUEUED",
        core_h_estimate=est,
    )

