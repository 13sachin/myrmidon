"""MiMiC coupled launch: CPMD (QM) + GROMACS (MM) started together.

mimic_run is validation-gated (the QM/MM analogue of gmx_grompp's gate) and
fully async. CPMD is the main program (its stdout is the QM log); GROMACS
runs as the coupled partner."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field, PrivateAttr

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import PREVIEW_HELP, ToolContext, precheck, tool
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.mimic.collect import coupled_findings, coupled_geometry
from myrmidon.engines.mimic.cpmd_input import (
    keyword_value,
    parse_cpmd,
    qm_geometry,
)
from myrmidon.engines.mimic.launch import (
    align_mimic_paths,
    stage_geometry,
    stage_restart,
)
from myrmidon.execution.backend import CoupledCmd, JobSpec, Resources
from myrmidon.execution.jobs import submit_job

#: the walltime a coupled run is booked for when nothing sizes it
FALLBACK_WALLTIME_MIN = 60


class MimicRunParams(BaseModel):
    #: the fields this submission's cost is computed from
    RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("cpmd_ranks", "gromacs_ranks", "cpus_per_task",
                                                  "walltime_min")

    preview: bool = Field(default=False, description=PREVIEW_HELP)

    cpmd_input: str = Field(description="reviewed run input from mimic_prepare")
    tpr: str = Field(description="GROMACS .tpr built for MiMiC (qmmm-active mdp)")
    restart_from: str | None = Field(
        default=None,
        description="the previous segment's CPMD RESTART file, to continue from it "
        "(the `restart` artifact of the earlier mimic_run). Required whenever the input "
        "asks to RESTART: CPMD reads the wavefunction from the file LATEST names in the "
        "working directory, and stops in ZHRWF with FILE NOT FOUND when it is absent.",
    )
    geometry_from: str | None = Field(
        default=None,
        description="a CPMD GEOMETRY file of this box to read the ions' coordinates from, "
        "needed when the input's RESTART line has GEOFILE (the `geometry` artifact of an "
        "earlier mimic_run: the previous segment's, or a snapshot). Default: the GEOMETRY "
        "beside restart_from, which the previous segment wrote at its end.",
    )
    cpmd_ranks: int | None = Field(
        default=None, ge=1, description="MPI ranks for CPMD (QM); unset, the node's cores "
        "the GROMACS ranks leave on a partition that bills whole nodes, else 4")
    gromacs_ranks: int | None = Field(
        default=None, ge=1, description="MPI ranks for GROMACS (MM); unset, one in six of "
        "the node's cores on a partition that bills whole nodes, else 1")
    cpus_per_task: int = Field(default=1, ge=1, description="OpenMP threads per rank of both "
                               "programs (OMP_NUM_THREADS)")
    deffnm: str = Field(default="mimic")
    walltime_min: int | None = Field(
        default=None, ge=1, description="batch walltime (minutes); unset, sized from the "
        "input's MAXSTEP at the seconds per step measured here, else 60")
    partition: str | None = Field(
        default=None,
        description="scheduler partition for THIS run (default: the machine config's). "
        "Large QM cells need the large-memory partition: a 100 Ry cell on 94 GB nodes "
        "loses its CPMD ranks right after the plane-wave setup",
    )
    account: str | None = Field(
        default=None,
        description=
        "scheduler account to bill for THIS submission instead of the machine config's "
        "default — the user may hold several project allocations",
    )

    def resources(self) -> Resources:
        """What this run books: both programs' ranks for the whole walltime. The study
        approval prices a node through this method; without it a chain of MiMiC runs
        would be presented to the human as 0.0 core-h."""
        return Resources(
            ntasks=(self.cpmd_ranks or 4) + (self.gromacs_ranks or 1),
            cpus_per_task=self.cpus_per_task,
            walltime_min=self.walltime_min or FALLBACK_WALLTIME_MIN,
            partition=self.partition,
            account=self.account,
        )

    stage: str = Field(
        default="qmmm",
        description="the stage this run belongs to. Every submission executes in its own "
        "numbered directory under it (01_<deffnm>, 02_<deffnm>, …), allocated by the "
        "harness and never reused: CPMD writes its log, RESTART.1, LATEST, GEOMETRY and "
        "ENERGIES under fixed names, so two runs in one directory would overwrite each "
        "other. The run's directory is in the submission summary and in every artifact "
        "path job_status returns.",
    )

    #: how the walltime was sized, when the call left it to the harness
    _sizing: str = PrivateAttr(default="")

    def sized(self, ctx: ToolContext) -> MimicRunParams:
        """The call with its walltime sized from the input's MAXSTEP at the seconds per
        coupled step measured here on the same QM region size and cutoff; the default
        otherwise, said."""
        from myrmidon.execution.pace import read_paces, study_paces
        from myrmidon.knowledge.stage import booked_minutes, qm_walltime

        ranked, filled = self.filled(ctx)
        if ranked is not self:
            return ranked.sized(ctx)._said(filled)
        if self.walltime_min is not None:
            return self
        try:
            sections = parse_cpmd(resolve_in_ws(ctx, self.cpmd_input).read_text())
            steps = keyword_value(sections, "&CPMD/MAXSTEP")
            cutoff = keyword_value(sections, "&SYSTEM/CUTOFF")
            minutes, why = (None, "the input states no MAXSTEP") if steps is None else \
                qm_walltime(
                    steps=int(float(steps)),
                    qm_atoms=(qm_geometry(sections) or {}).get("n_atoms"),
                    cutoff_ry=float(cutoff.split()[0]) if cutoff else None,
                    cores=(self.cpmd_ranks + self.gromacs_ranks) * self.cpus_per_task,
                    paces=[*study_paces(ctx.workspace), *read_paces(ctx.config.studies_root)],
                    compute=ctx.config.compute)
        except Exception as exc:  # noqa: BLE001 — the booking is the default, and says why
            minutes, why = None, f"the input could not be read ({exc})"
        booked = booked_minutes(minutes) if minutes is not None else FALLBACK_WALLTIME_MIN
        _, node = ctx.config.compute.billing(self.partition)
        if node.max_walltime_min and booked > node.max_walltime_min:
            booked = node.max_walltime_min
            why += f"; capped at the {booked} min the partition grants"
        out = self.model_copy(update={"walltime_min": booked})
        out._sizing = (f"walltime sized to {booked} min: {why}" if minutes is not None
                       else f"walltime {booked} min, not sized: {why}")
        return out

    def filled(self, ctx: ToolContext) -> tuple[MimicRunParams, str]:
        """The call with the ranks it leaves unset filled, and how: on a partition that
        bills whole nodes, the node's cores, one in six to GROMACS and the rest to CPMD,
        so the run uses what the booking pays for; else CPMD 4, GROMACS 1."""
        if self.cpmd_ranks is not None and self.gromacs_ranks is not None:
            return self, ""
        _, node = ctx.config.compute.billing(self.partition)
        cores = node.cores_per_node if node.node_exclusive and node.cores_per_node > 1 else 0
        mm = self.gromacs_ranks or (max(1, cores // 6) if cores else 1)
        qm = self.cpmd_ranks or (max(1, cores - mm) if cores else 4)
        said = (f"ranks filled: CPMD {qm} + GROMACS {mm}, the {cores}-core node the "
                "partition bills whole" if cores else "")
        return self.model_copy(update={"cpmd_ranks": qm, "gromacs_ranks": mm}), said

    def _said(self, filled: str) -> MimicRunParams:
        """This call, its sizing note followed by how its ranks were filled."""
        self._sizing = "; ".join(x for x in (self._sizing, filled) if x)
        return self


def mimic_gate(ctx: ToolContext, params: MimicRunParams):
    return coupled_findings(ctx, params.cpmd_input, params.tpr, params.restart_from,
                            params.geometry_from)


def geometry_for(ctx: ToolContext, params: MimicRunParams):
    """The GEOMETRY file a ``RESTART ... GEOFILE`` run reads its coordinates from."""
    return coupled_geometry(ctx, params.geometry_from, params.restart_from)


@tool(
    "mimic_run",
    description="Launch a MiMiC QM/MM run: CPMD and GROMACS started together as "
    "coupled MPI programs. Validation-gated: a blocking finding refuses. Async — poll "
    "with job_status.",
    # the QM side's files are fixed names in the run directory; the MM side's are the
    # GROMACS run's under deffnm, as its tpr's mdp asked; force-matching reference data
    # only when the CPMD input asks for it
    contract=Contract(
        consumes={"cpmd_input": "cpmd_input", "tpr": "tpr", "restart_from": "restart",
                  "geometry_from": "geometry"},
        produces=(
            Produces("qm_log", collected=True), Produces("restart", collected=True),
            Produces("geometry", collected=True), Produces("qm_trajectory", collected=True),
            Produces("energies", collected=True),
            Produces("fm_reference", when="cpmd_input.fm_reference", collected=True),
            Produces("log", collected=True),
            Produces("structure", collected=True),
            Produces("checkpoint", collected=True),
            Produces("energy", when="tpr.energies", collected=True),
            Produces("trajectory", when="tpr.frames", collected=True),
        )),
    tags={"mimic", "qmmm", "run"},
    requires={"mimicpy", "cpmd", "gromacs"},
    expensive=True,
)
def mimic_run(ctx: ToolContext, params: MimicRunParams) -> Observation:
    precheck(ctx, mimic_gate(ctx, params)[0])
    params = params.filled(ctx)[0]  # a call the sizing could not reach still runs ranked
    from myrmidon.engines.gromacs.common import gmx_bin

    # one submission, one directory, never reused: CPMD's output names are fixed, so a
    # second run in the same directory erases the first
    stage = ctx.workspace.run_dir(ctx.workspace.stage(params.stage), params.deffnm)
    inp = resolve_in_ws(ctx, params.cpmd_input)
    tpr = resolve_in_ws(ctx, params.tpr)
    consumed = [ArtifactRef.from_path(inp, "cpmd_input", root=ctx.workspace.root)]
    inp, aligned = align_mimic_paths(inp, stage)
    staged = []  # what the node reads besides the input and the tpr, as staged for it
    if params.restart_from:
        restart = resolve_in_ws(ctx, params.restart_from)
        consumed.append(
            ArtifactRef.from_path(restart, "restart", root=ctx.workspace.root)
        )
        staged.append(stage_restart(restart, stage))
    sections = parse_cpmd(inp.read_text(errors="replace"))
    if any(line.strip().upper().startswith("RESTART") and "GEOFILE" in line.upper()
           for line in sections.get("CPMD", [])):
        geometry = geometry_for(ctx, params)  # the gate refused the run without one
        consumed.append(ArtifactRef.from_path(geometry, "geometry", root=ctx.workspace.root))
        staged.append(stage_geometry(geometry, stage))
    cpmd = ctx.engines["cpmd"].path
    spec = JobSpec(
        argv=[cpmd, str(inp)],
        cwd=stage,
        stdout="cpmd.out",
        stderr="cpmd.err",
        mpi_ranks=params.cpmd_ranks,
        env={"OMP_NUM_THREADS": str(params.cpus_per_task)},
        # what the node reads: the run input as aligned into the run directory, the tpr,
        # and the restart and geometry staged beside them
        receipt_inputs=[str(inp), str(tpr), *map(str, staged)],
        resources=params.resources(),
        coupled=[
            CoupledCmd(
                argv=[gmx_bin(ctx), "mdrun", "-s", str(tpr), "-deffnm", params.deffnm],
                mpi_ranks=params.gromacs_ranks,
                stdout=f"{params.deffnm}.stdout",
                stderr=f"{params.deffnm}.stderr",
            )
        ],
    )
    job_id, est = submit_job(
        ctx, spec, kind="mimic", label=f"mimic:{params.deffnm}",
        # the prefix the MM half writes under: its own argv is in the partner command,
        # not in spec.argv, so collection cannot find it otherwise
        meta={"deffnm": params.deffnm},
    )
    return Observation(
        ok=True,
        summary=f"MiMiC run submitted as {job_id}: CPMD x{params.cpmd_ranks} + "
        f"GROMACS x{params.gromacs_ranks} in {ctx.workspace.relpath(stage)} "
        f"(booked {est} core-h" + (f"; {params._sizing}" if params._sizing else "") + "). "
        + (f"{aligned}. " if aligned else "")
        + "QM log: cpmd.out. Poll with job_status: once it has ended it returns the "
        "run's outputs, the CPMD log and its diagnosis; mimic_diagnose reads a "
        "running one.",
        command=spec.argv,
        inputs=consumed,
        artifacts=[ArtifactRef.from_path(inp, "cpmd_input", root=ctx.workspace.root)],
        job_id=job_id,
        state="QUEUED",
        # what this segment costs, so a chain's cap_core_h has something to sum and the
        # study state records the spend per segment rather than None
        core_h_estimate=est,
    )
