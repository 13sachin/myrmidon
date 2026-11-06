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
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.gromacs.collect import collect_run, spec_deffnm
from myrmidon.engines.inputs import require_input, resolve_in_ws
from myrmidon.engines.mimic.cpmd_input import (
    cpmd_findings,
    get_keyword,
    keyword_value,
    parse_cpmd,
    qm_geometry,
)
from myrmidon.engines.mimic.launch import (
    GEOMETRY_NAME,
    align_mimic_paths,
    stage_geometry,
    stage_restart,
)
from myrmidon.engines.mimic.system import pp_findings
from myrmidon.execution.backend import CoupledCmd, JobSpec, Resources
from myrmidon.execution.collect import collector, progress
from myrmidon.execution.jobs import submit_job
from myrmidon.knowledge.catalog import for_engine

#: the walltime a coupled run is booked for when nothing sizes it
FALLBACK_WALLTIME_MIN = 60


class MimicRunParams(BaseModel):
    #: the fields this submission's cost is computed from
    RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("cpmd_ranks", "gromacs_ranks", "walltime_min")

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
                    cores=self.cpmd_ranks + self.gromacs_ranks,
                    paces=[*study_paces(ctx.workspace), *read_paces(ctx.config.studies_root)])
        except Exception as exc:  # noqa: BLE001 — the booking is the default, and says why
            minutes, why = None, f"the input could not be read ({exc})"
        booked = booked_minutes(minutes) if minutes is not None else FALLBACK_WALLTIME_MIN
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
    warnings: list[Finding] = []
    checked = ["input_files", "cpmd_engine", "cpmd_sections", "charge_multiplicity", "qm_cell"]

    cpmd_info = ctx.engines.get("cpmd")
    if cpmd_info is None or not cpmd_info.available:
        warnings.append(
            Precondition(
                id="CPMD_UNAVAILABLE",
                message="CPMD is not available on this machine"
                + (f" — {cpmd_info.hint}" if cpmd_info else ""),
            )
        )

    inp, missing = require_input(ctx, "cpmd_input", params.cpmd_input, "INPUT_MISSING_CPMD",
                                 f"cpmd input not found: {params.cpmd_input}")
    if missing:
        warnings.append(missing)
        return warnings, checked
    _, missing = require_input(ctx, "tpr", params.tpr, message=f"tpr not found: {params.tpr}")
    if missing:
        warnings.append(missing)

    sections = parse_cpmd(inp.read_text())
    warnings.extend(cpmd_findings(sections, for_engine(ctx.engines, "cpmd"),
                                  params.restart_from))
    uncovered, ran = pp_findings(sections, inp.parent)
    if ran:
        checked.append("pseudopotentials")
        warnings.extend(uncovered)
    if "MIMIC" not in sections:
        warnings.append(
            Precondition(
                id="CPMD_NO_MIMIC_SECTION",
                message="input has no &MIMIC section — not a MiMiC run input",
            )
        )
    else:
        if get_keyword(sections["MIMIC"], "PATHS") is None:
            warnings.append(
                Precondition(
                    id="MIMIC_PATHS_MISSING",
                    message="&MIMIC has no PATHS — the coupled programs cannot find "
                    "each other",
                )
            )
    system = sections.get("SYSTEM", [])
    if get_keyword(system, "CHARGE") is None:
        warnings.append(
            Precondition(
                id="QM_CHARGE_MISSING",
                message="&SYSTEM has no CHARGE — run mimic_prepare (charge must be "
                "explicit, never assumed)",
            )
        )
    multiplicity = get_keyword(system, "MULTIPLICITY")
    sidecar = inp.with_name(inp.name + ".myrmidon.json")
    if multiplicity is None and not _reviewed(sidecar):
        warnings.append(
            Precondition(
                id="QM_MULTIPLICITY_UNCONFIRMED",
                message="the input states no multiplicity — run mimic_prepare (closed-shell "
                "runs carry it in the .myrmidon.json sidecar, open-shell runs in &SYSTEM)",
            )
        )
    checked.append("restart_available")
    wants_restart = any(
        line.strip().upper().startswith("RESTART") for line in sections.get("CPMD", [])
    )
    if wants_restart and not params.restart_from:
        warnings.append(
            Precondition(
                id="CPMD_RESTART_FILE_MISSING",
                message="the run input asks to RESTART, and no restart_from was given: "
                "CPMD reads the wavefunction from the file LATEST names in the working "
                "directory and stops in ZHRWF with FILE NOT FOUND when it is not there.",
                suggestion="pass restart_from=<the previous segment's `restart` artifact>, "
                "or render an input without a RESTART line for a run that starts fresh",
            )
        )
    if params.restart_from:
        restart, missing = require_input(ctx, "restart_from", params.restart_from,
                                         "INPUT_MISSING_RESTART",
                                         f"restart file not found: {params.restart_from}")
        if missing:
            warnings.append(missing)
        else:
            warnings.extend(restart_as_recorded(ctx, restart, params.restart_from))

    checked.append("geometry_available")
    wants_geofile = any(
        line.strip().upper().startswith("RESTART") and "GEOFILE" in line.upper()
        for line in sections.get("CPMD", [])
    )
    if params.geometry_from:
        _, missing = require_input(ctx, "geometry_from", params.geometry_from,
                                   "INPUT_MISSING_GEOMETRY",
                                   f"geometry file not found: {params.geometry_from}")
        if missing:
            warnings.append(missing)
    elif wants_geofile and geometry_for(ctx, params) is None:
        warnings.append(
            Precondition(
                id="CPMD_GEOFILE_WITHOUT_GEOMETRY",
                message="the run input's RESTART line has GEOFILE, so CPMD reads the ions' "
                "coordinates from a GEOMETRY file in the working directory, and no "
                "geometry_from was given nor a GEOMETRY found beside restart_from: CPMD "
                "stops in GEOFILE with DO NOT READ OLD COORDINATES.",
                suggestion="pass geometry_from=<the previous segment's `geometry` artifact> "
                "(or a snapshot of this box), or drop GEOFILE from the RESTART line",
            )
        )

    return warnings, checked


def _reviewed(sidecar) -> bool:
    """Whether this input's spin was settled for a stage: a sidecar beside it that the QM
    selection did not write (it states what the selection leaves, for a render to settle)."""
    import json

    try:
        return json.loads(sidecar.read_text()).get("written_by") != "qm_region"
    except (OSError, ValueError):
        return False


def restart_as_recorded(ctx: ToolContext, restart, given: str) -> list[Finding]:
    """The blocking finding when the restart a continuation names is not the file the
    ledger recorded under that path. A continuation is the continuation of a recorded run
    only while its restart hashes to what that run's collection recorded; a file replaced
    since — by another run writing into the same directory, or by hand — makes a chain the
    ledger cannot vouch for. A path no call ever produced has nothing to compare against
    and passes: the ledger records it as this call's input."""
    from myrmidon.core.artifacts import sha256_file

    recorded = ctx.workspace.ledger.last_output(ctx.workspace.relpath(restart))
    if recorded is None:
        return []
    now = sha256_file(restart)
    if now == recorded.sha256:
        return []
    return [Precondition(
        id="RESTART_NOT_AS_RECORDED",
        message=f"{given} is not the file the ledger recorded at that path: it hashes to "
                f"{now[:12]}…, the run that produced it recorded {recorded.sha256[:12]}…. "
                "A continuation from it would not continue the run the ledger shows.",
        suggestion="restart from the `restart` artifact job_status returned for the run "
                   "you mean to continue; its directory holds only that run's files",
    )]


RESTART_NAME = "RESTART.1"


LATEST_NAME = "LATEST"


def geometry_for(ctx: ToolContext, params: MimicRunParams):
    """The GEOMETRY file a ``RESTART ... GEOFILE`` run reads its coordinates from: the one
    named, else the one the previous segment wrote beside the restart it continues from;
    None when there is none. CPMD stops in GEOFILE without it (real CPMD 4.3, coupled)."""
    if params.geometry_from:
        return resolve_in_ws(ctx, params.geometry_from)
    if params.restart_from:
        beside = resolve_in_ws(ctx, params.restart_from).parent / GEOMETRY_NAME
        if beside.is_file():
            return beside
    return None


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


@progress("mimic")
def _mimic_progress(record) -> tuple[None, str]:
    """CPMD's own counter for a coupled run in flight: the GROMACS log of a coupled run
    counts nothing and names the nominal step budget."""
    from pathlib import Path

    from myrmidon.engines.mimic.diagnostics import parse_cpmd_output

    out = Path(record.spec.cwd) / "cpmd.out"
    if not out.exists():
        return None, ""
    facts = parse_cpmd_output(out.read_text(errors="replace"))
    steps, temp = facts.get("md_steps_completed"), facts.get("t_qm_last")
    parts = [f"step {steps}" if steps is not None else "",
             f"T(QM) {temp} K" if temp is not None else ""]
    return None, " · ".join(p for p in parts if p)


@collector("mimic")
def _collect_mimic_job(ctx: ToolContext, record, state, exit_code) -> Observation:
    """A MiMiC run that has ended: the MM engine's outputs by role, plus the CPMD log,
    the force-matching reference it wrote, and — for a run that failed — CPMD's own error
    message and the tail of its log, for the reader."""
    from pathlib import Path

    from myrmidon.engines.mimic.diagnostics import (
        incomplete_run_note,
        parse_cpmd_output,
        read_local_errors,
        run_completed,
    )
    from myrmidon.text import LOG_TAIL_LINES, tail

    obs = collect_run(ctx, record, state, exit_code)
    # the count the GROMACS half put in its own summary, before the QM half adds to it
    before = len(obs.artifacts)
    if before == 0 and state.value == "COMPLETED":
        # a coupled run that produced no MM outputs is either misconfigured or was
        # collected under the wrong prefix; neither is a success
        obs.findings.append(Finding(
            id="MM_OUTPUTS_NOT_FOUND",
            message=(f"the run completed but no MM outputs were found under "
                     f"'{spec_deffnm(record)}.*' in {record.spec.cwd} — the QM side is "
                     "collected below, the GROMACS trajectory and energies are not. "
                     "Anything fitted from them would have no recorded provenance."),
        ))
    cwd = Path(record.spec.cwd)
    root = ctx.workspace.root
    cpmd_out = cwd / "cpmd.out"
    if cpmd_out.exists():
        obs.artifacts.append(ArtifactRef.from_path(cpmd_out, "qm_log", root=root))
        # CPMD's own error file witnesses an abort whatever the exit code says, and a run
        # that ended normally without doing what it was asked — a wavefunction
        # optimisation stopped at its iteration cap, dynamics short of its step count —
        # exits 0 with a RESTART written and is not a success either. The log's tail is
        # what the cause is read from, so a failed run carries it along with the GROMACS
        # half's
        text = cpmd_out.read_text(errors="replace")
        cpmd_error = read_local_errors(cwd)
        if cpmd_error and not obs.diagnosis:
            obs.ok = False
            obs.diagnosis = f"CPMD aborted: {cpmd_error}"
            obs.summary = f"{record.job_id} {state.value.lower()}: {obs.diagnosis}"
        facts = parse_cpmd_output(text)
        if state.value == "COMPLETED":
            from myrmidon.execution.pace import cpmd_pace, cpmd_run_input, record_pace

            pace = cpmd_pace(cpmd_out, cpmd_run_input(record), record.spec.resources,
                             f"{root.name} {record.label or record.job_id}")
            if pace is not None:
                record_pace(ctx.config.studies_root, pace)
        if obs.ok and not run_completed(facts):
            obs.ok = False
            obs.diagnosis = ("CPMD ended before it did what it was asked: "
                             + incomplete_run_note(facts))
            obs.summary = f"{record.job_id} {state.value.lower()}: {obs.diagnosis}"
        if not obs.ok:
            obs.log_tail = "\n".join(part for part in (
                obs.log_tail, "--- cpmd.out ---", tail(text, LOG_TAIL_LINES)) if part)
    for fm_json in sorted(cwd.glob("FMTRAJECTORY*.json")):
        obs.artifacts.append(ArtifactRef.from_path(fm_json, "fm_reference", root=root))
    # what the next segment restarts from, and what a decision about it reads. Without
    # these a continuation has nothing to reference, and CPMD stops in ZHRWF when no
    # restart reaches its directory.
    for name, role in (
        (RESTART_NAME, "restart"), (LATEST_NAME, "restart_index"),
        ("ENERGIES", "energies"), ("GEOMETRY", "geometry"),
        ("TRAJECTORY", "qm_trajectory"),
    ):
        path = cwd / name
        if path.exists():
            obs.artifacts.append(ArtifactRef.from_path(path, role, root=root))
    if obs.ok and len(obs.artifacts) != before:
        # the GROMACS half counted its own before the QM half was collected
        obs.summary = obs.summary.replace(
            f"{before} output artifact(s)", f"{len(obs.artifacts)} output artifact(s)"
        )
    return obs
