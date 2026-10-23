"""MiMiC coupled launch: CPMD (QM) + GROMACS (MM) started together.

mimic_run is validation-gated (the QM/MM analogue of gmx_grompp's gate) and
fully async. CPMD is the main program (its stdout is the QM log); GROMACS
runs as the coupled partner.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import JobObservation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning
from myrmidon.engines.gromacs.results import (
    _spec_deffnm,
    gromacs_results,
    register_collector,
)
from myrmidon.engines.gromacs.run import get_jobstore
from myrmidon.engines.inputs import require_input, resolve_in_ws
from myrmidon.engines.mimic.cpmd_input import (
    get_keyword,
    has_keyword,
    parse_cpmd,
    qm_geometry,
)
from myrmidon.execution.backend import CoupledCmd, JobSpec, Resources


class MimicRunParams(BaseModel):
    #: the fields this submission's cost is computed from; a plan step must state them
    RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("cpmd_ranks", "gromacs_ranks", "walltime_min")

    cpmd_input: str = Field(description="reviewed run input from mimic_prepare")
    tpr: str = Field(description="GROMACS .tpr built for MiMiC (qmmm-active mdp)")
    restart_from: str | None = Field(
        default=None,
        description="the previous segment's CPMD RESTART file, to continue from it "
        "(the `restart` artifact of the earlier mimic_run). Required whenever the input "
        "asks to RESTART: CPMD reads the wavefunction from the file LATEST names in the "
        "working directory, and stops in ZHRWF with FILE NOT FOUND when it is absent.",
    )
    cpmd_ranks: int = Field(default=4, ge=1, description="MPI ranks for CPMD (QM)")
    gromacs_ranks: int = Field(default=1, ge=1, description="MPI ranks for GROMACS (MM)")
    deffnm: str = Field(default="mimic")
    walltime_min: int = Field(default=60, ge=1, description="batch walltime (minutes)")
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
            ntasks=self.cpmd_ranks + self.gromacs_ranks,
            walltime_min=self.walltime_min,
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
    acknowledge_warnings: list[str] = Field(default_factory=list)


def mimic_gate(ctx: ToolContext, params: MimicRunParams):
    from myrmidon.execution.accounting import fit_warning

    warnings: list[ValidationWarning] = []
    checked = ["input_files", "cpmd_engine", "cpmd_sections", "charge_multiplicity", "qm_cell"]

    # A coupled run asks for both programs' ranks at once, which is the submission most
    # likely to outgrow a workstation. Without a scheduler nothing queues it.
    checked.append("machine_fit")
    fit = fit_warning(
        Resources(ntasks=params.cpmd_ranks + params.gromacs_ranks,
                  walltime_min=params.walltime_min),
        ctx.config.compute, ctx.config.backend,
    )
    if fit is not None:
        warnings.append(fit)

    cpmd_info = ctx.engines.get("cpmd")
    if cpmd_info is None or not cpmd_info.available:
        warnings.append(
            ValidationWarning(
                id="CPMD_UNAVAILABLE",
                severity="block",
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
    if "MIMIC" not in sections:
        warnings.append(
            ValidationWarning(
                id="CPMD_NO_MIMIC_SECTION",
                severity="block",
                message="input has no &MIMIC section — not a MiMiC run input",
            )
        )
    else:
        if get_keyword(sections["MIMIC"], "PATHS") is None:
            warnings.append(
                ValidationWarning(
                    id="MIMIC_PATHS_MISSING",
                    severity="block",
                    message="&MIMIC has no PATHS — the coupled programs cannot find "
                    "each other",
                )
            )
        if get_keyword(sections["MIMIC"], "BOX") is None:
            warnings.append(
                ValidationWarning(
                    id="MIMIC_BOX_MISSING",
                    severity="warn",
                    message="&MIMIC has no BOX line — verify the whole-system box is "
                    "declared",
                )
            )
    system = sections.get("SYSTEM", [])
    if get_keyword(system, "CHARGE") is None:
        warnings.append(
            ValidationWarning(
                id="QM_CHARGE_MISSING",
                severity="block",
                message="&SYSTEM has no CHARGE — run mimic_prepare (charge must be "
                "explicit, never assumed)",
            )
        )
    multiplicity = get_keyword(system, "MULTIPLICITY")
    sidecar = inp.with_name(inp.name + ".myrmidon.json")
    if multiplicity is not None:
        if not has_keyword(sections.get("CPMD", []), "LSD"):
            warnings.append(
                ValidationWarning(
                    id="CPMD_MULTIPLICITY_WITHOUT_LSD",
                    severity="block",
                    message="&SYSTEM MULTIPLICITY requires LSD in &CPMD — CPMD 4.x aborts "
                    "in sysin otherwise ('Multiplicity requires local spin density "
                    "approximation'). For a closed-shell system omit MULTIPLICITY.",
                    suggestion="run mimic_prepare with the intended multiplicity",
                )
            )
    elif not sidecar.exists():
        warnings.append(
            ValidationWarning(
                id="QM_MULTIPLICITY_UNCONFIRMED",
                severity="block",
                message="multiplicity was never confirmed — run mimic_prepare (closed-shell "
                "runs carry it in the .myrmidon.json sidecar, open-shell runs in &SYSTEM)",
            )
        )
    geom = qm_geometry(sections)
    if geom is not None:
        vac = geom["min_vacuum_bohr"]
        cell = "x".join(str(v) for v in geom["cell_bohr"])
        if vac < 1.0:
            warnings.append(
                ValidationWarning(
                    id="QM_CELL_NO_VACUUM",
                    severity="block",
                    message=f"QM cell {cell} bohr is the bounding box of the QM atoms "
                    f"(min vacuum {vac} bohr) — CPMD/MiMiC segfaults in the short-range "
                    "potential with no vacuum around the QM region",
                    suggestion="re-run mimic_select_qm with padding_nm >= 0.3",
                )
            )
        elif vac < 5.0:
            warnings.append(
                ValidationWarning(
                    id="QM_CELL_TIGHT",
                    severity="warn",
                    message=f"only {vac} bohr between QM atoms and the cell wall "
                    f"(cell {cell} bohr) — density will touch the wall",
                    assumption="the isolated-molecule Poisson solver needs several bohr of vacuum",
                    suggestion="padding_nm >= 0.3 in mimic_select_qm",
                )
            )
    checked.append("restart_available")
    wants_restart = any(
        line.strip().upper().startswith("RESTART") for line in sections.get("CPMD", [])
    )
    if wants_restart and not params.restart_from:
        warnings.append(
            ValidationWarning(
                id="CPMD_RESTART_FILE_MISSING",
                severity="block",
                message="the run input asks to RESTART, and no restart_from was given: "
                "CPMD reads the wavefunction from the file LATEST names in the working "
                "directory and stops in ZHRWF with FILE NOT FOUND when it is not there.",
                suggestion="pass restart_from=<the previous segment's `restart` artifact>, "
                "or render an input without a RESTART line for a run that starts fresh",
            )
        )
    if params.restart_from and not wants_restart:
        warnings.append(
            ValidationWarning(
                id="CPMD_RESTART_FILE_UNUSED",
                severity="warn",
                message=f"restart_from names {params.restart_from}, but the run input has "
                "no RESTART line, so CPMD would start from scratch and the file would be "
                "ignored — the run would silently not be the continuation it looks like",
                suggestion="add the RESTART keyword to the input (the inputset's "
                "continuation stages carry it), or drop restart_from",
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

    checked.append("cpmd_restraints")
    if has_keyword(sections.get("ATOMS", []), "RESTRAINTS"):
        # A warning, not a refusal: the crash is a property of CPMD builds without PLUMED
        # hooks, not of the input, and a build that has them can run it.
        warnings.append(
            ValidationWarning(
                id="CPMD_RESTRAINTS_UNDER_MIMIC",
                severity="warn",
                message="&ATOMS carries a CONSTRAINTS/RESTRAINTS block. In MiMiC coupled "
                "runs CPMD's native restraints have segfaulted when the restraint force is "
                "first applied, after the wavefunction optimisation converged (a CPMD "
                "build without PLUMED hooks).",
                suggestion="for reference geometries at chosen distances, run displaced "
                "single points instead; acknowledge only on a CPMD build known to support "
                "restraints under MiMiC",
            )
        )
    warnings.extend(maxstep_findings(sections.get("CPMD", [])))
    return warnings, checked



def restart_as_recorded(ctx: ToolContext, restart, given: str) -> list[ValidationWarning]:
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
    return [ValidationWarning(
        id="RESTART_NOT_AS_RECORDED",
        severity="block",
        message=f"{given} is not the file the ledger recorded at that path: it hashes to "
                f"{now[:12]}…, the run that produced it recorded {recorded.sha256[:12]}…. "
                "A continuation from it would not continue the run the ledger shows.",
        suggestion="restart from the `restart` artifact job_status returned for the run "
                   "you mean to continue; its directory holds only that run's files",
    )]


def maxstep_findings(cpmd_section: list[str]) -> list[ValidationWarning]:
    """What to say about &CPMD MAXSTEP, which means two different things.

    Under ``MOLECULAR DYNAMICS`` it bounds ionic steps, so leaving it unset means a run
    of unknown length — worth asking about. Under ``WAVEFUNCTION OPTIMIZATION``, which is
    every single point, it bounds SCF iterations: CPMD's own limit is the right one, and
    a small value returns an unconverged wavefunction that still exits 0 and whose forces
    are unusable as reference data. Demanding MAXSTEP of a single point invites exactly
    that small value; CPMD_WAVEFUNCTION_STEP_CAP is what mimic_diagnose calls the result.
    """
    maxstep = get_keyword(cpmd_section, "MAXSTEP")
    optimising = has_keyword(cpmd_section, "WAVEFUNCTION OPTIMIZATION")
    if maxstep is None and not optimising:
        return [ValidationWarning(
            id="CPMD_MAXSTEP_UNSET",
            severity="warn",
            message="&CPMD has no MAXSTEP, and this run is dynamics",
            assumption="CPMD would use its compiled default number of MD steps",
            suggestion="set maxsteps in mimic_prepare to the number of MD steps this "
                       "stage should run",
        )]
    if maxstep is not None and optimising:
        text = maxstep.strip()
        if text.isdigit() and int(text) < 20:
            return [ValidationWarning(
                id="CPMD_SCF_LIMIT_TOO_LOW",
                severity="warn",
                message=(f"this is a WAVEFUNCTION OPTIMIZATION run, where MAXSTEP bounds "
                         f"SCF iterations rather than MD steps, and it is set to {text}"),
                assumption="the wavefunction would be taken as final after that many "
                           "iterations whether or not it converged",
                suggestion="leave maxsteps unset so CPMD uses its own limit — an "
                           "unconverged single point still exits 0, and its forces are "
                           "unusable as force-matching reference data",
            )]
    return []

#: CPMD reads its restart from ``RESTART`` and writes the next one to ``RESTART.1``.
#: Two names, not one: segment N's RESTART.1 and segment N+1's RESTART are the same bytes,
#: while N+1's own RESTART.1 is different. A continuation needs the predecessor's RESTART.1
#: under the name CPMD *reads*; a copy under the name it *writes* leaves nothing to read,
#: and CPMD stops in ZHRWF with FILE NOT FOUND before the first step.
RESTART_READ_NAME = "RESTART"
RESTART_NAME = "RESTART.1"
LATEST_NAME = "LATEST"


def align_mimic_paths(inp, stage):
    """Make the input's ``&MIMIC PATHS`` name the directory this run will execute in.

    PATHS is not a parameter — it is how MiMiC's two programs find each other, and it
    has to be the canonical absolute path of the directory mdrun runs in (MIMIC_PATHS_NOT_CWD).
    An input carried anywhere else still names where it was made: a later segment of a
    chain in its own stage, or a copy planted in a new workspace. Both fail as an MPI
    coupling error that names the symptom, not the cause.

    Returns ``(input_to_run, note)``. The original is never edited: an aligned copy is
    written into the stage, with its ``.myrmidon.json`` sidecar beside it so what the
    ledger hashes is what ran and its confirmed values travel with it.
    """
    import shutil

    from myrmidon.engines.mimic.cpmd_input import (
        parse_cpmd,
        paths_dir,
        render_cpmd,
        set_paths,
    )

    here = str(stage.resolve())
    sections = parse_cpmd(inp.read_text())
    named = paths_dir(sections)
    if named is None or named == here:
        return inp, ""

    target = stage / inp.name
    if target.resolve() == inp.resolve():
        target = stage / f"{inp.stem}.aligned{inp.suffix}"
    target.write_text(render_cpmd(set_paths(sections, here)))
    sidecar = inp.with_name(inp.name + ".myrmidon.json")
    if sidecar.is_file():
        shutil.copy2(sidecar, target.with_name(target.name + ".myrmidon.json"))
    return target, f"&MIMIC PATHS re-pointed from {named} to this run's directory"


def stage_restart(restart, stage) -> None:
    """Put the previous segment's restart where CPMD looks for it.

    The chain is otherwise invisible: nothing in a CPMD input names its predecessor,
    which is why a hand-run campaign's segment order survives only as directory
    numbering. Here the link is the ``restart_from`` parameter, so the registry records
    it as a hashed input and the ledger carries the edge.

    No ``LATEST`` is written. CPMD writes one at the end of a run naming what it wrote;
    planting one that names a file this segment has not written yet points CPMD at a
    file that does not exist.
    """
    import shutil

    target = stage / RESTART_READ_NAME
    if restart.resolve() != target.resolve():
        shutil.copy2(restart, target)


@tool(
    "mimic_run",
    description="Launch a MiMiC QM/MM run: CPMD and GROMACS started together as "
    "coupled MPI programs. Validation-gated (blocking findings refuse; warn-level "
    "must be acknowledged by id). Async — poll with job_status.",
    returns=JobObservation,
    tags={"mimic", "qmmm", "run"},
    requires={"mimicpy", "cpmd", "gromacs"},
    expensive=True,
    gate=mimic_gate,
)
def mimic_run(ctx: ToolContext, params: MimicRunParams) -> JobObservation:
    from myrmidon.engines.gromacs.common import gmx_bin

    # one submission, one directory, never reused: CPMD's output names are fixed, so a
    # second run in the same directory erases the first
    stage = ctx.workspace.run_dir(ctx.workspace.stage(params.stage), params.deffnm)
    inp = resolve_in_ws(ctx, params.cpmd_input)
    tpr = resolve_in_ws(ctx, params.tpr)
    consumed = [ArtifactRef.from_path(inp, "cpmd_input", root=ctx.workspace.root)]
    inp, aligned = align_mimic_paths(inp, stage)
    if params.restart_from:
        restart = resolve_in_ws(ctx, params.restart_from)
        consumed.append(
            ArtifactRef.from_path(restart, "restart", root=ctx.workspace.root)
        )
        stage_restart(restart, stage)
    cpmd = ctx.engines["cpmd"].path
    spec = JobSpec(
        argv=[cpmd, str(inp)],
        cwd=stage,
        stdout="cpmd.out",
        stderr="cpmd.err",
        mpi_ranks=params.cpmd_ranks,
        resources=Resources(
            ntasks=params.cpmd_ranks + params.gromacs_ranks,
            walltime_min=params.walltime_min,
            partition=params.partition,
            account=params.account,
        ),
        coupled=[
            CoupledCmd(
                argv=[gmx_bin(ctx), "mdrun", "-s", str(tpr), "-deffnm", params.deffnm],
                mpi_ranks=params.gromacs_ranks,
                stdout=f"{params.deffnm}.stdout",
                stderr=f"{params.deffnm}.stderr",
            )
        ],
    )
    from myrmidon.execution.accounting import estimate_core_h

    compute = getattr(ctx.config, "compute", None)
    est = estimate_core_h(spec.resources, compute) if compute else 0.0
    handle = ctx.backend.submit(spec)
    job_id = get_jobstore(ctx).add(
        ctx.backend.name, handle, spec, label=f"mimic:{params.deffnm}",
        # the prefix the MM half writes under: its own argv is in the partner command,
        # not in spec.argv, so collection cannot find it otherwise
        meta={"kind": "mimic", "core_h_estimate": est, "deffnm": params.deffnm},
    )
    return JobObservation(
        ok=True,
        summary=f"MiMiC run submitted as {job_id}: CPMD x{params.cpmd_ranks} + "
        f"GROMACS x{params.gromacs_ranks} in {ctx.workspace.relpath(stage)} "
        f"(booked {est} core-h). "
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


@register_collector("mimic")
def _collect_mimic_job(ctx: ToolContext, record, state, exit_code) -> JobObservation:
    """A MiMiC run that has ended: the MM engine's outputs by role, plus the CPMD log,
    the force-matching reference it wrote, and the CPMD diagnosis when the log has one."""
    from pathlib import Path

    from myrmidon.engines.mimic.diagnostics import (
        outranked_note,
        parse_cpmd_output,
        read_local_errors,
        signature_stands,
    )

    obs = gromacs_results(ctx, record, state, exit_code)
    # the count the GROMACS half put in its own summary, before the QM half adds to it
    before = len(obs.artifacts)
    if before == 0 and state.value == "COMPLETED":
        # a coupled run that produced no MM outputs is either misconfigured or was
        # collected under the wrong prefix; neither is a success
        obs.warnings.append(ValidationWarning(
            id="MM_OUTPUTS_NOT_FOUND", severity="warn",
            message=(f"the run completed but no MM outputs were found under "
                     f"'{_spec_deffnm(record)}.*' in {record.spec.cwd} — the QM side is "
                     "collected below, the GROMACS trajectory and energies are not. "
                     "Anything fitted from them would have no recorded provenance."),
        ))
    cwd = Path(record.spec.cwd)
    root = ctx.workspace.root
    cpmd_out = cwd / "cpmd.out"
    if cpmd_out.exists():
        obs.artifacts.append(ArtifactRef.from_path(cpmd_out, "qm_log", root=root))
        facts = parse_cpmd_output(cpmd_out.read_text(errors="replace"))
        # a signature is the diagnosis only when something witnesses an abort: the exit
        # code, a LocalError, or a log that did not complete what it was asked
        cpmd_error = read_local_errors(cwd)
        if facts.get("signature_id"):
            if signature_stands(facts, exit_code=exit_code, cpmd_error=cpmd_error):
                if not obs.diagnosis:
                    obs.diagnosis = facts["signature"]
                    obs.diagnosis_id = facts["signature_id"]
                    obs.ok = False
                    obs.summary = (f"{record.job_id} {state.value.lower()}: {obs.diagnosis} "
                                   f"[matched in cpmd.out: {facts['signature_span']!r}]")
            else:
                obs.warnings.append(ValidationWarning(
                    id="CPMD_SIGNATURE_IN_FINISHED_LOG", severity="warn",
                    message=outranked_note(facts, exit_code=exit_code),
                ))
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
