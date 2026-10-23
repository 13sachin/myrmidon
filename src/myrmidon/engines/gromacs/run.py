"""grompp (validation-gated) and mdrun (async submission).

gmx_grompp is the last gate before dynamics: the MDP sanity rules run first,
and warn-level findings must be acknowledged by id in the call. gmx_mdrun then
submits through the execution backend and returns immediately — poll with
job_status, which collects the results once the job has ended.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import JobObservation, NextAction, Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning
from myrmidon.engines.gromacs.common import run_gmx, stage_output, tail
from myrmidon.engines.gromacs.mdp import parse_mdp_file, validate_mdp
from myrmidon.engines.gromacs.prep import _artifact
from myrmidon.engines.inputs import require_input, resolve_in_ws
from myrmidon.execution.accounting import budget_warnings
from myrmidon.execution.backend import JobSpec, Resources
from myrmidon.execution.jobstore import JobStore, open_jobstore


def get_jobstore(ctx: ToolContext) -> JobStore:
    return open_jobstore(ctx.workspace)


class GromppParams(BaseModel):
    mdp: str = Field(description="run-parameter file (.mdp)")
    structure: str = Field(description="input coordinates (.gro)")
    topology: str = Field(description="topology (.top)")
    index: str | None = Field(
        default=None, description="index file (.ndx) for group references, e.g. QMMM-grps"
    )
    output: str = Field(default="topol.tpr")
    stage: str = Field(default="run", description="workspace stage to write into")
    max_warn: int = Field(default=0, description="passed to grompp -maxwarn")
    acknowledge_warnings: list[str] = Field(
        default_factory=list,
        description="ids of warn-level gate findings you explicitly accept",
    )


def grompp_gate(ctx: ToolContext, params: GromppParams):
    warnings: list[ValidationWarning] = []
    checked: list[str] = ["input_files"]
    inputs = [("mdp", params.mdp), ("structure", params.structure), ("topology", params.topology)]
    if params.index:
        inputs.append(("index", params.index))
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
        from myrmidon.core.file_origin import edit_findings

        warnings += edit_findings(ctx, params.mdp)

    checked.append("maxwarn")
    if params.max_warn > 0:
        warnings.append(
            ValidationWarning(
                id="GROMPP_MAXWARN",
                severity="warn",
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
    mdp_warnings, mdp_checked = validate_mdp(mdp_params)
    return warnings + mdp_warnings, checked + mdp_checked


_WARNING_BLOCK_RE = re.compile(r"WARNING \d+ \[[^\]]*\]:\s*\n((?:\s{2,}.*\n?)+)")
_NET_CHARGE_RE = re.compile(r"System has non-zero total charge:\s*(-?[\d.]+)")


def grompp_warnings(stderr: str) -> list[ValidationWarning]:
    """GROMACS' NOTEs and WARNINGs re-parsed into structured findings. A WARNING
    is what -maxwarn would have hidden; the net-charge one carries the number
    and the fix, because 'silence it' is the plausible wrong path."""
    found = [
        ValidationWarning(id="GROMPP_NOTE", severity="info", message=m.strip())
        for m in re.findall(r"NOTE \d+ \[[^\]]*\]:\s*\n((?:\s{2,}.*\n?)+)", stderr)
    ]
    charge = _NET_CHARGE_RE.search(stderr)
    for block in _WARNING_BLOCK_RE.findall(stderr):
        text = " ".join(block.split())
        if "net charge" in text.lower() and charge:
            found.append(
                ValidationWarning(
                    id="GROMPP_NET_CHARGE",
                    severity="warn",
                    message=f"system net charge {charge.group(1)} e with Ewald electrostatics: "
                    "GROMACS would add a uniform background charge and the dynamics would be "
                    "of the wrong system. Neutralize with gmx_genion — given this structure "
                    "and topology it builds its own run input — then grompp the neutral "
                    "system cleanly. " + text,
                )
            )
        else:
            found.append(ValidationWarning(id="GROMPP_WARNING", severity="warn", message=text))
    return found


@tool(
    "gmx_grompp",
    description="Preprocess mdp+structure+topology into a portable run input (.tpr). "
    "Runs the MDP validation gate first: blocking findings refuse; warn-level findings "
    "must be acknowledged by id via acknowledge_warnings. GROMACS' own notes are "
    "re-parsed into structured warnings.",
    returns=Observation,
    tags={"gromacs", "run"},
    requires={"gromacs"},
    expensive=True,
    gate=grompp_gate,
)
def gmx_grompp(ctx: ToolContext, params: GromppParams) -> Observation:
    stage = ctx.workspace.stage(params.stage)
    mdp = resolve_in_ws(ctx, params.mdp)
    structure = resolve_in_ws(ctx, params.structure)
    topology = resolve_in_ws(ctx, params.topology)
    tpr = stage_output(stage, params.output)
    args = [
        "grompp", "-f", str(mdp), "-c", str(structure), "-p", str(topology),
        "-o", str(tpr), "-po", str(stage / "mdout.mdp"),
        "-maxwarn", str(params.max_warn),
    ]
    if params.index:
        args += ["-n", str(resolve_in_ws(ctx, params.index))]
    proc = run_gmx(ctx, args, cwd=stage)
    command = list(proc.args)
    notes = grompp_warnings(proc.stderr)
    if proc.returncode != 0 or not tpr.exists():
        blocking = [w.id for w in notes if w.severity == "warn"]
        neutralise = [NextAction(
            kind="call", tool="gmx_genion", finding_ids=["GROMPP_NET_CHARGE"],
            known_parameters={"structure": params.structure, "topology": params.topology},
            reason="neutralise first: genion builds its own run input from the structure, "
            "then this grompp runs on the structure and topology it returns",
        )] if "GROMPP_NET_CHARGE" in blocking else []
        return Observation(
            ok=False,
            summary=f"grompp failed (exit {proc.returncode})"
            + (f" on grompp warning(s) {blocking} — read them, fix the cause; "
               "max_warn only hides them" if blocking else " — see log tail"),
            command=command,
            warnings=notes,
            log_tail=tail(proc.stderr, 40),
            next_actions=neutralise or None,
        )
    return Observation(
        ok=True,
        summary=f"Run input written: {ctx.workspace.relpath(tpr)}"
        + (f" ({len(notes)} grompp note(s) attached)" if notes else ""),
        command=command,
        warnings=notes,
        artifacts=[_artifact(ctx, tpr, "tpr")],
    )


class SubmitParams(BaseModel):
    """What every dynamics submission must state: the resources it books, so
    the core-hour estimate exists before the job does."""

    #: the fields this submission's cost is computed from; a plan step must state them
    RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("ntasks", "cpus_per_task", "walltime_min")

    ntasks: int = Field(default=1, ge=1, description="MPI ranks / tasks to book")
    cpus_per_task: int = Field(default=1, ge=1)
    walltime_min: int = Field(default=30, ge=1, description="scheduler time limit")
    account: str | None = Field(
        default=None,
        description=
        "scheduler account to bill for THIS submission instead of the machine config's "
        "default — the user may hold several project allocations",
    )
    acknowledge_warnings: list[str] = Field(
        default_factory=list, description="ids of warn-level gate findings you accept",
    )

    def resources(self) -> Resources:
        return Resources(
            ntasks=self.ntasks, cpus_per_task=self.cpus_per_task, walltime_min=self.walltime_min,
            account=self.account,
        )


class MdrunParams(SubmitParams):
    tpr: str = Field(description="run input (.tpr) from gmx_grompp")
    deffnm: str = Field(default="md", description="output file prefix")
    stage: str = Field(default="run")
    nthreads: int | None = Field(default=None, description="gmx mdrun -nt")


_TPR_FIELD_RE = re.compile(
    r"^\s*(nsteps|dt|init-step|integrator|natoms)\s*=\s*(\S+)", re.MULTILINE
)

#: integrators that minimise: they take steps, not time, so "the run is 0.05 ns" is not
#: a statement about them and a ns/day ceiling cannot judge their walltime
EM_INTEGRATORS = frozenset({"steep", "cg", "l-bfgs"})


def tpr_facts(ctx: ToolContext, tpr) -> dict[str, str]:
    """What the .tpr says about the run it carries, read with ``gmx dump -s``."""
    proc = run_gmx(ctx, ["dump", "-s", str(tpr)], cwd=tpr.parent)
    return dict(_TPR_FIELD_RE.findall(proc.stdout))


def tpr_run_length_ns(ctx: ToolContext, tpr) -> float | None:
    """Simulated time the .tpr carries (nsteps × dt, minus init-step), read
    with ``gmx dump -s`` — the number a walltime booking has to be able to cover."""
    fields = tpr_facts(ctx, tpr)
    try:
        nsteps, dt = int(fields["nsteps"]), float(fields["dt"])
    except (KeyError, ValueError):
        return None
    if nsteps < 0:
        return None  # -1: run until stopped
    return round(max(0, nsteps - int(fields.get("init-step", 0))) * dt / 1000.0, 6)


def throughput_warning(length_ns: float, walltime_min: int, ctx: ToolContext) -> ValidationWarning:
    """Does the booked walltime cover the run? The harness cannot know this
    machine's ns/day, but it can compute what the booking *requires* and compare
    it with the plausible ceiling in the machine config — a job that needs more
    stops at the time limit with a partial trajectory, which is not the run
    that was asked for."""
    cap = ctx.config.compute.plausible_ns_per_day
    required = length_ns / (walltime_min / 1440.0)
    reach = round(cap * walltime_min / 1440.0, 3)
    if required > cap:
        return ValidationWarning(
            id="WALLTIME_INSUFFICIENT", severity="warn", needs_human=True,
            message=f"the run is {length_ns:g} ns; {walltime_min} min of walltime would need "
            f"{required:.0f} ns/day, above this machine's plausible ceiling of {cap:g} ns/day "
            f"(compute.plausible_ns_per_day) — mdrun will stop at the time limit after roughly "
            f"{reach:g} ns, not the run that was asked for. Book enough walltime (and the "
            "core-hours that implies), split into checkpointed segments, or report that the "
            "run does not fit the allowance.",
        )
    return ValidationWarning(
        id="RUN_LENGTH", severity="info",
        message=f"the run is {length_ns:g} ns; {walltime_min} min needs {required:.0f} ns/day",
    )


def minimisation_warning(
    nsteps: int, n_atoms: int, walltime_min: int, ctx: ToolContext
) -> ValidationWarning:
    """Can the booked walltime cover a minimisation? Its size is steps × atoms, and the
    machine config says how fast that goes here. A minimisation killed at the time limit
    leaves no converged structure, and a box many times the intended size is the usual
    reason the booking falls short."""
    rate = ctx.config.compute.plausible_em_atom_steps_per_s
    capacity = rate * walltime_min * 60
    reach = int(capacity / max(n_atoms, 1))
    if nsteps * n_atoms > capacity:
        return ValidationWarning(
            id="EM_WALLTIME_INSUFFICIENT", severity="warn", needs_human=True,
            message=f"minimising {n_atoms} atoms for up to {nsteps} steps: at this "
            f"machine's plausible {rate:g} atom-steps/s "
            f"(compute.plausible_em_atom_steps_per_s) a {walltime_min} min booking "
            f"reaches about {reach} steps, so the job is killed at the time limit "
            "unless it converges first. Book more walltime, or check that the system is "
            "the size you meant — an unexpectedly large box is the usual cause.",
            suggestion="raise walltime_min, or check the atom count against the system "
            "you intended",
        )
    return ValidationWarning(
        id="EM_LENGTH", severity="info",
        message=f"minimising {n_atoms} atoms, at most {nsteps} steps; {walltime_min} min "
        f"covers about {reach} steps here",
    )


def mdrun_gate(ctx: ToolContext, params: MdrunParams):
    warnings: list[ValidationWarning] = []
    checked = ["input_files", "compute_budget"]
    tpr, missing = require_input(ctx, "tpr", params.tpr, message=f"tpr not found: {params.tpr}")
    if missing:
        warnings.append(missing)
    _, budget = budget_warnings(params.resources(), ctx.config.compute, ctx.config.backend)
    warnings += budget
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
                    minimisation_warning(steps, atoms, params.walltime_min, ctx)
                )
        else:
            length = tpr_run_length_ns(ctx, tpr)
            if length is not None:
                checked.append("walltime_covers_run")
                warnings.append(throughput_warning(length, params.walltime_min, ctx))
    return warnings, checked


def _submit(ctx: ToolContext, params: SubmitParams, argv: list[str], stage, label: str,
            stem: str) -> tuple[str, float]:
    est, _ = budget_warnings(params.resources(), ctx.config.compute)
    spec = JobSpec(
        argv=argv, cwd=stage, stdout=f"{stem}.stdout", stderr=f"{stem}.stderr",
        resources=params.resources(),
    )
    handle = ctx.backend.submit(spec)
    job_id = get_jobstore(ctx).add(
        ctx.backend.name, handle, spec, label=label, meta={"core_h_estimate": est},
    )
    return job_id, est


@tool(
    "gmx_mdrun",
    description="Run dynamics ASYNCHRONOUSLY through the execution backend. Returns a "
    "job_id immediately — poll with job_status, which also collects it. Never blocks. "
    "Books ntasks × cpus_per_task × walltime_min; estimates above the machine's approval "
    "threshold must be acknowledged (COMPUTE_BUDGET) after asking the user. Also checks "
    "that the walltime can cover the run length in the .tpr (WALLTIME_INSUFFICIENT).",
    returns=JobObservation,
    tags={"gromacs", "run"},
    requires={"gromacs"},
    expensive=True,
    gate=mdrun_gate,
)
def gmx_mdrun(ctx: ToolContext, params: MdrunParams) -> JobObservation:
    from myrmidon.engines.gromacs.common import gmx_bin

    stage = ctx.workspace.stage(params.stage)
    tpr = resolve_in_ws(ctx, params.tpr)
    argv = [gmx_bin(ctx), "mdrun", "-s", str(tpr), "-deffnm", params.deffnm]
    if params.nthreads is not None:
        argv += ["-nt", str(params.nthreads)]
    job_id, est = _submit(ctx, params, argv, stage, f"mdrun:{params.deffnm}", params.deffnm)
    return JobObservation(
        ok=True,
        summary=f"mdrun submitted via {ctx.backend.name} backend as {job_id} "
        f"(outputs {params.deffnm}.* in {ctx.workspace.relpath(stage)}; "
        f"booked {est} core-h). Poll with job_status.",
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
    deffnm: str = Field(default="md", description="output prefix (match the parent run)")
    stage: str = Field(default="run")
    nthreads: int | None = None


_CPT_OUTPUT_RE = re.compile(r"output filename\s*=\s*(\S+)")


def checkpoint_output_names(ctx: ToolContext, cpt) -> list[str]:
    """The output file names recorded in a checkpoint (``gmx dump -cp``) —
    what mdrun -append will compare VERBATIM against its own -deffnm expansion."""
    proc = run_gmx(ctx, ["dump", "-cp", str(cpt)], cwd=cpt.parent)
    return _CPT_OUTPUT_RE.findall(proc.stdout)


def continue_gate(ctx: ToolContext, params: CheckpointContinueParams):
    warnings: list[ValidationWarning] = []
    checked = ["input_files", "checkpoint_names", "compute_budget"]
    _, missing_tpr = require_input(ctx, "tpr", params.tpr,
                                   message=f"tpr not found: {params.tpr}")
    cpt, missing_cpt = require_input(ctx, "checkpoint", params.checkpoint,
                                     message=f"checkpoint not found: {params.checkpoint}")
    warnings += [w for w in (missing_tpr, missing_cpt) if w]
    if cpt is not None:
        names = checkpoint_output_names(ctx, cpt)
        stage = ctx.workspace.stage(params.stage)
        mismatched = [
            n for n in names
            if Path(n).stem != params.deffnm
            or (Path(n).is_absolute() and Path(n).parent != stage)
        ]
        if not names:
            warnings.append(ValidationWarning(
                id="CPT_UNREADABLE", severity="warn",
                message=f"could not read output names from {params.checkpoint} (gmx dump -cp); "
                "cannot verify that -deffnm matches the parent run",
            ))
        elif mismatched:
            warnings.append(ValidationWarning(
                id="CPT_NAME_MISMATCH", severity="block",
                message=f"the checkpoint records outputs {names[:4]} but this continuation "
                f"would write {params.deffnm}.* in {ctx.workspace.relpath(stage)}. mdrun "
                "-append compares these names verbatim (absolute stays absolute) and aborts; "
                "dropping -cpi instead would restart from the .tpr and silently NOT continue. "
                "Pass deffnm/stage exactly as the parent run used them.",
            ))
    _, budget = budget_warnings(params.resources(), ctx.config.compute, ctx.config.backend)
    checked.append("walltime_covers_run")
    return warnings + budget + [
        throughput_warning(params.extend_ps / 1000.0, params.walltime_min, ctx)
    ], checked


@tool(
    "checkpoint_continue",
    description="Extend a finished run from its checkpoint: convert-tpr adds time, "
    "mdrun continues from the .cpt with full provenance linkage to the parent run. "
    "The gate verifies the checkpoint's recorded output names match deffnm/stage "
    "(mdrun -append compares them verbatim) and books the core-hour estimate. "
    "Async — returns a job_id immediately.",
    returns=JobObservation,
    tags={"gromacs", "run"},
    requires={"gromacs"},
    expensive=True,
    gate=continue_gate,
)
def checkpoint_continue(ctx: ToolContext, params: CheckpointContinueParams) -> JobObservation:
    from myrmidon.engines.gromacs.common import gmx_bin

    stage = ctx.workspace.stage(params.stage)
    tpr = resolve_in_ws(ctx, params.tpr)
    cpt = resolve_in_ws(ctx, params.checkpoint)
    extended = stage / f"{params.deffnm}_ext.tpr"
    conv = run_gmx(
        ctx,
        ["convert-tpr", "-s", str(tpr), "-extend", str(params.extend_ps),
         "-o", str(extended)],
        cwd=stage,
    )
    if conv.returncode != 0 or not extended.exists():
        return JobObservation(
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
    return JobObservation(
        ok=True,
        summary=f"Continuation submitted as {job_id}: +{params.extend_ps} ps from "
        f"{ctx.workspace.relpath(cpt)} (booked {est} core-h). Poll with job_status.",
        command=argv,
        artifacts=[ArtifactRef.from_path(extended, "tpr", root=ctx.workspace.root)],
        job_id=job_id,
        state="QUEUED",
        core_h_estimate=est,
    )
