"""Force-matching fit tools: ``fm_plan`` writes an explicit ``fminput.dat`` (stride and QM
total charge are required; weight defaults surface as assumption warnings), ``fm_fit``
submits the fit as a job, and ``job_status`` collects it: fm.log, optimization_results.txt
and the optimized .itp parsed into typed facts, and the candidate topology assembled from
them."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import PREVIEW_HELP, ToolContext, precheck, tool
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.gromacs.common import stage_output, stage_topology
from myrmidon.engines.inputs import require_input, resolve_in_ws
from myrmidon.engines.mimic.fm import (
    gmx_data_flags,
    itp_atomtypes_warnings,
    rerun_mdp_warnings,
)
from myrmidon.engines.mimic.qm_region import mimicpy_bin
from myrmidon.execution.backend import JobSpec, Resources
from myrmidon.execution.jobs import submit_job

_DEFAULT_WEIGHTS = {"wv": 1.0, "we": 0.01, "wh": 0.01, "wq": 100000.0}




class FMPlanParams(BaseModel):
    stride: list[int] = Field(
        min_length=3, max_length=3,
        description="[first, last(exclusive), step] snapshot indices to fit — REQUIRED",
    )
    qm_total_charge: int = Field(
        description="net charge of the QM region (REQUIRED — confirm against mimic_prepare)"
    )
    wv: float | None = Field(default=None, description="DRESP potential weight")
    we: float | None = Field(default=None, description="DRESP field weight")
    wh: float | None = Field(default=None, description="DRESP hirshfeld weight")
    wq: float | None = Field(default=None, description="DRESP total-charge weight")
    eq_atoms: str = Field(default="use_atomtypes")
    reference_charges: str = Field(default="ff_charges")
    optimization_method: str = Field(default="simultaneous")
    extra: dict[str, str] = Field(
        default_factory=dict, description="extra fminput keys, passed through verbatim"
    )
    dataset_frames: int = Field(
        description="total configurations in the dataset (from fm_collect) — REQUIRED "
        "so the stride can be checked against reality",
    )
    output: str = Field(default="fminput.dat")
    stage: str = Field(default="fm")


@tool(
    "fm_plan",
    description="Write an explicit force-matching input (fminput.dat). Stride and QM "
    "total charge are REQUIRED; defaulted DRESP weights are surfaced as assumption "
    "warnings; the stride is validated against the dataset size (from fm_collect).",
    tags={"mimic", "fm"},
    requires={"mimicpy"},
    contract=Contract(produces=(Produces("fm_input"),)),
)
def fm_plan(ctx: ToolContext, params: FMPlanParams) -> Observation:
    first, last, step = params.stride
    if not (0 <= first < last and step >= 1):
        return Observation(
            ok=False,
            summary=f"invalid stride {params.stride}: need 0 <= first < last and step >= 1",
        )
    if last > params.dataset_frames:
        return Observation(
            ok=False,
            summary=f"stride end {last} exceeds the dataset ({params.dataset_frames} "
            "configurations) — re-collect or shrink the stride",
        )
    n_snap = len(range(first, last, step))
    warnings: list[Finding] = []
    weights: dict[str, float] = {}
    for key in ("wv", "we", "wh", "wq"):
        value = getattr(params, key)
        if value is None:
            weights[key] = _DEFAULT_WEIGHTS[key]
            warnings.append(
                Finding(
                    id=f"FM_WEIGHT_DEFAULT_{key.upper()}",
                    message=f"{key} not given — using {weights[key]}",
                    assumption=f"{key}={weights[key]} (a weight fits of a small solute "
                    "in water use; nothing derived it for this system)",
                )
            )
        else:
            weights[key] = value
    if n_snap < 100:
        warnings.append(
            Finding(
                id="FM_FEW_SNAPSHOTS",
                message=f"only {n_snap} snapshot(s) selected — a fit with fewer than 100 "
                "overfits and validates poorly",
                suggestion="widen the stride or extend the reference trajectory",
            )
        )

    stage = ctx.workspace.stage(params.stage)
    lines = [
        "# Force matching input (written by myrmidon fm_plan)",
        f"stride: [{first},{last},{step}]",
    ]
    lines += [f"{k}: {v:g}" for k, v in weights.items()]
    lines += [
        f"eq_atoms: {params.eq_atoms}",
        f"qm_total_charge: {params.qm_total_charge}",
        f"reference_charges: {params.reference_charges}",
        f"optimization_method: {params.optimization_method}",
    ]
    lines += [f"{k}: {v}" for k, v in params.extra.items()]
    out = stage_output(stage, params.output)
    out.write_text("\n".join(lines) + "\n")
    return Observation(
        ok=True,
        summary=f"FM plan written: {n_snap} snapshots (stride {params.stride}), "
        f"qm_total_charge {params.qm_total_charge}, method "
        f"{params.optimization_method} -> {ctx.workspace.relpath(out)}",
        findings=warnings,
        artifacts=[ArtifactRef.from_path(out, "fm_input", root=ctx.workspace.root)],
        n_snapshots=n_snap,
        stride=list(params.stride),
        weights=weights,
        qm_total_charge=params.qm_total_charge,
    )


class FMFitParams(BaseModel):
    #: the fields this submission's cost is computed from
    RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("n_processes",)

    preview: bool = Field(default=False, description=PREVIEW_HELP)

    topology: str
    coords: str
    fmdata: str = Field(description="FMTRAJECTORY .h5 (from fm_collect)")
    trajectory: str
    index: str = Field(description="index file with the QMatoms group (.ndx)")
    selection: str
    fm_input: str = Field(description="fminput.dat from fm_plan")
    mdp: str | None = Field(default=None, description="mdp for the classical rerun")
    mode: Literal["fm", "dresp", "opt_ff"] = Field(default="fm")
    n_processes: int = Field(default=12, ge=1)
    account: str | None = Field(
        default=None,
        description=
        "scheduler account to bill for THIS submission instead of the machine config's "
        "default — the user may hold several project allocations",
    )
    stage: str = Field(default="fm")

    def resources(self) -> Resources:
        """What the fit books: one task with ``n_processes`` cores, for the scheduler's
        default walltime — the request it submits, so the gate prices it the same way."""
        return Resources(ntasks=1, cpus_per_task=self.n_processes, account=self.account)


def fm_gate(ctx: ToolContext, params: FMFitParams):
    warnings: list[Finding] = []
    checked = ["input_files", "fm_input_keys"]
    inputs = [
        ("topology", params.topology), ("coords", params.coords),
        ("fmdata", params.fmdata), ("trajectory", params.trajectory),
        ("index", params.index), ("selection", params.selection),
        ("fm_input", params.fm_input),
    ]
    if params.mdp:
        inputs.append(("mdp", params.mdp))
    missing = False
    for label, path in inputs:
        _, absent = require_input(ctx, label, path)
        if absent:
            missing = True
            warnings.append(absent)
    if not missing and params.mdp:
        from myrmidon.engines.gromacs.mdp import parse_mdp_file

        mdp = parse_mdp_file(resolve_in_ws(ctx, params.mdp))
        checked.append("rerun_mdp_classical")
        warnings.extend(rerun_mdp_warnings(mdp))
    if not missing:
        checked.append("itp_atomtypes")
        warnings.extend(itp_atomtypes_warnings(resolve_in_ws(ctx, params.topology)))
        text = resolve_in_ws(ctx, params.fm_input).read_text()
        keys = {
            ln.split(":", 1)[0].strip()
            for ln in text.splitlines()
            if ":" in ln and not ln.lstrip().startswith(("#", "!"))
        }
        for required in ("stride", "qm_total_charge"):
            if required not in keys:
                warnings.append(
                    Precondition(
                        id="FM_INPUT_INCOMPLETE",
                        message=f"fminput has no '{required}' — write it with fm_plan "
                        "(explicit, never assumed)",
                    )
                )
    return warnings, checked


@tool(
    "fm_fit",
    description="Run the MiMiCPy force-matching fit (DRESP charges + bonded/LJ "
    "parameter optimization against QM/MM reference forces). Validation-gated and "
    "async — poll with job_status, which returns the parsed fit once it has ended.",
    tags={"mimic", "fm", "run"},
    requires={"mimicpy", "gromacs"},
    expensive=True,
    contract=Contract(consumes={"topology": "topology", "coords": "structure",
                                "fmdata": "fm_dataset", "trajectory": "trajectory",
                                "index": "index", "fm_input": "fm_input", "mdp": "mdp"},
                      produces=(Produces("fm_log", collected=True),
                                Produces("fitted_topology", collected=True))),
)
def fm_fit(ctx: ToolContext, params: FMFitParams) -> Observation:
    precheck(ctx, fm_gate(ctx, params)[0])
    from myrmidon.engines.gromacs.common import gmx_bin

    stage = ctx.workspace.stage(params.stage)
    top = stage_topology(stage, resolve_in_ws(ctx, params.topology))
    argv = [
        mimicpy_bin(ctx), "fm",
        "-top", str(top),
        "-coords", str(resolve_in_ws(ctx, params.coords)),
        "-fmdata", str(resolve_in_ws(ctx, params.fmdata)),
        "-trr", str(resolve_in_ws(ctx, params.trajectory)),
        "-ndx", str(resolve_in_ws(ctx, params.index)),
        "-sele", str(resolve_in_ws(ctx, params.selection)),
        "-fi", str(resolve_in_ws(ctx, params.fm_input)),
        "-gmx", gmx_bin(ctx),
        "-n_processes", str(params.n_processes),
        *gmx_data_flags(ctx),
    ]
    if params.mdp:
        argv += ["-mdp", str(resolve_in_ws(ctx, params.mdp))]
    if params.mode != "fm":
        argv.append(f"-{params.mode}")
    spec = JobSpec(
        argv=argv,
        cwd=stage,
        stdout="fm.stdout",
        stderr="fm.stderr",
        resources=params.resources(),
    )
    job_id, est = submit_job(ctx, spec, kind="fm_fit", label=f"fm:{params.mode}",
                             meta={"topology": params.topology})
    return Observation(
        ok=True,
        summary=f"force-matching fit submitted as {job_id} (mode {params.mode}, "
        f"{params.n_processes} processes) in {ctx.workspace.relpath(stage)}; booked "
        f"{est} core-h. "
        "Poll with job_status: once the fit has ended it returns the parsed fit, the "
        "fitted .itp(s) and the candidate topology assembled from them.",
        command=argv,
        job_id=job_id,
        state="QUEUED",
        core_h_estimate=est,
    )
