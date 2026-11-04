"""Force-matching fit tools: ``fm_plan`` writes an explicit ``fminput.dat`` (stride and QM
total charge are required; weight defaults surface as assumption warnings), ``fm_fit``
submits the fit as a job, and ``job_status`` collects it: fm.log, optimization_results.txt
and the optimized .itp parsed into typed facts, and the candidate topology assembled from
them."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import PREVIEW_HELP, ToolContext, precheck, tool
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.gromacs.common import stage_output, stage_topology, tail
from myrmidon.engines.gromacs.itp import atom_charges
from myrmidon.engines.inputs import require_input, resolve_in_ws
from myrmidon.engines.mimic.fm import (
    FORCE_OUTLIER_FACTOR,
    FORCE_UNITS,
    PARAMETER_FINDINGS,
    VALIDATION_RESIDUALS_FILENAME,
    fitted_pairs,
    gmx_data_flags,
    implausible_parameters,
    includes_of,
    itp_atomtypes_warnings,
    parse_fm_log,
    parse_optimization_results,
    parse_validation_force_residuals,
    rerun_mdp_warnings,
)
from myrmidon.engines.mimic.qm_region import mimicpy_bin
from myrmidon.execution.backend import JobSpec, JobState, Resources
from myrmidon.execution.collect import collector
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




def collect_fit(ctx: ToolContext, stage_dir: Path, original_itp: str | None = None,
                topology: str | None = None, *, completed: bool = True) -> Observation:
    """A force-matching fit that has ended, read into typed facts: snapshot counts,
    train/validation split, validation SDF/cost, optimizer status, wall time, the fitted
    charge deltas against the original .itp, and — given the topology the fit started
    from — the candidate topology with the fitted include(s) swapped in. This is what
    ``job_status`` returns for an ``fm_fit`` job; ``original_itp`` is inferred from the
    topology's includes when not given."""
    log_path = stage_dir / "fm.log"
    if not log_path.exists():
        return Observation(
            ok=False,
            summary=f"no fm.log at {ctx.workspace.relpath(log_path)} — has the fit run?",
        )
    facts = parse_fm_log(log_path.read_text(errors="replace"))
    results_file = log_path.parent / "optimization_results.txt"
    if results_file.exists():
        for key, value in parse_optimization_results(results_file.read_text()).items():
            facts.setdefault(key, value)

    artifacts = [ArtifactRef.from_path(log_path, "fm_log", root=ctx.workspace.root)]
    max_dq = mean_dq = n_q = None
    opt_itps: list[Path] = []
    for name in facts.get("itp_files", []):
        candidate = log_path.parent / name
        if candidate.exists() and name.startswith("opt_"):
            opt_itps.append(candidate)
            artifacts.append(
                ArtifactRef.from_path(candidate, "fitted_topology", root=ctx.workspace.root)
            )
    top = resolve_in_ws(ctx, topology) if topology else None
    if original_itp is None and top is not None and opt_itps:
        for inc in fitted_pairs(top, opt_itps):
            candidate_orig = top.parent / inc
            if candidate_orig.exists():
                original_itp = ctx.workspace.relpath(candidate_orig)
                break
    opt_itp = None
    if original_itp and opt_itps:
        # the fitted twin of the original: opt_<name>.itp, else the only one
        original = resolve_in_ws(ctx, original_itp)
        twin = log_path.parent / f"opt_{original.name}"
        opt_itp = twin if twin in opt_itps else (opt_itps[0] if len(opt_itps) == 1 else None)
    if original_itp and opt_itp is not None:
        before = dict(atom_charges(resolve_in_ws(ctx, original_itp).read_text()))
        after = dict(atom_charges(opt_itp.read_text()))
        common = sorted(set(before) & set(after))
        if common:
            deltas = [abs(after[a] - before[a]) for a in common]
            max_dq = round(max(deltas), 5)
            mean_dq = round(sum(deltas) / len(deltas), 5)
            n_q = len(common)

    forces: dict = {}
    residuals_csv = log_path.parent / VALIDATION_RESIDUALS_FILENAME
    if residuals_csv.exists():
        forces = parse_validation_force_residuals(residuals_csv.read_text(errors="replace"))
        if forces:
            artifacts.append(ArtifactRef.from_path(residuals_csv, "force_residuals",
                                                   root=ctx.workspace.root))

    ok = (completed and facts.get("completed", False)
          and facts.get("optimizer_success") is not False)
    warnings: list[Finding] = []
    if forces.get("outliers"):
        worst = forces["outliers"][:5]
        warnings.append(Finding(
            id="FM_FORCE_OUTLIER_ATOMS",
            message=f"{len(forces['outliers'])} QM atom(s) have a held-out force RMSE above "
            f"{FORCE_OUTLIER_FACTOR:g}× the median atom's ({forces['median_atom_rmse']} "
            f"{FORCE_UNITS}): "
            + ", ".join(f"{a['resname']}{a['resid']}:{a['name']} ({a['type']}) {a['rmse']}"
                        for a in worst)
            + (" ..." if len(forces["outliers"]) > 5 else "")
            + ". Look at these atoms' bonded terms and the QM/MM boundary before trusting "
            "the candidate.",
        ))
    # Every candidate, read against the topology it was fitted from where that can be
    # found, so a parameter that was replaced rather than refined is visible.
    pairs = fitted_pairs(top, opt_itps) if top is not None else {}
    by_finding: dict[str, list[str]] = {}
    for itp in opt_itps:
        source = next((top.parent / inc for inc, fitted in pairs.items()
                       if Path(fitted) == itp and (top.parent / inc).is_file()), None)
        rows_by_finding = implausible_parameters(
            itp.read_text(errors="replace"),
            source.read_text(errors="replace") if source is not None else None,
        )
        for finding, rows in rows_by_finding.items():
            by_finding.setdefault(finding, []).extend(f"{itp.name}: {r}" for r in rows)
    unphysical = by_finding.get("FM_UNPHYSICAL_PARAMS", [])
    for finding, rows in sorted(by_finding.items()):
        shown = "; ".join(rows[:3]) + (" ..." if len(rows) > 3 else "")
        warnings.append(Finding(
            id=finding,
            message=f"{len(rows)} fitted {PARAMETER_FINDINGS[finding]} ({shown})",
            suggestion="do not use this candidate as it stands: add reference snapshots, "
            "or restrict what the fit may change (charges only). Read the parameter "
            "tables in fm_report — validation scores can look acceptable while the "
            "parameters are not.",
        ))
    bits = []
    if unphysical:
        bits.append(f"{len(unphysical)} UNPHYSICAL bonded term(s)")
    if facts.get("n_snapshots") is not None:
        bits.append(f"{facts['n_snapshots']} snapshots")
    if facts.get("n_train") is not None:
        bits.append(f"{facts['n_train']}/{facts['n_val']} train/val")
    if facts.get("validation_sdf") is not None:
        bits.append(f"validation SDF {facts['validation_sdf']}")
    if opt_itps:
        bits.append(f"fitted itp(s): {', '.join(p.name for p in opt_itps)}")
    if max_dq is not None:
        bits.append(f"max |dq| {max_dq}")
    if forces:
        worst_type = forces["by_type"][0]
        bits.append(
            f"held-out bonded-force RMSE {forces['rmse']} {FORCE_UNITS} over "
            f"{forces['n_frames']} frames "
            f"(rel {forces['rel_rmse']}; worst type {worst_type['type']} {worst_type['rmse']})"
        )
    summary = (
        ("FM fit completed: " if ok else "FM fit NOT clean: ")
        + (", ".join(bits) if bits else "no facts parsed from fm.log")
        + (
            f". Optimizer: {facts.get('optimizer_message')}"
            if facts.get("optimizer_message")
            else ""
        )
    )
    obs = Observation(
        ok=ok,
        summary=summary,
        artifacts=artifacts,
        findings=warnings,
        n_unphysical_terms=len(unphysical),
        log_tail=None if ok else tail(log_path.read_text(errors="replace"), 25),
        n_snapshots=facts.get("n_snapshots"),
        n_train=facts.get("n_train"),
        n_val=facts.get("n_val"),
        validation_sdf=facts.get("validation_sdf"),
        validation_cost=facts.get("validation_cost"),
        optimizer_success=facts.get("optimizer_success"),
        optimizer_message=facts.get("optimizer_message"),
        qm_total_charge=facts.get("qm_total_charge"),
        wall_time_s=facts.get("wall_time_s"),
        max_abs_charge_delta=max_dq,
        mean_abs_charge_delta=mean_dq,
        n_charges=n_q,
        force_rmse=forces.get("rmse"),
        force_ref_rms=forces.get("ref_rms"),
        force_rel_rmse=forces.get("rel_rmse"),
        force_rmse_by_type=forces.get("by_type", []),
        force_outliers=forces.get("outliers", []),
        n_force_frames=forces.get("n_frames"),
    )
    if original_itp:
        orig_path = resolve_in_ws(ctx, original_itp)
        if orig_path.exists():
            obs.inputs.append(ArtifactRef.from_path(orig_path, "original_topology",
                                                    root=ctx.workspace.root))
    # the candidate topology: the fit's own topology with every fitted include swapped in
    if ok and top is not None and opt_itps:
        pairs = fitted_pairs(top, opt_itps)
        if pairs:
            cand = assemble_candidate(ctx, stage_dir, top, pairs)
            if cand.ok:
                known = {a.path for a in obs.artifacts}
                obs.artifacts += [a for a in cand.artifacts if a.path not in known]
                obs.summary += f". Candidate topology: {cand.summary}"
            else:
                obs.summary += f". No candidate topology: {cand.summary}"
        else:
            obs.summary += (". No candidate topology: none of the fitted itp(s) is named "
                            "after an include of the topology")
    return obs


def assemble_candidate(ctx: ToolContext, stage: Path, top: Path, replace: dict[str, Path],
                       output: str = "candidate.top") -> Observation:
    """The candidate topology: ``top`` copied into ``stage`` with its same-directory
    includes, each include named in ``replace`` swapped for the fitted file. Done by the
    fit's collection (``job_status`` on an ``fm_fit`` job); the result is a topology a
    comparison builds its rerun and minimisations from."""
    import shutil

    staged = stage_topology(stage, top)
    text = staged.read_text(errors="replace")
    includes = includes_of(staged)
    missing = [k for k in replace if k not in includes]
    if missing:
        return Observation(
            ok=False,
            summary=f"{missing} are not #include lines of {ctx.workspace.relpath(top)}; it "
            f"includes {includes}",
        )
    swapped = []
    for inc, fitted in replace.items():
        src = Path(fitted)
        dst = stage / src.name
        if dst.resolve() != src.resolve():
            shutil.copy(src, dst)
        text = text.replace(f'#include "{inc}"', f'#include "{src.name}"')
        swapped.append(f"{inc} -> {src.name}")
    out = stage / output
    out.write_text(text)
    if staged != out and staged.name != out.name:
        staged.unlink()
    root = ctx.workspace.root
    return Observation(
        ok=True,
        summary=f"{ctx.workspace.relpath(out)} ({'; '.join(swapped)})",
        artifacts=[ArtifactRef.from_path(out, "topology", root=root)]
        + [ArtifactRef.from_path(stage / Path(f).name, "fitted_topology", root=root)
           for f in replace.values()],
    )


@collector("fm_fit")
def _collect_fit_job(ctx: ToolContext, record, state, exit_code) -> Observation:
    meta = record.meta
    return collect_fit(
        ctx, Path(record.spec.cwd), original_itp=meta.get("original_itp"),
        topology=meta.get("topology"), completed=state == JobState.COMPLETED,
    )
