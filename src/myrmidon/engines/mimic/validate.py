"""fm_validate: compare candidate force fields against the QM/MM reference.

The comparison a force-matching campaign makes: given the reference dataset
(structure + aligned trajectory) and one or more candidate topologies, measure

- ``min_geometry`` — minimized monitor-pair distances vs the reference
  ensemble means (the cheap surrogate that predicted MD behaviour),
- ``md_geometry``  — monitor-pair distributions from existing candidate MD,
- ``force_rmse``   — held-out total-force RMSE on the QM atoms: candidate MM forces
  from a classical rerun over reference frames vs the dataset's QM/MM forces, per
  atom type and per atom (the number the fit's own validation SDF cannot give for a
  *stock* topology, and the direct test of whether a candidate reproduces the
  reference forces at all).

The heavy lifting runs asynchronously through the execution backend
(``validate_driver``); ``job_status`` collects the driver's JSON into a typed
observation when the job ends. Requires the ``analysis``
extra (MDAnalysis) in the execution environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import JobObservation, Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning
from myrmidon.engines.gromacs.common import gmx_bin
from myrmidon.engines.gromacs.results import register_collector
from myrmidon.engines.gromacs.run import get_jobstore
from myrmidon.engines.inputs import require_input, resolve_in_ws
from myrmidon.engines.mimic.fm import FORCE_UNITS, _stage_topology, rerun_mdp_warnings
from myrmidon.execution.backend import JobSpec, JobState, Resources
from myrmidon.text import tail

RESULTS_FILENAME = "validate_results.json"


def _gmx_data_dir(ctx: ToolContext) -> str | None:
    info = ctx.engines.get("gromacs")
    return info.data_dir if info is not None and info.data_dir else None


class EMSettings(BaseModel):
    nsteps: int = 1000
    emtol: float = 50.0


class FMValidateParams(BaseModel):
    #: the fields this submission's cost is computed from; a plan step must state them
    RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("ntomp",)

    structure: str = Field(description="reference .gro (defines atom order)")
    trajectory: str = Field(description="reference trajectory aligned to the dataset")
    topologies: dict[str, str] = Field(
        description="candidate name -> topology (.top) path", min_length=1
    )
    monitor_pairs: dict[str, tuple[int, int]] = Field(
        default_factory=dict,
        description="label -> (serial_a, serial_b), 1-based gro serials. Needed by the "
        "geometry metrics; force_rmse compares forces and needs none, so the commonest "
        "question — which of these topologies agrees best with the reference — asks for "
        "no atom serials",
    )
    metrics: list[str] | None = Field(
        default=None,
        description="min_geometry, md_geometry and/or force_rmse. Unset: follows from the "
        "inputs given — force_rmse when fmdata is given, min_geometry when monitor_pairs "
        "are, both when both are — and the observation says which it chose",
    )
    min_frames: list[int] = Field(
        default_factory=list,
        description="frame INDICES to minimize (empty = n_min_frames evenly spaced)",
    )
    n_min_frames: int = 5
    md_trajectories: dict[str, list[str]] = Field(
        default_factory=dict,
        description="candidate name -> MD trajectory paths (md_geometry metric)",
    )
    md_skip_ps: float = 500.0
    reference_stride: int = 1
    em: EMSettings = Field(default_factory=EMSettings)
    # force_rmse metric
    fmdata: str | None = Field(default=None, description="reference dataset .h5 (force_rmse)")
    selection: str | None = Field(default=None, description="QM selection file (force_rmse)")
    rerun_mdp: str | None = Field(
        default=None, description="classical rerun mdp with nstfout = 1 (force_rmse)")
    index: str | None = Field(default=None, description="optional .ndx for the rerun grompp")
    force_frames: list[int] = Field(
        default_factory=list,
        description="reference frame INDICES for the force comparison (empty = "
        "n_force_frames evenly spaced over the trajectory)",
    )
    n_force_frames: int = 50
    ntomp: int = 8
    account: str | None = Field(
        default=None,
        description=
        "scheduler account to bill for THIS submission instead of the machine config's "
        "default — the user may hold several project allocations",
    )
    stage: str = "validate"
    acknowledge_warnings: list[str] = Field(default_factory=list)

    def resources(self) -> Resources:
        """What a validation books: one task with ``ntomp`` cores, for the scheduler's
        default walltime — the request it submits, so a plan prices it the same way."""
        return Resources(ntasks=1, cpus_per_task=self.ntomp, account=self.account)

    #: True when metrics were not given and follow from the inputs (the gate says so)
    _metrics_from_inputs: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _metrics_from_what_was_given(self):
        # A fixed default metric would refuse a call that gave everything another metric
        # needs, for a metric nobody asked for. An unset choice follows the inputs.
        if self.metrics is None:
            self.metrics = ((["force_rmse"] if self.fmdata else [])
                            + (["min_geometry"] if self.monitor_pairs else []))
            self._metrics_from_inputs = True
        return self


_KNOWN_METRICS = {"min_geometry", "md_geometry", "force_rmse"}


def fm_validate_gate(ctx: ToolContext, params: FMValidateParams):
    """force_rmse needs the dataset, the selection and a classical rerun mdp that
    writes forces — checked before the queue wait, like fm_fit's gate."""
    warnings: list[ValidationWarning] = []
    checked = ["metrics_known", "monitor_pairs_for_metrics"]
    if not params.metrics:
        warnings.append(ValidationWarning(
            id="INPUT_MISSING_METRICS", severity="block",
            message="no comparison to make: metrics were not given, and neither the "
            "force inputs (fmdata) nor monitor_pairs were",
            suggestion="force_rmse compares each candidate's forces with the reference "
            "(needs fmdata, selection, rerun_mdp); min_geometry compares minimized "
            "distances (needs monitor_pairs). Give the inputs for one, or name it in "
            "metrics",
            needs=["fmdata", "monitor_pairs", "metrics"]))
        return warnings, checked
    if params._metrics_from_inputs:
        warnings.append(ValidationWarning(
            id="METRICS_FROM_INPUTS", severity="info",
            message=f"metrics not given: comparing by {', '.join(params.metrics)}",
            assumption="force_rmse because fmdata was given, min_geometry because "
            "monitor_pairs were; name metrics to choose otherwise"))
    geometry = [m for m in params.metrics if m in ("min_geometry", "md_geometry")]
    if geometry and not params.monitor_pairs:
        warnings.append(ValidationWarning(
            id="INPUT_MISSING_MONITOR_PAIRS", severity="block",
            message=f"{', '.join(geometry)} compares distances, so it needs "
            "monitor_pairs — label -> (serial_a, serial_b), 1-based serials in the "
            "reference .gro",
            suggestion="drop the geometry metric to compare forces alone, or name the "
            "pairs the study is about (a bond being fitted, a metal-donor distance)",
            needs=["monitor_pairs", "metrics"]))
    if "force_rmse" not in params.metrics:
        return warnings, checked
    checked.append("force_rmse_inputs")
    #: where each force_rmse input comes from, for a caller who has not run a campaign:
    #: a missing artifact that names the tool that makes it is a step, not a dead end
    HOW = {
        "fmdata": "the QM/MM reference dataset (.h5) — fm_collect registers one, or "
                  "cite the reference study that produced it",
        "selection": "the QM selection file — mimic_select_qm writes one",
        "rerun_mdp": "the classical rerun mdp — render it with inputset_apply "
                     "inputset=fm_classical_rerun output=<stage>/rerun.mdp (it sets "
                     "nstfout=1, which is the point of the rerun)",
    }
    for label, path in (("fmdata", params.fmdata), ("selection", params.selection),
                        ("rerun_mdp", params.rerun_mdp)):
        if not path:
            warnings.append(ValidationWarning(
                id=f"INPUT_MISSING_{label.upper()}", severity="block",
                message=f"force_rmse needs '{label}' — not given",
                suggestion=HOW[label], needs=[label]))
        else:
            _, missing = require_input(ctx, label, path)
            if missing:
                warnings.append(missing)
    if params.rerun_mdp and not any(w.id == "INPUT_MISSING_RERUN_MDP" for w in warnings):
        from myrmidon.engines.gromacs.mdp import parse_mdp_file

        checked.append("rerun_mdp_classical")
        warnings.extend(rerun_mdp_warnings(parse_mdp_file(resolve_in_ws(ctx, params.rerun_mdp))))
    return warnings, checked


@tool(
    "fm_validate",
    description="Validate candidate force-field topologies against the QM/MM "
    "reference: minimized monitor-pair geometry (min_geometry) and/or MD "
    "distance distributions (md_geometry), and/or the held-out total-force RMSE on the "
    "QM atoms per atom type (force_rmse: classical rerun of each candidate vs the "
    "dataset's QM/MM forces). Async: job_status collects the metrics when the job ends.",
    returns=Observation,
    tags={"mimic", "fm", "validate"},
    requires={"gromacs"},
    expensive=True,
    gate=fm_validate_gate,
)
def fm_validate(ctx: ToolContext, params: FMValidateParams) -> Observation:
    unknown = set(params.metrics) - _KNOWN_METRICS
    if unknown:
        return Observation(
            ok=False,
            summary=f"unknown metrics {sorted(unknown)}; supported: "
            f"{sorted(_KNOWN_METRICS)}",
        )
    if "md_geometry" in params.metrics and not params.md_trajectories:
        return Observation(
            ok=False,
            summary="md_geometry requested but md_trajectories is empty — "
            "provide candidate MD trajectories or drop the metric",
        )
    stage = ctx.workspace.stage(params.stage)
    candidates = {}
    for name, top in params.topologies.items():
        cand_dir = stage / "top" / name
        cand_dir.mkdir(parents=True, exist_ok=True)
        candidates[name] = str(_stage_topology(cand_dir, resolve_in_ws(ctx, top)))
    driver_params = {
        "stage": str(stage),
        "structure": str(resolve_in_ws(ctx, params.structure)),
        "trajectory": str(resolve_in_ws(ctx, params.trajectory)),
        "candidates": candidates,
        "monitor_pairs": {k: list(v) for k, v in params.monitor_pairs.items()},
        "metrics": params.metrics,
        "min_frames": params.min_frames,
        "n_min_frames": params.n_min_frames,
        "md_trajectories": {
            name: [str(resolve_in_ws(ctx, t)) for t in trajs]
            for name, trajs in params.md_trajectories.items()
        },
        "md_skip_ps": params.md_skip_ps,
        "reference_stride": params.reference_stride,
        "em": params.em.model_dump(),
        "fmdata": str(resolve_in_ws(ctx, params.fmdata)) if params.fmdata else None,
        "selection": str(resolve_in_ws(ctx, params.selection)) if params.selection else None,
        "rerun_mdp": str(resolve_in_ws(ctx, params.rerun_mdp)) if params.rerun_mdp else None,
        "index": str(resolve_in_ws(ctx, params.index)) if params.index else None,
        "force_frames": params.force_frames,
        "n_force_frames": params.n_force_frames,
        "gmx_data": _gmx_data_dir(ctx),
        "gmx": gmx_bin(ctx),
        "ntomp": params.ntomp,
        "out": RESULTS_FILENAME,
    }
    params_file = stage / "validate_params.json"
    params_file.write_text(json.dumps(driver_params, indent=1) + "\n")
    spec = JobSpec(
        argv=[sys.executable, "-m", "myrmidon.engines.mimic.validate_driver",
              str(params_file)],
        cwd=stage,
        stdout="validate.stdout",
        stderr="validate.stderr",
        resources=params.resources(),
    )
    handle = ctx.backend.submit(spec)
    job_id = get_jobstore(ctx).add(
        ctx.backend.name, handle, spec, label="fm_validate",
        meta={"kind": "fm_validate", "stage": params.stage})
    return JobObservation(
        ok=True,
        summary=f"validation submitted as {job_id}: metrics "
        f"{params.metrics} on {len(candidates)} candidate(s), "
        f"{len(params.monitor_pairs)} monitor pair(s). Poll with job_status: once it has "
        "ended it returns the per-candidate tables and MAE vs the reference.",
        command=spec.argv,
        job_id=job_id,
        state="QUEUED",
    )


class FMValidationObservation(Observation):
    job_id: str = ""
    state: str = ""
    exit_code: int | None = None
    core_h_estimate: float | None = None
    reference: dict = Field(default_factory=dict)
    min_geometry: dict = Field(default_factory=dict)
    md_geometry: dict = Field(default_factory=dict)
    mae: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    # force_rmse metric: candidate -> {rmse, ref_rms, rel_rmse, by_type, outliers, ...}
    force_rmse: dict[str, dict] = Field(default_factory=dict)
    force_units: str = FORCE_UNITS


def force_type_table(blocks: dict[str, dict]) -> str:
    """Per-atom-type force RMSE, candidates side by side (rows sorted by the first
    candidate's RMSE; the reference RMS is the same for all candidates)."""
    names = [n for n, b in blocks.items() if b.get("by_type")]
    if not names:
        return ""
    per = {n: {t["type"]: t for t in blocks[n]["by_type"]} for n in names}
    types = [t["type"] for t in blocks[names[0]]["by_type"]]
    lines = [f"{'type':10s} {'n':>4s} {'ref RMS':>10s}" + "".join(f" {n[:14]:>14s}" for n in names)]
    for t in types:
        first = per[names[0]][t]
        row = f"{t:10s} {first['n_atoms']:>4d} {first['ref_rms']:>10.4f}"
        for n in names:
            v = per[n].get(t)
            row += f" {v['rmse']:>14.4f}" if v else f" {'—':>14s}"
        lines.append(row)
    row = f"{'overall':10s} {'':>4s} {blocks[names[0]].get('ref_rms', 0.0):>10.4f}"
    for n in names:
        row += f" {blocks[n]['rmse']:>14.4f}"
    lines.append(row)
    return "\n".join(lines)


def _table(ref: dict, blocks: dict[str, dict]) -> str:
    names = list(blocks)
    lines = [
        f"{'pair':16s} {'reference':>12s}"
        + "".join(f" {n[:14]:>14s}" for n in names)
    ]
    for label, stats in ref.items():
        row = f"{label:16s} {stats['mean']:>12.3f}"
        for n in names:
            pair = (blocks[n].get("pairs") or {}).get(label)
            row += f" {pair['mean']:>14.3f}" if pair else f" {'—':>14s}"
        lines.append(row)
    row = f"{'MAE':16s} {'':>12s}"
    for n in names:
        mae = blocks[n].get("mae")
        row += f" {mae:>14.3f}" if mae is not None else f" {'—':>14s}"
    lines.append(row)
    return "\n".join(lines)


def collect_validation(ctx: ToolContext, stage: Path, *,
                       completed: bool = True) -> FMValidationObservation:
    """A validation job that has ended, read into per-candidate monitor-pair tables and
    MAE vs the QM/MM reference ensemble. This is what ``job_status`` returns for an
    ``fm_validate`` job."""
    results_path = stage / RESULTS_FILENAME
    if not results_path.exists():
        log = stage / "validate_driver.log"
        return FMValidationObservation(
            ok=False,
            summary=f"no {RESULTS_FILENAME} in {ctx.workspace.relpath(stage)}: the "
            + ("validation driver wrote no results" if completed
               else "job did not complete"),
            log_tail=tail(log.read_text(), 20) if log.exists() else None,
        )
    data = json.loads(results_path.read_text())
    if not data.get("ok"):
        log_tail = None
        log = stage / "validate_driver.log"
        if log.exists():
            log_tail = tail(log.read_text(), 20)
        return FMValidationObservation(
            ok=False,
            summary=f"validation driver failed: {data.get('error', 'unknown')}",
            log_tail=log_tail,
        )
    ref = data["reference"]
    parts = []
    mae: dict[str, dict[str, float | None]] = {}
    for metric in ("min_geometry", "md_geometry"):
        block = data.get(metric)
        if block:
            parts.append(f"[{metric}]\n" + _table(ref, block))
            mae[metric] = {n: block[n].get("mae") for n in block}
    ranked = [
        f"{metric}: " + ", ".join(
            f"{n}={v:.3f}" if v is not None else f"{n}=?"
            for n, v in sorted(vals.items(), key=lambda kv: (kv[1] is None, kv[1]))
        )
        for metric, vals in mae.items()
    ]
    forces = data.get("force_rmse") or {}
    if forces:
        good = {n: b for n, b in forces.items() if b.get("rmse") is not None}
        failed = [n for n in forces if n not in good]
        ranked.append(
            f"force_rmse ({FORCE_UNITS}, {len(data.get('force_frames', []))} frames): "
            + ", ".join(f"{n}={b['rmse']:.4f} (rel {b['rel_rmse']:.3f})" for n, b in
                        sorted(good.items(), key=lambda kv: kv[1]["rmse"]))
            + (f"; FAILED: {', '.join(failed)}" if failed else ""))
        table = force_type_table(good)
        if table:
            parts.append(f"[force_rmse per atom type, {FORCE_UNITS}]\n" + table)
        for n, b in good.items():
            if b.get("outliers"):
                parts.append(f"{n}: {len(b['outliers'])} outlier atom(s) — " + ", ".join(
                    f"{a['resname']}{a['resid']}:{a['name']} {a['rmse']:.4f}"
                    for a in b["outliers"][:5]))
    return FMValidationObservation(
        ok=True,
        summary="validation vs reference — " + "; ".join(ranked)
        + f". Reference: {data['reference_frames']} frames.\n"
        + "\n\n".join(parts),
        reference=ref,
        min_geometry=data.get("min_geometry", {}),
        md_geometry=data.get("md_geometry", {}),
        mae=mae,
        force_rmse=forces,
        artifacts=[ArtifactRef.from_path(results_path, "validation",
                                         root=ctx.workspace.root)],
    )


@register_collector("fm_validate")
def _collect_validate_job(ctx: ToolContext, record, state, exit_code) -> FMValidationObservation:
    return collect_validation(ctx, Path(record.spec.cwd), completed=state == JobState.COMPLETED)
