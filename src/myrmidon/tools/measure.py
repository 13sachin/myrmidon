"""Every observable through one verb: a deliverable as the mandate states it, or a stage's
property, dispatched by the property's row in knowledge/observables.py to the extractor that
measures it and resolved to the files that stage's run wrote."""

from __future__ import annotations

from pydantic import Field, PrivateAttr

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract
from myrmidon.core.lineage import latest_dataset, lineage, stage_run, stage_segments
from myrmidon.core.mandate import read_mandate
from myrmidon.core.observation import Observation
from myrmidon.core.tool import REGISTRY, ToolContext, tool
from myrmidon.core.validation import Finding
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.registry import fit_adapter
from myrmidon.knowledge.observables import (
    COUPLED,
    ENERGY_TERMS,
    PROPERTIES,
    compile_deliverable,
    measured_property,
)
from myrmidon.tools.analysis.convergence import series_settling
from myrmidon.tools.analysis.energy import energy_statistics
from myrmidon.tools.analysis.trajectory import TrajParams, measure_trajectory
from myrmidon.tools.diagnose import RunDiagnoseParams, run_diagnose
from myrmidon.tools.inputsets import system_file
from myrmidon.tools.join import segment_trajectory


class MeasureParams(TrajParams):
    stage: str = Field(default="", description="the stage whose run is measured (with property); "
                       "a deliverable's is its own")
    property: str = Field(default="", description="what to measure: rmsd, rmsf, "
                          "radius_of_gyration, min_distance, contacts, pair_distance, rdf, "
                          "thermodynamics, energy_drift, energy_terms, settling, dipole, "
                          "step_timing, qm_forces, force_rmse, boundary_bonds")
    edr: str | None = Field(default=None, description="energy_terms: an energy file (.edr)")
    terms: list[str] = Field(default_factory=list, description="energy_terms: terms as GROMACS "
                             "labels them: 'Potential', 'Total-Energy' (not 'Total Energy')")
    data: str | None = Field(default=None, description="settling: a time series, an .xvg or an "
                             ".npz a measurement wrote, or an .edr whose `series` are read")
    series: list[str] = Field(default_factory=list, description="settling: which series of the "
                              "file (legend names); empty = all; required for an .edr")
    max_drift_per_ns: float | None = Field(default=None, gt=0, description="settling: the "
                                           "researcher's criterion on the drift; unset, no verdict")
    max_halves_sigma: float | None = Field(default=None, gt=0, description="settling: the "
                                           "researcher's criterion on its half-means, in errors")
    min_settled_fraction: float = Field(default=0.5, gt=0, le=1, description="settling: no "
                                        "verdict when less than this fraction has settled")
    topology: str | None = Field(default=None, description="force_rmse: the topology whose "
                                 "forces are compared; a stage's is what it ran on")
    forces: str | None = Field(default=None, description="force_rmse: the rerun trajectory "
                               "carrying those forces; a stage's is what its run wrote")
    fmdata: str | None = Field(default=None, description="force_rmse: the reference dataset; "
                               "default: the latest collected")
    qm_selection: str | None = Field(default=None, description="force_rmse: the QM selection "
                                     "file; default: the system's")
    frames: list[int] = Field(default_factory=list, description="force_rmse: the dataset "
                              "configuration each rerun frame is; empty: frame k is k")
    label: str = Field(default="", description="force_rmse: the topology's name in tables")
    log_file: str | None = Field(default=None, description="dipole, step_timing, a coupled "
                                 "stage's thermodynamics: the run's QM log")
    #: what was set aside, the statistics asked for, the files found, or why it cannot be
    _said: list = PrivateAttr(default_factory=list)
    _statistics: list = PrivateAttr(default_factory=list)
    _read: list = PrivateAttr(default_factory=list)
    _refusal: str = PrivateAttr(default="")

    def sized(self, ctx: ToolContext) -> MeasureParams:
        """The call as what it measures is measured, so the ledger records that: the property
        it means, and for a deliverable or a stage the files that stage's run wrote."""
        named = measured_property(self.model_dump())
        update = {"property": named}
        if named in PROPERTIES and PROPERTIES[named][0] == "trajectory":
            update["metric"] = PROPERTIES[named][1]
        out = self.model_copy(update=update)
        if self.deliverable:
            return _from_deliverable(ctx, out)
        return _from_stage(ctx, out) if self.stage else out


def _extractor(mandate, stage: str, name: str) -> str:
    """Which extractor measures a property of a stage: a coupled stage's thermodynamics and
    drift are what its QM log states."""
    found = mandate.method.stage(stage) if mandate is not None else None
    if found is not None and found.regime != "mm" and name in COUPLED:
        return COUPLED[name]
    return PROPERTIES[name][0]


def _refused(params: MeasureParams, why: str) -> MeasureParams:
    out = params.model_copy()
    out._refusal = why
    return out


def _from_stage(ctx: ToolContext, params: MeasureParams) -> MeasureParams:
    """The call over a stage's latest collected run."""
    if params.property not in PROPERTIES:
        return _refused(params, f"no property {params.property!r}; known: "
                        + ", ".join(PROPERTIES))
    mandate = read_mandate(ctx.workspace)
    name = params.property
    update = {"terms": ENERGY_TERMS[name]} if name in ENERGY_TERMS and not params.terms else {}
    out = params.model_copy(update=update)
    return _on_run(ctx, out, _extractor(mandate, params.stage, name), params.stage, params.stage)


def _from_deliverable(ctx: ToolContext, params: MeasureParams) -> MeasureParams:
    """The call as the mandate measures the deliverable: its compiled parameters, on what its
    stage's latest collected run wrote, with what it set aside; or why it cannot be."""
    mandate = read_mandate(ctx.workspace)
    found = next((d for d in (mandate.deliverables if mandate else [])
                  if d.name == params.deliverable), None)
    if found is None:
        names = ", ".join(d.name for d in mandate.deliverables) if mandate else ""
        return _refused(params, f"the mandate names no deliverable {params.deliverable!r}"
                        + (f"; it names: {names}" if names else ""))
    compiled = compile_deliverable(mandate, found)
    if compiled.problems:
        return _refused(params, "; ".join(compiled.problems))
    if not found.property:
        return _refused(params, f"{found.name} is stated as a call of its own: make it")
    said = [Finding(id="DELIVERABLE_MEASURED_AS_APPROVED",
                    message=f"{key} {getattr(params, key)!r} set aside: {found.name} is "
                    f"measured with {key} {value!r}, as the mandate states")
            for key, value in compiled.params.items()
            if key in params.model_fields_set and getattr(params, key) != value]
    said += [Finding(id="DELIVERABLE_NOTE", message=m) for m in compiled.notes]
    out = params.model_copy(update=compiled.params)
    out = _on_run(ctx, out, compiled.extractor, found.over, found.name)
    out._said = [*said, *out._said]
    out._statistics = list(found.statistics)
    return out


def _on_run(ctx: ToolContext, params: MeasureParams, extractor: str, over: str, what: str
            ) -> MeasureParams:
    """The call with what its extractor reads of the stage's latest collected run: the
    trajectory and the .tpr it ran on, its energy file, its QM log, its forces. A trajectory it
    names stands, said when no run of that stage wrote it."""
    if extractor == "region" or not over:
        return params
    run = stage_run(ctx.workspace, over)
    if extractor == "trajectory" and params.trajectory:
        given = ArtifactRef.from_path(resolve_in_ws(ctx, params.trajectory), "trajectory",
                                      root=ctx.workspace.root)
        stages = lineage(ctx.workspace).stages_behind(given.sha256)
        if over in stages:
            return params
        out = params.model_copy()
        out._said = [Finding(
            id="DELIVERABLE_STAGE_MISMATCH",
            message=f"{what} is measured over stage {over}, and {params.trajectory} was written "
            "by " + (f"the run of stage {', '.join(sorted(stages))}" if stages
                     else "no run of a stage the mandate names"))]
        return out
    if run is None:
        return _refused(params, f"{what} is measured over stage {over}, and no run of it has "
                        f"been collected: stage_prepare stage={over}, stage_run it, collect it")
    outputs, inputs, runs = run
    roles = {}
    for a in outputs:
        roles.setdefault(a.role, []).append(a)
    update: dict = {}
    read: list[ArtifactRef] = []
    said: list[Finding] = []
    segments = stage_segments(ctx.workspace, over)
    if extractor == "trajectory" and len(
            {segment_trajectory(written) for _, written, _ in segments} - {None}) > 1:
        joined = REGISTRY.call("trajectory_join", ctx, {"stage": over})
        if not joined.ok:
            return _refused(params, f"{what} is measured over stage {over}'s "
                            f"{len(segments)} segments, and they could not be joined: "
                            + joined.summary)
        roles["trajectory"] = [a for a in joined.artifacts if a.role == "trajectory"]
        said.append(Finding(id="DELIVERABLE_OVER_SEGMENTS", message=f"{what} is measured over "
                            f"stage {over}'s {len(segments)} segments, joined in "
                            f"{joined.trajectory}"))

    def take(role: str, field: str) -> bool:
        found = roles.get(role)
        if not found:
            return False
        art = next((a for a in found if a.path.endswith(".xtc")), found[0])
        now = ArtifactRef.from_path(ctx.workspace.root / art.path, role, root=ctx.workspace.root)
        if now.sha256 != art.sha256:
            return False
        update[field] = art.path
        read.append(now)
        return True

    if extractor == "trajectory":
        if not take("trajectory", "trajectory"):
            return _refused(params, f"the run of stage {over} wrote no trajectory, or it changed "
                            "since: measure the run that wrote it, or run the stage again")
        tpr = next((a for a in inputs if a.role.split(".")[-1] == "tpr"), None)
        if tpr is not None and not params.structure:
            update["structure"] = tpr.path
    elif extractor == "energy":
        if not take("energy", "edr"):
            return _refused(params, f"the run of stage {over} wrote no energy file")
    elif extractor == "run_log":
        if not take("qm_log", "log_file"):
            return _refused(params, f"{params.property} is what a QM log states, and the run of "
                            f"stage {over} wrote none")
    elif extractor == "forces":
        if not take("trajectory", "forces"):
            return _refused(params, f"the run of stage {over} wrote no forces to compare")
        mandate = read_mandate(ctx.workspace)
        ran_on = next((a.path for a in inputs if a.role.split(".")[-1] == "topology"), None)
        update["topology"] = (params.topology or ran_on
                              or system_file(ctx, mandate, "topology", over))
        update["structure"] = params.structure or system_file(ctx, mandate, "structure", over)
        update["fmdata"] = params.fmdata or latest_dataset(ctx.workspace)
        update["qm_selection"] = (params.qm_selection
                                  or system_file(ctx, mandate, "qm_selection", over))
    out = params.model_copy(update=update)
    out._read = read
    if runs > 1:
        said.append(Finding(
            id="DELIVERABLE_LATEST_RUN",
            message=f"stage {over} has {runs} collected runs; {what} is measured on the latest. "
            "Measure another by naming its files"))
    out._said = said
    return out


def _measure_forces(ctx: ToolContext, params: MeasureParams) -> Observation:
    missing = [k for k in ("topology", "structure", "forces", "fmdata", "qm_selection")
               if not getattr(params, k)]
    if missing:
        return Observation(ok=False, summary=f"a force comparison needs {', '.join(missing)}: "
                           "name them, or the stage whose rerun wrote the forces")
    return fit_adapter().force_comparison(
        ctx, resolve_in_ws(ctx, params.topology), resolve_in_ws(ctx, params.structure),
        resolve_in_ws(ctx, params.forces), resolve_in_ws(ctx, params.fmdata),
        resolve_in_ws(ctx, params.qm_selection), list(params.frames), params.label)


def _measure_log(ctx: ToolContext, params: MeasureParams) -> Observation:
    obs = run_diagnose(ctx, RunDiagnoseParams(log=params.log_file or "", stage=params.stage))
    statistic = {"dipole": "dipole_debye_mean", "energy_drift": "eclassic_drift_ha_per_ps",
                 "thermodynamics": "mean_temperature_k"}.get(params.property)
    value = getattr(obs, statistic, None) if statistic else None
    if statistic and value is None and obs.ok:
        obs.ok = False
        obs.summary = (f"{params.property} is not in what this run's log states: "
                       + obs.summary)
    return obs


def _measure_region(ctx: ToolContext, params: MeasureParams) -> Observation:
    """How many boundary selections the latest QM selection made: what the region cuts."""
    found = [ev for ev in ctx.workspace.ledger.events() if ev.tool == "qm_region" and ev.ok
             and ev.kind == "call"]
    if not found:
        return Observation(ok=False, summary="no QM region has been selected in this study")
    bound = [c for c in found[-1].params.get("selections", [])
             if c.strip().lower().startswith("add-bound")]
    return Observation(ok=True, summary=f"{len(bound)} boundary selection(s): "
                       + ("; ".join(bound) or "the region cuts no bond"),
                       boundary_selections=len(bound))


_EXTRACTORS = {
    "trajectory": measure_trajectory,
    "energy": lambda ctx, p: energy_statistics(ctx, p.edr, p.terms),
    "settling": series_settling,
    "run_log": _measure_log,
    "forces": _measure_forces,
    "region": _measure_region,
}


@tool(
    "observable_measure",
    description="Measure any observable: deliverable=<name> as the mandate states it, or "
    "stage=<name> property=<name> on that stage's latest run, or a property over files you name. "
    "Harness-computed statistics, with the series saved as a file the report reads.",
    tags={"analysis"},
    contract=Contract(consumes={"structure": ("structure", "tpr"), "trajectory": "trajectory",
                                "reference": "trajectory", "edr": "energy",
                                "data": ("data", "energy"), "topology": "topology",
                                "forces": "trajectory", "fmdata": "reference_dataset",
                                "qm_selection": "qm_selection", "log_file": "qm_log"}),
)
def observable_measure(ctx: ToolContext, params: MeasureParams) -> Observation:
    if params._refusal:
        return Observation(ok=False, summary=params._refusal)
    if params.property not in PROPERTIES:
        return Observation(ok=False, summary="name the deliverable, or the property and the "
                           "stage or files it is measured over; properties: "
                           + ", ".join(PROPERTIES))
    mandate = read_mandate(ctx.workspace)
    extractor = _extractor(mandate, params.stage or next(
        (d.over for d in (mandate.deliverables if mandate else [])
         if d.name == params.deliverable), ""), params.property)
    if extractor == "trajectory" and not (params.structure and params.trajectory):
        return Observation(ok=False, summary="name the deliverable, or the structure and the "
                           "trajectory, or the stage whose run wrote them")
    if extractor == "energy" and not params.edr:
        return Observation(ok=False, summary="name the energy file, or the stage whose run "
                           "wrote it")
    if extractor == "settling" and not params.data:
        return Observation(ok=False, summary="name the series file settling is read from")
    obs = _EXTRACTORS[extractor](ctx, params)
    obs.findings = [*params._said, *obs.findings]
    obs.inputs = [*params._read, *obs.inputs]
    if params.deliverable and obs.ok:
        obs.deliverable = params.deliverable
        values = {"mean": getattr(obs, "mean", None), "max": getattr(obs, "largest", None),
                  "min": getattr(obs, "minimum", None), "sd": getattr(obs, "sd", None),
                  "final": getattr(obs, "final", None)}
        obs.statistics = {k: values[k] for k in params._statistics or values
                          if values.get(k) is not None}
        if obs.statistics:
            obs.summary = (f"{params.deliverable}: " + ", ".join(
                f"{k} {v:.3f}" for k, v in obs.statistics.items())
                + f" {getattr(obs, 'unit', '')}. " + obs.summary)
    return obs
