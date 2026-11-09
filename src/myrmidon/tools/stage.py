"""A mandate stage as one step: what its run reads, compiled in one call; and its run, submitted
in one call, whichever engine and job shape the stage comes to."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef, fresh_path, sha256_file
from myrmidon.core.contracts import Contract, includes_of
from myrmidon.core.lineage import stage_files, stage_left, stage_parameters
from myrmidon.core.lineage import stage_run as collected_run
from myrmidon.core.mandate import read_mandate
from myrmidon.core.observation import Observation, ToolError
from myrmidon.core.tool import PREVIEW_HELP, REGISTRY, ToolContext, precheck, tool
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.registry import (
    Layout,
    adapter,
    adapter_running,
    adapter_with,
    adapters,
    run_kinds,
)
from myrmidon.execution.backend import Resources
from myrmidon.execution.jobs import submit_job
from myrmidon.knowledge.inputsets import RENDERED_FILES
from myrmidon.knowledge.stage import TemplateStore, templates_for
from myrmidon.tools.inputsets import system_file
from myrmidon.tools.qm_region import region_selections

adapters()  # each engine registers what collects its jobs


class StagePrepareParams(BaseModel):
    stage: str = Field(description="a stage the study's mandate names (method.stages)")
    structure: str | None = Field(
        default=None, description="what the stage starts from; default: the structure the run "
        "it starts from left, or for a coupled stage and the first stage the system's")
    topology: str | None = Field(default=None, description="default: the one the stage's "
                                 "parameters_from names, else the system's; another, to run "
                                 "this one render on it")
    index: str | None = Field(default=None, description="default: the system's")
    restraint_reference: str | None = Field(
        default=None, description="coordinates position restraints hold atoms at; default: the "
        "structure the stage starts from")
    overrides: dict = Field(default_factory=dict, description="engine keys to set yourself, "
                            "each recorded as origin 'user' and said; every other value is "
                            "the mandate's, the method's or the system's")
    max_warn: int = Field(default=0, description="warnings the compile of the run input may "
                          "suppress: each is a physics decision")


def _starting_structure(ctx: ToolContext, mandate, stage) -> tuple[str | None, str]:
    """(the structure a stage's run input is compiled against, why when there is none yet): the
    one the run it starts from left; the first stage's, the system's. A stage extending itself
    starts from what the stage before it left. A coupled stage continuing a coupled one takes its
    coordinates from the restart, so it compiles against what the first of them did: the last
    classical structure before the chain. A parameter fit moves no atom: a stage after one starts
    on the structure the fit reads."""
    seen: set[str] = {stage.name}
    names = [s.name for s in mandate.method.stages]
    begins = mandate.method.starts_from(stage)
    if begins is not None and begins.name == stage.name:
        at = names.index(stage.name)
        begins = mandate.method.stages[at - 1] if at else None
    if begins is not None and begins.kind == "parameter_fit":
        reads = mandate.method.starts_from(begins) or begins  # as the fit's run reads it
        return system_file(ctx, mandate, "structure", reads.name), ""
    while stage.regime != "mm" and begins is not None and begins.regime != "mm":
        if begins.name in seen:
            begins = None
            break
        seen.add(begins.name)
        begins = mandate.method.starts_from(begins)
    if begins is None:
        return system_file(ctx, mandate, "structure"), ""
    run = collected_run(ctx.workspace, begins.name)
    left = next((a.path for a in (run[0] if run else []) if a.role == "structure"), None)
    return left, "" if left else (f"{begins.name}'s run has not been collected, and this stage "
                                  "starts from the structure it leaves")


def _prepare_findings(ctx: ToolContext, mandate, stage, structure: str | None) -> list[Finding]:
    """What each engine says stops the stage's render on these coordinates."""
    return [f for a in adapters() if a.prepare_findings
            for f in a.prepare_findings(ctx, mandate, stage, structure)]


def _partition(ctx: ToolContext, mandate, stage, structure: str | None,
               topology: str | None) -> Observation | None:
    """The stage's QM region partitioned by the harness from what the stage states (its
    subsystems; the padding and pseudopotentials its engine resolves from the stage and its QM
    method), as a ``qm_region`` call of its own in the ledger: on the first coupled stage of a
    topology, and again on a fresh coupled start from other coordinates, another topology or
    another padding than the stage states. None when the stage reads a partition already: the
    researcher's own selection (``qm_region`` called with selections of their own, or a
    selection file the mandate holds), the restart of the coupled stage it continues, which
    holds both halves, or one made from the same subsystems on this structure and topology at
    the padding the stage states, if it states one."""
    region = stage.qm_region
    engine = adapter_with("partition_qm")
    if (region is None or region.charge is None or stage.regime == "mm" or not structure
            or engine is None):
        return None  # a region with no charge is refused before anything is partitioned
    begins = mandate.method.starts_from(stage)
    if begins is not None and begins.regime != "mm" and begins.name != stage.name:
        return None
    ws = ctx.workspace
    topology = topology or system_file(ctx, mandate, "topology", stage.name)
    if not topology or not ws.resolve(structure).is_file():
        return None
    made = [ev for ev in ws.ledger.events()
            if ev.tool == "qm_region" and ev.ok and ev.kind == "call"]
    if not made and any(f.kind == "qm_selection" for f in mandate.system.files):
        return None
    padding, table = (engine.region_settings(mandate, stage) if engine.region_settings
                      else (region.padding_nm, ""))
    wanted = region_selections(ctx, topology, list(region.subsystems))[0]
    latest = next((ev for ev in reversed(made) if ev.params.get("stage") == stage.name),
                  made[-1] if made else None)
    if latest is not None:
        given = latest.params.get("selections")
        if given != region_selections(ctx, topology, latest.params.get("subsystems") or [])[0]:
            return None  # commands of the researcher's own: read as they are
        read = {i.role: i for i in latest.inputs}
        stated = region.padding_nm
        if (given == wanted and (not stated or latest.params.get("padding_nm") == stated)
                and read.get("coords") and read["coords"].sha256
                == sha256_file(ws.resolve(structure))
                and read.get("topology") and read["topology"].sha256
                == sha256_file(ws.resolve(topology))):
            return None
    call = {"topology": topology, "coords": structure, "subsystems": list(region.subsystems),
            "stage": stage.name, **({"padding_nm": padding} if padding else {})}
    if table:
        path = fresh_path(ws.stage(stage.name) / "pp_info.dat")
        path.write_text(table)
        call["pp_info"] = ws.relpath(path)
    return REGISTRY.call("qm_region", ctx, call)


def _refused(summary: str, code: str, findings: list, hint: str = "", **fields) -> Observation:
    return Observation(ok=False, summary=summary, findings=findings,
                       error=ToolError(code=code, message=summary, hint=hint), **fields)


@tool(
    "stage_prepare",
    description="Compile a mandate stage into everything its run reads, in one call: its "
    "engine files (each value's origin in a sidecar), complete with what the system holds, "
    "and the compiled run input; the engine's own checks come back as findings.",
    tags={"mandate", "run"},
    expensive=True,
    contract=Contract(consumes={"structure": "structure", "topology": "topology",
                                "index": "index", "restraint_reference": "structure"},
                      reads={"mdp": "mdp"}),
)
def stage_prepare(ctx: ToolContext, params: StagePrepareParams) -> Observation:
    mandate = read_mandate(ctx.workspace)
    stage = mandate.method.stage(params.stage) if mandate is not None else None
    if stage is None:
        names = ", ".join(s.name for s in mandate.method.stages) if mandate else ""
        return _refused(f"the mandate names no stage {params.stage!r}"
                        + (f"; its stages: {names}" if names else ""),
                        "MANDATE_STAGE_UNKNOWN", [])
    templates = templates_for(stage.model_dump(mode="json"), TemplateStore.default())
    if not templates:
        return _refused(f"no template tells an engine stage {stage.name} ({stage.kind}, "
                        f"{stage.regime}, {stage.ensemble})", "MANDATE_STAGE_TEMPLATE", [])
    structure, later = _starting_structure(ctx, mandate, stage)
    structure = params.structure or structure
    topology, waits = ((params.topology, "") if params.topology
                       else stage_parameters(ctx.workspace, mandate, stage))
    later = later or waits
    partitioned = _partition(ctx, mandate, stage, structure, topology)
    if partitioned is not None and not partitioned.ok:
        return _refused(f"stage {stage.name}'s QM region could not be partitioned: "
                        + partitioned.summary,
                        partitioned.error.code if partitioned.error else "QM_PARTITION_FAILED",
                        partitioned.findings)
    stopping = _prepare_findings(ctx, mandate, stage, structure)
    if stopping:
        return _refused(stopping[0].message, stopping[0].id, stopping,
                        stopping[0].suggestion or "")
    index = params.index or system_file(ctx, mandate, "index", stage.name)
    stage_dir = ctx.workspace.stage(stage.name)
    prepared: dict[str, str] = {}
    values: dict[str, dict] = {}
    findings: list = [*(partitioned.findings if partitioned is not None else [])]
    for template in templates:
        engine = adapter(template.engine)
        output = stage_dir / (template.file_name
                              or f"{stage.name}{RENDERED_FILES[template.renders][1]}")
        rendered = REGISTRY.call("inputset_apply", ctx, {
            "stage": stage.name, "output": ctx.workspace.relpath(output),
            "structure": structure, "topology": topology, "index": index,
            "overrides": {k: v for k, v in params.overrides.items() if engine.reads_key(k)}})
        findings += rendered.findings
        if not rendered.ok:
            return _refused(rendered.summary, rendered.error.code if rendered.error
                            else "STAGE_RENDER_FAILED", findings,
                            rendered.error.hint if rendered.error else "", files=prepared)
        kind = RENDERED_FILES[template.renders][0]
        prepared[kind] = ctx.workspace.relpath(output)
        values[kind] = rendered.values
    checked = _protocol_rules(ctx, mandate, stage, values)
    broken = [r for r in checked if r.status == "violated"]
    if broken:
        return _refused(f"{broken[0].id}: {broken[0].message} ({broken[0].detail})",
                        broken[0].id, [Precondition(id=r.id, message=f"{r.message} ({r.detail})")
                                       for r in broken], files=prepared)
    obs = _with_run_input(ctx, params, stage, stage_dir, templates, prepared, findings,
                          {"structure": structure, "topology": topology, "index": index,
                           "restraint_reference": params.restraint_reference}, later)
    obs.values = values
    obs.invariants = [r.model_dump() for r in checked]
    if partitioned is not None:  # what the selection the harness made holds, to review
        obs.partition = {key: getattr(partitioned, key, None) for key in (
            "selections", "qm_atoms", "declared_charge", "derived_multiplicity", "electrons",
            "guessed_elements", "min_vacuum_bohr", "qm_cell_bohr", "pseudopotentials")}
    obs.quiet = {"values"}  # each value and its origin is in the sidecar beside its file
    return obs


def _protocol_rules(ctx: ToolContext, mandate, stage, values: dict[str, dict]) -> list:
    """The rules of the mandate's protocol across its stages, over every file this stage
    just rendered and what the other stages already rendered."""
    from myrmidon.tools.inputsets import protocol_of, rendered_by_stage

    protocol = protocol_of(ctx, mandate)
    if protocol is None or not protocol.invariants:
        return []
    known = rendered_by_stage(ctx)
    known[stage.name] = {k: v for rendered in values.values() for k, v in rendered.items()}
    return protocol.check_invariants(known)


def _with_run_input(ctx: ToolContext, params: StagePrepareParams, stage, stage_dir: Path,
                    templates: list, prepared: dict[str, str], findings: list,
                    system: dict[str, str | None], later: str) -> Observation:
    """The stage's rendered files, with the run input the engine compiles from them when it
    has one and what it needs is there: what the compile read is this call's input."""
    root = ctx.workspace.root
    inputs: list[ArtifactRef] = []
    artifacts: list[ArtifactRef] = []
    command = None
    for template in templates:
        engine = adapter(template.engine)
        if engine.compile_run_input is None or template.auxiliary:
            continue
        kind = RENDERED_FILES[template.renders][0]
        files = {k: resolve_in_ws(ctx, v) for k, v in system.items() if v}
        compiled = engine.compile_run_input(ctx, ctx.workspace.resolve(prepared[kind]), files,
                                            stage_dir, stage.name, params.max_warn)
        if compiled is None:
            findings.append(_later(stage.name, later or "the stage's structure or topology is "
                                   "not among the system's files"))
            continue
        findings += compiled.findings
        if not compiled.ok:
            return _refused(compiled.summary, compiled.error.code, findings, compiled.error.hint,
                            files=prepared, log_tail=compiled.log_tail,
                            command=compiled.command)
        artifacts += compiled.artifacts
        command = compiled.command
        prepared |= {a.role: a.path for a in compiled.artifacts}
        inputs += [ArtifactRef.from_path(ctx.workspace.resolve(prepared[kind]), kind, root=root),
                   *(ArtifactRef.from_path(path, k, root=root) for k, path in files.items())]
        if "topology" in files:  # what the topology includes, the engine reads with it
            # the includes of the study, its project and the project's other studies, not the
            # engine's force-field library
            own = [root.resolve(), *([Path(ctx.workspace.origin).resolve()]
                                     if ctx.workspace.origin else [])]
            inputs += [ArtifactRef.from_path(path, "topology.include", root=root)
                       for path in includes_of("topology", files["topology"])
                       if any(path.resolve().is_relative_to(d) for d in own)
                       or ctx.workspace.sibling_path(path)]
    return Observation(
        ok=True,
        summary=f"stage {stage.name} prepared: " + ", ".join(sorted(prepared.values())),
        files=prepared, findings=findings, artifacts=artifacts, inputs=inputs, command=command,
        refs=[f"stage:{stage.name}"])


def _later(stage: str, why: str):
    return Finding(id="STAGE_RUN_INPUT_LATER",
                   message=f"stage {stage}'s run input is not compiled yet: {why}; prepare the "
                   "stage again once it is there")


class StageRunParams(BaseModel):
    #: the fields this submission's cost is computed from
    RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("cores", "walltime_min", "partition")

    preview: bool = Field(default=False, description=PREVIEW_HELP)
    stage: str = Field(description="a stage the study's mandate names; stage_prepare it first")
    walltime_min: int | None = Field(
        default=None, ge=1, description="scheduler time limit; unset, sized from the stage's "
        "length at the pace measured on this machine, within what the partition grants")
    cores: int | None = Field(
        default=None, ge=1, description="cores to divide among the engine's ranks and threads; "
        "unset, one node of the partition, which is what a whole-node partition bills")
    partition: str | None = Field(default=None, description="scheduler partition for this run "
                                  "(default: the machine config's)")
    account: str | None = Field(default=None, description="scheduler account to bill for this "
                                "run, instead of the machine config's")
    fit_mode: Literal["both", "charges", "bonded"] = Field(
        default="both", description="a parameter_fit: what it fits, charges (DRESP), bonded "
        "parameters, or both; a stage that states its fit_mode runs that")
    continue_run: bool = Field(
        default=False, description="go on from what this stage's last run left: a run stopped at "
        "its walltime, or one a stage that starts from itself extends")
    extend_ps: float | None = Field(default=None, gt=0, description="with continue_run: how many "
                                    "ps further than the run's end; unset, to the end it states")
    frames: list[int] = Field(default_factory=list, description="a stage of single points over "
                              "frames: the frames this run computes (the first alone, a probe); "
                              "unset, every frame of its scan no run has computed")
    snapshot: str | None = Field(default=None, description="a geometry snapshot_select wrote, for "
                                 "a coupled stage to start from in place of the restart's")
    inputs: dict[str, str] = Field(
        default_factory=dict, description="files by kind, each read in place of what the ledger "
        "says the stage reads (trajectory, restart, geometry, ...); the rest are resolved")
    layout: dict = Field(default_factory=dict, description="filled by the harness: the ranks and "
                         "threads the engine's rules chose, and why")

    def resources(self) -> Resources:
        """What the run books: its layout's ranks and threads for its walltime."""
        return Resources(
            ntasks=(self.layout.get("ntasks", 1) + self.layout.get("partner_ntasks", 0)),
            cpus_per_task=self.layout.get("cpus_per_task", 1),
            walltime_min=self.walltime_min or 60,
            partition=self.partition, account=self.account)

    def sized(self, ctx: ToolContext) -> StageRunParams:
        """The call with what it leaves to the harness filled: the files the ledger says the
        stage reads (so the gate holds each to the mandate), its layout on these cores, and its
        walltime within what the partition grants."""
        mandate = read_mandate(ctx.workspace)
        stage = mandate.method.stage(self.stage) if mandate is not None else None
        engine = _engine_of(stage)
        if engine is None:
            return self
        begins = mandate.method.starts_from(stage)
        # a stage that starts from itself extends a run that left something to extend; until one
        # has, its run is a first run
        continuing = self.continue_run or (
            begins is not None and begins.name == stage.name
            and any(a.role in ("checkpoint", "restart")
                    for a in stage_left(ctx.workspace, stage.name)))
        inputs = {**stage_files(ctx.workspace, stage.name), **self.inputs}
        inputs |= {**engine.stage_inputs(ctx, mandate, stage, inputs, continuing), **self.inputs}
        snapshot = self.snapshot or self.inputs.get("geometry")
        if snapshot:
            inputs["geometry"] = snapshot
        inputs = {k: v for k, v in inputs.items() if k in run_kinds() or k in self.inputs}
        files = {k: ctx.workspace.resolve(v) for k, v in inputs.items()}
        _, node = ctx.config.compute.billing(self.partition)
        cores = self.cores or node.cores_per_node
        found = engine.resolve_layout(cores, engine.layout_facts(files, stage))
        minutes, why = self.walltime_min, ""
        if minutes is None:
            minutes, why = engine.size_walltime(ctx, stage, files, found, found.cores,
                                                self.extend_ps, continuing, self.frames)
            minutes = minutes or ctx.config.compute.unmeasured_walltime_min
            if node.max_walltime_min and minutes > node.max_walltime_min:
                minutes = node.max_walltime_min
                why += f"; capped at the {minutes} min the partition grants: continue the stage"
        # a scan's run names the frames it computes, so a run beside it computes others
        frames = (engine.scan_frames(ctx, stage, self.frames) if engine.scan_frames
                  else None)
        return self.model_copy(update={
            # what the stage states the fit moves, unless the call says otherwise, which is
            # refused
            "fit_mode": (self.fit_mode if "fit_mode" in self.model_fields_set
                         else stage.fit_mode or self.fit_mode),
            "frames": self.frames if frames is None else frames,
            "inputs": inputs, "cores": cores, "walltime_min": minutes,
            "continue_run": continuing, "snapshot": snapshot,
            "layout": {"ntasks": found.ntasks, "cpus_per_task": found.cpus_per_task,
                       "partner_ntasks": found.partner_ntasks, "why": found.why,
                       "walltime_why": why}})


def _engine_of(stage):
    """The adapter that runs this stage; None for no stage or one no template tells."""
    if stage is None:
        return None
    templates = templates_for(stage.model_dump(mode="json"), TemplateStore.default())
    return adapter_running([t.engine for t in templates if not t.auxiliary])


def _run_findings(ctx: ToolContext, params: StageRunParams) -> list[Finding]:
    """What stops the stage's run: no such stage, or what its engine says of its inputs."""
    mandate = read_mandate(ctx.workspace)
    stage = mandate.method.stage(params.stage) if mandate is not None else None
    if stage is None:
        names = ", ".join(s.name for s in mandate.method.stages) if mandate else ""
        return [Precondition(id="MANDATE_STAGE_UNKNOWN",
                             message=f"the mandate names no stage {params.stage!r}"
                             + (f"; its stages: {names}" if names else ""),
                             needs=["stage"])]
    engine = _engine_of(stage)
    if engine is None:
        return [Precondition(id="MANDATE_STAGE_TEMPLATE",
                             message=f"no engine here runs stage {stage.name} ({stage.kind}, "
                             f"{stage.regime})")]
    files = {k: ctx.workspace.resolve(v) for k, v in params.inputs.items()}
    unknown = sorted(set(files) - run_kinds())
    if unknown:
        return [Precondition(id="STAGE_INPUT_UNKNOWN",
                             message=f"a stage's run reads no {', '.join(unknown)}; it reads "
                             + ", ".join(sorted(run_kinds())), needs=["inputs"])]
    missing = [Precondition(id=f"INPUT_MISSING_{kind.upper()}",
                            message=f"{kind} not found: {params.inputs[kind]}")
               for kind, path in files.items() if not path.is_file()]
    if missing:
        return missing
    scanned = stage.scan is not None and stage.scan.parameter == "frames"
    outside = sorted({int(f) for f in params.frames}
                     - ({int(f) for f in stage.scan.values} if scanned else set()))
    if outside:
        return [Precondition(id="STAGE_FRAMES_NOT_SCANNED",
                             message=f"frames {outside} are none of stage {stage.name}'s scan",
                             suggestion="name frames the stage's scan holds, or revise the "
                             "mandate's scan", needs=["frames"])]
    if scanned and engine.scan_frames and not params.frames \
            and engine.scan_frames(ctx, stage, []) == []:
        return [Precondition(
            id="STAGE_NOTHING_TO_RUN",
            message=f"every frame of stage {stage.name}'s scan has been computed, or is in a "
            "job not yet collected", needs=["frames"],
            suggestion=f"job_status on a running one; dataset_collect stage={stage.name}; or "
            "name frames to compute again")]
    if stage.fit_mode and params.fit_mode != stage.fit_mode:
        return [Precondition(
            id="STAGE_FIT_MODE",
            message=f"stage {stage.name} fits {stage.fit_mode}, as the mandate states; this "
            f"call asks for {params.fit_mode}", needs=["fit_mode"],
            suggestion="leave fit_mode out, or revise the mandate with the researcher's reason")]
    if params.extend_ps and not params.continue_run:
        return [Precondition(
            id="STAGE_EXTEND_WITHOUT_CONTINUE",
            message="extend_ps goes further than a run went, and this call starts one afresh",
            suggestion="continue_run: true, to extend the stage's last run",
            needs=["continue_run"])]
    if params.continue_run and not {"checkpoint", "restart"} & set(files):
        return [Precondition(
            id="STAGE_NOTHING_TO_CONTINUE",
            message=f"stage {stage.name} has left nothing to continue from: no run of it wrote a "
            "checkpoint or a restart, or its job has not been collected",
            suggestion=f"job_status on its job, or stage_run stage={stage.name} from the start")]
    return [*_source_findings(ctx, params, files), *engine.job_findings(
        ctx, stage, files, params.walltime_min or 60, params.extend_ps, params.continue_run)]


def _source_findings(ctx: ToolContext, params: StageRunParams, files: dict) -> list[Finding]:
    """Where the parameter files the run reads came from, and what was changed in them since
    they were rendered: said before the job, so a hand's edit is on the record."""
    from myrmidon.core.contracts import parameter_kinds
    from myrmidon.core.sourcing import parameter_source, sourced_finding, unsourced_finding
    from myrmidon.engines.file_origin import edit_findings

    named = [params.inputs[k] for k in files if k in parameter_kinds()]
    sources = [parameter_source(ctx.workspace, path) for path in named]
    found = [f for f in (unsourced_finding(sources), sourced_finding(sources)) if f is not None]
    for path in named:
        found += edit_findings(ctx, path)
    return found


@tool(
    "stage_run",
    description="Submit a mandate stage's run under the approved mandate: its compiled inputs and "
    "starting state come from the ledger, its ranks and threads from the engine's rules, its "
    "walltime from its length. Returns a job_id; job_status collects it.",
    tags={"mandate", "run"},
    expensive=True,
    contract=Contract(by_kind="inputs", consumes={"snapshot": "geometry"}),
)
def stage_run(ctx: ToolContext, params: StageRunParams) -> Observation:
    precheck(ctx, _run_findings(ctx, params))
    mandate = read_mandate(ctx.workspace)
    stage = mandate.method.stage(params.stage)
    engine = _engine_of(stage)
    files = {k: ctx.workspace.resolve(v) for k, v in params.inputs.items()}
    layout = Layout(params.layout["ntasks"], params.layout["cpus_per_task"],
                    params.layout["partner_ntasks"], params.layout["why"])
    job = engine.build_job(ctx, stage, files, layout, params)
    job_id, est = submit_job(ctx, job.spec, kind=job.kind, label=job.label, meta=job.meta)
    walltime_why = params.layout.get("walltime_why")
    return Observation(
        ok=True,
        summary=f"stage {stage.name} submitted as {job_id}: {job.summary} ({layout.why}; booked "
        f"{est} core-h for {params.walltime_min} min"
        + (f": {walltime_why}" if walltime_why else "") + "). Poll with job_status.",
        command=job.spec.argv, artifacts=job.artifacts, job_id=job_id, state="QUEUED",
        core_h_estimate=est, refs=[f"stage:{stage.name}"])
