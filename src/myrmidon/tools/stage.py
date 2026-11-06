"""A mandate stage as one step: what its run reads, compiled in one call."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract
from myrmidon.core.lineage import stage_run as collected_run
from myrmidon.core.mandate import read_mandate
from myrmidon.core.observation import Observation, ToolError
from myrmidon.core.tool import REGISTRY, ToolContext, tool
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.registry import adapter
from myrmidon.knowledge.inputsets import RENDERED_FILES
from myrmidon.knowledge.stage import TemplateStore, templates_for
from myrmidon.tools.inputsets import system_file


class StagePrepareParams(BaseModel):
    stage: str = Field(description="a stage the study's mandate names (method.stages)")
    structure: str | None = Field(
        default=None, description="what the stage starts from; default: the structure the run "
        "it starts from left, or for a coupled stage and the first stage the system's")
    topology: str | None = Field(default=None, description="default: the system's; a topology "
                                 "a fit produced, to run a stage on it")
    index: str | None = Field(default=None, description="default: the system's")
    overrides: dict = Field(default_factory=dict, description="engine keys to set yourself, "
                            "each recorded as origin 'user' and said; every other value is "
                            "the mandate's, the method's or the system's")
    max_warn: int = Field(default=0, description="warnings the compile of the run input may "
                          "suppress: each is a physics decision")


def _starting_structure(ctx: ToolContext, mandate, stage) -> tuple[str | None, str]:
    """(the structure a stage's run input is compiled against, why when there is none yet): a
    coupled stage's MM half always against the system's, as its engine takes the coordinates
    from the QM half; else the one the run it starts from left; the first stage, the system's."""
    system = system_file(ctx, mandate, "structure")
    begins = mandate.method.starts_from(stage)
    if stage.regime != "mm" or begins is None:
        return system, ""
    run = collected_run(ctx.workspace, begins.name)
    left = next((a.path for a in (run[0] if run else []) if a.role == "structure"), None)
    return left, "" if left else (f"{begins.name}'s run has not been collected, and this stage "
                                  "starts from the structure it leaves")


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
                                "index": "index"},
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
    topology = params.topology or system_file(ctx, mandate, "topology")
    index = params.index or system_file(ctx, mandate, "index")
    stage_dir = ctx.workspace.stage(stage.name)
    prepared: dict[str, str] = {}
    findings: list = []
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
        prepared[RENDERED_FILES[template.renders][0]] = ctx.workspace.relpath(output)
    return _with_run_input(ctx, params, stage, stage_dir, templates, prepared, findings,
                           {"structure": structure, "topology": topology, "index": index}, later)


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
        if engine.compile_run_input is None:
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
    return Observation(
        ok=True,
        summary=f"stage {stage.name} prepared: " + ", ".join(sorted(prepared.values())),
        files=prepared, findings=findings, artifacts=artifacts, inputs=inputs, command=command,
        refs=[f"stage:{stage.name}"])


def _later(stage: str, why: str):
    from myrmidon.core.validation import Finding

    return Finding(id="STAGE_RUN_INPUT_LATER",
                   message=f"stage {stage}'s run input is not compiled yet: {why}; prepare the "
                   "stage again once it is there")
