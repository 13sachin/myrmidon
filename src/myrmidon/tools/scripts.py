"""``script_run``: research scripts as first-class, provenance-bearing calls.

Most of the science in a campaign runs through hand-written scripts, which without a tool
of their own would bypass the ledger and the core-hour gate. ``script_run`` runs *any*
script through the harness and records what the tool-per-step approach records for engine
runs:

- the script, hashed, plus the git commit (and dirty flag) of the repo it
  lives in, so "which version of the analysis produced this" has an answer;
- declared inputs hashed before the run, declared outputs hashed after it
  (missing outputs fail the run even when the exit code is 0);
- its ``kind``: glue turns files into files, an analysis writes its numbers to a
  ``results`` file the harness reads back as the script's values, physics computes what an
  engine computes and the report says so;
- its placement: a script whose resources fit the machine's session envelope runs in the
  session and books nothing; one that does not is a job, with a core-hour estimate that
  the researcher's approved mandate covers and that is refused
  above ``compute.block_core_h``, optional ``after`` dependencies (afterok on tracked job
  ids) and an optional bounded wait so a short job returns its results in one call."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation
from myrmidon.core.tool import PREVIEW_HELP, ToolContext, precheck, tool
from myrmidon.core.validation import Finding, Precondition
from myrmidon.errors import JobNotFoundError
from myrmidon.execution.backend import JobSpec, JobState, Resources
from myrmidon.execution.collect import collector
from myrmidon.execution.jobs import refresh, submit_job
from myrmidon.execution.jobstore import open_jobstore
from myrmidon.execution.placement import Placement, place
from myrmidon.scripts.collect import (
    collect_outputs,
    collect_script_job,
    git_state,
    script_observation,
)

_INTERPRETERS = {".py": "python", ".sh": "sh", ".bash": "bash"}


class ScriptRunParams(BaseModel):
    #: the fields this submission's cost is computed from
    RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("ntasks", "cpus_per_task", "walltime_min")

    preview: bool = Field(default=False, description=PREVIEW_HELP)
    #: a call whose resources fit the session envelope runs in the session, not as a job
    SESSION_OK: ClassVar[bool] = True

    script: str = Field(description="script path: workspace-relative or absolute (e.g. a "
                        "file in the research repo)")
    kind: Literal["glue", "analysis", "physics"] = Field(
        description="what the script is. 'glue': turns files into files and claims no "
        "number (declare `outputs`). 'analysis': computes numbers and writes them to "
        "`results`, a JSON object of name → value the harness reads back as the script's "
        "values, never as an engine's measurement. 'physics': computes what an engine "
        "computes, so its numbers are the script's author's and the report says so — "
        "tool_lookup first",
    )
    results: str | None = Field(
        default=None,
        description="an analysis script's results file, relative to cwd or to the study "
        "root: a JSON object of name → value, hashed and read into the observation's "
        "`results`",
    )
    args: list[str] = Field(default_factory=list, description="command-line arguments")
    interpreter: str | None = Field(
        default=None,
        description="'python' (default for .py — the harness's own interpreter), 'sh', "
        "'bash', or a path; null for an executable script",
    )
    inputs: list[str] = Field(
        default_factory=list,
        description="files the script reads, relative to cwd or to the study root (hashed "
        "into provenance; must exist)",
    )
    outputs: list[str] = Field(
        default_factory=list,
        description="files or globs the script must produce, relative to cwd or to the "
        "study root; hashed at collection, missing ones fail the job and the refusal names "
        "where it looked",
    )
    cwd: str | None = Field(
        default=None,
        description="working directory (default: the workspace stage); outside the "
        "workspace is allowed but flagged",
    )
    stage: str = Field(default="scripts", description="workspace stage for logs/outputs")
    env: dict[str, str] = Field(default_factory=dict)
    ntasks: int = Field(
        default=1, ge=1,
        description="scheduler tasks; the script runs once PER TASK (srun replicates it). "
        "A wrapper that starts its own MPI engine keeps ntasks=1 with cpus_per_task set "
        "to the cores it needs, unsets PMI_*/PMIX_* and step-scoped SLURM_* variables, "
        "and starts the engine with `srun --overlap -n <ranks>` — a nested srun otherwise "
        "attaches the ranks to the outer one-task step",
    )
    cpus_per_task: int = Field(default=1, ge=1)
    gpus: int = Field(default=0, ge=0)
    walltime_min: int | None = Field(
        default=None, ge=1,
        description="minutes the script may run. Within the machine's session envelope "
        "(cores × minutes) it runs in the session and books nothing; above it, it is a job. "
        "Unset: a glue or analysis script gets the session's minutes, a physics one 30",
    )
    partition: str | None = None
    account: str | None = Field(
        default=None,
        description="Slurm account to bill instead of the machine default "
        "(e.g. a second project's budget)",
    )
    after: list[str] = Field(
        default_factory=list, description="tracked job ids that must complete OK first"
    )
    wait_seconds: int = Field(
        default=0, ge=0, le=600,
        description="block up to this long and return collected results if the script "
        "finishes (short analyses: one call instead of three)",
    )
    label: str | None = None

    def resources(self) -> Resources:
        """What this submission books. The study approval prices a node through this;
        a tool that submits without it would show a person 0.0 core-h for work that
        costs."""
        return script_resources(self)

    def sized(self, ctx: ToolContext) -> ScriptRunParams:
        """A walltime for a script that states none: glue and analysis that fit the
        session's cores run there, for as long as it allows; anything else books the
        default."""
        if self.walltime_min is not None:
            return self
        fits = self.ntasks * self.cpus_per_task <= ctx.config.session.max_cores and not self.gpus
        minutes = (ctx.config.session.max_minutes
                   if self.kind in ("glue", "analysis") and fits else DEFAULT_WALLTIME_MIN)
        return self.model_copy(update={"walltime_min": minutes})


def is_inside(root: Path, path: Path) -> bool:
    return path == root or str(path).startswith(str(root) + "/")


def script_interpreter(params: ScriptRunParams, script: Path) -> str | None:
    name = params.interpreter
    if name is None:
        name = _INTERPRETERS.get(script.suffix)
    if name == "python":
        return sys.executable  # the harness env; same filesystem on the compute nodes
    return name


#: what a script that states no walltime, and is not glue or analysis, books
DEFAULT_WALLTIME_MIN = 30


def script_resources(params: ScriptRunParams) -> Resources:
    return Resources(
        ntasks=params.ntasks,
        cpus_per_task=params.cpus_per_task,
        gpus=params.gpus,
        walltime_min=params.walltime_min or DEFAULT_WALLTIME_MIN,
        partition=params.partition,
        account=params.account,
    )


def _resolve(ctx: ToolContext, path: str) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ctx.workspace.root / p).resolve()


def _resolve_input(ctx: ToolContext, path: str, cwd: str | None) -> Path:
    """An input, looked for under the script's cwd and then under the study root — the two
    places a path in a study is written relative to, in the order its outputs are looked
    for (`scripts.collect`): what the script reads is what sits where it runs."""
    at_root = _resolve(ctx, path)
    if cwd is None or Path(path).is_absolute():
        return at_root
    at_cwd = (_resolve(ctx, cwd) / path).resolve()
    return at_cwd if at_cwd.exists() else at_root


def _placement(ctx: ToolContext, params: ScriptRunParams) -> Placement:
    return place(script_resources(params), ctx.config.session, after=bool(params.after))


def script_gate(ctx: ToolContext, params: ScriptRunParams):
    warnings: list[Finding] = []
    checked = ["script", "kind", "inputs", "interpreter", "dependencies", "cwd"]
    if params.kind == "glue" and not params.outputs:
        warnings.append(Precondition(
            id="SCRIPT_GLUE_NO_OUTPUT",
            message="a glue script turns files into files, and this one declares none",
            suggestion="declare what it writes in `outputs`; a script that computes numbers "
            "is `analysis` with a `results` file",
            needs=["outputs"],
        ))
    if params.kind == "analysis" and not params.results:
        warnings.append(Precondition(
            id="SCRIPT_ANALYSIS_NO_RESULTS",
            message="an analysis script's numbers exist only in its `results` file, and "
            "this one names none: whatever it prints is not read",
            suggestion="have the script write a JSON object of name → value and name that "
            "file in `results`",
            needs=["results"],
        ))
    if params.kind == "physics":
        warnings.append(Finding(
            id="SCRIPT_IS_PHYSICS",
            message="this script computes what an engine computes: its numbers are its "
            "author's, not an engine's, and the report will say so",
            suggestion="tool_lookup for the engine that computes this, and say in the "
            "report why the script is the source",
        ))
    if "\n" in params.script or len(params.script) > 4096 or \
            any(len(part) > 255 for part in Path(params.script).parts):
        # a whole program passed as the path: stat() on it raises "File name too long",
        # an OSError that would take the session down instead of being a finding
        return [Precondition(
            id="SCRIPT_NOT_A_PATH",
            message="script is a program's text, not a path to one: script_run runs a "
            "file in the study, so what it ran is kept on record",
            suggestion="write the program to a file in the study with file_write, then "
            "script_run that path",
            needs=["script"],
        )], checked
    script = _resolve(ctx, params.script)
    if not script.is_file():
        warnings.append(Precondition(
            id="SCRIPT_MISSING", message=f"script not found: {params.script}",
            needs=["script"],
        ))
    for i, path in enumerate(params.inputs):
        if not _resolve_input(ctx, path, params.cwd).exists():
            looked = " and ".join(dict.fromkeys(
                ([params.cwd] if params.cwd else []) + ["the study root"]))
            warnings.append(Precondition(
                id=f"INPUT_MISSING_{i}",
                message=f"input not found: {path}, looked for under {looked}",
                needs=["inputs"],
            ))
    interp = script_interpreter(params, script)
    if interp is not None and shutil.which(interp) is None and not Path(interp).is_file():
        warnings.append(Precondition(
            id="INTERPRETER_MISSING",
            message=f"interpreter '{interp}' not found anywhere on this machine",
        ))
    elif interp is None and script.is_file() and not (script.stat().st_mode & 0o111):
        warnings.append(Precondition(
            id="SCRIPT_NOT_EXECUTABLE",
            message=f"{params.script} is not executable and no interpreter was given",
        ))
    if params.after:
        store = open_jobstore(ctx.workspace)
        for job_id in params.after:
            try:
                store.get(job_id)
            except JobNotFoundError:
                warnings.append(Precondition(
                    id="AFTER_UNKNOWN_JOB",
                    message=f"dependency job '{job_id}' is not tracked in this workspace",
                ))

    if params.cwd is not None and not is_inside(ctx.workspace.root, _resolve(ctx, params.cwd)):
        warnings.append(Finding(
            id="CWD_OUTSIDE_WORKSPACE",
            message=f"cwd {params.cwd} is outside the workspace: outputs will be hashed "
            "but recorded with absolute paths, and the workspace does not own them",
        ))
    return warnings, checked


def _run_in_session(ctx: ToolContext, params: ScriptRunParams, spec: JobSpec, argv: list,
                    inputs: list[ArtifactRef], placement: Placement,
                    **common) -> Observation:
    """The script run here and now, bounded by the envelope's minutes, with no job record:
    the ledger event is its whole record."""
    ws = ctx.workspace
    cwd = Path(spec.cwd)
    limit = ctx.config.session.max_minutes
    killed = None
    try:
        with open(cwd / spec.stdout, "w") as out, open(cwd / spec.stderr, "w") as err:
            proc = subprocess.run(  # noqa: S603
                argv, cwd=cwd, env={**os.environ, **spec.env}, stdout=out, stderr=err,
                timeout=limit * 60, check=False,
            )
        exit_code = proc.returncode
        state = JobState.COMPLETED if exit_code == 0 else JobState.FAILED
    except subprocess.TimeoutExpired:
        exit_code, state = None, JobState.FAILED
        killed = (f"killed at the session envelope's {limit} min: work of that size is a "
                  "job — state walltime_min above the envelope and it is submitted")
    collected = collect_outputs(cwd, params.outputs, spec.stdout, spec.stderr, exit_code,
                                ws.root, results=params.results)
    if killed:
        collected.diagnosis = killed
    label = common.pop("label")
    return script_observation(
        label, state, exit_code, collected,
        f"{label} ran in the session: {len(collected.artifacts)} output artifact(s) in "
        f"{ws.relpath(cwd)}; booked nothing ({placement.reason}).",
        job_id="", state=state.value, exit_code=exit_code, command=argv, inputs=inputs,
        core_h_estimate=0.0, placement="session", **common)


@tool(
    "script_run",
    description="Run a research script (python, sh, executable) with provenance: script hash and "
                "git commit, declared inputs and outputs hashed, its kind (glue, analysis, "
                "physics). Inside the session envelope it runs at once; above it, it books a job "
                "under the mandate.",
    tags={"jobs", "scripts"},
    expensive=True,
)
def script_run(ctx: ToolContext, params: ScriptRunParams) -> Observation:
    precheck(ctx, script_gate(ctx, params)[0])
    ws = ctx.workspace
    script = _resolve(ctx, params.script)
    stage = ws.stage(params.stage)
    cwd = _resolve(ctx, params.cwd) if params.cwd else stage
    cwd.mkdir(parents=True, exist_ok=True)
    interp = script_interpreter(params, script)
    argv = ([interp] if interp else []) + [str(script), *params.args]
    token = uuid.uuid4().hex[:6]
    spec = JobSpec(
        argv=argv,
        cwd=cwd,
        env=params.env,
        stdout=f"{script.stem}-{token}.out",
        stderr=f"{script.stem}-{token}.err",
        resources=script_resources(params),
    )
    commit, dirty = git_state(script)
    inputs = [ArtifactRef.from_path(script, "script", root=ws.root)] + [
        ArtifactRef.from_path(_resolve_input(ctx, p, params.cwd), "input", root=ws.root)
        for p in params.inputs
    ]
    label = params.label or f"script:{script.stem}"
    identity = dict(
        script=params.script, interpreter=interp, git_commit=commit, git_dirty=dirty,
        outputs_expected=params.outputs, script_kind=params.kind,
    )
    placement = _placement(ctx, params)
    if placement.in_session:
        return _run_in_session(ctx, params, spec, argv, inputs, placement, label=label,
                               **identity)

    job_id, est = submit_job(
        ctx, spec, kind="script", label=label, after=params.after,
        meta={
            "script_kind": params.kind, "script": str(script),
            "outputs": params.outputs, "results": params.results,
            "git_commit": commit, "git_dirty": dirty,
        },
    )

    state = JobState.QUEUED
    exit_code = None
    if params.wait_seconds:
        deadline = time.monotonic() + params.wait_seconds
        interval = 0.2  # first look soon, then back off to every 2 s (as job_status does)
        while True:
            _, state, exit_code, _ = refresh(ctx, job_id)
            if state.terminal or time.monotonic() >= deadline:
                break
            time.sleep(min(interval, max(0.1, deadline - time.monotonic())))
            interval = min(2.0, interval * 2)

    where = ws.relpath(cwd)
    version = f" @ {commit}{'+dirty' if dirty else ''}" if commit else ""
    common = dict(
        job_id=job_id, state=state.value, exit_code=exit_code, command=argv,
        inputs=inputs, core_h_estimate=est, placement="job", **identity,
    )
    if not state.terminal:
        return Observation(
            ok=True,
            summary=f"{label} submitted via {ctx.backend.name} as {job_id}{version}; "
            f"est. {est} core-h; cwd {where}. Poll with job_status; once it has ended the "
            "same call collects the declared outputs.",
            **common,
        )
    store = open_jobstore(ws)
    collected = collect_script_job(store.get(job_id), ws.root)
    # collected here, with this call's ledger event: job_status will not collect it again
    store.set_meta(job_id, collected=True)
    return script_observation(
        label, state, exit_code, collected,
        f"{label} completed{version}: {len(collected.artifacts)} output artifact(s) in "
        f"{where}; est. {est} core-h.", **common)


@collector("script")
def _collect_script(ctx: ToolContext, record, state: JobState, exit_code: int | None):
    """A script job that has ended: its declared outputs hashed and its results read."""
    record = open_jobstore(ctx.workspace).get(record.job_id)  # state and exit code stored
    collected = collect_script_job(record, ctx.workspace.root)
    meta = record.meta
    label = record.label or record.job_id
    return script_observation(
        label, state, exit_code, collected,
        f"{label} completed: {len(collected.artifacts)} declared output artifact(s) hashed.",
        job_id=record.job_id, state=state.value, exit_code=exit_code,
        core_h_estimate=meta.get("core_h_estimate"), script=meta.get("script", ""),
        git_commit=meta.get("git_commit"), git_dirty=meta.get("git_dirty"),
        outputs_expected=meta.get("outputs", []), script_kind=meta.get("script_kind", ""),
        placement="job",
    )
