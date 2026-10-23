"""``script_run``: research scripts as first-class, provenance-bearing jobs.

Most of the science in a campaign runs through hand-written scripts and sbatch
templates, which without a tool of their own would bypass the ledger and the
core-hour gate. ``script_run`` runs *any* script through the execution backend
and records what the tool-per-step approach records for engine runs:

- the script, hashed, plus the git commit (and dirty flag) of the repo it
  lives in, so "which version of the analysis produced this" has an answer;
- declared inputs hashed before the run, declared outputs hashed after it
  (missing outputs fail the job even when the exit code is 0);
- a core-hour estimate that must be acknowledged above the machine's
  ``compute.warn_core_h`` and is refused above ``compute.block_core_h``;
- optional ``after`` dependencies (afterok on tracked job ids) and an optional
  bounded wait so short analyses return their results in one call.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import ScriptObservation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning
from myrmidon.engines.gromacs.run import get_jobstore
from myrmidon.errors import JobNotFoundError
from myrmidon.execution.accounting import budget_warnings, estimate_core_h
from myrmidon.execution.backend import JobSpec, JobState, Resources
from myrmidon.scripts.collect import collect_script_job, script_observation

_INTERPRETERS = {".py": "python", ".sh": "sh", ".bash": "bash"}


class ScriptRunParams(BaseModel):
    #: the fields this submission's cost is computed from; a plan step must state them
    RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("ntasks", "cpus_per_task", "walltime_min")

    script: str = Field(description="script path: workspace-relative or absolute (e.g. a "
                        "file in the research repo)")
    args: list[str] = Field(default_factory=list, description="command-line arguments")
    interpreter: str | None = Field(
        default=None,
        description="'python' (default for .py — the harness's own interpreter), 'sh', "
        "'bash', or a path; null for an executable script",
    )
    inputs: list[str] = Field(
        default_factory=list,
        description="files the script reads (hashed into provenance; must exist)",
    )
    outputs: list[str] = Field(
        default_factory=list,
        description="files or globs the script must produce, relative to cwd; hashed at "
        "collection, missing ones fail the job",
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
    walltime_min: int = Field(default=30, ge=1)
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
    acknowledge_warnings: list[str] = Field(
        default_factory=list,
        description="ids of warn-level gate findings you explicitly accept",
    )

    def resources(self) -> Resources:
        """What this submission books. The study approval prices a node through this;
        a tool that submits without it would show a person 0.0 core-h for work that
        costs."""
        return _resources(self)


def _resolve(ctx: ToolContext, path: str) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ctx.workspace.root / p).resolve()


def _inside(root: Path, path: Path) -> bool:
    return path == root or str(path).startswith(str(root) + "/")


def _interpreter(params: ScriptRunParams, script: Path) -> str | None:
    name = params.interpreter
    if name is None:
        name = _INTERPRETERS.get(script.suffix)
    if name == "python":
        return sys.executable  # the harness env; same filesystem on the compute nodes
    return name


def _resources(params: ScriptRunParams) -> Resources:
    return Resources(
        ntasks=params.ntasks,
        cpus_per_task=params.cpus_per_task,
        gpus=params.gpus,
        walltime_min=params.walltime_min,
        partition=params.partition,
        account=params.account,
    )


def git_state(path: Path) -> tuple[str | None, bool | None]:
    """(short commit, dirty) of the git repo containing ``path``; (None, None)
    when it is not under git or git is unavailable."""
    if shutil.which("git") is None:
        return None, None
    try:
        head = subprocess.run(  # noqa: S603
            ["git", "-C", str(path.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
        if head.returncode != 0:
            return None, None
        status = subprocess.run(  # noqa: S603
            ["git", "-C", str(path.parent), "status", "--porcelain",
             "--untracked-files=no"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    return head.stdout.strip(), bool(status.stdout.strip())


def script_gate(ctx: ToolContext, params: ScriptRunParams):
    warnings: list[ValidationWarning] = []
    checked = ["script", "inputs", "interpreter", "dependencies", "compute_budget", "cwd"]
    if "\n" in params.script or len(params.script) > 4096 or \
            any(len(part) > 255 for part in Path(params.script).parts):
        # a whole program passed as the path: stat() on it raises "File name too long",
        # an OSError that would take the session down instead of being a finding
        return [ValidationWarning(
            id="SCRIPT_NOT_A_PATH", severity="block",
            message="script is a program's text, not a path to one: script_run runs a "
            "file in the study, so what it ran is kept and replayable",
            suggestion="write the program to a file in the study with file_write, then "
            "script_run that path",
            needs=["script"],
        )], checked
    script = _resolve(ctx, params.script)
    if not script.is_file():
        warnings.append(ValidationWarning(
            id="SCRIPT_MISSING", severity="block", message=f"script not found: {params.script}",
            needs=["script"],
        ))
    for i, path in enumerate(params.inputs):
        if not _resolve(ctx, path).exists():
            warnings.append(ValidationWarning(
                id=f"INPUT_MISSING_{i}", severity="block", message=f"input not found: {path}",
                needs=["inputs"],
            ))
    interp = _interpreter(params, script)
    if interp is not None and shutil.which(interp) is None and not Path(interp).is_file():
        warnings.append(ValidationWarning(
            id="INTERPRETER_MISSING", severity="block",
            message=f"interpreter '{interp}' not found anywhere on this machine",
        ))
    elif interp is None and script.is_file() and not (script.stat().st_mode & 0o111):
        warnings.append(ValidationWarning(
            id="SCRIPT_NOT_EXECUTABLE", severity="block",
            message=f"{params.script} is not executable and no interpreter was given",
        ))
    if params.after:
        store = get_jobstore(ctx)
        for job_id in params.after:
            try:
                store.get(job_id)
            except JobNotFoundError:
                warnings.append(ValidationWarning(
                    id="AFTER_UNKNOWN_JOB", severity="block",
                    message=f"dependency job '{job_id}' is not tracked in this workspace",
                ))

    est, budget = budget_warnings(_resources(params), ctx.config.compute,
                                  ctx.config.backend)
    warnings.extend(budget)
    if params.cwd is not None and not _inside(ctx.workspace.root, _resolve(ctx, params.cwd)):
        warnings.append(ValidationWarning(
            id="CWD_OUTSIDE_WORKSPACE", severity="warn",
            message=f"cwd {params.cwd} is outside the workspace: outputs will be hashed "
            "but recorded with absolute paths, and the workspace does not own them",
        ))
    return warnings, checked


@tool(
    "script_run",
    description="Run ANY research script (python/sh/executable) as a tracked job with "
    "provenance: script hash + git commit, declared inputs hashed before and declared "
    "outputs hashed after (missing outputs fail the job), core-hour estimate gated by the "
    "machine's approval threshold, optional afterok dependencies. Returns a job_id "
    "immediately; pass wait_seconds to get results of short scripts in the same call. "
    "Prefer this over running scripts by hand: what bypasses it leaves no trace.",
    returns=ScriptObservation,
    tags={"jobs", "scripts"},
    expensive=True,
    gate=script_gate,
)
def script_run(ctx: ToolContext, params: ScriptRunParams) -> ScriptObservation:
    ws = ctx.workspace
    script = _resolve(ctx, params.script)
    stage = ws.stage(params.stage)
    cwd = _resolve(ctx, params.cwd) if params.cwd else stage
    cwd.mkdir(parents=True, exist_ok=True)
    interp = _interpreter(params, script)
    argv = ([interp] if interp else []) + [str(script), *params.args]
    token = uuid.uuid4().hex[:6]
    spec = JobSpec(
        argv=argv,
        cwd=cwd,
        env=params.env,
        stdout=f"{script.stem}-{token}.out",
        stderr=f"{script.stem}-{token}.err",
        resources=_resources(params),
    )
    commit, dirty = git_state(script)
    est = estimate_core_h(spec.resources, ctx.config.compute)
    inputs = [ArtifactRef.from_path(script, "script", root=ws.root)] + [
        ArtifactRef.from_path(_resolve(ctx, p), "input", root=ws.root) for p in params.inputs
    ]

    store = get_jobstore(ctx)
    if params.after:
        deps = []
        for job_id in params.after:
            rec = store.get(job_id)
            deps.append(ctx.backend.dep_for(rec.spec, rec.handle))
        handle = ctx.backend.submit_with_deps(spec, deps, mode="ok")
    else:
        handle = ctx.backend.submit(spec)
    label = params.label or f"script:{script.stem}"
    job_id = store.add(
        ctx.backend.name, handle, spec, label=label,
        meta={
            "kind": "script", "script": str(script), "outputs": params.outputs,
            "core_h_estimate": est, "git_commit": commit, "git_dirty": dirty,
        },
    )

    state = JobState.QUEUED
    exit_code = None
    if params.wait_seconds:
        deadline = time.monotonic() + params.wait_seconds
        interval = 0.2  # first look soon, then back off to every 2 s (as job_status does)
        while True:
            status = ctx.backend.poll(handle, spec)
            state, exit_code = status.state, status.exit_code
            if state.terminal or time.monotonic() >= deadline:
                break
            time.sleep(min(interval, max(0.1, deadline - time.monotonic())))
            interval = min(2.0, interval * 2)
        if state != JobState.QUEUED:
            store.update(job_id, state, exit_code)

    where = ws.relpath(cwd)
    version = f" @ {commit}{'+dirty' if dirty else ''}" if commit else ""
    common = dict(
        job_id=job_id, state=state.value, exit_code=exit_code, command=argv,
        inputs=inputs, core_h_estimate=est, script=params.script, interpreter=interp,
        git_commit=commit, git_dirty=dirty, outputs_expected=params.outputs,
    )
    if not state.terminal:
        return ScriptObservation(
            ok=True,
            summary=f"{label} submitted via {ctx.backend.name} as {job_id}{version}; "
            f"est. {est} core-h; cwd {where}. Poll with job_status; once it has ended the "
            "same call collects the declared outputs.",
            **common,
        )
    collected = collect_script_job(store.get(job_id), ws.root)
    return script_observation(
        label, state, exit_code, collected,
        f"{label} completed{version}: {len(collected.artifacts)} output artifact(s) in "
        f"{where}; est. {est} core-h.", **common)
