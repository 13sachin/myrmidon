"""Job tracking and result collection with crash-pattern diagnosis.

Failures come back as a structured diagnosis plus a truncated log tail — never
a raw log dump.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import JobObservation, Observation, RunProgress
from myrmidon.core.tool import ToolContext, tool
from myrmidon.engines.gromacs.run import get_jobstore
from myrmidon.execution.backend import JobState
from myrmidon.execution.jobstore import JobRecord
from myrmidon.text import tail

_ROLE_BY_SUFFIX = {
    ".log": "log",
    ".edr": "energy",
    ".xtc": "trajectory",
    ".trr": "trajectory",
    ".gro": "structure",
    ".cpt": "checkpoint",
}

#: what SLURM writes into the job's stderr when it kills it at the walltime. The state
#: says so too (JobState.TIMEOUT), but a local run only has the text.
_TIME_LIMIT = r"DUE TO TIME LIMIT|CANCELLED AT .*DUE TO TIME"

_CRASH_PATTERNS: list[tuple[str, str]] = [
    (
        r"LINCS WARNING",
        "LINCS constraint failure — bonds rotated too far in one step. Typical causes: "
        "timestep too large, bad starting geometry (clashes), or missing minimization. "
        "Minimize first and/or reduce dt.",
    ),
    (
        r"[Cc]annot settle water|settle.*error",
        "SETTLE failure — a water molecule blew up. Usually atom overlap after solvation "
        "or ionization; energy-minimize before dynamics.",
    ),
    (
        r"Fatal error",
        "GROMACS fatal error — see the extracted message in the log tail.",
    ),
    (
        r"[Ss]egmentation fault|SIGSEGV",
        "Segmentation fault — often an incompatible binary/architecture or corrupted "
        "input; re-run grompp and check the engine build.",
    ),
    (
        _TIME_LIMIT,
        "killed at its walltime by the scheduler — the booking was shorter than the run "
        "needed. Continue from the checkpoint with checkpoint_continue, or book more "
        "(gmx_mdrun walltime_min).",
    ),
]


def _spec_deffnm(record: JobRecord) -> str:
    """The prefix this run's output files carry.

    What the submitting tool recorded comes first. A coupled MiMiC launch puts CPMD in
    `spec.argv` and GROMACS in a partner command, so the argv holds no `-deffnm`; the
    partner's own argv is checked before the default, or collection would report success
    having found none of the MM outputs.
    """
    recorded = record.meta.get("deffnm")
    if recorded:
        return str(recorded)
    argv = record.spec.argv
    if "-deffnm" in argv:
        return argv[argv.index("-deffnm") + 1]
    for partner in record.spec.coupled or []:
        if "-deffnm" in partner.argv:
            return partner.argv[partner.argv.index("-deffnm") + 1]
    return "md"


def _parse_progress(log_path: Path) -> RunProgress | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    progress = RunProgress()
    steps = re.findall(r"^\s+Step\s+Time\s*\n\s*(\d+)\s+([\d.eE+-]+)", text, re.MULTILINE)
    if steps:
        progress.step = int(steps[-1][0])
        progress.time_ps = float(steps[-1][1])
    perf = re.findall(r"^Performance:\s+([\d.]+)", text, re.MULTILINE)
    if perf:
        progress.ns_per_day = float(perf[-1])
    if progress.step is None and progress.ns_per_day is None:
        return None
    return progress


def _parse_nsteps(text: str) -> int | None:
    """How many steps the run was asked for, from mdrun's own dump of the input
    parameters into the log — never from the mdp the harness thinks it used."""
    found = re.search(r"^\s*nsteps\s*=\s*(\d+)", text, re.MULTILINE)
    return int(found.group(1)) if found else None


def _walltime_diagnosis(log_path: Path, text: str) -> str:
    """A walltime kill, with how far it got: the number that decides whether to continue
    from the checkpoint or to book a bigger job."""
    progress = _parse_progress(log_path)
    step = progress.step if progress is not None else None
    reached = f"step {step}" if step is not None else "no completed step"
    asked = _parse_nsteps(text)
    of = f" of {asked}" if asked else ""
    share = ""
    if step is not None and asked:
        share = f" ({100 * step / asked:.0f}% of the run)"
    return (
        f"killed at its walltime after {reached}{of}{share} — the booking was shorter "
        "than the run needed. Continue from the checkpoint with checkpoint_continue, or "
        "book more (gmx_mdrun walltime_min)."
    )


def _diagnose(text: str) -> str | None:
    for pattern, diagnosis in _CRASH_PATTERNS:
        if re.search(pattern, text):
            return diagnosis
    return None


def _refresh(ctx: ToolContext, job_id: str) -> tuple[JobRecord, JobState, int | None, bool]:
    """(record, state, exit_code, changed) — ``changed`` is True when this poll
    moved the stored state, i.e. something actually happened."""
    store = get_jobstore(ctx)
    record = store.get(job_id)
    if record.state.terminal:
        return record, record.state, record.exit_code, False
    status = ctx.backend.poll(record.handle, record.spec)
    changed = status.state != record.state
    if changed:
        store.update(job_id, status.state, status.exit_code)
    return record, status.state, status.exit_code, changed


def pending_lines(ctx: ToolContext) -> list[str]:
    """One sentence per job this workspace is still waiting on, each after one poll.

    The look a session opens with. The job store holds the last state anybody asked for,
    so a job that ended while nobody was watching is indistinguishable there from one
    still queueing — each is refreshed against the backend here. A job that has ended
    names the call that collects it; collecting is `job_status`, exactly once, with the
    artifacts, and never this look.
    """
    store = get_jobstore(ctx)
    lines: list[str] = []
    for record in store.list():
        if record.state.terminal and record.meta.get("collected"):
            continue
        _, state, _, _ = _refresh(ctx, record.job_id)
        what = f" ({record.label})" if record.label else ""
        if state.terminal:
            lines.append(
                f"{record.job_id} {state.value}{what}, not yet collected — "
                f"call job_status job_id={record.job_id}"
            )
            continue
        bits = [f"{record.job_id} {state.value}{what}"]
        log = Path(record.spec.cwd) / f"{_spec_deffnm(record)}.log"
        progress = _parse_progress(log)
        if progress is not None and progress.step is not None:
            asked = _parse_nsteps(log.read_text(errors="replace"))
            bits.append(f"step {progress.step}" + (f" of {asked}" if asked else ""))
        est = record.meta.get("core_h_estimate")
        if est is not None:
            bits.append(f"{est} core-h booked")
        lines.append(", ".join(bits))
    return lines


class JobIdParams(BaseModel):
    job_id: str = Field(description="id returned by an async submission tool")


class JobStatusParams(JobIdParams):
    wait_seconds: int = Field(
        default=0,
        ge=0,
        le=600,
        description="block up to this long, returning as soon as the job reaches a "
        "terminal state. USE THIS (e.g. 60-300) instead of polling in a loop — each "
        "poll otherwise costs a full model turn.",
    )


#: what ``job_status`` returns once a job has ended, by ``JobRecord.meta["kind"]``. Engines
#: register theirs at import (fm_fit, fm_validate, mimic); scripts are handled here; a job
#: with no kind is a GROMACS run. One collector per kind, and only ``job_status`` calls them:
#: the agent learns a job's outcome from the same call that told it the job was running.
_COLLECTORS: dict[str, Callable[[ToolContext, JobRecord, JobState, int | None], Observation]] = {}


def register_collector(kind: str):
    def wrap(fn):
        _COLLECTORS[kind] = fn
        return fn
    return wrap


def collect_job(ctx: ToolContext, record: JobRecord, state: JobState,
                exit_code: int | None) -> Observation:
    """The outcome of a job that has ended: artifacts hashed, results parsed, or a
    crash-pattern diagnosis with the log tail — by job kind."""
    kind = record.meta.get("kind")
    if kind == "script":
        return _script_results(ctx, record.job_id, state, exit_code)
    if kind in _COLLECTORS:
        return _COLLECTORS[kind](ctx, record, state, exit_code)
    return gromacs_results(ctx, record, state, exit_code)


@tool(
    "job_status",
    description="Check one tracked job: state plus progress (step, time, ns/day) parsed "
    "from the engine log while it runs. Once it has ended, the same call collects it: "
    "output artifacts hashed into the ledger (a run's log/energy/trajectory/structure, a "
    "fit's parsed results and candidate topology, a validation's tables, a script's "
    "declared outputs), or a crash-pattern diagnosis plus the log tail on failure. There "
    "is no separate results tool. Pass wait_seconds (e.g. 120) to block until the job "
    "finishes or the wait expires — far cheaper than polling turn by turn.",
    returns=JobObservation,
    tags={"jobs"},
)
def job_status(ctx: ToolContext, params: JobStatusParams) -> Observation:
    import time as _time

    record, state, exit_code, changed = _refresh(ctx, params.job_id)
    if params.wait_seconds and not state.terminal:
        deadline = _time.monotonic() + params.wait_seconds
        # first look soon, then back off to every 5 s: a job that ends in half a second is
        # collected at once, and a scheduler is asked a handful of times more at most
        interval = 0.2
        while _time.monotonic() < deadline:
            _time.sleep(min(interval, max(0.1, deadline - _time.monotonic())))
            interval = min(5.0, interval * 2)
            record, state, exit_code, moved = _refresh(ctx, params.job_id)
            changed = changed or moved
            if state.terminal:
                break
    if state.terminal:
        return _collected(ctx, params.job_id, state, exit_code)

    deffnm = _spec_deffnm(record)
    progress = _parse_progress(Path(record.spec.cwd) / f"{deffnm}.log")
    bits = [f"{params.job_id} is {state.value}"]
    if progress and progress.step is not None:
        bits.append(f"step {progress.step} (t = {progress.time_ps} ps)")
    if progress and progress.ns_per_day is not None:
        bits.append(f"{progress.ns_per_day} ns/day")
    obs = JobObservation(
        ok=True,
        summary=", ".join(bits),
        job_id=params.job_id,
        state=state.value,
        exit_code=exit_code,
        progress=progress,
        core_h_estimate=record.meta.get("core_h_estimate"),
    )
    # A poll that found nothing new is not an event: on a long campaign unchanged polls
    # would otherwise be most of the ledger, burying the calls that mattered.
    # One that did see a change is still a look, not an action, so it is recorded as an
    # observation: the summary counts it apart and a replay waits instead.
    obs.record = changed
    obs.kind = "observe"
    return obs


def _collected(ctx: ToolContext, job_id: str, state: JobState,
               exit_code: int | None) -> Observation:
    """A job that has ended, collected exactly once into the ledger.

    The first look after the end is the collection — a ``call`` event carrying the
    artifacts (they are in the ledger because *this* call is the one that finds out).
    Every later look returns the same outcome as an observation.
    """
    store = get_jobstore(ctx)
    record = store.get(job_id)
    already = bool(record.meta.get("collected"))
    obs = collect_job(ctx, record, state, exit_code)
    for key, value in (
        ("job_id", job_id), ("state", state.value), ("exit_code", exit_code),
        ("core_h_estimate", record.meta.get("core_h_estimate")),
    ):
        if key in type(obs).model_fields and getattr(obs, key, None) in (None, ""):
            setattr(obs, key, value)
    if not obs.summary.startswith(job_id):
        obs.summary = f"{job_id} {state.value.lower()}: {obs.summary}"
    if already:
        obs.kind, obs.record = "observe", False
    else:
        store.set_meta(job_id, collected=True)
        obs.kind, obs.record = "call", True
    return obs


def gromacs_results(ctx: ToolContext, record: JobRecord, state: JobState,
                    exit_code: int | None) -> JobObservation:
    """A GROMACS run that has ended: its output files by role, or its diagnosis."""
    job_id = record.job_id
    cwd = Path(record.spec.cwd)
    deffnm = _spec_deffnm(record)
    outputs = sorted(cwd.glob(f"{deffnm}*"))
    artifacts = [
        ArtifactRef.from_path(p, _ROLE_BY_SUFFIX[p.suffix], root=ctx.workspace.root)
        for p in outputs
        if p.suffix in _ROLE_BY_SUFFIX
    ]
    log_text = ""
    log_path = cwd / f"{deffnm}.log"
    if log_path.exists():
        log_text += log_path.read_text(errors="replace")
    stderr_path = cwd / record.spec.stderr
    if stderr_path.exists():
        log_text += "\n" + stderr_path.read_text(errors="replace")

    if state is JobState.COMPLETED:
        progress = _parse_progress(log_path)
        return JobObservation(
            ok=True,
            summary=f"{job_id} completed: {len(artifacts)} output artifact(s). "
            "Analyze with gmx_energy / traj_analyze — do not report values not present "
            "in their observations.",
            job_id=job_id,
            state=state.value,
            exit_code=exit_code,
            progress=progress,
            artifacts=artifacts,
        )

    diagnosis = _diagnose(log_text)
    if state is JobState.TIMEOUT or re.search(_TIME_LIMIT, log_text):
        diagnosis = _walltime_diagnosis(log_path, log_text)
    tail_lines = tail(log_text, 30) if log_text else None
    return JobObservation(
        ok=False,
        summary=f"{job_id} {state.value.lower()}"
        + (f": {diagnosis}" if diagnosis else " with no recognized crash pattern"),
        job_id=job_id,
        state=state.value,
        exit_code=exit_code,
        diagnosis=diagnosis,
        artifacts=artifacts,
        log_tail=tail_lines,
    )


def _script_results(ctx: ToolContext, job_id: str, state: JobState, exit_code: int | None):
    from myrmidon.scripts.collect import collect_script_job, script_observation

    record = get_jobstore(ctx).get(job_id)  # re-read: state/exit code now stored
    collected = collect_script_job(record, ctx.workspace.root)
    meta = record.meta
    label = record.label or job_id
    return script_observation(
        label, state, exit_code, collected,
        f"{label} completed: {len(collected.artifacts)} declared output artifact(s) hashed.",
        job_id=job_id, state=state.value, exit_code=exit_code,
        core_h_estimate=meta.get("core_h_estimate"), script=meta.get("script", ""),
        git_commit=meta.get("git_commit"), git_dirty=meta.get("git_dirty"),
        outputs_expected=meta.get("outputs", []),
    )


@tool(
    "job_cancel",
    description="Cancel a tracked job (best effort).",
    returns=JobObservation,
    tags={"jobs"},
)
def job_cancel(ctx: ToolContext, params: JobIdParams) -> JobObservation:
    store = get_jobstore(ctx)
    record = store.get(params.job_id)
    if record.state.terminal:
        return JobObservation(
            ok=False,
            summary=f"{params.job_id} already {record.state.value}",
            job_id=params.job_id,
            state=record.state.value,
        )
    ctx.backend.cancel(record.handle, record.spec)
    store.update(params.job_id, JobState.CANCELLED)
    return JobObservation(
        ok=True,
        summary=f"{params.job_id} cancelled",
        job_id=params.job_id,
        state=JobState.CANCELLED.value,
    )
