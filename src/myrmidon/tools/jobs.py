"""Job tools: look at a tracked job, collect it once it has ended, cancel it."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.lineage import COLLECTS
from myrmidon.core.observation import Observation, ToolError
from myrmidon.core.tool import REGISTRY, ToolContext, tool
from myrmidon.core.validation import Precondition
from myrmidon.execution import receipt
from myrmidon.execution.backend import JobState
from myrmidon.execution.collect import collect_job, run_progress
from myrmidon.execution.jobs import refresh
from myrmidon.execution.jobstore import open_jobstore


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


@tool(
    "job_status",
    description="One tracked job: its state and progress while it runs; once ended, the same call "
                "collects it (outputs hashed into the ledger, or the log's last lines and the "
                "scheduler's word on a failure). Pass wait_seconds to block until it ends instead "
                "of polling turn by turn.",
    tags={"jobs"},
)
def job_status(ctx: ToolContext, params: JobStatusParams) -> Observation:
    record, state, exit_code, changed = refresh(ctx, params.job_id)
    if params.wait_seconds and not state.terminal:
        deadline = time.monotonic() + params.wait_seconds
        # first look soon, then back off to every 5 s: a job that ends in half a second is
        # collected at once, and a scheduler is asked a handful of times more at most
        interval = 0.2
        while time.monotonic() < deadline:
            time.sleep(min(interval, max(0.1, deadline - time.monotonic())))
            interval = min(5.0, interval * 2)
            record, state, exit_code, moved = refresh(ctx, params.job_id)
            changed = changed or moved
            if state.terminal:
                break
    if state.terminal:
        return collected(ctx, params.job_id, state, exit_code)

    progress, line = run_progress(record)
    obs = Observation(
        ok=True,
        summary=f"{params.job_id} is {state.value}" + (f", {line}" if line else ""),
        job_id=params.job_id,
        state=state.value,
        exit_code=exit_code,
        progress=progress,
        core_h_estimate=record.meta.get("core_h_estimate"),
    )
    # A poll that found nothing new is not an event: on a long campaign unchanged polls
    # would otherwise be most of the ledger, burying the calls that mattered.
    # One that did see a change is still a look, not an action, so it is recorded as an
    # observation: the summary counts it apart from the work.
    obs.record = changed
    obs.kind = "observe"
    return obs


def collected(ctx: ToolContext, job_id: str, state: JobState,
              exit_code: int | None) -> Observation:
    """A job that has ended, collected exactly once into the ledger.

    The first look after the end is the collection — a ``call`` event carrying the
    artifacts (they are in the ledger because *this* call is the one that finds out).
    Every later look returns the same outcome as an observation.
    """
    store = open_jobstore(ctx.workspace)
    record = store.get(job_id)
    already = bool(record.meta.get("collected"))
    obs = collect_job(ctx, record, state, exit_code)
    _held_to_receipt(ctx, obs, record)
    for key, value in (
        ("job_id", job_id), ("state", state.value), ("exit_code", exit_code),
        ("core_h_estimate", record.meta.get("core_h_estimate")),
    ):
        if getattr(obs, key, None) in (None, ""):
            setattr(obs, key, value)
    if not obs.summary.startswith(job_id):
        obs.summary = f"{job_id} {state.value.lower()}: {obs.summary}"
    if already:
        obs.kind, obs.record = "observe", False
    else:
        store.set_meta(job_id, collected=True)
        obs.kind, obs.record = "call", True
        # the job's outputs were made from what its submission read: the ledger's edge
        # from a run's files back to its inputs is this event naming that call
        if record.meta.get("event_id"):
            obs.refs = [*obs.refs, f"{COLLECTS}{record.meta['event_id']}"]
    return obs


def _held_to_receipt(ctx: ToolContext, obs: Observation, record) -> None:
    """The job's node receipt against the inputs its submission hashed: a file that changed
    before the engine read it — other than into what a job this one waited on wrote, by its
    collection in the ledger — ran an experiment nobody approved, so the result is refused."""
    moved, execution = receipt.changed(record.spec.receipt, record.meta.get("input_hashes", {}))
    if execution:
        obs.execution = execution
    if moved:
        written = _written_by(ctx, record.meta.get("after") or [])
        moved = {p: sha for p, sha in moved.items()
                 if written.get(str(Path(p).resolve())) != sha}
    if not moved:
        return
    names = ", ".join(ctx.workspace.relpath(Path(p)) for p in sorted(moved))
    refusal = Precondition(
        id="RECEIPT_INPUT_CHANGED",
        message=f"{names} changed between the submission and the job's start on the node: "
        "the run read other bytes than the ones the call was checked and booked against",
        suggestion="build the input again and submit it anew; the run's files are kept but "
        "not accepted")
    obs.ok = False
    obs.findings = [refusal, *obs.findings]
    obs.error = ToolError(code=refusal.id, message=refusal.message, hint=refusal.suggestion)
    obs.summary = f"{record.job_id} not accepted: {refusal.message}"


def _written_by(ctx: ToolContext, jobs: list[str]) -> dict[str, str]:
    """absolute path -> hash, for what the given jobs wrote as their collections recorded
    it; a job that has ended uncollected is collected first, so the ledger can say."""
    if not jobs:
        return {}
    for job_id in jobs:
        rec, state, _, _ = refresh(ctx, job_id)
        if state.terminal and not rec.meta.get("collected"):
            REGISTRY.call("job_status", ctx, {"job_id": job_id})
    root = Path(ctx.workspace.root)
    out = {}
    for event in ctx.workspace.ledger.events():
        if event.tool == "job_status" and event.ok and event.params.get("job_id") in jobs:
            for art in event.outputs:
                path = Path(art.path) if Path(art.path).is_absolute() else root / art.path
                out[str(path.resolve())] = art.sha256
    return out


@tool(
    "job_cancel",
    description="Cancel a tracked job (best effort).",
    tags={"jobs"},
)
def job_cancel(ctx: ToolContext, params: JobIdParams) -> Observation:
    store = open_jobstore(ctx.workspace)
    record = store.get(params.job_id)
    if record.state.terminal:
        return Observation(
            ok=False,
            summary=f"{params.job_id} already {record.state.value}",
            job_id=params.job_id,
            state=record.state.value,
        )
    ctx.backend.cancel(record.handle, record.spec)
    store.update(params.job_id, JobState.CANCELLED)
    return Observation(
        ok=True,
        summary=f"{params.job_id} cancelled",
        job_id=params.job_id,
        state=JobState.CANCELLED.value,
    )
