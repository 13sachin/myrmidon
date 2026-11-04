"""What a job shows while it runs and returns once it has ended: one reader per job kind,
registered by the module whose tool submits that kind."""

from __future__ import annotations

from collections.abc import Callable

from myrmidon.core.observation import Observation, RunProgress
from myrmidon.core.tool import ToolContext
from myrmidon.execution.backend import JobState
from myrmidon.execution.jobs import refresh
from myrmidon.execution.jobstore import JobRecord, open_jobstore

Collector = Callable[[ToolContext, JobRecord, JobState, "int | None"], Observation]
Progress = Callable[[JobRecord], "tuple[RunProgress | None, str]"]

#: a job record that names no kind is a GROMACS run
DEFAULT_KIND = "gromacs"

_COLLECTORS: dict[str, Collector] = {}
_PROGRESS: dict[str, Progress] = {}


def collector(kind: str):
    """Register what ``job_status`` returns for a job of ``kind`` once it has ended."""
    def wrap(fn: Collector) -> Collector:
        _COLLECTORS[kind] = fn
        return fn
    return wrap


def progress(kind: str):
    """Register the engine's own counter for a job of ``kind`` while it runs."""
    def wrap(fn: Progress) -> Progress:
        _PROGRESS[kind] = fn
        return fn
    return wrap


def kind_of(record: JobRecord) -> str:
    return record.meta.get("kind") or DEFAULT_KIND


def collect_job(ctx: ToolContext, record: JobRecord, state: JobState,
                exit_code: int | None) -> Observation:
    """The outcome of a job that has ended: artifacts hashed, results parsed, or a
    diagnosis with the log tail, by the collector its kind registered."""
    return _COLLECTORS[kind_of(record)](ctx, record, state, exit_code)


def run_progress(record: JobRecord) -> tuple[RunProgress | None, str]:
    """How far a running job has got, by its kind's reader; (None, "") when it has none."""
    reader = _PROGRESS.get(kind_of(record))
    return reader(record) if reader else (None, "")


def pending_lines(ctx: ToolContext) -> list[str]:
    """One sentence per job this study is still waiting on, each after one poll.

    The look a session opens with. The job store holds the last state anybody asked for,
    so a job that ended while nobody was watching is indistinguishable there from one
    still queueing: each is refreshed against the backend here. A job that has ended
    names the call that collects it; collecting is `job_status`, exactly once, with the
    artifacts, and never this look.
    """
    lines: list[str] = []
    for record in open_jobstore(ctx.workspace).list():
        if record.state.terminal and record.meta.get("collected"):
            continue
        _, state, _, _ = refresh(ctx, record.job_id)
        what = f" ({record.label})" if record.label else ""
        if state.terminal:
            lines.append(
                f"{record.job_id} {state.value}{what}, not yet collected — "
                f"call job_status job_id={record.job_id}"
            )
            continue
        bits = [f"{record.job_id} {state.value}{what}"]
        _, line = run_progress(record)
        if line:
            bits.append(line)
        est = record.meta.get("core_h_estimate")
        if est is not None:
            bits.append(f"{est} core-h booked")
        lines.append(", ".join(bits))
    return lines
