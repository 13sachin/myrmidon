"""``myrmidon watch``: what is happening, one line at a time.

An unattended run is not a conversation, so following it is not a transcript. The
feed here is the ledger — every tool call and gate refusal the registry records — plus
the state of the jobs and, while one is running, the engine's own step counter. Nothing
in it comes from a model's output stream, so the view does not know or care which agent
framework is driving.

Three questions a waiting person has, and nothing else earns a line: is it alive and
what is it doing, what did it decide and why, and is anything waiting on me.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from myrmidon.execution.backend import JobState
from myrmidon.execution.jobstore import JobRecord, JobStore

#: job states worth a line while they last; a finished job is reported once and dropped
LIVE = (JobState.QUEUED, JobState.RUNNING)


def _clock(when: datetime | None = None) -> str:
    return (when or datetime.now()).strftime("%H:%M:%S")


def header(ws) -> list[str]:
    """What the run is, before the feed starts. Printed once."""
    name = ws.meta.get("name", ws.root.name)
    lines = [f"myrmidon · {name} · {ws.root}"]
    jobs = _jobs(ws)
    booked = round(sum(r.meta.get("core_h_estimate") or 0.0 for r in jobs), 3)
    if jobs:
        lines.append(f"{len(jobs)} job(s) · {booked} core-h booked")
    lines.append("following the ledger — ctrl-c leaves the run alone")
    lines.append("─" * 74)
    return lines


def _jobs(ws, store: JobStore | None = None) -> list[JobRecord]:
    db = ws.state_dir / "jobs.sqlite"
    if not db.exists():
        return []
    return (store or JobStore(db)).list()


def running_progress(record: JobRecord) -> str:
    """The engine's own counter for a job in flight.

    Without this a twenty-minute segment is a frozen screen, and a frozen screen is
    indistinguishable from a hang. Both readers already exist for diagnosis; here they
    are what makes waiting legible.
    """
    cwd = Path(record.spec.cwd)
    if record.meta.get("kind") == "mimic":
        out = cwd / "cpmd.out"
        if not out.exists():
            return ""
        from myrmidon.engines.mimic.diagnostics import parse_cpmd_output

        facts = parse_cpmd_output(out.read_text(errors="replace"))
        steps = facts.get("md_steps_completed")
        temp = facts.get("t_qm_last")
        parts = [f"step {steps}" if steps is not None else "", 
                 f"T(QM) {temp} K" if temp is not None else ""]
        return " · ".join(p for p in parts if p)
    from myrmidon.engines.gromacs.results import _parse_progress

    argv = record.spec.argv
    deffnm = argv[argv.index("-deffnm") + 1] if "-deffnm" in argv else "md"
    progress = _parse_progress(cwd / f"{deffnm}.log")
    if progress is None:
        return ""
    parts = [f"step {progress.step}" if progress.step is not None else "",
             f"{progress.time_ps} ps" if progress.time_ps is not None else "",
             f"{progress.ns_per_day} ns/day" if progress.ns_per_day is not None else ""]
    return " · ".join(p for p in parts if p)


def _event_line(ev) -> str:
    when = _clock(ev.timestamp.astimezone() if ev.timestamp else None)
    if ev.kind == "gate_refused":
        return f"{when}  refused   {ev.tool:<18} — a finding needs a decision"
    if ev.tool == "plan_approve" and ev.ok:
        return (f"{when}  approval  {ev.tool:<18} "
                f"accepted {', '.join(ev.warnings_acknowledged) or 'nothing'}")
    if ev.kind == "observe":
        return ""  # a look is not a step; the job lines already say what is happening
    # a ledger event carries what a call consumed and produced, not the observation's
    # sentence — so the line says what changed on disk, which is the durable part
    made = ", ".join(f"{a.role}" for a in ev.outputs[:3])
    where = ev.outputs[0].path if ev.outputs else ""
    mark = "call" if ev.ok else "failed"
    tail = f"{'→ ' + where if where else ''}"
    if made and len(ev.outputs) > 1:
        tail += f" (+{len(ev.outputs) - 1} more: {made})"
    return f"{when}  {mark:<9} {ev.tool:<18} {tail}".rstrip()


class Feed:
    """What has happened since the last look.

    The ledger is append-only, so following it is remembering how many events have
    been seen. That count is all the state there is, which is why the view can be
    started, killed and started again from any terminal.
    """

    def __init__(self, ws) -> None:
        self.ws = ws
        # opened once: JobStore.__init__ runs its schema script, which takes a write
        # lock, and a view polling every two seconds must not fight the tools for it
        db = ws.state_dir / "jobs.sqlite"
        self.store = JobStore(db) if db.exists() else None
        self.seen_events = 0
        self.job_state: dict[str, str] = {}
        self.job_progress: dict[str, str] = {}
        self.first = True

    def poll(self) -> list[str]:
        """Only what is new. Called again, it says nothing until something changes."""
        lines: list[str] = []
        events = list(self.ws.ledger.events())
        for ev in events[self.seen_events:]:
            line = _event_line(ev)
            if line:
                lines.append(line)
        self.seen_events = len(events)

        for record in _jobs(self.ws, self.store):
            if self.job_state.get(record.job_id) != record.state.value:
                self.job_state[record.job_id] = record.state.value
                if not self.first or record.state in LIVE:
                    lines.append(
                        f"{_clock()}  job       {record.job_id:<18} {record.state.value}"
                        + (f"  {record.label}" if record.label else "")
                    )
            if record.state in LIVE:
                progress = running_progress(record)
                if progress and self.job_progress.get(record.job_id) != progress:
                    self.job_progress[record.job_id] = progress
                    lines.append(f"{_clock()}  progress  {record.job_id:<18} {progress}")
        self.first = False
        return lines


def follow(ws, poll_s: float = 2.0, once: bool = False) -> Iterator[str]:
    """The feed, forever — or once, which is what a script or a test wants."""
    feed = Feed(ws)
    while True:
        yield from feed.poll()
        if once:
            return
        time.sleep(poll_s)
