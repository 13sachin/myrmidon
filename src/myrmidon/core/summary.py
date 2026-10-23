"""Session summary: make the harness visible.

Counted from the workspace ledger and job store over a time window — what
went through the tools, what was refused, what it cost, what it produced.
Anything that bypassed the tools is absent by construction; the summary says
so, because that absence is the point.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime, timedelta

from myrmidon.core.observation import JobTally, SessionSummary, ToolTally
from myrmidon.core.provenance import ProvenanceEvent
from myrmidon.core.workspace import Workspace
from myrmidon.execution.accounting import spec_core_h
from myrmidon.execution.jobstore import JobStore

#: the calls that open a session's work in a study; the default window starts at the last
_SESSION_OPENERS = {"study_create", "study_open"}
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([mhd])$")
#: tools that hand a parameter file to an engine, and the parameter naming it. The
#: summary asks the same question the gate asked: does anything record where those
#: values came from?
_PARAM_FILE = {"gmx_grompp": "mdp", "mimic_prepare": "cpmd_input"}


def window_start(since: str | None, events: list[ProvenanceEvent]) -> datetime:
    """'session' (default) = the last workspace open/create event; '6h', '90m',
    '2d' = that long ago; anything else = ISO-8601 timestamp."""
    now = datetime.now(UTC)
    if not since or since == "session":
        for ev in reversed(events):
            if ev.tool in _SESSION_OPENERS:
                return ev.timestamp
        return events[0].timestamp if events else now
    if since == "all":
        return events[0].timestamp if events else now
    m = _DURATION.match(since.strip())
    if m:
        amount, unit = float(m.group(1)), m.group(2)
        delta = {"m": timedelta(minutes=amount), "h": timedelta(hours=amount),
                 "d": timedelta(days=amount)}[unit]
        return now - delta
    stamp = datetime.fromisoformat(since)
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def summarize(
    ws: Workspace, since: str | None = "session", backend=None, compute=None,
    poll_usage: bool = True,
) -> SessionSummary:
    events = list(ws.ledger.events())
    start = window_start(since, events)
    end = datetime.now(UTC)
    window = [ev for ev in events if ev.timestamp >= start]

    calls: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    refusals = 0
    observations = 0
    approvals = 0
    approved_acks: list[str] = []
    acknowledged: list[str] = []
    attached: list[str] = []
    roles: Counter[str] = Counter()
    hashes: set[str] = set()
    for ev in window:
        # a poll is a look at something already running, not a call the study made:
        # counted, but never as one of the actions
        if ev.kind == "observe":
            observations += 1
            continue
        calls[ev.tool] += 1
        if ev.tool == "plan_approve" and ev.ok:
            # a person's approval of the study's plan: the findings it acknowledged are
            # theirs, and a later call that relied on it names it (`approval`)
            approvals += 1
        if ev.kind == "gate_refused":
            refusals += 1
        elif not ev.ok:
            failed[ev.tool] += 1
        acknowledged += [w for w in ev.warnings_acknowledged if w not in acknowledged]
        if ev.approval:
            approved_acks += [w for w in ev.warnings_acknowledged if w not in approved_acks]
        attached += [ref for ref in ev.knowledge_refs if ref not in attached]
        for art in ev.outputs:
            if art.sha256 not in hashes:
                hashes.add(art.sha256)
                roles[art.role] += 1

    protocols_used: list[str] = []
    runs_sourced: list[str] = []
    runs_unsourced: list[str] = []
    for ev in window:
        if ev.tool == "inputset_apply" and ev.ok:
            name = ev.params.get("inputset")
            stage = ev.params.get("stage")
            label = f"{name} stage {stage}" if stage else str(name)
            if name and label not in protocols_used:
                protocols_used.append(label)
        key = _PARAM_FILE.get(ev.tool)
        if key and ev.kind == "call" and ev.params.get(key):
            from myrmidon.core.sourcing import parameter_source

            origin = parameter_source(ws, ev.params[key])
            line = f"{ev.tool} {origin.line}"
            (runs_sourced if origin.sourced else runs_unsourced).append(line)

    jobs: list[JobTally] = []
    core_h_est = 0.0
    core_h_actual = 0.0
    any_actual = False
    scripts: list[str] = []
    jobs_db = ws.state_dir / "jobs.sqlite"
    if jobs_db.exists():
        for rec in JobStore(jobs_db).list():
            submitted = datetime.fromisoformat(rec.submitted_at)
            if submitted.tzinfo is None:
                submitted = submitted.replace(tzinfo=UTC)
            if submitted < start.replace(microsecond=0):  # submitted_at has 1 s resolution
                continue
            est = rec.meta.get("core_h_estimate")
            if est is None and compute is not None:
                est = spec_core_h(rec.spec, compute)
            actual = None
            if poll_usage and backend is not None and rec.state.terminal:
                actual = backend.usage(rec.handle, rec.spec)
            if actual is not None:
                core_h_actual += actual
                any_actual = True
            if est is not None:
                core_h_est += est
            jobs.append(JobTally(
                job_id=rec.job_id, label=rec.label, state=rec.state.value,
                submitted_at=rec.submitted_at, core_h_estimate=est, core_h_actual=actual,
            ))
            if rec.meta.get("kind") == "script":
                commit = rec.meta.get("git_commit")
                dirty = "+dirty" if rec.meta.get("git_dirty") else ""
                tag = f" @ {commit}{dirty}" if commit else ""
                entry = f"{rec.meta.get('script')}{tag}"
                if entry not in scripts:
                    scripts.append(entry)

    n_tools = sum(calls.values())
    summary = SessionSummary(
        ok=True,
        summary=(
            f"{n_tools} tool call(s) across {len(calls)} tool(s) since "
            f"{start.isoformat(timespec='minutes')}; "
            + (f"{observations} job observation(s); " if observations else "")
            + f"{refusals} gate refusal(s); "
            f"{len(jobs)} job(s), {round(core_h_est, 1)} core-h booked"
            + (f", {round(core_h_actual, 1)} consumed" if any_actual else "")
            + f"; {len(hashes)} artifact(s). Work done outside these tools is not here."
        ),
        workspace=str(ws.root),
        window_start=start.isoformat(timespec="seconds"),
        window_end=end.isoformat(timespec="seconds"),
        n_events=len(window),
        tools=[ToolTally(tool=t, calls=n, failed=failed[t]) for t, n in calls.most_common()],
        observations=observations,
        gate_refusals=refusals,
        approvals=approvals,
        warnings_acknowledged=acknowledged,
        acknowledged_under_approval=[w for w in approved_acks if w in acknowledged],
        knowledge_refs=attached,
        jobs=jobs,
        core_h_submitted=round(core_h_est, 3),
        core_h_consumed=round(core_h_actual, 3) if any_actual else None,
        n_artifacts=len(hashes),
        artifacts_by_role=dict(roles),
        scripts=scripts,
        protocols_used=protocols_used,
        runs_sourced=runs_sourced,
        runs_unsourced=runs_unsourced,
    )
    summary.record = False  # reading the ledger is not an event in it
    return summary


def render(s: SessionSummary) -> str:
    """Plain-text report for the terminal."""
    lines = [
        f"myrmidon session summary — {s.workspace}",
        f"window: {s.window_start} → {s.window_end}",
        "",
        f"tool calls: {s.n_events - s.observations}"
        + (f" ({s.gate_refusals} refused by gates)" if s.gate_refusals else "")
        + (f", plus {s.observations} job observation(s)" if s.observations else ""),
    ]
    for t in s.tools:
        flag = f"  ({t.failed} failed)" if t.failed else ""
        lines.append(f"  {t.calls:4d}  {t.tool}{flag}")
    if s.approvals:
        lines.append(
            f"study approvals by the user: {s.approvals}"
            + (f" (accepted {', '.join(s.acknowledged_under_approval)})"
               if s.acknowledged_under_approval else "")
        )
    if s.warnings_acknowledged:
        own = [w for w in s.warnings_acknowledged if w not in s.acknowledged_under_approval]
        lines.append(
            f"acknowledged warnings: {', '.join(s.warnings_acknowledged)}"
            + (f" — {', '.join(own)} accepted by the agent itself, under no approval"
               if own and s.acknowledged_under_approval else "")
        )
    if s.knowledge_refs:
        lines.append(f"findings surfaced: {', '.join(s.knowledge_refs)}")
    lines.append("")
    if s.jobs:
        consumed = f", {s.core_h_consumed} consumed" if s.core_h_consumed is not None else ""
        lines.append(f"jobs: {len(s.jobs)} — {s.core_h_submitted} core-h booked{consumed}")
        for j in s.jobs:
            est = f"{j.core_h_estimate} core-h" if j.core_h_estimate is not None else "?"
            act = f" (used {j.core_h_actual})" if j.core_h_actual is not None else ""
            lines.append(f"  {j.job_id}  {j.state:<10} {est}{act}  {j.label or ''}")
    else:
        lines.append("jobs: none")
    if s.scripts:
        lines.append("scripts run through the harness:")
        lines += [f"  {sc}" for sc in s.scripts]
    if s.protocols_used:
        lines.append("inputsets applied: " + ", ".join(s.protocols_used))
    if s.runs_sourced or s.runs_unsourced:
        lines.append(
            f"engine inputs: {len(s.runs_sourced)} with a recorded source, "
            f"{len(s.runs_unsourced)} without"
        )
        for line in s.runs_unsourced:
            lines.append(f"  no source: {line}")
    roles = ", ".join(f"{n} {r}" for r, n in sorted(s.artifacts_by_role.items()))
    lines.append(f"artifacts: {s.n_artifacts}" + (f" ({roles})" if roles else ""))
    lines.append("")
    lines.append("Anything run outside these tools is not in the ledger and not above.")
    return "\n".join(lines)


def statusline(ws: Workspace) -> str:
    """One cheap line for a terminal status bar: the job states as of the last look.

    It never polls the backend — it is redrawn every few seconds, and a status bar that
    queried the scheduler each time would be a scheduler load, not a status bar. A job
    that ended is therefore still shown running here until something asks: opening the
    workspace does (`study_open` refreshes every pending job), and so does
    `job_status`, which is also what collects it.
    """
    name = ws.meta.get("name", ws.root.name)
    events = list(ws.ledger.events())
    parts = [f"myrmidon {name}", f"{len(events)} ev"]
    jobs_db = ws.state_dir / "jobs.sqlite"
    if jobs_db.exists():
        records = JobStore(jobs_db).list()
        states = Counter(r.state.value for r in records)
        live = {k: v for k, v in states.items() if k in ("QUEUED", "RUNNING")}
        if live:
            parts.append(" ".join(f"{v} {k.lower()}" for k, v in live.items()))
        booked = sum(
            (r.meta.get("core_h_estimate") or 0.0) for r in records if not r.state.terminal
        )
        if booked:
            parts.append(f"{round(booked, 1)} core-h in flight")
    if events:
        last = events[-1]
        parts.append(f"last: {last.tool}{'' if last.ok else ' ✗'}")
    return " · ".join(parts)
