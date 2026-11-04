"""Unattended continuation under the mandate: collect, rerun, wake.

A tick reconciles the study's jobs with the backend and collects each one that ended. Every
failed collection in the ledger is handled once, whoever collected it — this tick, an
earlier one, the agent in a session or a wake. A plain failure is rerun identically while
the mandate's `recovery.reruns` allow; a cancelled job is left as it was cancelled; anything
else, a rerun budget spent, every job ended with the study going on, or a newly approved
mandate wakes the agent: the same tools, the same gate, the same mandate, a few turns, with
the study, the log tail of what failed and the sensors' findings on its text inputs. The
wake is the model's turn exactly as in a session; nothing here interprets the study. Each
wake is written to `.myrmidon/wakes/index.jsonl` before it starts and again, with its
transcript's hash, when it ends. What a wake could not do because the gate refused it — the
ceiling, the system, a decision of the researcher's — becomes a question in the inbox, and
so do the mandate's deadline, its cap on the model's fixes, and a wake that left nothing
running and no report.

State is kept per approved mandate in `.myrmidon/recovery.json` and written before each
action it records, so a tick killed at any instant neither repeats a rerun nor loses a wake.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from myrmidon.core import mandate as mandates
from myrmidon.core.lineage import COLLECTS
from myrmidon.core.observation import JobState
from myrmidon.core.tool import REGISTRY, ToolContext, ToolRegistry
from myrmidon.errors import JobNotFoundError
from myrmidon.execution.jobstore import open_jobstore
from myrmidon.orchestration.report import trace_lines
from myrmidon.orchestration.spool import Inbox, Option

LOCK_FILENAME = "tick.lock"
RECOVERY_FILE = "recovery.json"
WAKES_DIR = "wakes"
#: the most turns one wake may take: a session's, as `myrmidon run` allows
WAKE_TURNS = 40
#: the study's last calls a wake is shown, so it need not look them up
RECENT_CALLS = 15
#: an input larger than this is not read for the wake's sensors (trajectories, checkpoints)
SENSED_BYTES = 1_000_000

#: what the gate refuses that only the researcher can grant
THEIRS = ("MANDATE_CEILING", "MANDATE_SYSTEM", "MANDATE_DECISION", "MANDATE_ABSENT")
CONTINUE = Option(id="continue",
                  text="wake the agent once more under the mandate as it stands")
STOP = Option(id="stop", text="leave the study where it is")
REVISE = Option(id="revise",
                text="revise the mandate in a session; its approval carries the study on")


@dataclass
class Report:
    blocked: str = ""
    collected: list[str] = field(default_factory=list)
    reran: list[str] = field(default_factory=list)
    running: list[str] = field(default_factory=list)
    woke: str = ""
    parked: list[str] = field(default_factory=list)
    done: bool = False

    def render(self) -> str:
        if self.blocked:
            return f"tick: {self.blocked}"
        lines = ["tick"]
        for label, items in (("collected", self.collected), ("reran", self.reran),
                             ("running", self.running), ("parked", self.parked)):
            lines += [f"  {label:<9} {item}" for item in items]
        if self.woke:
            lines.append(f"  woke      {self.woke}")
        if self.done:
            lines.append("the study's report is written after its last job: nothing to do")
        return "\n".join(lines)


@dataclass
class WakeResult:
    transcript: Path | None
    stop: str
    answer: str = ""


#: (prompt, transcript directory, most turns) -> how the wake ended
Wake = Callable[[str, Path, int], WakeResult]


@dataclass
class Failure:
    """A failed collection: the collecting event, the submission it collects, the job."""

    collection: object
    submission: object
    state: JobState
    exit_code: int | None
    rerun: object = None  # the observation of a rerun that did not go through


class State:
    """What the continuation has done. Facts about the study persist across mandates: the
    failures handled and those a wake has yet to carry, the jobs a tick collected, the
    rerun chains, the collections a wake was told of. What a mandate allowed starts afresh
    with a newly approved one: the reruns and fix wakes spent, the wakes the researcher
    added, the questions asked and answered, whether the agent was woken or the researcher
    stopped."""

    STUDY = {"handled": [], "pending": [], "collected": [], "chain": {}, "shown": []}
    MANDATE = {"digest": "", "reruns": {}, "fixes": 0, "granted": 0,
               "parked": [], "read": [], "stopped": False, "woke": False,
               "notified": False}
    FIELDS = {**STUDY, **MANDATE}

    def __init__(self, path: Path, digest: str):
        self.path = path
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        if data.get("digest") != digest:
            data = {**{k: data[k] for k in self.STUDY if k in data}, "digest": digest}
        for key, default in self.FIELDS.items():
            setattr(self, key, data.get(key, json.loads(json.dumps(default))))

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: getattr(self, k) for k in self.FIELDS},
                                  indent=1, sort_keys=True))
        tmp.replace(self.path)


def _lock(path: Path):
    handle = open(path, "a+")  # noqa: SIM115 — closed by the caller
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def tick(ctx: ToolContext, *, wake: Wake | None = None,
         registry: ToolRegistry = REGISTRY) -> Report:
    """One pass under the approved mandate. Nothing runs without one that holds; a second
    tick that finds the study locked returns at once saying so."""
    ws = ctx.workspace
    report = Report()
    mandate = mandates.holding(ws)
    if mandate is None:
        report.blocked = mandates.why_not_holding(ws) or "this study has no approved mandate"
        return report
    lock = _lock(ws.state_dir / LOCK_FILENAME)
    if lock is None:
        report.blocked = "another tick holds this study"
        return report
    was = ctx.unattended, ctx.person, ctx.model_driven
    ctx.unattended, ctx.person = True, None
    state = State(ws.state_dir / RECOVERY_FILE, mandate.digest)
    try:
        _pass(ctx, mandate, state, report, wake, registry)
    finally:
        state.save()
        ctx.unattended, ctx.person, ctx.model_driven = was
        lock.close()
    return report


def _pass(ctx, mandate, state: State, report: Report, wake, registry) -> None:
    ws = ctx.workspace
    inbox = Inbox(ws.root)
    _read_answers(inbox, state)
    if state.stopped:
        report.blocked = "the researcher stopped this study under this mandate"
        return
    waiting = [q for q in inbox.open_questions() if q.entry_id in state.parked]
    if waiting:
        report.parked = [f"{q.entry_id} {q.question}" for q in waiting]
        return
    _collect(ctx, state, report, registry)
    if _past_deadline(ws, mandate):
        _park(inbox, state, report, "The mandate's deadline has passed; nothing more runs "
              "under it.", [], [STOP, REVISE])
        return
    # failures a wake has yet to take, from an earlier tick that parked them, first
    waking = _failures(ws, state.pending)
    announced = len(waking)  # the failures already put to a wake or the researcher
    since = _approved_at(ws)
    for failure in _failures(ws, state.handled, new=True):
        state.handled.append(failure.collection.event_id)
        state.save()  # handled once, even if the tick dies inside the rerun
        if since is not None and _stamp(failure.collection.timestamp) < since:
            continue  # failed before any mandate: the session that saw it dealt with it
        if failure.state == JobState.CANCELLED and not failure.submission.params.get("after"):
            continue  # cancelled by a person or by the agent: left as it was
        # a failure the agent collected is the agent's to fix: rerunning it could undo
        # the fix it has already submitted
        if (failure.collection.params.get("job_id") not in state.collected
                or not _rerunnable(failure, state, mandate)):
            waking.append(failure)
            state.pending.append(failure.collection.event_id)
            continue
        root = state.chain.get(failure.submission.event_id, failure.submission.event_id)
        rerun = registry.call(failure.submission.tool, ctx, dict(failure.submission.params))
        if rerun.ok and getattr(rerun, "job_id", None) and rerun.provenance_id:
            state.reruns[root] = state.reruns.get(root, 0) + 1
            state.chain[rerun.provenance_id] = root
            state.save()
            report.reran.append(f"{failure.submission.tool} as {rerun.job_id}")
            report.running.append(rerun.job_id)
            continue
        if rerun.error is not None and rerun.error.code in THEIRS:
            state.pending.append(failure.collection.event_id)
            _park(inbox, state, report, f"The rerun of {failure.submission.tool} was refused: "
                  f"{rerun.error.message}", [rerun.summary], [CONTINUE, STOP, REVISE])
            return
        failure.rerun = rerun
        waking.append(failure)
        state.pending.append(failure.collection.event_id)
    unseen = [e.event_id for e in _collections(ws) if e.event_id not in state.shown
              and (since is None or _stamp(e.timestamp) >= since)]
    # while jobs run, only a failure the tick collected wakes the agent: one it collected
    # itself may already have its fix running
    if report.running and not any(f.collection.params.get("job_id") in state.collected
                                  for f in waking):
        return
    if not waking and state.woke and _reported(ws):
        report.done = True
        return
    if not waking and not unseen and state.woke and wake is not None:
        _park(inbox, state, report, "The agent's last turn left nothing running and no "
              "report.", _last_answer(ws), [CONTINUE, STOP, REVISE])
        return
    if waking and state.fixes >= mandate.recovery.model_fixes + state.granted:
        _park(inbox, state, report, "The mandate's model fixes are spent and a run still "
              "fails.", [_line(f) for f in waking], [CONTINUE, STOP, REVISE])
        return
    if wake is None:
        if unseen or waking[announced:] or not state.notified:
            inbox.notify("study", "The study waits on the agent and no wake model is "
                         "configured: open it in a session to carry it on.",
                         [_line(f) for f in waking])
        state.shown += unseen
        state.notified = True
        return
    _wake(ctx, mandate, state, report, wake, registry, waking, unseen, inbox)


def _rerunnable(failure: Failure, state: State, mandate) -> bool:
    """An identical rerun can only help a plain failure of a job that waits on nothing,
    within the mandate's reruns: a timeout repeats, a skipped dependant names the parent
    that failed, and one whose inputs changed under it would run the changed bytes."""
    root = state.chain.get(failure.submission.event_id, failure.submission.event_id)
    return (failure.state == JobState.FAILED
            and "RECEIPT_INPUT_CHANGED" not in failure.collection.knowledge_refs
            and not failure.submission.params.get("after")
            and state.reruns.get(root, 0) < mandate.recovery.reruns)


def _wake(ctx, mandate, state, report, wake, registry, waking, unseen, inbox) -> None:
    ws = ctx.workspace
    prompt = compose(ctx, mandate, waking, unseen, registry)
    directory = ws.state_dir / WAKES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    started = {"ts": _now(), "fix": bool(waking), "prompt_sha256": _sha(prompt.encode())}
    _index(directory, {**started, "event": "started"})
    # the wake is counted before it runs: a tick killed inside it does not redo it free
    state.shown += unseen
    state.fixes += bool(waking)
    state.woke = True
    state.pending = []
    state.save()
    before = len(list(ws.ledger.events()))
    result = wake(prompt, directory, WAKE_TURNS)
    ended = {**started, "event": "ended", "stop": result.stop, "answer": result.answer[:2000]}
    if result.transcript is not None and result.transcript.is_file():
        ended |= {"transcript": result.transcript.name,
                  "sha256": _sha(result.transcript.read_bytes())}
    _index(directory, ended)
    report.woke = result.stop + (f": {result.answer[:160]}" if result.answer else "")
    refused = [e for e in list(ws.ledger.events())[before:]
               if e.kind == "gate_refused" and set(e.knowledge_refs) & set(THEIRS)]
    if refused:
        _park(inbox, state, report, "The agent needed what the mandate does not grant; a "
              "revision is the researcher's.",
              [f"{e.tool}: {', '.join(sorted(set(e.knowledge_refs) & set(THEIRS)))}"
               for e in refused], [STOP, REVISE])


def compose(ctx, mandate, waking: list[Failure], unseen: list[str],
            registry=REGISTRY) -> str:
    """The wake's prompt, from the study alone: the mandate, what ended since the last
    turn, and for a failure its log's last lines and the sensors' findings on its text
    inputs."""
    from myrmidon.execution.jobs import booked_under_mandates

    ws = ctx.workspace
    events = list(ws.ledger.events())
    by_id = {e.event_id: e for e in events}
    spent = booked_under_mandates(ws, mandates.approved_digests(ws))
    lines = [
        f"You are carrying on the study at {ws.root} unattended; it is open. No person is "
        "present: what its approved mandate does not grant — a bigger ceiling, a different "
        "system, a decision of the researcher's — is not yours. Say it and end the turn; "
        "it is put to them.",
        "", mandates.render(mandate, spent), "",
        "The study's last calls:"]
    lines += [*trace_lines(events, RECENT_CALLS), ""]
    lines.append("What ended since the last turn:" if unseen or waking
                 else "Nothing has run under it yet: begin with its first stage.")
    for event_id in unseen:
        event = by_id.get(event_id)
        if event is not None:
            lines.append(f"- {event.params.get('job_id', '')}: "
                         + ("collected" if event.ok else "failed"))
    for failure in waking:
        lines += ["", _line(failure)]
        if failure.rerun is not None:
            lines.append(f"Its identical rerun did not go through: {failure.rerun.summary}")
        tail = failure.collection.log_tail or ""
        if tail:
            lines += ["Its log ends:", *tail.splitlines()[-20:]]
        for art in failure.submission.inputs:
            if art.size_bytes > SENSED_BYTES:
                continue
            check = registry.call("file_check", ctx, {"path": art.path})
            seen = [f.message for f in check.findings]
            if seen:
                lines += [f"The sensors on {art.path}:", *(f"  - {m}" for m in seen[:8])]
    lines += ["", "Carry the study on toward the deliverables — the next stage, or a fix to "
              "what failed and its run — within the ceiling. When every deliverable is in "
              "hand, call study_report."]
    return "\n".join(lines)


def _line(failure: Failure) -> str:
    job = failure.collection.params.get("job_id", "")
    code = f", exit {failure.exit_code}" if failure.exit_code is not None else ""
    return f"{failure.submission.tool} ({job}) ended {failure.state.value}{code}"


def _collect(ctx, state: State, report: Report, registry) -> None:
    """Every tracked job that has ended, collected through job_status; a job still
    running is named in the report."""
    from myrmidon.execution.jobs import existing_jobstore, refresh

    store = existing_jobstore(ctx.workspace)
    if store is None:
        return
    for rec in store.list():
        if rec.meta.get("collected"):
            continue
        _, job_state, _, _ = refresh(ctx, rec.job_id)
        if not job_state.terminal:
            report.running.append(rec.job_id)
            continue
        state.collected.append(rec.job_id)
        state.save()  # recorded as the tick's before the collection, which may not return
        registry.call("job_status", ctx, {"job_id": rec.job_id})
        report.collected.append(f"{rec.job_id} {job_state.value.lower()}")


def _failures(ws, ids: list[str], new: bool = False) -> list[Failure]:
    """The failed collections of submissions the ledger records — those named in ``ids``,
    or with ``new`` every one not named there: whoever collected it."""
    events = list(ws.ledger.events())
    by_id = {e.event_id: e for e in events}
    store = open_jobstore(ws)
    out = []
    for event in events:
        if (event.tool != "job_status" or event.kind != "call" or event.ok
                or (event.event_id in ids) is new):
            continue
        ref = next((r for r in event.knowledge_refs if r.startswith(COLLECTS)), "")
        submission = by_id.get(ref.removeprefix(COLLECTS))
        if submission is None:
            continue
        try:
            record = store.get(event.params.get("job_id", ""))
        except JobNotFoundError:
            continue  # its record is gone: nothing to rerun or report
        out.append(Failure(event, submission, record.state, record.exit_code))
    return out


def _collections(ws) -> list:
    return [e for e in ws.ledger.events() if e.tool == "job_status" and e.kind == "call"]


def _reported(ws) -> bool:
    """Whether the study's report was written after its last collection."""
    last = None
    for event in ws.ledger.events():
        if event.tool == "job_status" and event.kind == "call":
            last = None
        elif event.tool == "study_report" and event.ok:
            last = event
    return last is not None


def _last_answer(ws) -> list[str]:
    ended = [json.loads(line) for line in _index_lines(ws) if '"ended"' in line]
    return [f"the agent said: {ended[-1]['answer'][:400]}"] if ended and ended[-1].get(
        "answer") else []


def _stamp(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _approved_at(ws) -> datetime | None:
    """When the study's first mandate was approved: what failed before it was a session's."""
    first = next((a for a in mandates.approvals(ws) if a.timestamp is not None), None)
    return _stamp(first.timestamp) if first is not None else None


def _past_deadline(ws, mandate) -> bool:
    hours = mandate.recovery.deadline_h
    approved = [a for a in mandates.approvals(ws) if a.digest == mandate.digest]
    if not hours or not approved or approved[-1].timestamp is None:
        return False
    stamp = approved[-1].timestamp
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return (datetime.now(UTC) - stamp).total_seconds() > hours * 3600


def _park(inbox: Inbox, state: State, report: Report, question: str, evidence: list[str],
          options: list[Option]) -> None:
    entry = inbox.ask("study", question, evidence, options)
    state.parked.append(entry.entry_id)
    report.parked.append(f"{entry.entry_id} {question}")


def _read_answers(inbox: Inbox, state: State) -> None:
    """The researcher's answers to the questions asked under this mandate: `stop` ends
    it, `continue` adds one wake (a fix wake when one is needed), `revise` waits on the
    revision."""
    for line in inbox.all():
        if line.answers not in state.parked or line.answers in state.read:
            continue
        state.read.append(line.answers)
        if line.option == "stop":
            state.stopped = True
        elif line.option == "continue":
            state.granted += 1
            state.woke = False  # one more wake, even with nothing new to tell


def progress(ws) -> str:
    """One line on the continuation under the approved mandate, from its files: reruns,
    fix wakes against the mandate's, the last wake. Empty before any tick acted."""
    mandate = mandates.holding(ws)
    path = ws.state_dir / RECOVERY_FILE
    if mandate is None or not path.is_file():
        return ""
    state = State(path, mandate.digest)
    if not (state.reruns or state.woke or state.fixes):
        return ""
    ended = [json.loads(line) for line in _index_lines(ws) if '"ended"' in line]
    last = f"; last wake {ended[-1]['ts']} ({ended[-1]['stop']})" if ended else ""
    return (f"continuation: {sum(state.reruns.values())} rerun(s), {state.fixes} of "
            f"{mandate.recovery.model_fixes + state.granted} fix wake(s){last}")


def _index_lines(ws) -> list[str]:
    try:
        return (ws.state_dir / WAKES_DIR / "index.jsonl").read_text().splitlines()
    except OSError:
        return []


def _index(directory: Path, record: dict) -> None:
    with (directory / "index.jsonl").open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
