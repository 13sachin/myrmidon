"""The study's loop files: the event spool, the journal, the inbox, and the loop's own
working state. Every one lives in the study, so no process holds memory the study lacks
.

- ``events.jsonl`` — the **spool**: one line per event of six kinds — ``job`` (a tracked
  job ended, failed or was refused), ``sentinel`` (a predicate over parsed values did not
  hold, or the until-check was evaluated), ``checkpoint`` (a declared look at parsed data),
  ``human`` (an answer or a message), ``budget`` (a ceiling neared or hit), ``deadline``
  (a move ran past its expected duration). Appended by machinery, never rewritten; a tick
  handles each exactly once and records that in the loop state.
- ``journal.jsonl`` — decisions, ruled-out causes, suspicions and intentions, each citing
  ledger or spool events, written by machinery, a model or a person; only open entries are
  shown to a wake. Numbers stay in the ledger; the journal holds *why*.
- ``inbox.jsonl`` — parked questions with the evidence and the options and their costs,
  and the answers, by option id. A question is answered by a person (``myrmidon answer``)
  or by a wake; the next tick carries the answer out.
- ``.myrmidon/loop.json`` — what the loop was doing when it last wrote: the events it has
  handled, attempts per step, the action in flight. A tick killed between an action and
  its record is reconciled against the ledger and the job store, never repeated.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

SPOOL_FILENAME = "events.jsonl"
JOURNAL_FILENAME = "journal.jsonl"
INBOX_FILENAME = "inbox.jsonl"
LOOP_STATE_FILENAME = "loop.json"

EventKind = Literal["job", "sentinel", "checkpoint", "human", "budget", "deadline"]
JournalKind = Literal["decision", "ruled_out", "suspicion", "intention"]
Author = Literal["machinery", "model", "human"]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SpoolEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: "sp-" + uuid.uuid4().hex[:12])
    ts: str = Field(default_factory=now_iso)
    kind: EventKind
    subject: str  # the step (and arm) the event is about: "heating", "sp[frame2]"
    detail: dict[str, Any] = Field(default_factory=dict)
    plan_id: str = ""


class JournalEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: "jn-" + uuid.uuid4().hex[:12])
    ts: str = Field(default_factory=now_iso)
    kind: JournalKind
    by: Author
    text: str
    cites: list[str] = Field(default_factory=list)  # ledger event ids and spool event ids
    status: Literal["open", "closed"] = "open"
    subject: str = ""  # the step it is about, when it is about one
    # a line that closes an earlier entry names it; the entry itself is never rewritten
    closes: str | None = None


class Option(BaseModel):
    id: str  # what a person types: "rerun", "wait", "stop", ...
    action: str  # what the loop does: rerun | extend | wait | continue | stop | approve
    text: str
    cost_core_h: float = 0.0
    args: dict[str, Any] = Field(default_factory=dict)


class InboxEntry(BaseModel):
    """A parked question, or a notice (no options; nothing waits on it)."""

    entry_id: str = Field(default_factory=lambda: "in-" + uuid.uuid4().hex[:12])
    ts: str = Field(default_factory=now_iso)
    kind: Literal["question", "notice"] = "question"
    subject: str  # the branch that is parked: a step (and arm)
    question: str
    evidence: list[str] = Field(default_factory=list)  # one line each; ids at the end
    options: list[Option] = Field(default_factory=list)
    cites: list[str] = Field(default_factory=list)  # ledger and spool event ids
    # an answer line names the entry it answers and nothing else below is set
    answers: str | None = None
    option: str | None = None
    by: str | None = None  # a person's name, or "model:<name>"


class _Jsonl:
    def __init__(self, path: Path, model: type[BaseModel]):
        self.path = path
        self.model = model

    def append(self, record: BaseModel) -> str:
        with self.path.open("a") as fh:
            fh.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
        return next(iter(record.model_dump().values()))  # the id, every record's first field

    def all(self) -> list:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(self.model.model_validate_json(line))
        return out


class Spool(_Jsonl):
    def __init__(self, root: Path):
        super().__init__(Path(root) / SPOOL_FILENAME, SpoolEvent)

    def raise_event(self, kind: EventKind, subject: str, plan_id: str = "",
                    **detail) -> SpoolEvent:
        event = SpoolEvent(kind=kind, subject=subject, plan_id=plan_id, detail=detail)
        self.append(event)
        return event

    def already(self, kind: str, subject: str, **match) -> bool:
        """Whether an event of this kind about this subject, with these detail values, is
        in the spool — so a tick raises a sentinel or a deadline for a job once."""
        return any(ev.kind == kind and ev.subject == subject
                   and all(ev.detail.get(k) == v for k, v in match.items())
                   for ev in self.all())


class Journal(_Jsonl):
    def __init__(self, root: Path):
        super().__init__(Path(root) / JOURNAL_FILENAME, JournalEntry)

    def write(self, kind: JournalKind, by: Author, text: str, cites: list[str] | None = None,
              subject: str = "", status: str = "open") -> JournalEntry:
        entry = JournalEntry(kind=kind, by=by, text=text, cites=list(cites or []),
                             subject=subject, status=status)
        self.append(entry)
        return entry

    def close(self, entry_id: str, by: Author, text: str = "") -> JournalEntry:
        entry = JournalEntry(kind="decision", by=by, text=text or f"closes {entry_id}",
                             status="closed", closes=entry_id)
        self.append(entry)
        return entry

    def open_entries(self) -> list[JournalEntry]:
        closed = {e.closes for e in self.all() if e.closes}
        return [e for e in self.all()
                if e.status == "open" and e.closes is None and e.entry_id not in closed]


class Inbox(_Jsonl):
    def __init__(self, root: Path):
        super().__init__(Path(root) / INBOX_FILENAME, InboxEntry)

    def ask(self, subject: str, question: str, evidence: list[str], options: list[Option],
            cites: list[str] | None = None) -> InboxEntry:
        entry = InboxEntry(subject=subject, question=question, evidence=evidence,
                           options=options, cites=list(cites or []))
        self.append(entry)
        return entry

    def notify(self, subject: str, text: str, evidence: list[str] | None = None,
               cites: list[str] | None = None) -> InboxEntry:
        entry = InboxEntry(kind="notice", subject=subject, question=text,
                           evidence=list(evidence or []), cites=list(cites or []))
        self.append(entry)
        return entry

    def answer(self, entry_id: str, option: str, by: str) -> InboxEntry:
        """Record an answer. The option must be one the question offered."""
        question = self.get(entry_id)
        if question is None or question.kind != "question":
            raise KeyError(f"no open question {entry_id} in the inbox")
        if option not in {o.id for o in question.options}:
            raise ValueError(f"{option!r} is not an option of {entry_id}: "
                             + ", ".join(o.id for o in question.options))
        if self.answered(entry_id) is not None:
            raise ValueError(f"{entry_id} is already answered")
        line = InboxEntry(subject=question.subject, question="", answers=entry_id,
                          option=option, by=by)
        self.append(line)
        return line

    def get(self, entry_id: str) -> InboxEntry | None:
        return next((e for e in self.all() if e.entry_id == entry_id and e.answers is None),
                    None)

    def answered(self, entry_id: str) -> InboxEntry | None:
        return next((e for e in self.all() if e.answers == entry_id), None)

    def open_questions(self) -> list[InboxEntry]:
        answered = {e.answers for e in self.all() if e.answers}
        return [e for e in self.all()
                if e.kind == "question" and e.answers is None and e.entry_id not in answered]

    def notices(self) -> list[InboxEntry]:
        return [e for e in self.all() if e.kind == "notice" and e.answers is None]


class InFlight(BaseModel):
    """The action a tick was carrying out when it last wrote: enough to find its outcome
    in the ledger and the job store if the tick died before recording it."""

    action: str  # "run" | "rerun" | "extend" | "continue" | "collect"
    step: str
    arm: str = ""
    tool: str
    started: str  # ISO 8601, UTC; ledger events and jobs from here on are its outcome
    event: str = ""  # the spool event it answers, if any


class LoopState(BaseModel):
    """What the loop knows that the study's other records do not say on their own."""

    handled: list[str] = Field(default_factory=list)  # spool event ids
    attempts: dict[str, int] = Field(default_factory=dict)  # step[arm] -> reruns so far
    extensions: dict[str, int] = Field(default_factory=dict)  # step[arm] -> segments added
    # step[arm] -> the ledger event of the call that ran it (the loop's own record of
    # which call is which arm; the plan alignment claims a step once)
    runs: dict[str, list[str]] = Field(default_factory=dict)
    dropped: list[str] = Field(default_factory=list)  # step[arm] dropped on a signature
    stopped: list[str] = Field(default_factory=list)  # branches a person or a wake ended
    deadline_from: dict[str, str] = Field(default_factory=dict)  # step[arm] -> ISO start
    budget_notified: bool = False
    refused: dict[str, list[str]] = Field(default_factory=dict)  # step[arm] -> finding ids
    until_closed: bool = False  # the chain's criterion held, or a person accepted the run
    in_flight: InFlight | None = None

    @classmethod
    def load(cls, state_dir: Path) -> LoopState:
        path = Path(state_dir) / LOOP_STATE_FILENAME
        if path.is_file():
            return cls.model_validate_json(path.read_text())
        return cls()

    def save(self, state_dir: Path) -> None:
        path = Path(state_dir) / LOOP_STATE_FILENAME
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2) + "\n")
        os.replace(tmp, path)
