"""The study's journal and inbox. Both live in the study, appended and never rewritten,
so no process holds memory the study lacks.

- ``journal.jsonl`` — decisions, ruled-out causes, suspicions and intentions, each citing
  ledger events, written by machinery, a model or a person; open entries are what a wake
  and the report read. Numbers stay in the ledger; the journal holds *why*.
- ``inbox.jsonl`` — questions parked for the researcher with the evidence and the options,
  and the answers, by option id (``myrmidon answer``); the next tick carries an answer out.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

JOURNAL_FILENAME = "journal.jsonl"
INBOX_FILENAME = "inbox.jsonl"

JournalKind = Literal["decision", "ruled_out", "suspicion", "intention"]
Author = Literal["machinery", "model", "human"]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JournalEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: "jn-" + uuid.uuid4().hex[:12])
    ts: str = Field(default_factory=now_iso)
    kind: JournalKind
    by: Author
    text: str
    cites: list[str] = Field(default_factory=list)  # ledger event ids
    status: Literal["open", "closed"] = "open"
    subject: str = ""  # the stage it is about, when it is about one
    # a line that closes an earlier entry names it; the entry itself is never rewritten
    closes: str | None = None


class Option(BaseModel):
    id: str  # what a person types: "continue", "stop", "revise"
    text: str


class InboxEntry(BaseModel):
    """A parked question, or a notice (no options; nothing waits on it)."""

    entry_id: str = Field(default_factory=lambda: "in-" + uuid.uuid4().hex[:12])
    ts: str = Field(default_factory=now_iso)
    kind: Literal["question", "notice"] = "question"
    subject: str  # what it is about: "study", or a stage
    question: str
    evidence: list[str] = Field(default_factory=list)  # one line each; ids at the end
    options: list[Option] = Field(default_factory=list)
    cites: list[str] = Field(default_factory=list)  # ledger event ids
    # an answer line names the entry it answers and nothing else below is set
    answers: str | None = None
    option: str | None = None
    by: str | None = None  # a person's name


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
