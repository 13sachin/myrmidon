"""Append-only provenance ledger: the trail every study leaves behind.

One JSONL file per workspace. Every tool call becomes a ProvenanceEvent with
the exact validated params, the exact argv (when a binary ran), engine
versions, and hashed input/output artifacts: the record of what ran, read by the
report, the summary and anyone checking a number.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef

LEDGER_FILENAME = "myrmidon.ledger.jsonl"

#: the digest of the approved mandate a submitting call is made under ("" for none), set by
#: the registry, read by the ledger event and the job store, so a mandate counts its own jobs
CALL_MANDATE: ContextVar[str] = ContextVar("CALL_MANDATE", default="")
#: the findings a running tool's own checks noticed (core/tool.py ``precheck``), gathered by
#: the registry onto the call's result; None outside a registry call
CALL_FINDINGS: ContextVar[list | None] = ContextVar("CALL_FINDINGS", default=None)
#: the event id the running call will be recorded under, minted when the call starts, so a
#: job submitted during the call can name the event that submitted it
CALL_EVENT: ContextVar[str] = ContextVar("CALL_EVENT", default="")
#: the artifacts the call the registry is running consumes, hashed before the tool ran:
#: what a job it submits must still read on the node (execution/receipt.py)
CALL_INPUTS: ContextVar[list | None] = ContextVar("CALL_INPUTS", default=None)


def new_event_id() -> str:
    return "ev-" + uuid.uuid4().hex[:12]


class ProvenanceEvent(BaseModel):
    event_id: str = Field(default_factory=new_event_id)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tool: str
    params: dict = Field(default_factory=dict)
    command: list[str] | None = None
    engine_versions: dict[str, str] = Field(default_factory=dict)
    myrmidon_version: str = ""
    inputs: list[ArtifactRef] = Field(default_factory=list)
    outputs: list[ArtifactRef] = Field(default_factory=list)
    # warn-severity finding ids the call ran under, accepted by the person making it
    warnings_acknowledged: list[str] = Field(default_factory=list)
    ok: bool = True
    duration_s: float | None = None
    exit_code: int | None = None
    # the last lines of what a failed call's binary wrote, as the observation carried them:
    # the ledger is the one record of a call that ran in the session and left no job
    log_tail: str | None = None
    kind: str = "call"  # "call" | "observe" | "gate_refused"
    # ids of the findings the call surfaced (gate, advisory, diagnosis): the durable
    # knowledge in play, by the name the code gives it
    knowledge_refs: list[str] = Field(default_factory=list)
    # Hash of the tool's params model JSON schema at the time of the call, so a reader
    # of an old ledger can tell which tools have changed what they take since
    params_schema: str = ""
    # the digest of the approved mandate the call was made under (core/mandate.py): set on
    # a submission and on the approval itself
    mandate: str = ""


class ProvenanceLedger:
    def __init__(self, path: Path):
        self.path = path

    def append(self, event: ProvenanceEvent) -> str:
        with self.path.open("a") as fh:
            fh.write(json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n")
        return event.event_id

    def events(self) -> Iterator[ProvenanceEvent]:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield ProvenanceEvent.model_validate_json(line)

    def get(self, event_id: str) -> ProvenanceEvent | None:
        for ev in self.events():
            if ev.event_id == event_id:
                return ev
        return None

    def last_output(self, path: str) -> ArtifactRef | None:
        """The most recent record of ``path`` (workspace-relative) as a call's output: what
        the file hashed to when the harness produced it. None when no call produced it."""
        found = None
        for ev in self.events():
            for art in ev.outputs:
                if art.path == path:
                    found = art
        return found


def shquote(arg: str) -> str:
    """Quote one argument for a POSIX shell: the job backends write scripts with it."""
    if arg and all(c.isalnum() or c in "@%+=:,./-_" for c in arg):
        return arg
    return "'" + arg.replace("'", "'\\''") + "'"
