"""Episode transcripts: every message, tool call, result, and usage record.

One JSONL file per episode, keyed to the workspace ledger via provenance ids. The format
is documented, with every event and field, in ``docs/FORMATS.md``.

Every request to the model is in it verbatim (``model_call``): the messages added since
the previous request, the hash of the tool schemas offered (each set written once, as
``tool_schemas``), the settings, and the reply — reasoning included — or the error.
:func:`model_calls` rebuilds exactly what each request carried.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path


class Transcript:
    def __init__(self, directory: Path, episode_id: str | None = None):
        directory.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id or "ep-" + uuid.uuid4().hex[:8]
        self.path = directory / f"{self.episode_id}.jsonl"
        # sees each record as it is written (a terminal rendering the session)
        self.listener = None

    def emit(self, kind: str, **payload) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": kind,
            **payload,
        }
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, default=str, sort_keys=True) + "\n")
        if self.listener is not None:
            self.listener(record)

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


#: the transcript format (docs/FORMATS.md), stated by the `trace` event that opens every
#: session. Adding an optional field keeps it; removing, renaming or changing the meaning of
#: an event or a field moves it.
TRANSCRIPT_FORMAT = 1


class TraceError(ValueError):
    """A transcript whose trace does not add up to the history it claims."""


def digest(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, default=str)
                          .encode()).hexdigest()


def _replay(events: list[dict], check: bool):
    """(history, calls): the message history the trace implies after each event, and one
    entry per ``model_call``."""
    schemas: dict[str, list] = {}
    history: list[dict] = []
    calls: list[dict] = []
    for ev in events:
        kind = ev.get("kind")
        if kind == "tool_schemas":
            schemas[ev["sha256"]] = ev["schemas"]
        elif kind == "resumed":
            history = list(ev["replace"]) if "replace" in ev else history + ev["append"]
            if check and digest(history) != ev["sha256"]:
                raise TraceError(f"the history restored at call {len(calls) + 1} is not "
                                 "the one the trace records")
        elif kind == "model_call":
            history = history + ev["append"]
            call = {"model": ev["model"], "messages": history,
                    "tools": schemas.get(ev["tools"]), "settings": ev["settings"]}
            if check and call["tools"] is None:
                raise TraceError(f"call {ev['call']} names schemas {ev['tools'][:12]} the "
                                 "transcript never wrote")
            call["response" if "response" in ev else "error"] = ev.get("response",
                                                                         ev.get("error"))
            calls.append(call)
    return history, calls


def model_calls(events: list[dict]) -> list[dict]:
    """Each request as it was made: ``model``, ``messages``, ``tools``, ``settings``, and the
    ``response`` (or ``error``) it got. Raises :class:`TraceError` where the trace does not
    add up."""
    return _replay(events, check=True)[1]


def trace_history(events: list[dict], check: bool = True) -> list[dict]:
    """The message history as far as the trace has recorded it."""
    return _replay(events, check)[0]
