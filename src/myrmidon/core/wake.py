"""The wake: one typed model call that answers a prepared question.

The loop never gives a model a free turn. When the ladder runs out of envelope it builds
the question a person would get — what happened, the parsed evidence, the options and
their costs — and, when a wake model is configured, puts that question to the model first.
The context is a pure function of the study and the question (``compose``): the compact
plan with each step's state, open journal entries, the question's evidence and options,
nothing else. It is hashed and saved with the reply under ``.myrmidon/wakes/``, so a bad
decision is an experiment anyone can repeat through another endpoint.

The reply is one option id and a reason. ``park`` sends the question to the inbox for a
person; anything else is carried out exactly as a person's answer would be, and the ledger,
the journal and the inbox say it was the model's.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.spool import InboxEntry

DECIDE_TOOL = "decide"
SYSTEM = (
    "You are the wake of a simulation study loop. A move has stopped and the loop cannot "
    "settle it from its approved envelope. You are shown the plan, its state, the open "
    "journal and the evidence. Choose exactly one of the offered options by its id, and "
    "give the reason in one or two sentences citing the evidence. Choose `park` when the "
    "evidence does not settle it: a person decides then. You cannot spend what the options "
    "do not offer, and you never invent a number."
)


class WakeDecision(BaseModel):
    option: str = Field(description="the id of one offered option, or 'park'")
    reason: str = Field(description="one or two sentences citing the evidence")


def compose(t, entry: InboxEntry) -> tuple[str, str]:
    """(the context, its sha256) for a question, from the study alone."""
    plan = t.plan
    lines = [f"question: {plan.question}", f"plan {plan.plan_id}, shape {plan.shape}",
             f"envelope: {plan.envelope.model_dump()}", ""]
    if plan.until:
        lines.append(f"until: {plan.until.criterion} (ceiling {plan.until.ceiling}, "
                     f"check {plan.until.check} {plan.until.values})")
    if plan.arms:
        lines.append(f"arms: {plan.arms.varies} over {plan.arms.over} — {plan.arms.because}")
    lines.append("steps:")
    for step in plan.steps:
        for arm in t.arms_of(step):
            key = t.key(step, arm)
            lines.append(f"  {key}: {step.tool} — {t.status(key)}"
                         + (f", {step.estimate_core_h:g} core-h" if step.estimate_core_h
                            else ""))
    open_entries = t.journal.open_entries()
    if open_entries:
        lines += ["", "open journal:"]
        lines += [f"  [{e.kind}, {e.by}] {e.text}" for e in open_entries]
    lines += ["", f"branch: {entry.subject}", f"what stopped it: {entry.question}", "evidence:"]
    lines += [f"  - {line}" for line in entry.evidence if line]
    lines += ["", "options:"]
    lines += [f"  {o.id}: {o.text}" + (f" ({o.cost_core_h:g} core-h)" if o.cost_core_h else "")
              for o in entry.options]
    lines.append("  park: leave it to a person")
    text = "\n".join(lines)
    return text, hashlib.sha256(text.encode()).hexdigest()


def save_wake(directory: Path, entry: InboxEntry, context: str, digest: str, reply) -> Path:
    """Every wake on disk: the question, the exact context, its hash, and the reply."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{entry.entry_id}-{uuid.uuid4().hex[:6]}.json"
    path.write_text(json.dumps({
        "inbox": entry.entry_id, "subject": entry.subject, "context_sha256": digest,
        "context": context,
        "reply": reply.model_dump() if reply is not None else None,
    }, indent=2) + "\n")
    return path


def decide_schema() -> dict:
    return {"type": "function", "function": {
        "name": DECIDE_TOOL, "description": "your one choice",
        "parameters": WakeDecision.model_json_schema()}}


class ModelWake:
    """A wake backed by a completion function with the native loop's calling convention
    (`agent/pydantic_adapter.PydanticCompletion`): one request, one required tool call."""

    def __init__(self, completion_fn, model: str, connection: dict | None = None,
                 name: str = ""):
        self.completion_fn = completion_fn
        self.model = model
        self.connection = dict(connection or {})
        self.name = name or model

    def __call__(self, request):
        from myrmidon.core.loop import WakeReply

        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": request.context}]
        response = self.completion_fn(model=self.model, messages=messages,
                                      tools=[decide_schema()], tool_choice="required",
                                      **self.connection)
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            raise ValueError("the wake model replied with no decision")
        args = calls[0].function.arguments
        decision = WakeDecision.model_validate(json.loads(args) if isinstance(args, str)
                                               else args)
        return WakeReply(option=decision.option, reason=decision.reason, model=self.name)
