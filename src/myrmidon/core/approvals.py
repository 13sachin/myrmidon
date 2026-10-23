"""Human authority, as the ledger records it: one kind of approval, read one way.

An approval is a successful ``plan_approve`` event. The findings it acknowledged are the
ones a person accepted for this study — typically a budget above the machine's threshold —
and an unattended session (nobody at the keyboard) may acknowledge exactly those, and no
others, while the approval holds:

- an approval of a plan holds while ``plan.yaml`` is the file it approved (same sha256);
  edit the plan and it lapses;
- an approval with no plan attached is a standing grant for the study: a person accepting
  a cost before an unattended session starts (``myrmidon approve FINDING``), which holds
  before any plan exists;
- an approval scoped to one call (``call``: the call's identity, ``call_identity``) is a
  researcher's yes at the keyboard to exactly that call — the same tool with the same
  parameters — and holds once: the call that relies on it uses it up. A finding id names
  a kind of risk, not its size; a standing yes to ``COMPUTE_BUDGET`` for a 24 core-hour
  run would also have accepted a 2000 core-hour one.

A call that relied on an approval names its event (``ProvenanceEvent.approval``); an
acknowledgement under none is the agent's own, and the summary says so.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from myrmidon.core.artifacts import sha256_file
from myrmidon.core.provenance import ProvenanceEvent

PLAN_FILE = "plan.yaml"
APPROVE_TOOL = "plan_approve"


@dataclass(frozen=True)
class Approval:
    event_id: str
    findings: tuple[str, ...]
    approver: str  # "session" (through plan_approve), or the name `myrmidon approve` gave
    plan_sha256: str | None  # None: a standing grant, not tied to a plan
    call: str | None = None  # a call's identity: this approval accepts that call, once


def call_identity(tool: str, params: dict) -> str:
    """The identity of a call: its tool and validated parameters, acknowledgements aside
    (adding the acknowledgement is what the approval is for)."""
    body = {k: v for k, v in params.items() if k != "acknowledge_warnings"}
    blob = json.dumps({"tool": tool, "params": body}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def approvals(ws) -> list[Approval]:
    """Every approval recorded in the study, oldest first, whether or not it holds."""
    out = []
    for ev in ws.ledger.events():
        if ev.tool != APPROVE_TOOL or not ev.ok or ev.kind != "call":
            continue
        plan = next((ref for ref in ev.inputs if ref.role == "plan"), None)
        out.append(Approval(
            event_id=ev.event_id,
            findings=tuple(ev.warnings_acknowledged),
            approver=str(ev.params.get("approver") or "session"),
            plan_sha256=plan.sha256 if plan is not None else None,
            call=ev.params.get("call") or None,
        ))
    return out


def holding(ws) -> list[Approval]:
    """The approvals that still hold: standing grants, and plan approvals whose plan is
    unchanged."""
    if ws is None:
        return []
    plan = ws.root / PLAN_FILE
    current = sha256_file(plan) if plan.is_file() else None
    # a call-scoped approval is used up by the call that relied on it
    used = {ev.approval for ev in ws.ledger.events() if ev.approval and ev.ok}
    return [a for a in approvals(ws)
            if (a.plan_sha256 is None or a.plan_sha256 == current)
            and not (a.call and a.event_id in used)]


def approved_by(ws, call: str | None = None) -> dict[str, str]:
    """finding id -> the event id of the latest holding approval that accepted it, for a
    call with identity ``call``: standing and plan approvals hold for any call, a scoped
    one only for its own."""
    out: dict[str, str] = {}
    for approval in holding(ws):
        if approval.call is not None and approval.call != call:
            continue
        for finding in approval.findings:
            out[finding] = approval.event_id
    return out


def grant(ws, findings: list[str], approver: str, call: str | None = None) -> str:
    """Record an approval of ``findings`` by ``approver``: standing for the study
    (``myrmidon approve``), or, with ``call``, for that one call. Written as the same
    event a person's plan_approve leaves, so every reader treats it the same way."""
    import myrmidon

    params = {"approver": approver, **({"call": call} if call else {})}
    event = ProvenanceEvent(
        tool=APPROVE_TOOL, params=params, warnings_acknowledged=list(findings),
        ok=True, kind="call", myrmidon_version=myrmidon.__version__,
    )
    return ws.ledger.append(event)
