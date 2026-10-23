"""Validation warnings and the acknowledge-by-id gate inputset.

The "lack of assumption" principle, in code:

- Checks emit ValidationWarning objects with STABLE ids.
- ``block`` findings always refuse the operation.
- ``warn`` findings refuse unless the caller acknowledges each one BY ID in the
  tool call itself (``acknowledge_warnings=["MDP_DT_LARGE", ...]``) — the
  acknowledgment then lands in the transcript and the provenance ledger.
- A parameter that would be silently defaulted is surfaced as a warning whose
  ``assumption`` field states exactly what would be assumed.
- A ``warn`` finding marked ``needs_human`` (spend above the approval threshold,
  a booking that cannot finish its run) can only be acknowledged where a human
  is in the loop. In an unattended session (the native agent loop) it refuses
  like a block: the agent's job is then to report it, not to accept it — left to
  itself, a model acknowledges its own budget warning in the next call.
- A finding a person accepted in a study's approval that still holds (core/approvals.py: a
  plan_approve whose plan is unchanged, a researcher's yes to one call, or a standing
  grant from `myrmidon approve`) is acknowledged by that approval: it was acknowledged by
  id once, by a person, and the call need not restate it. The ledger records it as
  acknowledged, with the approval's event id. The approval names the ids, so it covers
  exactly those and no others; anything else still refuses.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Severity = Literal["block", "warn", "info"]


class ValidationWarning(BaseModel):
    id: str  # stable, SCREAMING_SNAKE, e.g. "MDP_DT_LARGE_NO_CONSTRAINTS"
    severity: Severity
    message: str
    assumption: str | None = None  # what would be silently assumed, spelled out
    suggestion: str | None = None
    needs_human: bool = False  # acknowledgeable only with a human in the loop
    # parameters any one of which, given, resolves the finding: what a refusal's
    # NextAction names as missing (core/tool.py)
    needs: list[str] | None = None


class GateDecision(BaseModel):
    """Outcome of running a gate against a set of acknowledgments."""

    allowed: bool
    blocking: list[ValidationWarning]  # block-severity: never acknowledgeable
    unacknowledged: list[ValidationWarning]  # warn-severity lacking an ack
    acknowledged: list[str]  # warning ids the caller accepted
    advisory: list[ValidationWarning]  # info-severity, never refuse
    checked: list[str]  # names of checks that ran — no silent skips
    # warn-severity findings that need a human and had none (unattended session):
    # refused even if the caller listed them in acknowledge_warnings
    human_required: list[ValidationWarning] = []


def decide(
    warnings: list[ValidationWarning],
    acknowledge: list[str],
    checked: list[str],
    unattended: bool = False,
    approved: list[str] | None = None,
) -> GateDecision:
    approved = approved or []
    blocking = [w for w in warnings if w.severity == "block"]
    warns = [w for w in warnings if w.severity == "warn"]
    advisory = [w for w in warnings if w.severity == "info"]
    human_required = (
        [w for w in warns if w.needs_human and w.id not in approved] if unattended else []
    )
    warns = [w for w in warns if w not in human_required]
    # acknowledged in the call, or once by a person in a holding approval
    accepted = set(acknowledge) | set(approved)
    unacknowledged = [w for w in warns if w.id not in accepted]
    acknowledged = sorted({w.id for w in warns if w.id in accepted})
    return GateDecision(
        allowed=not blocking and not unacknowledged and not human_required,
        blocking=blocking,
        unacknowledged=unacknowledged,
        acknowledged=acknowledged,
        advisory=advisory,
        checked=checked,
        human_required=human_required,
    )
