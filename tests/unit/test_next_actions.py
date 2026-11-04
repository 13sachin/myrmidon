"""A refusal says what stopped the call and the one move that changes it.

A tool checks its own preconditions inline (`precheck`): a blocking finding stops the call
with the finding as the error; a finding that is a person's stops a model's call and not a
person's own; anything else rides on the result. The registry records a stopped call as
refused, never as run."""

import pytest
from pydantic import BaseModel

from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolRegistry, precheck, tool
from myrmidon.core.validation import Finding, Precondition, Theirs


def _checked(finding: Finding) -> ToolRegistry:
    reg = ToolRegistry()

    class P(BaseModel):
        hours: int = 24

    @tool("spend", description="submit", registry=reg)
    def spend(ctx, params: P) -> Observation:
        precheck(ctx, [finding])
        return Observation(ok=True, summary="ran")

    return reg


_FIELDS = {"id": "SOMETHING", "message": "it cannot run",
           "suggestion": "give it what it needs"}


# (the finding's severity, a person makes the call, the call runs)
CHECKS = [
    ("block", False, False),
    ("block", True, False),
    ("warn", False, False),
    ("warn", True, True),
    ("info", False, True),
]


@pytest.mark.parametrize("severity,person,runs", CHECKS)
def test_a_tools_own_check_stops_or_informs(ws_ctx, severity, person, runs):
    ws_ctx.person = "researcher" if person else None
    kind = {"block": Precondition, "warn": Theirs, "info": Finding}[severity]
    obs = _checked(kind(**_FIELDS)).call("spend", ws_ctx, {})
    assert obs.ok is runs
    assert [f.id for f in obs.findings] == ["SOMETHING"]
    event = list(ws_ctx.workspace.ledger.events())[-1]
    if runs:
        assert obs.error is None and event.kind == "call"
    else:
        assert obs.error.code == "SOMETHING" and event.kind == "gate_refused"
        assert obs.error.hint == ("give it what it needs" if severity == "block"
                                  else obs.error.hint)
