"""Typed next actions: what a result makes reachable, where the harness knows it.

Generated only where the next step is certain — a refusal, an invalid call, no study, a
job still running, a file a registered parser reads — and never an approval the study has
not recorded. Without them a model loops: the same approved call refused again and again
without its acknowledgement, a blocking refusal repeated, a run directory listed over and
over with its CPMD output never parsed."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolRegistry, tool
from myrmidon.core.validation import ValidationWarning

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _gated(finding: ValidationWarning) -> ToolRegistry:
    reg = ToolRegistry()

    class P(BaseModel):
        hours: int = 24
        acknowledge_warnings: list[str] = []

    @tool("spend", description="submit", registry=reg, expensive=True,
          gate=lambda ctx, params: ([finding], ["budget"]))
    def spend(ctx, params: P) -> Observation:
        return Observation(ok=True, summary="ran")

    return reg


_BUDGET = ValidationWarning(id="COMPUTE_BUDGET", severity="warn", message="48 core-hours")


# (the finding needs a person, nobody at the keyboard, the action offered). A retry never
# carries an acknowledgement the study has not recorded; a person's finding asks for one.
OFFERS = [
    (False, False, "retry"),
    (True, False, "ask_user"),  # a person at an MCP client
    (True, True, "ask_user"),  # nobody
]


@pytest.mark.parametrize("needs_human,unattended,kind", OFFERS)
def test_a_refusal_never_acknowledges_for_the_model(ws_ctx, needs_human, unattended, kind):
    finding = _BUDGET.model_copy(update={"needs_human": needs_human})
    ws_ctx.unattended = unattended
    (action,) = _gated(finding).call("spend", ws_ctx, {"hours": 48}).next_actions
    assert action.kind == kind and action.finding_ids == ["COMPUTE_BUDGET"]
    assert "acknowledge_warnings" not in action.known_parameters
    assert action.needs_human is needs_human


def test_acknowledgements_are_recorded_once_each(ws_ctx):
    reg = _gated(_BUDGET)
    obs = reg.call("spend", ws_ctx, {"acknowledge_warnings": ["COMPUTE_BUDGET"] * 50})
    assert obs.ok
    event = list(ws_ctx.workspace.ledger.events())[-1]
    assert event.warnings_acknowledged == ["COMPUTE_BUDGET"]


def test_a_retrys_echoed_arguments_are_not_evidence(ws_ctx, tmp_path):
    """A refused call's numbers come back in its retry action; citing them is still
    citing the model's own guess."""
    from tests.unit.test_agent_loop import FakeMessage, FakeToolCall, scripted

    from myrmidon.agent.loop import AgentLoop

    reg = _gated(_BUDGET)  # not approved: refused, and the retry echoes the call
    loop = AgentLoop("fake", ws_ctx, registry=reg, transcript_dir=tmp_path / "t",
                     completion_fn=scripted(
                         FakeMessage(tool_calls=[FakeToolCall("spend", {"hours": 731})]),
                         FakeMessage(content="It would have used 731 hours."),
                         FakeMessage(content="It would have used 731 hours."),
                         FakeMessage(content="It would have used 731 hours.")))
    episode = loop.run("Spend.")
    assert "731" in episode.grounding.ungrounded

