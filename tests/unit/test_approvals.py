"""One approval story: a plan_approve in the ledger, read by the gate while it holds."""

import pytest

from myrmidon.core.approvals import approved_by, grant, holding
from myrmidon.core.validation import ValidationWarning


def _budget_tool():
    """A gated call whose only finding is a budget that needs a human."""
    from pydantic import BaseModel

    from myrmidon.core.observation import Observation
    from myrmidon.core.tool import ToolRegistry, tool

    reg = ToolRegistry()

    class P(BaseModel):
        acknowledge_warnings: list[str] = []

    def gate(ctx, params):
        return [ValidationWarning(id="COMPUTE_BUDGET", severity="warn", message="48 core-h",
                                  needs_human=True)], ["budget"]

    @tool("spend", description="x", returns=Observation, registry=reg, expensive=True,
          gate=gate)
    def spend(ctx, params: P) -> Observation:
        return Observation(ok=True, summary="submitted")

    return reg


# (nobody at the keyboard, a person's standing approval, ids the call restates, the call
# runs). The agent may acknowledge a finding that needs a person only when a person is there
# or has approved it; an approval acknowledges by itself (the call need not restate the
# id), and the ledger names the approval the acknowledgement relied on.
APPROVALS = [
    (False, False, ["COMPUTE_BUDGET"], True),
    (True, False, ["COMPUTE_BUDGET"], False),
    (True, True, ["COMPUTE_BUDGET"], True),
    (True, True, [], True),
]


@pytest.mark.parametrize("unattended,approved,acks,runs", APPROVALS)
def test_a_finding_for_a_person_is_accepted_only_by_a_person(ws_ctx, unattended, approved,
                                                              acks, runs):
    reg = _budget_tool()
    ws_ctx.unattended = unattended
    approval = grant(ws_ctx.workspace, ["COMPUTE_BUDGET"], approver="pi") if approved else ""
    obs = reg.call("spend", ws_ctx, {"acknowledge_warnings": acks})
    assert obs.ok is runs
    event = list(ws_ctx.workspace.ledger.events())[-1]
    if runs:
        assert event.warnings_acknowledged == ["COMPUTE_BUDGET"]
        assert event.approval == approval
    else:
        assert [w.id for w in obs.human_required] == ["COMPUTE_BUDGET"]


def test_a_plan_approval_lapses_when_the_plan_changes(ws_ctx):
    import myrmidon.basic_tools as basic

    basic.register_all()
    ws = ws_ctx.workspace
    plan = ws.root / "plan.yaml"
    plan.write_text("question: q\\n")
    from myrmidon.core.artifacts import ArtifactRef
    from myrmidon.core.provenance import ProvenanceEvent

    ws.ledger.append(ProvenanceEvent(
        tool="plan_approve", ok=True, warnings_acknowledged=["COMPUTE_BUDGET"],
        inputs=[ArtifactRef.from_path(plan, "plan", root=ws.root)]))
    assert list(approved_by(ws)) == ["COMPUTE_BUDGET"]
    plan.write_text("question: something else\\n")
    assert holding(ws) == [] and approved_by(ws) == {}


def _approve(*args):
    from typer.testing import CliRunner

    from myrmidon.cli import app

    return CliRunner().invoke(app, ["approve", *args])


def test_a_person_approves_from_the_command_line_and_an_unattended_call_relies_on_it(ws_ctx):
    reg = _budget_tool()
    ws_ctx.unattended = True
    ack = {"acknowledge_warnings": ["COMPUTE_BUDGET"]}
    assert not reg.call("spend", ws_ctx, ack).ok

    study = str(ws_ctx.workspace.root)
    out = _approve("COMPUTE_BUDGET", "--study", study, "--approver", "pi")
    assert out.exit_code == 0, out.output
    (approval,) = holding(ws_ctx.workspace)
    assert (approval.findings, approval.approver, approval.call) == (
        ("COMPUTE_BUDGET",), "pi", None)

    assert reg.call("spend", ws_ctx, ack).ok
    assert list(ws_ctx.workspace.ledger.events())[-1].approval == approval.event_id

