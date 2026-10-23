"""The native agent at a terminal: a researcher's decisions, multi-turn sessions, resume,
Ctrl-C, and the command. A scripted model in the OpenAI response shape; no network."""

import json

import pytest
from tests.unit.test_agent_loop import FakeMessage, FakeResponse, FakeToolCall

from myrmidon.agent.loop import AgentLoop
from myrmidon.core.approvals import approved_by, call_identity, grant
from myrmidon.core.validation import ValidationWarning


def _budget_registry(raise_interrupt: bool = False):
    """One gated tool whose finding needs a human, as a compute budget does."""
    from pydantic import BaseModel

    from myrmidon.core.observation import Observation
    from myrmidon.core.tool import ToolRegistry, tool

    reg = ToolRegistry()

    class P(BaseModel):
        hours: int = 24
        acknowledge_warnings: list[str] = []

    def gate(ctx, params):
        return [ValidationWarning(id="COMPUTE_BUDGET", severity="warn", needs_human=True,
                                  message=f"{params.hours} core-hours")], ["budget"]

    @tool("spend", description="submit a job", returns=Observation, registry=reg,
          expensive=True, gate=gate)
    def spend(ctx, params: P) -> Observation:
        if raise_interrupt:
            raise KeyboardInterrupt
        return Observation(ok=True, summary="submitted, cost 7.25 core-hours")

    return reg


def _model(*replies):
    """A completion function answering with ``replies`` in order, recording what it saw."""
    seen = []
    queue = list(replies)

    def fn(model, messages, tools, **kw):
        seen.append(json.loads(json.dumps(messages, default=str)))
        return FakeResponse(queue.pop(0))

    fn.seen = seen
    return fn


def _loop(ws_ctx, tmp_path, fn, registry, decide=None):
    return AgentLoop("fake", ws_ctx, registry=registry, completion_fn=fn,
                     transcript_dir=tmp_path / "t", on_decision=decide)


# --- approvals scoped to one call ---------------------------------------------------

def test_a_call_scoped_approval_holds_for_that_call_once(ws_ctx):
    ws = ws_ctx.workspace
    this = call_identity("spend", {"hours": 24})
    approval = grant(ws, ["COMPUTE_BUDGET"], approver="researcher", call=this)
    assert approved_by(ws, call=this) == {"COMPUTE_BUDGET": approval}
    assert approved_by(ws, call=call_identity("spend", {"hours": 2000})) == {}
    assert approved_by(ws) == {}  # not a standing grant
    # the acknowledgement it is for does not change the identity
    assert call_identity("spend", {"hours": 24, "acknowledge_warnings": ["X"]}) == this

    from myrmidon.core.provenance import ProvenanceEvent
    ws.ledger.append(ProvenanceEvent(tool="spend", ok=True, approval=approval))
    assert approved_by(ws, call=this) == {}  # used up


# --- the researcher at the keyboard -------------------------------------------------

# (the researcher's answer when asked — None: nobody there; the call runs). A yes is a
# call-scoped plan_approve in the ledger that the call then cites, and the model sees the
# result; a no, or nobody, leaves the refusal standing and nothing runs.
DECISIONS = [(True, True), (False, False), (None, False)]


@pytest.mark.parametrize("answer,runs", DECISIONS)
def test_only_the_researcher_decides_a_finding_that_needs_a_person(ws_ctx, tmp_path, answer,
                                                                   runs):
    asked = []
    decide = None if answer is None else (
        lambda tool, args, findings: asked.append((tool, [f.id for f in findings])) or answer)
    fn = _model(FakeMessage(tool_calls=[FakeToolCall("spend", {"hours": 24})]),
                FakeMessage(content="Done."))
    _loop(ws_ctx, tmp_path, fn, _budget_registry(), decide=decide).run("Run it.")
    assert asked == ([] if answer is None else [("spend", ["COMPUTE_BUDGET"])])
    events = list(ws_ctx.workspace.ledger.events())
    ran = [e for e in events if e.tool == "spend" and e.ok]
    assert len(ran) == int(runs)
    approvals = [e for e in events if e.tool == "plan_approve"]
    if runs:
        assert approvals[0].params["approver"] == "researcher" and approvals[0].params["call"]
        assert ran[0].approval == approvals[0].event_id
    else:
        assert approvals == []
    shown = fn.seen[1][-1]["content"]  # what the model was given back
    assert (json.loads(shown)["ok"] if shown.lstrip().startswith("{") else False) is runs


# --- a conversation -----------------------------------------------------------------

def test_a_resumed_session_keeps_its_history_and_what_it_may_cite(ws_ctx, tmp_path):
    from myrmidon.agent.session import ChatSession, load, new_state

    root = tmp_path / "sessions"
    first = _model(FakeMessage(tool_calls=[FakeToolCall("spend", {})]),
                   FakeMessage(content="Submitted."))
    loop = _loop(ws_ctx, tmp_path, first, _budget_registry(), decide=lambda *a: True)
    session = ChatSession.begin(loop, new_state("fake", "pydantic", None), root, "rules")
    session.ask("Run it.")
    state = load(root)
    assert state.asks == 1 and state.workspace == str(ws_ctx.workspace.root)

    # a new process: the same session, a fresh loop
    second = _model(FakeMessage(content="It cost 7.25 core-hours."))
    again = ChatSession.resume(_loop(ws_ctx, tmp_path, second, _budget_registry()),
                               state, root)
    episode = again.ask("What did it cost?")
    assert [m["role"] for m in second.seen[0]][:3] == ["system", "user", "assistant"]
    assert second.seen[0][-1]["content"] == "What did it cost?"
    assert episode.grounding.ok  # 7.25 came from a tool before the resume
    assert load(root).asks == 2


# --- the terminal -------------------------------------------------------------------

def _console():
    from rich.console import Console

    return Console(record=True, width=160, force_terminal=False)


_REFUSED = {"kind": "tool_result", "tool": "gmx_mdrun", "ok": False,
            "failure": "gate_refused", "observation_type": "JobObservation", "obs": {
                "summary": "gmx_mdrun refused by validation gate",
                "human_required": [{"id": "COMPUTE_BUDGET", "message": "48 core-hours"}]}}


def _study_events():
    """One question's records: a lookup, a refusal the agent acknowledges, the job it then
    submits, two polls (one unchanged), parsed evidence, a recovered bad call and a
    grounding bounce."""
    job = {"kind": "tool_result", "tool": "job_status", "ok": True,
           "observation_type": "JobObservation", "tags": ["jobs"], "provenance_id": "ev-3",
           "obs": {"summary": "job 847291 running · 42000 steps", "job_id": "847291",
                   "state": "running", "progress": {"step": 42000}}}
    return [
        {"kind": "tool_call", "tool": "knowledge_search", "args": {"query": "nvt"}},
        {"kind": "tool_result", "tool": "knowledge_search", "ok": True,
         "observation_type": "KnowledgeObservation", "tags": ["knowledge"],
         "obs": {"summary": "3 hits for nvt"}},
        {"kind": "tool_call", "tool": "gmx_mdrun", "args": {"tpr": "nvt.tpr"}},
        {"kind": "tool_result", "tool": "gmx_mdrun", "ok": False, "failure": "gate_refused",
         "observation_type": "JobObservation", "obs": {
             "summary": "refused", "unacknowledged": [
                 {"id": "MDP_NO_CONSTRAINTS", "message": "dt 2 fs without constraints"}]}},
        {"kind": "tool_call", "tool": "gmx_mdrun",
         "args": {"tpr": "nvt.tpr", "acknowledge_warnings": ["MDP_NO_CONSTRAINTS"]}},
        {"kind": "tool_result", "tool": "gmx_mdrun", "ok": True, "provenance_id": "ev-2",
         "observation_type": "JobObservation", "tags": ["gromacs", "run"],
         "obs": {"summary": "submitted NVT · job 847291 · 4 core-h", "job_id": "847291",
                 "state": "pending",
                 "artifacts": [{"role": "tpr", "path": "nvt.tpr"}]}},
        job, dict(job),
        {"kind": "tool_call", "tool": "nope", "args": {}, "error": "unknown-tool"},
        {"kind": "tool_result", "tool": "gmx_energy", "ok": True, "provenance_id": "ev-4",
         "observation_type": "EnergyObservation", "tags": ["analysis", "gromacs"],
         "obs": {"summary": "Temperature 300.2 K ± 1.1"}},
        {"kind": "grounding_bounce", "ungrounded": ["301"]},
        {"kind": "turn", "turn": 3, "prompt_tokens": 900, "completion_tokens": 40},
    ]


def test_hidden_records_stay_in_the_transcript_with_their_type(ws_ctx, tmp_path):
    from myrmidon.agent.chat import Renderer

    console = _console()
    fn = _model(FakeMessage(tool_calls=[FakeToolCall("spend", {"hours": 24})]),
                FakeMessage(content="Declined, so nothing ran."))
    loop = AgentLoop("fake", ws_ctx, registry=_budget_registry(), completion_fn=fn,
                     transcript_dir=tmp_path / "t", on_event=Renderer(console),
                     on_decision=lambda *a: False)
    loop.run("Run it.")
    out = console.export_text()
    assert "DECISION  declined by you: COMPUTE_BUDGET" in out
    assert "spend" not in out  # the call itself is the agent's working
    result = next(e for e in loop.transcript.events() if e["kind"] == "tool_result")
    assert result["observation_type"] == "ValidationReport" and result["tags"] == []
    assert any(e["kind"] == "tool_call" for e in loop.transcript.events())


def _patch_model(monkeypatch, *replies):
    from myrmidon.agent.pydantic_adapter import PydanticCompletion

    fn = _model(*replies)
    monkeypatch.setattr(PydanticCompletion, "__call__", lambda self, **kw: fn(**kw))
    return fn

