"""The loop's principles.

Each test is one promise of `myrmidon tick`, as rows where the promise has cases: nothing
is spent without a holding approval; a killed tick is reconciled, never repeated; a failed
run climbs the ladder — signature, one rerun, then a question; a sentinel is answered
inside the envelope and asked outside it; a deadline parks one branch and not another; an
answer, a person's or a wake model's, is carried out the same way.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest
from tests.unit.test_mimic import _full_mimic_setup

from myrmidon.core.loop import WakeReply, tick
from myrmidon.core.spool import Inbox, InFlight, Journal, LoopState, Spool
from myrmidon.core.tool import REGISTRY

RES = {"cpmd_ranks": 1, "gromacs_ranks": 1, "walltime_min": 10}


def _stage(ws_ctx, name, restart_from=None, sentinels=(), extra_method=None, inputset=None):
    """The three steps of one coupled stage, as the protocol writes them."""
    cites = inputset or f"inputset:mimic_cpmd_{name}@v2"
    return [
        {"name": f"{name}_method", "tool": "inputset_apply", "cites": cites,
         "produces": [f"cpmd_method_{name}"], "values": extra_method or {}},
        {"name": f"{name}_input", "tool": "mimic_prepare",
         "needs": ["cpmd_input_qm", f"cpmd_method_{name}"], "produces": [f"cpmd_input_{name}"],
         "values": {"cpmd_input": "cpmd_input_qm", "method": f"cpmd_method_{name}",
                    "total_charge": 0, "multiplicity": 1}},
        {"name": name, "tool": "mimic_run",
         "needs": ["tpr", f"cpmd_input_{name}"] + ([restart_from] if restart_from else []),
         "produces": [f"cpmd_restart_{name}"],
         "values": {"tpr": "tpr_path", "cpmd_input": f"cpmd_input_{name}", **RES,
                    **({"restart_from": restart_from} if restart_from else {})},
         "sentinels": list(sentinels)},
    ]


def _plan(ws_ctx, steps, approve=True, **plan):
    """A plan on a prepared QM/MM system (mock prepqm, confirmed charge), given its
    system input and tpr."""
    run_inp, tpr = _full_mimic_setup(ws_ctx)
    for step in steps:
        for k, v in list(step.get("values", {}).items()):
            if v == "cpmd_input_qm" and k == "cpmd_input" and step["tool"] == "mimic_prepare":
                step["values"][k] = run_inp
            if v == "tpr_path":
                step["values"][k] = tpr
        step["needs"] = [n for n in step.get("needs", []) if n != "cpmd_input_qm"] + \
            (["cpmd_input_qm"] if "cpmd_input_qm" in step.get("needs", []) else [])
    given = ["tpr", "cpmd_input_qm"]
    obs = REGISTRY.call("plan_write", ws_ctx, {"question": "the loop's own test", "given": given,
                                               "steps": steps, **plan})
    assert obs.ok, obs.summary
    assert not obs.unresolved, obs.unresolved
    if approve:
        ok = REGISTRY.call("plan_approve", ws_ctx, {"acknowledge_warnings": ["COMPUTE_BUDGET"]})
        assert ok.ok, ok.summary
    return obs


def _ticks(ctx, n=30, wake=None, pause=0.3):
    report = None
    for _ in range(n):
        report = tick(ctx, wake=wake)
        if report.done or report.blocked or (report.idle and not report.running):
            break
        time.sleep(pause)
    return report


ANNEAL = {"check": "final_temperature_below", "args": {"limit_k": 5.0}}


def test_nothing_is_spent_without_a_holding_approval(ws_ctx):
    """The loop runs an approved plan and nothing else. Editing the plan lapses the
    approval, and the next tick submits nothing."""
    _plan(ws_ctx, _stage(ws_ctx, "annealing", sentinels=[ANNEAL]), approve=False)
    report = tick(ws_ctx)
    assert report.blocked and "not approved" in report.blocked
    assert not any(ev.tool == "mimic_run" for ev in ws_ctx.workspace.ledger.events())
    assert REGISTRY.call("plan_approve", ws_ctx, {"acknowledge_warnings": ["COMPUTE_BUDGET"]}).ok
    path = ws_ctx.workspace.root / "plan.yaml"
    path.write_text(path.read_text().replace("walltime_min: 10", "walltime_min: 11"))
    assert "not approved" in tick(ws_ctx).blocked


def test_a_second_tick_finds_the_study_locked_and_does_nothing(ws_ctx):
    import fcntl

    _plan(ws_ctx, _stage(ws_ctx, "annealing"))
    lock = open(ws_ctx.workspace.state_dir / "loop.lock", "a+")  # noqa: SIM115
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        report = tick(ws_ctx)
    finally:
        lock.close()
    assert "another tick" in report.blocked and not report.submitted


def test_a_tick_killed_after_its_call_is_reconciled_and_never_resubmits(ws_ctx):
    """The action in flight is written before the call; a tick that died after it
    finds the call's ledger event and the job it submitted, and adopts them."""
    _plan(ws_ctx, _stage(ws_ctx, "annealing", sentinels=[ANNEAL]))
    first = tick(ws_ctx)
    assert any(s.startswith("annealing (run) job") for s in first.submitted)
    ws = ws_ctx.workspace
    state = LoopState.load(ws.state_dir)
    event = state.runs.pop("annealing")[0]
    started = (datetime.now(UTC) - timedelta(minutes=5)).isoformat(timespec="seconds")
    state.in_flight = InFlight(action="run", step="annealing", tool="mimic_run", started=started)
    state.save(ws.state_dir)
    second = tick(ws_ctx)
    assert any("recovered event " + event in m for m in second.messages), second.messages
    report = _ticks(ws_ctx)
    assert sum(1 for ev in ws.ledger.events() if ev.tool == "mimic_run" and ev.ok) == 1
    assert report.done


# a failed run climbs the ladder: (the mock's fault, reruns made, signature, parked?)
LADDER = [
    ("node_once", 1, None, False),  # a machine fault: the rerun does not reproduce it
    ("shake", 0, "CPMD_SHAKE_WRONG_GEOMETRY", True),  # a known signature: no rerun, a question
]


@pytest.mark.parametrize("fault,reruns,signature,parked", LADDER)
def test_a_failed_run_climbs_the_ladder(ws_ctx, monkeypatch, fault, reruns, signature, parked):
    _plan(ws_ctx, _stage(ws_ctx, "annealing", sentinels=[ANNEAL]))
    monkeypatch.setenv("MOCK_CPMD_FAIL", fault)
    report = _ticks(ws_ctx)
    ws = ws_ctx.workspace
    state = LoopState.load(ws.state_dir)
    assert state.attempts.get("annealing", 0) == reruns
    failed = [ev for ev in Spool(ws.root).all() if ev.kind == "job" and not ev.detail.get("ok")]
    assert failed and failed[0].detail.get("diagnosis_id") == signature
    questions = Inbox(ws.root).open_questions()
    assert bool(questions) is parked
    if parked:
        assert {o.id for o in questions[0].options} == {"rerun", "stop"}
        assert signature in questions[0].question
    else:
        assert report.done
        suspicion = [e for e in Journal(ws.root).all() if e.kind == "suspicion"]
        assert suspicion and suspicion[0].by == "machinery"


# a sentinel that fails: inside the envelope one more segment, outside it a question
HEAT = {"check": "t_qm_tail_in_range", "args": {"target_k": 300, "tolerance_k": 10}}
RAMP = {"&CPMD/TEMPERATURE RAMP": "${annealing.final_temperature_k} 300 0.1",
        "&CPMD/BERENDSEN IONS": "${annealing.final_temperature_k} 5000"}


def _heating_plan(ws_ctx, extend_segments):
    steps = _stage(ws_ctx, "annealing", sentinels=[ANNEAL]) + _stage(
        ws_ctx, "heating", restart_from="cpmd_restart_annealing", sentinels=[HEAT],
        extra_method=RAMP, inputset="inputset:mimic_cpmd_heating@v3")
    return _plan(ws_ctx, steps, envelope={"extend_segments": extend_segments})


def test_a_sentinel_inside_the_envelope_extends_the_stage_without_a_model(ws_ctx):
    _heating_plan(ws_ctx, extend_segments=1)
    report = _ticks(ws_ctx, n=40)
    assert report.done, report.messages
    ws = ws_ctx.workspace
    assert LoopState.load(ws.state_dir).extensions == {"heating": 1}
    held = [ev for ev in Spool(ws.root).all() if ev.kind == "sentinel"]
    assert len(held) == 1 and not held[0].detail["holds"]
    assert not Inbox(ws.root).open_questions()


def test_a_sentinel_outside_the_envelope_parks_with_options_and_an_answer_resumes(ws_ctx):
    _heating_plan(ws_ctx, extend_segments=0)
    report = _ticks(ws_ctx, n=40)
    ws = ws_ctx.workspace
    questions = Inbox(ws.root).open_questions()
    assert len(questions) == 1 and questions[0].subject == "heating"
    assert {o.id for o in questions[0].options} == {"extend", "continue", "stop"}
    assert not report.done
    # the person answers; the next tick carries the answer out as the envelope would have
    Inbox(ws.root).answer(questions[0].entry_id, "extend", by="tester")
    Spool(ws.root).raise_event("human", "heating", plan_id=report.plan_id,
                               inbox=questions[0].entry_id, option="extend", action="extend",
                               args={}, by="tester")
    report = _ticks(ws_ctx, n=40)
    assert report.done, report.messages
    assert LoopState.load(ws.state_dir).extensions == {"heating": 1}


def test_a_wake_model_answers_the_question_first_and_the_ledger_says_so(ws_ctx, tmp_path):
    """A wake is the same prepared question, answered by a typed model call; `park` sends
    it to the inbox. The context is saved with its hash, so the decision replays."""
    _heating_plan(ws_ctx, extend_segments=0)
    seen = []

    def wake(request):
        seen.append(request)
        return WakeReply(option="extend", reason="the ramp did not reach 300 K", model="test")

    report = _ticks(ws_ctx, n=40, wake=wake)
    assert report.done, report.messages
    assert len(seen) == 1 and "heating" in seen[0].context and seen[0].context_hash
    ws = ws_ctx.workspace
    answer = next(e for e in Inbox(ws.root).all() if e.answers)
    assert answer.by == "model:test" and answer.option == "extend"
    assert not Inbox(ws.root).open_questions()
    decision = [e for e in Journal(ws.root).all() if e.by == "model"]
    assert decision and "ramp" in decision[0].text
    saved = list((ws.state_dir / "wakes").glob("*.json"))
    assert len(saved) == 1 and seen[0].context_hash in saved[0].read_text()


def test_a_wake_that_parks_leaves_the_question_to_a_person(ws_ctx):
    _heating_plan(ws_ctx, extend_segments=0)
    report = _ticks(ws_ctx, n=40, wake=lambda request: WakeReply(option="park", reason="unsure"))
    assert not report.done
    questions = Inbox(ws_ctx.workspace.root).open_questions()
    assert len(questions) == 1 and questions[0].subject == "heating"


def test_a_deadline_parks_its_branch_and_waiting_resets_it(ws_ctx):
    """A job past its step's expected duration is an event; the envelope parks the branch
    with wait or cancel, and `wait` starts the clock again."""
    from myrmidon.engines import mock

    steps = _stage(ws_ctx, "annealing")
    steps[-1]["deadline_min"] = 30
    _plan(ws_ctx, steps)
    ws = ws_ctx.workspace
    with mock.activated(ws.root):
        import os

        os.environ["MOCK_SLURM_HOLD"] = "1"  # the mock scheduler keeps the job pending
        ctx = mock.mock_context(ws_ctx)
        first = tick(ctx)
        assert first.running, first.messages
        state = LoopState.load(ws.state_dir)
        state.deadline_from["annealing"] = (datetime.now(UTC) - timedelta(hours=2)).isoformat(
            timespec="seconds")
        state.save(ws.state_dir)
        report = tick(ctx)
        assert any(e.startswith("deadline annealing") for e in report.events)
        questions = Inbox(ws.root).open_questions()
        assert len(questions) == 1 and {o.id for o in questions[0].options} == {"wait", "cancel"}
        Inbox(ws.root).answer(questions[0].entry_id, "wait", by="tester")
        Spool(ws.root).raise_event("human", "annealing", plan_id=report.plan_id,
                                   inbox=questions[0].entry_id, option="wait", action="wait",
                                   args={}, by="tester")
        report = tick(ctx)
        assert not Inbox(ws.root).open_questions()
        since = LoopState.load(ws.state_dir).deadline_from["annealing"]
        assert datetime.fromisoformat(since) > datetime.now(UTC) - timedelta(minutes=1)
        assert not any(e.startswith("deadline") for e in report.events)


def test_the_plan_prices_its_ceiling_and_refuses_unknown_sentinels(ws_ctx):
    """The approval covers the plan at its ceiling — further segments, further
    arms, the envelope's extensions — and a sentinel the loop cannot evaluate is refused
    at writing."""
    run_inp, tpr = _full_mimic_setup(ws_ctx)
    step = {"name": "production", "tool": "mimic_run", "needs": ["tpr", "cpmd_input"],
            "values": {"cpmd_input": run_inp, "tpr": tpr, **RES},
            "sentinels": [{"check": "no_such_check", "args": {}}]}
    refused = REGISTRY.call("plan_write", ws_ctx, {"question": "q", "steps": [step],
                                                   "given": ["tpr", "cpmd_input"]})
    assert not refused.ok and "PLAN_SENTINEL_UNKNOWN" in {w.id for w in refused.blocking}
    step["sentinels"] = [HEAT]
    written = REGISTRY.call("plan_write", ws_ctx, {
        "question": "q", "steps": [step], "given": ["tpr", "cpmd_input"],
        "envelope": {"extend_segments": 2}})
    assert written.ok, written.summary
    assert written.ceiling_core_h == pytest.approx(3 * written.total_core_h, rel=1e-6)
    assert written.envelope["extend_segments"] == 2


def test_the_spool_kinds_are_the_documented_ones():
    import typing
    from pathlib import Path

    from myrmidon.core.spool import EventKind

    doc = (Path(__file__).resolve().parents[2] / "docs" / "FORMATS.md").read_text()
    spool = doc[doc.index("### Spool"):doc.index("### Journal")]
    for kind in typing.get_args(EventKind):
        assert f"`{kind}`" in spool, kind
