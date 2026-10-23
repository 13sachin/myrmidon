"""The loop as the agent reaches it: start a detached follower, one pass in the session,
where it stands, a person's answer — and what an unattended session may not do."""

import os
import time

from tests.unit.test_loop import ANNEAL, HEAT, RAMP, _plan, _stage, _ticks

from myrmidon.core.spool import Inbox
from myrmidon.core.tool import REGISTRY


def test_the_loop_runs_only_an_approved_plan_and_a_pass_reports_where_it_stands(ws_ctx):
    _plan(ws_ctx, _stage(ws_ctx, "annealing", sentinels=[ANNEAL]), approve=False)
    refused = REGISTRY.call("loop_start", ws_ctx, {})
    assert not refused.ok and "LOOP_PLAN_NOT_APPROVED" in {w.id for w in refused.blocking}
    assert REGISTRY.call("plan_approve", ws_ctx, {"acknowledge_warnings": ["COMPUTE_BUDGET"]}).ok
    first = REGISTRY.call("loop_tick", ws_ctx, {})
    assert first.ok and any(s.startswith("annealing (run) job") for s in first.submitted)
    assert not any(ev.tool == "loop_tick" for ev in ws_ctx.workspace.ledger.events())
    status = REGISTRY.call("loop_status", ws_ctx, {})
    assert status.ok and status.steps[-1]["step"] == "annealing"
    assert status.running or status.steps[-1]["state"] in ("running", "ended", "done")
    report = _ticks(ws_ctx)
    assert report.done and REGISTRY.call("loop_status", ws_ctx, {}).done


def test_an_answer_is_the_researchers_and_an_unattended_session_cannot_give_it(ws_ctx):
    steps = _stage(ws_ctx, "annealing", sentinels=[ANNEAL]) + _stage(
        ws_ctx, "heating", restart_from="cpmd_restart_annealing", sentinels=[HEAT],
        extra_method=RAMP, inputset="inputset:mimic_cpmd_heating@v3")
    _plan(ws_ctx, steps, envelope={"extend_segments": 0})
    _ticks(ws_ctx, n=40)
    status = REGISTRY.call("loop_status", ws_ctx, {})
    assert len(status.questions) == 1 and status.questions[0]["step"] == "heating"
    entry = status.questions[0]["entry_id"]
    ws_ctx.unattended = True
    refused = REGISTRY.call("loop_answer", ws_ctx, {"entry_id": entry, "option": "extend"})
    ws_ctx.unattended = False
    assert not refused.ok
    assert "LOOP_ANSWER_IS_THE_RESEARCHERS" in {w.id for w in refused.human_required}
    wrong = REGISTRY.call("loop_answer", ws_ctx, {"entry_id": entry, "option": "nope"})
    assert "LOOP_OPTION_UNKNOWN" in {w.id for w in wrong.blocking}
    answered = REGISTRY.call("loop_answer", ws_ctx, {
        "entry_id": entry, "option": "extend",
        "acknowledge_warnings": ["LOOP_ANSWER_IS_THE_RESEARCHERS"]})
    assert answered.ok, answered.summary
    assert Inbox(ws_ctx.workspace.root).answered(entry).option == "extend"
    report = _ticks(ws_ctx, n=40)
    assert report.done, report.messages


def test_a_detached_follower_outlives_the_session_and_stops_on_request(ws_ctx, cli_env):
    """loop_start spawns `myrmidon tick --follow` with its own session; the pid file says
    so, loop_status sees it alive, loop_stop ends it."""
    _plan(ws_ctx, _stage(ws_ctx, "annealing", sentinels=[ANNEAL]))
    started = REGISTRY.call("loop_start", ws_ctx, {"every_s": 5})
    assert started.ok, started.summary
    assert started.follower_alive and started.follower_pid
    try:
        again = REGISTRY.call("loop_start", ws_ctx, {"every_s": 5})
        assert again.follower_pid == started.follower_pid and "already" in again.summary
        assert REGISTRY.call("loop_status", ws_ctx, {}).follower_alive
        # the follower, not this session, carries the plan to its end on the mock scheduler
        for _ in range(120):
            status = REGISTRY.call("loop_status", ws_ctx, {})
            if status.done or status.questions:
                break
            time.sleep(0.25)
        assert status.done, (status.summary, status.questions)
    finally:
        stopped = REGISTRY.call("loop_stop", ws_ctx, {})
    assert stopped.ok and "stopped" in stopped.summary
    for _ in range(50):
        try:
            os.kill(started.follower_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    assert not REGISTRY.call("loop_status", ws_ctx, {}).follower_alive
    assert (ws_ctx.workspace.state_dir / "tick.log").exists()
