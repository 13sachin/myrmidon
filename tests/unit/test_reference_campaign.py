"""The acetone reference campaign runs through the loop on the mocks: the three
questions from the plan files in tests/reference_campaign/, approved
once and advanced by ticks until done — the walked-through instance of the execution shape
(docs/TUTORIAL.md) and the baseline every framework is scored against."""

import importlib.util
import json
from pathlib import Path

import pytest

from myrmidon.core.spool import Inbox, Journal, LoopState, Spool

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "reference_campaign", ROOT / "tests" / "reference_campaign" / "reference_campaign.py")
campaign = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(campaign)


def _run(ws_ctx, which):
    ws = ws_ctx.workspace
    campaign.stage_inputs(ws, which)
    written = campaign.write_and_approve(ws_ctx, which)
    report = campaign.run_until_settled(ws_ctx, max_ticks=60, pause_s=0.3)
    return written, report


def test_q1_repeat_until_extends_once_and_stops_when_the_criterion_holds(ws_ctx):
    """The first 200 ps settle too late for a verdict, one continuation of 200 ps from
    the checkpoint follows without a model call, and the criterion holds at 400 ps — two
    of the five segments the researcher approved."""
    written, report = _run(ws_ctx, "q1")
    assert written.shape == "chain" and written.ceiling_core_h > written.total_core_h
    assert report.done, report.messages
    ws = ws_ctx.workspace
    checks = [ev for ev in Spool(ws.root).all() if ev.kind == "sentinel"
              and ev.detail.get("until")]
    assert [c.detail["met"] for c in checks] == [None, True]
    assert "SERIES_SETTLED_LATE" in checks[0].detail["findings"]
    assert [c.detail["segments"] for c in checks] == [1, 2]
    state = LoopState.load(ws.state_dir)
    assert state.extensions == {"heat_run": 1} and state.until_closed
    tools = [ev.tool for ev in ws.ledger.events() if ev.kind == "call" and ev.ok]
    assert tools.count("checkpoint_continue") == 1
    assert not Inbox(ws.root).open_questions()
    assert all(e.by == "machinery" for e in Journal(ws.root).all())


def test_q2_chain_runs_every_stage_and_the_envelope_extends_the_heating(ws_ctx):
    """Four coupled stages from restart to restart; the heating sentinel catches the
    planted method fault in a job that exited 0, the envelope's one extension continues
    from the heating's own restart and reaches the target, production follows."""
    written, report = _run(ws_ctx, "q2")
    assert report.done, report.messages
    ws = ws_ctx.workspace
    events = Spool(ws.root).all()
    violated = [ev for ev in events if ev.kind == "sentinel" and not ev.detail.get("holds", True)]
    assert [(v.subject, v.detail["check"]) for v in violated] == [("heating", "t_qm_tail_in_range")]
    state = LoopState.load(ws.state_dir)
    assert state.extensions == {"heating": 1}
    runs = {ev.subject: ev.detail.get("action") for ev in events if ev.kind == "job"}
    assert runs == {"annealing": "run", "nve_test": "run", "heating": "extend",
                    "production": "run"}
    # heating's T_init came from the annealing run, parsed, never typed
    resolved = [ev for ev in ws.ledger.events() if "PLAN_REFERENCE_RESOLVED" in ev.knowledge_refs]
    assert resolved and all(ev.plan_step == "heating_method" for ev in resolved)
    # the restart chain is in the ledger: every later stage consumed the earlier's restart
    restarts = [ev.plan_step for ev in ws.ledger.events() if ev.tool == "mimic_run" and ev.ok
                and any(a.role == "restart" for a in ev.inputs)]
    assert restarts == ["nve_test", "heating", "heating", "production"]
    report_obs = campaign.write_report(ws_ctx)
    assert report_obs.ok, report_obs.summary
    text = (ws.root / "report.md").read_text()
    assert "heating, segment 2" in text and "production" in text


def test_q3_sweep_probes_first_then_runs_the_rest_and_joins(ws_ctx):
    """Three arms; the first single point runs alone and is checked before the other
    two are submitted; a step whose parameter takes a list joins the three."""
    written, report = _run(ws_ctx, "q3")
    assert written.shape == "sweep" and written.ceiling_core_h == pytest.approx(
        3 * written.total_core_h, rel=1e-3)
    assert report.done, report.messages
    ws = ws_ctx.workspace
    state = LoopState.load(ws.state_dir)
    events = list(ws.ledger.events())
    submitted = {}
    for index, ev in enumerate(events):
        if ev.tool == "mimic_run" and ev.ok:
            key = next(k for k, evs in state.runs.items() if ev.event_id in evs)
            submitted[key] = index
    assert list(submitted) == ["sp[frame_1]", "sp[frame_2]", "sp[frame_3]"]
    # the probe: the first point's job was collected before the other two were submitted
    probe_job = Spool(ws.root).all()[0].detail["job_id"]
    collected = next(i for i, ev in enumerate(events) if ev.tool == "job_status"
                     and ev.kind == "call" and ev.params.get("job_id") == probe_job)
    assert submitted["sp[frame_1]"] < collected < submitted["sp[frame_2]"]
    merge = next(ev for ev in ws.ledger.events() if ev.tool == "fm_merge" and ev.ok)
    assert len(merge.params["fm_jsons"]) == 3 and len(merge.params["trajectories"]) == 3
    report_obs = campaign.write_report(ws_ctx)
    assert report_obs.ok
    assert {v.step for v in report_obs.values} >= {"sp[frame_2]", "sp[frame_3]"}


def test_the_example_plans_are_what_the_tests_ran():
    """One source: the plan files, read by the example and by these tests."""
    for which in campaign.QUESTIONS:
        params = campaign.plan_params(which)
        assert params["steps"] and params["question"]
        json.dumps(params)  # plan_write's parameters, serialisable as a call
