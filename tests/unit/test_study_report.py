"""`study_report`: the report is the product of a study.

Every value in it is parsed from a run's own output and names the ledger event and file it
came from; the model writes prose around the values and never a value of its own.
"""

from pathlib import Path

import pytest

from myrmidon.core.report import run_values
from myrmidon.core.tool import REGISTRY

CPMD = Path(__file__).resolve().parents[1] / "fixtures" / "cpmd"

# What a coupled run's log states, from real CPMD 4.3 logs. (log, label -> value)
LOGS = [
    ("mimic_nve_test.out", {"ionic temperature control": "none (energy conserved)",
                            "MD steps completed": 1000, "simulated time": 0.2416,
                            "conserved-energy drift": -3.025e-4, "finished normally": True}),
    ("mimic_heating_berendsen.out", {"ionic temperature control": "Berendsen-style thermostat",
                                     "MD steps completed": 30, "final ionic temperature": 95.5,
                                     # the last 3 of 30 steps: 94.7, 95.3, 95.5 K
                                     "mean ionic temperature over the last 10 % of steps":
                                         95.1667,
                                     "finished normally": False}),
    ("acetone_bomd_running.out", {"MD steps completed": 2, "final ionic temperature": 0.4}),
]


@pytest.mark.parametrize("log,expected", LOGS)
def test_a_runs_values_are_read_from_its_log(log, expected):
    found = {label: value for label, value, _ in run_values((CPMD / log).read_text())}
    for label, value in expected.items():
        assert found[label] == (pytest.approx(value, rel=1e-3)
                                if isinstance(value, float) else value), label
    if "conserved-energy drift" not in expected:  # only an unthermostatted run has one
        assert "conserved-energy drift" not in found


# (the prose, the file name, the finding that refuses it)
NARRATED = [
    ({}, "report.md", None),
    ({"results": "The run is the production segment the plan asked for."}, "report.md", None),
    # a number the section states is the section's: the question asked for 100 ps
    ({"question": "The question is about 100 ps of dynamics."}, "report.md", None),
    ({"results": "It sampled 4242 MD steps."}, "report.md", "REPORT_NUMBER_UNGROUNDED"),
    ({"summary": "All went well."}, "report.md", "REPORT_SECTION_UNKNOWN"),
    ({}, "../report.md", "REPORT_OUTPUT_NOT_IN_STUDY"),
]


@pytest.mark.parametrize("narrative,output,finding", NARRATED)
def test_the_report_states_parsed_values_and_prose_around_them(ws_ctx, narrative, output,
                                                               finding):
    from tests.unit.test_mimic import _full_mimic_setup, _wait

    run_inp, tpr = _full_mimic_setup(ws_ctx)
    step = {"name": "production", "tool": "mimic_run", "needs": ["tpr", "cpmd_input"],
            "values": {"cpmd_input": run_inp, "tpr": tpr, "cpmd_ranks": 1,
                       "gromacs_ranks": 1, "walltime_min": 10}}
    assert REGISTRY.call("plan_write", ws_ctx, {
        "question": "100 ps of QM/MM dynamics of the solute?", "steps": [step],
        "given": ["tpr", "cpmd_input"], "overwrite": True}).ok
    assert REGISTRY.call("plan_approve", ws_ctx, {}).ok
    job = REGISTRY.call("mimic_run", ws_ctx, {})  # every value is the plan's
    assert job.ok, job.summary
    _wait(ws_ctx, job.job_id)

    obs = REGISTRY.call("study_report", ws_ctx, {"narrative": narrative, "output": output})
    if finding:
        assert not obs.ok
        assert [w.id for w in obs.blocking] == [finding]
        return
    assert obs.ok, obs.summary
    ledger = {ev.event_id: ev for ev in ws_ctx.workspace.ledger.events()}
    assert obs.values and all(ledger[v.event].tool == "mimic_run" for v in obs.values)
    assert {v.step for v in obs.values} == {"production"}
    assert (ws_ctx.workspace.root / output).is_file()
