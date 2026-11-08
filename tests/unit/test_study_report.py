"""`study_report`: the report is the product of a study.

Every value in it is parsed from a run's own output and names the ledger event and file it
came from; the model writes prose around the values and never a value of its own.
"""

import json
from pathlib import Path

import pytest
from tests.unit.findings_catalog import blocking

from myrmidon.core.tool import REGISTRY
from myrmidon.engines.facts import run_values

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CPMD = FIXTURES / "cpmd"

# What a run's log states, from real CPMD 4.3 and GROMACS 2021.6 logs, told apart by the
# log itself. (log, label -> value)
LOGS = [
    ("cpmd/mimic_nve_test.out", {"ionic temperature control": "none (energy conserved)",
                                 "MD steps completed": 1000, "simulated time": 0.2416,
                                 "conserved-energy drift": -3.025e-4,
                                 "finished normally": True}),
    ("cpmd/mimic_heating_berendsen.out", {
        "ionic temperature control": "Berendsen-style thermostat",
        "MD steps completed": 30, "final ionic temperature": 95.5,
        # the last 3 of 30 steps: 94.7, 95.3, 95.5 K
        "mean ionic temperature over the last 10 % of steps": 95.1667,
        "finished normally": False}),
    ("cpmd/acetone_bomd_running.out", {"MD steps completed": 2, "final ionic temperature": 0.4}),
    # a completed classical NPT run: mdrun's own averages over the run, and its last line
    ("gromacs/mdrun_npt_completed.log", {
        "MD steps completed": 100000, "simulated time": 200.0,
        "mean temperature over the run (mdrun's average)": 299.701,
        "mean pressure over the run (mdrun's average)": 2.1228,
        "mean potential energy over the run (mdrun's average)": -52550.7,
        "energy frames the averages are over": 1001,
        "performance": 390.119, "finished normally": True}),
    # a completed steepest-descent minimisation: its own closing lines, no simulated time
    ("gromacs/mdrun_em_completed.log", {
        "minimisation steps": 312, "minimisation converged to its force tolerance": True,
        "final potential energy": -122544.34, "final maximum force": 997.66,
        "finished normally": True}),
]


@pytest.mark.parametrize("log,expected", LOGS)
def test_a_runs_values_are_read_from_its_log(log, expected):
    found = {label: value for label, value, _ in run_values((FIXTURES / log).read_text())}
    for label, value in expected.items():
        assert found[label] == (pytest.approx(value, rel=1e-3)
                                if isinstance(value, float) else value), label
    if "conserved-energy drift" not in expected:  # only an unthermostatted run has one
        assert "conserved-energy drift" not in found


# (the prose, the file name, the finding that refuses it)
NARRATED = [
    ({}, "report.md", None),
    ({"results": "The run is the production segment the mandate asked for."}, "report.md",
     None),
    # the prose is the model's and stands as written; the values are the section's
    ({"question": "The question is about 100 ps of dynamics."}, "report.md", None),
    ({"results": "It sampled 4242 MD steps."}, "report.md", None),
    # a fixed section whatever the key's case and spacing; another name a section of its own
    ({"Results ": "Read as the results.", "What_ran": "Four stages.", "summary": "All went well.",
      "SUMMARY": "Said twice."}, "report.md", None),
    ({}, "../report.md", "REPORT_OUTPUT_NOT_IN_STUDY"),
]


@pytest.mark.parametrize("narrative,output,finding", NARRATED)
def test_the_report_states_parsed_values_and_prose_around_them(ws_ctx, narrative, output,
                                                               finding):
    from tests.conftest import REGION, coupled_study, wait_for

    coupled_study(ws_ctx, [{"name": "production", "kind": "dynamics", "regime": "qmmm",
                            "ensemble": "NVT", "temperature_k": 300, "steps": 5,
                            "qm_region": REGION}])  # a thermostatted stage writes its dipole
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "production"}).ok
    job = REGISTRY.call("stage_run", ws_ctx, {"stage": "production", "cores": 2,
                                              "walltime_min": 10})
    assert job.ok, job.summary
    wait_for(ws_ctx, job.job_id)

    obs = REGISTRY.call("study_report", ws_ctx, {"narrative": narrative, "output": output})
    if finding:
        assert not obs.ok
        assert [w.id for w in blocking(obs)] == [finding]
        return
    assert obs.ok, obs.summary
    ledger = {ev.event_id: ev for ev in ws_ctx.workspace.ledger.events()}
    assert obs.values and all(ledger[v.event].tool == "stage_run" for v in obs.values)
    assert [v.value for v in obs.values if v.label == "finished normally"] == [True]
    assert [v.file.endswith("/DIPOLE") for v in obs.values
            if v.label == "mean magnitude of the QM dipole"] == [True]
    written = (ws_ctx.workspace.root / output).read_text()
    assert all(text in written for text in narrative.values())
    record = json.loads((ws_ctx.workspace.root / "report.json").read_text())
    assert record["mandate_digest"] and record["rounds"] == []
    assert [row["stage"] for row in record["stages"]] == ["production"]
    assert record["stages"][0]["reached"] != "not run" and record["deviations"] == []
    assert len(record["values"]) == len(obs.values)
    headings = [line for line in written.splitlines() if line.startswith("## ")]
    assert len(set(headings)) == len(headings)
    if "summary" in narrative:
        assert written.index("## Summary") > written.index("## Record")
        assert written.index("Read as the results.") < written.index("## Deviations")


def test_an_analysis_is_reported_under_the_run_whose_files_it_read(ws_ctx):
    """Runs share a directory; an analysis of one run's energy file is that run's and no
    other's. Only a run whose collection recorded no files falls back to the directory."""
    from myrmidon.core.artifacts import ArtifactRef
    from myrmidon.core.provenance import ProvenanceEvent
    from myrmidon.orchestration.report import analysis_values, collection_outputs

    def collected_files(events, job):
        return {a.path for a in collection_outputs(events, job)}

    ws = ws_ctx.workspace
    (ws.root / "07_analysis").mkdir(exist_ok=True)
    xvg = ws.root / "07_analysis" / "md_energy.xvg"
    xvg.write_text('@ s0 legend "Temperature"\n0 299.0\n10 301.0\n20 300.0\n')
    xvg_ref = ArtifactRef.from_path(xvg, "data", root=ws.root)

    def ref(path):
        return ArtifactRef(path=path, role="edr", sha256="0" * 64, size_bytes=1)

    events = [
        ProvenanceEvent(tool="job_status", params={"job_id": "job-em"},
                        outputs=[ref("06_run/em.edr")]),
        ProvenanceEvent(tool="job_status", params={"job_id": "job-md"},
                        outputs=[ref("06_run/md.edr")]),
        ProvenanceEvent(tool="observable_measure", params={"terms": ["Temperature"]},
                        inputs=[ref("06_run/md.edr")], outputs=[xvg_ref]),
    ]
    assert collected_files(events, "job-md") == {"06_run/md.edr"}
    for job, n_values in (("job-em", 0), ("job-md", 3)):
        _, values = analysis_values(ws, events, "06_run", collected_files(events, job))
        assert len(values) == n_values, job
    _, values = analysis_values(ws, events, "06_run", collected_files(events, "job-none"))
    assert len(values) == 3


# (the extractor, the property, what its observation states, the statistics the deliverable
# asks for) -> the quantity the ledger keeps, and the statistics the report states of it
MEASURED = [
    ("trajectory", "rmsd", {"metric": "rmsd", "unit": "Å", "mean": 1.5, "sd": 0.2,
                            "minimum": 0.0, "largest": 2.0, "final": 1.9}, ["mean", "max"],
     ("rmsd", "Å", {"mean": 1.5, "max": 2.0})),
    ("energy", "thermodynamics", {"terms": {"Temperature": {
        "unit": "K", "mean": 300.2, "std": 2.1, "drift_per_ns": 0.4, "n_points": 9}}},
     ["mean"], ("Temperature", "K", {"mean": 300.2})),
    ("run_log", "energy_drift", {"eclassic_drift_ha_per_ps": -3.0e-4}, [],
     ("conserved-energy drift", "Ha/ps", {"value": -3.0e-4})),
    ("run_log", "dipole", {"dipole_debye_mean": 2.8, "dipole_debye_sd": 0.3,
                           "dipole_debye_last": 2.9}, ["mean", "sd"],
     ("QM dipole magnitude", "D", {"mean": 2.8, "sd": 0.3})),
    ("forces", "force_rmse", {"units": "Ha/bohr", "rmse": 0.012, "ref_rms": 0.05}, [],
     ("force RMSE", "Ha/bohr", {"value": 0.012})),
]


@pytest.mark.parametrize("extractor,prop,stated,asked,expected", MEASURED)
def test_a_deliverable_is_reported_as_its_measurement_read_it(extractor, prop, stated, asked,
                                                              expected):
    """The measuring call states each quantity it read with its statistics; the ledger keeps
    them and the report's row is them, whatever the property; unmeasured reads so."""
    from myrmidon.core.mandate import Mandate
    from myrmidon.core.observation import Observation
    from myrmidon.core.provenance import ProvenanceEvent
    from myrmidon.orchestration.report import deliverable_rows
    from myrmidon.tools.measure import MeasureParams, _asked, _quantities

    quantities = _asked(_quantities(extractor, MeasureParams(property=prop),
                                    Observation(ok=True, summary="", **stated)), asked)
    name, unit, statistics = expected
    assert (quantities[0]["name"], quantities[0]["unit"]) == (name, unit)
    assert quantities[0]["statistics"] == pytest.approx(statistics)

    mandate = Mandate.model_validate({"question": "q", "deliverables": [
        {"name": "it", "property": prop, "over": "production", "statistics": asked},
        {"name": "never", "property": prop, "over": "production"}]})
    event = ProvenanceEvent(tool="observable_measure", mandate="d" * 64, measured={
        "deliverable": "it", "file": "07_analysis/x", "quantities": quantities})
    stale = event.model_copy(update={"mandate": "", "event_id": "ev-unbound"})
    lines, values, record = deliverable_rows(mandate, [event, stale], {"d" * 64}, None)
    assert {(v.quantity, v.label, v.unit, v.event) for v in values if v.quantity == name} == {
        (name, k, unit, event.event_id) for k in statistics}
    assert record["it"]["event"] == event.event_id and not record["never"]["measured"]
    assert sum("NOT measured" in line for line in lines) == 1
