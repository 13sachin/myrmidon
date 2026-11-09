"""A fit's tables: the campaign report from the ledger — parsers on real-format .itp text,
round linking by artifact paths, the mocked chain end to end."""

import time

import pytest

from myrmidon.core.tool import REGISTRY
from myrmidon.engines.gromacs.itp import parse_itp_atoms, parse_itp_bonded
from myrmidon.orchestration.fit_report import bonded_diff, charge_diff

# real mimicpy output layout (a ligand itp from a metalloenzyme fit): mixed row formats,
# comments after ';', funct-9 dihedrals with one row per multiplicity
ORIG = """; original
[ moleculetype ]
;name nrexcl
 NDP 3

[ atoms ]
;   nr  type  resi  res  atom  cgnr  charge  mass
     1 NDP_CA   1 NDP  C6N   1  -0.2000  12.010 ; qtot -0.2
     2 NDP_H4   1 NDP  H26   2   0.1733   1.008
     3 NDP_CA   1 NDP  C5N   3  -0.2000  12.010

[ bonds ]
;   ai  aj funct  r  k
     1  2  1  1.0890e-01  2.8577e+05 ; C6N - H26
     1  3  1  1.3980e-01  3.8585e+05 ; C6N - C5N

[ angles ]
     1  3  2  1  1.1988e+02  4.0334e+02

[ dihedrals ] ; propers
     1  2  3  1  9  0.00  0.50000  1
     1  2  3  1  9  180.00  1.00000  2

[ moleculetype ]
 AKG 3
[ atoms ]
     1 AKG_c  1 AKG  C1  1  0.8000  12.010
[ bonds ]
     1  1  1
"""
FITTED = ORIG.replace("-0.2000  12.010 ; qtot -0.2", "-0.246791  12.010") \
    .replace("1.3980e-01  3.8585e+05", "0.1398       96462.5") \
    .replace("180.00  1.00000  2", "180.00  0.80000  2") \
    .replace("C1  1  0.8000", "C1  1  0.7500")


def test_parse_itp_atoms_and_bonded_real_layout():
    atoms = parse_itp_atoms(ORIG)
    assert list(atoms) == ["NDP", "AKG"]
    assert atoms["NDP"][0] == {"nr": 1, "type": "NDP_CA", "name": "C6N", "charge": -0.2}
    bonded = parse_itp_bonded(ORIG)
    assert bonded["NDP"]["bonds"][(1, 3, 1, 0)] == [0.1398, 385850.0]
    assert bonded["NDP"]["angles"][(1, 3, 2, 1, 0)] == [119.88, 403.34]
    # both multiplicities kept, disambiguated by occurrence
    assert bonded["NDP"]["dihedrals"][(1, 2, 3, 1, 9, 0)] == [0.0, 0.5, 1.0]
    assert bonded["NDP"]["dihedrals"][(1, 2, 3, 1, 9, 1)] == [180.0, 1.0, 2.0]
    # a bond row without parameters (force-field types) is not a fitted term
    assert "AKG" not in bonded


def test_charge_and_bonded_diff():
    c = charge_diff(parse_itp_atoms(ORIG), parse_itp_atoms(FITTED))
    assert c["n_atoms_compared"] == 4 and c["n_changed"] == 2
    assert c["rows"][0]["atom"] == "C1" and c["rows"][0]["dq"] == pytest.approx(-0.05)
    assert c["max_abs_dq"] == pytest.approx(0.05)
    assert {t["type"] for t in c["by_type"]} == {"AKG_c", "NDP_CA"}
    b = bonded_diff(parse_itp_bonded(ORIG), parse_itp_bonded(FITTED), parse_itp_atoms(FITTED),
                    floor_fraction=0.25)
    assert b["n_terms_compared"] == 5 and b["n_changed"] == 2
    kb = next(r for r in b["rows"] if r["section"] == "bonds")
    assert kb["atoms"] == "C6N-C5N" and kb["param"] == "kb"
    assert kb["rel_change"] == pytest.approx(-0.75) and kb["at_floor"] is True
    kd = next(r for r in b["rows"] if r["section"] == "dihedrals")
    assert kd["param"] == "kd" and kd["before"] == 1.0 and kd["after"] == 0.8
    assert b["n_at_floor"] == 1 and b["changed_by_section"] == {"bonds": 1, "dihedrals": 1}


def _wait(ctx, job_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = REGISTRY.call("job_status", ctx, {"job_id": job_id})
        if st.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
        time.sleep(0.1)
    raise AssertionError("fm job did not finish")



# a fit study's deliverables, one row per property kind it states: (property, the stage it is
# measured over, whether it can be measured here: the rerun never runs, the QM region is held)
FIT_DELIVERABLES = [("force_rmse", "rerun", False), ("qm_forces", "rerun", False),
                    ("boundary_bonds", "fit", True)]


@pytest.mark.parametrize("prop,over,measurable", FIT_DELIVERABLES)
def test_a_fit_studys_report_states_its_rounds_stages_and_deliverables(ws_ctx, prop, over,
                                                                       measurable):
    """A fit study's report.json carries the fit's rounds beside the stages and the
    deliverables, every one stated as measured or not, whatever its property."""
    import json

    from tests.conftest import REGION, coupled_study
    from tests.unit.test_fm import _dataset_files

    coupled_study(ws_ctx, [{"name": "fit", "kind": "parameter_fit", "qm_region": REGION},
                           {"name": "rerun", "kind": "single_point", "regime": "mm",
                            "parameters_from": "fit"}],
                  deliverables=[{"name": "it", "property": prop, "over": over}])
    files = _dataset_files(ws_ctx.workspace)
    assert REGISTRY.call("dataset_collect", ws_ctx, {
        "fmdata": files["fmdata"], "trajectory": files["trajectory"],
        "topology": files["topology"], "coords": files["coords"],
        "selection": files["selection"]}).ok
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "rerun",
                                                   "structure": files["coords"]}).ok
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "fit",
                                                   "overrides": {"stride": "[0,10,1]"}}).ok
    job = REGISTRY.call("stage_run", ws_ctx, {"stage": "fit", "cores": 2,
                                              "fit_mode": "charges"})
    assert job.ok, job.summary
    assert _wait(ws_ctx, job.job_id).ok

    report = REGISTRY.call("study_report", ws_ctx, {})
    assert report.ok, report.summary
    record = json.loads((ws_ctx.workspace.root / "report.json").read_text())
    assert [r["stage"] for r in record["rounds"]] == ["fit"]
    assert {row["stage"]: row["reached"] != "not run" for row in record["stages"]} == {
        "rerun": False, "fit": True}
    assert record["deliverables"]["it"]["property"] == prop
    # the parameter set its stage runs on, as the mandate names it
    assert record["deliverables"]["it"]["parameters_from"] == ("fit" if over == "rerun" else "")
    assert record["deliverables"]["it"]["measured"] is False  # nothing has measured it yet
    measure = REGISTRY.call("observable_measure", ws_ctx, {"deliverable": "it"})
    assert measure.ok is measurable, measure.summary
    assert REGISTRY.call("study_report", ws_ctx, {}).ok
    it = json.loads((ws_ctx.workspace.root / "report.json").read_text())["deliverables"]["it"]
    assert it["measured"] is measurable
    if measurable:
        assert it["event"] == measure.provenance_id and it["quantities"]
