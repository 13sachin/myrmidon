"""fm_report: the campaign report from the ledger — parsers on real-format .itp text,
round linking by artifact paths, the mocked chain end to end."""

import json
import time

import pytest

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.provenance import ProvenanceEvent
from myrmidon.core.tool import REGISTRY
from myrmidon.engines.mimic.fm_report import (
    bonded_diff,
    charge_diff,
    parse_itp_atoms,
    parse_itp_bonded,
)

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


def _mock_campaign(ws_ctx):
    """plan -> fit -> results -> candidate through the mocked tools, then a validation
    round planted as ledger events (the validate driver has its own tests)."""
    ws = ws_ctx.workspace
    prep = ws.stage("prep")
    for name in ("ref.h5", "traj.trr", "conf.gro", "sele.dat", "index.ndx"):
        (prep / name).write_text(f"mock {name}\n")
    (prep / "mock.itp").write_text(
        "[ moleculetype ]\n ACT 3\n[ atoms ]\n 1 c 1 ACT C1 1 0.81020 12.0\n"
        " 2 o 1 ACT O1 2 -0.58640 16.0\n 3 hc 1 ACT H1 3 0.12160 1.0\n")
    (prep / "topol.top").write_text('#include "mock.itp"\n[ system ]\nmock\n[ molecules ]\nACT 1\n')
    files = {"fmdata": "01_prep/ref.h5", "trajectory": "01_prep/traj.trr",
             "topology": "01_prep/topol.top", "coords": "01_prep/conf.gro",
             "selection": "01_prep/sele.dat", "index": "01_prep/index.ndx"}
    plan = REGISTRY.call("fm_plan", ws_ctx, {
        "stride": [0, 10, 1], "qm_total_charge": 0, "dataset_frames": 12,
        "extra": {"regularization": "True",
                  "regularization_force_constant_min_fraction": "0.25"}})
    assert plan.ok, plan.summary
    job = REGISTRY.call("fm_fit", ws_ctx, {**files, "fm_input": plan.artifacts[0].path})
    assert job.ok, job.summary
    # the poll that finds the fit ended collects it: the fitted itp, the original it was
    # named after (an input of the collection) and the candidate topology, in one event
    res = _wait(ws_ctx, job.job_id)
    assert res.state == "COMPLETED" and res.ok, res.summary
    assert any(a.role == "fitted_topology" for a in res.artifacts)
    cand_top = next(a.path for a in res.artifacts if a.role == "topology")
    assert cand_top.endswith("candidate.top")
    vstage = ws.stage("validate")
    data = {"ok": True, "reference_frames": 12, "min_frames": [3, 9],
            "reference": {"C1-O1": {"mean": 1.230, "std": 0.02}},
            "min_geometry": {
                "stock": {"n_frames_requested": 2, "n_frames_minimized": 2,
                          "pairs": {"C1-O1": {"mean": 1.259, "std": 0.0}}, "mae": 0.029},
                "fitted": {"n_frames_requested": 2, "n_frames_minimized": 2,
                           "pairs": {"C1-O1": {"mean": 1.237, "std": 0.0}}, "mae": 0.007}},
            "force_frames": [0, 5],
            "force_rmse": {
                "stock": {"n_frames": 2, "n_atoms": 3, "rmse": 0.02, "ref_rms": 0.05,
                          "rel_rmse": 0.4, "units": "Hartree/Bohr", "outliers": [],
                          "by_type": [{"type": "c", "element": "C", "n_atoms": 1,
                                       "n_samples": 2, "rmse": 0.03, "ref_rms": 0.06,
                                       "rel_rmse": 0.5}]},
                "fitted": {"n_frames": 2, "n_atoms": 3, "rmse": 0.01, "ref_rms": 0.05,
                           "rel_rmse": 0.2, "units": "Hartree/Bohr", "outliers": [],
                           "by_type": [{"type": "c", "element": "C", "n_atoms": 1,
                                        "n_samples": 2, "rmse": 0.012, "ref_rms": 0.06,
                                        "rel_rmse": 0.2}]}}}
    (vstage / "validate_results.json").write_text(json.dumps(data))
    ws.ledger.append(ProvenanceEvent(
        tool="fm_validate", params={"topologies": {"stock": "01_prep/topol.top",
                                                   "fitted": cand_top},
                                    "metrics": ["min_geometry"], "stage": "validate"}))
    ws.ledger.append(ProvenanceEvent(  # the job_status call that collected the validation
        tool="job_status", params={"job_id": "job-validate", "wait_seconds": 0},
        outputs=[ArtifactRef.from_path(vstage / "validate_results.json", "validation",
                                       root=ws.root)]))
    return cand_top

