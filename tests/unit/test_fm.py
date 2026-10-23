"""Force-matching tools: parsers vs real MiMiCPy output fixtures + mocked protocol."""

import time
from pathlib import Path

import pytest

from myrmidon.core.tool import REGISTRY
from myrmidon.engines.mimic.fm import (
    parse_fm_log,
    parse_fmcheck_report,
    parse_h5info,
    parse_optimization_results,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "fm"


def test_a_real_fit_log_and_its_results_file_agree():
    """Hand-checked against a real acetone/water fit's log; the results file the same
    fit wrote must say the same."""
    log = parse_fm_log((FIXTURES / "fm.log").read_text())
    assert log["n_snapshots"] == 1000
    assert log["n_train"] == 900 and log["n_val"] == 100
    assert log["validation_sdf"] == pytest.approx(0.655820)
    assert log["validation_cost"] == pytest.approx(0.118034)
    assert log["optimizer_success"] is True and "ftol" in log["optimizer_message"]
    assert log["dresp_completed"] and log["completed"]
    assert log["wall_time_s"] == pytest.approx(31.1)
    assert "opt_acetone.itp" in log["itp_files"]

    results = parse_optimization_results((FIXTURES / "optimization_results.txt").read_text())
    for key in ("validation_sdf", "validation_cost", "n_val"):
        assert results[key] == pytest.approx(log[key])
    assert results["n_optimizer_rows"] >= 4


def test_parse_real_fmcheck_and_h5info():
    check = parse_fmcheck_report((FIXTURES / "fmcheck_report.txt").read_text())
    assert check["trr_frames"] == 2000 and check["h5_frames"] == 2000
    assert check["status"] == "remapped" and check["remapped"] == 20
    assert check["index_aligned"] is False and check["dropped"] == 0
    info = parse_h5info((FIXTURES / "h5info_report.txt").read_text())
    assert info["frames"] == 2000 and info["qm_atoms"] == 10


def test_parse_itp_charges():
    text = "[ moleculetype ]\nACT 3\n[ atoms ]\n; nr type resnr res atom cgnr charge mass\n" \
           " 1 c 1 ACT C1 1 0.81020 12.0\n 2 o 1 ACT O1 2 -0.58640 16.0\n[ bonds ]\n 1 2 1\n"
    from myrmidon.engines.gromacs.itp import atom_charges

    assert atom_charges(text) == [("C1", 0.8102), ("O1", -0.5864)]


def _dataset_files(ws, stage="prep"):
    s = ws.stage(stage)
    for name in ("ref.h5", "traj.trr", "topol.top", "conf.gro", "sele.dat", "index.ndx"):
        (s / name).write_text(f"mock {name}\n")
    return {k: ws.relpath(s / v) for k, v in {
        "fmdata": "ref.h5", "trajectory": "traj.trr", "topology": "topol.top",
        "coords": "conf.gro", "selection": "sele.dat", "index": "index.ndx"}.items()}


def test_fm_collect(ws_ctx, monkeypatch):
    files = _dataset_files(ws_ctx.workspace)
    obs = REGISTRY.call("fm_collect", ws_ctx, {
        "fmdata": files["fmdata"], "trajectory": files["trajectory"],
        "topology": files["topology"], "coords": files["coords"],
        "selection": files["selection"]})
    assert obs.ok, obs.summary
    assert obs.frames == 12 and obs.qm_atoms == 5
    assert obs.index_aligned is True
    assert obs.fmdata_sha256 and obs.trr_sha256 and obs.fmdata_sha256 != obs.trr_sha256
    monkeypatch.setenv("MOCK_FMCHECK_DESYNC", "1")
    obs2 = REGISTRY.call("fm_collect", ws_ctx, {
        "fmdata": files["fmdata"], "trajectory": files["trajectory"],
        "topology": files["topology"], "coords": files["coords"],
        "selection": files["selection"]})
    assert obs2.ok and obs2.index_aligned is False
    assert {w.id for w in obs2.warnings} >= {"FM_TRR_H5_DESYNC", "FM_ALIGNED_PAIR_WRITTEN"}
    # the registered dataset is the aligned pair, not the originals
    paths = {a.role: a.path for a in obs2.artifacts}
    assert paths["fm_dataset"].endswith("aligned.h5")
    assert paths["trajectory"].endswith("aligned.trr")
    assert obs2.fmdata_sha256 != obs.fmdata_sha256

    no_fix = REGISTRY.call("fm_collect", ws_ctx, {
        "fmdata": files["fmdata"], "trajectory": files["trajectory"],
        "topology": files["topology"], "coords": files["coords"],
        "selection": files["selection"], "align_if_desynced": False})
    assert no_fix.ok
    assert not any(w.id == "FM_ALIGNED_PAIR_WRITTEN" for w in no_fix.warnings)


def test_fm_plan_defaults_and_bounds(ws_ctx):
    obs = REGISTRY.call("fm_plan", ws_ctx, {
        "stride": [0, 12, 1], "qm_total_charge": 0, "dataset_frames": 12})
    assert obs.ok, obs.summary
    assert obs.n_snapshots == 12
    assert obs.weights == {"wv": 1.0, "we": 0.01, "wh": 0.01, "wq": 100000.0}
    assert {w.id for w in obs.warnings} >= {"FM_WEIGHT_DEFAULT_WV", "FM_FEW_SNAPSHOTS"}
    text = (ws_ctx.workspace.root / obs.artifacts[0].path).read_text()
    assert "stride: [0,12,1]" in text and "qm_total_charge: 0" in text
    beyond = REGISTRY.call("fm_plan", ws_ctx, {
        "stride": [0, 100, 1], "qm_total_charge": 0, "dataset_frames": 12})
    assert not beyond.ok and "exceeds the dataset" in beyond.summary


def _wait(ctx, job_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = REGISTRY.call("job_status", ctx, {"job_id": job_id})
        if st.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
        time.sleep(0.1)
    raise AssertionError("fm job did not finish")


def test_fm_fit_failure_reported(ws_ctx, monkeypatch):
    files = _dataset_files(ws_ctx.workspace)
    plan = REGISTRY.call("fm_plan", ws_ctx, {
        "stride": [0, 10, 1], "qm_total_charge": 0, "dataset_frames": 12})
    monkeypatch.setenv("MOCK_FM_FAIL", "1")
    job = REGISTRY.call("fm_fit", ws_ctx, {**files, "fm_input": plan.artifacts[0].path})
    res = _wait(ws_ctx, job.job_id)
    assert res.state == "FAILED"
    assert not res.ok
    assert res.optimizer_success is False
    assert res.log_tail


def test_parse_cpmd_series_real_nose_log():
    from myrmidon.engines.mimic.diagnostics import parse_cpmd_series

    text = (Path(__file__).parent.parent / "fixtures" / "cpmd" / "nose_steps.out").read_text()
    steps = parse_cpmd_series(text)
    assert steps, "no MD steps parsed from the real Nose log"
    s2 = next(s for s in steps if s["nfi"] == 2)
    # hand-checked against the real log
    assert s2["temp_k"] == pytest.approx(301.6)
    assert s2["eks"] == pytest.approx(-1524.44666)
    assert s2["eclassic"] == pytest.approx(-1339.54573)
    assert s2["scf_iters"] and s2["scf_iters"] >= 15
    assert s2["t_qm"] is None  # this build prints 0.00 → treated as absent


def test_fm_frame_report(ws_ctx):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    stage = ws_ctx.workspace.stage("fm")
    dtype = np.dtype([
        ("id", "i4"), ("region", "i4"), ("coordinate", "f8", (3,)),
        ("electric_potential", "f8"), ("electric_field", "f8", (3,)),
        ("hirshfeld_charge", "f8"), ("force", "f8", (3,)),
    ])
    bohr = 1.0 / 0.529177210903
    with h5py.File(stage / "ref.h5", "w") as fh:
        for i in range(4):
            atoms = np.zeros(3, dtype=dtype)
            atoms["id"] = [10, 11, 500]
            atoms["region"] = [1, 1, 2]
            # atom 11 sits (2.0 + 0.1*i) Å from atom 10; frames 2,3 identical (duplicate)
            x = 2.0 + 0.1 * min(i, 2)
            atoms["coordinate"] = [[0, 0, 0], [x * bohr, 0, 0], [10 * bohr, 0, 0]]
            atoms["force"] = [[0.01 * (i + 1), 0, 0], [0, 0.02, 0], [0, 0, 0.5]]
            fh.create_group(f"config_{i}").create_dataset("atoms", data=atoms)

    log = stage / "seg.out"
    log.write_text(
        (Path(__file__).parent.parent / "fixtures" / "cpmd" / "nose_steps.out").read_text()
    )
    # gro ids 900/901 map to dataset ids 10/11 via the OVERLAPS section
    (stage / "run.inp").write_text(
        "&MIMIC\nPATHS\n1\n/x\nOVERLAPS\n2\n2 900 1 10\n2 901 1 11\n&END\n"
    )
    obs = REGISTRY.call("fm_frame_report", ws_ctx, {
        "fmdata": ws_ctx.workspace.relpath(stage / "ref.h5"),
        "monitor_pairs": {"pair": [900, 901], "gone": [10, 999]},
        "cpmd_logs": [ws_ctx.workspace.relpath(log)],
        "overlaps_input": ws_ctx.workspace.relpath(stage / "run.inp"),
    })
    assert obs.ok, obs.summary
    assert obs.n_frames == 4
    assert obs.monitor_stats["pair"]["min"] == pytest.approx(2.0, abs=1e-3)
    assert obs.monitor_stats["pair"]["max"] == pytest.approx(2.2, abs=1e-3)
    assert obs.monitor_missing == {"gone": 4}
    assert obs.n_near_duplicates == 1  # frames 2->3 identical QM coords
    assert obs.md_steps_from_logs and obs.temp_mean_k > 100
    assert obs.segments and obs.segments[0]["scf_max"] >= 15
    csv = (ws_ctx.workspace.root / obs.artifacts[0].path).read_text().splitlines()
    assert csv[0] == "frame,qm_fmax_au,qm_frms_au,qm_rmsd_prev_a,pair,gone"
    assert len(csv) == 5


CPMD_FM_BASE = """\
&MIMIC
PATHS
    1
/tmp/original
BOX
 30 30 30
OVERLAPS
    1
    2 5 1 1
&END
&CPMD
MIMIC
MOLECULAR DYNAMICS BO
&END
&SYSTEM
CUTOFF
 70.
&END
&ATOMS
*C_MT_BLYP KLEINMAN-BYLANDER
 LMAX=P
 1
 1.0 1.0 1.0
&END
"""


def test_restraint_injection_and_prepare():
    from myrmidon.engines.mimic.cpmd_input import (
        DistanceRestraint,
        add_distance_restraints,
        parse_cpmd,
        render_cpmd,
        set_paths,
    )

    sections = parse_cpmd(CPMD_FM_BASE)
    out = add_distance_restraints(
        sections, [DistanceRestraint(atom_i=3, atom_j=7, target_a=2.0, k_au=0.25)]
    )
    text = render_cpmd(out)
    assert "CONSTRAINTS" in text and "END CONSTRAINTS" in text
    assert "RESTRAINTS\n  1\n" in text  # count on its own line (CPMD coninp)
    # 2.0 Å = 3.77945 bohr
    assert "DIST 3 7 3.77945 0.2500" in text
    with pytest.raises(ValueError):
        add_distance_restraints(out, [DistanceRestraint(atom_i=1, atom_j=2, target_a=1.0)])
    moved = set_paths(out, "/scratch/window1")
    assert "/scratch/window1" in render_cpmd(moved)
    assert "/tmp/original" not in render_cpmd(moved)


MINI_GRO = (
    "mini\n    3\n"
    "    1ACT     C1    1   1.000   1.000   1.000\n"
    "    1ACT     O1    2   1.122   1.000   1.000\n"
    "    1SOL     OW    3   2.000   2.000   2.000\n"
    "   3.00000   3.00000   3.00000\n"
)


def test_displace_plan_moves_mobile_atom(ws_ctx):
    import math

    stage = ws_ctx.workspace.stage("prep")
    (stage / "base.gro").write_text(MINI_GRO)
    plan = REGISTRY.call("fm_displace_plan", ws_ctx, {
        "coords": ws_ctx.workspace.relpath(stage / "base.gro"),
        "pair": [1, 2], "pair_label": "C1-O1", "targets_a": [1.10, 1.35]})
    assert plan.ok, plan.summary
    assert plan.original_distance_a == pytest.approx(1.22, abs=1e-3)
    for win, target in zip(plan.windows, (1.10, 1.35), strict=True):
        text = (ws_ctx.workspace.root / win.coords).read_text().splitlines()
        def xyz(line):
            return [float(line[20:28]), float(line[28:36]), float(line[36:44])]
        a, b = xyz(text[2]), xyz(text[3])
        d = math.dist(a, b) * 10
        assert d == pytest.approx(target, abs=2e-2)  # %8.3f gro precision
        assert xyz(text[4]) == xyz(MINI_GRO.splitlines()[4])  # others untouched


def test_fm_merge_h5_sources(ws_ctx):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    stage = ws_ctx.workspace.stage("prep")
    dtype = np.dtype([("id", "i4"), ("region", "i4"), ("coordinate", "f8", (3,)),
                      ("force", "f8", (3,))])
    for name, n in (("a.h5", 3), ("b.h5", 2)):
        with h5py.File(stage / name, "w") as fh:
            for i in range(n):
                atoms = np.zeros(2, dtype=dtype)
                atoms["id"] = [1, 2]
                fh.create_group(f"config_{i}").create_dataset("atoms", data=atoms)
        (stage / name.replace(".h5", ".trr")).write_text(f"TRR{name}\n")
    obs = REGISTRY.call("fm_merge", ws_ctx, {
        "fm_jsons": [ws_ctx.workspace.relpath(stage / "a.h5"),
                     ws_ctx.workspace.relpath(stage / "b.h5")],
        "trajectories": [ws_ctx.workspace.relpath(stage / "a.trr"),
                         ws_ctx.workspace.relpath(stage / "b.trr")]})
    assert obs.ok, obs.summary
    assert obs.frames == 5
    merged = {a.role: a.path for a in obs.artifacts}
    with h5py.File(ws_ctx.workspace.root / merged["fm_dataset"], "r") as fh:
        assert sorted(fh.keys(), key=lambda k: int(k.split("_")[1]))[-1] == "config_4"
    assert "TRRa.h5" in (ws_ctx.workspace.root / merged["trajectory"]).read_text()


def test_fm_displace_plan_rigid_group(ws_ctx):
    from myrmidon.core.tool import REGISTRY
    ws = ws_ctx.workspace
    gro = ws.root / "rigid.gro"
    gro.write_text(
        "t\n"
        "4\n"
        "    1MG      MG    1   1.000   1.000   1.000\n"
        "    1AKG     O3    2   1.200   1.000   1.000\n"
        "    1AKG     C2    3   1.300   1.100   1.000\n"
        "    1SOL     OW    4   2.000   2.000   2.000\n"
        "   5.00000   5.00000   5.00000\n")
    obs = REGISTRY.call("fm_displace_plan", ws_ctx, {
        "coords": "rigid.gro", "pair": [1, 2], "pair_label": "Mg-O3",
        "targets_a": [4.0], "rigid_group": [2, 3], "stage": "rg"})
    assert obs.ok, obs.summary
    out = (ws.root / obs.windows[0].coords).read_text().splitlines()
    def xyz(i):
        ln = out[1 + i]
        return [float(ln[20:28]), float(ln[28:36]), float(ln[36:44])]
    o3, c2, ow = xyz(2), xyz(3), xyz(4)
    # mobile at target 4 A = 0.4 nm along +x from anchor
    assert abs(o3[0] - 1.4) < 1e-6 and abs(o3[1] - 1.0) < 1e-6
    # group partner translated by the SAME vector (internal geometry preserved)
    assert abs(c2[0] - 1.5) < 1e-6 and abs(c2[1] - 1.1) < 1e-6
    d_int = sum((a - b) ** 2 for a, b in zip(o3, c2, strict=True)) ** 0.5
    assert abs(d_int - (0.02 ** 0.5)) < 1e-6      # unchanged internal distance
    # non-group atom untouched
    assert ow == [2.0, 2.0, 2.0]
    # guards
    bad = REGISTRY.call("fm_displace_plan", ws_ctx, {
        "coords": "rigid.gro", "pair": [1, 2], "pair_label": "x",
        "targets_a": [4.0], "rigid_group": [3], "stage": "rg2"})
    assert not bad.ok and "must include the mobile" in bad.summary
    bad = REGISTRY.call("fm_displace_plan", ws_ctx, {
        "coords": "rigid.gro", "pair": [1, 2], "pair_label": "x",
        "targets_a": [4.0], "rigid_group": [1, 2], "stage": "rg3"})
    assert not bad.ok and "anchor" in bad.summary


def test_parse_validation_force_residuals():
    from myrmidon.engines.mimic.fm import parse_validation_force_residuals

    text = (
        "frame_index,gmx_id,name,type,resname,resid,element,is_bound,"
        "res_x,res_y,res_z,ref_x,ref_y,ref_z\n"
        "3,1,C1,c,ACT,1,C,0,3,4,0,2,2,2\n"
        "3,2,O1,o,ACT,1,O,0,0,0,1,2,2,2\n"
        "3,3,H1,hc,ACT,1,H,0,0,0,0,2,2,2\n"
        "9,1,C1,c,ACT,1,C,0,0,0,0,2,2,2\n"
        "9,2,O1,o,ACT,1,O,0,0,0,1,2,2,2\n"
        "9,3,H1,hc,ACT,1,H,0,2,0,0,2,2,2\n"
    )
    f = parse_validation_force_residuals(text)
    assert f["n_frames"] == 2 and f["n_atoms"] == 3 and f["n_samples"] == 6
    # |res|^2 per sample: 25,1,0,0,1,4 -> rmse sqrt(31/6); ref |.|^2 = 12 every sample
    assert f["rmse"] == pytest.approx((31 / 6) ** 0.5, abs=1e-4)
    assert f["ref_rms"] == pytest.approx(12 ** 0.5, abs=1e-4)
    by = {t["type"]: t for t in f["by_type"]}
    assert by["c"]["rmse"] == pytest.approx(12.5 ** 0.5, abs=1e-4)
    assert by["o"]["rmse"] == 1.0 and by["hc"]["rmse"] == pytest.approx(2 ** 0.5, abs=1e-4)
    assert by["c"]["rel_rmse"] == pytest.approx((12.5 / 12) ** 0.5, abs=1e-4)
    assert [t["type"] for t in f["by_type"]] == ["c", "hc", "o"]
    assert f["worst_atoms"][0]["name"] == "C1" and f["outliers"] == []
    # one atom far above the rest is an outlier
    text2 = text.replace("3,1,C1,c,ACT,1,C,0,3,4,0", "3,1,C1,c,ACT,1,C,0,30,40,0")
    f2 = parse_validation_force_residuals(text2)
    assert [a["name"] for a in f2["outliers"]] == ["C1"]
    assert parse_validation_force_residuals("frame_index,gmx_id\n") == {}


def test_parse_real_validation_force_residuals():
    """Fixture written by mimicpy's own compute_validation_sdf on the acetone QM region
    (synthetic frames, real writer); the per-type RMSE it logged alongside:
    o 647.50, hc 517.06, c 415.54, c3 347.06 (Hartree/Bohr; synthetic magnitudes)."""
    from myrmidon.engines.mimic.fm import parse_validation_force_residuals

    f = parse_validation_force_residuals((FIXTURES / "validation_force_residuals.csv").read_text())
    assert f["n_frames"] == 2 and f["n_atoms"] == 10 and f["n_samples"] == 20
    by = {t["type"]: t for t in f["by_type"]}
    assert set(by) == {"c", "o", "hc", "c3"}
    assert by["o"]["rmse"] == pytest.approx(647.50, abs=0.01)
    assert by["hc"]["rmse"] == pytest.approx(517.06, abs=0.01)
    assert by["c"]["rmse"] == pytest.approx(415.54, abs=0.01)
    assert by["c3"]["rmse"] == pytest.approx(347.06, abs=0.01)
    assert by["hc"]["n_atoms"] == 6 and by["hc"]["element"] == "H"
    assert f["worst_atoms"][0]["resname"] == "ACT"


FM_CANDIDATES = __import__("pathlib").Path(__file__).parent.parent / "fixtures" / "fm_candidates"



def _top(bonds="1 2 1 0.123 300000", angles="1 2 3 1 109.5 400", charge="0.100"):
    return (f"[ atoms ]\n 1 c 1 ACT C1 1 {charge} 12.0\n"
            f"[ bonds ]\n {bonds}\n[ angles ]\n {angles}\n")


def _metal(q):
    return (f"[ atoms ]\n 1 MG 1 MG MG 1 {q} 24.3\n"
            " 2 c 2 ACT C1 2 0.100 12.0\n[ bonds ]\n[ angles ]\n")


def _candidate(name):
    return (FM_CANDIDATES / f"{name}.itp").read_text()


# (fitted topology, the one it was fitted from or None, the findings). One synthetic
# topology per finding, so a threshold cannot quietly stop working; the two that need a
# baseline are skipped rather than guessed without one. A metal at its formal charge is
# not a fit artefact — only a fit that moved it is. And a real pair from an acetone fit:
# a naive fit whose three carbonyl angles collapsed to 0.002 degrees and still scored
# well, beside the regularised fit that did not.
PLAUSIBILITY = [
    (_top(), None, set()),
    (_top(angles="1 2 3 1 0.002 400"), None, {"FM_ANGLE_COLLAPSED"}),
    (_top(angles="1 2 3 1 179.9 400"), None, {"FM_ANGLE_COLLAPSED"}),
    (_top(angles="1 2 3 1 109.5 -9e5"), None, {"FM_UNPHYSICAL_PARAMS"}),
    (_top(bonds="1 2 1 0.9 3e5"), None, {"FM_BOND_LENGTH_IMPLAUSIBLE"}),
    (_top(charge="-3.5"), None, {"FM_CHARGE_IMPLAUSIBLE"}),
    (_top(bonds="1 2 1 0.123 30"), None, set()),
    (_top(bonds="1 2 1 0.123 30"), _top(), {"FM_FORCE_CONSTANT_IMPLAUSIBLE"}),
    (_top(charge="1.900"), _top(), {"FM_NET_CHARGE_DRIFTED"}),
    (_metal("2.000"), _metal("2.000"), set()),
    (_metal("2.010"), _metal("2.000"), set()),
    (_metal("2.800"), _metal("2.000"), {"FM_CHARGE_IMPLAUSIBLE", "FM_NET_CHARGE_DRIFTED"}),
    (_metal("2.800"), None, {"FM_CHARGE_IMPLAUSIBLE"}),
    (_candidate("good_reg"), _candidate("original"), set()),
    (_candidate("bad_naive"), _candidate("original"), {"FM_ANGLE_COLLAPSED"}),
]


@pytest.mark.parametrize("fitted,original,findings", PLAUSIBILITY,
                         ids=[f"{i}-{'+'.join(sorted(c[2])) or 'clean'}"
                              for i, c in enumerate(PLAUSIBILITY)])
def test_a_fitted_parameter_that_is_not_chemistry_is_a_finding(fitted, original, findings):
    from myrmidon.engines.mimic.fm import PARAMETER_FINDINGS, implausible_parameters

    found = implausible_parameters(fitted, original) if original else \
        implausible_parameters(fitted)
    assert set(found) == findings
    assert findings <= set(PARAMETER_FINDINGS)  # every finding has its explanation


def test_negative_force_constants_are_listed_and_a_collapsed_angle_is_not_one():
    """A check that reads the force-constant column only lists negative constants and
    sees nothing wrong with the collapsed fit — whose three angles the plausibility check
    above does find."""
    from myrmidon.engines.mimic.fm import implausible_parameters, unphysical_force_constants

    itp = ("[ atoms ]\n 1 c3 1 ACT C1 1 0.8 12.0\n[ bonds ]\n"
           "   2 5 1 0.1091966 -910570.31\n   1 4 1 0.1229 476976.0\n[ angles ]\n"
           "   1 2 5 1 110.01 -763.71 ; fitted\n   2 1 3 1 116.0 543.0\n")
    assert unphysical_force_constants(itp) == ["bonds 2 5: k = -910570",
                                               "angles 1 2 5: k = -763.71"]
    assert unphysical_force_constants(_candidate("bad_naive")) == []
    collapsed = implausible_parameters(_candidate("bad_naive"), _candidate("original"))
    assert len(collapsed["FM_ANGLE_COLLAPSED"]) == 3


# --- the snapshot campaign: frames, dataset, fit, candidate topology ---------------


def _write_trr(path, structure, n_frames=5):
    mda = pytest.importorskip("MDAnalysis")
    import numpy as np

    u = mda.Universe(str(structure))
    with mda.Writer(str(path), n_atoms=len(u.atoms)) as w:
        for i in range(n_frames):
            u.atoms.positions = u.atoms.positions + np.array([0.1, 0.0, 0.0])
            u.trajectory.ts.time = 10.0 * i
            w.write(u.atoms)


def test_traj_frames_picks_evenly_spaced_snapshots(ws_ctx):
    ws = ws_ctx.workspace
    inputs = ws.stage("inputs")
    (inputs / "ref.gro").write_text(MINI_GRO)
    _write_trr(inputs / "ref.trr", inputs / "ref.gro", n_frames=5)
    obs = REGISTRY.call("traj_frames", ws_ctx, {
        "structure": "01_inputs/ref.gro", "trajectory": "01_inputs/ref.trr", "n_frames": 3,
    })
    assert obs.ok, obs.summary
    assert obs.n_total == 5 and [f.index for f in obs.frames] == [0, 2, 4]
    assert [f.time_ps for f in obs.frames] == [0.0, 20.0, 40.0]
    for f in obs.frames:
        assert (ws.root / f.path).exists() and f.path.startswith("02_frames/frame_")
    assert {a.role for a in obs.artifacts} == {"structure"}
    explicit = REGISTRY.call("traj_frames", ws_ctx, {
        "structure": "01_inputs/ref.gro", "trajectory": "01_inputs/ref.trr", "frames": [1, 9],
    })
    assert not explicit.ok and "out of range [9]" in explicit.summary


def test_the_candidate_topology_swaps_the_fitted_itp(ws_ctx):
    """Assembled by the fit's collection (job_status on an fm_fit job) — a fitted
    opt_<name>.itp replaces the include <name>.itp of the topology the fit started from."""
    from myrmidon.engines.mimic.fm import _fitted_pairs, assemble_candidate

    ws = ws_ctx.workspace
    inputs = ws.stage("inputs")
    (inputs / "acetone.itp").write_text("; stock\n")
    (inputs / "system.top").write_text(
        '#include "amber99sb-ildn.ff/forcefield.itp"\n#include "acetone.itp"\n'
        "[ system ]\nacetone\n[ molecules ]\nACT 1\n"
    )
    fit = ws.stage("fit")
    (fit / "opt_acetone.itp").write_text("; fitted\n")
    (fit / "opt_ACT.itp").write_text("; fitted, but named after nothing the topology includes\n")
    pairs = _fitted_pairs(inputs / "system.top", [fit / "opt_acetone.itp", fit / "opt_ACT.itp"])
    assert pairs == {"acetone.itp": fit / "opt_acetone.itp"}
    obs = assemble_candidate(ws_ctx, fit, inputs / "system.top", pairs)
    assert obs.ok, obs.summary
    cand = ws.root / next(a.path for a in obs.artifacts if a.role == "topology")
    text = cand.read_text()
    assert '#include "opt_acetone.itp"' in text and '"acetone.itp"' not in text
    assert (cand.parent / "opt_acetone.itp").read_text() == "; fitted\n"
    assert (cand.parent / "acetone.itp").exists()  # staged include, untouched
    assert (inputs / "system.top").read_text().count("acetone.itp") == 1  # original untouched
    bad = assemble_candidate(ws_ctx, fit, inputs / "system.top", {"nope.itp": fit / "opt_ACT.itp"})
    assert not bad.ok and "not #include lines" in bad.summary


def test_fm_fit_refuses_a_mimic_rerun_mdp(ws_ctx):
    """A rerun mdp that still carried qmmm_grps cost a two-hour queue
    wait before mimicpy refused it. The gate refuses first."""
    ws = ws_ctx.workspace
    prep = ws.stage("prep")
    for name in ("t.top", "c.gro", "d.h5", "t.trr", "i.ndx", "s.dat"):
        (prep / name).write_text("x\n")
    (prep / "fminput.dat").write_text("stride: [0,1,1]\nqm_total_charge: 0\n")
    (prep / "bad.mdp").write_text("integrator = md\nnstfout = 1\nQMMM-grps = QMatoms\n")
    (prep / "good.mdp").write_text("integrator = md\nnstfout = 1\n")
    base = {"topology": "01_prep/t.top", "coords": "01_prep/c.gro", "fmdata": "01_prep/d.h5",
            "trajectory": "01_prep/t.trr", "index": "01_prep/i.ndx", "selection": "01_prep/s.dat",
            "fm_input": "01_prep/fminput.dat"}
    obs = REGISTRY.call("fm_fit", ws_ctx, {**base, "mdp": "01_prep/bad.mdp"})
    assert not obs.ok and [w.id for w in obs.blocking] == ["FM_RERUN_MDP_IS_MIMIC"]
    assert "qmmm_grps = QMatoms" in obs.blocking[0].message
    obs = REGISTRY.call("fm_fit", ws_ctx, {**base, "mdp": "01_prep/good.mdp"})
    assert obs.ok, obs.summary


def test_fm_merge_converts_each_same_named_json_separately(ws_ctx):
    """Three FMTRAJECTORY.json windows merged onto an h5 base
    became three copies of the last window (one stem-named conversion file)."""
    h5py = pytest.importorskip("h5py")
    import numpy as np

    ws = ws_ctx.workspace
    prep = ws.stage("prep")
    dtype = np.dtype([("id", "i4"), ("region", "i4"), ("coordinate", "f8", (3,)),
                      ("force", "f8", (3,))])
    with h5py.File(prep / "base.h5", "w") as fh:
        atoms = np.zeros(1, dtype=dtype)
        fh.create_group("config_0").create_dataset("atoms", data=atoms)
    (prep / "base.trr").write_text("TRRbase\n")
    for w in ("w1", "w2"):
        d = ws.stage(w)
        (d / "FMTRAJECTORY.json").write_text("{}\n")
        (d / "mimic.trr").write_text(f"TRR{w}\n")
    obs = REGISTRY.call("fm_merge", ws_ctx, {
        "fm_jsons": ["01_prep/base.h5", "02_w1/FMTRAJECTORY.json", "03_w2/FMTRAJECTORY.json"],
        "trajectories": ["01_prep/base.trr", "02_w1/mimic.trr", "03_w2/mimic.trr"],
    })
    assert obs.ok, obs.summary
    assert obs.frames == 1 + 2 + 2  # base + two mock windows of 2 configs each
    stage = ws.root / next(a.path for a in obs.artifacts if a.role == "fm_dataset")
    convs = sorted(p.name for p in stage.parent.glob("*_conv.h5"))
    assert convs == ["001_02_w1_FMTRAJECTORY_conv.h5", "002_03_w2_FMTRAJECTORY_conv.h5"]


def test_fm_collect_checks_across_the_dataset_and_its_tail(ws_ctx):
    ws = ws_ctx.workspace
    prep = ws.stage("prep")
    for name in ("d.h5", "t.trr", "t.top", "c.gro", "s.dat"):
        (prep / name).write_text("x\n")
    obs = REGISTRY.call("fm_collect", ws_ctx, {
        "fmdata": "01_prep/d.h5", "trajectory": "01_prep/t.trr", "topology": "01_prep/t.top",
        "coords": "01_prep/c.gro", "selection": "01_prep/s.dat", "check_frames": 4,
    })
    assert obs.ok, obs.summary
    log = (ws.root / "02_fm" / "fmcheck.log").read_text()
    assert "### fmcheck strided: -b 0 -e 12 -s 3" in log  # the mock reports 12 configs
    assert "### fmcheck tail: -b 9 -e 12" in log


def test_fm_results_registers_every_fitted_itp(ws_ctx):
    """A metalloenzyme fit wrote five opt_*.itp (ligands, two protein chains,
    the ion chain, water); only the first comma-separated token was parsed."""
    ws = ws_ctx.workspace
    stage = ws.stage("fit")
    (stage / "fm.log").write_text(
        "Number of snapshots used for fitting: 1106\n"
        "Successfully wrote .itp files for molecules containing QM atoms: "
        "non_bonded_ligands.itp, non_bonded_tip3p.itp\n"
        "Successfully updated topology with optimized QM interaction parameters\n"
        "Successfully wrote .itp files for molecules containing QM atoms: opt_ligands.itp, "
        "opt_topol_Protein_chain_A.itp, opt_tip3p.itp\n"
        "MiMiCPy Force matching workflow completed\nwall time: 0:23:26\n"
    )
    for name in ("opt_ligands.itp", "opt_topol_Protein_chain_A.itp", "opt_tip3p.itp"):
        (stage / name).write_text("[ atoms ]\n 1 MG 1 MG MG 1 1.9 24.3\n")
    (ws.root / "ligands.itp").write_text("[ atoms ]\n 1 MG 1 MG MG 1 2.0 24.3\n")
    from myrmidon.engines.mimic.fm import collect_fit

    obs = collect_fit(ws_ctx, stage, original_itp="ligands.itp")
    assert obs.ok, obs.summary
    fitted = sorted(a.path for a in obs.artifacts if a.role == "fitted_topology")
    assert [p.split("/")[-1] for p in fitted] == [
        "opt_ligands.itp", "opt_tip3p.itp", "opt_topol_Protein_chain_A.itp"]
    # charge deltas against the fitted twin of the original (opt_ligands.itp)
    assert obs.n_charges == 1 and abs(obs.max_abs_charge_delta - 0.1) < 1e-9
