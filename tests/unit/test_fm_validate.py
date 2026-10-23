"""fm_validate: driver math, end-to-end driver run (stub gmx), tool round trip."""

import json
import stat
from pathlib import Path

import numpy as np
import pytest

from myrmidon.core.tool import REGISTRY
from myrmidon.engines.mimic.validate import collect_validation
from myrmidon.engines.mimic.validate_driver import (
    main as driver_main,
)
from myrmidon.engines.mimic.validate_driver import (
    pair_distance,
    read_gro,
)

GRO = """three atoms
3
    1SOL     OW    1   0.100   0.100   0.100
    1SOL    HW1    2   0.150   0.100   0.100
    2ION     MG    3   0.950   0.100   0.100
   1.00000   1.00000   1.00000
"""


def _write_gro(path: Path) -> Path:
    path.write_text(GRO)
    return path


def test_read_gro_and_min_image(tmp_path):
    coords, box = read_gro(_write_gro(tmp_path / "s.gro"))
    assert coords.shape == (3, 3) and np.allclose(box, 1.0)
    # direct distance OW-HW1: 0.05 nm = 0.5 A
    assert pair_distance(coords, box, 1, 2) == pytest.approx(0.5)
    # OW-MG wraps through the boundary: 0.15 nm, not 0.85 nm
    assert pair_distance(coords, box, 1, 3) == pytest.approx(1.5)


@pytest.fixture
def stub_gmx(tmp_path, monkeypatch):
    """A fake gmx: grompp copies -c to a sentinel; mdrun 'minimizes' by
    copying that structure to <deffnm>.gro shifted so distances change."""
    mda = pytest.importorskip("MDAnalysis")  # noqa: F841
    script = tmp_path / "bin" / "gmx"
    script.parent.mkdir()
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "args = sys.argv[1:]\n"
        "def val(flag):\n"
        "    return args[args.index(flag) + 1]\n"
        "if args[0] == 'grompp':\n"
        "    shutil.copy(val('-c'), val('-o') + '.sourcegro')\n"
        "elif args[0] == 'mdrun':\n"
        "    src = val('-s') + '.sourcegro'\n"
        "    shutil.copy(src, val('-deffnm') + '.gro')\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def _make_reference(tmp_path):
    """Structure + 4-frame trr where the OW-HW1 distance is constant 0.5 A."""
    mda = pytest.importorskip("MDAnalysis")
    gro = _write_gro(tmp_path / "ref.gro")
    u = mda.Universe(str(gro))
    trr = tmp_path / "ref.trr"
    with mda.Writer(str(trr), n_atoms=3) as w:
        for i in range(4):
            u.trajectory.ts.time = float(i)
            w.write(u.atoms)
    return gro, trr


def test_driver_min_geometry_end_to_end(tmp_path, stub_gmx):
    gro, trr = _make_reference(tmp_path)
    top = tmp_path / "cand" / "topol.top"
    top.parent.mkdir()
    top.write_text("; stub topology\n")
    stage = tmp_path / "stage"
    stage.mkdir()
    params = {
        "stage": str(stage),
        "structure": str(gro),
        "trajectory": str(trr),
        "candidates": {"cand": str(top)},
        "monitor_pairs": {"OW-HW1": [1, 2], "OW-MG": [1, 3]},
        "metrics": ["min_geometry"],
        "min_frames": [0, 2],
        "n_min_frames": 2,
        "md_trajectories": {},
        "md_skip_ps": 0.0,
        "reference_stride": 1,
        "em": {"nsteps": 10, "emtol": 100.0},
        "gmx": stub_gmx,
        "ntomp": 1,
        "out": "validate_results.json",
    }
    pfile = stage / "p.json"
    pfile.write_text(json.dumps(params))
    assert driver_main(str(pfile)) == 0
    result = json.loads((stage / "validate_results.json").read_text())
    assert result["ok"], result
    assert result["reference"]["OW-HW1"]["mean"] == pytest.approx(0.5, abs=1e-3)
    block = result["min_geometry"]["cand"]
    assert block["n_frames_minimized"] == 2
    # the stub "minimizer" returns the input frame, so MAE vs reference ~ 0
    assert block["mae"] == pytest.approx(0.0, abs=1e-3)


def test_driver_md_geometry(tmp_path):
    gro, trr = _make_reference(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    params = {
        "stage": str(stage),
        "structure": str(gro),
        "trajectory": str(trr),
        "candidates": {},
        "monitor_pairs": {"OW-HW1": [1, 2]},
        "metrics": ["md_geometry"],
        "min_frames": [0],
        "n_min_frames": 1,
        "md_trajectories": {"cand": [str(trr)]},
        "md_skip_ps": 1.5,       # skips frames 0 and 1 of the 4
        "reference_stride": 1,
        "em": {"nsteps": 10, "emtol": 100.0},
        "gmx": "gmx-not-needed",
        "ntomp": 1,
        "out": "validate_results.json",
    }
    pfile = stage / "p.json"
    pfile.write_text(json.dumps(params))
    assert driver_main(str(pfile)) == 0
    result = json.loads((stage / "validate_results.json").read_text())
    block = result["md_geometry"]["cand"]
    assert block["n_frames_used"] == 2
    assert block["pairs"]["OW-HW1"]["mean"] == pytest.approx(0.5, abs=1e-3)
    assert block["mae"] == pytest.approx(0.0, abs=1e-3)


def test_fm_validate_results_parses_and_ranks(ws_ctx):
    stage = ws_ctx.workspace.stage("validate")
    data = {
        "ok": True,
        "reference": {"p1": {"mean": 2.1, "std": 0.07}},
        "reference_frames": 100,
        "min_frames": [10],
        "min_geometry": {
            "stock": {"n_frames_requested": 1, "n_frames_minimized": 1,
                      "pairs": {"p1": {"mean": 1.9}}, "mae": 0.2},
            "fitted": {"n_frames_requested": 1, "n_frames_minimized": 1,
                       "pairs": {"p1": {"mean": 2.05}}, "mae": 0.05},
        },
    }
    (stage / "validate_results.json").write_text(json.dumps(data))
    obs = collect_validation(ws_ctx, stage)  # what job_status returns for the ended job
    assert obs.ok, obs.summary
    assert obs.mae["min_geometry"] == {"stock": 0.2, "fitted": 0.05}
    # ranked best-first in the summary
    assert obs.summary.index("fitted=0.050") < obs.summary.index("stock=0.200")


# -- force_rmse: candidate rerun forces vs the dataset's QM/MM forces ---------------

ACETONE = Path(__file__).parent.parent / "fixtures" / "acetone"


@pytest.fixture
def force_stub_gmx(tmp_path):
    """grompp touches the tpr; mdrun -rerun copies the trr named by MOCK_RERUN_TRR
    to -o (the test prepares a rerun trr with known forces)."""
    script = tmp_path / "bin" / "gmx"
    script.parent.mkdir(exist_ok=True)
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import os, shutil, sys\n"
        "args = sys.argv[1:]\n"
        "def val(flag):\n"
        "    return args[args.index(flag) + 1]\n"
        "if args[0] == 'grompp':\n"
        "    open(val('-o'), 'w').write('tpr')\n"
        "elif args[0] == 'mdrun':\n"
        "    shutil.copy(os.environ['MOCK_RERUN_TRR'], val('-o'))\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def _acetone_case(tmp_path, monkeypatch):
    """Real mimicpy QM region on the acetone fixture; synthetic h5 reference forces in
    CPMD order (reversed, to exercise the mapping); a reference trr; a prepared rerun trr
    whose forces are reference + a known residual on the carbonyl carbon only."""
    import shutil

    h5py = pytest.importorskip("h5py")
    mda = pytest.importorskip("MDAnalysis")
    mimicpy = pytest.importorskip("mimicpy")  # noqa: F841
    from mimicpy.force_matching.qm_region import QMRegion
    from mimicpy.utils.constants import kjmolnm_to_au

    work = tmp_path / "case"
    shutil.copytree(ACETONE, work)
    top, gro, sele = work / "acetone_equilibrated.top", work / "acetone_equilibrated.gro", \
        work / "sele.dat"
    q = QMRegion(str(top), str(gro))
    q.setup_qm_region(sele)
    qm_ids = [int(g) for g in q.qm_atoms.index]  # GROMACS ids, topology order
    n_qm = len(qm_ids)
    ref = np.array([[0.01 * (i + 1), -0.005 * (i + 1), 0.002] for i in range(n_qm)])
    residual = np.zeros((n_qm, 3))
    carbonyl = [i for i, g in enumerate(qm_ids) if q.qm_atoms.loc[g, "type"] == "c"]
    residual[carbonyl[0]] = [0.003, 0.004, 0.0]  # |res| = 0.005 a.u. on the C=O carbon
    n_frames = 5
    # h5: 'atoms' compound dataset, QM rows in reversed CPMD order plus two MM atoms
    dtype = [("id", "i4"), ("region", "i4"), ("coordinate", "f8", (3,)),
             ("electric_potential", "f8"), ("electric_field", "f8", (3,)),
             ("hirshfeld_charge", "f8"), ("force", "f8", (3,))]
    cpmd_of = q.gmx_to_cpmd_map
    order = list(range(n_qm))[::-1]
    h5 = work / "ref.h5"
    with h5py.File(h5, "w") as f:
        for i in range(n_frames):
            data = np.zeros(n_qm + 2, dtype=dtype)
            for row, a in enumerate(order):
                data[row] = (cpmd_of[qm_ids[a]], 1, [0, 0, 0], 0.0, [0, 0, 0], 0.0, ref[a])
            data[n_qm] = (9001, 2, [1, 1, 1], 0.0, [0, 0, 0], 0.0, [0, 0, 0])
            data[n_qm + 1] = (9002, 2, [2, 2, 2], 0.0, [0, 0, 0], 0.0, [0, 0, 0])
            f.create_group(f"config_{i}").create_dataset("atoms", data=data)
    u = mda.Universe(str(gro))
    ref_trr = work / "ref.trr"
    with mda.Writer(str(ref_trr), n_atoms=len(u.atoms)) as w:
        for i in range(n_frames):
            u.trajectory.ts.time = float(i)
            w.write(u.atoms)
    # rerun trr: 3 frames with forces (GROMACS units kJ/mol/Å) = (ref + residual)/conversion
    forces = np.zeros((len(u.atoms), 3))
    for a, g in enumerate(qm_ids):
        forces[g - 1] = (ref[a] + residual[a]) / kjmolnm_to_au / 10.0
    rerun_trr = work / "prepared_rerun.trr"
    u.trajectory.ts.has_forces = True
    with mda.Writer(str(rerun_trr), n_atoms=len(u.atoms)) as w:
        for k in range(3):
            u.trajectory.ts.forces = forces
            u.trajectory.ts.time = float(k)
            w.write(u.atoms)
    monkeypatch.setenv("MOCK_RERUN_TRR", str(rerun_trr))
    (work / "rerun.mdp").write_text("integrator = md\nnstfout = 1\n")
    return work, top, gro, sele, h5, ref_trr, ref, residual, q


def test_driver_force_rmse_end_to_end(tmp_path, monkeypatch, force_stub_gmx):
    work, top, gro, sele, h5, ref_trr, ref, residual, q = _acetone_case(tmp_path, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir()
    params = {
        "stage": str(stage), "structure": str(gro), "trajectory": str(ref_trr),
        "candidates": {"cand": str(top)}, "monitor_pairs": {"C1-O1": [1, 2]},
        "metrics": ["force_rmse"], "min_frames": [], "n_min_frames": 2,
        "md_trajectories": {}, "md_skip_ps": 0.0, "reference_stride": 1,
        "em": {"nsteps": 10, "emtol": 100.0}, "gmx": force_stub_gmx, "ntomp": 1,
        "fmdata": str(h5), "selection": str(sele), "rerun_mdp": str(work / "rerun.mdp"),
        "index": None, "force_frames": [0, 2, 4], "n_force_frames": 3, "gmx_data": None,
        "out": "validate_results.json",
    }
    pfile = stage / "p.json"
    pfile.write_text(json.dumps(params))
    assert driver_main(str(pfile)) == 0
    result = json.loads((stage / "validate_results.json").read_text())
    assert result["ok"], result
    assert result["force_frames"] == [0, 2, 4]
    block = result["force_rmse"]["cand"]
    assert block["n_frames"] == 3 and block["n_atoms"] == len(q.qm_atoms)
    assert block["units"] == "Hartree/Bohr"
    by = {t["type"]: t for t in block["by_type"]}
    # only the carbonyl carbon (type c, one atom) carries a residual of |0.005|
    assert by["c"]["rmse"] == pytest.approx(0.005, abs=1e-5)
    for t in ("o", "c3", "hc"):
        assert by[t]["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert by["c"]["ref_rms"] > 0
    n_qm = len(q.qm_atoms)
    assert block["rmse"] == pytest.approx((0.005 ** 2 / n_qm) ** 0.5, abs=1e-5)
    assert [a["type"] for a in block["outliers"]] == ["c"]
    # the residual table is on disk in mimicpy's layout, with mapping through CPMD ids
    csv_text = (stage / block["file"]).read_text().splitlines()
    assert csv_text[0].startswith("frame_index,gmx_id,name,type,resname")
    assert len(csv_text) == 1 + 3 * n_qm
    assert (stage / "force_frames.trr").exists()


def test_fm_validate_force_rmse_gate_and_params(ws_ctx, monkeypatch):
    ws = ws_ctx.workspace
    _write_gro(ws.root / "ref.gro")
    (ws.root / "ref.trr").write_bytes(b"not a real trr")
    (ws.root / "cand.top").write_text("; top\n")
    (ws.root / "ref.h5").write_bytes(b"h5")
    (ws.root / "sele.dat").write_text("resname is ACT\n")
    (ws.root / "mimic.mdp").write_text("integrator = mimic\nQMMM-grps = QMatoms\nnstfout = 1\n")
    (ws.root / "rerun.mdp").write_text("integrator = md\nnstfout = 1\n")
    base = {"structure": "ref.gro", "trajectory": "ref.trr", "topologies": {"cand": "cand.top"},
            "monitor_pairs": {"a-b": (1, 2)}, "metrics": ["force_rmse"]}
    missing = REGISTRY.call("fm_validate", ws_ctx, base)
    assert not missing.ok
    assert {w.id for w in missing.blocking} >= {"INPUT_MISSING_FMDATA", "INPUT_MISSING_SELECTION",
                                                "INPUT_MISSING_RERUN_MDP"}
    mimic = REGISTRY.call("fm_validate", ws_ctx, {
        **base, "fmdata": "ref.h5", "selection": "sele.dat", "rerun_mdp": "mimic.mdp"})
    assert not mimic.ok and "FM_RERUN_MDP_IS_MIMIC" in {w.id for w in mimic.blocking}

    captured = {}
    monkeypatch.setattr(ws_ctx.backend, "submit", lambda spec: captured.setdefault("spec", spec)
                        and "h1")
    obs = REGISTRY.call("fm_validate", ws_ctx, {
        **base, "fmdata": "ref.h5", "selection": "sele.dat", "rerun_mdp": "rerun.mdp",
        "force_frames": [0, 5, 9]})
    assert obs.ok, obs.summary
    p = json.loads((ws.stage("validate") / "validate_params.json").read_text())
    assert p["metrics"] == ["force_rmse"] and p["force_frames"] == [0, 5, 9]
    assert p["fmdata"].endswith("ref.h5") and p["rerun_mdp"].endswith("rerun.mdp")


def test_unset_metrics_follow_the_inputs_given(ws_ctx):
    """A fixed default of min_geometry blocks without monitor_pairs, refusing a call that
    carries everything force_rmse needs for a metric it never asked for. Unset, the
    metric follows the inputs, and the gate says so."""
    from myrmidon.engines.mimic.validate import FMValidateParams, fm_validate_gate

    common = {"structure": "01_input/ref.gro", "trajectory": "01_input/ref.trr",
              "topologies": {"a": "01_input/a.top"}}
    forces = FMValidateParams(**common, fmdata="01_input/r.h5")
    assert forces.metrics == ["force_rmse"]
    warnings, _ = fm_validate_gate(ws_ctx, forces)
    ids = {w.id: w for w in warnings}
    assert "INPUT_MISSING_MONITOR_PAIRS" not in ids
    assert ids["METRICS_FROM_INPUTS"].severity == "info"
    assert "force_rmse" in ids["METRICS_FROM_INPUTS"].message

    assert FMValidateParams(**common, monitor_pairs={"a-b": (1, 2)}).metrics == ["min_geometry"]
    assert FMValidateParams(**common, fmdata="r.h5", monitor_pairs={"a-b": (1, 2)}
                            ).metrics == ["force_rmse", "min_geometry"]
    # a metric named is a metric chosen: no assumption to state
    named = FMValidateParams(**common, metrics=["force_rmse"], fmdata="01_input/r.h5")
    assert "METRICS_FROM_INPUTS" not in {w.id for w in fm_validate_gate(ws_ctx, named)[0]}

    neither = FMValidateParams(**common)
    warnings, _ = fm_validate_gate(ws_ctx, neither)
    assert [w.id for w in warnings] == ["INPUT_MISSING_METRICS"]
    assert "force_rmse" in warnings[0].suggestion and "monitor_pairs" in warnings[0].suggestion

