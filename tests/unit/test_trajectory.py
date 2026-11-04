"""`traj_analyze`: a trajectory metric is internal motion, under a stated preprocessing.

A fluctuation that counts the box drifting through space, or a molecule the box split in
two, is a number nobody asked for. The parser's promises: rigid-body motion is removed
before rmsd and rmsf; a split molecule is made whole only when asked and only when the
structure carries bonds; a profile over atoms is not a series over frames.
"""

import numpy as np
import pytest

from myrmidon.core.tool import REGISTRY

mda = pytest.importorskip("MDAnalysis")

RIGID_GRO = """\
rigid body
    4
    1MOL      C    1   1.000   1.000   1.000
    1MOL      O    2   1.150   1.000   1.000
    1MOL      H    3   1.000   1.120   1.000
    1MOL      N    4   1.000   1.000   1.090
   5.00000   5.00000   5.00000
"""
PAIR_PDB = """\
CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
ATOM      1  C1  MOL A   1       1.000   1.000   1.000  1.00  0.00           C
ATOM      2  C2  MOL A   1       3.000   1.000   1.000  1.00  0.00           C
CONECT    1    2
END
"""
PAIR_GRO = """\
pair without bonds
    2
    1MOL     C1    1   0.100   0.100   0.100
    1MOL     C2    2   0.300   0.100   0.100
   2.00000   2.00000   2.00000
"""


def _write_frames(path, structure, frames):
    """A trajectory of the given positions (Å), one frame per ps."""
    u = mda.Universe(str(structure))
    with mda.Writer(str(path), n_atoms=len(u.atoms)) as w:
        for i, positions in enumerate(frames):
            u.atoms.positions = positions
            u.trajectory.ts.time = float(i)
            w.write(u.atoms)


def _rigid_frames(structure, n=10):
    """The same body rotated about z by 20° per frame and carried 1 Å per frame along x."""
    base = mda.Universe(str(structure)).atoms.positions.copy()
    centre = base.mean(axis=0)
    frames = []
    for i in range(n):
        a = np.deg2rad(20.0 * i)
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        frames.append((base - centre) @ rot.T + centre + np.array([float(i), 0.0, 0.0]))
    return frames


@pytest.fixture
def rigid(ws_ctx):
    stage = ws_ctx.workspace.stage("input")
    (stage / "rigid.gro").write_text(RIGID_GRO)
    _write_frames(stage / "rigid.trr", stage / "rigid.gro", _rigid_frames(stage / "rigid.gro"))
    return ws_ctx


# (metric, extra parameters, axis, n_points, n_frames, has a final value)
RIGID = [
    ("rmsf", {}, "atom_index", 4, 10, False),
    ("rmsd", {}, "time_ps", 10, 10, True),
    ("rmsf", {"start_ps": 5}, "atom_index", 4, 5, False),
    ("rmsd", {"fit_selection": "name C O H", "reference_frame": 3}, "time_ps", 10, 10, True),
]


@pytest.mark.parametrize("metric,extra,axis,n_points,n_frames,has_final", RIGID)
def test_rigid_body_motion_is_not_internal_motion(rigid, metric, extra, axis, n_points,
                                                  n_frames, has_final):
    obs = REGISTRY.call("traj_analyze", rigid, {
        "structure": "01_input/rigid.gro", "trajectory": "01_input/rigid.trr",
        "metric": metric, "selection": "all", **extra})
    assert obs.ok, obs.summary
    assert obs.largest < 1e-3, f"{metric} sees motion in a rigid body: {obs.largest} Å"
    assert (obs.axis, obs.n_points, obs.n_frames) == (axis, n_points, n_frames)
    assert (obs.final is not None) is has_final
    assert obs.fit_selection == extra.get("fit_selection", "all")
    assert obs.reference.startswith(f"frame {extra.get('reference_frame', 0)}")
    assert any(t.startswith("superposition") for t in obs.transformations)
    if "start_ps" in extra:
        assert obs.time_window_ps == [5.0, 9.0]
    with np.load(rigid.workspace.root / obs.artifacts[0].path) as z:
        assert set(z.files) == {axis, metric} and len(z[metric]) == n_points


# (structure, unwrap, ok, the radius of gyration of two atoms 2 Å apart, one of them pushed
# across the box in the second frame). Whole molecules only when asked and only with bonds.
PERIODIC = [
    ("pair.pdb", True, True, 1.0),
    ("pair.pdb", False, True, 5.0),  # measured as the box split it, and said so
    ("pair.gro", True, False, None),  # no bonds in a .gro: refused, not guessed
]


@pytest.mark.parametrize("structure,unwrap,ok,mean_rg", PERIODIC)
def test_a_molecule_split_by_the_box_is_made_whole_only_when_asked(ws_ctx, structure,
                                                                    unwrap, ok, mean_rg):
    stage = ws_ctx.workspace.stage("input")
    (stage / "pair.pdb").write_text(PAIR_PDB)
    (stage / "pair.gro").write_text(PAIR_GRO)
    whole = np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
    split = np.array([[1.0, 1.0, 1.0], [-17.0, 1.0, 1.0]])
    _write_frames(stage / "pair.trr", stage / "pair.pdb", [whole, split])
    obs = REGISTRY.call("traj_analyze", ws_ctx, {
        "structure": f"01_input/{structure}", "trajectory": "01_input/pair.trr",
        "metric": "rg", "selection": "all", "unwrap": unwrap})
    assert obs.ok is ok, obs.summary
    if not ok:
        assert "bonds" in obs.summary
        return
    assert obs.mean == pytest.approx(mean_rg, abs=1e-6)
    assert any(t.startswith("unwrap") for t in obs.transformations) is unwrap


# (this trajectory's frames, the reference's, mean, reference mean): a pair is measured
# under the frame's box, so a pair the box split is measured across the boundary
WHOLE = np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
SPLIT = np.array([[1.0, 1.0, 1.0], [-17.0, 1.0, 1.0]])
FAR = np.array([[1.0, 1.0, 1.0], [4.0, 1.0, 1.0]])
PAIR_DISTANCES = [
    ([WHOLE, WHOLE], None, 2.0, None),
    ([WHOLE, SPLIT], None, 2.0, None),
    ([WHOLE, SPLIT], [FAR, FAR], 2.0, 3.0),
]


@pytest.mark.parametrize("frames,reference,mean,ref_mean", PAIR_DISTANCES)
def test_named_pairs_are_measured_under_the_box_and_against_a_reference(
        ws_ctx, frames, reference, mean, ref_mean):
    stage = ws_ctx.workspace.stage("input")
    (stage / "pair.pdb").write_text(PAIR_PDB)
    _write_frames(stage / "pair.trr", stage / "pair.pdb", frames)
    params = {"structure": "01_input/pair.pdb", "trajectory": "01_input/pair.trr",
              "metric": "pair_distance", "pairs": {"C1-C2": [1, 2]}}
    if reference is not None:
        _write_frames(stage / "ref.trr", stage / "pair.pdb", reference)
        params["reference"] = "01_input/ref.trr"
    obs = REGISTRY.call("traj_analyze", ws_ctx, params)
    assert obs.ok, obs.summary
    assert obs.pairs["C1-C2"]["mean"] == pytest.approx(mean, abs=1e-4)
    assert obs.pairs["C1-C2"].get("reference_mean") == (
        None if ref_mean is None else pytest.approx(ref_mean, abs=1e-4))
    assert obs.mae == (None if ref_mean is None else pytest.approx(abs(mean - ref_mean),
                                                                    abs=1e-4))
    with np.load(ws_ctx.workspace.root / obs.artifacts[0].path) as z:
        assert [str(x) for x in z["pairs"]] == ["C1-C2"]
        assert z["pair_distance"].shape == (len(frames), 1)


def test_frames_written_as_one_trajectory_keep_the_order_asked(rigid):
    """The k-th frame of the written trajectory is the k-th index asked for."""
    obs = REGISTRY.call("traj_frames", rigid, {
        "structure": "01_input/rigid.gro", "trajectory": "01_input/rigid.trr",
        "frames": [7, 2, 5], "output": "trajectory"})
    assert obs.ok, obs.summary
    assert [a.role for a in obs.artifacts] == ["trajectory"]
    u = mda.Universe(str(rigid.workspace.root / "01_input/rigid.gro"),
                     str(rigid.workspace.root / obs.artifacts[0].path))
    assert [ts.time for ts in u.trajectory] == [7.0, 2.0, 5.0]


def test_mindist_reports_minimum_and_contacts_maps_deposited_residues(ws_ctx):
    from myrmidon.knowledge.rules import save_residue_map

    stage = ws_ctx.workspace.stage("input")
    (stage / "pair.pdb").write_text(PAIR_PDB)
    save_residue_map(ws_ctx.workspace.root, {1: "A:MET189A"})
    f1 = np.array([[1.0, 1.0, 1.0], [3.5, 1.0, 1.0]])  # 2.5 A
    f2 = np.array([[1.0, 1.0, 1.0], [2.8, 1.0, 1.0]])  # 1.8 A
    _write_frames(stage / "pair.trr", stage / "pair.pdb", [f1, f2])

    md = REGISTRY.call("traj_analyze", ws_ctx, {
        "structure": "01_input/pair.pdb", "trajectory": "01_input/pair.trr",
        "metric": "mindist", "selection": "name C1", "selection2": "name C2"})
    assert md.ok, md.summary
    assert md.minimum == pytest.approx(1.8, abs=1e-3)
    assert md.min_time_ps == 1.0
    assert md.largest == pytest.approx(2.5, abs=1e-3)
    assert "min 1.800" in md.summary

    ct = REGISTRY.call("traj_analyze", ws_ctx, {
        "structure": "01_input/pair.pdb", "trajectory": "01_input/pair.trr",
        "metric": "contacts", "selection": "name C1", "selection2": "name C2",
        "cutoff_angstrom": 3.0})
    assert ct.ok, ct.summary
    assert len(ct.contacts) == 1
    assert ct.contacts[0]["resid"] == 1
    assert ct.contacts[0]["deposited"] == "A:MET189A"
    assert ct.contacts[0]["contact_fraction"] == 1.0
    assert ct.contacts[0]["min_distance_angstrom"] == pytest.approx(1.8, abs=1e-3)


def test_an_invalid_selection_is_refused_with_the_mdanalysis_spelling(ws_ctx):
    stage = ws_ctx.workspace.stage("input")
    (stage / "pair.pdb").write_text(PAIR_PDB)
    f1 = np.array([[1.0, 1.0, 1.0], [3.5, 1.0, 1.0]])
    _write_frames(stage / "pair.trr", stage / "pair.pdb", [f1])
    refused = REGISTRY.call("traj_analyze", ws_ctx, {
        "structure": "01_input/pair.pdb", "trajectory": "01_input/pair.trr",
        "metric": "rmsd", "selection": "resname JZ4 and heavy"})
    assert not refused.ok and "not name H*" in refused.summary


TRIO_PDB = """\
CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
ATOM      1  C1  MOL A   1       1.000   1.000   1.000  1.00  0.00           C
ATOM      2  H1  MOL A   1       1.900   1.000   1.000  1.00  0.00           H
ATOM      3  C2  LIG A   2       3.500   1.000   1.000  1.00  0.00           C
ATOM      4  H2  LIG A   2       4.400   1.000   1.000  1.00  0.00           H
END
"""

# (the two selections, a finding that they treat hydrogens differently): a distance from
# every atom of one side to the heavy atoms of the other is said; alike on both is not
SELECTION_PAIRS = [
    ("resname MOL", "resname LIG and not name H*", True),
    ("resname LIG and not name H*", "resname MOL", True),
    ("resname MOL and not name H*", "resname LIG and not name H*", False),
    ("resname MOL", "resname LIG", False),
]


@pytest.mark.parametrize("first,second,mixed", SELECTION_PAIRS)
def test_a_distance_between_unlike_selections_is_said(ws_ctx, first, second, mixed):
    stage = ws_ctx.workspace.stage("input")
    (stage / "trio.pdb").write_text(TRIO_PDB)
    frame = mda.Universe(str(stage / "trio.pdb")).atoms.positions.copy()
    _write_frames(stage / "trio.trr", stage / "trio.pdb", [frame, frame])
    obs = REGISTRY.call("traj_analyze", ws_ctx, {
        "structure": "01_input/trio.pdb", "trajectory": "01_input/trio.trr",
        "metric": "mindist", "selection": first, "selection2": second})
    assert obs.ok, obs.summary
    assert repr(second) in obs.summary  # both selections are named
    assert ("SELECTIONS_TREAT_HYDROGENS_DIFFERENTLY" in {f.id for f in obs.findings}) is mixed


CHAIN_PDB = """\
CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
ATOM      1  C1  MOL A   1       1.000   1.000   1.000  1.00  0.00           C
ATOM      2  C2  MOL A   1       2.500   1.000   1.000  1.00  0.00           C
ATOM      3  C3  MOL A   1       4.000   1.000   1.000  1.00  0.00           C
ATOM      4  C4  MOL A   1       4.000   2.500   1.000  1.00  0.00           C
CONECT    1    2
CONECT    2    3
CONECT    3    4
END
"""


def test_an_rmsd_fitted_on_part_of_a_molecule_unwraps_the_molecule(ws_ctx):
    """A fit on atoms that are not contiguous by themselves — a pocket's backbone inside a
    protein — is made whole with the molecule it belongs to, not refused with a traceback."""
    stage = ws_ctx.workspace.stage("input")
    (stage / "chain.pdb").write_text(CHAIN_PDB)
    frame = mda.Universe(str(stage / "chain.pdb")).atoms.positions.copy()
    _write_frames(stage / "chain.trr", stage / "chain.pdb", [frame, frame + 0.5])
    obs = REGISTRY.call("traj_analyze", ws_ctx, {
        "structure": "01_input/chain.pdb", "trajectory": "01_input/chain.trr",
        "metric": "rmsd", "selection": "name C4", "fit_selection": "name C1 C3",
        "unwrap": True})
    assert obs.ok, obs.summary
    assert obs.largest == pytest.approx(0.0, abs=1e-3)  # a rigid shift fits away


POCKET_PDB = """\
CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1
ATOM      1  C1  PRO A   1       1.000   5.000   5.000  1.00  0.00           C
ATOM      2  C2  PRO A   1       2.500   5.000   5.000  1.00  0.00           C
ATOM      3  C3  PRO A   1       4.000   5.000   5.000  1.00  0.00           C
ATOM      4  C4  PRO A   1       4.000   6.500   5.000  1.00  0.00           C
ATOM      5  O1  LIG B   2       2.500   3.500   5.000  1.00  0.00           O
ATOM      6  O2  LIG B   2       3.500   3.000   5.000  1.00  0.00           O
CONECT    1    2
CONECT    2    3
CONECT    3    4
CONECT    5    6
END
"""


def test_a_ligand_rmsd_fitted_on_its_pocket_survives_the_box_edge(ws_ctx):
    """The complex moves rigidly across the box and is wrapped atom by atom, splitting the
    protein and parting the ligand from its pocket: fitted on the pocket, the ligand has not
    moved, and its RMSD says so."""
    stage = ws_ctx.workspace.stage("input")
    (stage / "pocket.pdb").write_text(POCKET_PDB)
    p0 = mda.Universe(str(stage / "pocket.pdb")).atoms.positions.copy()
    _write_frames(stage / "pocket.trr", stage / "pocket.pdb", [p0, (p0 + [7.5, 0, 0]) % 10.0])
    obs = REGISTRY.call("traj_analyze", ws_ctx, {
        "structure": "01_input/pocket.pdb", "trajectory": "01_input/pocket.trr",
        "metric": "rmsd", "selection": "resname LIG", "fit_selection": "name C1 C3",
        "unwrap": True})
    assert obs.ok, obs.summary
    assert obs.largest == pytest.approx(0.0, abs=1e-2)
