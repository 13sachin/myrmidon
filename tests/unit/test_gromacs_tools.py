"""End-to-end GROMACS tool chain against the mock gmx binary."""

import time
from pathlib import Path

import pytest
from tests.unit.findings_catalog import blocking

from myrmidon.core.tool import REGISTRY
from myrmidon.engines.gromacs.itp import parse_itp_atoms

GOOD_MDP = (
    "integrator = md\ndt = 0.002\nconstraints = h-bonds\nnsteps = 5000\n"
    "tcoupl = v-rescale\nref_t = 300\ntau_t = 0.1\ntc_grps = System\n"
    "coulombtype = PME\nrcoulomb = 1.0\nrvdw = 1.0\n"
)
# Same but no thermostat: plants exactly one advisory finding.
NVE_MDP = (
    "integrator = md\ndt = 0.002\nconstraints = h-bonds\nnsteps = 5000\n"
    "coulombtype = PME\nrcoulomb = 1.0\nrvdw = 1.0\n"
)


def _write_pdb(ws):
    stage = ws.stage("fetch")
    pdb = stage / "input.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00\n"
        "END\n"
    )
    return ws.relpath(pdb)


def _wait_for_job(ctx, job_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        obs = REGISTRY.call("job_status", ctx, {"job_id": job_id})
        if obs.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return obs
        time.sleep(0.1)
    raise AssertionError("job never finished")


EM_MDP = (
    "integrator = steep\nemtol = 100.0\nemstep = 0.01\nnsteps = 50000\n"
    "cutoff-scheme = Verlet\nnstlist = 1\ncoulombtype = PME\nrcoulomb = 1.0\n"
    "rvdw = 1.0\npbc = xyz\n"
)


def test_energy_statistics_are_read_from_the_edr(ws_ctx):
    ws = ws_ctx.workspace
    run_stage = ws.stage("run")
    (run_stage / "md.edr").write_text("MOCKEDR\n")
    obs = REGISTRY.call(
        "observable_measure",
        ws_ctx,
        {"edr": ws.relpath(run_stage / "md.edr"), "terms": ["Potential", "Temperature"]},
    )
    assert obs.ok, obs.summary
    assert set(obs.terms) == {"Potential", "Temperature"}
    pot = obs.terms["Potential"]
    # mock writes -4200 + 0.5*t over 10 ps → drift 0.5/ps = 500/ns
    assert abs(pot.mean - (-4197.5)) < 1e-6
    assert abs(pot.drift_per_ns - 500.0) < 1e-6
    assert pot.n_points == 11
    assert obs.terms["Temperature"].unit == "K"
    assert obs.time_span_ps == 10.0


def test_a_run_account_is_booked_and_named(ws_ctx):
    """One user, several allocations: a submission may name the account it bills, the
    job spec carries it to the backend, and the budget line says so before the run."""
    from myrmidon.execution.accounting import budget_warnings
    from myrmidon.tools.stage import StageRunParams

    res = StageRunParams(stage="x", account="project_b").resources()
    assert res.account == "project_b"
    assert StageRunParams(stage="x").resources().account is None
    _, findings = budget_warnings(res, ws_ctx.config.compute, "slurm")
    assert "billed to account project_b" in findings[0].message
    _, findings = budget_warnings(StageRunParams(stage="x").resources(),
                                  ws_ctx.config.compute, "slurm")
    assert "billed to" not in findings[0].message




# (the mock's fault, what the scheduler's or engine's word on the end says, the engine's
# own line in the log tail): a crash is not classified, its words come back for the reader.
# A job the scheduler kills is told by the one fact that decides the next step: how far it
# got, and that it can continue.
CRASHES = [
    ("lincs", [], "LINCS WARNING"),
    ("walltime", ["killed at its walltime", "10995 of 50000", "continue_run"], None),
]


@pytest.mark.parametrize("mode,says,words", CRASHES)
def test_a_failed_run_returns_its_log_for_the_reader(ws_ctx, monkeypatch, mode, says, words):
    from tests.conftest import NVT_STAGE, approved_study

    ws = ws_ctx.workspace
    prep = REGISTRY.call("gmx_pdb2gmx", ws_ctx, {"structure": _write_pdb(ws),
                                                 "forcefield": "amber99sb-ildn",
                                                 "water": "tip3p"})
    approved_study(ws_ctx, next(a.path for a in prep.artifacts if a.role == "structure"),
                   next(a.path for a in prep.artifacts if a.role == "topology"), [NVT_STAGE])
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "nvt"}).ok
    monkeypatch.setenv("MOCK_GMX_FAIL", mode)
    submitted = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt"})
    results = _wait_for_job(ws_ctx, submitted.job_id)
    assert results.state == "FAILED" and not results.ok
    assert (results.diagnosis is None) is not bool(says)
    for phrase in says:
        assert phrase in results.diagnosis
    if words:
        assert words in results.log_tail


@pytest.mark.parametrize("grompp_fails", [False, True])
def test_a_restrained_run_input_holds_atoms_at_its_structure(ws_ctx, monkeypatch, grompp_fails):
    """`-DPOSRES*` with no reference given: grompp is given the structure as `-r`, the usual
    choice, and says so rather than refusing; when grompp fails, the note is never named among
    grompp's own warnings."""
    from myrmidon.engines.gromacs.run import grompp

    ws = ws_ctx.workspace
    run_stage = ws.stage("run")
    (run_stage / "nvt.mdp").write_text(GOOD_MDP + "define = -DPOSRES\n")
    prep = REGISTRY.call("gmx_pdb2gmx", ws_ctx, {"structure": _write_pdb(ws),
                                                 "forcefield": "amber99sb-ildn",
                                                 "water": "tip3p"})
    structure = next(a.path for a in prep.artifacts if a.role == "structure")
    topology = next(a.path for a in prep.artifacts if a.role == "topology")
    if grompp_fails:
        monkeypatch.setenv("MOCK_GMX_FAIL", "grompp")
    tpr = grompp(ws_ctx, run_stage / "nvt.mdp", ws.resolve(structure), ws.resolve(topology), None,
                 None, run_stage / "nvt.tpr", run_stage)
    assert tpr.ok is not grompp_fails, tpr.summary
    assert "REFERENCE" not in tpr.summary
    assert tpr.command[tpr.command.index("-r") + 1].endswith(structure)
    (note,) = [w for w in tpr.findings if w.id == "GROMPP_RESTRAINT_REFERENCE_DEFAULTED"]
    assert note.severity == "info"


GMX_FIXTURES = __import__("pathlib").Path(__file__).parents[1] / "fixtures" / "gromacs"


@pytest.mark.parametrize(("output", "skipped"), [
    ("make_ndx_commands", ["r XYZ", "foo bar"]),
    ("make_ndx_missing_group", ["1 | 40", "del 40"]),
])
def test_commands_make_ndx_skipped_are_read_from_its_output(output, skipped):
    from myrmidon.engines.gromacs.index import rejected_ndx_commands

    sent = (GMX_FIXTURES / f"{output}.in").read_text().splitlines()[:-1]
    real = (GMX_FIXTURES / f"{output}.out").read_text()
    assert [c for c, _ in rejected_ndx_commands(real, sent)] == skipped


# (commands, what make_ndx is sent): a quoted name that several default groups share for
# one set of atoms (a ligand's residue and its molecule) is the first one's number; a name
# whose groups differ, or that one group holds, is sent as written, and so is every command
# after one that renumbers or renames groups
DEFAULT_GROUPS = [("System", [1, 2, 3, 4]), ("Protein", [1, 2]), ("LIG", [3]),
                  ("Other", [3, 4]), ("LIG", [3]), ("CL", [4]), ("CL", [4, 5])]
QUOTED = [
    (['"Protein" | "LIG"'], ['"Protein" | 2']),
    (['"Protein" | "lig"'], ['"Protein" | 2']),  # GROMACS matches a name whatever its case
    (['"CL"'], ['"CL"']),
    (['"Other"', "name 7 Lig"], ['"Other"', "name 7 Lig"]),
    (['"LIG"', "del 0", '"LIG"'], ["2", "del 0", '"LIG"']),
]


@pytest.mark.parametrize(("commands", "sent"), QUOTED)
def test_a_name_default_groups_repeat_for_one_set_of_atoms_is_resolved(commands, sent):
    from myrmidon.engines.gromacs.index import resolve_repeated_names

    out, said = resolve_repeated_names(commands, DEFAULT_GROUPS)
    assert out == sent and bool(said) is (out != commands)


def _moltype(name, n):
    rows = "".join(f"{i} X 1 {name} A{i} {i} 0.0 1.0\n" for i in range(1, n + 1))
    return f"[ moleculetype ]\n{name} 3\n[ atoms ]\n{rows}\n"


#: the trimmed systems' topologies: a protein first, and an anion first with its ion last
TOPOLOGIES = {
    "trpcage_water.gro": _moltype("Protein_chain_A", 304) + _moltype("SOL", 3)
    + "[ system ]\nx\n[ molecules ]\nProtein_chain_A 1\nSOL 10\n",
    "acetate_water.gro": _moltype("OAC", 7) + _moltype("SOL", 3) + _moltype("NA", 1)
    + "[ system ]\nx\n[ molecules ]\nOAC 1\nSOL 20\nNA 1\n",
    "unreadable": '#include "elsewhere.itp"\n' + _moltype("SOL", 3)
    + "[ system ]\nx\n[ molecules ]\nProtein_chain_A 1\nSOL 10\n",
    "water_unreadable": _moltype("Protein_chain_A", 304) + '#include "elsewhere.itp"\n'
    + "[ system ]\nx\n[ molecules ]\nProtein_chain_A 1\nSOL 10\n",
}

# (structure, topology, group, the finding it is refused or warned on, the moleculetype
# reported). Restraints are numbered within their molecule: from 1 only for the first one.
GENRESTR = [
    ("trpcage_water.gro", "trpcage_water.gro", "Protein", None, "Protein_chain_A"),
    ("trpcage_water.gro", "trpcage_water.gro", "C-alpha", None, "Protein_chain_A"),
    ("acetate_water.gro", "acetate_water.gro", "OAC", None, "OAC"),
    ("acetate_water.gro", "acetate_water.gro", "NA", "GENRESTR_NOT_MOLECULE_NUMBERING", "NA"),
    ("acetate_water.gro", "acetate_water.gro", "non-Water", "GENRESTR_SPANS_MOLECULES", None),
    ("trpcage_water.gro", "trpcage_water.gro", "SOL", "GENRESTR_SPANS_MOLECULES", None),
    ("trpcage_water.gro", "unreadable", "SOL", "GENRESTR_LAYOUT_UNKNOWN", None),
    ("trpcage_water.gro", "water_unreadable", "Protein", None, "Protein_chain_A"),
]


@pytest.mark.parametrize(("structure", "topology", "group", "finding", "molecule"), GENRESTR)
def test_restraints_are_numbered_from_their_molecule(
        ws_ctx, structure, topology, group, finding, molecule):
    ws = ws_ctx.workspace
    stage = ws.stage("input")
    (stage / "conf.gro").write_text((GMX_FIXTURES / structure).read_text())
    (stage / "topol.top").write_text(TOPOLOGIES[topology])
    obs = REGISTRY.call("gmx_genrestr", ws_ctx, {
        "structure": ws.relpath(stage / "conf.gro"), "topology": ws.relpath(stage / "topol.top"),
        "group": group, "force_constants": [1000, 1000, 1000]})
    found = {w.id for w in obs.findings}
    assert (finding in found) if finding else not found, obs.summary
    assert obs.ok is (finding in (None, "GENRESTR_LAYOUT_UNKNOWN"))
    assert obs.molecule == molecule


def test_a_group_genrestr_does_not_have_comes_back_with_the_ones_it_has(ws_ctx):
    ws = ws_ctx.workspace
    stage = ws.stage("input")
    (stage / "conf.gro").write_text((GMX_FIXTURES / "acetate_water.gro").read_text())
    (stage / "topol.top").write_text(TOPOLOGIES["acetate_water.gro"])
    obs = REGISTRY.call("gmx_genrestr", ws_ctx, {
        "structure": ws.relpath(stage / "conf.gro"), "topology": ws.relpath(stage / "topol.top"),
        "group": "LIG", "force_constants": [1000, 1000, 1000]})
    assert not obs.ok
    assert "OAC" in {g.name for g in obs.available_groups}


LIGAND_FIXTURES = GMX_FIXTURES.parent / "ligand"

# (selection parameters, the finding it is refused on, fields of the selection). A
# deposited excerpt: three residues of chain A, nine atoms with a B location, the ligand,
# two phosphates, a buffer molecule, three crystal waters.
SELECTIONS = [
    ({"chains": ["A"], "ligand": "JZ4"}, None,
     {"protein_atoms": 23, "protein_residues": 3, "ligand_atoms": 10, "ligand_hydrogens": 0,
      "discarded": {"PO4": 2, "BME": 1, "HOH": 3}, "altloc": "A", "altloc_atoms": 9}),
    # kept heteroatoms leave the polymer's chain: pdb2gmx refuses a chain of mixed types
    ({"chains": ["A"], "ligand": None, "keep": ["HOH"]}, None,
     {"protein_atoms": 26, "ligand": None, "kept_hetero": {"HOH": 3}, "kept_chains": {"HOH": "B"},
      "discarded": {"JZ4": 1, "PO4": 2, "BME": 1}}),
    ({"chains": ["A"], "ligand": "JZ4", "altloc": "B"}, None, {"protein_atoms": 23}),
    ({"chains": ["B"], "ligand": "JZ4"}, "STRUCTURE_CHAIN_NOT_FOUND", {}),
    ({"chains": ["A"], "ligand": "ATP"}, "STRUCTURE_LIGAND_NOT_FOUND", {}),
    ({"chains": ["A"], "ligand": "PO4"}, "STRUCTURE_LIGAND_AMBIGUOUS", {}),
    ({"chains": ["A"], "ligand": "PO4", "ligand_resid": 166}, None, {"ligand_atoms": 5}),
    ({"chains": ["A"], "ligand": "JZ4", "altloc": "C"}, "STRUCTURE_ALTLOC_MISSING", {}),
]


@pytest.mark.parametrize(("params", "finding", "fields"), SELECTIONS)
def test_a_deposited_structure_splits_into_protein_ligand_and_discard(
        ws_ctx, params, finding, fields):
    ws = ws_ctx.workspace
    pdb = ws.stage("fetch") / "3htb.pdb"
    pdb.write_text((LIGAND_FIXTURES / "3htb_excerpt.pdb").read_text())
    obs = REGISTRY.call("structure_select", ws_ctx, {"structure": ws.relpath(pdb), **params})
    assert obs.ok is (finding is None), obs.summary
    assert finding is None or finding in {w.id for w in obs.findings}
    for key, value in fields.items():
        assert getattr(obs, key) == value, key
    if obs.ok:
        lines = (ws.root / obs.artifacts[0].path).read_text().splitlines()
        assert sum(ln.startswith(("ATOM", "HETATM")) for ln in lines) == obs.protein_atoms
        assert all(ln[16] == " " for ln in lines if ln.startswith("ATOM"))
        hetero = [ln for ln in lines if ln.startswith("HETATM")]
        if hetero:  # after the polymer's TER, under chain ids of their own, numbered from 1
            assert lines[lines.index(hetero[0]) - 1] == "TER"
            assert {ln[21] for ln in hetero} == set(obs.kept_chains.values())
            assert not set(obs.kept_chains.values()) & set(obs.chains)
            for name, n in obs.kept_hetero.items():
                numbers = [int(ln[22:26]) for ln in hetero if ln[17:20].strip() == name]
                assert sorted(set(numbers)) == list(range(1, n + 1))


@pytest.mark.parametrize(("build", "built", "missing"), [
    (True, {"A:THR21": ["CG2", "OG1"]}, {}),
    (False, {}, {"A:THR21": ["OG1", "CG2"]}),
])
def test_a_residue_the_deposit_leaves_incomplete_is_built_or_named(
        ws_ctx, build, built, missing):
    """pdb2gmx refuses a residue without its heavy atoms: they are built and listed, or the
    deposit's own REMARK 470 names them."""
    if build:
        pytest.importorskip("pdbfixer")
    ws = ws_ctx.workspace
    pdb = ws.stage("fetch") / "3htb.pdb"
    lines = (LIGAND_FIXTURES / "3htb_excerpt.pdb").read_text().splitlines()
    pdb.write_text("\n".join(["REMARK 470     THR A  21    OG1  CG2", *(
        ln for ln in lines if not (ln[17:26] == "THR A  21" and ln[12:16].strip() in
                                   ("OG1", "CG2")))]) + "\n")
    obs = REGISTRY.call("structure_select", ws_ctx, {
        "structure": ws.relpath(pdb), "chains": ["A"], "ligand": "JZ4",
        "build_missing_atoms": build})
    assert obs.ok, obs.summary
    assert {k: sorted(v) for k, v in obs.built_atoms.items()} == built
    assert obs.missing_atoms == missing
    assert obs.protein_atoms == 21 + 2 * build
    assert {"STRUCTURE_ATOMS_BUILT" if build else "STRUCTURE_ATOMS_MISSING"} <= {
        w.id for w in obs.findings}


#: a protein topology laid out as pdb2gmx writes one: force field, molecule, water, ions
PROTEIN_TOP = """\
#include "amber99sb-ildn.ff/forcefield.itp"

[ moleculetype ]
Protein_chain_A     3

[ atoms ]
     1         N3      1    MET      N      1     0.1592      14.01
     2         CT      1    MET     CA      2     0.0221      12.01
     3          C      1    MET      C      3     0.8187      12.01

#ifdef POSRES
#include "posre.itp"
#endif

; Include water topology
#include "amber99sb-ildn.ff/tip3p.itp"

; Include topology for ions
#include "amber99sb-ildn.ff/ions.itp"

[ system ]
Protein

[ molecules ]
; Compound        #mols
Protein_chain_A     1
"""
PROTEIN_GRO = ("protein\n    3\n    1MET      N    1   2.000   2.000   2.000\n"
               "    1MET     CA    2   2.100   2.000   2.000\n"
               "    1MET      C    3   2.200   2.000   2.000\n   4.00000   4.00000   4.00000\n")

# (the ligand's .itp, coordinates and restraints, the finding the assembly is refused on).
# The itp is a real acpype run's on the deposited ligand with hydrogens added.
ASSEMBLIES = [
    ("real", "real", range(1, 11), None),
    ("acpype_top", "real", range(1, 11), "COMPLEX_LIGAND_DEFAULTS"),
    ("real", "real", range(1, 31), "COMPLEX_RESTRAINTS_OUT_OF_RANGE"),
    ("real", "mock", range(1, 11), "COMPLEX_LIGAND_ATOM_ORDER"),
    ("real", "real", None, None),
]


@pytest.mark.parametrize(("itp", "coords", "restrained", "finding"), ASSEMBLIES)
def test_a_ligand_is_parametrised_and_placed_where_grompp_reads_it(
        ws_ctx, itp, coords, restrained, finding):
    ws = ws_ctx.workspace
    pdb = ws.stage("fetch") / "3htb.pdb"
    pdb.write_text((LIGAND_FIXTURES / "3htb_excerpt.pdb").read_text())
    sel = REGISTRY.call("structure_select", ws_ctx, {"structure": ws.relpath(pdb),
                                                     "chains": ["A"], "ligand": "JZ4"})
    ligand = sel.artifacts[1].path
    refused = REGISTRY.call("ligand_parametrize", ws_ctx, {
        "structure": ligand, "net_charge": 0, "hydrogens": "present", "atom_types": "gaff2",
        "charge_method": "bcc"})
    assert [w.id for w in blocking(refused)] == ["LIGAND_HYDROGENS_ABSENT"]
    par = REGISTRY.call("ligand_parametrize", ws_ctx, {
        "structure": ligand, "net_charge": 0, "hydrogens": "add_by_valence",
        "atom_types": "gaff2", "charge_method": "bcc"})
    assert par.ok, par.summary
    assert (par.molecule, par.net_charge, par.charge_method) == ("JZ4", 0.0, "AM1-BCC (sqm)")
    assert par.n_atoms == 10 + par.hydrogens_added
    radical = REGISTRY.call("ligand_parametrize", ws_ctx, {
        "structure": ligand, "net_charge": 1, "hydrogens": "add_by_valence",
        "atom_types": "gaff2", "charge_method": "bcc"})
    assert [w.id for w in blocking(radical)] == ["LIGAND_ODD_ELECTRONS"]
    made = {a.role: ws.root / a.path for a in par.artifacts}

    stage = ws.stage("input")
    (stage / "protein.top").write_text(PROTEIN_TOP)
    (stage / "posre.itp").write_text("[ position_restraints ]\n1 1 1000 1000 1000\n")
    (stage / "protein.gro").write_text(PROTEIN_GRO)
    real_itp = (LIGAND_FIXTURES / "jz4_gaff2_bcc.itp").read_text()
    (stage / "lig.itp").write_text(real_itp if itp == "real" else made["topology"].read_text())
    real_gro = "".join(f"    1  JZ4{a['name']:>5}{i:5d}   2.000   2.000   2.000\n"
                       for i, a in enumerate(parse_itp_atoms(real_itp)["JZ4"], start=1))
    (stage / "lig.gro").write_text(made["structure"].read_text() if coords == "mock"
                                   else f"JZ4\n   22\n{real_gro}   3.0 3.0 3.0\n")
    params = {"structure": ws.relpath(stage / "protein.gro"),
              "topology": ws.relpath(stage / "protein.top"),
              "ligand_structure": ws.relpath(stage / "lig.gro"),
              "ligand_itp": ws.relpath(stage / "lig.itp"), "ligand_restraints": None}
    if restrained is not None:
        (stage / "posre_lig.itp").write_text("[ position_restraints ]\n" + "".join(
            f"{i} 1 1000 1000 1000\n" for i in restrained))
        params["ligand_restraints"] = ws.relpath(stage / "posre_lig.itp")
    obs = REGISTRY.call("complex_assemble", ws_ctx, params)
    assert obs.ok is (finding is None), obs.summary
    if finding:
        assert finding in {w.id for w in obs.findings}
        return
    assert (obs.n_atoms, obs.ligand_atoms, obs.molecule) == (25, 22, "JZ4")
    assert obs.net_charge == 1.0
    assert obs.restrained_atoms == (10 if restrained else 0)
    top = (ws.root / next(a.path for a in obs.artifacts if a.role == "topology")).read_text()
    lines = [ln.strip() for ln in top.splitlines()]
    assert lines.index('#include "JZ4_atomtypes.itp"') < lines.index("[ moleculetype ]")
    assert lines.index("[ moleculetype ]") < lines.index('#include "JZ4.itp"') \
        < lines.index('#include "amber99sb-ildn.ff/tip3p.itp"')
    assert ("#ifdef POSRES_LIG" in lines) is bool(restrained)
    assert [ln.split() for ln in lines[-2:]] == [["Protein_chain_A", "1"], ["JZ4", "1"]]
    gro = (ws.root / next(a.path for a in obs.artifacts if a.role == "structure"))
    assert gro.read_text().splitlines()[-2][5:15].split() == ["JZ4", "H11"]
    assert sel.residue_map == {1: "A:MET1", 2: "A:ASN2", 3: "A:THR21"}
    # the facts the agent reasons over: a ligand with no hydrogens, a renumbered topology
    assert (sel.system_profile["ligand_hydrogens"], sel.system_profile["has_residue_offset"]) \
        == (0, True)
    assert obs.system_profile["ligand_atoms"] == 22


def _ar1(phi, n, seed, transient=0.0, tau_ps=0.0, dt_ps=2.0):
    import numpy as np

    rng = np.random.default_rng(seed)
    x, e = np.zeros(n), rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    t = np.arange(n) * dt_ps
    return t, x + (transient * np.exp(-t / tau_ps) if tau_ps else 0.0)


# (AR(1) coefficient, its exact statistical inefficiency (1 + phi) / (1 - phi))
INEFFICIENCY = [(0.0, 1.0), (0.5, 3.0), (0.8, 9.0)]


@pytest.mark.parametrize(("phi", "exact"), INEFFICIENCY)
def test_statistical_inefficiency_is_that_of_the_process(phi, exact):
    from myrmidon.analysis.convergence import statistical_inefficiency

    _, x = _ar1(phi, 50000, seed=3)
    assert statistical_inefficiency(x) == pytest.approx(exact, rel=0.15)


def test_the_settled_part_starts_after_the_transient_and_the_verdict_is_the_criterions():
    """A relaxation of 50 units with a 400 ps time constant on top of noise: the settled
    part begins once the transient is inside the noise, and no criterion means no verdict."""
    from myrmidon.analysis.convergence import describe

    t, y = _ar1(0.5, 5000, seed=0, transient=50.0, tau_ps=400.0)
    free = describe("T", t, y, "K", None, None)
    assert 1200 <= free.equilibrated_from_ps <= 3600
    assert free.criterion_met is None
    assert describe("T", t, y, "K", 1.0, 3.0).criterion_met is True
    # a 1 K/ns ramp: nearly all of it is discarded, and the flat-looking end is no verdict
    ramp = describe("T", t, y + 1e-3 * t, "K", 0.5, None)
    assert ramp.discarded_fraction > 0.5 and ramp.criterion_met is None
    whole = describe("T", t, y + 1e-3 * t, "K", 0.5, None, min_settled=0.0)
    assert whole.criterion_met is not None


@pytest.mark.parametrize(("name", "content", "ok"), [
    ("density.xvg", None, True),
    ("rmsd.npz", "time_ps", True),
    ("rmsf.npz", "atom_index", False),
])
def test_the_sensor_reads_time_series_and_only_those(ws_ctx, name, content, ok):
    import numpy as np

    ws = ws_ctx.workspace
    stage = ws.stage("analysis")
    t, y = _ar1(0.5, 400, seed=1)
    path = stage / name
    if content is None:
        rows = "".join(f"{a:.1f} {b + 1000:.4f}\n" for a, b in zip(t, y, strict=True))
        path.write_text('@    title "x"\n@ s0 legend "Density"\n' + rows)
    else:
        np.savez(path, **{content: t, name.split(".")[0]: y})
    obs = REGISTRY.call("observable_measure", ws_ctx, {"data": ws.relpath(path)})
    assert obs.ok is ok, obs.summary
    if ok:
        assert obs.series[0].criterion_met is None and obs.series[0].effective_samples > 10


FOUR_ATOMS_GRO = """\
four atoms
    4
    1MOL      C    1   1.000   1.000   1.000
    1MOL      O    2   1.150   1.000   1.000
    1MOL      H    3   1.000   1.120   1.000
    1MOL      N    4   1.000   1.000   1.090
   5.00000   5.00000   5.00000
"""
FOUR_ATOMS_TOP = """\
[ moleculetype ]
MOL 3
[ atoms ]
1 c 1 MOL C 1 0.5
2 o 1 MOL O 1 -0.5
3 h 1 MOL H 1 0.1
4 n 1 MOL N 1 -0.1
[ system ]
four atoms
[ molecules ]
MOL 1
"""


def _four_atoms(ws):
    mda = pytest.importorskip("MDAnalysis")
    stage = ws.stage("input")
    (stage / "mol.gro").write_text(FOUR_ATOMS_GRO)
    (stage / "mol.top").write_text(FOUR_ATOMS_TOP)
    (stage / "em.mdp").write_text(EM_MDP)
    (stage / "rerun.mdp").write_text(GOOD_MDP + "nstfout = 1\n")
    u = mda.Universe(str(stage / "mol.gro"))
    with mda.Writer(str(stage / "ref.trr"), n_atoms=4) as w:
        for i in range(4):
            u.atoms.positions = u.atoms.positions + 0.1
            u.trajectory.ts.time = float(i)
            w.write(u.atoms)
    return stage


def test_frames_are_minimised_and_reevaluated_under_one_topology(ws_ctx):
    """The two runs a force-field comparison is built from: frames of a reference minimised
    under a topology, measured against the reference; and the same frames re-evaluated for
    the topology's forces. Each is a stage, whichever job it comes to."""
    from tests.conftest import approved_study

    ws = ws_ctx.workspace
    _four_atoms(ws)
    approved_study(ws_ctx, "01_input/mol.gro", "01_input/mol.top", [
        {"name": "frames", "kind": "minimize", "steps": 10,
         "scan": {"parameter": "frames", "values": [0, 2, 3]}},
        {"name": "again", "kind": "single_point", "starts_from": "frames"}])
    frames = {"trajectory": "01_input/ref.trr"}
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "frames",
                                                   "structure": "01_input/mol.gro"}).ok
    job = REGISTRY.call("stage_run", ws_ctx, {"stage": "frames", "inputs": frames,
                                              "walltime_min": 5})
    assert job.ok, job.summary
    done = _wait_for_job(ws_ctx, job.job_id)
    assert done.ok, done.summary
    assert (done.n_requested, done.n_minimised, done.n_converged) == (3, 3, 3)
    minimised = next(a.path for a in done.artifacts if a.role == "trajectory")
    geometry = REGISTRY.call("observable_measure", ws_ctx, {
        "structure": "01_input/mol.gro", "trajectory": minimised, "metric": "pair_distance",
        "pairs": {"C-O": [1, 2]}, "reference": "01_input/ref.trr"})
    assert geometry.ok, geometry.summary
    assert geometry.n_frames == 3 and geometry.mae is not None

    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "again",
                                                   "structure": "01_input/mol.gro"}).ok
    rerun = REGISTRY.call("stage_run", ws_ctx, {"stage": "again", "inputs": frames,
                                                "walltime_min": 5})
    assert rerun.ok, rerun.summary
    collected = _wait_for_job(ws_ctx, rerun.job_id)
    assert collected.ok, collected.summary
    assert {"trajectory", "energy", "log"} <= {a.role for a in collected.artifacts}


# (the submission, whether the partition bills whole 48-core nodes, the (ranks, threads)
# booked when the call leaves them unset): a run fills the node with ranks; the frame
# minimiser is one driver process, so the node goes to its threads
GROMACS_FIXTURES = Path(__file__).parents[1] / "fixtures"
TRPCAGE_TOP = GROMACS_FIXTURES / "gromacs" / "pdb2gmx_trpcage" / "topol.top"
TRPCAGE_ATOMS = 304
TRPCAGE_ALL = 334  # 304 protein atoms and 10 three-site waters
SPLIT_NDX = (f"[ System ]\n{' '.join(map(str, range(1, TRPCAGE_ALL + 1)))}\n"
             f"[ Protein_Other ]\n{' '.join(map(str, range(1, TRPCAGE_ATOMS + 1)))}\n"
             "[ Water_and_ions ]\n"
             f"{' '.join(map(str, range(TRPCAGE_ATOMS + 1, TRPCAGE_ALL + 1)))}\n")

# (topology, index, force field, define, tc-grps): what a system's own files decide
SYSTEM_FACTS = [
    (GROMACS_FIXTURES / "acetone" / "acetone_equilibrated.top", None, "amber03", "", "System"),
    # pdb2gmx's own topology: the protein's POSRES, not the water's POSRES_WATER
    (TRPCAGE_TOP, None, "amber99sb-ildn", "-DPOSRES", "Protein Non-Protein"),
    # an index that splits the system into solute and Water_and_ions couples those two
    (TRPCAGE_TOP, SPLIT_NDX, "amber99sb-ildn", "-DPOSRES", "Protein_Other Water_and_ions"),
    ('#include "charmm36-jul2022.ff/forcefield.itp"\n', None, "charmm36-jul2022", "",
     "System"),
    ("[ system ]\nno force field named\n", None, None, "", "System"),
]


@pytest.mark.parametrize("top,ndx,family,define,tc_grps", SYSTEM_FACTS)
def test_what_a_systems_own_files_decide(tmp_path, top, ndx, family, define, tc_grps):
    from myrmidon.engines.gromacs.system import (
        coupling_groups,
        forcefield,
        restraint_defines,
        starting_state,
    )

    if isinstance(top, str):
        (tmp_path / "topol.top").write_text(top)
        top = tmp_path / "topol.top"
    index = None
    if ndx:
        index = tmp_path / "index.ndx"
        index.write_text(ndx)
    fam = forcefield(top)
    assert (fam.value if fam else None) == family
    assert restraint_defines(top, []).value == define
    groups = coupling_groups(top, [], index)
    assert groups.value == tc_grps and groups.origin.startswith("system:coupling_groups")
    # a structure with velocities continues a run; one without starts from rest
    for gro, continues in ((GROMACS_FIXTURES / "gromacs" / "trpcage_water.gro", False),
                           (GROMACS_FIXTURES / "acetone" / "acetone_equilibrated.gro", True)):
        state = {f.key: f.value for f in starting_state(gro)}
        assert state == {"continuation": "yes" if continues else "no",
                         "gen-vel": "no" if continues else "yes"}


# (the make_ndx commands, a repeated group name is said): a rename is said with what the
# group holds, and a name two groups share is a finding; commands written as one quoted
# string, a line each, are the same commands
RENAMES = [
    (["name 1 Renamed"], False),
    (["name 1 water"], True),  # GROMACS matches a group's name whatever its case
    ("'name 1 Renamed'", False),
    ('"r SOL\nname 1 Renamed"', True),  # both ran: r SOL makes a second SOL
    (['"SOL"', "name 1 Renamed"], True),  # a quoted name copies that group: a second SOL
]


@pytest.mark.parametrize("commands,repeated", RENAMES)
def test_a_rename_is_said_with_what_the_group_holds(ws_ctx, solvated, commands, repeated):
    obs = REGISTRY.call("gmx_make_ndx", ws_ctx, {
        "structure": solvated["structure"], "commands": commands, "output": "index.ndx"})
    assert obs.ok, obs.summary
    assert obs.summary.count("renamed") == 1 and obs.summary.count(" to ") == 1
    assert "atoms)" in obs.summary.split("renamed")[1]
    assert ("NDX_GROUP_NAME_REPEATED" in {f.id for f in obs.findings}) is repeated


# (the run's integrator, what the walltime kill advises): a minimisation has no checkpoint
STOPPED = [("md", "continue_run"), ("steep", "writes no checkpoint")]


@pytest.mark.parametrize("integrator,advice", STOPPED)
def test_a_run_killed_at_its_walltime_is_told_how_to_go_on(tmp_path, integrator, advice):
    from myrmidon.engines.gromacs.parse import walltime_diagnosis

    text = f"   integrator                     = {integrator}\n   nsteps = 50000\n"
    log = tmp_path / "run.log"
    log.write_text(text)
    assert advice in walltime_diagnosis(log, text)


def test_a_continuation_extends_the_run_input_that_wrote_the_checkpoint(ws_ctx):
    """Each extension goes further than the last; preparing the stage afresh starts a new line,
    and what continues then is that one, never an older extension in the same directory."""
    import re

    from tests.conftest import NVT_STAGE, approved_study, prepare_and_run, wait_for

    ws = ws_ctx.workspace
    stage = ws.stage("input")
    (stage / "conf.gro").write_text(
        "mock\n 3\n" + "    1ALA     CA    1   0.1   0.2   0.3\n" * 3 + "   2.5 2.5 2.5\n")
    (stage / "topol.top").write_text("[ system ]\nMock\n[ molecules ]\nProtein 1\n")
    approved_study(ws_ctx, "01_input/conf.gro", "01_input/topol.top", [NVT_STAGE])

    def steps_of(submitted):
        dump = ws.resolve(submitted.command[submitted.command.index("-s") + 1]).read_text()
        return int(re.search(r"nsteps\s*=\s*(\S+)", dump).group(1))

    refused = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt", "extend_ps": 1.0})
    assert not refused.ok
    _, first = prepare_and_run(ws_ctx, "nvt", {"structure": "01_input/conf.gro"})
    assert wait_for(ws_ctx, first.job_id).ok
    miswritten = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt", "extend_ps": 1.0})
    assert miswritten.error.code == "STAGE_EXTEND_WITHOUT_CONTINUE"
    reach = []
    for _ in range(2):
        extended = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt", "continue_run": True,
                                                       "extend_ps": 1.0})
        assert extended.ok, extended.summary
        assert wait_for(ws_ctx, extended.job_id).ok
        reach.append(steps_of(extended))
    assert reach[1] > reach[0] > NVT_STAGE["steps"]
    _, again = prepare_and_run(ws_ctx, "nvt", {"structure": "01_input/conf.gro"})
    assert wait_for(ws_ctx, again.job_id).ok
    goes_on = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt", "continue_run": True})
    assert goes_on.ok, goes_on.summary
    assert steps_of(goes_on) == NVT_STAGE["steps"]


# (what an output is named, the stage the tool writes into, the file — or refused): the
# file name, into the tool's own stage, whatever directory the call names with it — but
# never through another directory onto a file the stage already holds ("held.gro")
OUTPUTS = [("em.tpr", "01_em", "01_em/em.tpr"), ("01_em/em.tpr", "01_em", "01_em/em.tpr"),
           ("02_nvt/em.tpr", "01_em", "01_em/em.tpr"), ("a/01_em/em.tpr", "01_em",
                                                       "01_em/em.tpr"), ("..", "01_em", None),
           ("held.gro", "01_em", "01_em/held.gro"), ("02_nvt/held.gro", "01_em", None)]


@pytest.mark.parametrize("name,stage,written", OUTPUTS)
def test_an_output_is_written_into_the_tools_own_stage(tmp_path, name, stage, written):
    from myrmidon.engines.gromacs.common import stage_output
    from myrmidon.errors import MyrmidonError

    (tmp_path / stage).mkdir()
    (tmp_path / stage / "held.gro").write_text("")
    if written is None:
        with pytest.raises(MyrmidonError):
            stage_output(tmp_path / stage, name)
    else:
        assert stage_output(tmp_path / stage, name) == tmp_path / written


def test_the_parts_of_a_system_are_read_from_its_topology(tmp_path):
    from myrmidon.engines.gromacs.system import subsystems_of

    (tmp_path / "topol.top").write_text(
        "[ moleculetype ]\nProtein_chain_A 3\n[ atoms ]\n"
        "1 CT 1 ALA CA 1 0.1 12\n2 CT 1 ALA CB 2 0.0 12\n"
        "[ moleculetype ]\nLIG 3\n[ atoms ]\n1 c3 1 LIG C1 1 0.5 12\n2 c3 1 LIG C2 2 0.5 12\n"
        "[ moleculetype ]\nPOPC 3\n[ atoms ]\n1 c3 1 POPC C1 1 0 12\n2 c3 1 POPC C2 1 0 12\n"
        '#include "heme.itp"\n#include "ions.itp"\n'
        "[ system ]\nx\n[ molecules ]\n"
        "Protein_chain_A 1\nLIG 2\nPOPC 64\nHEM 1\nSOL 100\nNA 3\nCL 2\n")
    parts = subsystems_of(tmp_path / "topol.top", [])
    assert {k: v["kind"] for k, v in parts.items()} == {
        "protein": "protein", "ligand": "ligand", "membrane": "membrane",
        "cofactor": "cofactor", "solvent": "solvent", "ions": "ions"}
    assert parts["ligand"]["charge"] == 2.0  # both copies
    assert parts["cofactor"]["charge"] is None  # its definition was not read
    assert parts["ions"]["molecules"] == ["NA", "CL"]


BLOCK = "           Step           Time\n          {:d}       {:.5f}\n"
# (the log, its job's walltime) -> ns/day: a fresh run's segment from zero; a continuation
# appended after its parent's log, its own segment only
PACES = [
    ("There are: 33506 Atoms\n" + BLOCK.format(0, 0.0) + BLOCK.format(25000, 50.0), 7.2),
    ("There are: 33506 Atoms\n" + BLOCK.format(0, 0.0) + BLOCK.format(5000000, 10000.0)
     + "Reading checkpoint file md.cpt\n" + BLOCK.format(5000000, 10000.0)
     + BLOCK.format(5025000, 10050.0), 7.2),
    # a minimisation stopped at its walltime: its "Time" counts steps, no pace
    ("There are: 33506 Atoms\nInput Parameters:\n   integrator                     = steep\n"
     + BLOCK.format(0, 0.0) + BLOCK.format(10995, 10995.0), None),
]


@pytest.mark.parametrize("text,ns_per_day", PACES)
def test_a_run_stopped_at_its_walltime_measures_the_pace_its_retry_is_sized_from(
        tmp_path, text, ns_per_day):
    from myrmidon.execution.backend import Resources
    from myrmidon.execution.pace import stopped_pace

    log = tmp_path / "md.log"
    log.write_text(text)
    pace = stopped_pace(log, Resources(ntasks=48, walltime_min=10), "here")
    if ns_per_day is None:
        assert pace is None
    else:
        assert (pace.atoms, pace.cores, round(pace.ns_per_day, 2)) == (33506, 48, ns_per_day)
