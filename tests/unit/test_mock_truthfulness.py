"""The mock engines answer for the system they are given, not for the one they were written
against: a run on a new molecule is only a measurement if the mocks do not lie about it."""

from pathlib import Path

import pytest

from myrmidon.core.tool import REGISTRY

#: an anion and water: its net charge is in its [ atoms ], with no marker to say so
ANION_TOP = """\
[ moleculetype ]
; name nrexcl
ACE 3

[ atoms ]
  1  c   1 ACE C1  1  -0.10  12.01
  2  c   1 ACE C2  2   0.70  12.01
  3  o   1 ACE O1  3  -0.80  16.00
  4  o   1 ACE O2  4  -0.80  16.00

[ system ]
anion in water

[ molecules ]
ACE 1
SOL 200
"""


def _genion(ws_ctx, top_text, **params):
    ws = ws_ctx.workspace
    stage = ws.stage("prep")
    (stage / "topol.top").write_text(top_text)
    (stage / "ions.tpr").write_text("tpr\n")
    obs = REGISTRY.call("gmx_genion", ws_ctx, {
        "tpr": ws.relpath(stage / "ions.tpr"), "topology": ws.relpath(stage / "topol.top"),
        **params})
    assert obs.ok, obs.summary
    top = next(a.path for a in obs.artifacts if a.role == "topology")
    return obs, (ws.root / top).read_text()


# (topology, genion params, the [ molecules ] tail it writes). The charge is counted from
# [ atoms ]; a protein whose force field is not here carries it as a marker, removed after.
GENION = [
    (ANION_TOP, {}, ["SOL 199", "NA 1"]),
    (ANION_TOP, {"positive_ion": "K"}, ["SOL 199", "K 1"]),
    (ANION_TOP.replace("-0.10", "0.90"), {}, ["ACE 1", "SOL 200"]),
    ("; MOCK_NET_CHARGE 1\n[ molecules ]\nSOL 10\n", {}, ["SOL 9", "CL 1"]),
]


@pytest.mark.parametrize("top_text,params,tail", GENION)
def test_the_mock_genion_neutralises_the_charge_the_topology_has(ws_ctx, top_text, params,
                                                                 tail):
    _, top = _genion(ws_ctx, top_text, **params)
    assert top.rstrip().splitlines()[-2:] == tail
    assert "MOCK_NET_CHARGE" not in top


# (topology, mdp, the engine's own line). What real grompp 2021.6 refuses, the mock refuses:
# a net charge it can count, and temperature coupling with no groups — an input set that
# lacks tc-grps must not pass the mock when real grompp refuses it.
GROMPP_REFUSES = [
    (ANION_TOP, "integrator = steep\ncoulombtype = PME\n", "non-zero total charge: -1.0"),
    (ANION_TOP.replace("-0.10", "0.90"),
     "integrator = md\ndt = 0.002\nnsteps = 100\nconstraints = h-bonds\n"
     "coulombtype = PME\ntcoupl = v-rescale\ntau_t = 0.1\nref_t = 300\n",
     "Invalid T coupling input: 0 groups, 1 ref-t values and 1 tau-t values"),
]


@pytest.mark.parametrize("top_text,mdp,line", GROMPP_REFUSES)
def test_the_mock_grompp_refuses_what_gromacs_refuses(ws_ctx, top_text, mdp, line):
    ws = ws_ctx.workspace
    stage = ws.stage("prep")
    (stage / "topol.top").write_text(top_text)
    (stage / "conf.gro").write_text("x\n    1\n    1ACE     C1    1   1.0   1.0   1.0\n"
                                    "   2.0 2.0 2.0\n")
    (stage / "run.mdp").write_text(mdp)
    obs = REGISTRY.call("gmx_grompp", ws_ctx, {
        "mdp": ws.relpath(stage / "run.mdp"), "structure": ws.relpath(stage / "conf.gro"),
        "topology": ws.relpath(stage / "topol.top"), "output": "run.tpr"})
    assert not obs.ok
    assert line in (obs.log_tail or "")  # the engine's own line


def test_the_mock_energy_values_mean_what_their_term_says(ws_ctx):
    """A term-blind mock would give -4197.5 K for Temperature and fail a model doing
    everything right. Potential keeps its pinned drift; Temperature is a temperature."""
    ws = ws_ctx.workspace
    run_stage = ws.stage("run")
    (run_stage / "md.edr").write_text("MOCKEDR\n")
    obs = REGISTRY.call("gmx_energy", ws_ctx,
                        {"edr": ws.relpath(run_stage / "md.edr"),
                         "terms": ["Temperature", "Potential", "Pressure"]})
    assert obs.ok, obs.summary
    assert 250 < obs.terms["Temperature"].mean < 350
    assert abs(obs.terms["Potential"].mean - (-4197.5)) < 1e-6  # position-independent
    assert 0.5 < obs.terms["Pressure"].mean < 2


def test_the_mock_pdb2gmx_reports_what_gromacs_reports_for_the_structure(ws_ctx):
    """A mock that says 22 atoms and net charge 0 for every structure makes a model that
    reads it correctly skip neutralising a +1 protein; the mock reports what GROMACS does."""
    import re
    import shutil
    from pathlib import Path

    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    real = (fixtures / "gromacs" / "pdb2gmx_1l2y.out").read_text()
    n_atoms = int(re.search(r"residues with (\d+) atoms", real).group(1))
    charge = float(re.search(r"Total charge (-?[\d.]+)", real).group(1))
    ws = ws_ctx.workspace
    shutil.copy(fixtures / "1l2y_model1.pdb", ws.stage("input") / "1l2y.pdb")
    obs = REGISTRY.call("gmx_pdb2gmx", ws_ctx, {
        "structure": ws.relpath(ws.stage("input") / "1l2y.pdb"),
        "forcefield": "amber99sb-ildn", "water": "tip3p", "ignore_hydrogens": True})
    assert obs.ok, obs.summary
    assert (obs.n_atoms, obs.net_charge) == (n_atoms, charge)


# (coulombtype, whether grompp writes the run input). A net charge refuses only under
# Ewald electrostatics; under a plain cut-off GROMACS notes it and goes on
# (tests/fixtures/gromacs/grompp_netcharge_{cutoff,pme}.err, GROMACS 2021.6).
NET_CHARGE = [("cutoff", True), ("PME", False)]


@pytest.mark.parametrize("coulombtype,writes", NET_CHARGE)
def test_the_mock_grompp_refuses_a_net_charge_only_under_ewald(ws_ctx, coulombtype, writes):
    ws = ws_ctx.workspace
    stage = ws.stage("prep")
    (stage / "topol.top").write_text(ANION_TOP)
    (stage / "conf.gro").write_text("x\n    1\n    1ACE     C1    1   1.0   1.0   1.0\n"
                                    "   2.0 2.0 2.0\n")
    (stage / "run.mdp").write_text(f"integrator = steep\ncoulombtype = {coulombtype}\n")
    obs = REGISTRY.call("gmx_grompp", ws_ctx, {
        "mdp": ws.relpath(stage / "run.mdp"), "structure": ws.relpath(stage / "conf.gro"),
        "topology": ws.relpath(stage / "topol.top"), "output": "run.tpr",
        "acknowledge_warnings": ["MDP_NO_PME", "MDP_CUTOFF_MISMATCH"]})
    assert obs.ok is writes, obs.summary
    if not writes:
        assert [a.tool for a in obs.next_actions] == ["gmx_genion"]


def test_genion_given_a_structure_builds_its_own_run_input(ws_ctx):
    """A model given no run input for genion guesses a path that does not exist. Given the
    solvated structure, genion makes the one it reads, from a sourced input set, and
    neutralises."""
    import json

    ws = ws_ctx.workspace
    stage = ws.stage("prep")
    (stage / "topol.top").write_text(ANION_TOP)
    (stage / "conf.gro").write_text("x\n    1\n    1ACE     C1    1   1.0   1.0   1.0\n"
                                    "   2.0 2.0 2.0\n")
    obs = REGISTRY.call("gmx_genion", ws_ctx, {
        "structure": ws.relpath(stage / "conf.gro"),
        "topology": ws.relpath(stage / "topol.top")})
    assert obs.ok, obs.summary
    assert (obs.n_ions_positive, obs.n_ions_negative) == (1, 0)
    mdp = next(a.path for a in obs.artifacts if a.role == "mdp")
    sidecar = json.loads((ws.root / (mdp + ".myrmidon.json")).read_text())
    assert sidecar["inputset"] == "genion_run_input"
    neither = REGISTRY.call("gmx_genion", ws_ctx, {"topology": ws.relpath(stage / "topol.top")})
    assert (neither.ok, neither.failure) == (False, "invalid_params")


GMX_FIXTURES = Path(__file__).parents[1] / "fixtures" / "gromacs"
MOCK_GMX = Path(__file__).parents[2] / "src/myrmidon/engines/mock/bin/gmx"


def _mock_gmx(*args, stdin):
    import subprocess

    return subprocess.run([str(MOCK_GMX), *map(str, args)], input=stdin, capture_output=True,
                          text=True, check=False)


# (structure, the default groups real make_ndx listed for it): a protein with water, a
# small solute with water, an anion with water and a counter-ion
MAKE_NDX_DEFAULTS = [
    (GMX_FIXTURES / "trpcage_water.gro", "make_ndx_defaults_trpcage_water.groups"),
    (Path(__file__).parents[1] / "fixtures/acetone/acetone_equilibrated.gro",
     "make_ndx_defaults_acetone_solv.groups"),
    (GMX_FIXTURES / "acetate_water.gro", "make_ndx_defaults_acetate_water.groups"),
]


@pytest.mark.parametrize(("structure", "listing"), MAKE_NDX_DEFAULTS)
def test_the_mock_make_ndx_starts_from_the_groups_gromacs_does(tmp_path, structure, listing):
    import re

    out = _mock_gmx("make_ndx", "-f", structure, "-o", tmp_path / "x.ndx", stdin="q\n").stdout
    line = re.compile(r"^ *\d+ .*: +\d+ atoms$", re.MULTILINE)
    assert line.findall(out) == (GMX_FIXTURES / listing).read_text().splitlines()


@pytest.mark.parametrize("commands", ["make_ndx_commands", "make_ndx_missing_group"])
def test_the_mock_make_ndx_skips_and_numbers_as_gromacs_does(tmp_path, commands):
    """make_ndx carries on past a command it cannot apply, and a failed '1 | 40' still
    leaves its copy of group 1 behind, which shifts every later number."""
    from myrmidon.engines.gromacs.index import ndx_groups, rejected_ndx_commands

    stdin = (GMX_FIXTURES / f"{commands}.in").read_text()
    sent = stdin.splitlines()[:-1]
    mock = _mock_gmx("make_ndx", "-f", GMX_FIXTURES / "trpcage_water.gro",
                     "-o", tmp_path / "x.ndx", stdin=stdin)
    real = (GMX_FIXTURES / f"{commands}.out").read_text()
    assert rejected_ndx_commands(mock.stdout, sent) == rejected_ndx_commands(real, sent)
    if commands == "make_ndx_missing_group":
        groups = [f"[ {n} ] {len(m)}" for n, m in ndx_groups((tmp_path / "x.ndx").read_text())]
        real_groups = (GMX_FIXTURES / "make_ndx_missing_group.groups").read_text()
        assert groups == real_groups.splitlines()


def test_the_mock_genrestr_writes_what_gromacs_writes(tmp_path):
    _mock_gmx("make_ndx", "-f", GMX_FIXTURES / "trpcage_water.gro", "-o", tmp_path / "x.ndx",
              stdin=(GMX_FIXTURES / "make_ndx_commands.in").read_text())
    _mock_gmx("genrestr", "-f", GMX_FIXTURES / "trpcage_water.gro", "-n", tmp_path / "x.ndx",
              "-o", tmp_path / "p.itp", "-fc", 1000, 1000, 1000, stdin="15\n")
    assert (tmp_path / "p.itp").read_text() == (GMX_FIXTURES / "genrestr_lys.itp").read_text()
