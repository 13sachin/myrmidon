"""End-to-end GROMACS tool chain against the mock gmx binary."""

import time

import pytest

from myrmidon.core.tool import REGISTRY

GOOD_MDP = (
    "integrator = md\ndt = 0.002\nconstraints = h-bonds\nnsteps = 5000\n"
    "tcoupl = v-rescale\nref_t = 300\ntau_t = 0.1\ntc_grps = System\n"
    "coulombtype = PME\nrcoulomb = 1.0\nrvdw = 1.0\n"
)
# Same but no thermostat: plants exactly one warn-level finding to acknowledge.
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


def test_solvating_an_already_solvated_box_is_a_finding(ws_ctx):
    """Stage A's entry is the bare molecule, and the shipped equilibrated box is stage
    B's: solvating the box again changes the system silently."""
    import shutil

    fixtures = __import__("pathlib").Path(__file__).parent.parent / "fixtures" / "acetone"
    ws = ws_ctx.workspace
    stage = ws.stage("input")
    for name in ("acetone_equilibrated.gro", "acetone_stageA_entry.gro",
                 "acetone_stageA_entry.top"):
        shutil.copy(fixtures / name, stage / name)
    topology = ws.relpath(stage / "acetone_stageA_entry.top")

    refused = REGISTRY.call("gmx_solvate", ws_ctx, {
        "structure": ws.relpath(stage / "acetone_equilibrated.gro"), "topology": topology,
    })
    assert not refused.ok
    assert [w.id for w in refused.unacknowledged] == ["SOLVATE_ALREADY_SOLVATED"]
    assert "506" in refused.unacknowledged[0].message  # counted, not guessed

    entry = REGISTRY.call("gmx_solvate", ws_ctx, {
        "structure": ws.relpath(stage / "acetone_stageA_entry.gro"), "topology": topology,
    })
    assert entry.ok, entry.summary
    assert "already_solvated" in entry.checked  # the check ran and said nothing


EM_MDP = (
    "integrator = steep\nemtol = 100.0\nemstep = 0.01\nnsteps = 50000\n"
    "cutoff-scheme = Verlet\nnstlist = 1\ncoulombtype = PME\nrcoulomb = 1.0\n"
    "rvdw = 1.0\npbc = xyz\n"
)


def test_a_minimisation_that_cannot_fit_its_walltime_is_a_finding(ws_ctx):
    """A ns/day ceiling says nothing about a minimisation, so judged by it a 10-minute
    booking for a 22 378-atom box passes without remark. Its size is steps x atoms."""
    ws = ws_ctx.workspace
    stage = ws.stage("run")
    (stage / "mini.mdp").write_text(EM_MDP)

    def tpr_for(n_atoms):
        gro = stage / f"box{n_atoms}.gro"
        atoms = "".join(f"    1ACT     C1{i + 1:5d}   0.1   0.2   0.3\n"
                        for i in range(n_atoms))
        gro.write_text(f"mock\n{n_atoms:5d}\n{atoms}   2.5 2.5 2.5\n")
        (stage / "topol.top").write_text("[ system ]\nMock\n[ molecules ]\nACT 1\n")
        out = REGISTRY.call("gmx_grompp", ws_ctx, {
            "mdp": ws.relpath(stage / "mini.mdp"), "structure": ws.relpath(gro),
            "topology": ws.relpath(stage / "topol.top"), "stage": "run",
            "output": f"mini{n_atoms}.tpr",
        })
        assert out.ok, out.summary
        return next(a.path for a in out.artifacts if a.role == "tpr")

    refused = REGISTRY.call("gmx_mdrun", ws_ctx, {
        "tpr": tpr_for(22378), "deffnm": "big", "stage": "run", "walltime_min": 10,
    })
    assert not refused.ok
    assert "EM_WALLTIME_INSUFFICIENT" in [w.id for w in refused.unacknowledged]
    assert "22378 atoms" in refused.unacknowledged[0].message

    # the box the tutorial actually means fits, and says so as an info line
    fits = REGISTRY.call("gmx_mdrun", ws_ctx, {
        "tpr": tpr_for(1525), "deffnm": "small", "stage": "run", "walltime_min": 10,
    })
    assert fits.ok, fits.summary
    assert "walltime_covers_run" in fits.checked
    assert "EM_LENGTH" in [w.id for w in fits.advisory]
    # and it is not judged as if it were dynamics
    assert "RUN_LENGTH" not in [w.id for w in fits.advisory]


def test_gmx_energy_stats(ws_ctx):
    ws = ws_ctx.workspace
    run_stage = ws.stage("run")
    (run_stage / "md.edr").write_text("MOCKEDR\n")
    obs = REGISTRY.call(
        "gmx_energy",
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


def test_mdrun_account_is_booked_and_named(ws_ctx):
    """One user, several allocations: a submission may name the account it bills, the
    job spec carries it to the backend, and the budget line says so before the run."""
    from myrmidon.engines.gromacs.run import MdrunParams
    from myrmidon.execution.accounting import budget_warnings

    params = MdrunParams(tpr="x.tpr", account="ant1qmmm")
    res = params.resources()
    assert res.account == "ant1qmmm"
    assert MdrunParams(tpr="x.tpr").resources().account is None
    _, findings = budget_warnings(res, ws_ctx.config.compute, "slurm")
    assert "billed to account ant1qmmm" in findings[0].message
    _, findings = budget_warnings(MdrunParams(tpr="x.tpr").resources(),
                                  ws_ctx.config.compute, "slurm")
    assert "billed to" not in findings[0].message




# (how the mock mdrun fails, what the diagnosis must say). GROMACS crash patterns carry
# no stable id, so the parsed sentence is what there is to check. A job the scheduler
# kills is diagnosed by the one fact that decides the next step: how far it got, and that
# it can continue.
CRASHES = [
    ("lincs", ["LINCS"]),
    ("walltime", ["killed at its walltime", "10995 of 50000", "checkpoint_continue"]),
]


@pytest.mark.parametrize("mode,says", CRASHES)
def test_a_failed_mdrun_is_diagnosed_from_its_log(ws_ctx, monkeypatch, mode, says):
    ws = ws_ctx.workspace
    run_stage = ws.stage("run")
    (run_stage / "good.mdp").write_text(GOOD_MDP)
    prep = REGISTRY.call("gmx_pdb2gmx", ws_ctx, {"structure": _write_pdb(ws),
                                                 "forcefield": "amber99sb-ildn",
                                                 "water": "tip3p"})
    tpr = REGISTRY.call("gmx_grompp", ws_ctx, {
        "mdp": ws.relpath(run_stage / "good.mdp"),
        "structure": next(a.path for a in prep.artifacts if a.role == "structure"),
        "topology": next(a.path for a in prep.artifacts if a.role == "topology"),
    })
    assert tpr.ok, tpr.summary
    monkeypatch.setenv("MOCK_GMX_FAIL", mode)
    submitted = REGISTRY.call("gmx_mdrun", ws_ctx, {
        "tpr": next(a.path for a in tpr.artifacts if a.role == "tpr")})
    results = _wait_for_job(ws_ctx, submitted.job_id)
    assert results.state == "FAILED" and not results.ok
    for phrase in says:
        assert phrase in results.diagnosis


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
    found = {w.id for w in obs.warnings}
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


def test_make_ndx_refuses_an_index_it_did_not_build_as_asked(ws_ctx):
    ws = ws_ctx.workspace
    stage = ws.stage("input")
    (stage / "conf.gro").write_text((GMX_FIXTURES / "trpcage_water.gro").read_text())
    conf = ws.relpath(stage / "conf.gro")
    good = REGISTRY.call("gmx_make_ndx", ws_ctx, {"structure": conf,
                                                  "commands": ["r LYS & a NZ", "1 | 13"]})
    assert good.ok, good.summary
    assert [(g.name, g.atoms) for g in good.created] == [("LYS_&_NZ", 1), ("Protein_SOL", 334)]
    for commands in (["1 | 40", "name 15 LYS_NZ"], ["name 40 X"], ["r XYZ"]):
        bad = REGISTRY.call("gmx_make_ndx", ws_ctx, {"structure": conf, "commands": commands})
        assert not bad.ok and {w.id for w in bad.warnings} == {"NDX_COMMAND_REJECTED"}


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
    obs = REGISTRY.call("series_convergence", ws_ctx, {"data": ws.relpath(path)})
    assert obs.ok is ok, obs.summary
    if ok:
        assert obs.series[0].criterion_met is None and obs.series[0].effective_samples > 10
