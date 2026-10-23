"""MiMiC QM/MM tools against mock mimicpy/cpmd.x/mpirun binaries."""

import time
from pathlib import Path

import pytest

from myrmidon.core.tool import REGISTRY
from myrmidon.engines.mimic.cpmd_input import (
    get_keyword,
    parse_cpmd,
    render_cpmd,
    set_keyword,
)

CPMD_SAMPLE = """\
&CPMD
MIMIC
MOLECULAR DYNAMICS BO
&END

&MIMIC
PATHS
1
/tmp/run
BOX
  30.0 30.0 30.0
&END

&SYSTEM
CHARGE
  0
&END
"""


ACETONE_GRO = Path(__file__).parents[1] / "fixtures" / "acetone" / "acetone_stageA_entry.gro"

def test_cpmd_parse_render_roundtrip():
    sections = parse_cpmd(CPMD_SAMPLE)
    assert set(sections) == {"CPMD", "MIMIC", "SYSTEM"}
    assert get_keyword(sections["SYSTEM"], "CHARGE") == "0"
    assert get_keyword(sections["MIMIC"], "PATHS") == "1"
    assert get_keyword(sections["CPMD"], "MAXSTEP") is None

    sections["SYSTEM"] = set_keyword(sections["SYSTEM"], "CHARGE", "-1")
    sections["SYSTEM"] = set_keyword(sections["SYSTEM"], "MULTIPLICITY", "2")
    text = render_cpmd(sections)
    again = parse_cpmd(text)
    assert get_keyword(again["SYSTEM"], "CHARGE") == "-1"
    assert get_keyword(again["SYSTEM"], "MULTIPLICITY") == "2"


def _select_qm(ws_ctx):
    ws = ws_ctx.workspace
    stage = ws.stage("prep")
    for name in ("acetone_stageA_entry.top", "acetone.itp"):  # prepqm sums its charges
        (stage / name.replace("acetone_stageA_entry", "topol")).write_text(
            (ACETONE_GRO.parent / name).read_text())
    # a real acetone: the mock prepqm selects from the coordinates it is given
    (stage / "conf.gro").write_text(ACETONE_GRO.read_text())
    return REGISTRY.call(
        "mimic_select_qm",
        ws_ctx,
        {
            "topology": ws.relpath(stage / "topol.top"),
            "coords": ws.relpath(stage / "conf.gro"),
            "selections": ["add resname is ACT"],
            "padding_nm": 0.4,
        },
    )


def test_run_gate_blocks_unconfirmed_charge(ws_ctx):
    select = _select_qm(ws_ctx)
    cpmd_inp = next(a.path for a in select.artifacts if a.role == "cpmd_input")
    stage = ws_ctx.workspace.stage("qmmm")
    (stage / "mimic.tpr").write_text("MOCKTPR\n")
    refused = REGISTRY.call(
        "mimic_run",
        ws_ctx,
        {"cpmd_input": cpmd_inp, "tpr": ws_ctx.workspace.relpath(stage / "mimic.tpr")},
    )
    assert not refused.ok
    blocking_ids = {w.id for w in refused.blocking}
    # mock prepqm writes no CHARGE keyword into &SYSTEM? it does not — both block
    assert "QM_MULTIPLICITY_UNCONFIRMED" in blocking_ids


def _full_mimic_setup(ws_ctx):
    select = _select_qm(ws_ctx)
    cpmd_inp = next(a.path for a in select.artifacts if a.role == "cpmd_input")
    prepared = REGISTRY.call(
        "mimic_prepare",
        ws_ctx,
        {"cpmd_input": cpmd_inp, "total_charge": 0, "multiplicity": 1, "maxsteps": 5,
         "acknowledge_warnings": ["QM_CHARGE_MULTIPLICITY_UNCONFIRMED"]},
    )
    run_inp = next(a.path for a in prepared.artifacts if a.role == "cpmd_input")
    stage = ws_ctx.workspace.stage("qmmm")
    (stage / "mimic.tpr").write_text("MOCKTPR\n")
    return run_inp, ws_ctx.workspace.relpath(stage / "mimic.tpr")


def _wait(ctx, job_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = REGISTRY.call("job_status", ctx, {"job_id": job_id})
        if st.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
        time.sleep(0.1)
    raise AssertionError("mimic job never finished")


TIGHT_CELL_INPUT = """\
&MIMIC
PATHS
1
/tmp/x
BOX
 30 30 30
&END
&CPMD
MIMIC
MAXSTEP
 5
&END
&SYSTEM
CELL
 5.3 1.5 0.8 0.0 0.0 0.0
CHARGE
 0
&END
&ATOMS
*C_MT_BLYP KLEINMAN-BYLANDER
 LMAX=P
 2
 20.98 19.77 21.52
 26.32 27.53 25.79
&END
"""


# (total charge, multiplicity, MULTIPLICITY written, LSD written). CPMD accepts
# MULTIPLICITY only in a spin-polarised run, so a closed shell omits it — and the confirmed
# value still travels in the sidecar the run gate reads.
SPIN = [(-1, 1, None, False), (0, 2, "2", True)]


@pytest.mark.parametrize("charge,multiplicity,written,lsd", SPIN)
def test_prepare_writes_the_charge_and_the_spin_cpmd_will_accept(ws_ctx, charge,
                                                                 multiplicity, written, lsd):
    import json

    select = _select_qm(ws_ctx)
    cpmd_inp = next(a.path for a in select.artifacts if a.role == "cpmd_input")
    obs = REGISTRY.call("mimic_prepare", ws_ctx, {
        "cpmd_input": cpmd_inp, "total_charge": charge, "multiplicity": multiplicity,
        "maxsteps": 50, "acknowledge_warnings": ["QM_CHARGE_MULTIPLICITY_UNCONFIRMED"]})
    assert obs.ok, obs.summary
    assert (obs.charge, obs.multiplicity, obs.functional, obs.paths) == (
        charge, multiplicity, "BLYP", 1)
    out = next(a.path for a in obs.artifacts if a.role == "cpmd_input")
    sections = parse_cpmd((ws_ctx.workspace.root / out).read_text())
    assert get_keyword(sections["SYSTEM"], "CHARGE") == str(charge)
    assert get_keyword(sections["SYSTEM"], "MULTIPLICITY") == written
    assert any(line.strip() == "LSD" for line in sections["CPMD"]) is lsd
    assert get_keyword(sections["CPMD"], "MAXSTEP") == "50"
    meta = next(a.path for a in obs.artifacts if a.role == "run_metadata")
    assert json.loads((ws_ctx.workspace.root / meta).read_text())["multiplicity"] == multiplicity


def test_prepare_refuses_an_input_that_is_not_a_coupled_run(ws_ctx):
    stage = ws_ctx.workspace.stage("qmmm")
    (stage / "plain.inp").write_text("&CPMD\nOPTIMIZE WAVEFUNCTION\n&END\n")
    obs = REGISTRY.call("mimic_prepare", ws_ctx, {
        "cpmd_input": ws_ctx.workspace.relpath(stage / "plain.inp"),
        "total_charge": 0, "multiplicity": 1})
    assert not obs.ok and not obs.artifacts


def test_the_qm_cell_is_read_in_every_unit_cpmd_takes():
    from myrmidon.engines.mimic.cpmd_input import qm_geometry

    geom = qm_geometry(parse_cpmd(TIGHT_CELL_INPUT))
    assert geom["n_atoms"] == 2
    assert geom["cell_bohr"] == [5.3, 7.95, 4.24]  # relative to the first edge
    assert geom["min_vacuum_bohr"] < 0.1
    absolute = TIGHT_CELL_INPUT.replace("CELL\n 5.3 1.5 0.8", "CELL ABSOLUTE\n 20.0 20.0 20.0")
    assert qm_geometry(parse_cpmd(absolute))["cell_bohr"] == [20.0, 20.0, 20.0]
    angstrom = qm_geometry(parse_cpmd(absolute.replace("&SYSTEM\n", "&SYSTEM\nANGSTROM\n")))
    assert angstrom["cell_bohr"][0] == pytest.approx(37.79, abs=0.01)


# (a CPMD input as written, the check that ran, the finding that blocks it)
CPMD_GATES = [
    ("&MIMIC\nPATHS\n1\n/tmp/x\nBOX\n 30 30 30\n&END\n"
     "&CPMD\nMIMIC\nMAXSTEP\n 5\n&END\n&SYSTEM\nCHARGE\n 0\nMULTIPLICITY\n 1\n&END\n",
     None, "CPMD_MULTIPLICITY_WITHOUT_LSD"),
    # QM atoms with no vacuum around them segfault in the short-range coupling
    (TIGHT_CELL_INPUT, "qm_cell", "QM_CELL_NO_VACUUM"),
]


@pytest.mark.parametrize("text,check,finding", CPMD_GATES)
def test_a_cpmd_input_cpmd_would_reject_is_blocked_before_it_runs(ws_ctx, text, check,
                                                                  finding):
    stage = ws_ctx.workspace.stage("qmmm")
    (stage / "run.inp").write_text(text)
    (stage / "run.inp.myrmidon.json").write_text('{"multiplicity": 1}')
    report = REGISTRY.call("file_check", ws_ctx,
                           {"path": ws_ctx.workspace.relpath(stage / "run.inp")})
    assert not report.ok and report.file_kind == "cpmd"
    assert finding in {w.id for w in report.blocking}
    if check:
        assert check in report.checked


# --- segment chains: a continuation restarts from its predecessor -------------------


def _restarting_input(ws_ctx, run_inp):
    """The same reviewed input with a RESTART line, as an input set's continuation stage
    renders it (mimic_acetone_tutorial's nve_test, heating and production all carry one)."""
    path = ws_ctx.workspace.root / run_inp
    text = path.read_text().replace(
        "&CPMD", "&CPMD\n    RESTART COORDINATES VELOCITIES WAVEFUNCTION", 1
    )
    path.write_text(text)
    return run_inp


def test_a_restart_without_its_file_is_refused_before_it_is_spent(ws_ctx):
    """An input that asks to RESTART with no restart_from is refused before anything is
    booked: CPMD would stop in ZHRWF with FILE NOT FOUND after the job had been charged."""
    run_inp, tpr = _full_mimic_setup(ws_ctx)
    _restarting_input(ws_ctx, run_inp)
    refused = REGISTRY.call("mimic_run", ws_ctx, {"cpmd_input": run_inp, "tpr": tpr})
    assert not refused.ok
    assert "CPMD_RESTART_FILE_MISSING" in {w.id for w in refused.blocking}
    assert "restart_available" in refused.checked


def test_a_restart_file_with_no_restart_line_is_a_finding(ws_ctx):
    run_inp, tpr = _full_mimic_setup(ws_ctx)
    first = REGISTRY.call("mimic_run", ws_ctx, {"cpmd_input": run_inp, "tpr": tpr})
    done = _wait(ws_ctx, first.job_id)
    restart = next(a.path for a in done.artifacts if a.role == "restart")
    silent = REGISTRY.call("mimic_run", ws_ctx, {
        "cpmd_input": run_inp, "tpr": tpr, "restart_from": restart,
    })
    assert not silent.ok
    assert "CPMD_RESTART_FILE_UNUSED" in {w.id for w in silent.unacknowledged}


def test_each_submission_runs_in_its_own_directory_and_the_last_run_s_files_survive(ws_ctx):
    """The ledger's hash of a run's log must name a file that still exists after the next
    run in the same stage: CPMD's output names are fixed, so every submission gets a
    numbered directory of its own under the stage."""
    from myrmidon.core.artifacts import sha256_file

    run_inp, tpr = _full_mimic_setup(ws_ctx)
    first = _wait(ws_ctx, REGISTRY.call("mimic_run", ws_ctx, {
        "cpmd_input": run_inp, "tpr": tpr, "deffnm": "annealing"}).job_id)
    log = next(a for a in first.artifacts if a.role == "qm_log")
    restart = next(a.path for a in first.artifacts if a.role == "restart")
    _restarting_input(ws_ctx, run_inp)
    second = REGISTRY.call("mimic_run", ws_ctx, {
        "cpmd_input": run_inp, "tpr": tpr, "deffnm": "nve", "restart_from": restart})
    assert second.ok, second.summary
    second = _wait(ws_ctx, second.job_id)
    log2 = next(a for a in second.artifacts if a.role == "qm_log")

    root = ws_ctx.workspace.root
    here, there = (root / log.path).parent, (root / log2.path).parent
    assert here != there and here.parent == there.parent == ws_ctx.workspace.stage("qmmm")
    assert here.name == "01_annealing" and there.name == "02_nve"
    assert sha256_file(root / log.path) == log.sha256
    assert sha256_file(root / restart) == next(
        a.sha256 for a in first.artifacts if a.role == "restart")


def test_a_restart_that_no_longer_hashes_as_recorded_is_refused(ws_ctx):
    """A continuation continues the run the ledger shows only while its restart is the
    file that run's collection recorded."""
    run_inp, tpr = _full_mimic_setup(ws_ctx)
    first = _wait(ws_ctx, REGISTRY.call("mimic_run", ws_ctx,
                                        {"cpmd_input": run_inp, "tpr": tpr}).job_id)
    restart = next(a.path for a in first.artifacts if a.role == "restart")
    (ws_ctx.workspace.root / restart).write_bytes(b"another run's restart\n")
    _restarting_input(ws_ctx, run_inp)
    refused = REGISTRY.call("mimic_run", ws_ctx, {
        "cpmd_input": run_inp, "tpr": tpr, "restart_from": restart})
    assert not refused.ok
    assert "RESTART_NOT_AS_RECORDED" in {w.id for w in refused.blocking}


def test_a_collected_run_that_completed_is_not_failed_by_a_signature(ws_ctx, monkeypatch):
    """job_status on a coupled run that exited 0 with every step it asked for: a pattern
    in its log is a warning carrying the matched line, not the diagnosis."""
    from tests.unit.test_diagnosis import SCF_LINE, _finished_with

    from myrmidon.execution.backend import CoupledCmd, JobSpec, JobState, JobStatus
    from myrmidon.execution.jobstore import open_jobstore

    ws = ws_ctx.workspace
    stage = ws.stage("production")
    for name, text in (("mimic.log", "Step  Time\n 1000  242.0\n"), ("mimic.trr", "x"),
                       ("cpmd.out", _finished_with(SCF_LINE)())):
        (stage / name).write_text(text)
    store = open_jobstore(ws)
    spec = JobSpec(argv=["cpmd.x", "run.inp"], cwd=str(stage),
                   coupled=[CoupledCmd(argv=["gmx", "mdrun", "-deffnm", "mimic"])])
    job_id = store.add("local", "1", spec, label="mimic:mimic",
                       meta={"kind": "mimic", "core_h_estimate": 1.0, "deffnm": "mimic"})
    store.update(job_id, JobState.COMPLETED, 0)
    monkeypatch.setattr(type(ws_ctx.backend), "poll",
                        lambda self, handle, s: JobStatus(state=JobState.COMPLETED,
                                                          exit_code=0))
    obs = REGISTRY.call("job_status", ws_ctx, {"job_id": job_id})
    assert obs.ok and obs.diagnosis_id is None, obs.summary
    assert "CPMD_SIGNATURE_IN_FINISHED_LOG" in {w.id for w in obs.warnings}


def test_a_prepared_input_goes_where_the_path_says(ws_ctx):
    """A bare name lands beside its input (what every recorded graph passes); a path
    is workspace-relative, as every other tool's output is, never resolved against the
    input's directory; a path that escapes the workspace is refused, not a traceback."""
    select = _select_qm(ws_ctx)
    cpmd_inp = next(a.path for a in select.artifacts if a.role == "cpmd_input")
    common = {"cpmd_input": cpmd_inp, "total_charge": 0, "multiplicity": 1,
              "acknowledge_warnings": ["QM_CHARGE_MULTIPLICITY_UNCONFIRMED"]}

    beside = REGISTRY.call("mimic_prepare", ws_ctx, {**common, "output": "bare.inp"})
    assert beside.ok, beside.summary
    made = next(a.path for a in beside.artifacts if a.role == "cpmd_input")
    assert made == str(Path(cpmd_inp).parent / "bare.inp")

    placed = REGISTRY.call("mimic_prepare", ws_ctx,
                           {**common, "output": "01_elsewhere/run.inp"})
    assert placed.ok, placed.summary
    assert (ws_ctx.workspace.root / "01_elsewhere" / "run.inp").is_file()

    escaped = REGISTRY.call("mimic_prepare", ws_ctx, {**common, "output": "../out.inp"})
    assert not escaped.ok and "escapes the workspace" in escaped.summary


def test_a_coupled_run_collects_the_mm_half_it_actually_wrote(ws_ctx, monkeypatch):
    """The MM trajectory of a coupled run is what force matching fits against: `spec.argv`
    is CPMD's, so the `-deffnm` lookup must read the coupled command, not fall through to
    "md", match nothing, and report success with the trajectory absent from the ledger."""
    from myrmidon.core.tool import REGISTRY
    from myrmidon.engines.gromacs.results import _spec_deffnm
    from myrmidon.execution.backend import CoupledCmd, JobSpec, JobState, JobStatus
    from myrmidon.execution.jobstore import open_jobstore

    ws = ws_ctx.workspace
    stage = ws.stage("production")
    for name, text in (("mimic.log", "Step  Time\n 1000  242.0\n"), ("mimic.trr", "x"),
                       ("mimic.edr", "x"), ("cpmd.out", "PROGRAM CPMD ENDED\n")):
        (stage / name).write_text(text)

    store = open_jobstore(ws)
    spec = JobSpec(
        argv=["cpmd.x", "cpmd_run.inp"], cwd=str(stage),
        coupled=[CoupledCmd(argv=["gmx", "mdrun", "-s", "mimic.tpr", "-deffnm", "mimic"])],
    )
    job_id = store.add("local", "1", spec, label="mimic:mimic",
                       meta={"kind": "mimic", "core_h_estimate": 60.0, "deffnm": "mimic"})
    store.update(job_id, JobState.COMPLETED, 0)
    assert _spec_deffnm(store.get(job_id)) == "mimic"
    monkeypatch.setattr(type(ws_ctx.backend), "poll",
                        lambda self, handle, s: JobStatus(state=JobState.COMPLETED,
                                                          exit_code=0))

    obs = REGISTRY.call("job_status", ws_ctx, {"job_id": job_id})

    paths = {a.path for a in obs.artifacts}
    assert any(p.endswith("mimic.trr") for p in paths), sorted(paths)
    assert any(p.endswith("mimic.edr") for p in paths), sorted(paths)
    assert any(p.endswith("cpmd.out") for p in paths), sorted(paths)  # and the QM half
    assert not any(w.id == "MM_OUTPUTS_NOT_FOUND" for w in obs.warnings)


def test_maxstep_advice_depends_on_whether_the_run_is_dynamics(ws_ctx):
    """CPMD's MAXSTEP bounds MD steps under MOLECULAR DYNAMICS and SCF iterations under
    WAVEFUNCTION OPTIMIZATION. The gate asks for it only in dynamics: a single point
    capped at one SCF iteration exits 0 with an unconverged wavefunction and writes
    force-matching reference data from it."""
    from myrmidon.engines.mimic.launch import maxstep_findings

    dynamics = ["MOLECULAR DYNAMICS", "BO", "TIMESTEP", "  10.0"]
    single_point = ["WAVEFUNCTION OPTIMIZATION", "CONVERGENCE ORBITALS", "  1.0e-5"]

    def ids(section):
        return {w.id for w in maxstep_findings(section)}

    # dynamics with no MAXSTEP: a run of unknown length, still worth asking about
    assert "CPMD_MAXSTEP_UNSET" in ids(dynamics)
    # a single point with no MAXSTEP is correct, and must not be nagged into breaking
    assert ids(single_point) == set()
    # but a single point capped at one SCF iteration is the failure itself
    assert "CPMD_SCF_LIMIT_TOO_LOW" in ids([*single_point, "MAXSTEP", "  1"])
    # a sane SCF limit is nobody's business
    assert ids([*single_point, "MAXSTEP", "  200"]) == set()
    # and dynamics with a step count stays quiet
    assert ids([*dynamics, "MAXSTEP", "  1000"]) == set()


def _select(ws_ctx, selections):
    ws = ws_ctx.workspace
    stage = ws.stage("prep")
    for name in ("acetone_stageA_entry.top", "acetone.itp"):  # prepqm sums its charges
        (stage / name.replace("acetone_stageA_entry", "topol")).write_text(
            (ACETONE_GRO.parent / name).read_text())
    (stage / "conf.gro").write_text(ACETONE_GRO.read_text())
    return REGISTRY.call("mimic_select_qm", ws_ctx, {
        "topology": ws.relpath(stage / "topol.top"), "coords": ws.relpath(stage / "conf.gro"),
        "selections": selections, "padding_nm": 0.4})


def test_the_mock_selects_the_atoms_it_was_asked_for(ws_ctx):
    """The mock prepqm answered ten atoms whatever was selected, true only of acetone."""
    obs = _select(ws_ctx, ["add resname is ACT", "delete name is O1 or name is C2"])
    assert obs.ok, obs.summary
    assert obs.qm_atoms == 8
    ndx = next(a.path for a in obs.artifacts if a.role == "index")
    serials = (ws_ctx.workspace.root / ndx).read_text().split("\n")[1].split()
    assert len(serials) == 8
    inp = (ws_ctx.workspace.root / next(
        a.path for a in obs.artifacts if a.role == "cpmd_input")).read_text()
    # &ATOMS holds the selection: carbons and hydrogens, the deleted oxygen gone, and the
    # first carbon at the file's coordinates in bohr (1.212 nm = 22.9035 bohr)
    assert "*C_MT_BLYP.psp\n LMAX=P\n 2\n" in inp and "*H_MT_BLYP.psp" in inp
    assert "*O_MT_BLYP.psp" not in inp and "22.9035" in inp


PREPQM = Path(__file__).parents[1] / "fixtures" / "mimicpy"


def test_skipped_selection_commands_are_read_from_real_prepqm():
    """prepqm prints an error for a command it cannot apply, skips it and exits 0 with the
    rest; the commands it skipped are the ones the region is missing."""
    from myrmidon.engines.mimic.qm_region import rejected_commands

    commands = (PREPQM / "prepqm_rejected_commands.in").read_text().splitlines()[:-1]
    real = rejected_commands((PREPQM / "prepqm_rejected_commands.out").read_text(), commands)
    assert [cmd for cmd, _ in real] == [
        "rm name is H1", "add resname is LIG", "add resname ACT", "add foo is 3"]
    assert rejected_commands(
        (PREPQM / "prepqm_empty_region.out").read_text(), ["add resname is LIG"]
    )[0][0] == "add resname is LIG"


# (selection commands, finding id the call is refused or failed on). The mock skips what
# real prepqm skips, with its messages, so the tool reads both the same way.
SELECTIONS = [
    (["add resname is ACT", "rm name is H1"], "QM_SELECTION_COMMAND_UNKNOWN"),
    (["add resname is ACT", "view"], "QM_SELECTION_COMMAND_UNKNOWN"),
    (["clear resname is ACT"], "QM_SELECTION_COMMAND_UNKNOWN"),
    (["add"], "QM_SELECTION_COMMAND_UNKNOWN"),
    (["add resname is ACT", "add resname is LIG"], "QM_SELECTION_REJECTED"),
    (["add resname ACT"], "QM_SELECTION_REJECTED"),
    (["add foo is 3", "add resname is ACT"], "QM_SELECTION_REJECTED"),
    (["add within 0.3 of resname ACT"], "QM_SELECTION_REJECTED"),  # the mock lacks `within`
    (["add resname is LIG"], "QM_SELECTION_REJECTED"),  # nothing matched, nothing written
]


@pytest.mark.parametrize(("selections", "finding"), SELECTIONS)
def test_a_selection_prepqm_would_skip_is_not_a_qm_region(ws_ctx, selections, finding):
    obs = _select(ws_ctx, selections)
    assert not obs.ok
    assert finding in {w.id for w in [*obs.warnings, *getattr(obs, "blocking", [])]}


# (selection, the charge prepqm writes, the MM charge sum it says it rounded): a whole
# neutral solute sums to an integer; with one atom deleted it does not
CHARGE_ROUNDING = [
    (["add resname is ACT"], 0, None),
    (["add resname is ACT", "delete name is O1"], 1, 0.5864),
]


@pytest.mark.parametrize(("selections", "charge", "rounded_from"), CHARGE_ROUNDING)
def test_the_charge_prepqm_rounded_is_reported_with_its_sum(ws_ctx, selections, charge,
                                                            rounded_from):
    obs = _select(ws_ctx, selections)
    assert obs.ok
    assert (obs.declared_charge, obs.rounded_from) == (charge, rounded_from)
    assert ("QM_CHARGE_ROUNDED" in {w.id for w in obs.warnings}) == (rounded_from is not None)


# (topology, coordinates, the commands of a real prepqm capture in tests/fixtures/mimicpy):
# a solute with one atom deleted, and one residue's side chain cut from a protein. Both
# regions' charges sum to a non-integer, which prepqm rounds into &SYSTEM CHARGE.
PREPQM_CAPTURES = [
    ("acetone/acetone_equilibrated.top", "acetone/acetone_equilibrated.gro",
     "prepqm_rejected_commands"),
    ("mimicpy/trpcage_atoms.top", "gromacs/trpcage_water.gro", "prepqm_lys_side_chain"),
]


@pytest.mark.parametrize("top,coords,capture", PREPQM_CAPTURES)
def test_the_mock_prepqm_selects_and_charges_as_real_prepqm_does(tmp_path, top, coords,
                                                                   capture):
    import subprocess

    from myrmidon.engines.gromacs.index import ndx_groups
    from myrmidon.engines.mimic.qm_region import rejected_commands

    fixtures = Path(__file__).parents[1] / "fixtures"
    stdin = (PREPQM / f"{capture}.in").read_text()
    commands = stdin.splitlines()[:-1]
    mock = Path(__file__).parents[2] / "src/myrmidon/engines/mock/bin/mimicpy"
    proc = subprocess.run(
        [str(mock), "prepqm", "-top", str(fixtures / top), "-coords", str(fixtures / coords),
         "-ff", str(fixtures / "acetone"), "-out", "cpmd.inp", "-ndx", "index.ndx"],
        cwd=tmp_path, input=stdin, capture_output=True, text=True, check=True)
    if (PREPQM / f"{capture}.out").exists():
        real = (PREPQM / f"{capture}.out").read_text()
        assert rejected_commands(proc.stdout, commands) == rejected_commands(real, commands)
    assert proc.stderr.splitlines() == (PREPQM / f"{capture}.err").read_text().splitlines()
    system = parse_cpmd((PREPQM / f"{capture}.system").read_text())["SYSTEM"]
    written = parse_cpmd((tmp_path / "cpmd.inp").read_text())["SYSTEM"]
    assert get_keyword(written, "CHARGE") == get_keyword(system, "CHARGE")
    real_ndx = ndx_groups((PREPQM / f"{capture}.ndx").read_text())
    assert [m for _, m in ndx_groups((tmp_path / "index.ndx").read_text())] == [
        m for _, m in real_ndx]


# (an approved plan step's charge and multiplicity, or None for no plan; the call's;
# whether the call runs unattended). A model stating the QM charge is not the researcher
# confirming it; a plan a person approved with those values is.
SPIN_STATE = [
    (None, (0, 1), False),
    ((0, 1), (0, 1), True),
    ((0, 1), (1, 2), False),
]


@pytest.mark.parametrize("planned,given,runs", SPIN_STATE)
def test_a_qm_charge_and_multiplicity_nobody_approved_needs_a_person(ws_ctx, planned,
                                                                     given, runs):
    select = _select_qm(ws_ctx)
    cpmd_inp = next(a.path for a in select.artifacts if a.role == "cpmd_input")
    if planned is not None:
        step = {"name": "prepare", "tool": "mimic_prepare",
                "values": {"cpmd_input": cpmd_inp, "total_charge": planned[0],
                           "multiplicity": planned[1]}}
        assert REGISTRY.call("plan_write", ws_ctx, {
            "question": "q", "system": "s", "given": [], "steps": [step]}).ok
        assert REGISTRY.call("plan_approve", ws_ctx, {}).ok
    ws_ctx.unattended = True
    obs = REGISTRY.call("mimic_prepare", ws_ctx, {
        "cpmd_input": cpmd_inp, "total_charge": given[0], "multiplicity": given[1]})
    assert obs.ok is runs, obs.summary
    if not runs:
        finding = next(w for w in obs.human_required
                       if w.id == "QM_CHARGE_MULTIPLICITY_UNCONFIRMED")
        assert finding.needs_human


# A stage's run input is the stage's method on the system the QM selection made once: what a
# staged MiMiC run (anneal, NVE test, heat, produce) changes between stages is the method and
# never the system. Each row is a stage of a real staged run: its input made by mimic_prepare
# from the annealing system input and the stage's method must be the input it ran with.
STAGES = Path(__file__).parents[1] / "fixtures" / "mimic_stages"


@pytest.mark.parametrize("stage", ["nve_test", "heating", "production"])
def test_a_stage_input_is_its_method_on_the_system_made_once(ws_ctx, stage):
    from myrmidon.engines.mimic.cpmd_input import parse_cpmd

    here = ws_ctx.workspace.stage(stage)
    for name in ("system_from_prepqm.inp", f"{stage}_method.inp"):
        (here / name).write_text((STAGES / name).read_text())
    rel = ws_ctx.workspace.relpath(here)
    obs = REGISTRY.call("mimic_prepare", ws_ctx, {
        "cpmd_input": f"{rel}/system_from_prepqm.inp", "method": f"{rel}/{stage}_method.inp",
        "total_charge": 0, "multiplicity": 1, "output": "run.inp",
        "acknowledge_warnings": ["QM_CHARGE_MULTIPLICITY_UNCONFIRMED"]})
    assert obs.ok, obs.summary

    def lines(path):
        return {k: [x.strip() for x in v if x.strip()]
                for k, v in parse_cpmd(path.read_text()).items()}

    made, real = lines(here / "run.inp"), lines(STAGES / f"{stage}_run_real.inp")
    assert list(made) == list(real)
    assert made == real
