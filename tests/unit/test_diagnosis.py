"""What a CPMD/MiMiC run's own output says, parsed: the failure signature by its stable id,
and the numbers of a run that worked. Real logs are fixtures where we have them; the inline
snippets are copied from real runs (the comments say which kind of run).
"""

from pathlib import Path

import pytest

from myrmidon.core.tool import REGISTRY
from myrmidon.engines.mimic.diagnostics import parse_cpmd_output

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
CPMD = FIXTURES / "cpmd"
SINGLEPOINT = FIXTURES / "cpmd_singlepoint"


def _fixture(path):
    return lambda: path.read_text()


def _text(s):
    return lambda: s


def _diagnose(ws_ctx, cpmd_out, err=None, local_error=None):
    stage = Path(ws_ctx.workspace.stage("qmmm"))
    (stage / "cpmd.out").write_text(cpmd_out())
    if err:
        (stage / "cpmd.err").write_text(err())
    if local_error:
        (stage / "LocalError-0-0-0.log").write_text(local_error())
    return REGISTRY.call("mimic_diagnose", ws_ctx, {"stage": "qmmm"})


# (cpmd.out, stderr, LocalError log, the signature). A generic line that comes after the
# cause — an MPI abort, a killed task — must not stand in for it.
SIGNATURES = [
    (_fixture(CPMD / "multiplicity_without_lsd.out"), None,
     _fixture(CPMD / "multiplicity_without_lsd.LocalError.log"),
     "CPMD_MULTIPLICITY_WITHOUT_LSD"),
    (_text(" ATRHO| CHARGE(R-SPACE):   24.0\n"), _fixture(CPMD / "segfault_tight_cell.err"),
     None, "CPMD_SEGFAULT"),
    (_text(" MIMIC INTERFACE\n"), _text(
        "[node01:1931243:0:1931243] Caught signal 8 (Floating point exception: "
        "integer divide by zero)\n==== backtrace (tid:1931240) ====\n"
        " 1 0x0000000000495e34 proppt_utils_mp_propcal_()  ???:0\n"), None,
     "CPMD_PROPCAL_SIGFPE"),
    (_text(" MIMIC INTERFACE\n"), _text(
        "Program:    MiMiCCommLib version:    2.0.2\n"
        "Source file:    .../mcl/src/transport/MPMDTransport.cpp line:    64\n"
        "Unidentified client: /work/run/\n"), None, "MIMIC_PATHS_NOT_CWD"),
    # CPMD died in SHAKE before the handshake; GROMACS then reported an MPI failure
    (_text(" PCG DID NOT CONVERGE AFTER SHAKE_CG_ITER ITERATIONS\n"),
     _text("MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD\n"),
     _text(" error message: NEW_SHAKE| SHAKE failed\n"), "CPMD_SHAKE_WRONG_GEOMETRY"),
    # a rank killed after the plane-wave setup is memory
    (_text(" NUMBER OF PLANE WAVES FOR DENSITY CUTOFF:        6150000\n"),
     _text("srun: error: n1: tasks 0-41: Terminated\nsrun: error: n1: task 17: Killed\n"),
     None, "CPMD_KILLED_OUT_OF_MEMORY"),
    (_text(" MIMIC INTERFACE\n"),
     _text("MPI_Allreduce(...): Message truncated; 48 bytes received but buffer size is 4\n"),
     None, "MIMIC_UNEVEN_RANK_LAYOUT"),
    # the scheduler's time limit is not a CPMD failure
    (_text("   1  0.0001  -36.1  -36.0  0.1  1.2\n"),
     _text("slurmstepd: error: *** STEP 123.0 ON n1 CANCELLED AT 1999-12-31T23:59:59 "
           "DUE TO TIME LIMIT ***\nsrun: error: n1: task 3: Killed\n"),
     None, "SCHEDULER_TIME_LIMIT"),
    (_text("       1  0.00050  300.0  -17.123456  -17.120000\n"
           " ! SELF-CONSISTENCY NOT REACHED IN 100 STEPS\n"), None, None,
     "CPMD_SCF_NOT_CONVERGED"),
    # a single point that hit MAXSTEP exits 0 and writes a reference that looks good
    (_fixture(SINGLEPOINT / "unconverged.out"), None, None, "CPMD_WAVEFUNCTION_STEP_CAP"),
]


@pytest.mark.parametrize("cpmd_out,err,local_error,signature", SIGNATURES,
                         ids=[s[3] for s in SIGNATURES])
def test_a_failed_run_is_named_by_its_signature(ws_ctx, cpmd_out, err, local_error,
                                                signature):
    obs = _diagnose(ws_ctx, cpmd_out, err, local_error)
    assert not obs.ok
    assert obs.diagnosis_id == signature


FINISHED = _fixture(CPMD / "mimic_nve_finished.out")  # a real 1000-step coupled NVE run
SCF_LINE = " ! SELF-CONSISTENCY NOT REACHED IN 100 STEPS\n"
BANNER = " *                      AVERAGED QUANTITIES"


def _finished_with(line):
    return lambda: FINISHED().replace(BANNER, line + BANNER, 1)


# (cpmd.out, LocalError log, ok, diagnosis, the warning): a run that ended normally with
# every step it asked for is not failed by a pattern unless something witnesses an abort
COMPLETED = [
    (FINISHED, None, True, None, None),
    (_finished_with(SCF_LINE), None, True, None, "CPMD_SIGNATURE_IN_FINISHED_LOG"),
    (_finished_with(SCF_LINE), _text(" error message: SCF failed\n"), False,
     "CPMD_SCF_NOT_CONVERGED", None),
]


@pytest.mark.parametrize("cpmd_out,local_error,ok,diagnosis,warning", COMPLETED,
                         ids=["clean", "signature_in_finished_log", "witnessed_by_local_error"])
def test_a_completed_run_outranks_a_signature_unless_an_abort_is_witnessed(
        ws_ctx, cpmd_out, local_error, ok, diagnosis, warning):
    obs = _diagnose(ws_ctx, cpmd_out, local_error=local_error)
    assert obs.ok is ok, obs.summary
    assert obs.diagnosis_id == diagnosis
    assert obs.md_steps_completed == obs.md_steps_requested == 1000
    assert [w.id for w in obs.warnings] == ([warning] if warning else [])
    if diagnosis or warning:
        assert obs.signature_span and "SELF-CONSISTENCY" in obs.signature_span


def test_the_timing_table_of_a_finished_run_is_not_an_abort():
    """The end-of-run timing table names every routine that ran, SHAKE's among them."""
    facts = parse_cpmd_output(FINISHED())
    assert "new_shake" in FINISHED()
    assert facts["finished_normally"] and "signature_id" not in facts


@pytest.mark.parametrize("mode,signature", [
    ("lsd", "CPMD_MULTIPLICITY_WITHOUT_LSD"),
    ("segv", "CPMD_SEGFAULT"),
    ("scf", "CPMD_SCF_NOT_CONVERGED"),
])
def test_the_mock_fails_the_way_the_real_engine_does(ws_ctx, monkeypatch, mode, signature):
    """Every mock run stands on this: a mock failure mode reads as the real signature."""
    from tests.unit.test_mimic import _full_mimic_setup, _wait

    run_inp, tpr = _full_mimic_setup(ws_ctx)
    monkeypatch.setenv("MOCK_CPMD_FAIL", mode)
    job = REGISTRY.call("mimic_run", ws_ctx, {"cpmd_input": run_inp, "tpr": tpr})
    assert _wait(ws_ctx, job.job_id).state == "FAILED"
    diag = REGISTRY.call("mimic_diagnose", ws_ctx, {})
    assert not diag.ok and diag.diagnosis_id == signature


def test_a_running_bomd_log_reads_its_energies_and_timings():
    facts = parse_cpmd_output((CPMD / "acetone_bomd_running.out").read_text())
    # values hand-checked against the real CPMD 4.3 log
    assert facts["md_steps_completed"] == 2
    assert facts["energy_first"] == pytest.approx(-44.61803)  # EKS, not ECLASSIC
    assert facts["energy_last"] == pytest.approx(-44.61965)
    assert facts["eclassic_last"] == pytest.approx(-44.61750)
    assert facts["scf_last_iterations"] >= 1
    assert facts["t_qm_last"] == pytest.approx(2.81) and facts["t_mm_last"] == pytest.approx(0.42)
    assert not facts["finished_normally"] and "diagnosis" not in facts


def test_a_converged_single_point_reads_as_a_run_that_worked(ws_ctx):
    """A converged single point's numbers, parsed from its log."""
    obs = _diagnose(ws_ctx, _fixture(SINGLEPOINT / "converged.out"))
    assert obs.ok, obs.summary
    assert obs.wavefunction_converged is True
    assert obs.scf_iterations == 24
    assert obs.gemax_last == pytest.approx(7.504e-6)
    assert obs.gemax_criterion == pytest.approx(1.0e-5)
    assert obs.energy_total == pytest.approx(-44.93909977)
    assert obs.md_steps_completed is None  # a single point is not dynamics


def test_the_step_cap_is_seen_though_the_run_finished(ws_ctx):
    obs = _diagnose(ws_ctx, _fixture(SINGLEPOINT / "unconverged.out"))
    assert obs.finished_normally and not obs.ok  # finishing is not the test
    assert obs.wavefunction_converged is False
    assert obs.scf_iterations == 1 and obs.scf_max_iterations == 1
    assert obs.gemax_last > obs.gemax_criterion


def test_an_md_log_still_parses_as_dynamics(ws_ctx):
    """The wavefunction-optimisation parser must not touch a BO-MD log: its table is
    `NFI TEMPP EKS ...`, not `NFI GEMAX CNORM ETOT DETOT`."""
    obs = _diagnose(ws_ctx, _text(
        "     NFI    TEMPP       EKS      ECLASSIC          DIS     TCPU\n"
        "       1   300.00   -36.5177   -44.9391   0.000E+00     1.00\n"
        "       2   299.10   -36.5180   -44.9390   1.000E-04     1.00\n"
        " CPU TIME : 0 HOURS 0 MINUTES 2.00 SECONDS\n PROGRAM CPMD ENDED AT: x\n"))
    assert obs.md_steps_completed == 2
    assert obs.wavefunction_converged is None and obs.scf_iterations is None


def test_a_log_nothing_can_be_read_from_is_not_a_run_that_worked(ws_ctx):
    """A log that ended normally but holds no step table is not a run that worked: a
    verdict of "finished, nothing parsed" would send a person to read the log by hand."""
    obs = _diagnose(ws_ctx, _text(
        "PROGRAM CPMD STARTED AT: x\n CPU TIME : 0 HOURS 0 MINUTES 1.00 SECONDS\n"
        " PROGRAM CPMD ENDED AT: x\n"))
    assert not obs.ok


# (real CPMD 4.3 MiMiC log, what controls the ionic temperature, whether a drift is read).
# ECLASSIC is conserved only where the header says the temperature is not controlled.
ION_CONTROL = [
    ("mimic_nve_test.out", None, True),
    ("mimic_heating_berendsen.out", "Berendsen-style thermostat", False),
    ("nose_steps.out", "not stated in the header", False),  # a log cut above its table
]


@pytest.mark.parametrize(("log", "control", "drift"), ION_CONTROL)
def test_energy_drift_is_read_only_where_energy_is_conserved(log, control, drift):
    facts = parse_cpmd_output((CPMD / log).read_text())
    assert facts["ion_temperature_control"] == control
    assert ("eclassic_drift_ha_per_ps" in facts) is drift
    if drift:
        # 1000 steps of 10 a.u.; the slope cross-checked by an independent fit (awk)
        assert facts["md_time_ps"] == pytest.approx(999 * 10 * 2.4188843265857e-5)
        assert facts["eclassic_drift_ha_per_ps"] == pytest.approx(-3.0245e-4, rel=1e-3)
        assert 0 < facts["eclassic_fluctuation_ha"] < 2e-4
