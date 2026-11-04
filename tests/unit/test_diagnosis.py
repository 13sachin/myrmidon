"""What a CPMD/MiMiC run's own output says, parsed: the numbers of a run that worked, and,
for one that did not, the engine's own words handed back for the reader. Real logs are
fixtures where we have them; the inline snippets are copied from real runs (the comments say
which kind of run).
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


def _diagnose(ws_ctx, cpmd_out, err=None, local_error=None, dipole=None):
    stage = Path(ws_ctx.workspace.stage("qmmm"))
    (stage / "cpmd.out").write_text(cpmd_out())
    if dipole:
        (stage / "DIPOLE").write_text(dipole())
    if err:
        (stage / "cpmd.err").write_text(err())
    if local_error:
        (stage / "LocalError-0-0-0.log").write_text(local_error())
    return REGISTRY.call("mimic_diagnose", ws_ctx, {"stage": "qmmm"})


# (cpmd.out, stderr, LocalError log, the engine's words the reader finds in log_tail,
# CPMD's own error message when it left one). Nothing names the cause: the lines do.
FAILED = [
    (_fixture(CPMD / "multiplicity_without_lsd.out"), None,
     _fixture(CPMD / "multiplicity_without_lsd.LocalError.log"),
     "INCOMPATIBLE WITH LDA", "Multiplicity requires local spin density"),
    (_text(" ATRHO| CHARGE(R-SPACE):   24.0\n"), _fixture(CPMD / "segfault_tight_cell.err"),
     None, "Caught signal 11", "Caught signal 11"),
    # the line CPMD 4.3's RECPNEW writes for a pseudopotential it cannot find, before the
    # MPI abort that follows it
    (_text(" RECPNEW| ECPFILE NOT FOUND S_MT_BLYP.psp\n"),
     _text("application called MPI_Abort(MPI_COMM_WORLD, 999) - process 0\n"), None,
     "ECPFILE NOT FOUND", None),
    # CPMD died in SHAKE before the handshake; GROMACS then reported an MPI failure
    (_text(" PCG DID NOT CONVERGE AFTER SHAKE_CG_ITER ITERATIONS\n"),
     _text("MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD\n"),
     _text(" error message: NEW_SHAKE| SHAKE failed\n"), "SHAKE_CG_ITER", "SHAKE failed"),
    (_text("       1  0.00050  300.0  -17.123456  -17.120000\n"
           " ! SELF-CONSISTENCY NOT REACHED IN 100 STEPS\n"), None, None,
     "SELF-CONSISTENCY NOT REACHED", None),
]


@pytest.mark.parametrize("cpmd_out,err,local_error,words,error", FAILED)
def test_a_failed_run_returns_the_engines_own_words(ws_ctx, cpmd_out, err, local_error,
                                                     words, error):
    obs = _diagnose(ws_ctx, cpmd_out, err, local_error)
    assert not obs.ok
    assert words in obs.log_tail
    assert ("--- stderr ---" in obs.log_tail) is bool(err)
    if error:
        assert error in obs.cpmd_error and obs.diagnosis.startswith("CPMD aborted")
    else:
        assert obs.cpmd_error is None and obs.diagnosis is None


FINISHED = _fixture(CPMD / "mimic_nve_finished.out")  # a real 1000-step coupled NVE run
SCF_LINE = " ! SELF-CONSISTENCY NOT REACHED IN 100 STEPS\n"
BANNER = " *                      AVERAGED QUANTITIES"


def _finished_with(line):
    return lambda: FINISHED().replace(BANNER, line + BANNER, 1)


#: four steps of a real coupled production's DIPOLE file, in the cell of FINISHED's log
DIPOLE = _fixture(CPMD / "mimic_production.DIPOLE")

# (cpmd.out, LocalError log, DIPOLE file, ok, diagnosis, the dipole's mean in debye): a run
# that ended normally with every step it asked for is judged by its facts and by CPMD's own
# error file, never by a line of text; the dipole is the file's total times the cell volume
COMPLETED = [
    (FINISHED, None, None, True, None, None),
    (FINISHED, None, DIPOLE, True, None, 8.3668),
    (_finished_with(SCF_LINE), None, None, True, None, None),
    (_finished_with(SCF_LINE), _text(" error message: SCF failed\n"), None, False,
     "CPMD aborted: SCF failed", None),
]


@pytest.mark.parametrize("cpmd_out,local_error,dipole,ok,diagnosis,debye", COMPLETED,
                         ids=["clean", "with_dipole", "a_stray_line",
                              "witnessed_by_local_error"])
def test_a_completed_run_is_judged_by_its_facts_and_cpmds_error_file(
        ws_ctx, cpmd_out, local_error, dipole, ok, diagnosis, debye):
    obs = _diagnose(ws_ctx, cpmd_out, local_error=local_error, dipole=dipole)
    assert getattr(obs, "dipole_debye_mean", None) == (
        pytest.approx(debye, abs=1e-3) if debye else None)
    assert obs.mean_temperature_k is not None
    assert obs.ok is ok, obs.summary
    assert obs.diagnosis == diagnosis
    assert obs.md_steps_completed == obs.md_steps_requested == 1000
    assert (obs.log_tail is None) is ok


@pytest.mark.parametrize("mode,words,error", [
    ("lsd", "INCOMPATIBLE WITH LDA", "Multiplicity requires"),
    ("segv", "Caught signal 11", "mimic_short_range"),
    ("scf", "SELF-CONSISTENCY NOT REACHED", None),
    ("pp", "ECPFILE NOT FOUND", None),
])
def test_the_mock_fails_the_way_the_real_engine_does(ws_ctx, monkeypatch, mode, words, error):
    """Every mock run stands on this: a mock failure leaves the real engine's words."""
    from tests.unit.test_mimic import _full_mimic_setup, _wait

    run_inp, tpr = _full_mimic_setup(ws_ctx)
    monkeypatch.setenv("MOCK_CPMD_FAIL", mode)
    job = REGISTRY.call("mimic_run", ws_ctx, {"cpmd_input": run_inp, "tpr": tpr})
    assert _wait(ws_ctx, job.job_id).state == "FAILED"
    diag = REGISTRY.call("mimic_diagnose", ws_ctx, {})
    assert not diag.ok and words in diag.log_tail
    assert (error or "") in (diag.cpmd_error or "")


def test_a_single_point_at_its_step_cap_is_collected_as_a_run_that_did_not_work(
        ws_ctx, monkeypatch):
    """The job ends with exit 0 and a RESTART; its own facts say the wavefunction did not
    converge, so collection does not call it a success, and the log goes with it."""
    from tests.unit.test_mimic import _full_mimic_setup, _wait

    run_inp, tpr = _full_mimic_setup(ws_ctx)
    monkeypatch.setenv("MOCK_CPMD_FAIL", "stepcap")
    job = REGISTRY.call("mimic_run", ws_ctx, {"cpmd_input": run_inp, "tpr": tpr})
    obs = _wait(ws_ctx, job.job_id)
    assert obs.state == "COMPLETED" and not obs.ok
    assert obs.diagnosis and "MAXIMUM NUMBER OF STEP" in obs.log_tail
    assert any(a.role == "restart" for a in obs.artifacts)


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


def test_a_runs_ionic_temperature_is_what_the_next_stage_starts_from(ws_ctx):
    """A heating run's final and tail-mean ionic temperature, as its log states them: the
    numbers a following stage's ramp starts from."""
    obs = _diagnose(ws_ctx, _fixture(CPMD / "mimic_heating_berendsen.out"))
    assert obs.final_temperature_k == pytest.approx(95.5, rel=1e-3)
    assert obs.tail_mean_temperature_k == pytest.approx(95.1667, rel=1e-3)
