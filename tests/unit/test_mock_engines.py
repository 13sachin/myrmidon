"""The mock engines ship with the harness and behave the same wherever they run."""

import os
from dataclasses import replace

import pytest

from myrmidon.core.tool import REGISTRY
from myrmidon.engines import mock


def test_the_mock_engines_are_part_of_the_package():
    """`serve --mock` must work from an installed harness, not only from a checkout."""
    import myrmidon

    assert mock.BIN.is_relative_to(os.path.dirname(myrmidon.__file__))
    mock.check_bin()
    assert {"gmx", "cpmd.x", "mimicpy", "sbatch", "squeue", "sacct"} <= {
        p.name for p in mock.BIN.iterdir()}


def test_a_mock_context_reaches_no_real_engine_a_machine_config_names(ws_ctx):
    """A machine config names its engines by absolute path; on the mocks none of them is used."""
    configured = ws_ctx.config.model_copy(update={
        "gromacs_bin": "/opt/real/gmx", "cpmd_bin": "/opt/real/cpmd.x",
        "mimicpy_bin": "/opt/real/mimicpy"})
    with mock.activated(ws_ctx.workspace.root):
        engines = mock.mock_context(replace(ws_ctx, config=configured)).engines
    assert {engines[name].path for name in ("gromacs", "cpmd", "mimicpy")} == {
        str(mock.BIN / binary) for binary in ("gmx", "cpmd.x", "mimicpy")}


def test_a_mock_job_is_seen_running_exactly_once_whatever_the_host(ws_ctx):
    """The logical clock: the poll sequence of a submitted job is a fact of the session,
    not of how fast this machine happened to run the detached script."""
    from tests.conftest import NVT_STAGE, approved_study

    ws = ws_ctx.workspace
    stage = ws.stage("input")
    (stage / "conf.gro").write_text(
        "mock\n 3\n" + "    1ALA     CA    1   0.1   0.2   0.3\n" * 3 + "   2.5 2.5 2.5\n")
    (stage / "topol.top").write_text("[ system ]\nMock\n[ molecules ]\nProtein 1\n")
    stages = [{**NVT_STAGE, "name": f"p{n}", "starts_from": ""} for n in range(2)]
    sequences = []
    with mock.activated(ws.root):
        ctx = mock.mock_context(ws_ctx)
        approved_study(ctx, "01_input/conf.gro", "01_input/topol.top", stages)
        for n in range(2):
            prepared = REGISTRY.call("stage_prepare", ctx, {"stage": f"p{n}",
                                                            "structure": "01_input/conf.gro"})
            assert prepared.ok, prepared.summary
            job = REGISTRY.call("stage_run", ctx, {"stage": f"p{n}", "walltime_min": 5})
            assert job.ok, job.summary
            sequences.append([REGISTRY.call("job_status", ctx, {"job_id": job.job_id}).state
                              for _ in range(3)])
    assert sequences[0] == sequences[1] == ["RUNNING", "COMPLETED", "COMPLETED"]


@pytest.mark.parametrize("fails,exit_code", [(False, 0), (True, 7)])
def test_the_mock_srun_runs_a_multi_prog_step_as_the_real_one_does(tmp_path, fails, exit_code):
    """A coupled launch's wrappers write a program's stdout only from its first rank."""
    import subprocess

    from myrmidon.config import SlurmDefaults
    from myrmidon.execution.backend import CoupledCmd, JobSpec, Resources
    from myrmidon.execution.slurm import SlurmBackend

    spec = JobSpec(argv=["echo", "qm"], cwd=tmp_path, stdout="qm.out", stderr="qm.err", mpi_ranks=3,
                   resources=Resources(ntasks=4),
                   coupled=[CoupledCmd(argv=["sh", "-c", f"echo mm; exit {7 if fails else 0}"],
                                       mpi_ranks=1, stdout="mm.out", stderr="mm.err")])
    backend = SlurmBackend(SlurmDefaults(coupled_launch="multi-prog"))
    script = backend.render_script(spec, "job")
    srun = mock.BIN / "srun"
    done = subprocess.run(["sh", "-c", script.replace("srun ", f"{srun} ")], cwd=tmp_path,
                          capture_output=True, text=True, check=False)
    assert done.returncode == exit_code, done.stderr
    assert (tmp_path / "qm.out").read_text() == "qm\n"
    assert (tmp_path / "mm.out").read_text() == "mm\n"
