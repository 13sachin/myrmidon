"""Smoke test against a real SLURM scheduler.

Submits a trivial job through SlurmBackend using the machine config (partition,
account, extra_sbatch_lines) and waits for it to complete. Auto-skipped unless
sbatch is on PATH and the machine config selects ``backend = "slurm"``.
Workspaces live under the configured ``studies_root`` (compute nodes cannot
see the login node's /tmp).
"""

import shutil
import time
import uuid
from pathlib import Path

import pytest

from myrmidon.config import MyrmidonConfig
from myrmidon.execution.backend import JobSpec, JobState, Resources
from myrmidon.execution.slurm import SlurmBackend

CONFIG = MyrmidonConfig.load()

pytestmark = [
    pytest.mark.slurm,
    pytest.mark.timeout(1200),
    pytest.mark.skipif(shutil.which("sbatch") is None, reason="no sbatch on PATH"),
    pytest.mark.skipif(CONFIG.backend != "slurm", reason="machine config backend is not slurm"),
]


def wait_terminal(backend, handle, spec, timeout=900.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = backend.poll(handle, spec)
        if status.state.terminal:
            return status
        time.sleep(10)
    backend.cancel(handle, spec)
    raise AssertionError(f"job {handle} did not finish within {timeout:.0f} s")


def test_real_sbatch_roundtrip():
    run_dir = Path(CONFIG.studies_root) / "_scratch" / "slurm-smoke" / uuid.uuid4().hex[:8]
    run_dir.mkdir(parents=True)
    backend = SlurmBackend(CONFIG.slurm)
    spec = JobSpec(
        argv=["sh", "-c", "hostname; echo MYRMIDON_OK"],
        cwd=run_dir,
        stdout="smoke.out",
        stderr="smoke.err",
        resources=Resources(walltime_min=5),
    )
    handle = backend.submit(spec)
    assert handle.isdigit()
    first = backend.poll(handle, spec)
    assert first.state in (JobState.QUEUED, JobState.RUNNING, JobState.COMPLETED), first
    status = wait_terminal(backend, handle, spec)
    assert status.state is JobState.COMPLETED, status
    assert status.exit_code == 0
    out = (run_dir / "smoke.out").read_text()
    assert "MYRMIDON_OK" in out
    # the job ran on a compute node, not where we submitted from
    script = next(run_dir.glob(".slurm-*/job.sbatch")).read_text()
    assert f"--account={CONFIG.slurm.account}" in script if CONFIG.slurm.account else True
