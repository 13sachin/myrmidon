"""The mock engines ship with the harness and behave the same wherever they run."""

import os

from myrmidon.core.tool import REGISTRY
from myrmidon.engines import mock


def test_the_mock_engines_are_part_of_the_package():
    """`serve --mock` must work from an installed harness, not only from a checkout."""
    import myrmidon

    assert mock.BIN.is_relative_to(os.path.dirname(myrmidon.__file__))
    mock.check_bin()
    assert {"gmx", "cpmd.x", "mimicpy", "sbatch", "squeue", "sacct"} <= {
        p.name for p in mock.BIN.iterdir()}


def test_a_mock_job_is_seen_running_exactly_once_whatever_the_host(ws_ctx):
    """The logical clock: the poll sequence of a submitted job is a fact of the session,
    not of how fast this machine happened to run the detached script."""
    ws = ws_ctx.workspace
    stage = ws.stage("run")
    sequences = []
    for n in range(2):
        (stage / f"p{n}.tpr").write_text("MOCKTPR\n")
        with mock.activated(ws.root):
            ctx = mock.mock_context(ws_ctx)
            job = REGISTRY.call("gmx_mdrun", ctx, {
                "tpr": ws.relpath(stage / f"p{n}.tpr"), "deffnm": f"p{n}",
                "walltime_min": 5})
            assert job.ok, job.summary
            states = [REGISTRY.call("job_status", ctx, {"job_id": job.job_id}).state
                      for _ in range(3)]
        sequences.append(states)
    assert sequences[0] == sequences[1] == ["RUNNING", "COMPLETED", "COMPLETED"]
