"""What a session that comes back to a finished job can see.

A job outlives the session that submitted it: the user closes the terminal, the
scheduler runs the job, and the next session has a new server process with no workspace
and a job store holding whatever state was last asked for. Everything here is about that
gap.

The division of labour these tests hold to: opening a workspace *looks* (it refreshes
state and says what it sees, and writes no ledger event), `job_status` *collects*,
exactly once, and owns the artifacts.
"""


import pytest

from myrmidon.core.tool import REGISTRY


def _submit_mdrun(ctx, planted):
    """A real submission through the registry: a stage prepared and run on the mock engines."""
    from tests.conftest import NVT_STAGE, approved_study, prepare_and_run

    approved_study(ctx, "01_input/conf.gro", "01_input/topol.top", [NVT_STAGE])
    return prepare_and_run(ctx, "nvt", walltime_min=5)[1].job_id


@pytest.fixture
def submitted(ws_ctx):
    """A workspace with one job submitted through the tools, as a session leaves it."""
    ws = ws_ctx.workspace
    stage = ws.stage("input")
    (stage / "conf.gro").write_text(
        "mock\n 3\n" + "    1ALA     CA    1   0.1   0.2   0.3\n" * 3 + "   2.5 2.5 2.5\n")
    (stage / "topol.top").write_text("[ system ]\nMock\n[ molecules ]\nProtein 1\n")
    (stage / "nvt.mdp").write_text(
        "integrator = md\ndt = 0.002\nnsteps = 500\nconstraints = h-bonds\n"
        "constraint_algorithm = lincs\ntcoupl = v-rescale\ntc-grps = System\n"
        "tau_t = 0.1\nref_t = 300\ncoulombtype = PME\ngen_vel = yes\ngen_seed = 1\n")
    return ws_ctx, _submit_mdrun(ws_ctx, stage)


def _wait(ctx, job_id):
    """Let the job finish, without collecting it: the state the next session meets."""
    from myrmidon.execution.jobs import refresh as _refresh

    for _ in range(300):
        _, state, _, _ = _refresh(ctx, job_id)
        if state.terminal:
            return state
        import time

        time.sleep(0.2)
    raise AssertionError("mock job never finished")


def _fresh_context(ctx):
    """A second session: a new ToolContext over the same machine config, as the MCP
    server builds one — no workspace, because nobody has said which study this is."""
    from myrmidon.core.tool import ToolContext

    return ToolContext(
        config=ctx.config, workspace=None, backend=type(ctx.backend)(),
        engines=ctx.engines, inputsets=ctx.inputsets,
    )


def test_open_does_not_collect_and_job_status_collects_exactly_once(submitted):
    """Opening is a look: it refreshes state and writes no ledger event. The artifacts
    land when job_status finds the job ended, once."""
    ctx, job_id = submitted
    _wait(ctx, job_id)
    ws = ctx.workspace

    REGISTRY.call("study_open", _fresh_context(ctx), {})
    REGISTRY.call("study_open", _fresh_context(ctx), {})
    opens = [ev for ev in ws.ledger.events() if ev.tool == "study_open"]
    assert all(not ev.outputs for ev in opens), "a look must not carry artifacts"

    first = REGISTRY.call("job_status", ctx, {"job_id": job_id})
    second = REGISTRY.call("job_status", ctx, {"job_id": job_id})
    assert first.kind == "call" and first.artifacts
    assert second.kind == "observe"
    collected = [ev for ev in ws.ledger.events()
                 if ev.tool == "job_status" and ev.kind == "call"]
    assert len(collected) == 1 and collected[0].outputs


