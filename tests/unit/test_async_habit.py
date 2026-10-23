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
    """A real submission through the registry: grompp, then mdrun on the mock engines."""
    grompp = REGISTRY.call("gmx_grompp", ctx, {
        "mdp": "01_input/nvt.mdp", "structure": "01_input/conf.gro",
        "topology": "01_input/topol.top", "stage": "run", "output": "nvt.tpr",
    })
    assert grompp.ok, grompp.summary
    run = REGISTRY.call("gmx_mdrun", ctx, {
        "tpr": "02_run/nvt.tpr", "deffnm": "nvt", "stage": "run", "walltime_min": 5,
    })
    assert run.ok, run.summary
    return run.job_id


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
    from myrmidon.engines.gromacs.results import _refresh

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


def test_open_says_where_a_running_job_got_to(ws_ctx, monkeypatch):
    """A job still running reports its own step counter — the difference between
    "waiting" and "hung". Both numbers are read out of the engine's log, here a real
    one; only the scheduler's answer is stubbed, because no process is behind it."""
    from myrmidon.execution.backend import JobState, JobStatus

    ws = ws_ctx.workspace
    _one_fake_job(ws_ctx, ws, label="mdrun:heat")
    (ws.root / "heat.log").write_text(
        "   nsteps = 100000\n\n           Step           Time\n"
        "           4000        8.00000\n\nPerformance:       79.2\n"
    )
    monkeypatch.setattr(type(ws_ctx.backend), "poll",
                        lambda self, handle, spec: JobStatus(state=JobState.RUNNING))

    obs = REGISTRY.call("study_open", ws_ctx, {})

    assert len(obs.jobs) == 1, obs.jobs
    assert "RUNNING (mdrun:heat)" in obs.jobs[0]
    assert "step 4000 of 100000" in obs.jobs[0]
    assert "16.0 core-h booked" in obs.jobs[0]
    assert "still running" in obs.summary


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


def _one_fake_job(ctx, ws, label: str):
    """A RUNNING record in the store, with no real process behind it."""
    from myrmidon.execution.backend import JobSpec, JobState
    from myrmidon.execution.jobstore import open_jobstore

    store = open_jobstore(ws)
    spec = JobSpec(argv=["gmx", "mdrun", "-deffnm", "heat"], cwd=str(ws.root))
    job_id = store.add("local", "0", spec, label=label, meta={"core_h_estimate": 16.0})
    store.update(job_id, JobState.RUNNING, None)
    return job_id
