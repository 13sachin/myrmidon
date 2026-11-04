"""script_run: research scripts as provenance-bearing jobs (local backend)."""

import subprocess
import time

import pytest
from tests.unit.findings_catalog import blocking, unapproved

from myrmidon.core.tool import REGISTRY

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t", "HOME": "/nonexistent-home",
}


def _write(ws, rel, text, mode=None):
    path = ws.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if mode is not None:
        path.chmod(mode)
    return rel


def _wait_terminal(ctx, job_id, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        obs = REGISTRY.call("job_status", ctx, {"job_id": job_id})
        if obs.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return obs
        time.sleep(0.1)
    raise AssertionError("job never finished")


# (the call's kind and resources, where the run is placed, what it books): the same script
# is a job above the session envelope (1 core × 30 min) and runs in the session within it,
# and an analysis's results file is read back as its numbers.
PLACEMENTS = [
    ({"kind": "glue", "walltime_min": 30}, "job", 0.5),
    ({"kind": "glue", "walltime_min": 2}, "session", 0.0),
    ({"kind": "analysis", "walltime_min": 2, "results": "sum.json"}, "session", 0.0),
]


@pytest.mark.parametrize("extra,placement,core_h", PLACEMENTS)
def test_script_run_waits_hashes_inputs_and_outputs(ws_ctx, extra, placement, core_h):
    ws = ws_ctx.workspace
    _write(ws, "01_data/in.txt", "3 4\n")
    script = _write(
        ws, "analysis/sum.py",
        "import sys, pathlib, json\n"
        "a, b = pathlib.Path(sys.argv[1]).read_text().split()\n"
        "pathlib.Path('sum.txt').write_text(str(int(a) + int(b)) + '\\n')\n"
        "pathlib.Path('sum.json').write_text(json.dumps({'sum': int(a) + int(b)}))\n",
    )
    obs = REGISTRY.call(
        "script_run", ws_ctx,
        {"script": script, "args": [str(ws.root / "01_data/in.txt")],
         "inputs": ["01_data/in.txt"], "outputs": ["sum.txt"], "wait_seconds": 20, **extra},
    )
    assert obs.ok, obs.summary
    assert obs.state == "COMPLETED" and obs.exit_code == 0
    assert obs.placement == placement and obs.core_h_estimate == core_h
    assert bool(obs.job_id) == (placement == "job")  # a session run has no job record
    assert obs.script_kind == extra["kind"]
    assert ws.ledger.get(obs.provenance_id).kind == "call"  # an action, counted as work
    assert {a.role for a in obs.inputs} == {"script", "input"}
    analysis = extra["kind"] == "analysis"
    assert [a.role for a in obs.artifacts] == (["output", "data"] if analysis else ["output"])
    assert (ws.root / obs.artifacts[0].path).read_text() == "7\n"
    assert obs.results == ({"sum": 7} if analysis else {})
    assert obs.command[0].endswith("python") or "python" in obs.command[0]
    event = ws.ledger.get(obs.provenance_id)
    assert {a.role for a in event.inputs} == {"script", "input"}  # inputs in the ledger
    assert event.outputs[0].role == "output"


def test_missing_declared_output_fails_even_on_exit_zero(ws_ctx):
    ws = ws_ctx.workspace
    script = _write(ws, "analysis/noop.py", "print('done')\n")
    obs = REGISTRY.call(
        "script_run", ws_ctx,
        {"script": script, "kind": "glue", "outputs": ["result.csv"], "wait_seconds": 20},
    )
    assert not obs.ok
    assert obs.state == "COMPLETED" and obs.exit_code == 0
    assert obs.outputs_missing == ["result.csv"]
    assert "looked for in" in obs.summary and "the study root" in obs.summary


def test_inputs_and_outputs_resolve_against_cwd_and_the_study_root_alike(ws_ctx):
    """A script run inside a stage directory edits a file there: the input named relative
    to cwd and the output named relative to the study root are the same file, found either
    way, and both are hashed under their study-relative path."""
    ws = ws_ctx.workspace
    stage = ws.stage("select")
    (stage / "protein.pdb").write_text("ATOM      1  N   ALA A   1\n")
    script = _write(ws, "analysis/fix_chain.py",
                    "import pathlib\n"
                    "p = pathlib.Path('protein.pdb')\n"
                    "p.write_text(p.read_text().replace(' A ', ' B '))\n")
    rel = ws.relpath(stage)
    obs = REGISTRY.call("script_run", ws_ctx, {
        "script": script, "kind": "glue", "cwd": rel, "inputs": ["protein.pdb"],
        "outputs": [f"{rel}/protein.pdb"], "walltime_min": 2, "wait_seconds": 20})
    assert obs.ok, obs.summary
    assert [a.path for a in obs.artifacts] == [f"{rel}/protein.pdb"]
    assert any(a.path == f"{rel}/protein.pdb" and a.role == "input" for a in obs.inputs)
    assert (stage / "protein.pdb").read_text() == "ATOM      1  N   ALA B   1\n"
    refused = REGISTRY.call("script_run", ws_ctx, {
        "script": script, "kind": "glue", "cwd": rel, "inputs": ["absent.pdb"],
        "outputs": ["protein.pdb"], "walltime_min": 2})
    assert not refused.ok
    [finding] = [w for w in blocking(refused) if w.id.startswith("INPUT_MISSING")]
    assert rel in finding.message and "the study root" in finding.message
    assert str(ws.root) not in finding.message  # the study's names, not the machine's


def test_python_exception_is_diagnosed(ws_ctx):
    ws = ws_ctx.workspace
    script = _write(ws, "analysis/boom.py", "raise ValueError('bad input 42')\n")
    obs = REGISTRY.call("script_run", ws_ctx, {"script": script, "kind": "glue",
                                               "outputs": ["never.txt"], "wait_seconds": 20})
    assert not obs.ok
    assert obs.state == "FAILED"
    assert "ValueError: bad input 42" in obs.diagnosis
    assert obs.log_tail and "Traceback" in obs.log_tail


# (the call, the blocking findings it must carry). A whole program passed as the path is
# refused by the gate, not raised out of it as "File name too long"; a kind without what
# it needs (glue's outputs, an analysis's results) is refused before anything runs.
GATE_BLOCKS = [
    ({"script": "nope.py", "kind": "glue", "outputs": ["x"], "inputs": ["absent.dat"]},
     {"SCRIPT_MISSING", "INPUT_MISSING_0"}),
    ({"script": 'python3 -c "\n# write a gro\n' + "x = 1\n" * 60 + '"', "kind": "glue",
      "outputs": ["x"]}, {"SCRIPT_NOT_A_PATH"}),
    ({"script": "a" * 300 + ".py", "kind": "glue", "outputs": ["x"]}, {"SCRIPT_NOT_A_PATH"}),
    ({"script": "nope.py", "kind": "glue"}, {"SCRIPT_GLUE_NO_OUTPUT"}),
    ({"script": "nope.py", "kind": "analysis"}, {"SCRIPT_ANALYSIS_NO_RESULTS"}),
]


@pytest.mark.parametrize("call,ids", GATE_BLOCKS)
def test_gate_blocks_what_it_cannot_run(ws_ctx, call, ids):
    refused = REGISTRY.call("script_run", ws_ctx, call)
    assert not refused.ok
    assert ids <= {w.id for w in blocking(refused)}
    assert ws_ctx.workspace.ledger.get(refused.provenance_id).kind == "gate_refused"


def test_compute_budget_gate(ws_ctx):
    ws = ws_ctx.workspace
    script = _write(ws, "analysis/ok.py", "open('ok.txt', 'w').write('1')\n")
    glue = {"script": script, "kind": "glue", "outputs": ["ok.txt"], "walltime_min": 30}
    ws_ctx.person = None
    refused = REGISTRY.call("script_run", ws_ctx, glue)
    assert not refused.ok
    assert [w.id for w in unapproved(refused)] == ["MANDATE_ABSENT"]
    assert "0.5 core-h" in unapproved(refused)[0].message

    from tests.unit.test_approvals import _mandate, _sign

    _mandate(ws_ctx)
    _sign(ws_ctx)
    ok = REGISTRY.call("script_run", ws_ctx, {**glue, "wait_seconds": 20})
    assert ok.ok, ok.summary
    assert ws.ledger.get(ok.provenance_id).mandate

    ws_ctx.config.compute.node_exclusive = True
    ws_ctx.config.compute.cores_per_node = 48
    ws_ctx.config.compute.block_core_h = 20.0
    blocked = REGISTRY.call("script_run", ws_ctx, {**glue, "walltime_min": 60})
    assert not blocked.ok
    assert [w.id for w in blocking(blocked)] == ["COMPUTE_BUDGET_EXCEEDED"]
    assert "48.0 core-h" in blocking(blocked)[0].message  # 1 task rounds up to a node

    # within the envelope nothing is booked, so no budget finding: a physics script's one
    # finding says its numbers are its author's, and the report carries it
    physics = REGISTRY.call("script_run", ws_ctx, {**glue, "kind": "physics", "walltime_min": 2})
    assert physics.ok, physics.summary
    assert "SCRIPT_IS_PHYSICS" in {w.id for w in physics.findings}


def test_git_commit_and_dirty_flag_recorded(ws_ctx, tmp_path):
    repo = tmp_path / "research"
    repo.mkdir()
    script = repo / "analyze.py"
    script.write_text("open('out.txt', 'w').write('x')\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=_GIT_ENV)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=_GIT_ENV)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "v1"], check=True, env=_GIT_ENV)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    call = {"script": str(script), "kind": "glue", "outputs": ["out.txt"], "wait_seconds": 20}
    obs = REGISTRY.call("script_run", ws_ctx, call)
    assert obs.ok, obs.summary
    assert obs.git_commit == head and obs.git_dirty is False
    assert obs.inputs[0].role == "script" and obs.inputs[0].path == str(script)

    script.write_text("open('out.txt', 'w').write('y')\n")
    dirty = REGISTRY.call("script_run", ws_ctx, call)
    assert dirty.git_commit == head and dirty.git_dirty is True


def test_unchanged_polls_are_not_ledger_events(ws_ctx):
    ws = ws_ctx.workspace
    script = _write(ws, "analysis/slow.py",
                    "import time; time.sleep(1.5); open('slow.txt', 'w').write('x')\n")
    obs = REGISTRY.call("script_run", ws_ctx, {"script": script, "kind": "glue",
                                               "outputs": ["slow.txt"], "walltime_min": 30})
    before = sum(1 for ev in ws.ledger.events() if ev.tool == "job_status")
    seen_running = False
    for _ in range(10):
        st = REGISTRY.call("job_status", ws_ctx, {"job_id": obs.job_id})
        if st.state == "RUNNING":
            seen_running = True
        time.sleep(0.1)
    assert seen_running
    during = sum(1 for ev in ws.ledger.events() if ev.tool == "job_status")
    assert during - before <= 1  # only the QUEUED→RUNNING transition, not every poll
    final = _wait_terminal(ws_ctx, obs.job_id)
    assert final.provenance_id is not None  # the terminal transition is recorded




# (who changes the queued job's input, whether its result is accepted): nobody; a hand
# while it waits; the job it waits on, which writes it as its declared output
RECEIPTS = [(None, True), ("hand", False), ("dependency", True)]


@pytest.mark.parametrize("changer,accepted", RECEIPTS)
def test_a_job_whose_input_changed_before_it_ran_is_not_accepted(ws_ctx, changer, accepted):
    """The node hashes the job's inputs as it starts and writes its execution record; an
    input edited while the job waited in the queue is caught at collection, by hash — unless
    the job it waited on wrote it, as its collection in the ledger says."""
    ws = ws_ctx.workspace
    _write(ws, "01_scripts/data.txt", "first\n")  # where the scripts run
    first = _write(ws, "analysis/first.py",
                   "import time\ntime.sleep(2)\n"
                   + ("open('data.txt', 'w').write('rewritten')\n" if changer == "dependency"
                      else "") + "open('slow.txt', 'w').write('1')\n")
    second = _write(ws, "analysis/second.py",
                    "open('out.txt', 'w').write(open('data.txt').read())\n")
    ws_ctx.person = "researcher"
    queue = REGISTRY.call("script_run", ws_ctx, {
        "script": first, "kind": "glue", "walltime_min": 30,
        "outputs": ["slow.txt", *(["data.txt"] if changer == "dependency" else [])]})
    job = REGISTRY.call("script_run", ws_ctx, {"script": second, "kind": "glue",
                                               "walltime_min": 30,
                                               "inputs": ["01_scripts/data.txt"],
                                               "outputs": ["out.txt"],
                                               "after": [queue.job_id]})
    assert queue.ok and job.ok, job.summary
    if changer == "hand":  # the queued job has not started: its input is rewritten under it
        (ws.root / "01_scripts" / "data.txt").write_text("edited\n")
    done = REGISTRY.call("job_status", ws_ctx, {"job_id": job.job_id, "wait_seconds": 30})
    assert done.ok is accepted, done.summary
    assert ("RECEIPT_INPUT_CHANGED" in {f.id for f in done.findings}) is not accepted
    assert done.execution["exit_code"] == 0 and done.execution["started"]
