"""script_run: research scripts as provenance-bearing jobs (local backend)."""

import dataclasses
import subprocess
import time

import pytest

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


def test_script_run_waits_hashes_inputs_and_outputs(ws_ctx):
    ws = ws_ctx.workspace
    _write(ws, "01_data/in.txt", "3 4\n")
    script = _write(
        ws, "analysis/sum.py",
        "import sys, pathlib\n"
        "a, b = pathlib.Path(sys.argv[1]).read_text().split()\n"
        "pathlib.Path('sum.txt').write_text(str(int(a) + int(b)) + '\\n')\n",
    )
    obs = REGISTRY.call(
        "script_run", ws_ctx,
        {"script": script, "args": [str(ws.root / "01_data/in.txt")],
         "inputs": ["01_data/in.txt"], "outputs": ["sum.txt"], "wait_seconds": 20},
    )
    assert obs.ok, obs.summary
    assert obs.state == "COMPLETED" and obs.exit_code == 0
    assert obs.core_h_estimate == 0.5  # 1 core × 30 min default
    assert {a.role for a in obs.inputs} == {"script", "input"}
    assert [a.role for a in obs.artifacts] == ["output"]
    assert (ws.root / obs.artifacts[0].path).read_text() == "7\n"
    assert obs.command[0].endswith("python") or "python" in obs.command[0]
    event = ws.ledger.get(obs.provenance_id)
    assert {a.role for a in event.inputs} == {"script", "input"}  # inputs in the ledger
    assert event.outputs[0].role == "output"


def test_missing_declared_output_fails_even_on_exit_zero(ws_ctx):
    ws = ws_ctx.workspace
    script = _write(ws, "analysis/noop.py", "print('done')\n")
    obs = REGISTRY.call(
        "script_run", ws_ctx,
        {"script": script, "outputs": ["result.csv"], "wait_seconds": 20},
    )
    assert not obs.ok
    assert obs.state == "COMPLETED" and obs.exit_code == 0
    assert obs.outputs_missing == ["result.csv"]


def test_python_exception_is_diagnosed(ws_ctx):
    ws = ws_ctx.workspace
    script = _write(ws, "analysis/boom.py", "raise ValueError('bad input 42')\n")
    obs = REGISTRY.call("script_run", ws_ctx, {"script": script, "wait_seconds": 20})
    assert not obs.ok
    assert obs.state == "FAILED"
    assert "ValueError: bad input 42" in obs.diagnosis
    assert obs.log_tail and "Traceback" in obs.log_tail


# (the call, the blocking findings it must carry). A whole program passed as the path is
# refused by the gate, not raised out of it as "File name too long".
GATE_BLOCKS = [
    ({"script": "nope.py", "inputs": ["absent.dat"]}, {"SCRIPT_MISSING", "INPUT_MISSING_0"}),
    ({"script": 'python3 -c "\n# write a gro\n' + "x = 1\n" * 60 + '"'},
     {"SCRIPT_NOT_A_PATH"}),
    ({"script": "a" * 300 + ".py"}, {"SCRIPT_NOT_A_PATH"}),
]


@pytest.mark.parametrize("call,ids", GATE_BLOCKS)
def test_gate_blocks_what_it_cannot_run(ws_ctx, call, ids):
    refused = REGISTRY.call("script_run", ws_ctx, call)
    assert not refused.ok
    assert ids <= {w.id for w in refused.blocking}
    assert ws_ctx.workspace.ledger.get(refused.provenance_id).kind == "gate_refused"


def test_a_gate_that_raises_refuses_the_call_and_the_session_goes_on(ws_ctx, monkeypatch):
    spec, fn = REGISTRY._tools["script_run"]

    def broken(ctx, params):
        raise OSError(36, "File name too long")

    monkeypatch.setitem(REGISTRY._tools, "script_run",
                        (dataclasses.replace(spec, gate=broken), fn))
    refused = REGISTRY.call("script_run", ws_ctx, {"script": "x.py"})
    assert (refused.ok, refused.failure) == (False, "raised")
    assert ws_ctx.workspace.ledger.get(refused.provenance_id).kind == "gate_refused"


def test_compute_budget_gate(ws_ctx):
    ws = ws_ctx.workspace
    script = _write(ws, "analysis/ok.py", "print(1)\n")
    ws_ctx.config.compute.warn_core_h = 0.1
    refused = REGISTRY.call("script_run", ws_ctx, {"script": script})
    assert not refused.ok
    assert [w.id for w in refused.unacknowledged] == ["COMPUTE_BUDGET"]
    assert "0.5 core-h" in refused.unacknowledged[0].message

    ok = REGISTRY.call(
        "script_run", ws_ctx,
        {"script": script, "acknowledge_warnings": ["COMPUTE_BUDGET"], "wait_seconds": 20},
    )
    assert ok.ok, ok.summary
    assert ws.ledger.get(ok.provenance_id).warnings_acknowledged == ["COMPUTE_BUDGET"]

    ws_ctx.config.compute.node_exclusive = True
    ws_ctx.config.compute.cores_per_node = 48
    ws_ctx.config.compute.block_core_h = 20.0
    blocked = REGISTRY.call(
        "script_run", ws_ctx,
        {"script": script, "walltime_min": 60, "acknowledge_warnings": ["COMPUTE_BUDGET"]},
    )
    assert not blocked.ok
    assert [w.id for w in blocked.blocking] == ["COMPUTE_BUDGET_EXCEEDED"]
    assert "48.0 core-h" in blocked.blocking[0].message  # 1 task rounds up to a node


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

    obs = REGISTRY.call(
        "script_run", ws_ctx,
        {"script": str(script), "outputs": ["out.txt"], "wait_seconds": 20},
    )
    assert obs.ok, obs.summary
    assert obs.git_commit == head and obs.git_dirty is False
    assert obs.inputs[0].role == "script" and obs.inputs[0].path == str(script)

    script.write_text("open('out.txt', 'w').write('y')\n")
    dirty = REGISTRY.call("script_run", ws_ctx, {"script": str(script), "wait_seconds": 20})
    assert dirty.git_commit == head and dirty.git_dirty is True


def test_unchanged_polls_are_not_ledger_events(ws_ctx):
    ws = ws_ctx.workspace
    script = _write(ws, "analysis/slow.py", "import time; time.sleep(1.5)\n")
    obs = REGISTRY.call("script_run", ws_ctx, {"script": script})
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


def test_an_output_declared_as_an_absolute_glob_is_collected(ws_ctx):
    """pathlib refuses an absolute pattern ("Non-relative patterns are unsupported"), and a
    script whose outputs are declared by absolute path — the natural spelling when its cwd
    is not the workspace — must still produce artifacts rather than crash collection. The
    pattern is globbed from its own anchor."""
    ws = ws_ctx.workspace
    out = ws.root / "results"
    script = _write(
        ws, "analysis/write.py",
        "import pathlib, sys\n"
        "d = pathlib.Path(sys.argv[1]); d.mkdir(exist_ok=True)\n"
        "(d / 'a.txt').write_text('1\\n'); (d / 'b.txt').write_text('2\\n')\n",
    )
    obs = REGISTRY.call(
        "script_run", ws_ctx,
        {"script": script, "args": [str(out)], "outputs": [f"{out}/*.txt"],
         "wait_seconds": 20},
    )
    assert obs.ok, obs.summary
    assert obs.state == "COMPLETED" and obs.outputs_missing == []
    assert sorted(pathlib_name(a.path) for a in obs.artifacts) == ["a.txt", "b.txt"]


def pathlib_name(path: str) -> str:
    from pathlib import Path

    return Path(path).name
