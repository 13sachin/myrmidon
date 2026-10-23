"""session_summary / `myrmidon summary`: the harness made visible."""



from myrmidon.core.summary import render, statusline
from myrmidon.core.tool import REGISTRY


def _script(ws, name, body):
    path = ws.root / "analysis" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return ws.relpath(path)


def test_session_summary_counts_everything_that_went_through(ws_ctx):
    ws = ws_ctx.workspace
    REGISTRY.call("file_write", ws_ctx, {"path": "notes/a.txt", "content": "hi"})
    ok = _script(ws, "ok.py", "open('r.txt','w').write('1')\n")
    run = REGISTRY.call(
        "script_run", ws_ctx, {"script": ok, "outputs": ["r.txt"], "wait_seconds": 20}
    )
    assert run.ok, run.summary
    refused = REGISTRY.call("script_run", ws_ctx, {"script": "missing.py"})
    assert not refused.ok
    n_before = sum(1 for _ in ws.ledger.events())

    obs = REGISTRY.call("session_summary", ws_ctx, {"since": "session"})
    assert obs.ok
    assert sum(1 for _ in ws.ledger.events()) == n_before  # reading is not an event
    tools = {t.tool: t for t in obs.tools}
    assert tools["file_write"].calls == 1
    assert tools["script_run"].calls == 2 and tools["script_run"].failed == 0
    assert obs.gate_refusals == 1
    assert len(obs.jobs) == 1 and obs.jobs[0].state == "COMPLETED"
    assert obs.core_h_submitted == 0.5
    assert obs.core_h_consumed is not None and obs.core_h_consumed >= 0
    assert obs.n_artifacts == 2  # notes/a.txt + r.txt
    assert obs.artifacts_by_role == {"file": 1, "output": 1}
    assert obs.scripts and obs.scripts[0].endswith("ok.py")
    assert "not here" in obs.summary  # the bypass caveat is always stated

    text = render(obs)
    assert "script_run" in text and "core-h" in text
    line = statusline(ws)
    assert line.startswith("myrmidon test-study") and "last: script_run" in line

