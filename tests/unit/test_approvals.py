"""The researcher's approval: of a finding that is theirs, by their own call; of a mandate,
recorded in the ledger and read by the gate while it holds; and what runs unattended under
it."""

import pytest
from tests.unit.findings_catalog import blocking, unapproved

from myrmidon.core.tool import REGISTRY
from myrmidon.core.validation import Theirs


def _budget_tool():
    """A booking whose only finding at the registry's gate is a budget that is the
    researcher's to accept."""
    from typing import ClassVar

    from pydantic import BaseModel

    from myrmidon.core.observation import Observation
    from myrmidon.core.tool import ToolRegistry, precheck, tool
    from myrmidon.execution.backend import Resources

    reg = ToolRegistry()
    reg.submission_gate = lambda ctx, params, name: ([Theirs(
        id="SPEND_THEIRS", message="48 core-h")], ["budget"])

    class P(BaseModel):
        RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("ntasks",)
        ntasks: int = 1
        preview: bool = False

        def resources(self) -> Resources:
            return Resources(ntasks=self.ntasks)

    @tool("spend", description="x", registry=reg, expensive=True)
    def spend(ctx, params: P) -> Observation:
        precheck(ctx, [])
        return Observation(ok=True, summary="submitted")

    return reg


# (a person makes the call, the call runs). A finding that is the researcher's is accepted
# by their own call at the terminal; a model's call is refused and says whose it is.
APPROVALS = [(False, False), (True, True)]


@pytest.mark.parametrize("person,runs", APPROVALS)
def test_a_finding_for_a_person_is_accepted_only_by_a_person(ws_ctx, person, runs):
    reg = _budget_tool()
    ws_ctx.person = "researcher" if person else None
    obs = reg.call("spend", ws_ctx, {})
    assert obs.ok is runs
    event = list(ws_ctx.workspace.ledger.events())[-1]
    if runs:
        assert event.warnings_acknowledged == ["SPEND_THEIRS"]
    else:
        assert [w.id for w in unapproved(obs)] == ["SPEND_THEIRS"]
        assert obs.error.code == "SPEND_THEIRS" and "researcher" in obs.error.hint


def _approve(*args):
    from typer.testing import CliRunner

    from myrmidon.cli import app

    return CliRunner().invoke(app, ["approve", *args])


# --- the mandate ------------------------------------------------------------------------


def _mandate(ws_ctx, **params):
    """A mandate over one system file, written through the tool."""
    stage = ws_ctx.workspace.stage("input")
    (stage / "conf.gro").write_text("acetone in water\n")
    call = {"question": "Equilibrate the box", "ceiling_core_h": 10.0,
            "system_files": [{"path": f"{stage.name}/conf.gro", "kind": "structure"}],
            **params}
    return REGISTRY.call("mandate_write", ws_ctx, call), stage / "conf.gro"


def _sign(ws_ctx) -> str:
    """The researcher's yes to the mandate on disk, as `myrmidon approve` records it."""
    from myrmidon.core import mandate as mandates

    ws = ws_ctx.workspace
    return mandates.record_approval(ws, mandates.read_mandate(ws), approver="researcher",
                                    via="terminal")


def test_a_mandate_holds_for_the_digest_approved_with_the_system_it_hashed(ws_ctx):
    import asyncio
    import json

    from myrmidon.core import mandate as mandates
    from myrmidon.server.mcp_server import build_server

    ws = ws_ctx.workspace
    unpriced, _ = _mandate(ws_ctx, ceiling_core_h=None, questions=["a?", "b?", "c?"])
    assert unpriced.status == "needs_input" and "core-hours" in unpriced.questions[0]
    written, system = _mandate(ws_ctx)
    assert written.ok and written.status == "ready", written.summary
    assert mandates.holding(ws) is None and "not approved" in mandates.why_not_holding(ws)
    _sign(ws_ctx)
    assert mandates.holding(ws).digest == written.digest
    written_files = mandates.read_mandate(ws).system.files

    system.write_text("another system\n")
    assert mandates.holding(ws) is None and "conf.gro" in mandates.why_not_holding(ws)
    system.write_text("acetone in water\n")
    assert mandates.holding(ws).digest == written.digest

    refused, _ = _mandate(ws_ctx, ceiling_core_h=20.0)
    assert not refused.ok and "revise" in refused.summary
    # a revision states what changes, through the server that hands the tool every field
    revised = json.loads(asyncio.run(build_server(ctx=ws_ctx).call_tool("mandate_write", {
        "ceiling_core_h": 20.0, "revise": True, "reason": "a longer run"})).content[0].text)
    assert (revised["ok"], revised["data"]["revision"], revised["data"]["approved"]) == (
        True, 2, False)
    now = mandates.read_mandate(ws)
    assert (now.question, now.system.files) == ("Equilibrate the box", written_files)
    assert mandates.holding(ws) is None
    assert len(list((ws.state_dir / mandates.ARCHIVE_DIR).iterdir())) >= 2


# (a mandate approved, its ceiling, a person makes the call) -> the finding that decides,
# whether the job is submitted. Outside an approved mandate only a person's own call books;
# under one, the job fits what the ceiling has left or waits on the researcher.
SUBMISSIONS = [
    (False, 10.0, False, "MANDATE_ABSENT", False),
    (False, 10.0, True, "MANDATE_ABSENT", True),
    (True, 10.0, False, "MANDATE_WITHIN", True),
    (True, 0.3, False, "MANDATE_CEILING", False),
]


@pytest.mark.parametrize("approved,ceiling,person,finding,submitted", SUBMISSIONS)
def test_a_job_books_under_an_approved_mandate_within_its_ceiling(
        ws_ctx, approved, ceiling, person, finding, submitted):
    ws = ws_ctx.workspace
    written, _ = _mandate(ws_ctx, ceiling_core_h=ceiling)
    if approved:
        _sign(ws_ctx)
    (ws.root / "analysis").mkdir()
    (ws.root / "analysis" / "ok.py").write_text("open('ok.txt', 'w').write('1')\n")
    ws_ctx.person = "researcher" if person else None
    obs = REGISTRY.call("script_run", ws_ctx, {"script": "analysis/ok.py", "kind": "glue",
                                               "outputs": ["ok.txt"], "walltime_min": 30})
    assert obs.ok is submitted, obs.summary
    event = ws.ledger.get(obs.provenance_id)
    found = {w.id for w in [*obs.findings, *unapproved(obs)]}
    assert finding in found | set(event.warnings_acknowledged)  # a person's call accepts it
    if submitted and approved:
        from myrmidon.execution.jobs import open_jobstore

        assert event.mandate == written.digest
        assert open_jobstore(ws).get(obs.job_id).meta["mandate"] == written.digest


def test_a_person_approves_the_mandate_they_read_from_the_command_line(
        ws_ctx, cli_env, monkeypatch):
    import json

    from myrmidon.client_registry import REGISTRY as CLIENTS
    from myrmidon.core import mandate as mandates

    for desc in CLIENTS.values():  # the command refuses inside an agent's shell
        if desc.nested_env:
            monkeypatch.delenv(desc.nested_env, raising=False)
    written, _ = _mandate(ws_ctx)
    study = str(ws_ctx.workspace.root)
    shown = _approve("--show", "--json", "--study", study)
    assert shown.exit_code == 0, shown.output
    view = json.loads(shown.output)
    assert (view["kind"], view["digest"], view["blocked"]) == ("mandate", written.digest, "")
    detailed = json.loads(_approve("--show", "--json", "--detail", "--study", study).output)
    # the researcher reads the system, the stages and the receipt; the model and an auditor
    # read the files, what the engines are told, the recovery terms and the full digest
    for text, detail in ((view["text"], False), (detailed["text"], True)):
        assert "QUESTION" in text and written.digest[:12] in text
        assert (written.digest in text) is detail and ("RECOVERY" in text) is detail
        assert all((f.path in text) is detail
                   for f in mandates.read_mandate(ws_ctx.workspace).system.files)

    stale = _approve("--yes", "--study", study, "--digest", "0" * 12)
    assert stale.exit_code == 1 and mandates.approvals(ws_ctx.workspace) == []
    file = ws_ctx.workspace.root / mandates.MANDATE_FILE
    text = file.read_text()
    file.write_text(text + "added_by_hand: 1\n")  # a field the file does not define
    broken = _approve("--show", "--study", study)
    assert broken.exit_code == 1 and "does not parse" in broken.output
    file.write_text(text)
    signed = _approve("--yes", "--study", study, "--digest", written.digest[:12], "--json")
    assert signed.exit_code == 0, signed.output
    (approval,) = mandates.approvals(ws_ctx.workspace)
    assert approval.digest == written.digest and approval.event_id == json.loads(
        signed.output)["event"]


def test_a_job_runs_only_on_the_approved_system_with_the_researchers_values(ws_ctx, solvated,
                                                                           monkeypatch):
    """The honest path submits; a run input built from another structure, or rendered with
    another value of a key the researcher decided, is refused by name, before anything is
    booked; preview says the same and books nothing; a collected run descends from the
    system through the ledger."""
    from myrmidon.core import mandate as mandates
    from myrmidon.core.lineage import lineage
    from myrmidon.execution.jobs import existing_jobstore

    ws = ws_ctx.workspace
    stage = {"kind": "dynamics", "ensemble": "NVT", "temperature_k": 300, "steps": 5000}
    written = REGISTRY.call("mandate_write", ws_ctx, {
        "question": "Equilibrate the box at 300 K", "ceiling_core_h": 10.0,
        "system_files": [{"path": solvated["structure"], "kind": "structure"},
                         {"path": solvated["topology"], "kind": "topology"}],
        "stages": [{"name": "nvt", **stage}, {"name": "nvt_again", **stage, "starts_from": "nvt"}],
        # an empty decision ("no restraint define") is met by a file that sets none
        "decisions_yours": [{"said": "300 K", "target": "gromacs.dynamics",
                             "keys": {"ref_t": 300, "define": ""}}]})
    assert written.status == "ready", written.summary
    _sign(ws_ctx)
    system = {"structure": solvated["structure"]}
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "nvt", **system}).ok
    rendered = [e for e in ws.ledger.events() if e.tool == "inputset_apply"][-1]
    assert any(r.startswith("inputset:gromacs.dynamics@v") for r in rendered.knowledge_refs)

    preview = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt", "preview": True})
    assert preview.ok and existing_jobstore(ws) is None, preview.summary
    # a preview sees the tool's own checks too, beside the mandate's
    missing = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt", "preview": True,
                                                  "inputs": {"tpr": "02_run/none.tpr"}})
    assert not missing.ok and existing_jobstore(ws) is None
    assert any(f.id.startswith("INPUT_MISSING") for f in missing.findings)
    job = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt"})
    assert job.ok, job.summary
    # a run that crashes is a deviation, never the result
    monkeypatch.setenv("MOCK_GMX_FAIL", "lincs")
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "nvt_again", **system}).ok
    crashed = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt_again"})
    assert crashed.ok, crashed.summary
    REGISTRY.call("job_status", ws_ctx, {"job_id": crashed.job_id, "wait_seconds": 30})
    monkeypatch.delenv("MOCK_GMX_FAIL")

    other = ws.stage("other") / "conf.gro"
    other.write_text((ws.root / solvated["structure"]).read_text().replace(
        "Acetone", "Another", 1) + "\n")
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "nvt_again",
                                                   "structure": ws.relpath(other)}).ok
    refused = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt_again", "preview": True})
    assert not refused.ok and [w.id for w in blocking(refused)] == ["MANDATE_SYSTEM"]
    assert ws.relpath(other) in blocking(refused)[0].message

    hotter = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "nvt_again", **system,
                                                     "overrides": {"ref_t": 310}})
    assert hotter.ok, hotter.summary
    refused = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt_again"})
    assert not refused.ok and [w.id for w in blocking(refused)] == ["MANDATE_DECISION"]
    assert "310" in blocking(refused)[0].message and "300 K" in blocking(refused)[0].message

    # a rendered file edited by hand is still the set's: the decision reads it on disk
    import re

    prepared = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "nvt_again", **system})
    assert prepared.ok, prepared.summary
    path = prepared.files["mdp"]
    hand = re.sub(r"(?m)^(ref[-_]t\s*=.*)$", lambda m: m.group(1).replace("300", "310"),
                  ws.resolve(path).read_text())
    assert "310" in hand
    edited = REGISTRY.call("file_write", ws_ctx, {"path": path, "content": hand})
    assert edited.ok, edited.summary
    refused = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt_again", "preview": True})
    assert {w.id for w in blocking(refused)} == {"MANDATE_DECISION"}, refused.summary
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "nvt_again", **system}).ok

    # a run input no recorded call built reaches nothing of the system
    foreign = ws.stage("elsewhere") / "foreign.tpr"
    foreign.write_text(ws.resolve(prepared.files["tpr"]).read_text() + "   from another study\n")
    refused = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt_again", "preview": True,
                                                  "inputs": {"tpr": ws.relpath(foreign)}})
    assert [w.id for w in blocking(refused)] == ["MANDATE_SYSTEM"], refused.summary

    done = REGISTRY.call("job_status", ws_ctx, {"job_id": job.job_id, "wait_seconds": 30})
    assert done.ok, done.summary
    confout = next(a for a in done.artifacts if a.role == "structure")
    held = {f.sha256 for f in mandates.holding(ws).system.files}
    verdict = lineage(ws).fits(confout.path, confout.sha256, "structure", held)
    assert verdict.ok and verdict.reached

    # the report reads the trace: the mandate, the job under it, what the gate refused
    report = REGISTRY.call("study_report", ws_ctx, {})
    assert report.ok, report.summary
    text = (ws.root / report.path).read_text()
    assert "Equilibrate the box at 300 K" in text and job.job_id in text
    results = text.split("## Results")[1].split("## Deviations")[0]
    deviations = text.split("## Deviations")[1]
    assert crashed.job_id not in results and f"job {crashed.job_id}" in deviations
    assert "MANDATE_DECISION" in text  # a refused submission; a preview is only a look


# (a value as a file writes it, the researcher's constraint, met)
CONSTRAINTS = [
    ("300 300", 300, True),
    ("300 310", 300, False),
    ("no", False, True),
    ("yes", False, False),
    ("500. 300. 30.", "500 300 30", True),
    ("0.002", {"max": 0.001}, False),
    ("0.001", [0.001, 0.0005], True),
    ("h-bonds", "H-bonds", True),
]


@pytest.mark.parametrize("value,constraint,met", CONSTRAINTS)
def test_a_decision_reads_the_value_as_the_engine_writes_it(value, constraint, met):
    from myrmidon.core.lineage import satisfies

    assert satisfies(value, constraint) is met


def test_a_cpmd_keyword_is_read_by_its_whole_words():
    from myrmidon.engines.mimic.cpmd_input import keyword_value

    sections = {"CPMD": [" TEMPERATURE RAMP", "  500. 300. 30.", " MAXSTEP", "  1000"]}
    assert keyword_value(sections, "&CPMD/TEMPERATURE RAMP") == "500. 300. 30."
    assert keyword_value(sections, "&CPMD/TEMPERATURE") is None
    assert keyword_value(sections, "&CPMD/MAXSTEP") == "1000"


def test_the_walk_is_linear_in_a_chain_of_restarts():
    """Each segment reads two files of the one before: without memoizing, the walk
    doubles per segment."""
    from types import SimpleNamespace as NS

    from myrmidon.core.lineage import Lineage

    events = [NS(event_id="e0", tool="t", ok=True, kind="call", knowledge_refs=[],
                 inputs=[NS(path="sys", role="structure", sha256="S")],
                 outputs=[NS(path="a0", sha256="a0"), NS(path="b0", sha256="b0")])]
    for n in range(1, 60):
        events.append(NS(event_id=f"e{n}", tool="t", ok=True, kind="call",
                         knowledge_refs=[], inputs=[
                             NS(path=f"a{n-1}", role="restart", sha256=f"a{n-1}"),
                             NS(path=f"b{n-1}", role="geometry", sha256=f"b{n-1}")],
                         outputs=[NS(path=f"a{n}", sha256=f"a{n}"),
                                  NS(path=f"b{n}", sha256=f"b{n}")]))
    ws = NS(ledger=NS(events=lambda: iter(events)))
    lin = Lineage(ws, lambda tool, param: {"structure": "structure"}.get(param, "restart"),
                  identity={"structure", "restart"}, parameters=set())
    verdict = lin.fits("a59", "a59", "restart", {"S"})
    assert verdict.ok and verdict.reached and len(lin._memo) <= 120


def test_a_staged_run_input_is_rendered_from_its_stages_template(ws_ctx):
    """A coupled run input is compiled from its stage's template onto what the QM selection
    wrote: the decision for that stage reads it, and no other stage's."""
    from tests.conftest import REGION, coupled_study

    from myrmidon.core.artifacts import ArtifactRef
    from myrmidon.core.lineage import lineage

    coupled_study(ws_ctx, [{"name": "reference", "kind": "single_point", "regime": "qmmm",
                            "qm_region": REGION}])
    prepared = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "reference"})
    assert prepared.ok, prepared.summary
    ws = ws_ctx.workspace
    run_input = ArtifactRef.from_path(ws.resolve(prepared.files["cpmd_input"]), "cpmd_input",
                                      root=ws.root)
    assert lineage(ws).rendered_from(run_input.path, run_input.sha256).startswith(
        "cpmd.single_point@v")


# (the decision's target and keys, whether the mandate is written). A lock on a key its
# target reads is written; one nothing reads — on a tool, a set, or anywhere — would hold
# nothing the researcher approved, and the mandate is not written
LOCKS = [
    ("script_run", {"max_segments": 5}, False),
    ("*", {"production_ns": 1}, False),
    ("gromacs.production", {"nsteps": 500000, "dt": 0.002}, True),
    ("gromacs.nvt", {"define": "-DPOSRES -DPOSRES_LIG"}, True),
    ("*", {"ref_t": 300}, True),
    ("*", {"&SYSTEM/CHARGE": 0}, True),
    ("*", {"SYSTEM/CHARGE": 0}, False),  # the gate reads a CPMD keyword by its & spelling
    ("*", {}, True),  # a choice the system files or the ceiling hold: words, no keys
    ("gromacs.dynamics", {"nsteps": 500000}, True),  # a stage template, as a purpose set
    ("gromacs.interactions", {"nsteps": 1}, False),  # only extended: no render cites it
    ("gromacs.genion", {"nstxout": 0}, False),  # genion's own run input, no job's
    ("forcefield.amber", {"rcoulomb": 1.0}, False),  # read through its purpose set
    ("box_qmmm_solute", {"distance_nm": 1.2}, False),  # renders no file
]


@pytest.mark.parametrize("target,keys,written", LOCKS)
def test_a_decision_nothing_reads_is_not_written(ws_ctx, target, keys, written):
    obs, _ = _mandate(ws_ctx, decisions_yours=[{"said": "x", "target": target, "keys": keys}])
    assert obs.ok is written, obs.summary
    if not written:
        assert obs.error.code == "MANDATE_TARGET_UNREAD"


def test_a_booking_tools_check_that_raises_is_a_refusal_never_a_crash(ws_ctx):
    from typing import ClassVar

    from pydantic import BaseModel

    from myrmidon.core.observation import Observation
    from myrmidon.core.tool import ToolRegistry, precheck, tool
    from myrmidon.execution.backend import Resources

    reg = ToolRegistry()
    reg.submission_gate = lambda ctx, params, name: ([], [])

    class P(BaseModel):
        RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("ntasks",)
        ntasks: int = 1
        preview: bool = False

        def resources(self) -> Resources:
            return Resources(ntasks=self.ntasks)

    def broken(ctx, params):
        raise OSError("stat failed")

    @tool("spend", description="x", registry=reg, expensive=True)
    def spend(ctx, params: P) -> Observation:
        precheck(ctx, broken(ctx, params))
        return Observation(ok=True, summary="submitted")  # pragma: no cover

    obs = reg.call("spend", ws_ctx, {})
    assert not obs.ok and obs.error.code == "GATE_RAISED"
    assert list(ws_ctx.workspace.ledger.events())[-1].kind == "gate_refused"


# --- unattended continuation under the mandate -----------------------------------------


def _ticks_until(ws_ctx, wake, limit=60):
    """Tick until the study is done, parked or blocked; the last report."""
    import time

    from myrmidon.orchestration.continuation import tick

    for _ in range(limit):
        report = tick(ws_ctx, wake=wake)
        if report.done or report.parked or report.blocked:
            return report
        time.sleep(0.2)
    raise AssertionError(f"no tick settled: {report.render()}")


def _recorded(ws):
    import json

    state = json.loads((ws.state_dir / "recovery.json").read_text())
    index = ws.state_dir / "wakes" / "index.jsonl"
    wakes = [json.loads(line) for line in index.read_text().splitlines()] if index.exists() \
        else []
    return state, [w for w in wakes if w["event"] == "ended"]


# (failures before a success — None submits nothing —, the mandate's reruns and model fixes,
# who collects the job first, whether it is cancelled) -> the reruns made, the fix wakes,
# whether a question is parked. A failure is handled once whoever collected it; a cancelled
# job is left as it is; a wake that leaves nothing running and no report is a question.
CONTINUATIONS = [
    (None, 1, 0, "tick", False, 0, 0, False),
    (1, 1, 0, "tick", False, 1, 0, False),
    (5, 1, 1, "tick", False, 1, 1, True),
    (5, 1, 0, "tick", False, 1, 0, True),
    (5, 1, 1, "agent", False, 0, 1, True),
    (5, 1, 0, "tick", True, 0, 0, False),
]


@pytest.mark.parametrize("fails,reruns,fixes,collector,cancel,made,fix_wakes,parked",
                         CONTINUATIONS)
def test_a_tick_collects_reruns_and_wakes_the_agent_under_the_mandate(
        ws_ctx, fails, reruns, fixes, collector, cancel, made, fix_wakes, parked):
    import json

    from myrmidon.orchestration.continuation import WakeResult, tick
    from myrmidon.orchestration.spool import Inbox

    ws = ws_ctx.workspace
    _mandate(ws_ctx, reruns=reruns, model_fixes=fixes, deliverables=[{"name": "the count"}])
    _sign(ws_ctx)
    (ws.root / "analysis").mkdir()
    (ws.root / "analysis" / "flaky.py").write_text(
        "import pathlib, sys, time\np = pathlib.Path('tries.txt')\n"
        "n = len(p.read_text().splitlines()) if p.exists() else 0\n"
        "p.write_text('x\\n' * (n + 1))\ntime.sleep(0.5)\n"
        f"sys.exit(1 if n < {fails or 0} else 0)\n")
    if fails is not None:
        ws_ctx.person = "researcher"
        first = REGISTRY.call("script_run", ws_ctx, {
            "script": "analysis/flaky.py", "kind": "glue", "outputs": ["tries.txt"],
            "walltime_min": 30})
        assert first.ok and first.job_id, first.summary
        if cancel:
            assert REGISTRY.call("job_cancel", ws_ctx, {"job_id": first.job_id}).ok
        if collector == "agent":
            REGISTRY.call("job_status", ws_ctx, {"job_id": first.job_id, "wait_seconds": 30})
        ws_ctx.person = None

    def wake(prompt, directory, turns):
        started = json.loads((directory / "index.jsonl").read_text().splitlines()[-1])
        if not started["fix"]:  # a fix wake that fixes nothing leaves the study idle
            REGISTRY.call("study_report", ws_ctx, {})
        transcript = directory / f"{started['prompt_sha256'][:8]}.jsonl"
        transcript.write_text(prompt)
        return WakeResult(transcript, "completed")

    report = _ticks_until(ws_ctx, wake)
    state, wakes = _recorded(ws)
    assert sum(state["reruns"].values()) == made and len(state["chain"]) == made
    assert sum(w["fix"] for w in wakes) == fix_wakes == state["fixes"]
    assert all(len(w["sha256"]) == 64 and w["transcript"] for w in wakes)
    assert bool(report.parked) is parked and report.done is not parked
    failed = [e.event_id for e in ws.ledger.events()
              if e.tool == "job_status" and e.kind == "call" and not e.ok]
    assert sorted(state["handled"]) == sorted(failed)
    if not parked:
        return
    # the researcher answers: `continue` adds a wake, `stop` ends the mandate's run, and a
    # revised mandate, once approved, starts afresh
    inbox = Inbox(ws.root)
    question = inbox.open_questions()[-1]
    inbox.answer(question.entry_id, "continue", by="researcher")
    before = len(wakes)
    report = _ticks_until(ws_ctx, wake)
    state, wakes = _recorded(ws)
    assert state["granted"] == 1 and len(wakes) == before + 1
    if not report.parked:  # the wake it added finished the study
        return
    inbox.answer(inbox.open_questions()[-1].entry_id, "stop", by="researcher")
    assert "stopped" in tick(ws_ctx, wake=wake).blocked
    _mandate(ws_ctx, reruns=reruns, model_fixes=fixes, ceiling_core_h=20.0, revise=True,
             reason="a larger ceiling", deliverables=[{"name": "the count"}])
    _sign(ws_ctx)
    revised = tick(ws_ctx, wake=wake)
    assert not revised.blocked and not revised.reran  # the old failures stay handled
    assert _recorded(ws)[0]["stopped"] is False


def test_without_a_wake_model_the_researcher_is_told_once(ws_ctx):
    """A failure waiting on the agent, with no model to wake, is one notice, not one a
    tick."""
    from myrmidon.orchestration.continuation import tick
    from myrmidon.orchestration.spool import Inbox

    ws = ws_ctx.workspace
    _mandate(ws_ctx, reruns=0, model_fixes=2)
    _sign(ws_ctx)
    (ws.root / "analysis").mkdir()
    (ws.root / "analysis" / "fail.py").write_text("import sys\nsys.exit(1)\n")
    ws_ctx.person = "researcher"
    job = REGISTRY.call("script_run", ws_ctx, {"script": "analysis/fail.py", "kind": "glue",
                                               "outputs": ["none.txt"],
                                               "walltime_min": 30}).job_id
    REGISTRY.call("job_status", ws_ctx, {"job_id": job, "wait_seconds": 30})
    ws_ctx.person = None
    for _ in range(4):
        tick(ws_ctx)
    assert len(Inbox(ws.root).notices()) == 1


def test_a_deliverable_measured_otherwise_than_the_mandate_states_is_said(ws_ctx):
    """The approved mandate states how a number is measured; a call that shares its first
    parameter and sets another otherwise is told, unless it matches another deliverable
    wholly; a mandate written without params keeps its digest."""
    from myrmidon.core.mandate import deliverable_findings

    refused, _ = _mandate(ws_ctx, deliverables=[{
        "name": "ligand RMSD", "tool": "observable_measure",
        "params": {"metric": "rmsd", "fit": "x"}}])
    assert not refused.ok and "takes no 'fit'" in refused.summary
    _mandate(ws_ctx, deliverables=[
        {"name": "ligand RMSD", "tool": "observable_measure",
         "params": {"metric": "rmsd", "fit_selection": "backbone"}},
        {"name": "backbone RMSD", "tool": "observable_measure",
         "params": {"metric": "rmsd", "fit_selection": "name CA"}},
        {"name": "the count"}])
    _sign(ws_ctx)
    from myrmidon.core.mandate import read_mandate

    assert "params" not in read_mandate(ws_ctx.workspace).body()["deliverables"][2]
    ws = ws_ctx.workspace
    assert deliverable_findings(ws, "observable_measure", {"metric": "rmsd",
                                                     "fit_selection": "backbone"}) == []
    assert deliverable_findings(ws, "observable_measure", {"metric": "rmsd",
                                                     "fit_selection": "name CA"}) == []
    assert deliverable_findings(ws, "observable_measure", {"metric": "mindist"}) == []
    said = deliverable_findings(ws, "observable_measure", {"metric": "rmsd",
                                                     "fit_selection": "protein"})
    assert {f.id for f in said} == {"DELIVERABLE_MEASURED_OTHERWISE"} and len(said) == 2
    assert "'protein'" in said[0].message


# (how the run's collection ended, what it says) -> whether its checkpoint descends from the
# system. A run stopped at its walltime still wrote its checkpoint from what was submitted;
# one whose node read other inputs than were submitted wrote it from nothing on record.
COLLECTIONS = [(True, [], True), (False, [], True), (False, ["RECEIPT_INPUT_CHANGED"], False)]


@pytest.mark.parametrize("ok,refs,reached", COLLECTIONS)
def test_a_runs_checkpoint_is_its_submissions_however_it_ended(ok, refs, reached):
    from types import SimpleNamespace as NS

    from myrmidon.core.lineage import Lineage

    def ev(eid, tool, inputs, outputs, ok=True, refs=()):
        return NS(event_id=eid, tool=tool, ok=ok, kind="call", knowledge_refs=list(refs),
                  inputs=[NS(path=p, role=r, sha256=p) for p, r in inputs],
                  outputs=[NS(path=p, sha256=p) for p in outputs])

    events = [ev("e1", "grompp", [("sys.gro", "structure")], ["md.tpr"]),
              ev("e2", "mdrun", [("md.tpr", "tpr")], []),
              ev("e3", "job_status", [], ["md.cpt"], ok=ok, refs=["collects:e2", *refs])]
    kinds = {"structure": "structure", "tpr": "tpr"}
    lin = Lineage(NS(ledger=NS(events=lambda: iter(events))),
                  lambda tool, param: kinds.get(param), identity={"structure", "checkpoint"},
                  parameters=set())
    verdict = lin.fits("md.cpt", "md.cpt", "checkpoint", {"sys.gro"})
    assert (verdict.ok and verdict.reached) is reached
