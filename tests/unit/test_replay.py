"""Replay a study from its ledger through the tools, and the ledger-only report."""


from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.replay import (
    plan,
    planted_files,
    render,
    replay,
    stage_key,
)
from myrmidon.core.tool import REGISTRY
from myrmidon.core.workspace import Workspace

MDP = (
    "integrator = md\ndt = 0.002\nnsteps = 5000\nconstraints = h-bonds\ncoulombtype = PME\n"
    "tcoupl = v-rescale\ntc-grps = System\ntau_t = 0.1\nref_t = 300\n"
    "gen_vel = yes\ngen_temp = 300\n"
)


def _wait(ctx, job_id):
    for _ in range(200):
        st = REGISTRY.call("job_status", ctx, {"job_id": job_id, "wait_seconds": 1})
        if st.state in ("COMPLETED", "FAILED"):
            return st
    raise AssertionError("job did not finish")


def _study(ws_ctx):
    """A small study: planted inputs, a written mdp, grompp, mdrun, results, energy."""
    ws = ws_ctx.workspace
    inputs = ws.stage("input")
    (inputs / "conf.gro").write_text("mock\n 3\n" + "    1ALA     CA    1   0.1   0.2   0.3\n" * 3
                                     + "   2.5 2.5 2.5\n")
    (inputs / "topol.top").write_text("[ system ]\nMock\n[ molecules ]\nProtein 1\n")
    calls = [
        ("study_open", {}),
        ("file_write", {"path": "01_input/nvt.mdp", "content": MDP}),
        ("file_check", {"path": "01_input/nvt.mdp"}),
        ("gmx_grompp", {"mdp": "01_input/nvt.mdp", "structure": "01_input/conf.gro",
                        "topology": "01_input/topol.top", "output": "nvt.tpr", "stage": "nvt"}),
        ("gmx_mdrun", {"tpr": "02_nvt/nvt.tpr", "deffnm": "nvt", "stage": "02_nvt",
                       "walltime_min": 5}),
    ]
    for name, params in calls:
        obs = REGISTRY.call(name, ws_ctx, params)
        assert obs.ok, (name, obs.summary)
    job = obs.job_id
    res = _wait(ws_ctx, job)  # the poll that finds it ended is the collection
    assert res.ok, res.summary
    en = REGISTRY.call("gmx_energy", ws_ctx, {"edr": "02_nvt/nvt.edr", "terms": ["Temperature"]})
    assert en.ok, en.summary
    return ws, job


def test_stage_key_folds_double_numbering():
    assert stage_key("03_02_nvt/nvt.edr") == stage_key("02_nvt/nvt.edr") == "nvt/nvt.edr"
    assert stage_key("topol.top") == "topol.top"
    assert stage_key("100_sp_0096/cpmd.inp") == "sp_0096/cpmd.inp"


def test_a_study_past_99_stages_finds_and_numbers_its_stages(ws_ctx):
    ws = ws_ctx.workspace
    for i in range(1, 100):
        (ws.root / f"{i:02d}_s{i}").mkdir()
    hundredth = ws.stage("sp_0096")
    assert hundredth.name == "100_sp_0096"
    assert ws.stage("sp_0096") == hundredth  # the second tool in the stage finds it
    assert ws.stage("sp_0097").name == "101_sp_0097"
    assert ws.stages()[98:] == ["99_s99", "100_sp_0096", "101_sp_0097"]
    (ws.root / "50_s50").rmdir()
    assert ws.stage("next").name == "102_next"  # a removed number is not handed out again


def test_replay_reproduces_the_study_through_the_tools(ws_ctx, tmp_path):
    src, old_job = _study(ws_ctx)
    events = list(src.ledger.events())
    actions = dict((ev.tool, a) for ev, a in plan(events))
    assert actions["study_open"].startswith("skipped")
    assert actions["gmx_grompp"] == "replay"
    # a poll is an observation, not an action: the replay waits on the job instead; the
    # poll that collected the job is a call, and is replayed (after waiting on the twin)
    by_kind = {(ev.tool, ev.kind): a for ev, a in plan(events)}
    assert by_kind.get(("job_status", "observe"), "skipped: observe") == "skipped: observe"
    assert by_kind[("job_status", "call")] == "replay"
    assert planted_files(src, events) == ["01_input/conf.gro", "01_input/topol.top"]

    report = replay(src, ws_ctx, tmp_path / "twin")
    assert report.n_failed == 0, render(report)
    replayed = [s.tool for s in report.steps if s.action == "replayed"]
    assert replayed == ["file_write", "gmx_grompp", "gmx_mdrun", "job_status", "gmx_energy"]
    new_job = report.job_map[old_job]
    assert new_job != old_job and new_job.startswith("job-")
    twin = Workspace.open(tmp_path / "twin")
    assert (twin.root / "01_input" / "conf.gro").exists()  # planted inputs copied
    assert twin.stages() == ["01_input", "02_nvt", "03_analysis"]
    status = {a.path: a.status for a in report.artifacts}
    assert status["01_input/nvt.mdp"] == "identical"
    assert status["02_nvt/nvt.tpr"] == "identical"
    assert status["02_nvt/nvt.edr"] == "identical"  # the mock is deterministic
    # the twin's own ledger is a full record: the replay is itself replayable
    twin_events = [e for e in twin.ledger.events() if e.kind == "call" and e.ok]
    assert [e.tool for e in twin_events] == replayed
    assert [e.tool for e in twin.ledger.events() if e.kind == "observe"] == ["job_status"] * sum(
        1 for e in twin.ledger.events() if e.kind == "observe"
    )
    text = render(report)
    assert "5 tool call(s) replayed, 0 failed" in text and old_job in text


def test_replay_refuses_an_existing_target(ws_ctx, tmp_path):
    src, _ = _study(ws_ctx)
    import pytest

    with pytest.raises(FileExistsError):
        replay(src, ws_ctx, src.root)


def test_replay_reproduces_a_protocol_rendered_file(ws_ctx, tmp_path):
    """inputset_apply is deterministic, so a replay must reproduce the mdp and its
    sidecar byte for byte — including the acknowledgement that allowed the override."""
    obs = REGISTRY.call(
        "inputset_apply",
        ws_ctx,
        {
            "inputset": "em_solvated_biomolecule",
            "output": "01_input/em.mdp",
            "overrides": {"rvdw": 1.2, "nsteps": 5000},
            "acknowledge_warnings": ["INPUTSET_VALUE_OVERRIDDEN"],
        },
    )
    assert obs.ok, obs.summary

    report = replay(ws_ctx.workspace, ws_ctx, tmp_path / "twin")
    assert report.n_failed == 0, render(report)
    assert [s.tool for s in report.steps if s.action == "replayed"] == ["inputset_apply"]
    status = {a.path: a.status for a in report.artifacts}
    assert status["01_input/em.mdp"] == "identical"
    assert status["01_input/em.mdp.myrmidon.json"] == "identical"


def test_a_planted_file_edited_in_place_is_still_planted(ws_ctx):
    """gmx_solvate rewrites the topology it was given, and reports it as an output.
    Counting that as "produced" would leave the replay with nothing to plant, so the very
    call that produced it could not run. A call that both consumes and produces a path
    modified it; the twin still needs the original."""
    from myrmidon.core.provenance import ProvenanceEvent
    from myrmidon.core.replay import planted_files

    ws = ws_ctx.workspace
    stage = ws.stage("input")
    (stage / "acetone.top").write_text("; planted topology\n")
    (stage / "acetone.gro").write_text("mock\n 1\nx\n 1 1 1\n")
    top = ws.relpath(stage / "acetone.top")
    gro = ws.relpath(stage / "acetone.gro")

    def ref(path, role):
        return ArtifactRef.from_path(ws.root / path, role, root=ws.root)

    edited_in_place = ProvenanceEvent(
        tool="gmx_solvate",
        params={},
        inputs=[ref(gro, "structure"), ref(top, "topology")],
        outputs=[ref(top, "topology")],
        ok=True,
    )
    assert top in planted_files(ws, [edited_in_place])

    # a file a tool genuinely created is not planted: the replay rebuilds it
    (stage / "made.top").write_text("; created by a tool\n")
    made = ws.relpath(stage / "made.top")
    created = ProvenanceEvent(
        tool="gmx_pdb2gmx", params={}, inputs=[], outputs=[ref(made, "topology")], ok=True,
    )
    assert made not in planted_files(ws, [created, edited_in_place])


def test_a_retired_tool_in_an_old_ledger_is_skipped_not_replayed():
    """A ledger may hold events of a tool the registry does not have; a replay of such a
    study passes over them, as it does any navigation call (which declares itself:
    ToolSpec.navigation)."""
    from myrmidon.core.provenance import ProvenanceEvent
    from myrmidon.core.replay import plan

    events = [ProvenanceEvent(tool=t, params={}, ok=True, kind="call", myrmidon_version="x")
              for t in ("lesson_record", "lesson_search", "skill_lookup")]
    assert [why for _, why in plan(events)] == ["skipped: session/knowledge/viewer tool"] * 3
