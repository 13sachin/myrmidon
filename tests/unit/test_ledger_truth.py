"""What the ledger has to record before a study can be read as a graph.

Two claims: what a call consumed is recorded for every tool, and a poll is recorded as
a look rather than as one of the study's actions.

A study's data-flow edges — "this grompp consumed that inputset_apply's mdp" — are
recorded by the registry, hashed, for every tool, never recovered by matching path
strings inside ``params`` against earlier outputs. The acetone tutorial ledgers are the
fixtures: the graph is generated from a study that actually ran, not one a test invented.
"""


import pytest
from pydantic import BaseModel

from myrmidon.core.artifacts import consumed_inputs
from myrmidon.core.observation import Observation
from myrmidon.core.tool import REGISTRY, ToolRegistry, tool

MDP = (
    "integrator = md\ndt = 0.002\nnsteps = 5000\nconstraints = h-bonds\ncoulombtype = PME\n"
    "tcoupl = v-rescale\ntc-grps = System\ntau_t = 0.1\nref_t = 300\n"
    "gen_vel = yes\ngen_temp = 300\n"
)
GRO = "mock\n 3\n" + "    1ALA     CA    1   0.1   0.2   0.3\n" * 3 + "   2.5 2.5 2.5\n"
TOP = "[ system ]\nMock\n[ molecules ]\nProtein 1\n"


def _plant(ws):
    stage = ws.stage("input")
    (stage / "conf.gro").write_text(GRO)
    (stage / "topol.top").write_text(TOP)
    (stage / "nvt.mdp").write_text(MDP)
    return stage


def _event(ws, tool_name):
    return [ev for ev in ws.ledger.events() if ev.tool == tool_name][-1]


def test_every_file_a_call_reads_is_an_input_and_an_output_is_the_next_calls_input(ws_ctx):
    """A stage's preparation records what it compiled from, each under the kind it is and with
    the hash it had; its run reads the run input it wrote, by hash — the edge the study's data
    flow is made of."""
    from tests.conftest import NVT_STAGE, approved_study

    from myrmidon.core.artifacts import sha256_file

    ws = ws_ctx.workspace
    _plant(ws)
    approved_study(ws_ctx, "01_input/conf.gro", "01_input/topol.top", [NVT_STAGE])
    prepared = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "nvt"})
    assert prepared.ok, prepared.summary
    inputs = {ref.role: ref for ref in _event(ws, "stage_prepare").inputs}
    assert {r: inputs[r].path for r in ("structure", "topology")} == {
        "structure": "01_input/conf.gro", "topology": "01_input/topol.top"}
    assert inputs["topology"].sha256 == sha256_file(ws.root / "01_input/topol.top")
    assert inputs["mdp"].path == prepared.files["mdp"]

    tpr = next(a for a in prepared.artifacts if a.role == "tpr")
    run = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt", "walltime_min": 5})
    assert run.ok, run.summary
    consumed = {r.role: r for r in _event(ws, "stage_run").inputs}
    assert (consumed["inputs.tpr"].path, consumed["inputs.tpr"].sha256) == (tpr.path, tpr.sha256)


def test_a_later_study_of_the_project_is_told_what_an_earlier_one_wrote_and_reads_it(ws_ctx):
    """What an earlier study's collected runs wrote is listed by study_open in a later study of
    the same project, each file by the <study>/<path> the later study names it by; a call there
    reads it and records it, hashed, as its input."""
    from tests.conftest import NVT_STAGE, approved_study, wait_for

    ws = ws_ctx.workspace
    _plant(ws)
    approved_study(ws_ctx, "01_input/conf.gro", "01_input/topol.top", [NVT_STAGE])
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "nvt"}).ok
    run = REGISTRY.call("stage_run", ws_ctx, {"stage": "nvt", "walltime_min": 5})
    done = wait_for(ws_ctx, run.job_id)
    assert done.ok, done.summary
    assert REGISTRY.call("study_create", ws_ctx, {"name": "later", "description": "x"}).ok
    opened = REGISTRY.call("study_open", ws_ctx, {})
    (earlier,) = opened.studies
    assert (earlier["name"], earlier["question"], earlier["reported"]) == (
        ws.root.name, "Run the stages", False)
    wrote = {a.role: a for a in done.artifacts}
    named = {o["role"]: o["path"] for o in earlier["outputs"] if o["stage"] == "nvt"}
    assert set(named) == set(wrote)
    later = ws_ctx.workspace
    (read,) = consumed_inputs({"structure": named["structure"]}, later.root, later.origin)
    assert (read.path, read.sha256) == (str(ws.root / wrote["structure"].path),
                                        wrote["structure"].sha256)


@pytest.fixture
def study_and_project(ws_ctx, tmp_path):
    """A study with planted inputs, harness state, and a project directory beside it
    holding the researcher's own file; plus a file in neither."""
    ws = ws_ctx.workspace
    stage = _plant(ws)
    (stage / "frame_0.gro").write_text(GRO)
    (stage / "frame_1.gro").write_text(GRO)
    (ws.state_dir / "jobs.sqlite").write_text("x")
    project = tmp_path / "project"
    (project / "box").mkdir(parents=True)
    (project / "box" / "heat.gro").write_text(GRO)
    (tmp_path / "elsewhere.mdp").write_text(MDP)
    # an earlier study of the project, and a study of another project, beside this one
    from myrmidon.core.workspace import Workspace

    for name, origin in (("equil", project), ("theirs", tmp_path / "other")):
        made = Workspace.create(ws.root.parent / name, name, origin=origin)
        (made.stage("npt") / "confout.gro").write_text(GRO)
    return ws.root, project, tmp_path


# (the call's params, with {project} and {elsewhere} filled in; the inputs it records as
# role -> path). Only the study and the researcher's project are the study's inputs: not a
# GROMACS library file, not a stray path, not harness state.
CONSUMED = [
    ({"structure": "01_input/conf.gro", "solvent": "spc216.gro", "mdp": "{elsewhere}"},
     {"structure": "01_input/conf.gro"}),
    ({"db": ".myrmidon/jobs.sqlite"}, {}),
    ({"frames": ["01_input/frame_0.gro", "01_input/frame_1.gro"],
      "extra": {"index": "01_input/topol.top", "count": 3}, "n": 20},
     {"frames[0]": "01_input/frame_0.gro", "frames[1]": "01_input/frame_1.gro",
      "extra.index": "01_input/topol.top"}),
    # the researcher's own file, named relative to the project or absolutely
    ({"structure": "box/heat.gro"}, {"structure": "{project}/box/heat.gro"}),
    ({"structure": "{project}/box/heat.gro"}, {"structure": "{project}/box/heat.gro"}),
    # an earlier study's file, named <study>/<path> or absolutely; not another project's,
    # and a bare name is never searched for across studies
    ({"structure": "equil/01_npt/confout.gro"},
     {"structure": "{studies}/equil/01_npt/confout.gro"}),
    ({"structure": "{studies}/equil/01_npt/confout.gro"},
     {"structure": "{studies}/equil/01_npt/confout.gro"}),
    ({"structure": "theirs/01_npt/confout.gro"}, {}),
    ({"structure": "confout.gro"}, {}),
]


@pytest.mark.parametrize("params,recorded", CONSUMED)
def test_what_a_call_consumes_is_recorded_with_where_it_sat(study_and_project, params,
                                                            recorded):
    import json

    root, project, tmp = study_and_project
    fill = {"project": str(project), "elsewhere": str(tmp / "elsewhere.mdp"),
            "studies": str(root.parent)}
    params = json.loads(json.dumps(params).replace("{project}", fill["project"])
                        .replace("{elsewhere}", fill["elsewhere"])
                        .replace("{studies}", fill["studies"]))
    recorded = {k: v.format(**fill) for k, v in recorded.items()}
    refs = consumed_inputs(params, root, project)
    assert {r.role: r.path for r in refs} == recorded


def test_a_tool_that_records_its_own_inputs_keeps_them_as_they_were_read(ws_ctx):
    """gmx_solvate names the topology it copied out of the way: the
    registry's walk must not displace or duplicate the tool's own record, and the hash
    recorded is the state the call started from."""
    from myrmidon.core.artifacts import sha256_file

    ws = ws_ctx.workspace
    stage = _plant(ws)
    (stage / "boxed.gro").write_text(GRO)
    before = (stage / "topol.top").read_text()
    obs = REGISTRY.call("gmx_solvate", ws_ctx, {
        "structure": "01_input/boxed.gro", "topology": "01_input/topol.top",
    })
    assert obs.ok, obs.summary
    recorded = _event(ws, "gmx_solvate").inputs
    assert [(r.role, r.path) for r in recorded] == [(r.role, r.path) for r in obs.inputs]
    assert len({r.path for r in recorded}) == len(recorded)
    top_in = next(r for r in recorded if r.role == "topology")
    assert (stage / "topol.top").read_text() == before  # planted input left alone
    assert top_in.sha256 == sha256_file(stage / "topol.top")


def test_a_refused_call_records_what_it_would_have_consumed(ws_ctx, tmp_path):
    reg = ToolRegistry()

    class P(BaseModel):
        structure: str

    def gate(ctx, params):
        from myrmidon.core.validation import Precondition

        return [Precondition(id="X1", message="no")], ["x"]

    @tool("gated_thing", description="x", expensive=True, registry=reg)
    def gated_thing(ctx, params: P) -> Observation:
        from myrmidon.core.tool import precheck

        precheck(ctx, gate(ctx, params)[0])
        return Observation(ok=True, summary="ran")  # pragma: no cover - refused

    _plant(ws_ctx.workspace)
    obs = reg.call("gated_thing", ws_ctx, {"structure": "01_input/conf.gro"})
    assert not obs.ok
    ev = [e for e in ws_ctx.workspace.ledger.events() if e.tool == "gated_thing"][-1]
    assert ev.kind == "gate_refused"
    assert [r.path for r in ev.inputs] == ["01_input/conf.gro"]


# --- polls are observations, not actions; the poll that finds the job ended is the
# --- collection, and that one is an action -------------------------------------------


def _run_a_job(ws_ctx):
    from tests.conftest import NVT_STAGE, approved_study, prepare_and_run

    ws = ws_ctx.workspace
    _plant(ws)
    approved_study(ws_ctx, "01_input/conf.gro", "01_input/topol.top", [NVT_STAGE])
    _, run = prepare_and_run(ws_ctx, "nvt", walltime_min=5)
    for _ in range(200):
        st = REGISTRY.call("job_status", ws_ctx, {"job_id": run.job_id, "wait_seconds": 1})
        if st.state in ("COMPLETED", "FAILED"):
            return run.job_id, st
    raise AssertionError("job did not finish")


def test_a_poll_is_a_look_and_the_collection_is_a_call(ws_ctx):
    job_id, last = _run_a_job(ws_ctx)
    polls = [ev for ev in ws_ctx.workspace.ledger.events() if ev.tool == "job_status"]
    assert polls, "the poll that saw the job finish is worth recording"
    # exactly one of them is the collection: the first look after the end, carrying the
    # artifacts; every other poll that changed something is a look, never an action
    calls = [ev for ev in polls if ev.kind == "call"]
    assert len(calls) == 1 and {a.role for a in calls[0].outputs} >= {"log", "energy"}
    assert {ev.kind for ev in polls if ev is not calls[0]} <= {"observe"}
    # and looking again records nothing: the outcome is already in the ledger
    again = REGISTRY.call("job_status", ws_ctx, {"job_id": job_id})
    assert again.ok and again.kind == "observe" and not again.provenance_id
    assert len([ev for ev in ws_ctx.workspace.ledger.events() if ev.tool == "job_status"]) \
        == len(polls)


# --- the fields a graph needs, and the schema it was written against ----------------


def test_every_call_records_the_params_schema_it_was_made_against(ws_ctx):
    from myrmidon.core.tool import params_schema_hash

    ws = ws_ctx.workspace
    _plant(ws)
    obs = REGISTRY.call("file_check", ws_ctx, {"path": "01_input/nvt.mdp"})
    assert obs.ok, obs.summary
    ev = _event(ws, "file_check")
    assert ev.params_schema == params_schema_hash(REGISTRY.get("file_check").params_model)
    assert len(ev.params_schema) == 12
