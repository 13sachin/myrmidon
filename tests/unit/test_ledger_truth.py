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


def test_every_file_a_call_names_is_an_input_and_an_output_is_the_next_calls_input(ws_ctx):
    """grompp names three study files and the ledger records three inputs, each with the
    parameter it arrived as and the hash it had when the call started; mdrun's tpr is
    grompp's tpr, by hash — the edge the study's data flow is made of."""
    from myrmidon.core.artifacts import sha256_file

    ws = ws_ctx.workspace
    _plant(ws)
    grompp = REGISTRY.call("gmx_grompp", ws_ctx, {
        "mdp": "01_input/nvt.mdp", "structure": "01_input/conf.gro",
        "topology": "01_input/topol.top", "output": "nvt.tpr", "stage": "nvt",
    })
    assert grompp.ok, grompp.summary
    inputs = {ref.role: ref for ref in _event(ws, "gmx_grompp").inputs}
    assert {r: i.path for r, i in inputs.items()} == {
        "mdp": "01_input/nvt.mdp", "structure": "01_input/conf.gro",
        "topology": "01_input/topol.top"}
    assert inputs["topology"].sha256 == sha256_file(ws.root / "01_input/topol.top")

    tpr = next(a for a in grompp.artifacts if a.role == "tpr")
    run = REGISTRY.call("gmx_mdrun", ws_ctx, {
        "tpr": tpr.path, "deffnm": "nvt", "stage": "02_nvt", "walltime_min": 5,
    })
    assert run.ok, run.summary
    consumed = _event(ws, "gmx_mdrun").inputs
    assert [(r.role, r.path, r.sha256) for r in consumed] == [("tpr", tpr.path, tpr.sha256)]


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
]


@pytest.mark.parametrize("params,recorded", CONSUMED)
def test_what_a_call_consumes_is_recorded_with_where_it_sat(study_and_project, params,
                                                            recorded):
    import json

    root, project, tmp = study_and_project
    fill = {"project": str(project), "elsewhere": str(tmp / "elsewhere.mdp")}
    params = json.loads(json.dumps(params).replace("{project}", fill["project"])
                        .replace("{elsewhere}", fill["elsewhere"]))
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
        acknowledge_warnings: list[str] = []

    def gate(ctx, params):
        from myrmidon.core.validation import ValidationWarning

        return [ValidationWarning(id="X1", severity="block", message="no")], ["x"]

    @tool("gated_thing", description="x", expensive=True, gate=gate, registry=reg)
    def gated_thing(ctx, params: P) -> Observation:  # pragma: no cover - refused
        return Observation(ok=True, summary="ran")

    _plant(ws_ctx.workspace)
    obs = reg.call("gated_thing", ws_ctx, {"structure": "01_input/conf.gro"})
    assert not obs.ok
    ev = [e for e in ws_ctx.workspace.ledger.events() if e.tool == "gated_thing"][-1]
    assert ev.kind == "gate_refused"
    assert [r.path for r in ev.inputs] == ["01_input/conf.gro"]


# --- polls are observations, not actions; the poll that finds the job ended is the
# --- collection, and that one is an action -------------------------------------------


def _run_a_job(ws_ctx):
    ws = ws_ctx.workspace
    _plant(ws)
    grompp = REGISTRY.call("gmx_grompp", ws_ctx, {
        "mdp": "01_input/nvt.mdp", "structure": "01_input/conf.gro",
        "topology": "01_input/topol.top", "output": "nvt.tpr", "stage": "nvt",
    })
    assert grompp.ok, grompp.summary
    run = REGISTRY.call("gmx_mdrun", ws_ctx, {
        "tpr": "02_nvt/nvt.tpr", "deffnm": "nvt", "stage": "02_nvt", "walltime_min": 5,
    })
    assert run.ok, run.summary
    for _ in range(200):
        st = REGISTRY.call("job_status", ws_ctx, {"job_id": run.job_id, "wait_seconds": 1})
        if st.state in ("COMPLETED", "FAILED"):
            return run.job_id, st
    raise AssertionError("job did not finish")


def test_a_poll_is_a_look_the_collection_is_a_call_and_replay_repeats_only_the_call(
        ws_ctx):
    from myrmidon.core.replay import plan

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

    by_kind = {(ev.tool, ev.kind): action
               for ev, action in plan(list(ws_ctx.workspace.ledger.events()))}
    assert by_kind[("gmx_mdrun", "call")] == "replay"
    assert by_kind[("job_status", "call")] == "replay"  # the collection, after waiting
    assert by_kind.get(("job_status", "observe"), "skipped: observe") == "skipped: observe"


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


def test_a_replay_names_every_tool_whose_params_changed_since_the_ledger():
    """A changed params model must not break the replay of an old ledger silently: the
    drift is named, by tool, with how many events were recorded against it — and a tool
    missing from the registry is named too."""
    from myrmidon.core.provenance import ProvenanceEvent
    from myrmidon.core.replay import schema_drift
    from myrmidon.core.tool import params_schema_hash

    current = params_schema_hash(REGISTRY.get("file_check").params_model)
    drift = schema_drift([
        ProvenanceEvent(tool="gmx_grompp", params_schema="0" * 12),
        ProvenanceEvent(tool="gmx_grompp", params_schema="0" * 12),
        ProvenanceEvent(tool="file_check", params_schema=current),
        ProvenanceEvent(tool="gmx_mdrun"),  # a ledger older than the field
        ProvenanceEvent(tool="gmx_retired", params_schema="abc"),
    ])
    assert {d.tool: d.n_events for d in drift} == {"gmx_grompp": 2, "gmx_retired": 1}
    missing = next(d for d in drift if d.tool == "gmx_retired").current
    assert "registered" in missing and missing != current  # a reason, not a hash
