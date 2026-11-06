"""Registry invariants and the gate/provenance machinery."""

import json
from typing import ClassVar

import pytest
from pydantic import BaseModel
from tests.unit.findings_catalog import in_group

from myrmidon.core.observation import Observation
from myrmidon.core.tool import REGISTRY, ToolRegistry, ToolSpec, precheck
from myrmidon.core.validation import Finding, Precondition, Theirs


def _load_all_tools():
    import myrmidon.tools as basic

    basic.register_all()


def test_registry_invariants():
    """Every registered tool: Observation return type, JSON-serializable schema."""
    _load_all_tools()
    specs = REGISTRY.specs()
    assert len(specs) >= 15
    for spec in specs:
        schema = spec.params_model.model_json_schema()
        json.dumps(schema)  # must round-trip
        assert spec.description, spec.name


class _DummyParams(BaseModel):
    value: int = 1


def _make_registry_with_gated_tool(warn_severity="warn"):
    reg = ToolRegistry()

    def gate(ctx, params):
        return (
            [{"block": Precondition, "warn": Theirs, "info": Finding}[warn_severity](
                id="TEST_WARNING", message="test")],
            ["test_check"],
        )

    def fn(ctx, params):
        precheck(ctx, gate(ctx, params)[0])
        return Observation(ok=True, summary=f"ran with {params.value}")

    reg.register(
        ToolSpec(
            name="dummy_gated",
            description="test tool",
            params_model=_DummyParams,
            expensive=True,
            needs_workspace=False,
        ),
        fn,
    )
    return reg


# (finding severity, the call runs, where the finding is). A block refuses; a warn is the
# researcher's and refuses a caller who is not one; an info is recorded and the call runs.
GATE = [
    ("warn", False, "unapproved"),
    ("info", True, "advisory"),
    ("block", False, "blocking"),
]


@pytest.mark.parametrize("severity,runs,held_in", GATE)
def test_a_block_and_a_researchers_finding_refuse_and_an_info_runs(ctx, severity, runs,
                                                                    held_in):
    reg = _make_registry_with_gated_tool(severity)
    ctx.person = None
    obs = reg.call("dummy_gated", ctx, {"value": 2})
    assert obs.ok is runs
    if not runs:
        assert obs.failure == "gate_refused"
    assert [w.id for w in in_group(obs, held_in)] == ["TEST_WARNING"]


class _BooksUnnamed(BaseModel):
    """Books resources without naming the fields a plan would price it from."""
    walltime_min: int = 60

    def resources(self):
        return None


class _BooksUnchecked(BaseModel):
    """Books resources and names them, but its function does not check before it acts."""
    RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("value",)
    value: int = 1
    preview: bool = False

    def resources(self):
        return None


# A tool the registry refuses to hold: (params, expensive?, what the refusal names)
UNREGISTRABLE = [
    # the gate would price it as free
    (_BooksUnnamed, True, "RESOURCE_FIELDS"),
    # its own checks could not be collected before it books
    (_BooksUnchecked, True, "precheck"),
]


@pytest.mark.parametrize("params,expensive,names", UNREGISTRABLE)
def test_a_tool_that_breaks_a_registry_rule_is_not_registered(params, expensive, names):
    with pytest.raises(Exception, match=names):
        ToolRegistry().register(
            ToolSpec(name="bad", description="x", params_model=params, expensive=expensive),
            lambda ctx, p: Observation(ok=True, summary="x"),
        )


def test_a_tool_described_at_length_is_not_registered():
    with pytest.raises(Exception, match="at most 300"):
        ToolRegistry().register(
            ToolSpec(name="wordy", description="x" * 301, params_model=_DummyParams),
            lambda ctx, p: Observation(ok=True, summary="x"))


def test_invalid_params_return_failed_observation(ctx):
    _load_all_tools()
    obs = REGISTRY.call("fetch_pdb", ctx, {"wrong_field": 1})
    assert not obs.ok and obs.failure == "invalid_params"


def test_provenance_recorded_per_call(ws_ctx):
    events_before = len(list(ws_ctx.workspace.ledger.events()))
    obs = REGISTRY.call("study_open", ws_ctx, {})
    assert obs.ok
    events = list(ws_ctx.workspace.ledger.events())
    assert len(events) == events_before + 1
    assert obs.provenance_id == events[-1].event_id
    assert events[-1].tool == "study_open"


def test_a_ledger_event_names_the_findings_its_call_surfaced(ws_ctx):
    """Knowledge lives in gates, parsers and diagnostics; the event says which of it was
    in play, by id. A tool's own guard names no finding, so its event names none."""
    obs = REGISTRY.call("inputset_apply", ws_ctx, {
        "inputset": "gromacs.rerun", "method": "forcefield.amber", "output": "01_input/rerun.mdp",
        "overrides": {"integrator": "sd"},
    })
    ids = [f.id for f in obs.findings]
    assert obs.ok and ids and len(ids) == len(set(ids))  # once each
    event = list(ws_ctx.workspace.ledger.events())[-1]
    assert event.kind == "call"
    assert "INPUTSET_VALUE_OVERRIDDEN" in event.knowledge_refs

    guard = REGISTRY.call("file_write", ws_ctx, {"path": "../escape.mdp", "content": "x\n"})
    assert not guard.ok
    assert list(ws_ctx.workspace.ledger.events())[-1].knowledge_refs == []


def test_unexpected_exception_returns_structured_failure(ctx):
    """A tool bug must not crash the server — it becomes an ok=False Observation."""
    reg = ToolRegistry()

    class P(BaseModel):
        pass

    def exploding(ctx, params):
        raise ValueError("dependency blew up")

    reg.register(
        ToolSpec(
            name="exploding",
            description="x",
            params_model=P,
            needs_workspace=False,
        ),
        exploding,
    )
    obs = reg.call("exploding", ctx, {})
    assert not obs.ok and obs.failure == "raised"
    assert "dependency blew up" in obs.log_tail  # the traceback, for diagnosis



def test_a_findings_kind_is_its_class_and_survives_a_record():
    """`severity` is not a field one can pass (a block cannot be built as information by
    mistake), and a finding read back from JSON keeps the kind it was written as."""
    import pytest

    from myrmidon.core.observation import Observation

    with pytest.raises(ValueError):
        Finding(id="X", message="m", severity="block")
    obs = Observation(ok=False, summary="s", findings=[
        Precondition(id="A", message="a"), Theirs(id="B", message="b"),
        Finding(id="C", message="c")])
    back = Observation.model_validate_json(obs.model_dump_json())
    assert [f.severity for f in back.findings] == ["block", "warn", "info"]


def test_two_calls_making_one_stage_at_once_share_it(tmp_path, monkeypatch):
    """A stage another call made between this one's look and its mkdir is this one's too;
    a file in the way is an error, never a loop."""
    from myrmidon.core.workspace import Workspace

    ws = Workspace.create(tmp_path / "study", name="s")
    monkeypatch.setattr(ws, "stages", lambda: [])  # the look saw no stage yet
    (ws.root / "01_analysis").mkdir()  # what the call beside it made meanwhile
    assert ws.stage("analysis") == ws.root / "01_analysis"
    (ws.root / "01_b").write_text("not a directory")
    with pytest.raises(FileExistsError):
        ws.stage("b")


# (a parameter as a client's model spelled it, what the tool receives). A spelling whose
# meaning is not in doubt is read as meant; anything else is left for validation to refuse.
SPELLINGS = [
    ({"concentration_mol_l": "None"}, {"concentration_mol_l": None}),
    ({"concentration_mol_l": "null"}, {"concentration_mol_l": None}),
    ({"concentration_mol_l": 0.15}, {"concentration_mol_l": 0.15}),
    ({"questions": "How long?"}, {"questions": ["How long?"]}),
    ({"questions": '["a?", "b?"]'}, {"questions": ["a?", "b?"]}),
    ({"question": "None"}, {"question": "None"}),  # a string field keeps its words
    ({"concentration_mol_l": ""}, {"concentration_mol_l": None}),  # left empty: no value
    ({"question": ""}, {"question": ""}),
]


@pytest.mark.parametrize("given,read", SPELLINGS)
def test_a_parameter_spelled_as_a_string_is_read_as_meant(given, read):
    from pydantic import BaseModel

    from myrmidon.core.tool import lenient

    class Params(BaseModel):
        question: str = ""
        questions: list[str] = []
        concentration_mol_l: float | None = 0.15

    assert lenient(Params, given) == read


@pytest.mark.parametrize("tool,args", [
    ("mandate_write", {"question": "q", "questions": '["How long, in ps?"]'}),
    ("mandate_write", {"question": "q", "questions": "How long would you like it?"}),
    ("stage_run", {"stage": "x", "walltime_min": "None"}),
    # nested lists written as one word or as nothing, a proposal with what it would lock
    ("mandate_write", {"question": "q", "subsystems": {
        "ligand": {"kind": "ligand", "resnames": "JZ4", "chains": None, "resid": 164}},
        "decisions_proposed": [{"what": "x", "locks": {"npt.duration_ps": 200}}]}),
    # a revision states only what changes; make_ndx's own end written out
    ("mandate_write", {"revise": True, "reason": "a bigger ceiling", "ceiling_core_h": 96}),
    ("gmx_make_ndx", {"structure": "x.gro", "commands": ["q"]}),
])
def test_the_published_schema_admits_the_spellings_the_registry_reads(tool, args):
    """A client checks a call against the schema before the server sees it: the string
    spellings pass it, and the field's own constraints stay on its own type."""
    from mcp.server.mcpserver.utilities.func_metadata import func_metadata

    import myrmidon.tools
    from myrmidon.core.tool import REGISTRY
    from myrmidon.server.mcp_server import _make_handler

    myrmidon.tools.register_all()
    model = func_metadata(_make_handler(REGISTRY.get(tool), None)).arg_model
    model.model_validate(args)
    if tool == "mandate_write":
        array = model.model_json_schema()["properties"]["questions"]["anyOf"][0]
        assert array["maxItems"] == 3
