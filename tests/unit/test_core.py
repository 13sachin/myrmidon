"""Registry invariants and the gate/provenance machinery."""

import json
from typing import ClassVar

import pytest
from pydantic import BaseModel, Field

from myrmidon.core.observation import Observation
from myrmidon.core.tool import REGISTRY, ToolRegistry, ToolSpec
from myrmidon.core.validation import ValidationWarning


def _load_all_tools():
    import myrmidon.basic_tools as basic

    basic.register_all()


def test_registry_invariants():
    """Every registered tool: Observation return type, JSON-serializable schema."""
    _load_all_tools()
    specs = REGISTRY.specs()
    assert len(specs) >= 15
    for spec in specs:
        assert issubclass(spec.returns, Observation), spec.name
        schema = spec.params_model.model_json_schema()
        json.dumps(schema)  # must round-trip
        assert spec.description, spec.name
        if spec.expensive:
            assert "acknowledge_warnings" in spec.params_model.model_fields, spec.name


class _DummyParams(BaseModel):
    value: int = 1
    acknowledge_warnings: list[str] = Field(default_factory=list)


def _make_registry_with_gated_tool(warn_severity="warn"):
    reg = ToolRegistry()

    def gate(ctx, params):
        return (
            [ValidationWarning(id="TEST_WARNING", severity=warn_severity, message="test")],
            ["test_check"],
        )

    def fn(ctx, params):
        return Observation(ok=True, summary=f"ran with {params.value}")

    reg.register(
        ToolSpec(
            name="dummy_gated",
            description="test tool",
            params_model=_DummyParams,
            returns=Observation,
            expensive=True,
            needs_workspace=False,
            gate=gate,
        ),
        fn,
    )
    return reg


# (finding severity, ids acknowledged in the call, the call runs, where the finding is)
GATE = [
    ("warn", [], False, "unacknowledged"),
    ("warn", ["TEST_WARNING"], True, None),
    ("block", ["TEST_WARNING"], False, "blocking"),
]


@pytest.mark.parametrize("severity,acks,runs,held_in", GATE)
def test_a_warning_runs_only_acknowledged_and_a_block_never(ctx, severity, acks, runs,
                                                             held_in):
    reg = _make_registry_with_gated_tool(severity)
    obs = reg.call("dummy_gated", ctx, {"value": 2, "acknowledge_warnings": acks})
    assert obs.ok is runs
    if held_in:
        assert obs.failure == "gate_refused"
        assert [w.id for w in getattr(obs, held_in)] == ["TEST_WARNING"]


class _NoAck(BaseModel):
    value: int = 1


class _BooksUnnamed(BaseModel):
    """Books resources without naming the fields a plan would price it from."""
    walltime_min: int = 60
    acknowledge_warnings: list[str] = Field(default_factory=list)

    def resources(self):
        return None


class _AsksForNothing(BaseModel):
    """Declares a researcher's field it does not have: a plan could never collect it."""
    RESEARCHER_FIELDS: ClassVar[tuple[str, ...]] = ("charge",)
    value: int = 1


# A tool the registry refuses to hold: (params, expensive?, what the refusal names)
UNREGISTRABLE = [
    (_AsksForNothing, False, "RESEARCHER_FIELDS"),
    (_NoAck, True, "acknowledge_warnings"),
    # a plan would price it as free
    (_BooksUnnamed, True, "RESOURCE_FIELDS"),
]


@pytest.mark.parametrize("params,expensive,names", UNREGISTRABLE)
def test_a_tool_that_breaks_a_registry_rule_is_not_registered(params, expensive, names):
    with pytest.raises(Exception, match=names):
        ToolRegistry().register(
            ToolSpec(name="bad", description="x", params_model=params,
                     returns=Observation, expensive=expensive),
            lambda ctx, p: Observation(ok=True, summary="x"),
        )


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
    refused = REGISTRY.call("inputset_apply", ws_ctx, {
        "inputset": "fm_classical_rerun", "output": "01_input/rerun.mdp",
        "overrides": {"integrator": "sd"},
    })
    assert not refused.ok and refused.unacknowledged and not refused.warnings  # once each
    event = list(ws_ctx.workspace.ledger.events())[-1]
    assert event.kind == "gate_refused"
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
            returns=Observation,
            needs_workspace=False,
        ),
        exploding,
    )
    obs = reg.call("exploding", ctx, {})
    assert not obs.ok and obs.failure == "raised"
    assert "dependency blew up" in obs.log_tail  # the traceback, for diagnosis

