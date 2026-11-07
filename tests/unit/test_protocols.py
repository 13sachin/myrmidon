"""Protocols: published pipelines, and the rule that they are never a rule.

An input set says what a value should be. A protocol says what sequence a published pipeline
runs. A mandate is one study's instance of a sequence, with its system, its decisions, its
ceiling and its approval. Each object does one job: an input set carries no ordered `stages`
list of its own, and a protocol states no values.

The load-bearing test in this file is the one that proves a researcher may ignore a
published pipeline entirely. "Classical equilibration before QM/MM" is one protocol, not a
law, and the harness has no standing to hold it as one.
"""

import pytest
import yaml
from tests.conftest import fixture_sets

from myrmidon.core.tool import REGISTRY
from myrmidon.knowledge.protocols import ProtocolStore

SHIPPED = ("mimic_qmmm_solvated_solute", "fm_reference_singlepoint")


def _wf(tmp_path, **over):
    body = {
        "id": "toy",
        "title": "a toy pipeline",
        "applicability": "testing",
        "given": ["structure"],
        "steps": [{"name": "a", "tool": "stage_prepare", "needs": ["structure"],
                   "produces": ["tpr"]}],
        "sources": [{"kind": "in_house", "text": "this test", "evidence": ["ev-1"]}],
        **over,
    }
    directory = tmp_path / "protocols"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{body['id']}.yaml").write_text(yaml.safe_dump(body))
    return ProtocolStore(path=directory, builtin_path=tmp_path / "absent")


# --- the shipped store ------------------------------------------------------------


@pytest.mark.parametrize("name", SHIPPED)
def test_each_stage_of_a_shipped_protocol_is_prepared_by_one_step(name):
    """A coupled stage's mdp and CPMD input are one stage_prepare call that cites the whole
    stage, so a rule between the two files reads one step, and no step names the same call
    twice."""
    protocol = ProtocolStore.builtin().get(name)
    prepared = [s.cited() for s in protocol.steps if s.tool == "stage_prepare"]
    assert sorted(prepared) == sorted({(s.stage.name, "") for s in protocol.steps if s.stage})


# --- what the schema refuses -------------------------------------------------------


def test_a_protocol_without_a_source_is_refused(tmp_path):
    """The input set store's rule, for the same reason: a sequence nobody published and
    nobody can follow back is one person's habit, not knowledge."""
    directory = tmp_path / "protocols"
    directory.mkdir(parents=True)
    (directory / "bare.yaml").write_text(yaml.safe_dump({
        "id": "bare", "title": "t", "applicability": "a",
        "steps": [{"name": "a", "tool": "stage_prepare"}],
    }))
    with pytest.raises(ValueError):
        ProtocolStore(path=directory, builtin_path=tmp_path / "absent").get("bare")


# --- a study's relation to a protocol -----------------------------------------------


def test_an_adapted_sequence_still_has_to_hold_together():
    """Free to depart, not free to be broken.

    Dropping the heating step while keeping the production step that restarts from it is
    not a new pipeline, it is a sequence that reads a file nothing writes. The check names
    the role, not the convention; dropping a step nothing later reads passes."""
    from myrmidon.knowledge.protocols import unsatisfied_roles

    protocol = ProtocolStore.builtin().get("mimic_qmmm_solvated_solute")
    assert protocol.unsatisfied() == {}
    broken = unsatisfied_roles(protocol.given,
                               [s for s in protocol.steps if s.name != "heating"])
    assert "cpmd_restart_heated" in broken.get("production", [])
    last = protocol.steps[-1].name
    assert unsatisfied_roles(protocol.given,
                             [s for s in protocol.steps if s.name != last]) == {}


def test_the_shipped_pack_says_which_systems_each_recipe_is_for():
    """A recipe distilled from one tutorial system must not be sent every system under
    that system's name. Each recipe states the class of system it holds for; a value that
    is one system's fact is derived, not shipped; an id a study cited under an alias still
    resolves, and its version says whether the values changed."""
    from myrmidon.knowledge.inputsets import resolve

    sets, protocols = fixture_sets(), ProtocolStore.builtin()
    unclassed = [i.id for i in [*sets.all(), *protocols.all()] if not i.system_class]
    assert not unclassed
    for item, store in [(i, sets) for i in sets.all()] + [(p, protocols)
                                                           for p in protocols.all()]:
        for old in item.aliases:
            assert store.get(old).id == item.id
    box = resolve(sets.get("box_qmmm_solute"))
    assert "distance_nm" in box.missing and "distance_nm" not in box.values


def test_a_protocol_wires_roles_and_states_no_values(tmp_path):
    """A protocol may say which role feeds which parameter; a value is an input set's."""
    step = {"name": "a", "tool": "stage_prepare", "needs": ["structure"], "produces": ["tpr"]}
    ok = _wf(tmp_path, steps=[dict(step, values={"structure": "structure"})])
    assert ok.get("toy").steps[0].values == {"structure": "structure"}
    with pytest.raises(ValueError, match="not roles it needs"):
        _wf(tmp_path / "x", steps=[dict(step, values={"maxwarn": "3"})]).get("toy")


def test_a_protocol_is_priced_on_the_machine_that_looks_it_up(ctx):
    """A protocol holds no core-hour: its stages are priced on this machine's billing, so a
    node billed at twice the rate prices them at twice the core-hours, in a lookup as in a
    preview."""
    from myrmidon.config import PartitionCompute

    def priced(billed: float):
        ctx.config.compute.partitions = {"p": PartitionCompute(
            cores_per_node=128, billed_per_node_h=billed, node_exclusive=True)}
        ctx.config.compute.default_partition = "p"
        found = REGISTRY.call("protocol_lookup", ctx, {"query": "mimic_qmmm_solvated_solute"})
        preview = REGISTRY.call("protocol_preview", ctx,
                                {"protocol": "mimic_qmmm_solvated_solute"})
        assert preview.ok, preview.summary
        assert found.protocols[0].total_core_h == preview.estimate_core_h > 0
        # the classical stages' step is a force field's, which no system names here
        assert [u.split()[0] for u in found.protocols[0].unpriced] == ["equilibrate"]
        return preview.estimate_core_h

    assert priced(256.0) == pytest.approx(2 * priced(128.0), rel=1e-3)
    assert not REGISTRY.call("protocol_preview", ctx, {"protocol": "no_such"}).ok


def test_a_protocols_stages_are_built_on_the_system_and_priced_for_its_length(
        ws_ctx, solvated):
    """Given the built system, each prepared stage is rendered and built by the engine's
    preprocessor in a throwaway study — the study's ledger records none of it — and a run
    is priced for the length the study sets."""
    before = len(list(ws_ctx.workspace.ledger.events()))
    obs = REGISTRY.call("protocol_preview", ws_ctx, {
        "protocol": "mimic_qmmm_solvated_solute", **solvated,
        "overrides": {"heat_prep": {"nsteps": 25000}}})
    assert obs.ok, obs.text
    assert "minimise_prep" in obs.text and "renders and builds" in obs.text
    assert "does NOT" not in obs.text
    heat = REGISTRY.call("protocol_preview", ws_ctx, {
        "protocol": "mimic_qmmm_solvated_solute", **solvated})
    assert obs.estimate_core_h < heat.estimate_core_h  # a shorter heating run costs less
    assert len(list(ws_ctx.workspace.ledger.events())) - before == 2  # the two previews
    assert not list((ws_ctx.workspace.state_dir / "preview").iterdir())
