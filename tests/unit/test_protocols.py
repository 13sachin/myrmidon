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
        "stages": [{"name": "a", "kind": "minimize"}],
        "sources": [{"kind": "in_house", "text": "this test", "evidence": ["ev-1"]}],
        **over,
    }
    directory = tmp_path / "protocols"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{body['id']}.yaml").write_text(yaml.safe_dump(body))
    return ProtocolStore(path=directory, builtin_path=tmp_path / "absent")


# --- the shipped store ------------------------------------------------------------


def test_a_mandate_is_written_from_a_protocol(ws_ctx, solvated):
    """A protocol is what a mandate is instantiated from: named alone, its stages, their
    deliverables and its Hamiltonian are the mandate's, and the researcher's edits are the
    mandate's own."""
    protocol = ProtocolStore.builtin().get("mimic_qmmm_solvated_solute")
    written = REGISTRY.call("mandate_write", ws_ctx, {
        "question": "acetone under QM/MM", "ceiling_core_h": 5000.0,
        "protocol": protocol.id,
        "system_files": [{"path": solvated["structure"], "kind": "structure"},
                         {"path": solvated["topology"], "kind": "topology"}]})
    assert written.ok, written.summary
    from myrmidon.core.mandate import read_mandate

    held = read_mandate(ws_ctx.workspace)
    assert held.method.protocol == protocol.ref
    assert [s.name for s in held.method.stages] == [s.name for s in protocol.stages]
    assert [d.name for d in held.deliverables] == [d.name for d in protocol.deliverables]
    assert held.system.hamiltonian.qm.method == protocol.hamiltonian.qm.method
    # a protocol states no charge: the region's is the topology's, proposed with its source
    assert held.method.stage("production").qm_region.charge == 0
    assert "production.qm_region.charge" in {k for p in held.decisions.proposed for k in p.locks}


# --- what the schema refuses -------------------------------------------------------


def test_a_protocol_without_a_source_is_refused(tmp_path):
    """The input set store's rule, for the same reason: a sequence nobody published and
    nobody can follow back is one person's habit, not knowledge."""
    directory = tmp_path / "protocols"
    directory.mkdir(parents=True)
    (directory / "bare.yaml").write_text(yaml.safe_dump({
        "id": "bare", "title": "t", "applicability": "a",
        "stages": [{"name": "a", "kind": "minimize"}],
    }))
    with pytest.raises(ValueError):
        ProtocolStore(path=directory, builtin_path=tmp_path / "absent").get("bare")


# --- a study's relation to a protocol -----------------------------------------------


# (what a protocol states beside its stage `a`, whether it holds together): free to depart,
# not free to be broken
SEQUENCES = [
    ({"stages": [{"name": "a", "kind": "minimize"},
                 {"name": "b", "kind": "dynamics", "ensemble": "NVE", "steps": 10,
                  "starts_from": "a"}]}, True),
    ({"stages": [{"name": "a", "kind": "minimize", "starts_from": "b"},
                 {"name": "b", "kind": "dynamics", "ensemble": "NVE", "steps": 10}]}, False),
    ({"invariants": [{"id": "X", "equal": ["$a.dt", "$c.dt"], "message": "m"}]}, False),
    ({"deliverables": [{"name": "d", "property": "rmsd", "over": "c"}]}, False),
]


@pytest.mark.parametrize("over,holds", SEQUENCES)
def test_an_adapted_sequence_still_has_to_hold_together(tmp_path, over, holds):
    """A stage starts from an earlier one; a rule and a deliverable name its stages."""
    store = _wf(tmp_path, **over)
    if holds:
        assert store.get("toy").stages
    else:
        with pytest.raises(ValueError):
            store.get("toy")


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
        "overrides": {"equilibrate": {"nsteps": 25000}}})
    assert obs.ok, obs.text
    assert "minimise" in obs.text and "renders and builds" in obs.text
    assert "does NOT" not in obs.text
    heat = REGISTRY.call("protocol_preview", ws_ctx, {
        "protocol": "mimic_qmmm_solvated_solute", **solvated})
    assert obs.estimate_core_h < heat.estimate_core_h  # a shorter heating run costs less
    assert len(list(ws_ctx.workspace.ledger.events())) - before == 2  # the two previews
    assert not list((ws_ctx.workspace.state_dir / "preview").iterdir())
