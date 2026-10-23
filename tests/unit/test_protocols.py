"""Protocols: published pipelines, and the rule that they are never a rule.

An input set says what a value should be. A protocol says what sequence a published pipeline
runs. A plan is one study's instance of a sequence, with its system, its costs and its
approval. Each object does one job: an input set carries no ordered `stages` list of its
own, and a protocol states no values.

The load-bearing test in this file is the one that proves a researcher may ignore a
published pipeline entirely. "Classical equilibration before QM/MM" is one protocol, not a
law, and the harness has no standing to hold it as one.
"""

import pytest
import yaml

from myrmidon.core.tool import REGISTRY
from myrmidon.knowledge.protocols import ProtocolStore

SHIPPED = ("mimic_qmmm_solvated_solute", "fm_reference_singlepoint")


def _wf(tmp_path, **over):
    body = {
        "id": "toy",
        "title": "a toy pipeline",
        "applicability": "testing",
        "given": ["structure"],
        "steps": [{"name": "a", "tool": "gmx_grompp", "needs": ["structure"],
                   "produces": ["tpr"]}],
        "sources": [{"kind": "in_house", "text": "this test", "evidence": ["ev-1"]}],
        **over,
    }
    directory = tmp_path / "protocols"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{body['id']}.yaml").write_text(yaml.safe_dump(body))
    return ProtocolStore(path=directory, builtin_path=tmp_path / "absent")


# --- the shipped store ------------------------------------------------------------


def test_the_acetone_protocol_prices_itself_from_what_the_machine_booked():
    """A cost column of dashes is decoration; this one carries measured numbers, and
    every estimate names its basis — a price with no source is a number like any other."""
    protocol = ProtocolStore.builtin().get("mimic_qmmm_solvated_solute")
    assert all(s.estimate_core_h is not None for s in protocol.steps)
    assert all(s.estimate_basis for s in protocol.steps)
    assert sum(s.estimate_core_h for s in protocol.steps) == 240.0


# --- what the schema refuses -------------------------------------------------------


def test_a_protocol_without_a_source_is_refused(tmp_path):
    """The input set store's rule, for the same reason: a sequence nobody published and
    nobody can follow back is one person's habit, not knowledge."""
    directory = tmp_path / "protocols"
    directory.mkdir(parents=True)
    (directory / "bare.yaml").write_text(yaml.safe_dump({
        "id": "bare", "title": "t", "applicability": "a",
        "steps": [{"name": "a", "tool": "gmx_grompp"}],
    }))
    with pytest.raises(ValueError):
        ProtocolStore(path=directory, builtin_path=tmp_path / "absent").get("bare")


# --- a plan's relation to a protocol ------------------------------------------------


def test_adapting_a_published_protocol_is_free_and_visible(ws_ctx):
    """The point of the whole split.

    A researcher who starts from a pipeline and changes it is doing the work. It costs no
    acknowledgement — the finding is `info` — and it is recorded, so the study says what it
    departed from and where. If this ever becomes a warn, the harness has started asking
    researchers to justify doing research.
    """
    protocol = ProtocolStore.builtin().get("mimic_qmmm_solvated_solute")
    dropped = ("nve_test", "heating", "production")
    theirs = [s.model_dump(exclude_none=True) for s in protocol.steps
              if s.name not in dropped]
    theirs.append({"name": "longer_production", "tool": "mimic_run",
                   "needs": ["tpr_mimic", "cpmd_restart_annealed"],
                   "produces": ["trajectory_long"], "estimate_core_h": 200.0})

    obs = REGISTRY.call("plan_write", ws_ctx, {
        "question": "Anneal then sample straight away, for longer",
        "from_protocol": "mimic_qmmm_solvated_solute",
        "given": protocol.given,
        "steps": theirs,
        "acknowledge_warnings": ["COMPUTE_BUDGET"],
    })
    assert obs.ok, obs.summary
    adapted = next(w for w in obs.advisory if w.id == "PLAN_ADAPTS_PROTOCOL")
    assert adapted.severity == "info"
    assert "added longer_production" in adapted.message
    assert "left out heating, nve_test, production" in adapted.message
    assert obs.cites == f"protocol:mimic_qmmm_solvated_solute@v{protocol.version}"


def test_an_adapted_plan_still_has_to_hold_together(ws_ctx):
    """Free to depart, not free to be broken.

    Dropping the heating step while keeping the production step that restarts from it is
    not a new pipeline, it is a plan that reads a file nothing writes. The refusal names
    the role, not the convention.
    """
    protocol = ProtocolStore.builtin().get("mimic_qmmm_solvated_solute")
    refused = REGISTRY.call("plan_write", ws_ctx, {
        "question": "skip the heating and see",
        "from_protocol": "mimic_qmmm_solvated_solute",
        "given": protocol.given,
        "steps": [s.model_dump(exclude_none=True) for s in protocol.steps
                  if s.name != "heating"],
        "acknowledge_warnings": ["COMPUTE_BUDGET"],
    })
    assert not refused.ok
    finding = next(w for w in refused.blocking if w.id == "PLAN_STEP_INPUT_UNSATISFIED")
    assert "production" in finding.message and "cpmd_restart_heated" in finding.message



def test_the_shipped_pack_says_which_systems_each_recipe_is_for():
    """A recipe distilled from one tutorial system must not be sent every system under
    that system's name. Each recipe states the class of system it holds for; a value that
    is one system's fact is derived, not shipped; an id a study cited under an alias still
    resolves, and its version says whether the values changed."""
    from myrmidon.knowledge.inputsets import InputSetStore, resolve

    sets, protocols = InputSetStore.builtin(), ProtocolStore.builtin()
    unclassed = [i.id for i in [*sets.all(), *protocols.all()] if not i.system_class]
    assert not unclassed
    for item, store in [(i, sets) for i in sets.all()] + [(p, protocols)
                                                           for p in protocols.all()]:
        for old in item.aliases:
            assert store.get(old).id == item.id
    box = resolve(sets.get("box_qmmm_solute"))
    assert "distance_nm" in box.missing and "distance_nm" not in box.values


def test_the_qmmm_sequence_asks_the_researcher_for_its_chemistry(ws_ctx):
    """Written as a plan, the published QM/MM sequence cannot be approved until the
    researcher has said which atoms are quantum, their charge and spin, and how long each
    stage runs."""
    obs = REGISTRY.call("plan_write", ws_ctx, {
        "question": "staged QM/MM", "from_protocol": "mimic_qmmm_solvated_solute"})
    assert obs.ok, obs.summary
    asked = {u.key for u in obs.unresolved if u.kind == "researcher"}
    assert {"selections", "total_charge", "multiplicity", "nsteps", "&CPMD/MAXSTEP"} <= asked
    # the charge is the system's, asked once in each stage's mimic_prepare — never again in
    # a stage's method
    assert "&SYSTEM/CHARGE" not in asked


def test_a_protocol_wires_roles_and_states_no_values(tmp_path):
    """A protocol may say which role feeds which parameter; a value is an input set's."""
    step = {"name": "a", "tool": "gmx_grompp", "needs": ["structure"], "produces": ["tpr"]}
    ok = _wf(tmp_path, steps=[dict(step, values={"structure": "structure"})])
    assert ok.get("toy").steps[0].values == {"structure": "structure"}
    with pytest.raises(ValueError, match="not roles it needs"):
        _wf(tmp_path / "x", steps=[dict(step, values={"maxwarn": "3"})]).get("toy")
