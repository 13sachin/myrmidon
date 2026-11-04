"""InputSet store (schema v2, three layers), inputset_lookup, and checkpoint_continue."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from tests.conftest import fixture_sets
from tests.unit.findings_catalog import blocking

from myrmidon.core.tool import REGISTRY
from myrmidon.engines.gromacs.mdp import parse_mdp, render_mdp
from myrmidon.knowledge.catalog import load, shipped_versions
from myrmidon.knowledge.inputsets import InputSetStore, Source, resolve

ACETONE = Path(__file__).parents[1] / "fixtures" / "acetone"


def test_builtin_protocols_have_resolvable_sources():
    """A citation nobody can follow is not a source: the shipped stage templates and
    methods, and any input set, each cite what a reader can follow."""
    from myrmidon.knowledge.stage import MethodStore, TemplateStore

    shipped = [*TemplateStore.builtin().all(), *MethodStore.builtin().all(),
               *fixture_sets().all()]
    assert len(shipped) >= 10
    for item in shipped:
        assert item.sources, item.id
        for source in item.sources:
            assert source.locator or source.sha256 or source.evidence, item.id
    # the key catalogs too: each loads, and says what engine output it is
    for engine in ("gromacs", "cpmd"):
        for version in shipped_versions(engine):
            catalog = load(engine, version)
            assert catalog.sources and catalog.keys, (engine, version)


def _write(directory: Path, protocol_id: str, **extra) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "id": protocol_id,
        "title": f"{protocol_id} title",
        "applicability": "testing",
        "parameters": {"dt": 0.002},
        "sources": [{"kind": "in_house", "text": "this test", "evidence": ["ev-1"]}],
        **extra,
    }
    path = directory / f"{protocol_id}.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


def test_required_from_user_is_listed_not_supplied(tmp_path):
    project = tmp_path / "proj"
    _write(
        project,
        "flat",
        renders="cpmd_inp",
        parameters={
            "total_charge": {
                "required_from_user": True,
                "derivable": "sum of topology charges over the selection",
            },
            "&SYSTEM/CUTOFF": {"value": 70.0, "source": 0},
        },
    )
    inputset = InputSetStore(path=project, builtin_path=tmp_path / "absent").get("flat")
    assert inputset.values() == {"&SYSTEM/CUTOFF": 70.0}
    # chemistry is never a default: it is listed, not supplied
    assert "total_charge" in inputset.unsupplied()
    assert "total_charge" not in inputset.values()


def test_a_value_without_a_source_someone_can_follow_is_refused(tmp_path):
    """A citation nobody can follow is not a source, and neither is an
    index into a list that does not have it."""
    with pytest.raises(ValueError, match="no DOI or URL"):
        Source.from_citation("Somebody said so once, in a corridor")
    project = tmp_path / "proj"
    _write(project, "bad", parameters={"dt": {"value": 0.002, "source": 7}})
    with pytest.raises(ValueError, match="points at source 7"):
        InputSetStore(path=project, builtin_path=tmp_path / "absent").all()


def _acetone_facts() -> dict:
    """What an engine input reads from a real system: acetone in SPC water (amber03)."""
    from myrmidon.engines.gromacs.system import (
        coupling_groups,
        restraint_defines,
        starting_state,
    )

    top = ACETONE / "acetone_equilibrated.top"
    facts = [coupling_groups(top, []), restraint_defines(top, []),
             *starting_state(ACETONE / "acetone_equilibrated.gro")]
    return {f.key: f for f in facts}


def test_every_rendered_value_names_where_it_came_from():
    store = fixture_sets()
    inputset, amber = store.get("gromacs.nvt"), store.get("forcefield.amber")
    r = resolve(inputset, overrides={"ref_t": 310, "nsteps": 50000, "nstcalcenergy": 100},
                method=amber, facts=_acetone_facts())
    assert r.origin["dt"] == "inputset:forcefield.amber@v1"  # the method's
    assert r.origin["integrator"] == "inputset:gromacs.nvt@v1"  # the purpose's
    assert r.origin["tc-grps"].startswith("system:coupling_groups")  # the system's
    assert r.origin["ref_t"] == r.origin["nsteps"] == "user"
    assert r.owner["dt"] == "method" and r.owner["tc-grps"] == "system"
    assert r.overrides == {"ref_t": {"inputset": 300, "user": 310}}
    assert r.additions == ["nstcalcenergy"]  # a key no input set mentions
    assert r.complete
    # an override is the set's key however grompp would read it spelled
    assert resolve(inputset, overrides={"Ref-T": 310}, method=amber,
                   facts=_acetone_facts()).overrides == r.overrides
    # the file says so line by line, and the comments do not change what GROMACS reads
    text = render_mdp(r.values, r.origin)
    lines = {ln.split("=")[0].strip(): ln for ln in text.splitlines() if "=" in ln}
    assert lines["dt"].endswith("; inputset:forcefield.amber@v1")
    assert lines["nsteps"].endswith("; user")
    assert parse_mdp(text)["dt"] == "0.002"


def test_inputset_apply_lands_in_the_ledger_with_the_artifact_hash(ws_ctx):
    before = len(list(ws_ctx.workspace.ledger.events()))
    obs = REGISTRY.call(
        "inputset_apply",
        ws_ctx,
        {"inputset": "gromacs.minimise", "method": "forcefield.amber", "output": "02_run/em.mdp"},
    )
    assert obs.ok, obs.summary
    events = list(ws_ctx.workspace.ledger.events())
    assert len(events) == before + 1
    event = events[-1]
    assert event.tool == "inputset_apply"
    assert event.params["inputset"] == "gromacs.minimise"
    assert any(a.role == "mdp" and a.sha256 for a in event.outputs)


# (the overrides, a deviation is recorded, what the result records). Changing a value the
# input set states is a deviation on record, with the why the journal holds; adding a key is
# free.
OVERRIDES = [
    ({"rvdw": 1.4}, True, dict(overrides={"rvdw": {"inputset": 1.0, "user": 1.4}},
                               additions=[])),
    ({"nsteps": 20000, "rvdw": 1.4}, True,
     dict(overrides={"nsteps": {"inputset": 50000, "user": 20000},
                     "rvdw": {"inputset": 1.0, "user": 1.4}}, additions=[])),
    ({"nstcalcenergy": 100}, False, dict(overrides={}, additions=["nstcalcenergy"])),
]


@pytest.mark.parametrize("overrides,deviates,recorded", OVERRIDES)
def test_a_changed_value_is_a_deviation_on_record(ws_ctx, overrides, deviates, recorded):
    call = {"inputset": "gromacs.minimise", "method": "forcefield.amber",
            "output": "02_run/em.mdp", "overrides": overrides}
    obs = REGISTRY.call("inputset_apply", ws_ctx, call)
    assert obs.ok, obs.summary
    assert ("INPUTSET_VALUE_OVERRIDDEN" in {w.id for w in obs.findings}) is deviates
    assert (obs.overrides, obs.additions) == (recorded["overrides"], recorded["additions"])
    for key in overrides:
        assert obs.origin[key] == "user"


def test_inputset_apply_blocks_when_the_protocol_refuses_to_supply_a_value(tmp_path, ws_ctx):
    project = tmp_path / "proj"
    _write(
        project,
        "needs_a_decision",
        parameters={
            "integrator": "steep",
            "ref_t": {
                "required_from_user": True,
                "derivable": "the temperature the experiment was done at",
            },
        },
    )
    ctx = replace(ws_ctx, inputsets=InputSetStore(path=project, builtin_path=tmp_path / "none"))
    refused = REGISTRY.call(
        "inputset_apply", ctx, {"inputset": "needs_a_decision", "output": "02_run/x.mdp"}
    )
    assert not refused.ok
    finding = next(w for w in blocking(refused) if w.id == "INPUTSET_REQUIRED_PARAM_MISSING")
    assert finding.assumption  # says what would be assumed instead

    obs = REGISTRY.call(
        "inputset_apply",
        ctx,
        {"inputset": "needs_a_decision", "output": "02_run/x.mdp", "overrides": {"ref_t": 300}},
    )
    assert obs.ok, obs.summary
    assert obs.origin["ref_t"] == "user"


def test_a_cpmd_parameter_key_must_name_its_section(tmp_path, ws_ctx):
    project = tmp_path / "proj"
    _write(project, "qm", renders="cpmd_inp", parameters={"CUTOFF": 70.0})
    ctx = replace(ws_ctx, inputsets=InputSetStore(path=project, builtin_path=tmp_path / "none"))
    obs = REGISTRY.call("inputset_apply", ctx, {"inputset": "qm", "output": "a.inp"})
    assert not obs.ok
    assert not (ws_ctx.workspace.root / "a.inp").exists()


def test_changing_a_fit_setting_is_recorded_as_yours(ws_ctx, tmp_path):
    project = tmp_path / "inputsets"
    _write(project, "fm_bonded_only", parameters={
        "stride": {"value": [0, 1000, 5], "source": 0},
        "regularization_relative_fraction": {"value": 0.10,
                                             "rationale": "a starting point", "source": 0},
    })
    ws_ctx.inputsets = InputSetStore(path=project, site_path=tmp_path / "none",
                                     builtin_path=tmp_path / "absent")
    obs = REGISTRY.call("inputset_apply", ws_ctx, {
        "inputset": "fm_bonded_only", "output": "01_input/p02.dat",
        "overrides": {"regularization_relative_fraction": 0.02},
    })
    assert obs.ok, obs.summary
    text = (ws_ctx.workspace.root / "01_input" / "p02.dat").read_text()
    assert "regularization_relative_fraction: 0.02  # user" in text
    assert "INPUTSET_VALUE_OVERRIDDEN" in {w.id for w in obs.findings}
