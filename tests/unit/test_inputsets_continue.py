"""InputSet store (schema v2, three layers), inputset_lookup, and checkpoint_continue."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from myrmidon.core.tool import REGISTRY
from myrmidon.engines.gromacs.mdp import parse_mdp, render_mdp, validate_mdp
from myrmidon.knowledge.inputsets import InputSetStore, Source, resolve


def test_builtin_protocols_have_resolvable_sources():
    """A citation nobody can follow is not a source."""
    inputsets = InputSetStore.builtin().all()
    assert len(inputsets) >= 2
    for inputset in inputsets:
        assert inputset.sources, inputset.id
        for source in inputset.sources:
            assert source.locator or source.sha256 or source.evidence, inputset.id


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


def test_shipped_protocols_render_to_valid_mdp():
    """Every shipped inputset must pass the harness's own mdp audit.

    A store nobody checks against the validator is how a default becomes a fiction.
    Warn-level findings are allowed: an input set that deliberately does not set the
    run length should say so by leaving nsteps out, not by inventing one.
    """
    # a MiMiC mdp needs an index group that only exists once prepqm has run
    allowed = {"MDP_MIMIC_NO_QMMM_GRPS"}
    audited = 0
    for inputset in InputSetStore.builtin().all():
        if inputset.renders not in (None, "mdp"):
            continue
        resolution = resolve(inputset)
        if not resolution.values:
            continue
        text = render_mdp(resolution.values, resolution.origin)
        warnings, checked = validate_mdp(parse_mdp(text))
        assert checked, inputset.id
        blocking = {w.id for w in warnings if w.severity == "block"}
        assert not (blocking - allowed), f"{inputset.id}: {sorted(blocking)}"
        audited += 1
    assert audited >= 6, f"only {audited} mdp recipes audited — the loop stopped covering them"


def test_every_rendered_value_names_where_it_came_from():
    inputset = InputSetStore.builtin().get("nvt_equilibration_protein")
    r = resolve(inputset, overrides={"ref_t": 310, "nsteps": 50000})
    assert r.origin["dt"] == "inputset:nvt_equilibration_protein@v2"
    assert r.origin["ref_t"] == r.origin["nsteps"] == "user"
    assert r.overrides == {"ref_t": {"inputset": 300, "user": 310}}
    assert r.additions == ["nsteps"]  # a key the inputset never mentions
    assert r.complete
    # the file says so line by line, and the comments do not change what GROMACS reads
    text = render_mdp(r.values, r.origin)
    lines = {ln.split("=")[0].strip(): ln for ln in text.splitlines() if "=" in ln}
    assert lines["dt"].endswith("; inputset:nvt_equilibration_protein@v2")
    assert lines["nsteps"].endswith("; user")
    assert parse_mdp(text)["dt"] == "0.002"


def test_inputset_apply_lands_in_the_ledger_with_the_artifact_hash(ws_ctx):
    before = len(list(ws_ctx.workspace.ledger.events()))
    obs = REGISTRY.call(
        "inputset_apply",
        ws_ctx,
        {"inputset": "em_solvated_biomolecule", "output": "02_run/em.mdp"},
    )
    assert obs.ok, obs.summary
    events = list(ws_ctx.workspace.ledger.events())
    assert len(events) == before + 1
    event = events[-1]
    assert event.tool == "inputset_apply"
    assert event.params["inputset"] == "em_solvated_biomolecule"
    assert any(a.role == "mdp" and a.sha256 for a in event.outputs)


AXIS_NSTEPS = (
    "question: How far does minimisation have to run?\n"
    "steps:\n  - name: em\n    tool: gmx_grompp\n"
    "    cites: inputset:em_solvated_biomolecule@v1\n"
    "arms:\n  varies: nsteps\n  over: [1000, 20000]\n"
    "  because: to see whether the extra steps change the final energy\n"
)

# (the study's plan, the overrides, gated, what the result records). Changing a value the
# input set states needs an acknowledgement — unless it is the axis the study declared it
# varies: that is the experiment, not a deviation, and it is still recorded. The study,
# not the input set's author, says which value that is. Adding a key is free.
OVERRIDES = [
    (None, {"rvdw": 1.4}, True,
     dict(overrides={"rvdw": {"inputset": 1.0, "user": 1.4}}, on_axis=[], additions=[])),
    (AXIS_NSTEPS, {"nsteps": 20000}, False,
     dict(overrides={"nsteps": {"inputset": 1000, "user": 20000}}, on_axis=["nsteps"],
          additions=[])),
    (AXIS_NSTEPS, {"nsteps": 20000, "rvdw": 1.4}, True,
     dict(overrides={"nsteps": {"inputset": 1000, "user": 20000},
                     "rvdw": {"inputset": 1.0, "user": 1.4}}, on_axis=["nsteps"],
          additions=[])),
    (None, {"constraints": "h-bonds"}, False,
     dict(overrides={}, on_axis=[], additions=["constraints"])),
]


@pytest.mark.parametrize("plan,overrides,gated,recorded", OVERRIDES)
def test_a_changed_value_is_acknowledged_unless_it_is_the_studys_axis(ws_ctx, plan, overrides,
                                                                      gated, recorded):
    if plan:
        (Path(ws_ctx.workspace.root) / "plan.yaml").write_text(plan)
    call = {"inputset": "em_solvated_biomolecule", "output": "02_run/em.mdp",
            "overrides": overrides}
    if gated:
        refused = REGISTRY.call("inputset_apply", ws_ctx, call)
        assert not refused.ok
        assert [w.id for w in refused.unacknowledged] == ["INPUTSET_VALUE_OVERRIDDEN"]
        assert not (ws_ctx.workspace.root / "02_run/em.mdp").exists()
        call["acknowledge_warnings"] = ["INPUTSET_VALUE_OVERRIDDEN"]
    obs = REGISTRY.call("inputset_apply", ws_ctx, call)
    assert obs.ok, obs.summary
    assert (obs.overrides, obs.on_axis, obs.additions) == (
        recorded["overrides"], recorded["on_axis"], recorded["additions"])
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
    finding = next(w for w in refused.blocking if w.id == "INPUTSET_REQUIRED_PARAM_MISSING")
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


def test_minimization_recipe_comes_from_the_protocol_not_from_a_string():
    """The harness carries no hardcoded recipe in code: the driver renders its
    minimisation from the input set."""
    from myrmidon.engines.mimic import validate_driver

    assert not hasattr(validate_driver, "EM_MDP")
    text = validate_driver.em_mdp_text({"nsteps": 2000, "emtol": 100.0})
    assert "inputset: em_solvated_biomolecule v1" in text
    params = parse_mdp(text)
    assert params["integrator"] == "steep"
    assert params["nsteps"] == "2000"
    assert "nsteps        = 2000  ; user" in text


def test_changing_a_fit_setting_is_recorded_as_yours(ws_ctx, tmp_path):
    project = tmp_path / "inputsets"
    _write(project, "fm_bonded_only", parameters={
        "stride": {"value": [0, 1000, 5], "source": 0},
        "regularization_relative_fraction": {"value": 0.10,
                                             "rationale": "a starting point", "source": 0},
    })
    (Path(ws_ctx.workspace.root) / "plan.yaml").write_text(
        "question: Which regularisation fits best?\n"
        "steps:\n"
        "  - name: fit\n"
        "    tool: fm_fit\n"
        "    cites: inputset:fm_bonded_only@v1\n"
        "arms:\n"
        "  varies: regularization_relative_fraction\n"
        "  over: [0.10, 0.02]\n"
        "  because: the held-out force error is what decides it\n"
    )
    ws_ctx.inputsets = InputSetStore(path=project, site_path=tmp_path / "none",
                                     builtin_path=tmp_path / "absent")
    obs = REGISTRY.call("inputset_apply", ws_ctx, {
        "inputset": "fm_bonded_only", "output": "01_input/p02.dat",
        "overrides": {"regularization_relative_fraction": 0.02},
    })
    assert obs.ok, obs.summary
    text = (ws_ctx.workspace.root / "01_input" / "p02.dat").read_text()
    assert "regularization_relative_fraction: 0.02  # user" in text
    assert "INPUTSET_VALUE_ON_DECLARED_AXIS" in {w.id for w in obs.advisory}
