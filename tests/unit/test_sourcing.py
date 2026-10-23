"""Where a parameter file's values came from, and what the gates say about it.

A run-parameter file reaching an engine was either rendered from an input set, supplied
by the user, or typed by whoever was driving. The ledger knows which, and the record
says so.
"""

import json
from pathlib import Path

import pytest

from myrmidon.core.sourcing import parameter_source
from myrmidon.core.tool import REGISTRY

MDP = (
    "integrator = md\ndt = 0.002\nnsteps = 5000\nconstraints = h-bonds\ncoulombtype = PME\n"
    "tcoupl = v-rescale\ntc-grps = System\ntau_t = 0.1\nref_t = 300\ngen_vel = yes\n"
    "gen_temp = 300\ngen_seed = 42\n"
)


ACETONE_GRO = Path(__file__).parents[1] / "fixtures" / "acetone" / "acetone_stageA_entry.gro"

def _inputs(ws_ctx):
    """conf.gro and topol.top planted as workspace inputs, as a user would."""
    stage = ws_ctx.workspace.stage("input")
    (stage / "conf.gro").write_text("MOCK\n    1\nX\n 1 1 1\n")
    (stage / "topol.top").write_text("; mock topology\n")
    return (
        ws_ctx.workspace.relpath(stage / "conf.gro"),
        ws_ctx.workspace.relpath(stage / "topol.top"),
    )


def _select_qm(ws_ctx, template=None):
    ws = ws_ctx.workspace
    stage = ws.stage("prep")
    for name in ("acetone_stageA_entry.top", "acetone.itp"):  # prepqm sums its charges
        (stage / name.replace("acetone_stageA_entry", "topol")).write_text(
            (Path(__file__).parents[1] / "fixtures" / "acetone" / name).read_text())
    # a real acetone: the mock prepqm selects from the coordinates it is given
    (stage / "conf.gro").write_text(ACETONE_GRO.read_text())
    call = {
        "topology": ws.relpath(stage / "topol.top"),
        "coords": ws.relpath(stage / "conf.gro"),
        "selections": ["add resname is ACT"],
        "padding_nm": 0.4,
    }
    if template is not None:
        call["template"] = template
    return REGISTRY.call("mimic_select_qm", ws_ctx, call)


def _planted(ws_ctx):
    """A file no tool produced: the user's own."""
    gro, _ = _inputs(ws_ctx)
    return gro


def _written(ws_ctx):
    """Typed in the session by whoever was driving."""
    assert REGISTRY.call("file_write", ws_ctx, {"path": "01_input/nvt.mdp", "content": MDP}).ok
    return "01_input/nvt.mdp"


def _rendered(ws_ctx):
    obs = REGISTRY.call("inputset_apply", ws_ctx, {
        "inputset": "em_solvated_biomolecule", "output": "01_input/em.mdp",
        "overrides": {"nsteps": 500}, "acknowledge_warnings": ["INPUTSET_VALUE_OVERRIDDEN"]})
    assert obs.ok, obs.summary
    return "01_input/em.mdp"


def _foreign_sidecar(ws_ctx):
    """mimic_prepare writes a sidecar too; only inputset_apply's counts as a source."""
    stage = ws_ctx.workspace.stage("input")
    (stage / "cpmd.inp").write_text("&CPMD\n&END\n")
    (stage / "cpmd.inp.myrmidon.json").write_text(
        json.dumps({"charge": 0, "multiplicity": 1, "confirmed_by": "mimic_prepare"}))
    return "01_input/cpmd.inp"


def _derived(ws_ctx):
    """prepqm builds cpmd.inp from a template, and the template carries the functional,
    the cutoff and the coupling — so the question follows the hop."""
    applied = REGISTRY.call("inputset_apply", ws_ctx, {
        "inputset": "fm_singlepoint_cpmd", "output": "01_input/template.inp",
        "overrides": {"&SYSTEM/CHARGE": "0.0"}})
    assert applied.ok, applied.summary
    obs = _select_qm(ws_ctx, template="01_input/template.inp")
    assert obs.ok, obs.summary
    return next(a.path for a in obs.artifacts if a.role == "cpmd_input")


# (how the file came to be, provenance, sourced, the input set it names, via)
SOURCES = [
    (_planted, "user_input", True, None, []),
    (_written, "authored", False, None, []),
    (_rendered, "inputset", True, "em_solvated_biomolecule", []),
    (_foreign_sidecar, "user_input", True, None, []),
    (_derived, "derived", True, "fm_singlepoint_cpmd", ["01_input/template.inp"]),
]


@pytest.mark.parametrize("make,provenance,sourced,inputset,via", SOURCES,
                         ids=[s[0].__name__.strip("_") for s in SOURCES])
def test_every_parameter_file_says_where_its_values_came_from(ws_ctx, make, provenance,
                                                               sourced, inputset, via):
    origin = parameter_source(ws_ctx.workspace, make(ws_ctx))
    assert (origin.provenance, origin.sourced, origin.inputset) == (provenance, sourced,
                                                                    inputset)
    assert list(origin.via or []) == via


def _grompp_on(make):
    def run(ws_ctx):
        gro, top = _inputs(ws_ctx)
        mdp = make(ws_ctx)
        return REGISTRY.call("gmx_grompp", ws_ctx, {
            "mdp": mdp, "structure": gro, "topology": top, "stage": "run"})
    run.__name__ = f"grompp{make.__name__}"
    return run


def _nvt_rendered(ws_ctx):
    obs = REGISTRY.call("inputset_apply", ws_ctx, {
        "inputset": "nvt_equilibration_protein", "output": "01_input/nvt.mdp",
        "overrides": {"nsteps": 5000}})
    assert obs.ok, obs.summary
    return "01_input/nvt.mdp"


def _mimic_prepare_unsourced(ws_ctx):
    obs = _select_qm(ws_ctx)  # no template: prepqm's own defaults, nothing recorded
    assert obs.ok, obs.summary
    generated = next(a.path for a in obs.artifacts if a.role == "cpmd_input")
    return REGISTRY.call("mimic_prepare", ws_ctx,
                         {"cpmd_input": generated, "total_charge": 0, "multiplicity": 1,
                          "acknowledge_warnings": ["QM_CHARGE_MULTIPLICITY_UNCONFIRMED"]})


# Advisory by design: the store has to cover the common recipes before a warning here
# would be anything but reflexively acknowledged. (gate call, the advisory it carries)
GATES = [
    (_grompp_on(_written), "PARAM_UNSOURCED"),
    (_grompp_on(_nvt_rendered), "PARAM_SOURCED"),
    (_mimic_prepare_unsourced, "PARAM_UNSOURCED"),
]


@pytest.mark.parametrize("call,advisory", GATES, ids=[g[0].__name__.strip("_") for g in GATES])
def test_the_gate_states_the_source_and_does_not_refuse(ws_ctx, call, advisory):
    obs = call(ws_ctx)
    assert obs.ok, obs.summary
    assert "parameter_source" in obs.checked
    ids = {w.id for w in obs.advisory}
    assert advisory in ids and ({"PARAM_SOURCED", "PARAM_UNSOURCED"} - {advisory}).isdisjoint(ids)


