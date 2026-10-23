"""What the user changed in an engine file, and what that means.

The file is the interface. A researcher opens
`heat.mdp` in any editor and changes a number; nothing in the file is myrmidon's, so
the harness notices by diffing the file against the sidecar `inputset_apply` left.
A value on the experiment the study declared becomes theirs; any other value the
inputset states is a
finding to acknowledge; a key the input set never mentions is an addition.
"""

import re
from pathlib import Path

import pytest

from myrmidon.core.file_origin import (
    file_origin,
)
from myrmidon.core.tool import REGISTRY

HEAT = {"inputset": "npt_equilibration_solute_in_water",
        "output": "01_input/heat.mdp", "overrides": {"gen_seed": 20260913, "nsteps": 100000}}


#: The study says what it varies. Here it is the velocity seed — which is what a study
#: running replicas of one setup varies, and the reason the seed is the caller's to set.
#: The study declares its own axis and the gate reads that, rather than an input set's
#: author, who never saw the study, marking a key tunable.
PLAN = """
question: Do three seeds of this equilibration agree?
steps:
  - name: heat
    tool: gmx_grompp
    cites: inputset:npt_equilibration_solute_in_water@v2
arms:
  varies: gen_seed
  over: [20260913, 777, 31337]
  metric: mean temperature
  because: three replicas, to put an error bar on the equilibrated temperature
"""


@pytest.fixture
def rendered(ws_ctx):
    (Path(ws_ctx.workspace.root) / "plan.yaml").write_text(PLAN)
    obs = REGISTRY.call("inputset_apply", ws_ctx, HEAT)
    assert obs.ok, obs.summary
    return ws_ctx


def _edit(ws, pattern, replacement):
    path = ws.root / "01_input" / "heat.mdp"
    path.write_text(re.sub(pattern, replacement, path.read_text(), flags=re.M))


def test_a_rendered_file_is_all_the_protocols(rendered):
    origin = file_origin(rendered, "01_input/heat.mdp")
    assert origin.provenance == "inputset"
    assert origin.inputset == "npt_equilibration_solute_in_water"
    assert {k.status for k in origin.keys} == {"unchanged"}
    assert not origin.edited and origin.findings == []
    # except the one value the caller set, which the sidecar already knew was theirs
    seed = next(k for k in origin.keys if k.key == "gen_seed")
    assert seed.origin == "user" and seed.value == "20260913"
    assert len([k for k in origin.keys if k.origin.startswith("inputset:")]) > 20


def _append(ws, text):
    path = ws.root / "01_input" / "heat.mdp"
    path.write_text(path.read_text() + text)


# (the hand edit; key -> status it now has; finding -> severity). A value on the axis the
# study declared becomes the researcher's; any other stated value is a finding to
# acknowledge; an added key is an addition and a dropped one a decision; GROMACS spelling
# ('-' for '_', case) is not an edit at all.
EDITS = [
    ([(r"^gen_seed\s*=.*$", "gen_seed = 777")],
     {"gen_seed": "on_declared_axis"}, {"INPUTSET_VALUE_ON_DECLARED_AXIS": "info"}),
    ([(r"^integrator\s*=.*$", "integrator = steep")],
     {"integrator": "changed"}, {"INPUTSET_VALUE_FIXED": "warn"}),
    ([(r"^tcoupl\s*=.*$", ""), "\nfreezegrps = Protein\n"],
     {"tcoupl": "removed", "freezegrps": "added"},
     {"INPUTSET_VALUE_ADDED": "info", "INPUTSET_VALUE_REMOVED": "warn"}),
    ([(r"^DispCorr\s*=\s*EnerPres.*$", "dispcorr = EnerPres"),
      (r"^cutoff-scheme\s*=\s*Verlet.*$", "cutoff_scheme = Verlet")],
     {}, {}),
]


@pytest.mark.parametrize("edits,statuses,findings", EDITS)
def test_a_hand_edit_is_classified_by_whose_value_it_was(rendered, edits, statuses,
                                                         findings):
    for edit in edits:
        if isinstance(edit, str):
            _append(rendered.workspace, edit)
        else:
            _edit(rendered.workspace, *edit)
    origin = file_origin(rendered, "01_input/heat.mdp")
    assert {k.key: k.status for k in origin.keys if k.status != "unchanged"} == statuses
    assert {f.id: f.severity for f in origin.findings} == findings
    assert origin.edited is bool(statuses)


def test_a_cpmd_input_round_trips_too(ws_ctx):
    obs = REGISTRY.call("inputset_apply", ws_ctx, {
        "inputset": "fm_singlepoint_cpmd",
        "output": "01_input/sp.inp", "overrides": {"&SYSTEM/CHARGE": "0.0"}})
    assert obs.ok, obs.summary
    origin = file_origin(ws_ctx, "01_input/sp.inp")
    assert origin.keys and {k.status for k in origin.keys} == {"unchanged"}
    assert "&SYSTEM/CUTOFF" in {k.key for k in origin.keys}
    path = ws_ctx.workspace.root / "01_input" / "sp.inp"
    path.write_text(path.read_text().replace(" CUTOFF\n  70.", " CUTOFF\n  30."))
    edited = file_origin(ws_ctx, "01_input/sp.inp")
    cutoff = next(k for k in edited.keys if k.key == "&SYSTEM/CUTOFF")
    assert cutoff.status in ("changed", "on_declared_axis") and cutoff.value == "30."
    assert any(f.id.startswith("INPUTSET_VALUE_") for f in edited.findings)


# (the hand edit, the finding grompp raises, it refuses). grompp is the last gate before
# dynamics: an edited value arrives there, not after the run.
GATE = [
    ((r"^integrator\s*=.*$", "integrator = steep"), "INPUTSET_VALUE_FIXED", True),
    ((r"^gen_seed\s*=.*$", "gen_seed = 777"), "INPUTSET_VALUE_ON_DECLARED_AXIS", False),
]


@pytest.mark.parametrize("edit,finding,refuses", GATE)
def test_the_gate_reports_the_edit_before_the_run(rendered, edit, finding, refuses):
    ws = rendered.workspace
    stage = ws.stage("input")
    (stage / "conf.gro").write_text(
        "mock\n 3\n" + "    1ALA     CA    1   0.1   0.2   0.3\n" * 3 + "   2.5 2.5 2.5\n"
    )
    (stage / "topol.top").write_text("[ system ]\nMock\n[ molecules ]\nProtein 1\n")
    _edit(ws, *edit)
    call = {"mdp": "01_input/heat.mdp", "structure": "01_input/conf.gro",
            "topology": "01_input/topol.top", "stage": "run"}
    obs = REGISTRY.call("gmx_grompp", rendered, call)
    if refuses:
        assert not obs.ok and finding in [w.id for w in obs.unacknowledged]
        obs = REGISTRY.call("gmx_grompp", rendered, call | {"acknowledge_warnings": [finding]})
        event = [ev for ev in ws.ledger.events() if ev.tool == "gmx_grompp"][-1]
        assert event.warnings_acknowledged == [finding]
    else:
        assert finding in [w.id for w in obs.advisory]
    assert obs.ok, obs.summary
