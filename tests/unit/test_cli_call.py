"""The command line for the registry.

The rule these tests hold to: the command line never has a capability the registry
lacks. `myrmidon call` is that rule in code, so what is tested is that a person at a
terminal reaches the same registry, the same gate and the same ledger an agent does.
"""


import pytest
from typer.testing import CliRunner

from myrmidon.cli import app

FIXTURES = __import__("pathlib").Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def planted(ws_ctx):
    """A workspace with the acetone inputs in it, as stage A starts."""
    import shutil

    ws = ws_ctx.workspace
    stage = ws.stage("input")
    for name in ("gro", "top"):
        shutil.copy(FIXTURES / "acetone" / f"acetone_stageA_entry.{name}",
                    stage / f"acetone.{name}")
    shutil.copy(FIXTURES / "acetone" / "acetone.itp", stage / "acetone.itp")
    return ws


# --- the command itself --------------------------------------------------------------


def test_a_call_from_the_terminal_lands_in_the_ledger(runner, cli_env, planted):
    result = runner.invoke(app, [
        "call", "gmx_editconf", "--study", str(planted.root),
        "structure=01_input/acetone.gro", "box_type=cubic", "distance_nm=1.0384",
    ])
    assert result.exit_code == 0, result.output
    assert "02_prep/boxed.gro" in result.output
    events = [ev for ev in planted.ledger.events() if ev.tool == "gmx_editconf"]
    assert len(events) == 1 and events[0].ok
    assert events[0].params["distance_nm"] == 1.0384  # not the string


def test_a_gate_refusal_exits_one_and_an_advisory_is_printed_with_the_call(
    runner, cli_env, planted
):
    common = ["call", "inputset_apply", "--study", str(planted.root),
              "method=forcefield.amber", "output=01_input/mini.mdp"]
    refused = runner.invoke(app, [*common, "inputset=gromacs.nonsense"])
    assert refused.exit_code == 1, refused.output
    assert "block" in refused.output
    assert not (planted.root / "01_input" / "mini.mdp").exists()

    ran = runner.invoke(app, [*common, "inputset=gromacs.minimise",
                              'overrides={"integrator": "md"}'])
    assert ran.exit_code == 0, ran.output
    assert "INPUTSET_VALUE_OVERRIDDEN" in ran.output
    assert (planted.root / "01_input" / "mini.mdp").is_file()
    kinds = [ev.kind for ev in planted.ledger.events() if ev.tool == "inputset_apply"]
    assert kinds == ["gate_refused", "call"]
