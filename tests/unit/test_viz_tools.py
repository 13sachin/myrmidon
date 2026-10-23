"""Visualization tool chain against the mock chimerax REST server."""

import shutil
from pathlib import Path

import pytest

from myrmidon.core.tool import REGISTRY

FIXTURE_PDB = Path(__file__).parent.parent / "fixtures" / "1l2y_model1.pdb"


@pytest.fixture
def viz_ctx(ws_ctx):
    yield ws_ctx
    REGISTRY.call("viz_command", ws_ctx, {"close_session": True})


def _open_fixture(ctx):
    ws = ctx.workspace
    stage = ws.stage("fetch")
    shutil.copy(FIXTURE_PDB, stage / "1l2y.pdb")
    return REGISTRY.call("viz_open", ctx, {"structure": ws.relpath(stage / "1l2y.pdb")})


def test_viz_open_reports_grounded_structure_info(viz_ctx):
    obs = _open_fixture(viz_ctx)
    assert obs.ok, obs.summary
    assert obs.session_started is True
    assert obs.n_models == 1
    assert obs.n_atoms == 304  # every ATOM record of model 1, parsed not assumed
    assert obs.chains == {"A": 20}
    assert obs.ligands == []
    assert obs.image is not None
    png = viz_ctx.workspace.root / obs.image
    assert png.is_file() and png.stat().st_size > 0
    assert {a.role for a in obs.artifacts} == {"snapshot"}
    assert obs.provenance_id is not None


def test_viz_open_reports_ligand_names_with_specs(viz_ctx):
    """Ligand entries carry the residue name (real 1.12 ``info residues``
    reports attribute ``name`` by default), so NAP vs ICT is never guessed."""
    ws = viz_ctx.workspace
    stage = ws.stage("fetch")
    pdb = stage / "mini.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "HETATM    3  PA  NAP A1515       5.000   0.000   0.000  1.00  0.00           P\n"
        "HETATM    4  C1  ICT A1616       9.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    obs = REGISTRY.call("viz_open", viz_ctx, {"structure": ws.relpath(pdb)})
    assert obs.ok, obs.summary
    assert obs.ligands == ["/A:1515 NAP", "/A:1616 ICT"]


def test_viz_command_surfaces_chimerax_errors(viz_ctx):
    _open_fixture(viz_ctx)
    obs = REGISTRY.call(
        "viz_command",
        viz_ctx,
        {"commands": ["color /A red", "bogus_command_xyz", "lighting soft"]},
    )
    assert not obs.ok
    assert "bogus_command_xyz" in obs.summary
    assert obs.executed == ["color /A red", "bogus_command_xyz"]  # stopped at failure


def test_log_level_errors_count_as_failure(viz_ctx):
    """A null top-level "error" with log-level error text is still a failure
    (seen on real ChimeraX 1.12, e.g. when the command parameter is lost)."""
    from myrmidon.viz.session import VizSession

    _open_fixture(viz_ctx)
    session = VizSession.attach(viz_ctx.workspace.state_dir)
    raw = session._request("", timeout=10)  # mock answers like real: log error only
    assert raw["error"] is None and raw["log messages"]["error"]
    result = session.run("")
    assert not result.ok
    assert "parameter missing" in result.error

