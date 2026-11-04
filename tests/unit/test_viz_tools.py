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


