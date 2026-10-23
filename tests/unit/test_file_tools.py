"""file_write / file_read / file_check: confined, and a look is not a step."""

import pytest

from myrmidon.core.tool import REGISTRY


def test_a_look_at_a_file_is_checked_and_is_not_a_step(ws_ctx):
    REGISTRY.call(
        "file_write",
        ws_ctx,
        {"path": "01_run/bad.mdp", "content": "integrator = md\ndt = 0.005\nnsteps = 10\n"},
    )
    report = REGISTRY.call("file_check", ws_ctx, {"path": "01_run/bad.mdp"})
    assert not report.ok  # dt too large blocks
    assert any(w.id == "MDP_DT_TOO_LARGE" for w in report.blocking)
    assert report.file_kind == "mdp" and "parameter_source" in report.checked
    looks = [ev for ev in ws_ctx.workspace.ledger.events() if ev.tool == "file_check"]
    assert looks and {ev.kind for ev in looks} == {"observe"}


@pytest.fixture
def project(ws_ctx, tmp_path, monkeypatch):
    """A study made from a project directory that holds the researcher's own files, the
    way bare `myrmidon` in that directory makes it — plus the project's hidden files (a
    study made from $HOME has ~/.config/myrmidon/secrets.env inside its project) and a
    file in neither."""
    project = tmp_path / "project"
    (project / "box").mkdir(parents=True)
    (project / "box" / "topol.top").write_text("[ system ]\nbox\n")
    (project / ".config" / "myrmidon").mkdir(parents=True)
    (project / ".config" / "myrmidon" / "secrets.env").write_text("export KEY=hunter2\n")
    (tmp_path / "elsewhere.txt").write_text("not yours")
    monkeypatch.chdir(project)
    assert REGISTRY.call("study_create", ws_ctx, {"name": "nvt", "description": "x"}).ok
    (ws_ctx.workspace.root / "mine.txt").write_text("the study's")
    return project


# (tool, path with {project}/{tmp} filled in, allowed). What file_read returns goes to the
# model's server: the study and the project, never beyond, never the project's hidden
# files. file_write writes only inside the study.
CONFINEMENT = [
    ("file_read", "mine.txt", True),
    ("file_read", "", True),  # the study's own root lists
    ("file_read", "box/topol.top", True),  # the researcher's path, relative to the project
    ("file_read", "{project}/box/topol.top", True),
    ("file_read", ".config/myrmidon/secrets.env", False),
    ("file_read", "{project}/.config", False),
    ("file_read", "{tmp}/elsewhere.txt", False),
    ("file_read", "../elsewhere.txt", False),
    ("file_read", "/etc/passwd", False),
    ("file_write", "notes.txt", True),
    ("file_write", "../outside.txt", False),
    ("file_write", "a/../../outside.txt", False),
    ("file_write", "/etc/passwd", False),
    ("file_write", "{project}/box/new.txt", False),
]


@pytest.mark.parametrize("tool,path,allowed", CONFINEMENT)
def test_reads_stay_in_the_study_and_project_and_writes_in_the_study(
        ws_ctx, project, tmp_path, tool, path, allowed):
    path = path.format(project=project, tmp=tmp_path)
    params = {"path": path, "content": "x"} if tool == "file_write" else {"path": path}
    obs = REGISTRY.call(tool, ws_ctx, params)
    assert obs.ok is allowed, (path, obs.summary)
    if not allowed:
        assert "hunter2" not in obs.summary and "not yours" not in obs.summary
