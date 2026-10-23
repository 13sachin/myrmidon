"""`--config FILE`: the file is the whole configuration."""

import pytest
from typer.testing import CliRunner

from myrmidon.cli import app
from myrmidon.config import ComputeDefaults, MyrmidonConfig
from myrmidon.errors import ConfigError


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A machine with a config, a project file above the working directory and a
    MYRMIDON_* variable — every source the usual layering reads."""
    home = tmp_path / "home"
    (home / ".config" / "myrmidon").mkdir(parents=True)
    (home / ".config" / "myrmidon" / "config.toml").write_text(
        'backend = "slurm"\ncpmd_bin = "machine_cpmd"\n[compute]\nwarn_core_h = 5000.0\n')
    project = tmp_path / "project"
    (project / "sub").mkdir(parents=True)
    (project / ".myrmidon.toml").write_text('client = "qwen"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("MYRMIDON_CONFIG", raising=False)
    monkeypatch.setenv("MYRMIDON_GROMACS_BIN", "gmx_from_shell")
    monkeypatch.chdir(project / "sub")
    return tmp_path


def test_a_config_file_is_read_alone(machine):
    path = machine / "only.toml"
    path.write_text('cpmd_bin = "mine"\n')
    config = MyrmidonConfig.from_file(path)
    assert config.cpmd_bin == "mine"
    # nothing from the machine config, the project file or the shell
    assert config.backend == "local"
    assert config.client == "claude"
    assert config.gromacs_bin == "gmx"
    assert config.compute == ComputeDefaults()


def test_a_missing_config_file_is_an_error_not_the_defaults(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        MyrmidonConfig.from_file(tmp_path / "nope.toml")


def _scripted_run(monkeypatch, tmp_path, args, *replies):
    """`myrmidon run` with a scripted model, a config file and a fresh HOME."""
    from tests.unit.test_agent_loop import FakeMessage, scripted

    from myrmidon.agent import pydantic_adapter

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MYRMIDON_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    replies = replies or (FakeMessage(content="Nothing to measure."),)
    model = replies[0] if callable(replies[0]) else scripted(*replies)
    monkeypatch.setattr(pydantic_adapter, "PydanticCompletion", lambda **kw: model)
    path = tmp_path / "c.toml"
    path.write_text(f'studies_root = "{tmp_path / "studies"}"\n')
    return CliRunner().invoke(app, ["run", "a question", "--model", "fake",
                                    "--config", str(path), *args])


def test_run_writes_its_transcript_into_the_study(monkeypatch, tmp_path):
    from myrmidon.core.workspace import Workspace

    ws = Workspace.create(tmp_path / "studies" / "s1", "s1")
    out = _scripted_run(monkeypatch, tmp_path, ["--study", str(ws.root)])
    assert out.exit_code == 0, out.output
    written = list((ws.root / ".myrmidon" / "transcripts").glob("*.jsonl"))
    assert len(written) == 1 and f"transcript: {written[0]}" in out.output
    assert not (tmp_path / "episodes").exists(), "never into the working directory"


def test_run_exits_by_how_it_ended(monkeypatch, tmp_path):
    from tests.unit.test_agent_loop import FakeMessage

    def unreachable(**kw):
        raise ConnectionError("endpoint down")

    assert _scripted_run(monkeypatch, tmp_path, []).exit_code == 0
    out = _scripted_run(monkeypatch, tmp_path, [], unreachable)
    assert out.exit_code == 3 and "endpoint down" in out.output
    untraced = FakeMessage(content="The temperature was 412.7 K.")
    out = _scripted_run(monkeypatch, tmp_path, [], *[untraced] * 5)
    assert out.exit_code == 1, "an answer whose number nothing measured is not an answer"


def test_the_last_transcript_event_says_how_the_run_ended(monkeypatch, tmp_path):
    import json

    def unreachable(**kw):
        raise ConnectionError("endpoint down")

    _scripted_run(monkeypatch, tmp_path, [], unreachable)
    (path,) = (tmp_path / "studies" / ".sessions" / "runs").glob("*.jsonl")
    last = json.loads(path.read_text().splitlines()[-1])
    assert (last["kind"], last["stop_reason"]) == ("episode_end", "MODEL_ERROR")


def test_run_mock_puts_the_fake_engines_and_scheduler_in_front(monkeypatch, tmp_path):
    import os

    from myrmidon.core.workspace import Workspace
    from myrmidon.engines import mock

    # `--mock` changes this process's environment for good, as `serve --mock` does
    monkeypatch.setenv("PATH", os.environ["PATH"])
    for key in ("MOCK_SLURM_DIR", "MOCK_SLURM_CLOCK"):
        monkeypatch.delenv(key, raising=False)
    seen = []
    real = mock.mock_context
    monkeypatch.setattr(mock, "mock_context", lambda ctx: seen.append(real(ctx)) or seen[-1])
    ws = Workspace.create(tmp_path / "studies" / "s1", "s1")
    out = _scripted_run(monkeypatch, tmp_path, ["--study", str(ws.root), "--mock"])
    assert out.exit_code == 0, out.output
    (ctx,) = seen
    assert "mock" in ctx.engines["gromacs"].version
    assert ctx.workspace.root == ws.root  # the study survives the swap
    assert os.environ["PATH"].split(os.pathsep)[0] == str(mock.BIN)
    assert os.environ["MOCK_SLURM_DIR"] == str(ws.root / ".mock_slurm")
