"""Named model endpoints: several servers side by side, chosen per run, and a key that is
handed to the model call and to nothing else."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from myrmidon.cli import app
from myrmidon.endpoints import EndpointError, load_endpoints

KEY = "sk-test-0123456789-must-never-be-written"
TOML = """
[endpoints.site-a]
api_base = "https://site-a.example/v1/"
key_env = "SITE_A_KEY"

[endpoints.site-b]
api_base = "https://site-b.example/v1"
key_env = "SITE_B_KEY"
"""


@pytest.fixture
def endpoints_file(tmp_path, monkeypatch):
    path = tmp_path / "endpoints.toml"
    path.write_text(TOML)
    monkeypatch.setenv("MYRMIDON_ENDPOINTS", str(path))
    return path


# (the endpoint table, the refusal). A secret never sits in the config file.
BAD_ENDPOINTS = [
    ('api_base = "https://x/v1"\nkey_env = "X"\napi_key = "sk-1"', "belongs in the environment"),
    ('api_base = "https://x/v1"\nkey_env = "sk-abc def"', "names an environment variable"),
    ('api_base = "x.example"\nkey_env = "X"', "http"),
]


@pytest.mark.parametrize("body,refusal", BAD_ENDPOINTS)
def test_a_key_written_into_the_file_is_refused(tmp_path, body, refusal):
    path = tmp_path / "e.toml"
    path.write_text(f"[endpoints.x]\n{body}\n")
    with pytest.raises(EndpointError, match=refusal):
        load_endpoints(path)


class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


def test_the_endpoints_command_says_set_or_not_and_never_the_key(endpoints_file,
                                                                monkeypatch):
    monkeypatch.setenv("SITE_A_KEY", KEY)
    monkeypatch.delenv("SITE_B_KEY", raising=False)
    out = CliRunner().invoke(app, ["endpoints"]).output
    assert "site-a" in out and "$SITE_A_KEY set" in out and "$SITE_B_KEY NOT SET" in out
    assert KEY not in out


def test_a_model_error_that_echoes_the_key_does_not_record_it(ctx, tmp_path):
    from myrmidon.agent.loop import AgentLoop, StopReason

    def failing(model, messages, tools, **kwargs):
        raise RuntimeError(f"401 Unauthorized: invalid key {kwargs['api_key']}")

    loop = AgentLoop("openai/m", ctx, transcript_dir=tmp_path / "eps", completion_fn=failing,
                     connection={"api_base": "https://x/v1", "api_key": KEY}, endpoint="x")
    episode = loop.run("Anything.")
    assert episode.stop_reason is StopReason.MODEL_ERROR
    assert KEY not in episode.model_dump_json() and "[api_key]" in episode.error
    assert KEY not in Path(episode.transcript_path).read_text()
