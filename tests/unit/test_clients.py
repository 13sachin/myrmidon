"""An agent client is wired through files myrmidon writes: the server entry in the
client's own config, and for a client that runs harness code beside the person, that
code in the client's profile. These pin the Hermes wiring: where the profile is, what the
config gets, what the plugin is handed, and what the plugin refuses."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from myrmidon.client_entries import READERS, find_client_entries
from myrmidon.client_registry import get as client
from myrmidon.server import install


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """A Hermes profile of its own, under a home of its own, with no real endpoints."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home / "hermes"))
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.setenv("MYRMIDON_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("MYRMIDON_ENDPOINTS", str(tmp_path / "endpoints.toml"))
    return home / "hermes"


# (environment, the profile under ~): $HERMES_HOME wins, else the profile suffix
HOMES = [
    ({"HERMES_HOME": "/elsewhere/profile"}, Path("/elsewhere/profile")),
    ({"HERMES_DATA_DIR_SUFFIX": "-research"}, Path("/h/.hermes-research")),
    ({}, Path("/h/.hermes")),
]


@pytest.mark.parametrize("env,expected", HOMES)
def test_the_hermes_profile_follows_its_environment(monkeypatch, env, expected):
    for key in ("HERMES_HOME", "HERMES_DATA_DIR_SUFFIX"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert client("hermes").user_config(Path("/h")) == expected / "config.yaml"


# (the server entry's args, the command) -> the command line a view runs
LINES = [
    (["serve"], ["plan", "show"], ["plan", "show"]),
    (["-m", "myrmidon.cli", "serve"], ["status"], ["-m", "myrmidon.cli", "status"]),
]


@pytest.mark.parametrize("args,command,expected", LINES)
def test_a_view_runs_the_servers_own_program(args, command, expected):
    from myrmidon.clients.hermes._harness import cli  # importable now that nothing shadows it

    assert cli({"command": "/env/bin/x", "args": args}, command) == ["/env/bin/x", *expected]


def test_the_hermes_config_takes_the_entry_once_and_keeps_a_commented_file(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  default: m\n")
    entry = {"timeout": 660, "command": "/env/bin/myrmidon", "args": ["serve"]}
    assert install.merge_hermes_server(config, "myrmidon", entry) == (True, [])
    assert install.merge_hermes_server(config, "myrmidon", entry) == (False, [])
    assert READERS["hermes-yaml"](config)["myrmidon"] == entry
    assert install.enable_hermes_plugin(config) == ("updated", [])
    assert install.enable_hermes_plugin(config) == ("unchanged", [])

    commented = tmp_path / "hand.yaml"
    commented.write_text("# mine\nmodel:\n  default: m\n")
    changed, notes = install.merge_hermes_server(commented, "myrmidon", entry)
    assert (changed, len(notes)) == (False, 1) and "mcp_servers" in notes[0]
    what, notes = install.enable_hermes_plugin(commented)
    assert (what, len(notes)) == ("not written", 1) and "enabled" in notes[0]
    assert commented.read_text().startswith("# mine")


def test_init_project_puts_the_plugin_in_the_profile_and_trusts_the_project(hermes_home,
                                                                            tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    cwd = repo / "study-dir"
    cwd.mkdir()
    written = dict(install.hermes_project_extras(cwd, "myrmidon", True))

    plugin = hermes_home / "plugins" / "myrmidon"
    provider = hermes_home / "plugins" / "model-providers" / "myrmidon"
    for shipped in ("__init__.py", "_harness.py", "plugin.yaml", "server.json"):
        assert written[str(plugin / shipped)] == "created"
    for shipped in ("__init__.py", "plugin.yaml", "endpoints.json"):
        assert written[str(provider / shipped)] == "created"
    server = json.loads((plugin / "server.json").read_text())
    assert server == install.server_entry() and "--study" not in server["args"]
    assert json.loads((provider / "endpoints.json").read_text())["endpoints"] == []

    import yaml

    config = yaml.safe_load((hermes_home / "config.yaml").read_text())
    assert config["plugins"]["enabled"] == ["myrmidon"]
    assert config["skills"]["trusted_project_dirs"] == [str(repo.resolve())]
    assert (repo / ".hermes" / "skills").is_dir()  # the root's, not the subdirectory's

    again = install.hermes_project_extras(cwd, "myrmidon", True)
    assert {what for _, what in again} == {"unchanged"}

    install.merge_hermes_server(hermes_home / "config.yaml", "myrmidon",
                                install.client_entry(client("hermes"), scope="user"))
    (found,) = [e for e in find_client_entries(cwd, hermes_home.parent) if e["client"] == "hermes"]
    assert (found["scope"], found["args"]) == ("user", ["serve"])


def _load_by_path(name: str, file: Path):
    """A file as Hermes loads it: by path, in an interpreter without myrmidon. A plugin's
    `__init__.py` is loaded as a package so that `._harness` beside it resolves."""
    search = [str(file.parent)] if file.name == "__init__.py" else None
    spec = importlib.util.spec_from_file_location(name, file, submodule_search_locations=search)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _load_installed_plugin(plugin_dir: Path):
    return _load_by_path("myrmidon_hermes_plugin_under_test", plugin_dir / "__init__.py")


class _Ctx:
    def __init__(self):
        self.hooks: dict[str, object] = {}
        self.commands: dict[str, object] = {}

    def register_hook(self, name, fn):
        self.hooks[name] = fn

    def register_command(self, name, fn, _description=""):
        self.commands[name] = fn


# (the tool, its arguments, blocked)
CALLS = [
    ("terminal", {"command": "myrmidon approve --yes"}, True),
    ("execute_code", {"code": "subprocess.run(['myrmidon', 'approve', '--yes'])"}, True),
    ("terminal", {"command": "python -m myrmidon.cli approve --digest x"}, True),
    ("terminal", {"command": "myrmidon status"}, False),
    ("terminal", {"command": "grep approve myrmidon.ledger.jsonl; cat core/mandate.py"}, False),
    ("mcp__myrmidon__mandate_show", {}, False),
]


@pytest.mark.parametrize("tool,args,blocked", CALLS)
def test_the_plugin_refuses_the_models_approval_in_every_spelling(hermes_home, monkeypatch,
                                                                  tool, args, blocked):
    install.install_hermes_plugin()
    ctx = _Ctx()
    _load_installed_plugin(hermes_home / "plugins" / "myrmidon").register(ctx)
    verdict = ctx.hooks["pre_tool_call"](tool_name=tool, args=args, session_id="s")
    assert (verdict or {}).get("action") == ("block" if blocked else None)
    assert set(ctx.commands) == {"status", "plan", "inbox", "answer", "approve"}


def test_the_plugin_looks_at_the_study_once_per_session_within_a_bound(hermes_home,
                                                                       monkeypatch):
    install.install_hermes_plugin()
    module = _load_installed_plugin(hermes_home / "plugins" / "myrmidon")
    runs: list[list[str]] = []

    def fake_run(argv, cwd, timeout=120, env=None):
        runs.append(argv)
        return 0, "a study, open", ""

    monkeypatch.setattr(module, "run", fake_run)
    ctx = _Ctx()
    module.register(ctx)
    look = ctx.hooks["pre_llm_call"]
    assert look(session_id="s1", is_first_turn=True)["context"].endswith("a study, open")
    assert look(session_id="s1", is_first_turn=True) is None
    assert look(session_id="s2", is_first_turn=False) is None
    assert len(runs) == 1 and runs[0][-2:] == ["study", "open"]
    for n in range(64):
        look(session_id=f"other-{n}", is_first_turn=True)
    assert look(session_id="s1", is_first_turn=True) is not None  # forgotten past the bound


def test_approve_yes_is_bound_to_the_digest_shown_in_that_session(hermes_home, monkeypatch):
    """`/approve yes` records the digest `/approve` last showed in the same session, and
    only while the file on disk still has it; a revision in between is shown, not
    recorded."""
    import json as _json

    install.install_hermes_plugin()
    module = _load_installed_plugin(hermes_home / "plugins" / "myrmidon")
    disk = {"digest": "a" * 64}
    recorded: list[list[str]] = []

    def fake_run(argv, cwd, timeout=120, env=None):
        if "--show" in argv:
            return 0, _json.dumps({"kind": "mandate", "digest": disk["digest"], "text": "M",
                                   "question": "q", "ceiling_core_h": 10}), ""
        recorded.append(argv)
        return 0, _json.dumps({"ok": True, "event": "ev-1", "approver": "r",
                               "summary": "approved"}), ""

    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(module, "server_entry", lambda: {"command": "myrmidon",
                                                         "args": ["serve"]})
    approve = module._approve
    assert "Read it first" in approve("yes", session_id="s1")
    approve("", session_id="s1")
    disk["digest"] = "b" * 64  # a revision lands after s1 read the mandate
    assert "changed since you read it" in approve("yes", session_id="s1") and not recorded
    assert "Read it first" in approve("yes", session_id="s2")  # s1's look is not s2's
    assert "Approved" in approve("yes", session_id="s1")
    assert recorded and "b" * 64 in recorded[0]
