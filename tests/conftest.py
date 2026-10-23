import copy
import os
from pathlib import Path

import pytest

from myrmidon.config import MyrmidonConfig
from myrmidon.core.tool import ToolContext
from myrmidon.engines import mock
from myrmidon.engines.discovery import discover
from myrmidon.execution.local import LocalBackend
from myrmidon.knowledge.inputsets import InputSetStore
from myrmidon.knowledge.protocols import ProtocolStore

MOCKBIN = mock.BIN


@pytest.fixture
def mock_engines(monkeypatch):
    """Put the mock gmx/sbatch binaries first on PATH."""
    monkeypatch.setenv("PATH", f"{MOCKBIN}:{os.environ['PATH']}")


_DISCOVERED = {}


def _mock_engines(config):
    """The mock engines, discovered once: discovery runs each binary for its version, the
    answer is the same for every test, and doing it per test was most of the suite's setup.
    A copy each time, so no test changes another's."""
    if "engines" not in _DISCOVERED:
        _DISCOVERED["engines"] = discover(config)
    return copy.deepcopy(_DISCOVERED["engines"])


@pytest.fixture
def ctx(tmp_path, mock_engines):
    """A ToolContext wired for tests: mock engines, tmp workspaces, local backend."""
    import myrmidon.basic_tools as basic

    basic.register_all()
    config = MyrmidonConfig(studies_root=tmp_path / "workspaces")
    return ToolContext(
        config=config,
        workspace=None,
        backend=LocalBackend(),
        engines=_mock_engines(config),
        inputsets=InputSetStore.builtin(),
        protocols=ProtocolStore.builtin(),
    )


@pytest.fixture
def ws_ctx(ctx):
    """ctx with an active workspace."""
    from myrmidon.core.tool import REGISTRY

    obs = REGISTRY.call(
        "study_create", ctx, {"name": "test-study", "description": "unit test"}
    )
    assert obs.ok, obs.summary
    return ctx


@pytest.fixture
def planted_run(ws_ctx):
    """A workspace in which one grompp and one mdrun have gone through the registry on
    the mock engine and the job has been collected — a run with a ledger and a job store,
    which is what `watch` follows."""
    from myrmidon.core.tool import REGISTRY

    ws = ws_ctx.workspace
    stage = ws.stage("input")
    (stage / "conf.gro").write_text(
        "mock\n 3\n" + "    1ALA     CA    1   0.1   0.2   0.3\n" * 3 + "   2.5 2.5 2.5\n")
    (stage / "topol.top").write_text("[ system ]\nMock\n[ molecules ]\nProtein 1\n")
    (stage / "nvt.mdp").write_text(
        "integrator = md\ndt = 0.002\nnsteps = 500\nconstraints = h-bonds\n"
        "constraint_algorithm = lincs\ntcoupl = v-rescale\ntc-grps = System\n"
        "tau_t = 0.1\nref_t = 300\ncoulombtype = PME\ngen_vel = yes\ngen_seed = 1\n")
    grompp = REGISTRY.call("gmx_grompp", ws_ctx, {
        "mdp": "01_input/nvt.mdp", "structure": "01_input/conf.gro",
        "topology": "01_input/topol.top", "output": "nvt.tpr", "stage": "nvt"})
    assert grompp.ok, grompp.summary
    tpr = next(a.path for a in grompp.artifacts if a.path.endswith(".tpr"))
    run = REGISTRY.call("gmx_mdrun", ws_ctx, {"tpr": tpr, "deffnm": "nvt", "stage": "nvt",
                                              "walltime_min": 5})
    assert run.ok, run.summary
    REGISTRY.call("job_status", ws_ctx, {"job_id": run.job_id, "wait_seconds": 20})
    return ws_ctx, ws


@pytest.fixture
def cli_env(tmp_path, monkeypatch, mock_engines):
    """What the `myrmidon` command sees in a test: a machine config of its own.

    The CLI builds its own ToolContext from MyrmidonConfig.load(), so without this a
    test on a cluster machine would read the real config and submit real jobs.
    """
    config = tmp_path / "config.toml"
    root = tmp_path / "workspaces"
    config.write_text(
        'backend = "local"\n'
        f'studies_root = "{root}"\n'
    )
    monkeypatch.setenv("MYRMIDON_CONFIG", str(config))
    return root


REPO_ROOT = Path(__file__).resolve().parent.parent
#: what a myrmidon project leaves in a directory. A release snapshot ships a generic
#: `.mcp.json` of its own, so the invariant is that a test changes none of this — not
#: that none of it exists.
PROJECT_MARKERS = (REPO_ROOT / ".mcp.json", REPO_ROOT / ".claude" / "skills")
_MARKERS_AT_START = {path: path.exists() for path in PROJECT_MARKERS}


@pytest.fixture(autouse=True)
def _never_make_the_checkout_a_project():
    """No test may turn the directory it runs in into a myrmidon project.

    A launcher test that stubs `os.chdir` runs the launcher against the checkout and
    writes `.mcp.json`, `.claude/skills/` and the guidance block into `CLAUDE.md`. The
    product refuses that with the `CLAUDECODE` check, which these tests unset because
    they are testing the launcher; this is what is left to catch it.
    """
    yield
    for path, existed in _MARKERS_AT_START.items():
        assert path.exists() == existed, (
            f"a test changed {path}: a project marker it must not touch"
        )


@pytest.fixture(autouse=True)
def _model_retries_without_waiting(monkeypatch):
    """A failing scripted model is retried as a real one would be, without the waits."""
    monkeypatch.setattr("myrmidon.agent.loop.MODEL_RETRY_WAITS_S", (0.0, 0.0))
