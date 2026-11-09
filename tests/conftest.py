import copy
import os
import shutil
from pathlib import Path

import pytest

from myrmidon.config import MyrmidonConfig
from myrmidon.core.tool import ToolContext
from myrmidon.engines import mock
from myrmidon.engines.discovery import discover
from myrmidon.execution.local import LocalBackend
from myrmidon.knowledge.inputsets import InputSetStore
from myrmidon.knowledge.protocols import ProtocolStore

#: input sets the machinery tests resolve: a pack standing in for the package's, which ships
#: none (a stage's values are its templates')
FIXTURE_INPUTSETS = Path(__file__).parent / "fixtures" / "inputsets"


def fixture_sets() -> InputSetStore:
    return InputSetStore(builtin_path=FIXTURE_INPUTSETS)

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
    import myrmidon.tools as basic

    basic.register_all()
    config = MyrmidonConfig(studies_root=tmp_path / "workspaces")
    # an attended context with a person at the keyboard, so a test may approve a plan;
    # `unattended` and `person` are the two axes a test sets to say otherwise
    return ToolContext(
        config=config,
        workspace=None,
        backend=LocalBackend(),
        engines=_mock_engines(config),
        inputsets=fixture_sets(),
        protocols=ProtocolStore.builtin(),
        person="researcher",
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


ACETONE = Path(__file__).parent / "fixtures" / "acetone"


@pytest.fixture
def solvated(ws_ctx):
    """The system files an engine input reads its facts from — acetone in SPC water
    (amber03, GAFF): its topology and its equilibrated structure, in the study."""
    stage = ws_ctx.workspace.stage("input")
    shutil.copy(ACETONE / "acetone_equilibrated.top", stage / "topol.top")
    shutil.copy(ACETONE / "acetone.itp", stage / "acetone.itp")
    shutil.copytree(ACETONE / "amber03.ff", stage / "amber03.ff")
    shutil.copy(ACETONE / "acetone_equilibrated.gro", stage / "conf.gro")
    rel = ws_ctx.workspace.relpath
    return {"topology": rel(stage / "topol.top"), "structure": rel(stage / "conf.gro")}


@pytest.fixture
def planted_run(ws_ctx):
    """A workspace in which one prepared stage has run on the mock engine under an approved
    mandate and the job has been collected — a run with a ledger and a job store, which is
    what `watch` follows."""
    ws = ws_ctx.workspace
    stage = ws.stage("input")
    (stage / "conf.gro").write_text(
        "mock\n 3\n" + "    1ALA     CA    1   0.1   0.2   0.3\n" * 3 + "   2.5 2.5 2.5\n")
    (stage / "topol.top").write_text("[ system ]\nMock\n[ molecules ]\nProtein 1\n")
    approved_study(ws_ctx, "01_input/conf.gro", "01_input/topol.top", [NVT_STAGE])
    prepared, run = prepare_and_run(ws_ctx, "nvt", walltime_min=5)
    _registry().call("job_status", ws_ctx, {"job_id": run.job_id, "wait_seconds": 20})
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
    # the input sets the machinery tests resolve stand in for the package's pack here too
    monkeypatch.setattr(InputSetStore, "builtin_dir", classmethod(lambda cls: FIXTURE_INPUTSETS))
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


def _registry():
    from myrmidon.core.tool import REGISTRY

    return REGISTRY


#: a classical stage any test can run
NVT_STAGE = {"name": "nvt", "kind": "dynamics", "ensemble": "NVT", "temperature_k": 300,
             "steps": 500}


def approved_study(ctx, structure, topology, stages, **extra):
    """A mandate of these stages over this system, written and approved by the researcher."""
    from myrmidon.core import mandate as mandates

    written = _registry().call("mandate_write", ctx, {
        "question": "Run the stages", "ceiling_core_h": 5000.0,
        "system_files": [{"path": structure, "kind": "structure"},
                         {"path": topology, "kind": "topology"}],
        "hamiltonian": {"mm": "amber"}, "stages": stages, **extra})
    assert written.ok and written.status == "ready", written.summary
    held = mandates.read_mandate(ctx.workspace)
    mandates.record_approval(ctx.workspace, held, approver="researcher", via="terminal")
    return held


def prepare_and_run(ctx, stage, prepare=None, **run):
    """A stage prepared, and its run submitted: (the preparation, the submission)."""
    prepared = _registry().call("stage_prepare", ctx, {"stage": stage, **(prepare or {})})
    assert prepared.ok, prepared.summary
    submitted = _registry().call("stage_run", ctx, {"stage": stage, **run})
    assert submitted.ok, submitted.summary
    return prepared, submitted


#: a coupled stage's region, on the solute of ``coupled_study``
REGION = {"subsystems": ["solute"], "charge": 0}
#: what a coupled study is computed with
QM_HAMILTONIAN = {"mm": "amber", "qm": {"method": "blyp_mt", "cutoff_ry": 70},
                  "coupling": {"scheme": "mimic"}}


def coupled_study(ws_ctx, stages, project: Path | None = None, **extra):
    """Acetone in water, its QM region selected, under an approved mandate of these stages;
    its files in the study, or in ``project``, the directory the study was made from."""
    import json
    import shutil

    from myrmidon.core import mandate as mandates
    from myrmidon.core.workspace import META_FILENAME

    ws = ws_ctx.workspace
    stage = ws.stage("input") if project is None else project
    for name in ("acetone_equilibrated.top", "acetone.itp", "acetone_equilibrated.gro"):
        (stage / name).write_text((ACETONE / name).read_text())
    shutil.copytree(ACETONE / "amber03.ff", stage / "amber03.ff")
    rel = ws.relpath
    if project is not None:  # named as the researcher names them, from the project
        meta = json.loads((ws.root / META_FILENAME).read_text())
        (ws.root / META_FILENAME).write_text(json.dumps(meta | {"origin": str(project)}))
        rel = lambda path: str(path.relative_to(project))  # noqa: E731
    top, gro = rel(stage / "acetone_equilibrated.top"), rel(stage / "acetone_equilibrated.gro")
    selected = _registry().call("qm_region", ws_ctx, {
        "topology": top, "coords": gro, "selections": ["add resname is ACT"], "padding_nm": 0.4})
    assert selected.ok, selected.summary
    written = _registry().call("mandate_write", ws_ctx, {
        "question": "Run acetone in water under QM/MM", "ceiling_core_h": 5000.0,
        "system_files": [{"path": gro, "kind": "structure"}, {"path": top, "kind": "topology"}],
        "subsystems": {"solute": {"kind": "ligand", "resnames": ["ACT"]},
                       "solvent": {"kind": "solvent", "molecules": ["SOL"]}},
        "hamiltonian": QM_HAMILTONIAN, "stages": stages, **extra})
    assert written.ok and written.status == "ready", written.summary
    held = mandates.read_mandate(ws)
    mandates.record_approval(ws, held, approver="researcher", via="terminal")
    return held


def wait_for(ctx, job_id, seconds=15):
    """A job waited for and collected."""
    import time

    deadline = time.time() + seconds
    while time.time() < deadline:
        done = _registry().call("job_status", ctx, {"job_id": job_id})
        if done.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return done
        time.sleep(0.1)
    raise AssertionError(f"{job_id} never finished")
