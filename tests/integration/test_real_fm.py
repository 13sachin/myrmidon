"""Integration test: real MiMiCPy force matching through the harness.

Point MYRMIDON_FM_DATASET at a directory holding an acetone/water reference dataset
(ours has 2000 QM/MM configurations); the test skips when it is unset, because where
that data lives is a fact about your machine. fm_collect validates the dataset (our real
dataset is known to be trr<->h5 desynced by one frame — mimicpy remaps), fm_plan
selects 100 snapshots, fm_fit runs DRESP + parameter optimization (through SLURM
when the machine config says so), fm_results parses the outcome.
"""

import os
import shutil
import time
import uuid
from pathlib import Path

import pytest

from myrmidon.config import MyrmidonConfig
from myrmidon.core.tool import REGISTRY, ToolContext
from myrmidon.engines.discovery import discover
from myrmidon.server.mcp_server import build_backend

CONFIG = MyrmidonConfig.load()
ENGINES = discover(CONFIG)
# No default: a dataset path is a fact about one machine and does not belong in the
# repository. Unset means skip, which is what it does on any machine but the one holding it.
_DATASET_ENV = os.environ.get("MYRMIDON_FM_DATASET")
DATASET = Path(_DATASET_ENV) if _DATASET_ENV else None

pytestmark = [
    pytest.mark.mimic,
    pytest.mark.timeout(3600),
    pytest.mark.skipif(
        not all(ENGINES[e].available for e in ("gromacs", "mimicpy"))
        or "mock" in ENGINES["gromacs"].version,
        reason="real GROMACS + mimicpy not available",
    ),
    pytest.mark.skipif(
        DATASET is None or not (DATASET / "FMTRAJECTORY.h5").exists(),
        reason="set MYRMIDON_FM_DATASET to a directory holding FMTRAJECTORY.h5",
    ),
]


@pytest.fixture
def ctx(tmp_path):
    import myrmidon.basic_tools as basic

    basic.register_all()
    if CONFIG.backend == "slurm":
        root = Path(CONFIG.studies_root) / "_scratch" / "integration" / uuid.uuid4().hex[:8]
    else:
        root = tmp_path / "workspaces"
    config = CONFIG.model_copy(update={"studies_root": root})
    return ToolContext(
        config=config,
        backend=build_backend(config),
        engines=ENGINES,
    )


def call(ctx, tool_name, **kw):
    obs = REGISTRY.call(tool_name, ctx, kw)
    assert obs.ok, f"{tool_name}: {obs.summary}\n{obs.log_tail or ''}"
    return obs


def wait_job(ctx, job_id, timeout=2400):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = REGISTRY.call("job_status", ctx, {"job_id": job_id, "wait_seconds": 30})
        if st.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
    raise AssertionError(f"{job_id} did not finish within {timeout}s")


def test_force_matching_on_reference_dataset(ctx):
    call(ctx, "study_create", name="acetone-fm", description="real FM integration")
    ws = ctx.workspace
    inputs = ws.stage("inputs")
    for name in ("acetone.top", "acetone.itp", "acetone.gro", "index.ndx",
                 "sele.dat", "mimic.mdp"):
        shutil.copy(DATASET / name, inputs / name)
    rel = lambda name: ws.relpath(inputs / name)  # noqa: E731

    ds = call(
        ctx, "fm_collect",
        fmdata=str(DATASET / "FMTRAJECTORY.h5"), trajectory=str(DATASET / "mimic_traj.trr"),
        topology=rel("acetone.top"), coords=rel("acetone.gro"), selection=rel("sele.dat"),
    )
    assert ds.qm_atoms == 10
    # This real dataset's trr and h5 sampling drift apart: only a small subset of frames
    # truly match. fm_collect registers exactly that consistent, aligned subset.
    assert ds.frames is not None and ds.frames >= 20
    assert ds.fmdata_sha256 and ds.trr_sha256
    dataset = {a.role: str(ws.root / a.path) for a in ds.artifacts}

    plan = call(
        ctx, "fm_plan",
        stride=[0, ds.frames, 1], qm_total_charge=0, dataset_frames=ds.frames,
    )
    assert plan.n_snapshots == ds.frames

    job = call(
        ctx, "fm_fit",
        topology=rel("acetone.top"), coords=rel("acetone.gro"),
        fmdata=dataset["fm_dataset"], trajectory=dataset["trajectory"],
        index=rel("index.ndx"), selection=rel("sele.dat"),
        fm_input=plan.artifacts[0].path, mdp=rel("mimic.mdp"), n_processes=12,
    )
    done = wait_job(ctx, job.job_id)
    assert done.state == "COMPLETED", f"{done.summary}\n{done.log_tail or ''}"

    res = call(ctx, "fm_results", original_itp=rel("acetone.itp"))
    assert res.n_snapshots == ds.frames
    assert res.optimizer_success is True
    assert res.validation_sdf is not None and res.validation_sdf > 0
    assert res.n_charges == 10
    assert res.max_abs_charge_delta is not None and 0 < res.max_abs_charge_delta < 1.0
    assert {a.role for a in res.artifacts} >= {"fm_log", "fitted_topology"}

    tools = {e.tool for e in ws.ledger.events()}
    assert tools >= {"fm_collect", "fm_plan", "fm_fit", "fm_results"}
