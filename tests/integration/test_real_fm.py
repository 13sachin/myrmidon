"""Integration test: real MiMiCPy force matching through the harness.

Point MYRMIDON_FM_DATASET at a directory holding an acetone/water reference dataset
(ours has 2000 QM/MM configurations); the test skips when it is unset, because where
that data lives is a fact about your machine. dataset_collect validates the dataset (our real
dataset is known to be trr<->h5 desynced by one frame — mimicpy remaps), a parameter_fit stage
fits it with DRESP + parameter optimization (through SLURM when the machine config says so), and
its collection parses the outcome.
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
    import myrmidon.tools as basic

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


def test_force_matching_on_reference_dataset(ctx, approve_mandate):
    call(ctx, "study_create", name="acetone-fm", description="real FM integration")
    ws = ctx.workspace
    inputs = ws.stage("inputs")
    for name in ("acetone.top", "acetone.itp", "acetone.gro", "index.ndx", "sele.dat"):
        shutil.copy(DATASET / name, inputs / name)
    rel = lambda name: ws.relpath(inputs / name)  # noqa: E731

    ds = call(
        ctx, "dataset_collect",
        fmdata=str(DATASET / "FMTRAJECTORY.h5"), trajectory=str(DATASET / "mimic_traj.trr"),
        topology=rel("acetone.top"), coords=rel("acetone.gro"), selection=rel("sele.dat"),
    )
    assert ds.qm_atoms == 10
    # This real dataset's trr and h5 sampling drift apart: only a small subset of frames
    # truly match. dataset_collect registers exactly that consistent, aligned subset.
    assert ds.frames is not None and ds.frames >= 20
    assert ds.fmdata_sha256 and ds.trr_sha256

    approve_mandate(
        ctx, "Fit force-field parameters to the reference dataset", 256.0,
        stages=[{"name": "fit", "kind": "parameter_fit",
                 "qm_region": {"subsystems": ["solute"], "charge": 0}}],
        subsystems={"solute": {"kind": "ligand", "resnames": ["ACT"]}},
        structure=rel("acetone.gro"), topology=rel("acetone.top"), index=rel("index.ndx"),
        qm_selection=rel("sele.dat"))
    prepared = call(ctx, "stage_prepare", stage="fit")
    # the fit's input and its classical rerun's, rendered by the one call
    assert prepared.files["fm_input"].endswith("fminput.dat")
    assert prepared.files["mdp"].endswith("fit.mdp")
    job = call(ctx, "stage_run", stage="fit", walltime_min=30)
    done = wait_job(ctx, job.job_id)
    assert done.state == "COMPLETED", f"{done.summary}\n{done.log_tail or ''}"

    assert done.n_snapshots == ds.frames
    assert done.optimizer_success is True
    assert done.validation_sdf is not None and done.validation_sdf > 0
    assert done.n_charges == 10
    assert done.max_abs_charge_delta is not None and 0 < done.max_abs_charge_delta < 1.0
    assert {a.role for a in done.artifacts} >= {"fm_log", "fitted_topology", "topology"}
    # mimicpy names a fitted file after the include it rewrote, as the mock does, and the
    # candidate topology is assembled from it
    fitted = {Path(a.path).name for a in done.artifacts if a.role == "fitted_topology"}
    assert fitted & {"opt_acetone.itp", "resp_acetone.itp"}, fitted
    assert any(Path(a.path).name == "candidate.top" for a in done.artifacts if a.role == "topology")

    report = call(ctx, "study_report")
    assert report.rounds and report.rounds[0]["clean"]
    tools = {e.tool for e in ws.ledger.events()}
    assert tools >= {"dataset_collect", "stage_prepare", "stage_run", "study_report"}
