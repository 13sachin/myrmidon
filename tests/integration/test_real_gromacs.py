"""Integration test against a real GROMACS installation.

The full classical chain on Trp-cage (1L2Y, 20 residues) in water:
prep → ionize → minimize → gated grompp → async NVT → energy/RMSD → the ledger.

Auto-skipped when no real GROMACS is discovered. Runs in about a minute on a
laptop with the local backend; on a cluster whose machine config selects
``backend = "slurm"`` the mdrun steps go through real sbatch and the workspace
lives under the configured ``studies_root`` (compute nodes cannot see the
login node's /tmp).
"""

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
FIXTURE = Path(__file__).parent.parent / "fixtures" / "1l2y_model1.pdb"

pytestmark = [
    pytest.mark.gromacs,
    pytest.mark.timeout(2400 if CONFIG.backend == "slurm" else 600),
    pytest.mark.skipif(
        shutil.which(CONFIG.gromacs_bin) is None,
        reason=f"real GROMACS ({CONFIG.gromacs_bin}) not on PATH",
    ),
]


@pytest.fixture
def real_ctx(tmp_path):
    import myrmidon.tools as basic

    basic.register_all()
    if CONFIG.backend == "slurm":
        root = Path(CONFIG.studies_root) / "_scratch" / "integration" / uuid.uuid4().hex[:8]
    else:
        root = tmp_path / "workspaces"
    config = CONFIG.model_copy(update={"studies_root": root})
    ctx = ToolContext(
        config=config,
        backend=build_backend(config),
        engines=discover(config),
    )
    if not ctx.engines["gromacs"].available:
        pytest.skip(ctx.engines["gromacs"].hint)
    if "mock" in ctx.engines["gromacs"].version:
        pytest.skip("mock gmx shadows the real one on PATH")
    return ctx


def call(ctx, tool_name, **kw):
    obs = REGISTRY.call(tool_name, ctx, kw)
    assert obs.ok, f"{tool_name}: {obs.summary}\n{obs.log_tail or ''}"
    return obs


def art(obs, role):
    return next(a.path for a in obs.artifacts if a.role == role)


def wait_job(ctx, job_id, timeout=None):
    timeout = timeout or (900 if CONFIG.backend == "slurm" else 420)
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = REGISTRY.call("job_status", ctx, {"job_id": job_id})
        if st.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
        time.sleep(5 if CONFIG.backend == "slurm" else 2)
    raise AssertionError(f"{job_id} did not finish within {timeout}s")


def test_full_chain_on_trp_cage(real_ctx, approve_mandate):
    ctx = real_ctx
    call(ctx, "study_create", name="1l2y-nvt", description="integration test")
    ws = ctx.workspace

    fetch_stage = ws.stage("fetch")
    shutil.copy(FIXTURE, fetch_stage / "1l2y.pdb")

    # --- prep ---
    prep = call(
        ctx,
        "gmx_pdb2gmx",
        structure=ws.relpath(fetch_stage / "1l2y.pdb"),
        forcefield="amber99sb-ildn",
        water="tip3p",
        ignore_hydrogens=True,
    )
    assert prep.n_atoms > 300
    assert prep.net_charge == pytest.approx(1.0)  # Trp-cage is +1

    boxed = call(
        ctx, "gmx_editconf", structure=art(prep, "structure"),
        box_type="cubic", distance_nm=1.0,
    )
    solv = call(
        ctx, "gmx_solvate", structure=art(boxed, "structure"),
        topology=art(prep, "topology"),
    )
    assert solv.n_solvent and solv.n_solvent > 400

    # --- neutralize: genion builds its own run input ---
    ionized = call(
        ctx, "gmx_genion", structure=art(solv, "structure"),
        topology=art(prep, "topology"), neutralize=True,
    )
    assert ionized.n_ions_negative == 1  # one Cl- neutralizes the +1

    approve_mandate(
        ctx, "Minimize and run 10 ps of NVT on Trp-cage", 64.0,
        stages=[{"name": "em", "kind": "minimize", "steps": 200},
                {"name": "nvt", "kind": "dynamics", "ensemble": "NVT", "temperature_k": 300,
                 "duration_ps": 10, "sampling_ps": 1}],
        structure=art(ionized, "structure"), topology=art(ionized, "topology"))

    # --- energy minimization, then NVT from the structure it leaves ---
    call(ctx, "stage_prepare", stage="em")
    em_job = call(ctx, "stage_run", stage="em")
    em_done = wait_job(ctx, em_job.job_id)
    assert em_done.state == "COMPLETED", em_done.summary

    prepared = call(ctx, "stage_prepare", stage="nvt")
    assert "tpr" in prepared.files
    nvt_job = call(ctx, "stage_run", stage="nvt")
    nvt_done = wait_job(ctx, nvt_job.job_id)
    assert nvt_done.state == "COMPLETED", nvt_done.summary
    results = call(ctx, "job_status", job_id=nvt_job.job_id)
    roles = {a.role for a in results.artifacts}
    assert {"log", "energy", "trajectory"} <= roles

    # --- analysis: numbers come from the harness, sanity-checked here ---
    energy = call(
        ctx, "observable_measure",
        edr=next(a.path for a in results.artifacts if a.role == "energy"),
        terms=["Potential", "Temperature"],
    )
    temp = energy.terms["Temperature"]
    assert 250 < temp.mean < 350, f"temperature off: {temp.mean} K"
    pot = energy.terms["Potential"]
    assert pot.mean < 0, "solvated system potential must be negative"

    # topology from .gro, not .tpr: MDAnalysis lags new GROMACS tpx versions
    rmsd = call(
        ctx, "observable_measure",
        structure=next(a.path for a in results.artifacts if a.role == "structure"),
        trajectory=next(a.path for a in results.artifacts if a.role == "trajectory"),
        metric="rmsd", selection="protein",
    )
    assert rmsd.n_frames >= 2
    assert 0 <= rmsd.final < 5.0, f"RMSD suspicious: {rmsd.final} Å over 10 ps"

    # --- provenance: the ledger holds every engine command that ran ---
    events = list(ws.ledger.events())
    assert len(events) >= 12
    commands = " ".join(" ".join(ev.command) for ev in events if ev.command)
    for fragment in ("pdb2gmx", "solvate", "genion", "grompp", "mdrun"):
        assert fragment in commands
