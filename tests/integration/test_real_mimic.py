"""Integration test against real MiMiC: CPMD + MiMiC-enabled GROMACS + mimicpy.

Acetone (10 QM atoms, BLYP, 70 Ry) in 506 SPC waters — the MiMiCPy tutorial
system. select QM region → gated grompp with index → reviewed CPMD input →
validate → coupled launch → diagnose. Auto-skipped unless all three engines are
discovered. On SLURM the workspace lives under the configured studies_root.
"""

import os
import shutil
import time
import uuid
from pathlib import Path

import pytest
from tests.unit.findings_catalog import blocking

from myrmidon.config import MyrmidonConfig
from myrmidon.core.tool import REGISTRY, ToolContext
from myrmidon.engines.discovery import discover
from myrmidon.server.mcp_server import build_backend

CONFIG = MyrmidonConfig.load()
ENGINES = discover(CONFIG)
FIXTURES = Path(__file__).parent.parent / "fixtures" / "acetone"
MAXSTEPS = 20
# Unpatched CPMD 4.3 traps (propcal MOD(nfi, 0)) on every non-I/O rank after the first MD
# step; the st26_fm_gcc build carries the propin broadcast fix. Set to 1 on unpatched builds.
CPMD_RANKS = int(os.environ.get("MYRMIDON_TEST_CPMD_RANKS", "4"))

pytestmark = [
    pytest.mark.mimic,
    pytest.mark.timeout(3600),
    pytest.mark.skipif(
        not all(ENGINES[e].available for e in ("gromacs", "cpmd", "mimicpy"))
        or "mock" in ENGINES["gromacs"].version,
        reason="real CPMD + GROMACS + mimicpy not all available",
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


def art(obs, role):
    return next(a.path for a in obs.artifacts if a.role == role)


def wait_job(ctx, job_id, timeout=2400):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = REGISTRY.call("job_status", ctx, {"job_id": job_id, "wait_seconds": 30})
        if st.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
    raise AssertionError(f"{job_id} did not finish within {timeout}s")


def test_acetone_qmmm_chain(ctx, approve_mandate):
    call(ctx, "study_create", name="acetone-qmmm", description="real MiMiC integration")
    ws = ctx.workspace
    inputs = ws.stage("inputs")
    for f in FIXTURES.iterdir():
        if f.is_dir():
            shutil.copytree(f, inputs / f.name)
        else:
            shutil.copy(f, inputs / f.name)
    rel = lambda name: ws.relpath(inputs / name)  # noqa: E731

    # --- QM region via real prepqm: what the selection leaves of its chemistry ---
    sel = call(
        ctx, "qm_region",
        topology=rel("acetone_equilibrated.top"), coords=rel("acetone_equilibrated.gro"),
        selections=["add resname is ACT"], pp_info=rel("pp_info.dat"), padding_nm=0.4,
    )
    assert sel.qm_atoms == 10
    assert sel.min_vacuum_bohr is not None and sel.min_vacuum_bohr > 7  # 0.4 nm = 7.56 bohr
    assert sel.declared_charge == 0
    assert (sel.derived_multiplicity, sel.electrons) == (1, 24)
    assert sel.guessed_elements["c3"] == "C" and "PP_LMAX_ALL_S_ON_HEAVY_ATOMS" not in {
        f.id for f in sel.findings}
    assert {a.role for a in sel.artifacts} >= {"cpmd_input", "index", "qm_selection"}

    approve_mandate(
        ctx, "Run 20 coupled QM/MM steps of acetone in water", 128.0,
        stages=[{"name": "cool", "kind": "minimize", "regime": "qmmm", "steps": MAXSTEPS,
                 "temperature_k": 300, "qm_region": {"subsystems": ["solute"], "charge": 0}}],
        hamiltonian={"mm": "amber", "qm": {"method": "blyp_mt", "cutoff_ry": 70},
                     "coupling": {"scheme": "mimic"}},
        subsystems={"solute": {"kind": "ligand", "resnames": ["ACT"]},
                    "solvent": {"kind": "solvent", "molecules": ["SOL"]}},
        structure=rel("acetone_equilibrated.gro"), topology=rel("acetone_equilibrated.top"))

    # --- one call: the complete run input and its compiled MM half, through grompp ---
    prepared = call(ctx, "stage_prepare", stage="cool")
    assert {"cpmd_input", "mdp", "tpr"} <= set(prepared.files)
    report = call(ctx, "file_check", path=prepared.files["cpmd_input"],
                  tpr=prepared.files["tpr"])
    assert not blocking(report), report.summary

    # --- coupled launch ---
    job = call(ctx, "stage_run", stage="cool", cores=CPMD_RANKS + 1)
    done = wait_job(ctx, job.job_id)
    assert done.state == "COMPLETED", f"{done.summary}\n{done.log_tail or ''}"

    # --- diagnostics are parsed from CPMD output, never asserted by the LLM ---
    diag = call(ctx, "run_diagnose", stage="cool")
    assert diag.finished_normally
    assert diag.md_steps_completed == MAXSTEPS
    assert diag.energy_last is not None and diag.energy_last < -30  # acetone/BLYP ≈ -36 a.u.
    run_dir = next(ws.stage("cool").glob("*_cool"))
    assert (run_dir / "cool.log").exists() and (run_dir / "ENERGIES").exists()

    # --- a matched snapshot: both engines write at the stage's one stride ---
    snapshot = call(ctx, "snapshot_select", stage="cool", n_frames=1)
    assert {"geometry", "structure"} <= {a.role for a in snapshot.artifacts}

    events = list(ws.ledger.events())
    tools = {e.tool for e in events}
    assert tools >= {"qm_region", "stage_prepare", "stage_run"}
