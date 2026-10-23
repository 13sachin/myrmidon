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


def art(obs, role):
    return next(a.path for a in obs.artifacts if a.role == role)


def wait_job(ctx, job_id, timeout=2400):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = REGISTRY.call("job_status", ctx, {"job_id": job_id, "wait_seconds": 30})
        if st.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
    raise AssertionError(f"{job_id} did not finish within {timeout}s")


def test_acetone_qmmm_chain(ctx):
    call(ctx, "study_create", name="acetone-qmmm", description="real MiMiC integration")
    ws = ctx.workspace
    inputs = ws.stage("inputs")
    for f in FIXTURES.iterdir():
        shutil.copy(f, inputs / f.name)
    rel = lambda name: ws.relpath(inputs / name)  # noqa: E731

    # --- QM region via real prepqm (charge reported, not trusted) ---
    sel = call(
        ctx, "mimic_select_qm",
        topology=rel("acetone_equilibrated.top"), coords=rel("acetone_equilibrated.gro"),
        selections=["add resname is ACT"], template=rel("template.inp"),
        pp_info=rel("pp_info.dat"), padding_nm=0.4,
    )
    assert sel.qm_atoms == 10
    assert sel.min_vacuum_bohr is not None and sel.min_vacuum_bohr > 7  # 0.4 nm = 7.56 bohr
    assert sel.declared_charge == 0
    assert any(w.id == "QM_CHARGE_UNCONFIRMED" for w in sel.warnings)

    # --- tpr through the gate, with the QM index group ---
    tpr = call(
        ctx, "gmx_grompp", mdp=rel("mimic.mdp"), structure=rel("acetone_equilibrated.gro"),
        topology=rel("acetone_equilibrated.top"), index=art(sel, "index"),
        output="mimic.tpr", stage="qmmm", max_warn=1,
    )

    # --- explicit charge/multiplicity; short run ---
    prep = call(
        ctx, "mimic_prepare", cpmd_input=art(sel, "cpmd_input"),
        total_charge=0, multiplicity=1, maxsteps=MAXSTEPS,
        acknowledge_warnings=["QM_CHARGE_MULTIPLICITY_UNCONFIRMED"],
    )
    assert prep.maxsteps == MAXSTEPS and prep.paths == 1
    report = call(ctx, "file_check", path=art(prep, "cpmd_input"), tpr=art(tpr, "tpr"))
    assert not report.blocking and not report.unacknowledged, report.summary

    # --- coupled launch ---
    job = call(
        ctx, "mimic_run", cpmd_input=art(prep, "cpmd_input"), tpr=art(tpr, "tpr"),
        cpmd_ranks=CPMD_RANKS, gromacs_ranks=1,
    )
    done = wait_job(ctx, job.job_id)
    assert done.state == "COMPLETED", f"{done.summary}\n{done.log_tail or ''}"

    # --- diagnostics are parsed from CPMD output, never asserted by the LLM ---
    diag = call(ctx, "mimic_diagnose")
    assert diag.finished_normally
    assert diag.md_steps_completed == MAXSTEPS
    assert diag.energy_last is not None and diag.energy_last < -30  # acetone/BLYP ≈ -36 a.u.
    stage = ws.stage("qmmm")
    assert (stage / "mimic.log").exists() and (stage / "ENERGIES").exists()

    events = list(ws.ledger.events())
    tools = {e.tool for e in events}
    assert tools >= {"mimic_select_qm", "gmx_grompp", "mimic_prepare", "mimic_run"}
