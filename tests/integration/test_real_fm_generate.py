"""Generate a clean trr/h5 force-matching reference pair via the harness.

One MiMiC BO-MD run with CPMD FORCE_MATCHING pace 1 and GROMACS nstxout=1 writes
FMTRAJECTORY.json and mimic.trr in lockstep; dataset_collect converts, aligns and
registers the pair; a quick fit proves it is fit-ready. Opt-in (a real
multi-minute QM/MM run): set MYRMIDON_FM_GENERATE=1. Steps via
MYRMIDON_FM_STEPS (default 300), CPMD ranks via MYRMIDON_TEST_CPMD_RANKS.
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
STEPS = int(os.environ.get("MYRMIDON_FM_STEPS", "300"))
CPMD_RANKS = int(os.environ.get("MYRMIDON_TEST_CPMD_RANKS", "16"))

pytestmark = [
    pytest.mark.mimic,
    pytest.mark.timeout(3600),
    pytest.mark.skipif(
        os.environ.get("MYRMIDON_FM_GENERATE") != "1",
        reason="reference-data generation is opt-in: set MYRMIDON_FM_GENERATE=1",
    ),
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


def wait_job(ctx, job_id, timeout=3000):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = REGISTRY.call("job_status", ctx, {"job_id": job_id, "wait_seconds": 60})
        if st.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
    raise AssertionError(f"{job_id} did not finish within {timeout}s")


def test_generate_reference_pair(ctx, approve_mandate):
    call(ctx, "study_create", name="acetone-fm-ref",
         description=f"clean FM reference pair, {STEPS} steps")
    ws = ctx.workspace
    inputs = ws.stage("inputs")
    for f in FIXTURES.iterdir():
        if f.is_dir():
            shutil.copytree(f, inputs / f.name)
        else:
            shutil.copy(f, inputs / f.name)
    rel = lambda name: ws.relpath(inputs / name)  # noqa: E731

    sel = call(
        ctx, "qm_region",
        topology=rel("acetone_equilibrated.top"), coords=rel("acetone_equilibrated.gro"),
        selections=["add resname is ACT"], pp_info=rel("pp_info.dat"), padding_nm=0.4,
    )
    region = {"subsystems": ["solute"], "charge": sel.declared_charge}
    # a frame each step, so each reference record pairs with a GROMACS frame
    approve_mandate(
        ctx, "Generate a force-matching reference pair, and fit it", 320.0,
        stages=[{"name": "reference", "kind": "dynamics", "regime": "qmmm", "ensemble": "NVE",
                 "steps": STEPS, "sampling_ps": 2.418884326585747e-4, "qm_region": region},
                {"name": "rerun", "kind": "single_point", "regime": "mm"},
                {"name": "fit", "kind": "parameter_fit", "qm_region": region}],
        hamiltonian={"mm": "amber", "qm": {"method": "blyp_mt", "cutoff_ry": 70},
                     "coupling": {"scheme": "mimic"}},
        subsystems={"solute": {"kind": "ligand", "resnames": ["ACT"]},
                    "solvent": {"kind": "solvent", "molecules": ["SOL"]}},
        structure=rel("acetone_equilibrated.gro"), topology=rel("acetone_equilibrated.top"))
    prepared = call(ctx, "stage_prepare", stage="reference", overrides={
        "&CPMD/FORCE_MATCHING": 1, "nstfout": 1})
    report = call(ctx, "file_check", path=prepared.files["cpmd_input"],
                  tpr=prepared.files["tpr"])
    assert not blocking(report), report.summary

    job = call(ctx, "stage_run", stage="reference", cores=CPMD_RANKS + 1)
    done = wait_job(ctx, job.job_id)
    assert done.state == "COMPLETED", f"{done.summary}\n{done.log_tail or ''}"
    diag = call(ctx, "run_diagnose", stage="reference")
    assert diag.md_steps_completed == STEPS

    ds = call(ctx, "dataset_collect", stage="reference")
    # a lockstep run must lose at most the trivial first-frame offset
    assert ds.qm_atoms == 10
    assert ds.frames >= STEPS - 2, ds.summary
    assert not ds.dropped or ds.dropped <= 2

    fitting = call(ctx, "stage_prepare", stage="fit")
    assert fitting.files["fm_input"].endswith("fminput.dat")
    # the bonded target's classical rerun: an mdp of its own, never the coupled run's
    call(ctx, "stage_prepare", stage="rerun", structure=rel("acetone_equilibrated.gro"))
    fit = call(ctx, "stage_run", stage="fit")
    fit_done = wait_job(ctx, fit.job_id)
    assert fit_done.state == "COMPLETED", f"{fit_done.summary}\n{fit_done.log_tail or ''}"
    assert fit_done.optimizer_success is True
    assert fit_done.n_snapshots == ds.frames
    print(f"\nREFERENCE PAIR: {[a.path for a in ds.artifacts]}")
