"""Generate a clean trr/h5 force-matching reference pair via the harness.

One MiMiC BO-MD run with CPMD FORCE_MATCHING pace 1 and GROMACS nstxout=1 writes
FMTRAJECTORY.json and mimic.trr in lockstep; fm_collect converts, aligns and
registers the pair; a quick fm_fit proves it is fit-ready. Opt-in (a real
multi-minute QM/MM run): set MYRMIDON_FM_GENERATE=1. Steps via
MYRMIDON_FM_STEPS (default 300), CPMD ranks via MYRMIDON_TEST_CPMD_RANKS.
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


def wait_job(ctx, job_id, timeout=3000):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = REGISTRY.call("job_status", ctx, {"job_id": job_id, "wait_seconds": 60})
        if st.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
    raise AssertionError(f"{job_id} did not finish within {timeout}s")


def test_generate_reference_pair(ctx):
    call(ctx, "study_create", name="acetone-fm-ref",
         description=f"clean FM reference pair, {STEPS} steps")
    ws = ctx.workspace
    inputs = ws.stage("inputs")
    for f in FIXTURES.iterdir():
        shutil.copy(f, inputs / f.name)
    rel = lambda name: ws.relpath(inputs / name)  # noqa: E731

    sel = call(
        ctx, "mimic_select_qm",
        topology=rel("acetone_equilibrated.top"), coords=rel("acetone_equilibrated.gro"),
        selections=["add resname is ACT"], template=rel("fm_template.inp"),
        pp_info=rel("pp_info.dat"), padding_nm=0.4,
    )
    tpr = call(
        ctx, "gmx_grompp", mdp=rel("mimic_fm.mdp"), structure=rel("acetone_equilibrated.gro"),
        topology=rel("acetone_equilibrated.top"), index=art(sel, "index"),
        output="mimic.tpr", stage="qmmm", max_warn=2,
    )
    prep = call(
        ctx, "mimic_prepare", cpmd_input=art(sel, "cpmd_input"),
        total_charge=0, multiplicity=1, maxsteps=STEPS,
        acknowledge_warnings=["QM_CHARGE_MULTIPLICITY_UNCONFIRMED"],
    )
    report = call(ctx, "file_check", path=art(prep, "cpmd_input"), tpr=art(tpr, "tpr"))
    assert not report.blocking, report.summary

    job = call(
        ctx, "mimic_run", cpmd_input=art(prep, "cpmd_input"), tpr=art(tpr, "tpr"),
        cpmd_ranks=CPMD_RANKS, gromacs_ranks=1,
    )
    done = wait_job(ctx, job.job_id)
    assert done.state == "COMPLETED", f"{done.summary}\n{done.log_tail or ''}"
    diag = call(ctx, "mimic_diagnose")
    assert diag.md_steps_completed == STEPS

    qmmm = ws.stage("qmmm")
    assert (qmmm / "FMTRAJECTORY.json").exists() and (qmmm / "mimic.trr").exists()

    ds = call(
        ctx, "fm_collect",
        fmdata=ws.relpath(qmmm / "FMTRAJECTORY.json"), trajectory=ws.relpath(qmmm / "mimic.trr"),
        topology=rel("acetone_equilibrated.top"), coords=rel("acetone_equilibrated.gro"),
        selection=ws.relpath(_write_sele(inputs)),
    )
    # a lockstep run must lose at most the trivial first-frame offset
    assert ds.qm_atoms == 10
    assert ds.frames >= STEPS - 2, ds.summary
    assert not ds.dropped or ds.dropped <= 2

    plan = call(ctx, "fm_plan", stride=[0, ds.frames, 1], qm_total_charge=0,
                dataset_frames=ds.frames)
    dataset = {a.role: str(ws.root / a.path) for a in ds.artifacts}
    fit = call(
        ctx, "fm_fit",
        topology=rel("acetone_equilibrated.top"), coords=rel("acetone_equilibrated.gro"),
        fmdata=dataset["fm_dataset"], trajectory=dataset["trajectory"],
        index=art(sel, "index"), selection=ws.relpath(inputs / "sele.dat"),
        fm_input=plan.artifacts[0].path, mdp=rel("mimic_fm.mdp"), n_processes=12,
    )
    fit_done = wait_job(ctx, fit.job_id)
    assert fit_done.state == "COMPLETED", f"{fit_done.summary}\n{fit_done.log_tail or ''}"
    res = call(ctx, "fm_results", original_itp=rel("acetone.itp"))
    assert res.optimizer_success is True
    assert res.n_snapshots == plan.n_snapshots

    print(f"\nREFERENCE PAIR: {dataset['fm_dataset']}\n                {dataset['trajectory']}")


def _write_sele(inputs: Path) -> Path:
    sele = inputs / "sele.dat"
    if not sele.exists():
        sele.write_text("add resname is ACT\n")
    return sele
