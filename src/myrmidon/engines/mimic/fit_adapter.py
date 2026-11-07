"""MiMiCPy's force-matching fit, the engine of a parameter_fit stage, as engine-blind code
sees it."""

from __future__ import annotations

from myrmidon.core.validation import Check
from myrmidon.engines.gromacs.itp import BONDED_N_ATOMS, parse_itp_atoms, parse_itp_bonded
from myrmidon.engines.mimic import fit_collect, fm, fm_dataset
from myrmidon.engines.registry import EngineAdapter, FitReaders, StageRender, StageText

FEW_SNAPSHOTS = Check(id="FM_FEW_SNAPSHOTS", sources=(
    "https://mimic-project.org/en/latest/tutorials/acetone_tutorial.html",))


def _stage_text(request: StageRender) -> StageText:
    """A fit's input, and whether its stride leaves it too few snapshots to fit."""
    n = fm.stride_count(request.values)
    findings = [FEW_SNAPSHOTS.finding(
        f"the fit's stride selects only {n} snapshot(s)",
        means="a parameter fit on fewer than 100 snapshots overfits them and validates poorly",
        suggestion="widen the stride, or collect more reference frames")] if (
        n is not None and n < 100) else []
    return StageText(fm.render_fminput(request.values, request.origin, request.header),
                     findings=findings)


ADAPTER = EngineAdapter(
    name="mimicpy",
    renders=("fminput",),
    render=lambda values, origin, header: fm.render_fminput(values, origin, header),
    stage_text=_stage_text,
    reads_key=lambda key: not key.startswith("&"),
    input_facts=fm.dataset_facts,
    resolve_layout=fit_collect.resolve_layout,
    layout_facts=fit_collect.layout_facts,
    stage_inputs=fit_collect.stage_inputs,
    job_findings=fit_collect.job_findings,
    size_walltime=fit_collect.size_walltime,
    build_job=fit_collect.build_job,
    collect_dataset=fm_dataset.collect_dataset,
    log_names=("fm.log",),
    is_log_text=lambda text: "Running Force matching in MiMiCPy" in text[:16384],
    is_log=lambda path: path.name == "fm.log" and path.is_file(),
    run_facts=fm.parse_fm_log,
    diagnose=fit_collect.diagnose_log,
    force_comparison=fit_collect.force_comparison,
    run_reads=fit_collect.FIT_INPUTS,
    checks=(FEW_SNAPSHOTS,),
    fit_readers=FitReaders(
        parse_fm_log=fm.parse_fm_log, parse_fmcheck_report=fm.parse_fmcheck_report,
        parse_optimization_results=fm.parse_optimization_results,
        parse_validation_force_residuals=fm.parse_validation_force_residuals,
        residuals_file=fm.VALIDATION_RESIDUALS_FILENAME, force_type_table=fm.force_type_table,
        unphysical_force_constants=fm.unphysical_force_constants,
        parse_itp_atoms=parse_itp_atoms, parse_itp_bonded=parse_itp_bonded,
        bonded_n_atoms=BONDED_N_ATOMS),
)
