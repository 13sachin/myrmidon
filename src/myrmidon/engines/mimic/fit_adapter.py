"""MiMiCPy's force-matching fit, the engine of a parameter_fit stage, as engine-blind code
sees it."""

from __future__ import annotations

from myrmidon.engines.mimic import fit_collect, fm
from myrmidon.engines.registry import EngineAdapter, StageRender, StageText
from myrmidon.knowledge.sensors import evaluate


def _stage_text(request: StageRender) -> StageText:
    """A fit's input, and what the sensor records make of how many snapshots it fits."""
    return StageText(
        fm.render_fminput(request.values, request.origin, request.header),
        findings=evaluate("fm_input", {"n_snapshots": fm.stride_count(request.values)}))


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
)
