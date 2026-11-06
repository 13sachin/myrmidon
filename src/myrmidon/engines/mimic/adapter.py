"""CPMD, the QM half of a coupled MiMiC run, as engine-blind code sees it."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from myrmidon.engines.mimic import collect, cpmd_input, diagnostics, layout, system, trajectory
from myrmidon.engines.registry import EngineAdapter, StageRender, StageText
from myrmidon.knowledge.sensors import evaluate


def _system_facts(files: Mapping[str, Path], _data_dir: Path | None,
                  rules: set[str] | None) -> dict:
    return system.system_facts(files.get("cpmd_input"), rules)


def _run_log(record) -> Path | None:
    stdout = Path(record.spec.cwd) / (record.spec.stdout or "")
    return stdout if stdout.is_file() and diagnostics.is_cpmd_output(stdout) else None


def _boundary_atoms(system_input: Path) -> bool:
    """Whether the selection behind this input marked boundary atoms, as its sidecar says."""
    try:
        sidecar = json.loads(system_input.with_name(system_input.name + ".myrmidon.json")
                             .read_text())
    except (OSError, ValueError):
        return False
    return bool(sidecar.get("boundary_atoms"))


def _stage_text(request: StageRender) -> StageText:
    """A stage's run input: its purpose and method on the system the QM selection made once,
    with the stage's charge and spin; the template alone while no selection has run."""
    from myrmidon.engines.file_origin import cpmd_values

    stated = cpmd_input.render_cpmd_from_values(request.values, request.header)
    region = request.stage.qm_region
    system_input = request.system.get("cpmd_input")
    if system_input is None or region is None:
        return StageText(stated)
    charge, multiplicity = region.charge, region.multiplicity
    sections = cpmd_input.apply_method(cpmd_input.parse_cpmd(system_input.read_text()),
                                       cpmd_input.parse_cpmd(stated))
    sections = cpmd_input.set_spin(sections, charge, multiplicity)
    bound = _boundary_atoms(system_input)
    electrons = None if bound else system.electron_count(sections, charge)
    findings = evaluate("qm_region", {
        "electrons": electrons, "multiplicity": multiplicity,
        "spin_parity_matches": None if electrons is None
        else electrons % 2 == (multiplicity - 1) % 2})
    text = cpmd_input.render_cpmd(sections)
    held = {" ".join(k.upper().split()) for k in request.values}
    return StageText(text, findings=findings, sidecar={
        "charge": charge, "multiplicity": multiplicity, "electrons": electrons,
        "system_input": system_input.name,
        "system_keys": sorted(k for k in cpmd_values(text)
                              if " ".join(k.upper().split()) not in held)})


ADAPTER = EngineAdapter(
    name="cpmd",
    file_kinds=("cpmd_input",),
    system_facts=_system_facts,
    renders=("cpmd_inp",),
    render=lambda values, _origin, header: cpmd_input.render_cpmd_from_values(values, header),
    stage_text=_stage_text,
    log_names=("cpmd.out",),
    reads_key=lambda key: key.startswith("&"),
    is_log_text=diagnostics.is_cpmd_text,
    is_log=diagnostics.is_cpmd_output,
    run_facts=diagnostics.cpmd_facts,
    fact_labels=diagnostics.RUN_FACTS,
    run_log=_run_log,
    resolve_layout=layout.resolve_layout,
    layout_facts=layout.layout_facts,
    stage_inputs=collect.stage_inputs,
    job_findings=collect.job_findings,
    size_walltime=collect.size_walltime,
    build_job=collect.build_job,
    diagnose=collect.diagnose_log,
    cut_snapshots=trajectory.cut_snapshots,
)
