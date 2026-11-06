"""GROMACS as engine-blind code sees it."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from myrmidon.engines.gromacs import layout, mdp, parse, run, system
from myrmidon.engines.gromacs.batch import collect_minimized
from myrmidon.engines.gromacs.collect import collect_run, diagnose_log
from myrmidon.engines.gromacs.collect import run_progress as mdrun_progress
from myrmidon.engines.registry import EngineAdapter
from myrmidon.execution.collect import collector, progress

collector("gromacs")(collect_run)
progress("gromacs")(mdrun_progress)
collector("minimize_frames")(collect_minimized)


def _system_facts(files: Mapping[str, Path], data_dir: Path | None,
                  rules: set[str] | None) -> dict:
    return system.system_facts(files.get("topology"), files.get("index"), files.get("structure"),
                               data_dir, rules)


def _system_parts(files: Mapping[str, Path], data_dir: Path | None) -> dict:
    top = files.get("topology")
    if top is None:
        return {}
    residuetypes = data_dir / "residuetypes.dat" if data_dir is not None else None
    return system.subsystems_of(top, [data_dir] if data_dir is not None else [], residuetypes)


def _run_log(record) -> Path | None:
    from myrmidon.engines.gromacs.collect import spec_deffnm

    log = Path(record.spec.cwd) / f"{spec_deffnm(record)}.log"
    return log if log.is_file() and parse.is_gromacs_log(log) else None


ADAPTER = EngineAdapter(
    name="gromacs",
    file_kinds=("topology", "index", "structure"),
    system_facts=_system_facts,
    system_parts=_system_parts,
    renders=("mdp",),
    render=lambda values, origin, header: mdp.render_mdp(values, origin, header),
    reads_key=lambda key: not key.startswith("&"),
    compile_run_input=run.compile_stage_run_input,
    is_log_text=parse.is_gromacs_text,
    is_log=parse.is_gromacs_log,
    run_facts=parse.run_facts,
    fact_labels=parse.GROMACS_FACTS,
    run_log=_run_log,
    resolve_layout=layout.resolve_layout,
    layout_facts=layout.layout_facts,
    stage_inputs=run.stage_inputs,
    job_findings=run.job_findings,
    size_walltime=run.size_walltime,
    build_job=run.build_job,
    diagnose=diagnose_log,
)
