"""CPMD, the QM half of a coupled MiMiC run, as engine-blind code sees it."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from myrmidon.engines.mimic import diagnostics, system
from myrmidon.engines.registry import EngineAdapter


def _system_facts(files: Mapping[str, Path], _data_dir: Path | None,
                  rules: set[str] | None) -> dict:
    return system.system_facts(files.get("cpmd_input"), rules)


def _run_log(record) -> Path | None:
    stdout = Path(record.spec.cwd) / (record.spec.stdout or "")
    return stdout if stdout.is_file() and diagnostics.is_cpmd_output(stdout) else None


ADAPTER = EngineAdapter(
    name="cpmd",
    file_kinds=("cpmd_input",),
    system_facts=_system_facts,
    is_log_text=diagnostics.is_cpmd_text,
    is_log=diagnostics.is_cpmd_output,
    run_facts=diagnostics.cpmd_facts,
    fact_labels=diagnostics.RUN_FACTS,
    run_log=_run_log,
)
