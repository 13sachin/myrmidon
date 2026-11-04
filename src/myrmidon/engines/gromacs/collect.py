"""A GROMACS run as a job: the prefix its outputs carry, what it shows while it runs, and
what it returns once it has ended."""

from __future__ import annotations

import re
from pathlib import Path

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation, RunProgress
from myrmidon.core.tool import ToolContext
from myrmidon.engines.gromacs.parse import (
    ROLE_BY_SUFFIX,
    TIME_LIMIT,
    parse_nsteps,
    parse_progress,
    walltime_diagnosis,
)
from myrmidon.execution.backend import JobState
from myrmidon.execution.jobstore import JobRecord
from myrmidon.text import LOG_TAIL_LINES, tail


def spec_deffnm(record: JobRecord) -> str:
    """The prefix this run's output files carry.

    What the submitting tool recorded comes first. A coupled MiMiC launch puts CPMD in
    `spec.argv` and GROMACS in a partner command, so the argv holds no `-deffnm`; the
    partner's own argv is checked before the default, or collection would report success
    having found none of the MM outputs.
    """
    recorded = record.meta.get("deffnm")
    if recorded:
        return str(recorded)
    argv = record.spec.argv
    if "-deffnm" in argv:
        return argv[argv.index("-deffnm") + 1]
    for partner in record.spec.coupled or []:
        if "-deffnm" in partner.argv:
            return partner.argv[partner.argv.index("-deffnm") + 1]
    return "md"


def run_progress(record: JobRecord) -> tuple[RunProgress | None, str]:
    """mdrun's own counter for a run in flight, and the line that says it."""
    log = Path(record.spec.cwd) / f"{spec_deffnm(record)}.log"
    progress = parse_progress(log)
    if progress is None:
        return None, ""
    bits = []
    if progress.step is not None:
        asked = parse_nsteps(log.read_text(errors="replace"))
        bits.append(f"step {progress.step}" + (f" of {asked}" if asked else ""))
    if progress.time_ps is not None:
        bits.append(f"{progress.time_ps} ps")
    if progress.ns_per_day is not None:
        bits.append(f"{progress.ns_per_day} ns/day")
    return progress, " · ".join(bits)


def collect_run(ctx: ToolContext, record: JobRecord, state: JobState,
                    exit_code: int | None) -> Observation:
    """A GROMACS run that has ended: its output files by role, or, when it failed, the
    tail of its log and stderr for the reader to diagnose, and the scheduler's word when
    it was a walltime kill."""
    job_id = record.job_id
    cwd = Path(record.spec.cwd)
    deffnm = spec_deffnm(record)
    outputs = sorted(cwd.glob(f"{deffnm}*"))
    artifacts = [
        ArtifactRef.from_path(p, ROLE_BY_SUFFIX[p.suffix], root=ctx.workspace.root)
        for p in outputs
        if p.suffix in ROLE_BY_SUFFIX
    ]
    log_text = ""
    log_path = cwd / f"{deffnm}.log"
    if log_path.exists():
        log_text += log_path.read_text(errors="replace")
    stderr_path = cwd / record.spec.stderr
    if stderr_path.exists():
        log_text += "\n" + stderr_path.read_text(errors="replace")

    if state is JobState.COMPLETED:
        progress = parse_progress(log_path)
        from myrmidon.execution.pace import gromacs_pace, record_pace

        pace = gromacs_pace(log_path, record.spec.resources,
                            f"{ctx.workspace.root.name} {record.label or job_id}")
        if pace is not None:
            record_pace(ctx.config.studies_root, pace)
        return Observation(
            ok=True,
            summary=f"{job_id} completed: {len(artifacts)} output artifact(s). "
            "Analyze with gmx_energy / traj_analyze — do not report values not present "
            "in their observations.",
            job_id=job_id,
            state=state.value,
            exit_code=exit_code,
            progress=progress,
            artifacts=artifacts,
        )

    diagnosis = None
    if state is JobState.TIMEOUT:
        from myrmidon.execution.pace import record_pace, stopped_pace

        pace = stopped_pace(log_path, record.spec.resources,
                            f"{ctx.workspace.root.name} {record.label or job_id}")
        if pace is not None:
            record_pace(ctx.config.studies_root, pace)
    if state is JobState.TIMEOUT or re.search(TIME_LIMIT, log_text):
        diagnosis = walltime_diagnosis(log_path, log_text)
    tail_lines = tail(log_text, LOG_TAIL_LINES) if log_text else None
    return Observation(
        ok=False,
        summary=f"{job_id} {state.value.lower()}"
        + (f": {diagnosis}" if diagnosis
           else ": read the engine's own words in log_tail" if tail_lines
           else " and left no log"),
        job_id=job_id,
        state=state.value,
        exit_code=exit_code,
        diagnosis=diagnosis,
        artifacts=artifacts,
        log_tail=tail_lines,
    )
