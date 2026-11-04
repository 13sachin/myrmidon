"""How fast runs go on this machine, measured: what a stage is priced at.

A completed GROMACS run's log states its atoms and its ns/day. The study's own runs are
the first measure (the same system, the same booking); every run collected anywhere on the
machine is kept in ``studies_root/.pace.json``, the measure for a study that has run
nothing yet, scaled by atoms. Nothing here is a benchmark typed into the repository.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from myrmidon.knowledge.stage import Pace

PACE_FILE = ".pace.json"
#: the most recent measurements kept per engine and booking
KEEP = 20


def _path(studies_root) -> Path:
    return Path(studies_root).expanduser() / PACE_FILE


def read_paces(studies_root) -> list[Pace]:
    """The machine's measurements, most recent first; none when the cache is unreadable."""
    try:
        rows = json.loads(_path(studies_root).read_text())
        return [Pace.model_validate(r) for r in reversed(rows)]
    except (OSError, ValueError):
        return []


def record_pace(studies_root, pace: Pace) -> None:
    """Add one measurement to the machine's cache, keeping the latest per booking."""
    path = _path(studies_root)
    try:
        rows = json.loads(path.read_text()) if path.is_file() else []
    except (OSError, ValueError):
        rows = []
    # a run collected again is measured once
    rows = [r for r in rows if not (pace.where and r.get("where") == pace.where)]
    rows.append(pace.model_dump(exclude_none=True))
    same = [r for r in rows if (r.get("engine"), r.get("cores"), r.get("partition"))
            == (pace.engine, pace.cores, pace.partition)]
    drop = same[:-KEEP] if len(same) > KEEP else []
    rows = [r for r in rows if r not in drop]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rows, indent=1))
        tmp.replace(path)
    except OSError:
        pass  # a cache that cannot be written prices the next study cold, nothing worse


def gromacs_pace(log: Path, resources, where: str) -> Pace | None:
    """The pace a completed mdrun log states, for the booking it ran on."""
    from myrmidon.engines.gromacs.parse import run_facts

    try:
        facts = run_facts(log.read_text(errors="replace"))
    except OSError:
        return None
    if not facts.get("atoms") or not facts.get("performance_ns_per_day"):
        return None
    return Pace(engine="gromacs", atoms=facts["atoms"],
                cores=resources.ntasks * resources.cpus_per_task,
                partition=resources.partition or "", ns_per_day=facts["performance_ns_per_day"],
                where=where)


def cpmd_pace(cpmd_out: Path, run_input: Path | None, resources, where: str) -> Pace | None:
    """The seconds per QM/MM step a coupled run's CPMD log states, for the QM atoms its run
    input holds and the booking it ran on."""
    from myrmidon.engines.mimic.cpmd_input import keyword_value, parse_cpmd, qm_geometry
    from myrmidon.engines.mimic.diagnostics import seconds_per_step

    if run_input is None:
        return None
    try:
        seconds = seconds_per_step(cpmd_out.read_text(errors="replace"))
        sections = parse_cpmd(run_input.read_text())
    except OSError:
        return None
    geometry, cutoff = qm_geometry(sections), keyword_value(sections, "&SYSTEM/CUTOFF")
    if seconds is None or not geometry:
        return None  # a step is priced by the QM region it was measured on
    try:
        cutoff_ry = float(cutoff) if cutoff else None
    except ValueError:
        cutoff_ry = None
    return Pace(engine="cpmd", atoms=geometry["n_atoms"],
                cores=resources.ntasks * resources.cpus_per_task,
                partition=resources.partition or "", s_per_step=seconds, cutoff_ry=cutoff_ry,
                where=where)


#: where an mdrun log begins a job's own segment: a fresh start, or a continuation's
SEGMENT_START = re.compile(r"^(Reading checkpoint file|Started mdrun|Input Parameters:)",
                           re.MULTILINE)
STEP_TIME = re.compile(r"^\s+Step\s+Time\s*\n\s*\d+\s+([\d.eE+-]+)", re.MULTILINE)


def stopped_pace(log: Path, resources, where: str) -> Pace | None:
    """The pace a run stopped at its walltime reached: the simulated time this job's own
    segment covered — from the first step block after its start, which a continuation's
    log appends after its parent's — over the walltime it was booked, a slight
    underestimate (start-up counts as running)."""
    from myrmidon.engines.gromacs.parse import run_facts

    try:
        text = log.read_text(errors="replace")
    except OSError:
        return None
    facts = run_facts(text)
    if "minimisation_steps" in facts:
        return None  # a minimiser's "Time" counts steps: it measures no dynamics pace
    starts = list(SEGMENT_START.finditer(text))
    times = [float(t) for t in STEP_TIME.findall(text[starts[-1].start() if starts else 0:])]
    covered = times[-1] - times[0] if len(times) > 1 else 0.0
    if not facts.get("atoms") or covered <= 0 or not resources.walltime_min:
        return None
    return Pace(engine="gromacs", atoms=facts["atoms"],
                cores=resources.ntasks * resources.cpus_per_task,
                partition=resources.partition or "",
                ns_per_day=covered / 1000 / (resources.walltime_min / 1440), where=where)


def study_paces(ws) -> list[Pace]:
    """The paces this study's completed runs measured, most recent first."""
    from myrmidon.engines.gromacs.collect import spec_deffnm
    from myrmidon.execution.backend import JobState
    from myrmidon.execution.jobs import existing_jobstore

    store = existing_jobstore(ws)  # a study that has run nothing has measured nothing
    out = []
    for rec in reversed(store.list() if store is not None else []):
        kind = rec.meta.get("kind", "gromacs")
        stopped = rec.state is JobState.TIMEOUT and kind == "gromacs"
        if not (rec.state is JobState.COMPLETED and kind in ("gromacs", "mimic") or stopped):
            continue
        where = f"{ws.root.name} {rec.label or rec.job_id}"
        cwd = Path(rec.spec.cwd)
        log = cwd / f"{spec_deffnm(rec)}.log"
        pace = (stopped_pace(log, rec.spec.resources, where) if stopped else
                gromacs_pace(log, rec.spec.resources, where) if kind == "gromacs" else
                cpmd_pace(cwd / "cpmd.out", cpmd_run_input(rec), rec.spec.resources, where))
        if pace is not None:
            out.append(pace)
    return out


def cpmd_run_input(rec) -> Path | None:
    """The CPMD input a coupled job ran: the argument after its executable."""
    argv = rec.spec.argv
    found = Path(rec.spec.cwd) / argv[1] if len(argv) > 1 else None
    return found if found is not None and found.is_file() else None
