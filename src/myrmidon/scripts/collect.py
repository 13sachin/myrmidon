"""Collect a finished script job: declared outputs hashed, failure diagnosed.

Shared by ``script_run`` (when it waits) and ``job_status`` (which collects a job
the first time it sees it ended) so a script job
is collected identically whichever tool the agent reaches for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.execution.backend import JobState
from myrmidon.execution.jobstore import JobRecord
from myrmidon.text import tail

_GLOB_CHARS = set("*?[")

_DIAGNOSES: list[tuple[str, str]] = [
    (r"Traceback \(most recent call last\)", "Python exception — see the last log line"),
    (r"DUE TO TIME LIMIT", "SLURM killed the job at its walltime — raise walltime_min"),
    (r"Out Of Memory|MemoryError|oom-kill", "out of memory"),
    (r"command not found|No such file or directory", "missing executable or path"),
    (r"ModuleNotFoundError|ImportError", "Python import failed — wrong interpreter/env?"),
]


@dataclass
class ScriptCollection:
    artifacts: list[ArtifactRef] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    log_tail: str | None = None
    diagnosis: str | None = None


def _tail(path: Path, n: int = 20) -> str:
    return tail(path.read_text(errors="replace"), n) if path.exists() else ""


def collect_script_job(record: JobRecord, root: Path) -> ScriptCollection:
    cwd = Path(record.spec.cwd)
    out = ScriptCollection()
    for pattern in record.meta.get("outputs", []):
        if _GLOB_CHARS & set(pattern):
            # pathlib refuses an absolute glob ("Non-relative patterns are unsupported");
            # an output declared with an absolute path (a script whose cwd is outside the
            # workspace) is globbed from its own anchor instead
            pat = Path(pattern)
            if pat.is_absolute():
                base, rel = Path(pat.anchor), str(pat.relative_to(pat.anchor))
            else:
                base, rel = cwd, pattern
            matches = sorted(p for p in base.glob(rel) if p.is_file())
        else:
            candidate = cwd / pattern
            matches = [candidate] if candidate.is_file() else []
        if not matches:
            out.missing.append(pattern)
        out.artifacts += [ArtifactRef.from_path(p, "output", root=root) for p in matches]

    stdout_tail = _tail(cwd / record.spec.stdout)
    stderr_tail = _tail(cwd / record.spec.stderr)
    text = stdout_tail + "\n" + stderr_tail
    if record.exit_code not in (0, None) or out.missing:
        out.log_tail = "\n".join(s for s in (stdout_tail, stderr_tail) if s) or None
        for pattern, diagnosis in _DIAGNOSES:
            if re.search(pattern, text):
                out.diagnosis = diagnosis
                if diagnosis.startswith("Python exception"):
                    last = [ln for ln in stderr_tail.splitlines() if ln.strip()]
                    if last:
                        out.diagnosis = f"Python exception: {last[-1].strip()[:200]}"
                break
    return out


def script_observation(label: str, state: JobState, exit_code: int | None,
                       collected: ScriptCollection, done: str, /, **fields):
    """A script job that has ended, as one observation: ``done`` when it completed with every
    declared output, else what failed — the same for the submission that waited and for
    the job_status that collected it."""
    from myrmidon.core.observation import ScriptObservation

    if state == JobState.COMPLETED and not collected.missing:
        return ScriptObservation(ok=True, summary=done, artifacts=collected.artifacts, **fields)
    why = (
        f"{state.value.lower()} (exit {exit_code})" if state != JobState.COMPLETED
        else f"completed but {len(collected.missing)} declared output(s) missing: "
        f"{collected.missing}"
    )
    return ScriptObservation(
        ok=False,
        summary=f"{label} {why}" + (f" — {collected.diagnosis}" if collected.diagnosis else ""),
        artifacts=collected.artifacts, outputs_missing=collected.missing,
        diagnosis=collected.diagnosis, log_tail=collected.log_tail, **fields,
    )
