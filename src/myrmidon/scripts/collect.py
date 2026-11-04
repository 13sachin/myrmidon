"""Collect a finished script: declared outputs hashed, results read, failure diagnosed.

Shared by ``script_run`` (a script run in the session, or a job it waited for) and
``job_status`` (which collects a job the first time it sees it ended) so a script is
collected identically whichever tool the agent reaches for.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    results: dict[str, Any] = field(default_factory=dict)  # an analysis script's numbers
    log_tail: str | None = None
    diagnosis: str | None = None
    looked_in: list[str] = field(default_factory=list)  # where the outputs were looked for


def _tail(path: Path, n: int = 20) -> str:
    return tail(path.read_text(errors="replace"), n) if path.exists() else ""


def _read_results(cwd: Path, name: str, out: ScriptCollection, root: Path) -> None:
    """The results file an analysis script declared: a JSON object of name → value, hashed
    like an output and read into the collection, looked for where the outputs are. Missing
    or not an object, it is a missing output, since the numbers it promised do not exist."""
    found = _matches(name, cwd) or _matches(name, root)
    if not found:
        out.missing.append(name)
        return
    path = found[0]
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        out.missing.append(name)
        out.diagnosis = f"results file {name} is not JSON: {exc}"
        return
    if not isinstance(data, dict):
        out.missing.append(name)
        out.diagnosis = f"results file {name} must hold a JSON object of name → value"
        return
    out.results = data
    out.artifacts.append(ArtifactRef.from_path(path, "data", root=root))


def where(directory: Path, root: Path) -> str:
    """A directory as the study names it: relative to the study root, the root itself as
    'the study root', and one outside as it is."""
    try:
        rel = str(directory.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(directory)
    return "the study root" if rel == "." else rel


def _matches(pattern: str, base: Path) -> list[Path]:
    if _GLOB_CHARS & set(pattern):
        # pathlib refuses an absolute glob ("Non-relative patterns are unsupported");
        # an output declared with an absolute path (a script whose cwd is outside the
        # workspace) is globbed from its own anchor instead
        pat = Path(pattern)
        if pat.is_absolute():
            base, pattern = Path(pat.anchor), str(pat.relative_to(pat.anchor))
        return sorted(p for p in base.glob(pattern) if p.is_file())
    candidate = base / pattern
    return [candidate] if candidate.is_file() else []


def collect_outputs(cwd: Path, patterns: list[str], stdout: str, stderr: str,
                    exit_code: int | None, root: Path,
                    results: str | None = None) -> ScriptCollection:
    """The declared outputs, each looked for under the script's cwd and then under the
    study root — the two places a path in a study is written relative to."""
    out = ScriptCollection()
    out.looked_in = [where(d, root) for d in dict.fromkeys((cwd, root))]
    for pattern in patterns:
        matches = _matches(pattern, cwd) or _matches(pattern, root)
        if not matches:
            out.missing.append(pattern)
        out.artifacts += [ArtifactRef.from_path(p, "output", root=root) for p in matches]
    if results:
        _read_results(cwd, results, out, root)

    stdout_tail = _tail(cwd / stdout)
    stderr_tail = _tail(cwd / stderr)
    text = stdout_tail + "\n" + stderr_tail
    if exit_code not in (0, None) or out.missing:
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


def collect_script_job(record: JobRecord, root: Path) -> ScriptCollection:
    return collect_outputs(Path(record.spec.cwd), record.meta.get("outputs", []),
                           record.spec.stdout, record.spec.stderr, record.exit_code, root,
                           results=record.meta.get("results"))


def script_observation(label: str, state: JobState, exit_code: int | None,
                       collected: ScriptCollection, done: str, /, **fields):
    """A script that has ended, as one observation: ``done`` when it completed with every
    declared output, else what failed — the same for a run in the session, for the
    submission that waited and for the job_status that collected it."""
    from myrmidon.core.observation import Observation

    if state == JobState.COMPLETED and not collected.missing:
        return Observation(ok=True, summary=done, artifacts=collected.artifacts,
                                 results=collected.results, **fields)
    why = (
        f"{state.value.lower()} (exit {exit_code})" if state != JobState.COMPLETED
        else f"completed but {len(collected.missing)} declared output(s) missing: "
        f"{collected.missing}, looked for in {' and '.join(collected.looked_in)}"
    )
    return Observation(
        ok=False,
        summary=f"{label} {why}" + (f" — {collected.diagnosis}" if collected.diagnosis else ""),
        artifacts=collected.artifacts, outputs_missing=collected.missing,
        results=collected.results, diagnosis=collected.diagnosis, log_tail=collected.log_tail,
        **fields,
    )


def git_state(path: Path) -> tuple[str | None, bool | None]:
    """(short commit, dirty) of the git repo containing ``path``; (None, None)
    when it is not under git or git is unavailable."""
    if shutil.which("git") is None:
        return None, None
    try:
        head = subprocess.run(  # noqa: S603
            ["git", "-C", str(path.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
        if head.returncode != 0:
            return None, None
        status = subprocess.run(  # noqa: S603
            ["git", "-C", str(path.parent), "status", "--porcelain",
             "--untracked-files=no"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    return head.stdout.strip(), bool(status.stdout.strip())
