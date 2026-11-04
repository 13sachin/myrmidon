"""The node receipt: what a job's inputs were when it ran, written where it ran.

The submitting call hashes the job's inputs into the job store. The job script hashes the
same files on the compute node as the engine is about to read them (``input_hashes.before``)
and writes ``execution.json`` — the job, when it started and finished, its exit code.
Collection compares: a file that changed between submission and start, other than by a job
this one waited on, ran an experiment nobody approved, and its result is not accepted. The
receipt lives in its own directory beside the run, named in the job spec.
"""

from __future__ import annotations

import json
from pathlib import Path

from myrmidon.core.provenance import shquote

BEFORE = "input_hashes.before"
EXECUTION = "execution.json"
_NOW = "$(date -u +%Y-%m-%dT%H:%M:%SZ)"


def wrap(receipt: str | None, inputs: list[str], job: str) -> tuple[str, str]:
    """(the shell line before the engine, the one after it, which reads the engine's exit
    status from ``$rc``); empty for a job with no receipt. ``job`` is the shell expression
    naming the job on the node."""
    if not receipt or not inputs:
        return "", ""
    where = shquote(receipt)
    files = " ".join(shquote(p) for p in inputs)
    before = (f"sha256sum {files} > {where}/{BEFORE} 2>/dev/null; "
              f"started={_NOW}")
    after = (f"printf '{{\"job\": \"%s\", \"started\": \"%s\", \"finished\": \"%s\", "
             f"\"exit_code\": %s}}\\n' \"{job}\" \"$started\" \"{_NOW}\" \"$rc\" "
             f"> {where}/{EXECUTION}")
    return before, after


def read_hashes(path: Path) -> dict[str, str]:
    """path -> sha256 from a ``sha256sum`` listing; empty when there is none."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        sha, _, name = line.partition("  ")
        if sha and name:
            out[name] = sha
    return out


def changed(receipt: str | None, expected: dict[str, str]
            ) -> tuple[dict[str, str] | None, dict]:
    """(each input whose hash on the node differs from the one recorded at submission, with
    the hash the node saw — None when the node wrote no hashes —, the job's execution
    record)."""
    if not receipt or not expected:
        return None, {}
    where = Path(receipt)
    try:
        execution = json.loads((where / EXECUTION).read_text())
    except (OSError, ValueError):
        execution = {}
    on_node = read_hashes(where / BEFORE)
    if not on_node:
        return None, execution
    return ({p: on_node.get(p, "") for p, sha in expected.items() if on_node.get(p) != sha},
            execution)
