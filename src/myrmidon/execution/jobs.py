"""Submitting a job and looking at it again: the one path every tool that books the
scheduler goes through."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import NamedTuple

from myrmidon.core.artifacts import sha256_file
from myrmidon.core.provenance import CALL_INPUTS
from myrmidon.core.tool import ToolContext
from myrmidon.execution.accounting import spec_core_h
from myrmidon.execution.backend import JobSpec, JobState
from myrmidon.execution.jobstore import JOBSTORE_FILENAME, JobRecord, JobStore, open_jobstore


class SubmittedJob(NamedTuple):
    job_id: str
    core_h_estimate: float


def submit_job(ctx: ToolContext, spec: JobSpec, *, kind: str, label: str,
               meta: dict | None = None, after: list[str] | None = None) -> SubmittedJob:
    """Submit ``spec`` and record it in the study's job store with its kind (the collector
    that reads it when it ends), its core-hour estimate, and what that collector needs.
    ``after`` names tracked jobs that must complete OK first."""
    est = spec_core_h(spec, ctx.config.compute)
    store = open_jobstore(ctx.workspace)
    hashes = _receipt(ctx, spec)
    if hashes:
        meta = {**(meta or {}), "input_hashes": hashes, **({"after": after} if after else {})}
    if after:
        deps = [ctx.backend.dep_for(rec.spec, rec.handle) for rec in map(store.get, after)]
        handle = ctx.backend.submit_with_deps(spec, deps, mode="ok")
    else:
        handle = ctx.backend.submit(spec)
    job_id = store.add(ctx.backend.name, handle, spec, label=label,
                       meta={**(meta or {}), "kind": kind, "core_h_estimate": est})
    return SubmittedJob(job_id, est)


def _receipt(ctx: ToolContext, spec: JobSpec) -> dict[str, str]:
    """Set the job's node receipt up (execution/receipt.py): the files the node will read —
    those the tool names on the spec, else the inputs the submitting call consumed — by
    absolute path and their hash now. Returns them."""
    root = Path(ctx.workspace.root)
    if spec.receipt_inputs:
        hashes = {str(Path(p)): sha256_file(Path(p)) for p in spec.receipt_inputs
                  if Path(p).is_file()}
    else:
        hashes = {}
        for art in CALL_INPUTS.get() or []:
            path = Path(art.path) if Path(art.path).is_absolute() else root / art.path
            if path.is_file():
                hashes[str(path)] = art.sha256
    if hashes:
        where = Path(spec.cwd) / f".receipt-{uuid.uuid4().hex[:8]}"
        where.mkdir(parents=True, exist_ok=True)
        spec.receipt, spec.receipt_inputs = str(where), sorted(hashes)
    return hashes


def existing_jobstore(workspace) -> JobStore | None:
    """The study's job store if it has one. Never creates it: a look writes nothing."""
    db = workspace.state_dir / JOBSTORE_FILENAME
    return JobStore(db) if db.exists() else None


def refresh(ctx: ToolContext, job_id: str) -> tuple[JobRecord, JobState, int | None, bool]:
    """(record, state, exit_code, changed) after one poll; ``changed`` is True when the
    poll moved the stored state, i.e. something actually happened."""
    store = open_jobstore(ctx.workspace)
    record = store.get(job_id)
    if record.state.terminal:
        return record, record.state, record.exit_code, False
    status = ctx.backend.poll(record.handle, record.spec)
    changed = status.state != record.state
    if changed:
        store.update(job_id, status.state, status.exit_code)
    return record, status.state, status.exit_code, changed


def booked_under_mandates(workspace, digests: set[str]) -> float:
    """Core-hours booked by the study's jobs submitted under any of these mandates: their
    estimates, the upper bound the scheduler may charge."""
    store = existing_jobstore(workspace)
    if store is None or not digests:
        return 0.0
    return round(sum(float(rec.meta.get("core_h_estimate") or 0.0) for rec in store.list()
                     if rec.meta.get("mandate") in digests), 3)
