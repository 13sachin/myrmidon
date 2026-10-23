"""Job store: sqlite tracking of async jobs, persistent across server restarts.

``meta`` is a free JSON column for what the submitting tool needs back at
collection time (declared script outputs, the core-hour estimate, ...).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.errors import JobNotFoundError
from myrmidon.execution.backend import JobSpec, JobState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    handle TEXT NOT NULL,
    spec TEXT NOT NULL,
    state TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    finished_at TEXT,
    exit_code INTEGER,
    label TEXT,
    meta TEXT
);
"""


class JobRecord(BaseModel):
    job_id: str
    backend: str
    handle: str
    spec: JobSpec
    state: JobState
    submitted_at: str
    finished_at: str | None = None
    exit_code: int | None = None
    label: str | None = None
    meta: dict = Field(default_factory=dict)


JOBSTORE_FILENAME = "jobs.sqlite"


def open_jobstore(workspace) -> JobStore:
    """The workspace's job store — the one durable record of every submission."""
    return JobStore(workspace.state_dir / JOBSTORE_FILENAME)


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "meta" not in columns:  # stores created before the column existed
                conn.execute("ALTER TABLE jobs ADD COLUMN meta TEXT")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add(
        self,
        backend: str,
        handle: str,
        spec: JobSpec,
        label: str | None = None,
        meta: dict | None = None,
    ) -> str:
        from myrmidon.core.provenance import CALL_EVENT, CALL_PLAN

        # the call that submitted it and the plan that call was made under, so a plan's cost
        # counts its own jobs and a reused step can name the run it reuses
        plan_id, plan_step = CALL_PLAN.get()
        if plan_id:
            meta = {**(meta or {}), "plan_id": plan_id, "plan_step": plan_step}
        if CALL_EVENT.get():
            meta = {**(meta or {}), "event_id": CALL_EVENT.get()}
        job_id = "job-" + uuid.uuid4().hex[:8]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, backend, handle, spec, state, submitted_at,"
                " label, meta) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    backend,
                    handle,
                    spec.model_dump_json(),
                    JobState.QUEUED.value,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    label,
                    json.dumps(meta or {}),
                ),
            )
        return job_id

    def get(self, job_id: str) -> JobRecord:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT job_id, backend, handle, spec, state, submitted_at, finished_at,"
                " exit_code, label, meta FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise JobNotFoundError(f"unknown job '{job_id}'")
        return JobRecord(
            job_id=row[0],
            backend=row[1],
            handle=row[2],
            spec=JobSpec.model_validate(json.loads(row[3])),
            state=JobState(row[4]),
            submitted_at=row[5],
            finished_at=row[6],
            exit_code=row[7],
            label=row[8],
            meta=json.loads(row[9]) if row[9] else {},
        )

    def update(self, job_id: str, state: JobState, exit_code: int | None = None) -> None:
        finished = (
            datetime.now(UTC).isoformat(timespec="seconds") if state.terminal else None
        )
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET state = ?, exit_code = COALESCE(?, exit_code),"
                " finished_at = COALESCE(?, finished_at) WHERE job_id = ?",
                (state.value, exit_code, finished, job_id),
            )
            if cur.rowcount == 0:
                raise JobNotFoundError(f"unknown job '{job_id}'")

    def set_meta(self, job_id: str, **fields) -> None:
        record = self.get(job_id)
        record.meta.update(fields)
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET meta = ? WHERE job_id = ?",
                (json.dumps(record.meta), job_id),
            )

    def list(self, pending_only: bool = False) -> list[JobRecord]:
        with self._conn() as conn:
            rows = conn.execute("SELECT job_id FROM jobs ORDER BY submitted_at").fetchall()
        records = [self.get(r[0]) for r in rows]
        if pending_only:
            records = [r for r in records if not r.state.terminal]
        return records
