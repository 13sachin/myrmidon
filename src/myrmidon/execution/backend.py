"""Execution backend contract.

Jobs are long-running; the agent loop must never block on one. ``submit``
returns a handle immediately; ``poll`` is cheap and side-effect free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class Resources(BaseModel):
    ntasks: int = 1
    cpus_per_task: int = 1
    gpus: int = 0
    walltime_min: int = 60
    partition: str | None = None
    account: str | None = None


class CoupledCmd(BaseModel):
    """A second executable launched alongside the main argv (MiMiC: the MM
    partner program). Backends decorate with mpirun/srun per mpi_ranks."""

    argv: list[str]
    mpi_ranks: int | None = None
    stdout: str = "coupled.out"
    stderr: str = "coupled.err"


class JobSpec(BaseModel):
    argv: list[str]
    cwd: Path
    env: dict[str, str] = Field(default_factory=dict)
    stdout: str = "job.out"
    stderr: str = "job.err"
    resources: Resources = Field(default_factory=Resources)
    mpi_ranks: int | None = None  # main argv runs under mpirun/srun with this many ranks
    coupled: list[CoupledCmd] | None = None  # MiMiC two-program launch


class JobState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    # a scheduler kill at the walltime: a failure whose cause is already known, and the
    # one state that says so. Folded into FAILED it would read as "no recognized crash
    # pattern".
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"

    @property
    def terminal(self) -> bool:
        return self in (
            JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT,
        )


class JobStatus(BaseModel):
    state: JobState
    exit_code: int | None = None
    detail: str | None = None


class ChainDep(BaseModel):
    """A dependency edge for chained submission: the predecessor's handle plus
    whatever the backend needs to observe its completion (SLURM: the handle is
    enough; local: the run-dir exit sentinel of the predecessor)."""

    handle: str
    exit_sentinel: Path | None = None


class ExecutionBackend(ABC):
    name: str

    @abstractmethod
    def submit(self, spec: JobSpec) -> str:
        """Start the job and return a backend handle immediately."""

    @abstractmethod
    def poll(self, handle: str, spec: JobSpec) -> JobStatus:
        """Cheap, idempotent state check."""

    @abstractmethod
    def cancel(self, handle: str, spec: JobSpec) -> None:
        """Best-effort cancellation."""

    def usage(self, handle: str, spec: JobSpec) -> float | None:
        """Core-hours actually consumed by a finished job, if the backend can
        tell (SLURM accounting; local: elapsed × cores). None when unknown."""
        return None

    # -- chained submission (optional capability) --------------------------
    def dep_for(self, spec: JobSpec, handle: str) -> ChainDep:
        """Describe a submitted job as a dependency for a later submission."""
        return ChainDep(handle=handle)

    def submit_with_deps(
        self, spec: JobSpec, deps: list[ChainDep], mode: str = "ok"
    ) -> str:
        """Submit ``spec`` to start only after ``deps`` finish.

        ``mode`` is ``"ok"`` (predecessors must succeed; the job is skipped or
        held otherwise) or ``"any"`` (start once they are merely finished —
        the convention for early-exit continuation segments).
        Backends without dependency support raise.
        """
        raise NotImplementedError(f"backend {self.name!r} cannot chain jobs")
