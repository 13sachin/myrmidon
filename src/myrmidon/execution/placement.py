"""Where a call runs: in the session, or as a job.

The machine config's ``[session]`` envelope says what the session — a login node, a
workstation's foreground — may carry: so many cores for so many minutes. A call whose
resources fit runs synchronously in the session and books nothing; one that does not, asks
for a GPU, or waits on tracked jobs is submitted and priced like any job. One stated policy,
read the same way by the gate, the run and a protocol's price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from myrmidon.config import SessionEnvelope
from myrmidon.execution.backend import Resources


@dataclass(frozen=True)
class Placement:
    where: Literal["session", "job"]
    reason: str

    @property
    def in_session(self) -> bool:
        return self.where == "session"


def place(resources: Resources, envelope: SessionEnvelope, *, after: bool = False) -> Placement:
    """Session or job, and why, for a call asking for ``resources``."""
    cores = max(1, resources.ntasks) * max(1, resources.cpus_per_task)
    want = f"{cores} core(s) × {resources.walltime_min} min"
    have = f"{envelope.max_cores} core(s) × {envelope.max_minutes} min"
    if after:
        return Placement("job", "waits on tracked jobs, which only the scheduler orders")
    if resources.gpus:
        return Placement("job", "asks for a GPU, which the session has none of")
    if cores > envelope.max_cores or resources.walltime_min > envelope.max_minutes:
        return Placement("job", f"{want} exceeds the session envelope of {have}")
    return Placement("session", f"{want} fits the session envelope of {have}")
