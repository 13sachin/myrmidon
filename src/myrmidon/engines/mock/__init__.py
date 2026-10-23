"""Mock engines and scheduler: the harness with nothing real behind it.

``bin/`` holds executable fakes (``gmx``, ``cpmd.x``, ``mimicpy``, ``chimerax``, ``sbatch``,
``squeue``, ``sacct``, ``scancel``, ``srun``, ``mpirun``) that print output recorded from
real runs and write the files the real engines would. They ship with the package, so
``myrmidon serve --mock`` and ``myrmidon run --mock`` work from an installed harness, and
the unit tests use the same ones.

The mock scheduler keeps its state in a directory (``MOCK_SLURM_DIR``), one JSON file per
job, and runs on a logical clock: a job is reported RUNNING for a fixed number of
``squeue`` queries before it lands, whatever the host's speed. The mocks read every other
behaviour switch from ``MOCK_*`` variables (``MOCK_GMX_FAIL``, ``MOCK_SLURM_HOLD``, ...), so
a run that must not be steered by its shell strips them first.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from myrmidon.core.tool import ToolContext

#: the mock binaries, put first on PATH to stand in for the real ones
BIN = Path(__file__).resolve().parent / "bin"
#: prefix of the variables the mocks read; an inherited one changes what they do
ENV_PREFIX = "MOCK_"
#: squeue queries a mock job reports RUNNING before it lands: one, so a session meets a
#: running job exactly once
SLURM_CLOCK = 1


def scheduler_env(state_dir: Path) -> dict[str, str]:
    """The mock scheduler's environment: its job records in ``state_dir/.mock_slurm``, and
    the logical clock instead of wall-clock timing."""
    return {"MOCK_SLURM_DIR": str(Path(state_dir) / ".mock_slurm"),
            "MOCK_SLURM_CLOCK": str(SLURM_CLOCK)}


def strip_env() -> dict[str, str]:
    """Remove inherited ``MOCK_*`` variables; return them so the caller can restore them."""
    removed = {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIX)}
    for key in removed:
        del os.environ[key]
    return removed


def check_bin() -> None:
    """Raise if the mock binaries are missing or not executable (an install that lost the
    mode bits would otherwise fail later as "engine not found")."""
    if not BIN.is_dir():
        raise RuntimeError(f"mock engines not available: {BIN} missing")
    broken = sorted(p.name for p in BIN.iterdir() if not os.access(p, os.X_OK))
    if broken:
        raise RuntimeError(f"mock engines not executable in {BIN}: {', '.join(broken)}")


def enter(state_dir: Path) -> None:
    """Point this process at the mock engines and scheduler, for good: no inherited
    ``MOCK_*`` switch, the mocks first on PATH, the scheduler's state under ``state_dir``.
    For a process that is the mock session (``serve --mock``); a caller that must restore
    its environment uses :func:`activated`."""
    check_bin()
    strip_env()
    os.environ["PATH"] = f"{BIN}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ.update(scheduler_env(state_dir))


@contextmanager
def activated(state_dir: Path) -> Iterator[None]:
    """:func:`enter` for the duration of the block, then the environment as it was."""
    saved = {k: os.environ.get(k) for k in ("PATH", *scheduler_env(state_dir))}
    inherited = strip_env()
    try:
        enter(state_dir)
        yield
    finally:
        strip_env()
        os.environ.update(inherited)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def mock_context(ctx: ToolContext) -> ToolContext:
    """``ctx`` on the mock engines (``gmx``, ``cpmd.x``, the mock ``sbatch``), with the
    machine's compute accounting kept, so budget gates fire the way they would for real.
    Call it with the mocks on PATH (:func:`enter`), or discovery finds the real engines."""
    from myrmidon.engines.discovery import discover
    from myrmidon.execution.slurm import SlurmBackend

    config = ctx.config.model_copy(update={
        "gromacs_bin": "gmx", "cpmd_bin": "cpmd.x", "backend": "slurm",
        # the mock sbatch runs the script with sh on this node: no module prelude
        "slurm": ctx.config.slurm.model_copy(update={"extra_sbatch_lines": []}),
    })
    # `replace`, not a fresh ToolContext: every field the session carries is inherited and
    # only what the mock changes is named. Listing fields by hand would drop any store the
    # session carries, and a call that is correct against the real context would be
    # refused in the mock one. A store added to the world must reach every driver by default.
    return replace(
        ctx,
        config=config,
        backend=SlurmBackend(config.slurm, cores_per_node=config.compute.cores_per_node
                             if config.compute.node_exclusive else None),
        engines=discover(config),
    )
