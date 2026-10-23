"""Local backend: detached subprocesses that survive harness restarts.

The job is wrapped in ``sh -c '...; echo $? > .exit'`` launched in its own
session (setsid). State is derived from files, not from a parent process:

- ``<run>/.pid``   — process-group leader pid, written at submit
- ``<run>/.exit``  — exit code sentinel, written when the command finishes

so the MCP server can restart and still poll correctly.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import uuid
from pathlib import Path

from myrmidon.core.provenance import shquote
from myrmidon.execution.backend import (
    ChainDep,
    ExecutionBackend,
    JobSpec,
    JobState,
    JobStatus,
)

#: exit code written when an afterok dependency finished unsuccessfully — the
#: dependent job is skipped, and poll() reports it FAILED with this code.
DEP_FAILED_EXIT = 78


class LocalBackend(ExecutionBackend):
    name = "local"

    def __init__(self, mpirun: str = "mpirun"):
        self.mpirun = mpirun

    def _run_dir(self, spec: JobSpec, handle: str) -> Path:
        return Path(spec.cwd) / f".run-{handle}"

    def _decorate(self, argv: list[str], mpi_ranks: int | None) -> str:
        if mpi_ranks:
            argv = [self.mpirun, "-np", str(mpi_ranks), *argv]
        return " ".join(shquote(a) for a in argv)

    def dep_for(self, spec: JobSpec, handle: str) -> ChainDep:
        return ChainDep(handle=handle, exit_sentinel=self._run_dir(spec, handle) / ".exit")

    def submit_with_deps(
        self, spec: JobSpec, deps: list[ChainDep], mode: str = "ok"
    ) -> str:
        """Chain locally by waiting on predecessor exit sentinels.

        The wait/guard preamble runs inside the detached job itself, so the
        harness never blocks. In ``ok`` mode a failed predecessor makes this
        job skip with exit code ``DEP_FAILED_EXIT`` (mirroring SLURM afterok
        holding the dependent job).
        """
        sentinels = [d.exit_sentinel for d in deps]
        if any(s is None for s in sentinels):
            raise ValueError("local chain deps need exit sentinels (use dep_for)")
        waits = [
            f"until [ -f {shquote(str(s))} ]; do sleep 1; done" for s in sentinels
        ]
        guards = []
        if mode == "ok":
            guards = [
                f'[ "$(cat {shquote(str(s))})" = "0" ] || '
                f"{{ echo {DEP_FAILED_EXIT} > __EXIT__; exit {DEP_FAILED_EXIT}; }}"
                for s in sentinels
            ]
        preamble = "; ".join(waits + guards)
        return self.submit(spec, preamble=preamble)

    def submit(self, spec: JobSpec, preamble: str | None = None) -> str:
        handle = uuid.uuid4().hex[:8]
        run_dir = self._run_dir(spec, handle)
        run_dir.mkdir(parents=True, exist_ok=True)
        cwd = Path(spec.cwd)
        exit_file = shquote(str(run_dir / ".exit"))
        cmd = self._decorate(spec.argv, spec.mpi_ranks)
        if preamble:
            # the guard must write the real sentinel of THIS submission
            cmd = f"{preamble.replace('__EXIT__', exit_file)}; {cmd}"
        # The outer sh backgrounds the job, records its pid, and exits; we reap
        # the outer immediately, and the job reparents to init — the harness
        # never accumulates zombie children, and the job survives our restart.
        if spec.coupled:
            # MiMiC-style: all programs start concurrently; the job fails if
            # any of them fails (first nonzero exit code wins).
            parts = [f"{cmd} > {shquote(spec.stdout)} 2> {shquote(spec.stderr)} & p0=$!"]
            for i, cc in enumerate(spec.coupled, start=1):
                ccmd = self._decorate(cc.argv, cc.mpi_ranks)
                parts.append(
                    f"{ccmd} > {shquote(cc.stdout)} 2> {shquote(cc.stderr)} & p{i}=$!"
                )
            n = len(spec.coupled)
            parts.append("code=0")
            for i in range(n + 1):
                parts.append(f"wait $p{i}; c=$?; [ $c -ne 0 ] && [ $code -eq 0 ] && code=$c")
            parts.append(f"echo $code > {exit_file}")
            inner = "(" + "; ".join(parts) + ")"
        else:
            inner = (
                f"({cmd} > {shquote(spec.stdout)} 2> {shquote(spec.stderr)}; "
                f"echo $? > {exit_file})"
            )
        wrapper = f"{inner} & echo $! > {shquote(str(run_dir / '.pid'))}"
        env = {**os.environ, **spec.env}
        proc = subprocess.Popen(  # noqa: S603
            ["sh", "-c", wrapper],
            cwd=cwd,
            env=env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        (run_dir / ".pgid").write_text(str(proc.pid))
        proc.wait()  # outer sh exits as soon as the job is backgrounded
        return handle

    def poll(self, handle: str, spec: JobSpec) -> JobStatus:
        run_dir = self._run_dir(spec, handle)
        exit_file = run_dir / ".exit"
        if exit_file.exists():
            try:
                code = int(exit_file.read_text().strip())
            except ValueError:
                return JobStatus(state=JobState.UNKNOWN, detail="unreadable exit sentinel")
            state = JobState.COMPLETED if code == 0 else JobState.FAILED
            return JobStatus(state=state, exit_code=code)
        pid_file = run_dir / ".pid"
        if not pid_file.exists():
            return JobStatus(state=JobState.UNKNOWN, detail="no pid file — never submitted?")
        pid = int(pid_file.read_text().strip())
        if _alive(pid):
            return JobStatus(state=JobState.RUNNING)
        return JobStatus(
            state=JobState.FAILED,
            detail="process gone without exit sentinel (killed or machine rebooted)",
        )

    def usage(self, handle: str, spec: JobSpec) -> float | None:
        """Elapsed wall time (pid file → exit sentinel mtimes) × requested cores."""
        run_dir = self._run_dir(spec, handle)
        pid_file, exit_file = run_dir / ".pid", run_dir / ".exit"
        if not (pid_file.exists() and exit_file.exists()):
            return None
        elapsed = max(0.0, exit_file.stat().st_mtime - pid_file.stat().st_mtime)
        res = spec.resources
        cores = max(1, spec.mpi_ranks or res.ntasks) * max(1, res.cpus_per_task)
        return round(elapsed * cores / 3600.0, 4)

    def cancel(self, handle: str, spec: JobSpec) -> None:
        run_dir = self._run_dir(spec, handle)
        pgid_file = run_dir / ".pgid"
        if not pgid_file.exists():
            return
        pgid = int(pgid_file.read_text().strip())
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
