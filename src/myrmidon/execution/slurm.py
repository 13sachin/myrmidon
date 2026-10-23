"""SLURM backend: rendered sbatch scripts, squeue polling with sacct fallback.

State machine per poll:
1. ``squeue -j <id>`` — if the job is listed, map its SLURM state.
2. Otherwise ``sacct -j <id>`` — terminal state + exit code.
3. Neither knows the job → UNKNOWN (accounting lag or purged history).
"""

from __future__ import annotations

import re
import subprocess
import time
import uuid
from pathlib import Path

from myrmidon.config import SlurmDefaults
from myrmidon.core.provenance import shquote
from myrmidon.errors import MyrmidonError
from myrmidon.execution.backend import (
    ChainDep,
    ExecutionBackend,
    JobSpec,
    JobState,
    JobStatus,
)

_SQUEUE_STATE = {
    "PENDING": JobState.QUEUED,
    "CONFIGURING": JobState.QUEUED,
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "SUSPENDED": JobState.RUNNING,
    "COMPLETED": JobState.COMPLETED,
    "FAILED": JobState.FAILED,
    "TIMEOUT": JobState.TIMEOUT,
    "OUT_OF_MEMORY": JobState.FAILED,
    "NODE_FAIL": JobState.FAILED,
    "CANCELLED": JobState.CANCELLED,
}


def _walltime(minutes: int) -> str:
    return f"{minutes // 60:d}:{minutes % 60:02d}:00"


class SlurmBackend(ExecutionBackend):
    name = "slurm"

    def __init__(
        self,
        defaults: SlurmDefaults | None = None,
        sbatch: str = "sbatch",
        squeue: str = "squeue",
        sacct: str = "sacct",
        scancel: str = "scancel",
        use_srun: bool = True,
        cores_per_node: int | None = None,
        env_script: str | None = None,
    ):
        self.defaults = defaults or SlurmDefaults()
        # sourced at the top of every batch script (config.env_script)
        self.env_script = str(Path(env_script).expanduser()) if env_script else None
        self.sbatch = sbatch
        self.squeue = squeue
        self.sacct = sacct
        self.scancel = scancel
        self.use_srun = use_srun
        # Node-exclusive sites bill nodes × cores_per_node, while sacct's AllocCPUS
        # counts hardware threads (twice the cores on a hyperthreaded node) — set this to
        # account the way the site does.
        self.cores_per_node = cores_per_node

    # -- script rendering -------------------------------------------------
    def render_script(self, spec: JobSpec, job_name: str) -> str:
        res = spec.resources
        partition = res.partition or self.defaults.partition
        account = res.account or self.defaults.account
        # Coupled programs redirect their own stdout/stderr inside the script, so
        # the batch-level streams must not share those files (SLURM would
        # truncate/interleave the QM log).
        batch_out = "slurm-batch.out" if spec.coupled else spec.stdout
        batch_err = "slurm-batch.err" if spec.coupled else spec.stderr
        lines = [
            "#!/bin/sh",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --ntasks={res.ntasks}",
            f"#SBATCH --cpus-per-task={res.cpus_per_task}",
            f"#SBATCH --time={_walltime(res.walltime_min)}",
            f"#SBATCH --output={batch_out}",
            f"#SBATCH --error={batch_err}",
        ]
        if res.gpus:
            lines.append(f"#SBATCH --gpus={res.gpus}")
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        if account:
            lines.append(f"#SBATCH --account={account}")
        if self.env_script and not any(
            self.env_script in ln for ln in self.defaults.extra_sbatch_lines
        ):
            from myrmidon.core.envscript import source_line

            lines.append(source_line(self.env_script))
        lines.extend(self.defaults.extra_sbatch_lines)
        for key, value in spec.env.items():
            lines.append(f"export {key}={shquote(value)}")
        def decorate(argv: list[str], mpi_ranks: int | None) -> str:
            joined = " ".join(shquote(a) for a in argv)
            if self.use_srun:
                if mpi_ranks:
                    return f"srun --ntasks={mpi_ranks} --exact {joined}"
                return f"srun {joined}"
            return joined

        if spec.coupled and self.defaults.coupled_launch == "multi-prog":
            lines.extend(self._render_multi_prog(spec))
        elif spec.coupled:
            # MiMiC-style concurrent programs inside one allocation: each gets
            # its own srun step; the job fails if any program fails.
            lines.append(
                f"{decorate(spec.argv, spec.mpi_ranks)} > {shquote(spec.stdout)} "
                f"2> {shquote(spec.stderr)} & p0=$!"
            )
            for i, cc in enumerate(spec.coupled, start=1):
                lines.append(
                    f"{decorate(cc.argv, cc.mpi_ranks)} > {shquote(cc.stdout)} "
                    f"2> {shquote(cc.stderr)} & p{i}=$!"
                )
            lines.append("code=0")
            for i in range(len(spec.coupled) + 1):
                lines.append(
                    f"wait $p{i}; c=$?; "
                    "if [ $c -ne 0 ] && [ $code -eq 0 ]; then code=$c; fi"
                )
            lines.append("exit $code")
        else:
            lines.append(decorate(spec.argv, spec.mpi_ranks))
        lines.append("")
        return "\n".join(lines)

    def _render_multi_prog(self, spec: JobSpec) -> list[str]:
        """One ``srun --multi-prog`` step: contiguous rank ranges per program.

        Each program runs through a small wrapper so that only its first rank
        writes the program's stdout (CPMD/GROMACS print from rank 0); stderr of
        every rank is appended to the program's stderr file.
        """
        programs = [(spec.argv, spec.mpi_ranks or 1, spec.stdout, spec.stderr)]
        programs += [(cc.argv, cc.mpi_ranks or 1, cc.stdout, cc.stderr) for cc in spec.coupled]
        lines: list[str] = []
        conf: list[str] = []
        first = 0
        for i, (argv, ranks, stdout, stderr) in enumerate(programs):
            last = first + ranks - 1
            joined = " ".join(shquote(a) for a in argv)
            wrapper = f"multiprog-{i}.sh"
            lines += [
                f"cat > {wrapper} <<'MYRMIDON_PROG'",
                "#!/bin/sh",
                f'if [ "$SLURM_PROCID" = "{first}" ]; then',
                f"  exec {joined} > {shquote(stdout)} 2>> {shquote(stderr)}",
                "fi",
                f"exec {joined} > /dev/null 2>> {shquote(stderr)}",
                "MYRMIDON_PROG",
                f"chmod +x {wrapper}",
                f": > {shquote(stderr)}",
            ]
            conf.append(f"{first}-{last} ./{wrapper}")
            first = last + 1
        lines += ["cat > multiprog.conf <<'MYRMIDON_CONF'", *conf, "MYRMIDON_CONF"]
        lines.append("export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}")
        extra = " ".join(shquote(a) for a in self.defaults.srun_args)
        lines.append(
            f"srun --ntasks={first} {extra + ' ' if extra else ''}--multi-prog multiprog.conf"
        )
        return lines

    # -- backend interface -------------------------------------------------
    def submit_with_deps(
        self, spec: JobSpec, deps: list[ChainDep], mode: str = "ok"
    ) -> str:
        kind = "afterok" if mode == "ok" else "afterany"
        dependency = f"{kind}:" + ":".join(d.handle for d in deps)
        return self.submit(spec, dependency=dependency)

    def retry_wait_for(self, message: str, attempt: int) -> int | None:
        """Seconds to wait before retrying a refused submission, or None to give up.

        A per-user submit limit (a development partition allowing a handful of submitted
        jobs) is a fact about the site, so the patterns and the waits come from the
        machine config and every submission gets the same treatment — a campaign must not
        carry its own regex.
        """
        backoff = self.defaults.submit_retry_backoff_s
        if attempt >= len(backoff):
            return None
        if not any(re.search(p, message, re.I) for p in self.defaults.submit_retry_on):
            return None
        return backoff[attempt]

    def submit(self, spec: JobSpec, dependency: str | None = None) -> str:
        run_dir = Path(spec.cwd) / f".slurm-{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        script = run_dir / "job.sbatch"
        script.write_text(self.render_script(spec, job_name=f"myrmidon-{run_dir.name}"))
        argv = [self.sbatch, str(script)]
        if dependency:
            argv.insert(1, f"--dependency={dependency}")
        attempt = 0
        while True:
            proc = subprocess.run(  # noqa: S603
                argv,
                cwd=spec.cwd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0:
                break
            refusal = proc.stderr.strip() or proc.stdout.strip()
            wait_s = self.retry_wait_for(refusal, attempt)
            if wait_s is None:
                raise MyrmidonError(
                    f"sbatch failed (exit {proc.returncode}): {refusal[:500]}"
                )
            with (run_dir / "submit.log").open("a") as fh:
                fh.write(
                    f"attempt {attempt + 1} refused: {refusal[:300]}\n"
                    f"  matches the site's submit-limit patterns; retrying in {wait_s} s\n"
                )
            time.sleep(wait_s)
            attempt += 1
        m = re.search(r"Submitted batch job (\d+)", proc.stdout)
        if not m:
            raise MyrmidonError(f"could not parse sbatch output: {proc.stdout[:200]!r}")
        return m.group(1)

    def poll(self, handle: str, spec: JobSpec) -> JobStatus:
        squeue = subprocess.run(  # noqa: S603
            [self.squeue, "-j", handle, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        state_text = squeue.stdout.strip().splitlines()
        if squeue.returncode == 0 and state_text:
            raw = state_text[0].strip()
            state = _SQUEUE_STATE.get(raw, JobState.UNKNOWN)
            return JobStatus(state=state, detail=f"squeue: {raw}")

        sacct = subprocess.run(  # noqa: S603
            [self.sacct, "-j", handle, "-n", "-X", "-P", "-o", "State,ExitCode"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        for line in sacct.stdout.strip().splitlines():
            fields = line.strip().split("|")
            if len(fields) < 2:
                continue
            raw = fields[0].split()[0].rstrip("+")  # e.g. "CANCELLED by 123"
            exit_code = None
            m = re.match(r"(\d+):", fields[1])
            if m:
                exit_code = int(m.group(1))
            state = _SQUEUE_STATE.get(raw, JobState.UNKNOWN)
            return JobStatus(state=state, exit_code=exit_code, detail=f"sacct: {raw}")
        return JobStatus(
            state=JobState.UNKNOWN,
            detail="job unknown to both squeue and sacct (accounting lag?)",
        )

    def cancel(self, handle: str, spec: JobSpec) -> None:
        subprocess.run(  # noqa: S603
            [self.scancel, handle], capture_output=True, text=True, timeout=60
        )

    def usage(self, handle: str, spec: JobSpec) -> float | None:
        """Core-hours from sacct: AllocNodes × cores_per_node × ElapsedRaw on
        node-exclusive sites, else AllocCPUS × ElapsedRaw."""
        try:
            sacct = subprocess.run(  # noqa: S603
                [self.sacct, "-j", handle, "-n", "-X", "-P",
                 "-o", "AllocNodes,AllocCPUS,ElapsedRaw"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        for line in sacct.stdout.strip().splitlines():
            fields = line.strip().split("|")
            if len(fields) < 3 or not all(f.isdigit() for f in fields[:3]):
                continue
            nodes, cpus, elapsed = (int(f) for f in fields[:3])
            cores = nodes * self.cores_per_node if self.cores_per_node else cpus
            return round(cores * elapsed / 3600.0, 3)
        return None
