"""SlurmBackend against the mock sbatch/squeue/sacct/scancel binaries."""

import time
from pathlib import Path

import pytest

from myrmidon.execution.backend import CoupledCmd, JobSpec, Resources


def wait_terminal(backend, handle, spec, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = backend.poll(handle, spec)
        if status.state.terminal:
            return status
        time.sleep(0.05)
    raise AssertionError("job did not reach a terminal state")


REFUSING_SBATCH = """#!/bin/sh
n=$(cat "$COUNTER" 2>/dev/null || echo 0)
n=$((n + 1))
echo $n > "$COUNTER"
if [ "$n" -le 2 ]; then
  echo "sbatch: error: Batch job submission failed: Job violates accounting/QOS policy" >&2
  echo "sbatch: error: QOSMaxSubmitJobPerUserLimit" >&2
  exit 1
fi
echo "Submitted batch job 4242"
"""


def _refusing_sbatch(tmp_path, monkeypatch):
    """A scheduler that refuses twice on the per-user submit limit, then accepts: a
    development QOS with MaxSubmitPU=4, MaxJobsPU=2 refuses a campaign's next single
    point with QOSMaxSubmitJobPerUserLimit while the earlier ones are queued."""
    fake = tmp_path / "sbatch-refusing"
    fake.write_text(REFUSING_SBATCH)
    fake.chmod(0o755)
    monkeypatch.setenv("COUNTER", str(tmp_path / "attempts"))
    return str(fake)


def _coupled_spec(tmp_path):
    return JobSpec(
        argv=["cpmd.x", "run.inp"],
        cwd=tmp_path,
        stdout="cpmd.out",
        stderr="cpmd.err",
        mpi_ranks=4,
        resources=Resources(ntasks=6),
        coupled=[
            CoupledCmd(
                argv=["gmx_mpi_d", "mdrun", "-deffnm", "mimic"],
                mpi_ranks=2,
                stdout="mimic.stdout",
                stderr="mimic.stderr",
            )
        ],
    )



# How a partition charges, as the scheduler states it (execution/billing.py). The fixtures
# are real `sinfo -o "%P|%X|%Y|%Z|%c|%G"` and `scontrol show partition --oneliner` output
# from a site that bills hardware threads at CPU=0.5 (2 × 24 cores, 96 CPUs per node), so
# neither the CPU count nor the core count alone is the price.
# (partition, cores per node, billed per node-hour, nodes whole?, note)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "slurm"
SCHEDULED = [
    ("batch", 48, 48.0, True, ""),
    ("gpus", 40, 40.0, True, ""),
    ("booster", 48, 48.0, True, ""),
    # mixed node shapes: fewest cores, dearest node — an upper bound
    ("maint", 40, 48.0, True, ""),
]


@pytest.mark.parametrize("name,cores,billed,exclusive,note", SCHEDULED)
def test_a_partition_is_priced_as_its_scheduler_bills_it(name, cores, billed, exclusive, note):
    from myrmidon.execution.billing import parse_partitions

    partitions, default = parse_partitions((FIXTURES / "sinfo_partitions.txt").read_text(),
                                           (FIXTURES / "scontrol_partitions.txt").read_text())
    assert default == "batch"
    node = partitions[name]
    assert (node.cores_per_node, node.billed_per_node_h, node.node_exclusive, node.note) == \
        (cores, billed, exclusive, note)


# Billing rules no fixture above exercises. (sinfo line, scontrol line, the partition)
BILLED = [
    # no weights: SLURM bills one per CPU
    ("cpu|2|32|1|64|(null)", "PartitionName=cpu OverSubscribe=NO",
     (64, 64.0, False, "")),
    # GPUs carry their own weight; memory is billed but not priced, and says so
    ("acc|2|16|2|64|gpu:a100:4(S:0-1)",
     "PartitionName=acc OverSubscribe=EXCLUSIVE TRESBillingWeights=CPU=0.5,Mem=0.1G,GRES/gpu=8",
     (32, 64.0, True, "billing weights not priced: mem")),
]


@pytest.mark.parametrize("sinfo,scontrol,expected", BILLED)
def test_a_billing_rule_is_read_as_slurm_applies_it(sinfo, scontrol, expected):
    from myrmidon.execution.billing import parse_partitions

    (node,) = parse_partitions(sinfo, scontrol)[0].values()
    assert (node.cores_per_node, node.billed_per_node_h, node.node_exclusive, node.note) == \
        expected


# (the job, whether its program is threaded): every srun step gets the job's cores per task
STEPS = [(Resources(ntasks=1, cpus_per_task=48), None), (Resources(ntasks=48), 48)]


@pytest.mark.parametrize("res,ranks", STEPS)
def test_a_job_step_gets_the_cores_per_task_the_job_booked(tmp_path, res, ranks):
    from myrmidon.execution.slurm import SlurmBackend

    spec = JobSpec(argv=["gmx", "mdrun"], cwd=tmp_path, stdout="o", stderr="e",
                   mpi_ranks=ranks, resources=res)
    lines = SlurmBackend().render_script(spec, "job").splitlines()
    export = lines.index("export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}")
    assert export < next(i for i, line in enumerate(lines) if line.startswith("srun"))
