"""SlurmBackend against the mock sbatch/squeue/sacct/scancel binaries."""

import time
from pathlib import Path

import pytest

from myrmidon.execution.backend import CoupledCmd, JobSpec, Resources
from myrmidon.execution.slurm import SlurmBackend


@pytest.fixture
def slurm_env(tmp_path, mock_engines, monkeypatch):
    monkeypatch.setenv("MOCK_SLURM_DIR", str(tmp_path / "slurm-state"))
    return tmp_path


def wait_terminal(backend, handle, spec, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = backend.poll(handle, spec)
        if status.state.terminal:
            return status
        time.sleep(0.05)
    raise AssertionError("job did not reach a terminal state")


def test_usage_accounts_nodes_not_hyperthreads(slurm_env):
    """sacct's AllocCPUS counts hardware threads (96 on a 48-core node with SMT); a
    node-exclusive site bills nodes × cores_per_node."""
    spec = JobSpec(argv=["sh", "-c", "sleep 0.5"], cwd=slurm_env)
    backend = SlurmBackend()
    handle = backend.submit(spec)
    assert backend.usage(handle, spec) is None  # still running: no accounting yet
    wait_terminal(backend, handle, spec)
    assert backend.usage(handle, spec) == round(96 * 120 / 3600, 3)  # raw AllocCPUS
    exclusive = SlurmBackend(cores_per_node=48)
    assert exclusive.usage(handle, spec) == round(48 * 120 / 3600, 3)


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
