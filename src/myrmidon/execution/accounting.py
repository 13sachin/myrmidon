"""Compute accounting: the number every submission must show before it runs.

Estimates are walltime upper bounds — what the scheduler *may* charge — so a
gate can demand acknowledgment before the job exists. Actual usage comes from
the backend (``ExecutionBackend.usage``) once the job finishes.
"""

from __future__ import annotations

import math

from myrmidon.config import ComputeDefaults
from myrmidon.core.validation import ValidationWarning
from myrmidon.execution.backend import JobSpec, Resources


def cores_booked(resources: Resources, compute: ComputeDefaults) -> int:
    """Cores the scheduler will charge for: the request, rounded up to whole nodes of the
    partition it goes to where that partition allocates nodes whole."""
    _, node = compute.billing(resources.partition)
    requested = max(1, resources.ntasks) * max(1, resources.cpus_per_task)
    if node.node_exclusive and node.cores_per_node > 0:
        nodes = math.ceil(requested / node.cores_per_node)
        return nodes * node.cores_per_node
    return requested


def estimate_core_h(resources: Resources, compute: ComputeDefaults) -> float:
    """What the booking is billed: cores booked × walltime, at the partition's rate per
    core-hour (``config.PartitionCompute``, read from the scheduler by ``init``)."""
    _, node = compute.billing(resources.partition)
    rate = node.billed_per_node_h / node.cores_per_node
    return round(cores_booked(resources, compute) * rate * resources.walltime_min / 60.0, 3)


def describe_booking(resources: Resources, compute: ComputeDefaults) -> str:
    """How a booking's price was arrived at, said the same way by the submission gate and
    the plan: "96 core(s) × 120 min on partition batch (whole nodes)"."""
    name, node = compute.billing(resources.partition)
    rate = node.billed_per_node_h / node.cores_per_node
    return (f"{cores_booked(resources, compute)} core(s) × {resources.walltime_min} min"
            + (f" on partition {name}" if name else "")
            + (" (whole nodes)" if node.node_exclusive else "")
            + (f" at {rate:g} per core-hour" if rate != 1 else ""))


def spec_core_h(spec: JobSpec, compute: ComputeDefaults) -> float:
    """Estimate for a whole JobSpec, including MiMiC coupled programs."""
    res = spec.resources
    ranks = spec.mpi_ranks or 0
    for cc in spec.coupled or []:
        ranks += cc.mpi_ranks or 0
    if ranks > res.ntasks:
        res = res.model_copy(update={"ntasks": ranks})
    return estimate_core_h(res, compute)


def fit_warning(
    res: Resources, compute: ComputeDefaults, backend: str,
) -> ValidationWarning | None:
    """Does this machine have the cores the submission is asking for?

    On a scheduler the answer is the scheduler's business: it queues the job until a
    node with room appears, and asking for more cores than the login node has is
    normal. Without one there is no queue — the ranks all start at once on the cores
    that exist, contend for them, and the run takes longer than doing it with fewer.
    That is the workstation's version of the budget gate biting, and it is worth
    saying before the run rather than after.

    Silent when ``cores_per_node`` is 1, the field's default, which means the config
    does not record how big the machine is. ``myrmidon init`` writes the real count.
    """
    if backend != "local" or compute.cores_per_node <= 1:
        return None
    want = max(1, res.ntasks) * max(1, res.cpus_per_task)
    if want <= compute.cores_per_node:
        return None
    return ValidationWarning(
        id="LOCAL_OVERSUBSCRIBED", severity="warn", needs_human=True,
        message=f"this submission wants {want} cores and the machine has "
        f"{compute.cores_per_node} — with no scheduler to queue behind, the ranks will "
        "contend and the run will be slower than the same work on fewer ranks. Lower the "
        "rank count, or acknowledge to run it anyway",
    )


def budget_warnings(
    res: Resources, compute: ComputeDefaults, backend: str = "slurm",
) -> tuple[float, list[ValidationWarning]]:
    """The core-hour gate every submission runs through: the estimate, and the
    finding it produces — a hard refusal above ``block_core_h``, an
    acknowledge-by-id warning above ``warn_core_h``, else an info line so the
    number is on record before the job exists.

    On a machine with no scheduler it also asks whether the cores being requested
    exist at all (see :func:`fit_warning`)."""
    est = estimate_core_h(res, compute)
    detail = describe_booking(res, compute) + (
        f", billed to account {res.account}" if res.account else "")
    if compute.block_core_h is not None and est > compute.block_core_h:
        w = ValidationWarning(
            id="COMPUTE_BUDGET_EXCEEDED", severity="block",
            message=f"estimated {est} core-h ({detail}) exceeds the machine's hard limit of "
            f"{compute.block_core_h} core-h — split the job or ask the user to raise the limit",
        )
    elif est > compute.warn_core_h:
        w = ValidationWarning(
            id="COMPUTE_BUDGET", severity="warn", needs_human=True,
            message=f"estimated {est} core-h ({detail}) is above the {compute.warn_core_h} "
            "core-h approval threshold — get the user's approval, then acknowledge",
        )
    else:
        w = ValidationWarning(
            id="COMPUTE_ESTIMATE", severity="info", message=f"estimated {est} core-h ({detail})",
        )
    fit = fit_warning(res, compute, backend)
    return est, [w] + ([fit] if fit else [])
