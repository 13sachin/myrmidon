"""The budget gate, and the machine without a scheduler.

On a cluster an oversized request queues until a node has room, and the budget gate is
the thing that bites. With no scheduler nothing queues: the ranks start together on the
cores that exist and contend. The gate says so before the run, and a person decides.
"""

import pytest

from myrmidon.config import ComputeDefaults
from myrmidon.execution.accounting import budget_warnings, estimate_core_h
from myrmidon.execution.backend import Resources

WORKSTATION = ComputeDefaults(cores_per_node=8, node_exclusive=False)
CLUSTER = ComputeDefaults(cores_per_node=48, node_exclusive=True, block_core_h=1000.0)
# cores_per_node = 1 is the old default and means 'unknown', not 'one core'
UNKNOWN = ComputeDefaults(cores_per_node=1, node_exclusive=False)

# (request, machine, backend, the findings the budget gate raises)
BUDGET = [
    (Resources(ntasks=8, walltime_min=10), WORKSTATION, "local", {"COMPUTE_ESTIMATE"}),
    (Resources(ntasks=9, walltime_min=10), WORKSTATION, "local",
     {"COMPUTE_ESTIMATE", "LOCAL_OVERSUBSCRIBED"}),
    (Resources(ntasks=2, cpus_per_task=8, walltime_min=10), WORKSTATION, "local",
     {"COMPUTE_ESTIMATE", "LOCAL_OVERSUBSCRIBED"}),
    # fitting the machine and costing much are independent: the ceiling judges the cost
    (Resources(ntasks=8, walltime_min=600), WORKSTATION, "local", {"COMPUTE_ESTIMATE"}),
    (Resources(ntasks=16, walltime_min=30), WORKSTATION, "local",
     {"COMPUTE_ESTIMATE", "LOCAL_OVERSUBSCRIBED"}),
    # a scheduler queues instead; an unknown machine size raises nothing about the fit
    (Resources(ntasks=480, walltime_min=10), CLUSTER, "slurm", {"COMPUTE_ESTIMATE"}),
    (Resources(ntasks=4800, walltime_min=20), CLUSTER, "slurm", {"COMPUTE_BUDGET_EXCEEDED"}),
    (Resources(ntasks=64, walltime_min=10), UNKNOWN, "local", {"COMPUTE_ESTIMATE"}),
]


@pytest.mark.parametrize("request_,machine,backend,expected", BUDGET)
def test_the_budget_gate_names_cost_and_fit(request_, machine, backend, expected):
    est, findings = budget_warnings(request_, machine, backend)
    assert {w.id for w in findings} == expected
    assert est == estimate_core_h(request_, machine)
    for w in findings:
        if w.id in ("COMPUTE_BUDGET_EXCEEDED", "LOCAL_OVERSUBSCRIBED"):
            assert w.severity == ("block" if w.id == "COMPUTE_BUDGET_EXCEEDED" else "info")


@pytest.mark.parametrize("cpmd_ranks,flagged", [(8, True), (3, False)])
def test_a_coupled_run_asks_for_both_programs_ranks(ws_ctx, cpmd_ranks, flagged):
    from myrmidon.execution.gate import submission_gate
    from myrmidon.tools.stage import StageRunParams

    ws_ctx.config.compute.cores_per_node = 4
    ws_ctx.config.backend = "local"

    warnings, checked = submission_gate(
        ws_ctx,
        StageRunParams(stage="run", walltime_min=30, layout={
            "ntasks": cpmd_ranks, "cpus_per_task": 1, "partner_ntasks": 1}),
        "stage_run",
    )
    assert "machine_fit" in checked
    assert any(w.id == "LOCAL_OVERSUBSCRIBED" for w in warnings) is flagged


# (a machine's cores, the system's QM cell edge and cutoff, MM atoms, the layout it resolves
# to: CPMD ranks, threads per rank, GROMACS ranks). CPMD's ranks stay within the mesh's planes
# along x; GROMACS's are a count its decomposition factors; every core of a node is used.
COUPLED_LAYOUTS = [
    (8, "18.6", "70", 1500, (7, 1, 1)),
    (48, "18.6", "70", 1500, (45, 1, 3)),
    (128, "18.6", "70", 1500, (61, 2, 3)),
    (128, "18.6", "70", 50000, (54, 2, 10)),
    (256, "18.6", "70", 50000, (54, 4, 10)),
    (256, "60.0", "70", 50000, (214, 1, 42)),
    (2, "18.6", "70", 1500, (1, 1, 1)),
]


@pytest.mark.parametrize("cores,cell,cutoff,atoms,layout", COUPLED_LAYOUTS)
def test_a_coupled_run_divides_its_cores_as_its_engines_can_use_them(cores, cell, cutoff, atoms,
                                                                     layout):
    from myrmidon.engines.gromacs.layout import smooth_at_most
    from myrmidon.engines.mimic.layout import fft_planes, resolve_layout

    found = resolve_layout(cores, {"cell_bohr": cell, "cutoff_ry": cutoff, "atoms": atoms})
    assert (found.ntasks, found.cpus_per_task, found.partner_ntasks) == layout
    assert found.ntasks <= fft_planes(float(cell), float(cutoff))
    assert smooth_at_most(found.partner_ntasks, [2, 3, 5, 7]) == found.partner_ntasks
    assert found.cores <= cores


# the REAL SPACE MESH a real CPMD log states at 70 Ry for these cell edges
@pytest.mark.parametrize("edge,planes", [(18.6, 100), (21.0, 112), (22.9, 126),
                                         (18.6 * 1.1, 112), (18.6 * 0.9, 90)])
def test_the_planes_along_a_cell_edge_are_the_ones_cpmd_lays_out(edge, planes):
    from myrmidon.engines.mimic.layout import fft_planes

    assert fft_planes(edge, 70.0) == planes


# (cores, atoms, integrator, ranks, threads): every core a rank down to the atoms-per-rank
# floor, the rest threads; the one integrator that runs on one rank gets them all as threads
GROMACS_LAYOUTS = [
    (8, 100000, None, 8, 1),
    (48, 2000, None, 3, 16),
    (128, 2000, None, 4, 32),
    (128, 100000, None, 128, 1),
    (256, 100000, None, 128, 2),
    (128, 100000, "l-bfgs", 1, 128),
    (128, None, None, 128, 1),
]


@pytest.mark.parametrize("cores,atoms,integrator,ranks,threads", GROMACS_LAYOUTS)
def test_a_classical_run_gets_the_ranks_its_atoms_can_use(cores, atoms, integrator, ranks,
                                                          threads):
    from myrmidon.engines.gromacs.layout import resolve_layout

    found = resolve_layout(cores, {"atoms": atoms, "integrator": integrator})
    assert (found.ntasks, found.cpus_per_task) == (ranks, threads)
    assert found.cores <= cores


def test_a_walltime_the_partition_cannot_grant_is_refused(tmp_path):
    from myrmidon.config import ComputeDefaults, PartitionCompute
    from myrmidon.execution.accounting import budget_warnings
    from myrmidon.execution.backend import Resources

    compute = ComputeDefaults(partitions={"devel": PartitionCompute(
        cores_per_node=128, billed_per_node_h=128.0, max_walltime_min=120)})
    within = budget_warnings(Resources(walltime_min=120, partition="devel"), compute)[1]
    beyond = budget_warnings(Resources(walltime_min=121, partition="devel"), compute)[1]
    assert "WALLTIME_EXCEEDS_PARTITION_LIMIT" not in {f.id for f in within}
    refused = next(f for f in beyond if f.id == "WALLTIME_EXCEEDS_PARTITION_LIMIT")
    assert refused.severity == "block"
