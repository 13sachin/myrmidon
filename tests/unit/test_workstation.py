"""The budget gate, and the machine without a scheduler.

On a cluster an oversized request queues until a node has room, and the budget gate is
the thing that bites. With no scheduler nothing queues: the ranks start together on the
cores that exist and contend. The gate says so before the run, and a person decides.
"""

import pytest

from myrmidon.config import ComputeDefaults
from myrmidon.execution.accounting import budget_warnings, estimate_core_h
from myrmidon.execution.backend import Resources

WORKSTATION = ComputeDefaults(cores_per_node=8, node_exclusive=False, warn_core_h=10.0)
CLUSTER = ComputeDefaults(cores_per_node=48, node_exclusive=True, warn_core_h=10.0)
# cores_per_node = 1 is the old default and means 'unknown', not 'one core'
UNKNOWN = ComputeDefaults(cores_per_node=1, node_exclusive=False, warn_core_h=10.0)

# (request, machine, backend, the findings the budget gate raises)
BUDGET = [
    (Resources(ntasks=8, walltime_min=10), WORKSTATION, "local", {"COMPUTE_ESTIMATE"}),
    (Resources(ntasks=9, walltime_min=10), WORKSTATION, "local",
     {"COMPUTE_ESTIMATE", "LOCAL_OVERSUBSCRIBED"}),
    (Resources(ntasks=2, cpus_per_task=8, walltime_min=10), WORKSTATION, "local",
     {"COMPUTE_ESTIMATE", "LOCAL_OVERSUBSCRIBED"}),
    # fitting the machine and costing more than the threshold are independent
    (Resources(ntasks=8, walltime_min=600), WORKSTATION, "local", {"COMPUTE_BUDGET"}),
    (Resources(ntasks=16, walltime_min=30), WORKSTATION, "local",
     {"COMPUTE_ESTIMATE", "LOCAL_OVERSUBSCRIBED"}),
    # a scheduler queues instead; an unknown machine size raises nothing about the fit
    (Resources(ntasks=480, walltime_min=10), CLUSTER, "slurm", {"COMPUTE_BUDGET"}),
    (Resources(ntasks=64, walltime_min=10), UNKNOWN, "local", {"COMPUTE_BUDGET"}),
]


@pytest.mark.parametrize("request_,machine,backend,expected", BUDGET)
def test_the_budget_gate_names_cost_and_fit(request_, machine, backend, expected):
    est, findings = budget_warnings(request_, machine, backend)
    assert {w.id for w in findings} == expected
    assert est == estimate_core_h(request_, machine)
    for w in findings:
        if w.id in ("COMPUTE_BUDGET", "LOCAL_OVERSUBSCRIBED"):
            assert w.severity == "warn" and w.needs_human  # a person's call, never the agent's


@pytest.mark.parametrize("cpmd_ranks,flagged", [(8, True), (3, False)])
def test_a_coupled_run_asks_for_both_programs_ranks(ws_ctx, cpmd_ranks, flagged):
    from myrmidon.engines.mimic.launch import MimicRunParams, mimic_gate

    ws_ctx.config.compute.cores_per_node = 4
    inp = ws_ctx.workspace.stage("input") / "run.inp"
    inp.write_text("&MIMIC\n  PATHS\n  1\n  /tmp\n&END\n&CPMD\n&END\n")
    (ws_ctx.workspace.stage("input") / "run.tpr").write_bytes(b"\0")

    warnings, checked = mimic_gate(
        ws_ctx,
        MimicRunParams(cpmd_input="01_input/run.inp", tpr="01_input/run.tpr",
                       cpmd_ranks=cpmd_ranks, gromacs_ranks=1),
    )
    assert "machine_fit" in checked
    assert any(w.id == "LOCAL_OVERSUBSCRIBED" for w in warnings) is flagged
