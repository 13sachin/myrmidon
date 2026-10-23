"""`plan_write`: the verb that puts a study's plan on disk, and what it refuses.

A plan says what a study is doing, in what order, for how much. These are the checks that
justify the verb's existence — every one of them is something a plan can be wrong about
before a single core-hour is spent.
"""


import pytest

from myrmidon.config import PartitionCompute
from myrmidon.core.tool import REGISTRY

STEPS = [
    {"name": "minimise", "tool": "gmx_grompp",
     "cites": "inputset:em_solute_in_water@v1",
     "values": {"mdp": "em.mdp", "structure": "structure", "topology": "topology"},
     "needs": ["structure", "topology"], "produces": ["tpr_min"]},
    {"name": "minimise_run", "tool": "gmx_mdrun",
     "values": {"tpr": "tpr_min", "ntasks": 1, "cpus_per_task": 1, "walltime_min": 30},
     "needs": ["tpr_min"], "produces": ["structure_minimised"]},
    {"name": "heat", "tool": "gmx_grompp", "cites": "inputset:npt_equilibration_solute_in_water@v2",
     "values": {"mdp": "npt.mdp", "structure": "structure_minimised", "topology": "topology",
                "nsteps": 100000},
     "needs": ["structure_minimised", "topology"], "produces": ["tpr_heat"]},
    {"name": "heat_run", "tool": "gmx_mdrun",
     "values": {"tpr": "tpr_heat", "ntasks": 16, "cpus_per_task": 1, "walltime_min": 60},
     "needs": ["tpr_heat"], "produces": ["structure_equilibrated"]},
]


def _write(ctx, **over):
    call = {
        "question": "Is acetone in water equilibrated after 200 ps?",
        "system": "one acetone in TIP3P water",
        "given": ["structure", "topology"],
        "steps": STEPS,
        **over,
    }
    return REGISTRY.call("plan_write", ctx, call)


# What a plan can be wrong about before a core-hour is spent: (what is changed, the
# finding, where it holds the write). Order is data flow, so a gap in it is a finding — a
# step deleted from an edited plan, or a QM/MM step reading an equilibrated structure that
# nothing makes. The harness has no opinion on *which* order; only that it can run.
REFUSALS = [
    (dict(steps=[s for s in STEPS if s["name"] != "minimise_run"]),
     "PLAN_STEP_INPUT_UNSATISFIED", "blocking"),
    (dict(question="Run QM/MM on acetone", steps=[
        {"name": "qmmm", "tool": "mimic_run",
         "needs": ["structure_equilibrated", "cpmd_template"],
         "produces": ["trajectory_qmmm"], "estimate_core_h": 40.0}]),
     "PLAN_STEP_INPUT_UNSATISFIED", "blocking"),
    # a tool parameter beside `values` instead of in it would be dropped silently
    (dict(steps=STEPS[:1] + [dict(STEPS[1], walltime_min=20)] + STEPS[2:]),
     "PLAN_STEP_UNKNOWN_FIELD", "blocking"),
    (dict(steps=[dict(STEPS[0], cites="inputset:em_solute_in_water@v7")] + STEPS[1:]),
     "PLAN_CITES_OTHER_VERSION", "unacknowledged"),
    # `arms.varies` has to name a real key, or it exempts nothing and says nothing
    (dict(arms={"varies": "temperature", "over": [300, 310], "because": "density"}),
     "PLAN_AXIS_NAMES_NOTHING", "unacknowledged"),
]


@pytest.mark.parametrize("change,finding,held_in", REFUSALS)
def test_a_plan_that_cannot_run_as_written_is_not_written(ws_ctx, change, finding, held_in):
    refused = _write(ws_ctx, **change)
    assert not refused.ok
    assert finding in {w.id for w in getattr(refused, held_in)}
    assert not (ws_ctx.workspace.root / "plan.yaml").exists()


def test_writing_a_plan_is_never_gated_on_cost_and_runs_nothing(ws_ctx):
    """Proposing is not approving: 16.5 core-h is over the threshold,
    and writing the plan anyway is the point — it is what a person needs in front of them
    before they can approve anything. Writing books no compute and submits no job."""
    obs = _write(ws_ctx)
    assert obs.ok, obs.summary
    assert "COMPUTE_ESTIMATE" in {w.id for w in obs.advisory}
    assert not any(w.needs_human for w in obs.warnings + obs.advisory)
    events = list(ws_ctx.workspace.ledger.events())
    assert events[-1].tool == "plan_write"
    assert not any(ev.tool.startswith(("gmx_mdrun", "mimic_run")) for ev in events)


def test_approving_is_the_act_that_needs_a_person(ws_ctx):
    assert _write(ws_ctx).ok
    refused = REGISTRY.call("plan_approve", ws_ctx, {})
    assert not refused.ok
    budget = next(w for w in refused.unacknowledged if w.id == "COMPUTE_BUDGET")
    assert budget.needs_human

    ok = REGISTRY.call("plan_approve", ws_ctx, {"acknowledge_warnings": ["COMPUTE_BUDGET"]})
    assert ok.ok, ok.summary
    event = next(
        ev for ev in ws_ctx.workspace.ledger.events() if ev.tool == "plan_approve" and ev.ok
    )
    assert "COMPUTE_BUDGET" in event.warnings_acknowledged


def test_approving_a_plan_that_changed_since_you_read_it_is_refused(ws_ctx):
    """The digest is what a person approved; approving a different one is approving
    something nobody showed them."""
    first = _write(ws_ctx)
    assert first.ok
    longer = [dict(s) for s in STEPS]
    longer[-1] = dict(longer[-1], values={**longer[-1]["values"], "walltime_min": 240})
    assert _write(ws_ctx, steps=longer, overwrite=True).ok

    refused = REGISTRY.call("plan_approve", ws_ctx, {
        "digest": first.digest, "acknowledge_warnings": ["COMPUTE_BUDGET"]})
    assert not refused.ok
    assert [w.id for w in refused.blocking] == ["PLAN_CHANGED_SINCE_YOU_READ_IT"]


# A plan's cost is computed from the resources each step states, by the function a
# submission is priced with — never a figure the model gives. (the run step's
# values, the figure it is given, the machine it is written on and the one it is approved
# on, its cost, the finding at writing, the block at approval)
SMALL, NODES = {}, {"cores_per_node": 48, "node_exclusive": True}
# a scheduler's partitions, as `init` reads them: a booking goes to the default here
GPUS = {"partitions": {"batch": PartitionCompute(cores_per_node=48, billed_per_node_h=48.0),
                       "gpus": PartitionCompute(cores_per_node=40, billed_per_node_h=40.0)},
        "default_partition": "gpus"}
# a shared partition that bills each core twice
DEAR = {"partitions": {"shared": PartitionCompute(cores_per_node=48, billed_per_node_h=96.0,
                                                  node_exclusive=False)},
        "default_partition": "shared"}
RUN = {"tpr": "tpr", "ntasks": 16, "cpus_per_task": 1, "walltime_min": 60}
PRICED = [
    (RUN, None, SMALL, SMALL, 16.0, None, None),
    (RUN, 4.0, SMALL, SMALL, 16.0, "PLAN_ESTIMATE_COMPUTED", None),
    # a booking rounds up to whole nodes where nodes are exclusive
    (dict(RUN, ntasks=4), None, NODES, NODES, 48.0, None, None),
    # priced on the nodes of the partition it goes to, as that partition bills them
    (RUN, None, GPUS, GPUS, 40.0, None, None),
    (RUN, None, DEAR, DEAR, 32.0, None, None),
    # a tool default nobody chose is an assumption, not a resource
    ({"tpr": "tpr", "walltime_min": 60}, 16.0, SMALL, SMALL, None, "PLAN_STEP_UNRESOLVED",
     "PLAN_NOT_EXECUTABLE"),
    # costs computed on one machine are not this one's
    (RUN, None, SMALL, NODES, 16.0, None, "PLAN_COST_STALE"),
]


@pytest.mark.parametrize("values,given,written_on,approved_on,cost,finding,block", PRICED)
def test_a_plan_costs_what_its_resources_cost(ws_ctx, values, given, written_on,
                                              approved_on, cost, finding, block):
    compute = ws_ctx.config.compute
    step = {"name": "run", "tool": "gmx_mdrun", "values": values, "needs": ["tpr"],
            **({"estimate_core_h": given} if given is not None else {})}
    for key, value in written_on.items():
        setattr(compute, key, value)
    obs = _write(ws_ctx, steps=[step], given=["tpr"])
    assert obs.ok, obs.summary
    assert obs.steps[0].get("estimate_core_h") == cost
    if finding:
        assert finding in {w.id for w in obs.advisory}

    for key, value in approved_on.items():
        setattr(compute, key, value)
    approved = REGISTRY.call("plan_approve", ws_ctx, {"acknowledge_warnings": ["COMPUTE_BUDGET"]})
    assert approved.ok is (block is None), approved.summary
    assert [w.id for w in getattr(approved, "blocking", [])] == ([block] if block else [])


# Approvable means executable: a plan may not be approved while a value its
# execution will need is missing. (the step, what is unresolved as (key, kind))
SELECT = {"name": "qm", "tool": "mimic_select_qm", "cites": "inputset:mimic_qm_region@v3",
          "needs": ["topology", "structure"],
          "values": {"topology": "topology", "coords": "structure"}}
PRODUCE = {"name": "produce", "tool": "mimic_run", "cites": "inputset:mimic_cpmd_production@v3",
           "needs": ["tpr_mimic", "cpmd_input"],
           "values": {"cpmd_input": "cpmd_input", "tpr": "tpr_mimic"}}
DECIDED = {"&CPMD/MAXSTEP": 2000,
           "cpmd_ranks": 8, "gromacs_ranks": 2, "walltime_min": 120}
UNRESOLVED = [
    # the selection is the researcher's; the padding is derivable, so somebody must derive it
    (SELECT, {("selections", "researcher"), ("padding_nm", "parameter")}),
    (dict(SELECT, values={**SELECT["values"], "selections": ["add resname is ACT"],
                          "padding_nm": 0.4}), set()),
    # the tool's own declaration holds whatever the step cites: chemistry stays the
    # researcher's when the step cites the file it renders
    (dict(SELECT, cites="inputset:mimic_cpmd_template@v1"),
     {("selections", "researcher"), ("padding_nm", "parameter")}),
    ({"name": "prepare", "tool": "mimic_prepare", "needs": ["cpmd_input_qm"],
      "values": {"cpmd_input": "cpmd_input_qm"}},
     {("total_charge", "researcher"), ("multiplicity", "researcher")}),
    # rendering an input set names it by citing it
    ({"name": "template", "tool": "inputset_apply", "cites": "inputset:mimic_cpmd_template@v1",
      "needs": []}, set()),
    # what the input set leaves to the researcher, and the resources the cost comes from
    (PRODUCE, {("&CPMD/MAXSTEP", "researcher"),
               ("cpmd_ranks, gromacs_ranks, walltime_min", "resources")}),
    (dict(PRODUCE, values={**PRODUCE["values"], **DECIDED}), set()),
    # a file parameter is stated by the role that feeds it
    (dict(PRODUCE, values=DECIDED), {("cpmd_input", "file"), ("tpr", "file")}),
    # a value the cited input set states is stated
    ({"name": "box", "tool": "gmx_editconf", "cites": "inputset:box_qmmm_solute@v2",
      "needs": ["structure"], "values": {"structure": "structure"}},
     {("distance_nm", "parameter")}),
]


@pytest.mark.parametrize("step,unresolved", UNRESOLVED)
def test_a_plan_is_approvable_only_when_it_is_executable(ws_ctx, step, unresolved):
    obs = _write(ws_ctx, steps=[step], given=step["needs"])
    assert obs.ok, obs.summary
    assert {(u.key, u.kind) for u in obs.unresolved} == unresolved
    approved = REGISTRY.call("plan_approve", ws_ctx, {"acknowledge_warnings": ["COMPUTE_BUDGET"]})
    assert approved.ok is not unresolved, approved.summary
    assert [w.id for w in getattr(approved, "blocking", [])] == \
        (["PLAN_NOT_EXECUTABLE"] if unresolved else [])


# An approved plan binds the call that carries out its step: a value the call
# leaves unset is the plan's, a value it sets otherwise is a departure a person decides. An
# input set rendered for a step that cites it takes that step's values as its overrides, so
# a decision collected at planning reaches the file the run reads.
# (the step, approve the plan?, the call's overrides, ok?, finding, overrides recorded)
APPLY = {"name": "mdp", "tool": "inputset_apply",
         "cites": "inputset:em_solute_in_water@v1", "produces": ["mdp"],
         "values": {"inputset": "em_solute_in_water", "overrides": {"nstenergy": 500}}}
CITING = {"name": "heat", "tool": "gmx_grompp",
          "cites": "inputset:npt_equilibration_solute_in_water@v2",
          "needs": ["structure", "topology"], "produces": ["tpr_heat"],
          "values": {"mdp": "npt.mdp", "structure": "structure", "topology": "topology",
                     "nsteps": 50000}}
STAGE_METHOD = {"name": "production_method", "tool": "inputset_apply",
                "cites": "inputset:mimic_cpmd_production@v3", "produces": ["cpmd_method"],
                "values": {"&CPMD/MAXSTEP": 2000}}
BOUND = [
    (APPLY, False, {"nstenergy": 10}, True, None, {"nstenergy": 10}),
    (APPLY, True, None, True, "PLAN_VALUES_APPLIED", {"nstenergy": 500}),
    (APPLY, True, {"nstenergy": 10}, False, "PLAN_STEP_DIFFERS", None),
    (CITING, True, None, True, "PLAN_VALUES_APPLIED", {"nsteps": 50000}),
    (CITING, True, {"nsteps": 10}, False, "PLAN_STEP_DIFFERS", None),
    # a step that renders a stage's method carries that set's keys as its own values
    (STAGE_METHOD, True, None, True, "PLAN_VALUES_APPLIED", {"&CPMD/MAXSTEP": 2000}),
]


@pytest.mark.parametrize("step,approve,overrides,ok,finding,recorded", BOUND)
def test_an_approved_plan_binds_the_call_that_carries_out_its_step(
        ws_ctx, step, approve, overrides, ok, finding, recorded):
    assert _write(ws_ctx, steps=[step], given=step.get("needs", [])).ok
    if approve:
        assert REGISTRY.call("plan_approve", ws_ctx, {}).ok
    inputset = step["cites"].removeprefix("inputset:").partition("@")[0]
    call = {"inputset": inputset,
            **({"overrides": overrides} if overrides is not None else {})}
    obs = REGISTRY.call("inputset_apply", ws_ctx, call)
    assert obs.ok == ok, obs.summary
    if finding == "PLAN_STEP_DIFFERS":
        differs = next(w for w in obs.unacknowledged if w.id == finding)
        assert differs.needs_human
    elif finding:
        assert finding in {w.id for w in obs.advisory}
    if recorded is not None:
        event = [ev for ev in ws_ctx.workspace.ledger.events()
                 if ev.tool == "inputset_apply"][-1]
        assert event.params["overrides"] == recorded


# A plan names files before they exist. (the plan's steps as (tool, path), the files the
# study has, the call's path or None to leave it out, ok?, the finding). A name no file has
# is a guess the call may correct; a real file is the approved one; a required value left
# out is the plan's; a call is bound to the step it fits, not the first of its tool.
GUESSED = [
    ([("file_read", "topol.top")], ["01_prep/processed.top"], "01_prep/processed.top",
     True, "PLAN_VALUE_GUESSED"),
    ([("file_read", "01_prep/a.top")], ["01_prep/a.top", "01_prep/b.top"], "01_prep/b.top",
     False, "PLAN_STEP_DIFFERS"),
    ([("file_read", "01_prep/a.top")], ["01_prep/a.top"], None, True, "PLAN_VALUES_APPLIED"),
    ([("file_read", "01_prep/a.top"), ("file_read", "01_prep/b.top")],
     ["01_prep/a.top", "01_prep/b.top"], "01_prep/b.top", True, None),
]


@pytest.mark.parametrize("steps,files,path,ok,finding", GUESSED)
def test_a_plan_value_that_names_no_file_is_a_guess_not_an_approval(
        ws_ctx, steps, files, path, ok, finding):
    root = ws_ctx.workspace.root
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text("[ system ]\n")
    plan_steps = [{"name": f"read{i}", "tool": tool, "values": {"path": p}}
                  for i, (tool, p) in enumerate(steps)]
    assert _write(ws_ctx, steps=plan_steps, given=[]).ok
    assert REGISTRY.call("plan_approve", ws_ctx, {}).ok
    obs = REGISTRY.call("file_read", ws_ctx, {} if path is None else {"path": path})
    assert obs.ok == ok, obs.summary
    found = {w.id for key in ("advisory", "warnings", "unacknowledged")
             for w in getattr(obs, key, None) or []}
    assert (finding in found) if finding else not found & {
        "PLAN_STEP_DIFFERS", "PLAN_VALUE_GUESSED"}


# A value that is a fact about an earlier step's run is approved as a rule and filled in from
# that run's parsed output once it has completed: heating starts from the
# temperature annealing actually ended at. (the reference heating's ramp makes, the finding)
ANNEAL = {"name": "anneal", "tool": "mimic_run", "needs": ["tpr", "cpmd_input"],
          "values": {"cpmd_input": "cpmd_input", "tpr": "tpr", "cpmd_ranks": 1,
                     "gromacs_ranks": 1, "walltime_min": 10}}


def _heat(ramp: str) -> dict:
    return {"name": "heat", "tool": "inputset_apply",
            "cites": "inputset:mimic_cpmd_heating@v3", "produces": ["cpmd_method_heating"],
            "values": {"&CPMD/TEMPERATURE RAMP": ramp,
                       "&CPMD/BERENDSEN IONS": "${anneal.final_temperature_k} 5000"}}


REFERENCED = [
    ("${anneal.final_temperature_k} 300 50", None),
    ("${anneal.tail_mean_temperature_k} 300 50", None),
    ("${annealing.final_temperature_k} 300 50", "PLAN_REFERENCE_INVALID"),  # no such step
    ("${heat.final_temperature_k} 300 50", "PLAN_REFERENCE_INVALID"),  # not an earlier one
    ("${anneal.final_pressure} 300 50", "PLAN_REFERENCE_INVALID"),  # no such fact
]


@pytest.mark.parametrize("ramp,finding", REFERENCED)
def test_a_reference_names_an_earlier_runs_fact(ws_ctx, ramp, finding):
    obs = _write(ws_ctx, steps=[ANNEAL, _heat(ramp)], given=["tpr", "cpmd_input"])
    if finding:
        assert not obs.ok
        assert {w.id for w in obs.blocking} == {finding}
    else:
        assert obs.ok, obs.summary
        assert not obs.unresolved  # a reference states the value: the plan is executable


def test_a_reference_is_filled_from_the_run_once_it_has_completed(ws_ctx):
    from tests.unit.test_mimic import _full_mimic_setup, _wait

    from myrmidon.core.report import run_facts, run_output

    run_inp, tpr = _full_mimic_setup(ws_ctx)
    anneal = dict(ANNEAL, values={**ANNEAL["values"], "cpmd_input": run_inp, "tpr": tpr})
    assert _write(ws_ctx, steps=[anneal, _heat("${anneal.final_temperature_k} 300 50")],
                  given=["tpr", "cpmd_input"]).ok
    assert REGISTRY.call("plan_approve", ws_ctx, {}).ok
    heat = {"inputset": "mimic_cpmd_heating", "output": "05_heat/heating.inp"}

    early = REGISTRY.call("inputset_apply", ws_ctx, heat)  # annealing has not run
    assert not early.ok
    assert "PLAN_REFERENCE_UNRESOLVED" in {w.id for w in early.blocking}

    job = REGISTRY.call("mimic_run", ws_ctx, {})
    _wait(ws_ctx, job.job_id)
    out, _ = run_output(ws_ctx, job.provenance_id)
    final = run_facts(out.read_text())["final_temperature_k"]
    obs = REGISTRY.call("inputset_apply", ws_ctx, heat)
    assert obs.ok, obs.summary
    assert "PLAN_REFERENCE_RESOLVED" in {w.id for w in obs.advisory}
    event = [ev for ev in ws_ctx.workspace.ledger.events() if ev.tool == "inputset_apply"][-1]
    assert event.params["overrides"]["&CPMD/TEMPERATURE RAMP"] == f"{final:.6g} 300 50"


# A file an earlier step makes reaches a step only through a parameter it is wired to
#: a stage's input made with no `method` would keep the annealing method
# without a word. (the input step's extra values, what is unresolved as (key, kind))
METHOD = {"name": "heat_method", "tool": "inputset_apply",
          "cites": "inputset:mimic_cpmd_heating@v3", "produces": ["cpmd_method_heating"],
          "values": {"&CPMD/TEMPERATURE RAMP": "5 300 50", "&CPMD/BERENDSEN IONS": "5 5000"}}
INPUT = {"name": "heat_input", "tool": "mimic_prepare",
         "needs": ["cpmd_input_qm", "cpmd_method_heating"], "produces": ["cpmd_input_heating"],
         "values": {"cpmd_input": "cpmd_input_qm", "total_charge": 0, "multiplicity": 1}}
WIRED = [
    ({}, {("cpmd_method_heating", "wiring")}),
    ({"method": "cpmd_method_heating"}, set()),
]


@pytest.mark.parametrize("extra,unresolved", WIRED)
def test_a_file_an_earlier_step_makes_is_wired_to_a_parameter(ws_ctx, extra, unresolved):
    step = dict(INPUT, values={**INPUT["values"], **extra})
    obs = _write(ws_ctx, steps=[METHOD, step], given=["cpmd_input_qm"])
    assert obs.ok, obs.summary
    assert {(u.key, u.kind) for u in obs.unresolved} == unresolved


def test_a_call_that_leaves_a_wired_file_out_is_refused(ws_ctx):
    step = dict(INPUT, values={**INPUT["values"], "method": "cpmd_method_heating"})
    assert _write(ws_ctx, steps=[METHOD, step], given=["cpmd_input_qm"]).ok
    assert REGISTRY.call("plan_approve", ws_ctx, {}).ok
    stage = ws_ctx.workspace.stage("qmmm")
    (stage / "cpmd.inp").write_text("&MIMIC\n&END\n")
    obs = REGISTRY.call("mimic_prepare", ws_ctx, {
        "cpmd_input": f"{ws_ctx.workspace.relpath(stage)}/cpmd.inp"})  # no method
    assert not obs.ok
    assert "PLAN_ROLE_UNNAMED" in {w.id for w in obs.blocking}
