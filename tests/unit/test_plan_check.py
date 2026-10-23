"""`plan_check`: the plan against the ledger — did we do what we said?

A plan is a prospective ledger and a ledger is an executed plan: the same list in two
tenses. That is what makes this a diff rather than new machinery, and it is why no
progress pointer was ever needed.

The tests are ordered by how certain each answer is. Protocols and core-hours are exact,
because both sides name an id or a number. Step matching is exact only when a call says
which step it belongs to — nothing sets that today, so the fallback is a sequence
alignment, and the observation says which kind of answer it gave.
"""

import pytest

from myrmidon.core.tool import REGISTRY

STEPS = [
    {"name": "em", "tool": "gmx_grompp", "cites": "inputset:em_solvated_biomolecule@v1",
     "values": {"mdp": "em.mdp", "structure": "structure", "topology": "topology"},
     "needs": ["structure", "topology"], "produces": ["tpr_em"]},
    {"name": "em_run", "tool": "gmx_mdrun",
     "values": {"tpr": "tpr_em", "ntasks": 8, "cpus_per_task": 1, "walltime_min": 60},
     "needs": ["tpr_em"], "produces": ["structure_minimised"]},
]


@pytest.fixture
def planned(ws_ctx):
    obs = REGISTRY.call("plan_write", ws_ctx, {
        "question": "Does this system minimise cleanly?",
        "given": ["structure", "topology"],
        "steps": STEPS,
    })
    assert obs.ok, obs.summary
    return ws_ctx


def _check(ctx):
    obs = REGISTRY.call("plan_check", ctx, {})
    assert obs.ok, obs.summary
    return obs


def test_an_unapproved_plan_says_so(planned):
    """A plan is written freely; nothing runs against its budget until a person agrees."""
    obs = _check(planned)
    assert obs.approval_holds is False
    assert obs.approved_digest is None
    assert "not approved" in obs.summary


def test_an_approved_plan_holds_until_it_changes(planned):
    approved = REGISTRY.call("plan_approve", planned, {
        "acknowledge_warnings": ["COMPUTE_BUDGET"]})
    assert approved.ok, approved.summary

    obs = _check(planned)
    assert obs.approval_holds
    assert obs.approved_digest == obs.digest
    assert "COMPUTE_BUDGET" in obs.acknowledged

    # the digest is computed from the plan's content, so one comparison answers both
    # "was it approved" and "is it still the plan that was approved"
    path = planned.workspace.root / "plan.yaml"
    path.write_text(path.read_text().replace("estimate_core_h: 8.0", "estimate_core_h: 800.0"))

    after = _check(planned)
    assert after.approval_holds is False
    assert after.approved_digest is None
    assert "changed after approval" in after.summary
    assert after.planned_core_h == 800.0  # and the read is of what is on disk now


def test_reading_the_plan_is_a_look_not_a_step(planned):
    """It must not appear in the study's own record as work."""
    _check(planned)
    assert not any(
        ev.tool == "plan_check" for ev in planned.workspace.ledger.events()
    )



# A plan is scoped to its question: a call made for another question in the
# same study, or before the plan existed, carries out none of its steps and books nothing
# against it.
NVT = [
    {"name": "grompp", "tool": "gmx_grompp", "needs": ["structure", "topology"],
     "produces": ["tpr"],
     "values": {"mdp": "nvt.mdp", "structure": "structure", "topology": "topology"}},
    {"name": "mdrun", "tool": "gmx_mdrun", "needs": ["tpr"], "produces": ["trajectory"],
     "values": {"tpr": "tpr", "ntasks": 1, "cpus_per_task": 1, "walltime_min": 6}},
]


def _nvt(ctx, question, **over):
    obs = REGISTRY.call("plan_write", ctx, {
        "question": question, "given": ["structure", "topology"], "steps": NVT, **over})
    assert obs.ok, obs.summary
    return obs


def _run_nvt(ctx):
    stage = ctx.workspace.stage("input")
    (stage / "conf.gro").write_text(
        "mock\n 3\n" + "    1ALA     CA    1   0.1   0.2   0.3\n" * 3 + "   2.5 2.5 2.5\n")
    (stage / "topol.top").write_text("[ system ]\nMock\n[ molecules ]\nProtein 1\n")
    (stage / "nvt.mdp").write_text(
        "integrator = md\ndt = 0.002\nnsteps = 500\nconstraints = h-bonds\n"
        "constraint_algorithm = lincs\ntcoupl = v-rescale\ntc-grps = System\n"
        "tau_t = 0.1\nref_t = 300\ncoulombtype = PME\ngen_vel = yes\ngen_seed = 1\n")
    grompp = REGISTRY.call("gmx_grompp", ctx, {
        "mdp": "01_input/nvt.mdp", "structure": "01_input/conf.gro",
        "topology": "01_input/topol.top", "output": "nvt.tpr", "stage": "nvt"})
    assert grompp.ok, grompp.summary
    tpr = next(a.path for a in grompp.artifacts if a.path.endswith(".tpr"))
    run = REGISTRY.call("gmx_mdrun", ctx, {"tpr": tpr, "deffnm": "nvt", "stage": "nvt",
                                           "walltime_min": 6})
    assert run.ok, run.summary


# (the plan the calls were made under: None, or its question and whether it was approved;
# the question of the plan checked afterwards; whether it says it revises; steps done;
# core-h booked for it; matched by step name)
SCOPED = [
    (None, "NVT of the solvated box?", False, [], 0.0, False),
    (("NVT of the solvated box?", False), "QM/MM of the solute?", False, [], 0.0, False),
    (("NVT of the solvated box?", False), "NVT of the solvated box?", False,
     ["grompp", "mdrun"], 0.1, False),
    (("NVT of the solvated box?", False), "A 100 ps NVT of the box?", True,
     ["grompp", "mdrun"], 0.1, False),
    # an approved plan binds the calls, and they name the steps they carried out
    (("NVT of the solvated box?", True), "NVT of the solvated box?", False,
     ["grompp", "mdrun"], 0.1, True),
]


@pytest.mark.parametrize("made_under,question,revises,done,booked,exact", SCOPED)
def test_a_plan_is_carried_out_only_by_its_own_calls(ws_ctx, made_under, question, revises,
                                                     done, booked, exact):
    if made_under is not None:
        _nvt(ws_ctx, made_under[0])
        if made_under[1]:
            assert REGISTRY.call("plan_approve", ws_ctx, {}).ok
    _run_nvt(ws_ctx)
    if made_under is None or question != made_under[0] or revises:
        _nvt(ws_ctx, question, overwrite=made_under is not None, revises=revises)

    obs = _check(ws_ctx)
    assert obs.done == done
    assert obs.booked_core_h == booked
    assert obs.exact is exact
    events = [ev for ev in ws_ctx.workspace.ledger.events() if ev.tool == "gmx_mdrun"]
    assert (events[-1].plan_step == "mdrun") is exact
    assert (events[-1].plan_id == obs.plan_id) is bool(done)


# Reuse is declared: an earlier run may carry out a step of a new question's
# plan, but only when the step names it, and the plan then says so. Offered as a candidate,
# never filled in. (which earlier call each step reuses, by the tool that made it, or an
# event id; the finding; steps reused; steps still pending)
DECLARED = [
    ({}, None, [], ["grompp", "mdrun"]),
    ({"grompp": "gmx_grompp", "mdrun": "gmx_mdrun"}, None, ["grompp", "mdrun"], []),
    ({"grompp": "gmx_grompp"}, None, ["grompp"], ["mdrun"]),
    # a step reuses a call of its own tool, which exists and succeeded
    ({"mdrun": "gmx_grompp"}, "PLAN_REUSE_INVALID", None, None),
    ({"mdrun": "ev-000000000000"}, "PLAN_REUSE_INVALID", None, None),
]


@pytest.mark.parametrize("reuse,finding,reused,pending", DECLARED)
def test_a_step_uses_an_earlier_run_only_by_naming_it(ws_ctx, reuse, finding, reused,
                                                      pending):
    _run_nvt(ws_ctx)  # an earlier question, asked before this plan existed
    offered = _nvt(ws_ctx, "QM/MM of the solute?")
    assert {c.step: bool(c.job) for c in offered.reuse_candidates} == \
        {"grompp": False, "mdrun": True}  # the run is named by its job too
    event = {c.tool: c.event for c in offered.reuse_candidates}
    steps = [dict(s, reuse=event.get(reuse[s["name"]], reuse[s["name"]]))
             if s["name"] in reuse else s for s in NVT]
    written = REGISTRY.call("plan_write", ws_ctx, {
        "question": "QM/MM of the solute?", "given": ["structure", "topology"],
        "steps": steps, "overwrite": True})
    if finding:
        assert not written.ok
        assert [w.id for w in written.blocking] == [finding]
        return
    assert written.ok, written.summary
    assert {c.step for c in written.reuse_candidates} == set(pending)

    obs = _check(ws_ctx)
    assert (obs.reused, obs.pending, obs.done) == (reused, pending, [])
    assert obs.booked_core_h == 0.0  # what an earlier question booked is not this plan's
