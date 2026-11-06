"""The gate every call that books resources passes: where it runs, and what it costs.

The registry applies it to any tool whose params state ``resources()``, judging its
findings with the tool's own checks, so no submitting tool can leave the core-hour check
out. A job runs under the
researcher's approved mandate (core/mandate.py) and within its ceiling, less what the
study's jobs under it have already booked; outside one, only a person's own call submits
it.
"""

from __future__ import annotations

from myrmidon.core import mandate as mandates
from myrmidon.core.lineage import submission_findings
from myrmidon.core.provenance import CALL_MANDATE
from myrmidon.core.tool import ToolContext
from myrmidon.core.validation import Finding, Theirs
from myrmidon.execution.accounting import budget_warnings
from myrmidon.execution.jobs import booked_under_mandates
from myrmidon.execution.placement import place


def submission_gate(ctx: ToolContext, params, tool: str
                    ) -> tuple[list[Finding], list[str]]:
    """A call whose params allow the session (``SESSION_OK``) and fit its envelope runs
    there and books nothing; any other is a job, priced, and held to the approved mandate:
    its ceiling, its system (every input descends from it, by the ledger) and the values
    the researcher decided (core/lineage.py); without a scheduler, also to the cores the
    machine has."""
    resources = params.resources()
    checked = []
    if getattr(type(params), "SESSION_OK", False):
        checked.append("placement")
        placement = place(resources, ctx.config.session,
                          after=bool(getattr(params, "after", None)))
        if placement.in_session:
            return [Finding(
                id="RUNS_IN_SESSION",
                message=f"runs in the session and books nothing: {placement.reason}",
            )], checked
    checked.append("compute_budget")
    if ctx.config.backend == "local":
        checked.append("machine_fit")
    est, findings = budget_warnings(resources, ctx.config.compute, ctx.config.backend)
    ws = ctx.workspace
    # the registry found the approved mandate this call is made under, if one holds
    held = mandates.read_mandate(ws) if CALL_MANDATE.get() else None
    if held is not None and held.digest == CALL_MANDATE.get():
        checked += ["mandate_ceiling", "mandate_system", "mandate_decisions"]
        return [*_within(ws, held, est, findings), *submission_findings(
            ws, held, tool, params.model_dump(mode="json", exclude_none=True),
            resources.model_dump(mode="json", exclude_none=True), ctx.config.backend,
            expect=lock_expectations(ctx, held))
        ], checked
    checked.append("mandate")
    return [*(w for w in findings if w.id != "COMPUTE_ESTIMATE"),
            Theirs(
                id="MANDATE_ABSENT",
                message=f"not submitted: the job needs {est:g} core-h, and "
                f"{mandates.why_not_holding(ws)}",
                suggestion="mandate_write with the question, the system and the ceiling; "
                "say its cost and digest and end the turn: the researcher approves it")
            ], checked


def lock_expectations(ctx: ToolContext, held: mandates.Mandate):
    """``expect(kind, stage)`` for the submission check: the keys a parameter file of that
    kind must hold for the researcher's locks on that stage, compiled from the mandate over
    the facts its system files hold (knowledge/stage.py). None when nothing is locked."""
    if not any(d.locks for d in held.decisions.yours):
        return None
    from myrmidon.knowledge.inputsets import ENGINE_OF_RENDERS, RENDERED_FILES
    from myrmidon.knowledge.stage import MethodStore, TemplateStore, expected_keys, methods_for

    facts = mandate_facts(ctx, held)
    ff = facts.get("forcefield")
    templates, methods = TemplateStore.default(), MethodStore.default()
    hamiltonian = methods_for(held, methods, ff.value if ff is not None else None)
    engine_of = {kind: ENGINE_OF_RENDERS.get(renders)
                 for renders, (kind, _) in RENDERED_FILES.items()}

    def expect(kind: str, stage: str | None):
        engine = engine_of.get(kind)
        if engine is None:
            return {}, []
        return expected_keys(held, stage, engine, templates=templates, methods=hamiltonian,
                             facts=facts)
    return expect


def mandate_facts(ctx: ToolContext, mandate: mandates.Mandate) -> dict:
    """The facts the mandate's system files hold: its topology's force field, restraints
    and groups, its index's QM group, its structure's starting state."""
    from myrmidon.engines.facts import data_dir_of, system_facts

    files = {}
    for f in mandate.system.files:
        path = ctx.workspace.resolve(f.path)
        if path.is_file():
            files.setdefault(f.kind, path)
    return system_facts(files, data_dir_of(ctx.engines))


def _within(ws, held: mandates.Mandate, est: float, findings: list[Finding]):
    """The price against what the mandate has left. Its ceiling is the researcher's
    approval of the spend, so the machine's approval threshold has no say under it; the
    machine's hard limit and the fit still do."""
    spent = booked_under_mandates(ws, mandates.approved_digests(ws))
    ceiling = held.cost.ceiling_core_h
    left = round(ceiling - spent, 3)
    yield from findings
    if est > left:
        yield Theirs(
            id="MANDATE_CEILING",
            message=f"not submitted: the job needs {est:g} core-h and the mandate has "
            f"{left:g} of its {ceiling:g} core-h left ({spent:g} booked)",
            suggestion="say so and end the turn: the researcher raises the ceiling "
            "(mandate_write with revise=true, which they approve) or stops here")
        return
    yield Finding(
        id="MANDATE_WITHIN",
        message=f"{est:g} core-h under mandate revision {held.revision} "
        f"({held.digest[:12]}): {left - est:g} of {ceiling:g} core-h left after it")
