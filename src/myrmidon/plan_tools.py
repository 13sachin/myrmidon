"""`plan_write`: the verb that puts a study's plan on disk.

A plan says what this study is doing, in what order, for how much (`core/plan.py`);
this tool is what produces the `plan.yaml` that `inputset_apply` reads.

**The tool does not invent the plan.** Assembling steps from a question is the agent's
work, done against `inputset_lookup`; what happens here is everything that can be checked
rather than asserted:

- every step's `tool` is a registered tool, and every `cites` names an input set that is
  actually installed, at the version it claims;
- no step needs an artifact role that nothing earlier in the plan produces — referential
  integrity, not an opinion about which pipeline is right: a study may skip any published
  step by bringing that artifact itself, and a new sequence is what experimenting looks
  like. It is also what catches a step deleted from an edited plan;
- the declared axis is a key some cited inputset actually sets, so `arms.varies` cannot
  quietly name nothing and exempt nothing;
- every step's cost is **computed** from the scheduler resources it states, by the function
  a submission is priced with, never typed;
- **approvable means executable**: every value execution will need — what a
  cited input set leaves to the researcher, a submitting step's resources, a parameter the
  tool requires that nothing states — is listed as ``unresolved``, and ``plan_approve``
  refuses while that list is not empty, because the planning conversation is when the
  researcher is there to answer;
- the total goes through the same budget thresholds a submission does, and
  **acknowledging that finding is the approval** — recorded in the ledger against
  `Plan.digest()`, so an edited plan is not covered by it.

Writing a plan runs nothing and costs nothing. It is the cheapest moment at which a study
can be wrong in a way somebody notices.
"""

from __future__ import annotations

import uuid
from typing import Annotated, NamedTuple

import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from myrmidon.core.observation import (
    JournalObservation,
    PlanCheckObservation,
    PlanObservation,
    ProtocolHit,
    ProtocolObservation,
    ReportObservation,
    ReuseCandidate,
    Unresolved,
)
from myrmidon.core.plan import Arms, Envelope, Plan, PlanStep, Until
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning

PLAN_FILE = "plan.yaml"
#: the tool the loop continues a repeating step with, by the step's own tool
CONTINUATION_TOOL = {"gmx_mdrun": "checkpoint_continue", "mimic_run": "mimic_run"}
#: the parameter a step's cited input set renders into, by what the set renders: a grompp
#: step citing an mdp input set reads the rendered file as its `mdp`, so that file is not a
#: value the researcher states, and the loop renders it before the call
RENDER_PARAM = {"mdp": "mdp", "cpmd_inp": "template"}


def extension_core_h(plan: Plan, compute) -> float:
    """What one more segment of the repeating step costs: its continuation priced from
    ``until.extend``'s resources the way the continuation's submission will be, falling
    back to the step's own resources."""
    if plan.until is None:
        return 0.0
    step = plan.step(plan.until.repeats)
    tool = CONTINUATION_TOOL.get(step.tool, step.tool)
    values = {k: v for k, v in step.values.items()}
    values.update(plan.until.extend)
    price = price_step(step.model_copy(update={"tool": tool, "values": values}), compute)
    return price.core_h or 0.0


def ceiling_core_h(plan: Plan, compute) -> float:
    """The most the plan may cost under its own ceilings — the figure an approval covers
    and the loop refuses to book past: one pass, the further segments an `until` may add
    (cost × ceiling), the further arms (cost × arms), and the segments the envelope
    lets a watched coupled stage gain."""
    arm_steps = set(plan.arm_steps())
    arm = sum(s.estimate_core_h or 0.0 for s in plan.steps if s.name in arm_steps) \
        if arm_steps else None
    extendable = sum((s.estimate_core_h or 0.0) * (len(plan.arms.over) if plan.arms and
                                                   s.name in arm_steps else 1)
                     for s in plan.steps if s.tool == "mimic_run" and s.sentinels)
    return round(plan.ceiling_core_h(extension_core_h(plan, compute), arm)
                 + plan.envelope.extend_segments * extendable, 3)


class PlanWriteParams(BaseModel):
    question: str = Field(
        description="what this study is trying to find out, in the user's own words"
    )
    steps: list[dict] = Field(
        default_factory=list,
        description="ordered steps. Each: name, tool, optional cites "
        "('inputset:<id>@v<N>' from inputset_lookup), values (the parameters the step's "
        "tool will be called with — every one of them, e.g. resources and overrides; a "
        "key outside `values` is refused), needs and produces "
        "(artifact roles — 'structure', 'topology', 'tpr_heat', 'trajectory'). A step "
        "that submits a job states its scheduler resources in `values` (e.g. ntasks, "
        "cpus_per_task, walltime_min; mimic_run: cpmd_ranks, gromacs_ranks, walltime_min): "
        "its cost is computed from them, never given. A step needing a role no earlier "
        "step produces is refused: roles are what order a plan, so there is no 'after'. "
        "`reuse`: the event id of an earlier call whose result the step uses instead of "
        "running (plan_write offers candidates); the plan then shows it as reused. A value "
        "that is a fact about an earlier step's run is written as a reference, e.g. "
        "'${annealing.final_temperature_k} 300 50': the rule is approved, and the number is "
        "parsed from that run when it has completed."
    )
    system: str = Field(default="", description="what the study is about, one line")
    from_protocol: str | None = Field(
        default=None,
        description="a published pipeline to start from, by id (see the protocol store). "
        "With no `steps`, the plan is that sequence. With `steps`, they are the plan and "
        "the protocol is recorded as where it started — adapting a published pipeline is "
        "the work, not a deviation, and costs no acknowledgement.",
    )
    given: list[str] = Field(
        default_factory=list,
        description="artifact roles supplied from outside the plan — the files the user "
        "brings. A role that is neither given nor produced is a missing step.",
    )
    arms: dict | None = Field(
        default=None,
        description="the experiment this study declares: {varies, over, metric, because}. "
        "An override of the key named by 'varies' then needs no acknowledgement, because "
        "the plan has already said that is the experiment and said why. Anything else "
        "changed is still a deviation to acknowledge.",
    )
    until: dict | None = Field(
        default=None,
        description="for a chain: {criterion, check, repeats, ceiling, values, extend}. "
        "'ceiling' is required — a chain with no ceiling is the loop that never ends; it "
        "counts segments, the first included. `check` is the tool the loop decides the "
        "criterion with (series_convergence on the repeating gmx_mdrun step's energies), "
        "`values` its parameters — the researcher's tolerance, e.g. {series: ['Density'], "
        "max_drift_per_ns: 5, max_halves_sigma: 2} — and `extend` the continuation's "
        "{extend_ps, ntasks, cpus_per_task, walltime_min}, priced into the ceiling.",
    )
    envelope: dict | None = Field(
        default=None,
        description="what the loop may do without asking, approved with the plan "
        ": {rerun_on_failure: reruns of identical inputs a failed step gets "
        "before it parks (default 1), extend_segments: segments a coupled stage may gain "
        "when its sentinel fails (default 0), drop_fraction: the share of a sweep's arms "
        "that may fail on a known signature and be dropped (default 0), on_deadline: 'park' "
        "or 'wait'}. A step may carry `sentinels` [{check, args}] and `deadline_min`.",
    )
    invariants: list[dict] = Field(
        default_factory=list,
        description="rules relating values that DIFFERENT steps render, as "
        "{id, equal|not_equal, message} over '$step.key' references. A rule inside one "
        "parameter set belongs to that inputset instead.",
    )
    notes: str = ""
    overwrite: bool = Field(
        default=False,
        description="replace an existing plan. The digest changes, so any approval "
        "recorded against the old one stops covering this study.",
    )
    revises: bool = Field(
        default=False,
        description="with overwrite: the new plan answers the same question as the one it "
        "replaces, reworded — the calls already made for it still count. Not needed when "
        "the question is unchanged. A different question starts a new plan, and nothing "
        "done for the old one carries out its steps.",
    )
    acknowledge_warnings: list[str] = Field(default_factory=list)


def _protocol(ctx: ToolContext, protocol_id: str):
    store = getattr(ctx, "protocols", None)
    if store is None:
        # Without this the caller would get PLAN_MALFORMED: the protocol silently does
        # not load, its steps never arrive, and an empty plan is blamed on the agent. A
        # harness that is missing a store says so itself.
        return None, ValidationWarning(
            id="PLAN_PROTOCOL_STORE_MISSING",
            severity="block",
            message="this session has no protocol store, so from_protocol cannot be "
            "resolved — the harness is misconfigured, your call is not wrong",
            suggestion="give the steps directly, or report this: every ToolContext should "
            "carry a ProtocolStore",
        )
    try:
        return store.get(protocol_id), None
    except (KeyError, AttributeError):
        known = ", ".join(sorted(w.id for w in store.all())) or "none"
        return None, ValidationWarning(
            id="PLAN_PROTOCOL_UNKNOWN",
            severity="block",
            message=f"no protocol {protocol_id!r} in any layer; known: {known}",
            suggestion="omit from_protocol and give the steps yourself — a new sequence "
            "is not an error",
        )


def _build(
    params: PlanWriteParams, protocol=None
) -> tuple[Plan | None, ValidationWarning | None]:
    """The plan, or the finding that says why it is not one.

    With a protocol and no steps of its own, the plan is that published sequence. With
    steps, they are the plan and the protocol is only what it started from: a researcher
    adapting a pipeline is doing the work, so nothing here makes them justify it.
    """
    steps = params.steps
    given = list(params.given)
    invariants = list(params.invariants)
    if protocol is not None and not steps:
        steps = [s.model_dump(exclude_none=True) for s in protocol.steps]
        given = given or list(protocol.given)
        invariants = invariants or [i.model_dump(exclude_none=True) for i in protocol.invariants]
    # a key the step does not have would be dropped without a word, and the plan the
    # researcher approves would not be the one the agent described
    known = set(PlanStep.model_fields)
    for s in steps:
        stray = sorted(set(s) - known) if isinstance(s, dict) else []
        if stray:
            return None, ValidationWarning(
                id="PLAN_STEP_UNKNOWN_FIELD",
                severity="block",
                message=f"step {s.get('name', '?')!r} has field(s) a step does not have: "
                f"{', '.join(stray)}",
                suggestion="a step's parameters for its tool go inside its `values` object, "
                "e.g. {\"name\": \"run\", \"tool\": \"gmx_mdrun\", \"values\": "
                "{\"walltime_min\": 20}}",
            )
    try:
        return Plan(
            question=params.question,
            system=params.system,
            cites=(
                f"protocol:{protocol.id}@v{protocol.version}" if protocol is not None else None
            ),
            steps=[PlanStep(**s) for s in steps],
            given=given,
            arms=Arms(**params.arms) if params.arms else None,
            until=Until(**params.until) if params.until else None,
            invariants=invariants,
            envelope=Envelope(**params.envelope) if params.envelope else Envelope(),
            notes=params.notes,
        ), None
    except (ValueError, TypeError) as exc:
        return None, ValidationWarning(
            id="PLAN_MALFORMED",
            severity="block",
            message=f"this is not a plan: {exc}",
            suggestion="give at least one step with a name and a tool, or name a protocol "
            "to start from; see the description of `steps`",
        )


def _cited_protocol(ctx: ToolContext, cites: str):
    """(inputset, finding) for one step's `cites` string."""
    ref = cites.removeprefix("inputset:")
    pid, _, version = ref.partition("@v")
    store = getattr(ctx, "inputsets", None)
    if store is None:
        return None, None
    try:
        inputset = store.get(pid)
    except (KeyError, AttributeError):
        known = ", ".join(sorted(p.id for p in store.all())) or "none"
        return None, ValidationWarning(
            id="PLAN_CITES_UNKNOWN_PROTOCOL",
            severity="block",
            message=f"no inputset {pid!r} in any layer; known: {known}",
            suggestion="find one with inputset_lookup",
        )
    if version and str(inputset.version) != version:
        return inputset, ValidationWarning(
            id="PLAN_CITES_OTHER_VERSION",
            severity="warn",
            message=f"the plan cites {pid} at v{version}; this machine has "
            f"v{inputset.version}",
            assumption="the values would be this machine's, not the ones the plan was "
            "written against",
            suggestion="re-read it with inputset_lookup and cite the version you got",
        )
    return inputset, None


class StepPrice(NamedTuple):
    core_h: float | None  # None: the step does not state the resources its cost needs
    basis: str
    missing: list[str]  # resource fields left unstated, or stated as no valid value


def price_step(step: PlanStep, compute) -> StepPrice:
    """What a step costs on this machine, computed the way its submission will be.

    A tool whose params have ``resources()`` submits a job, and its cost is the booking of
    the resources the step states, priced by ``execution/accounting.py`` — the function
    the submission gate prices the job with, so the plan a person approves and the job
    that runs carry one number. Every field the params list in ``RESOURCE_FIELDS`` must be
    stated: a tool default nobody chose is an assumption, and an unstated cost is exactly
    what the approval exists to show. A tool that submits nothing runs in the session.
    """
    from myrmidon.core.tool import REGISTRY, UnknownToolError
    from myrmidon.execution.accounting import describe_booking, estimate_core_h

    try:
        model = REGISTRY.get(step.tool).params_model
    except UnknownToolError:  # PLAN_STEP_TOOL_UNKNOWN says so
        return StepPrice(None, "", [])
    if not callable(getattr(model, "resources", None)):
        return StepPrice(0.0, "runs in the session, not on the scheduler", [])
    stated, missing = {}, []
    # every value the step states for its tool, so a `partition` is priced on its own
    # nodes; the resource fields must be among them (the registry refuses a tool without)
    for name in {*step.values, *model.RESOURCE_FIELDS} & set(model.model_fields):
        info = model.model_fields[name]
        kind = Annotated[(info.annotation, *info.metadata)] if info.metadata else info.annotation
        try:
            stated[name] = TypeAdapter(kind).validate_python(step.values[name])
        except (KeyError, ValidationError):
            if name in model.RESOURCE_FIELDS:
                missing.append(name)
    if missing:
        return StepPrice(None, "", sorted(missing, key=model.RESOURCE_FIELDS.index))
    resources = model.model_construct(**stated).resources()
    return StepPrice(estimate_core_h(resources, compute),
                     describe_booking(resources, compute), [])


def _priced(plan: Plan, compute) -> tuple[Plan, list[str]]:
    """(the plan with every step's cost computed, steps whose given figure was replaced)."""
    steps, replaced = [], []
    for step in plan.steps:
        if step.reuse:  # runs nothing: what it books was booked by the call it names
            steps.append(step.model_copy(update={
                "estimate_core_h": 0.0, "estimate_basis": f"reuses {step.reuse}"}))
            continue
        price = price_step(step, compute)
        given = step.estimate_core_h
        if not price.missing and given is not None and given != price.core_h:
            replaced.append(f"{step.name} was given {given}, costs {price.core_h}")
        steps.append(step.model_copy(update={
            "estimate_core_h": price.core_h, "estimate_basis": price.basis or None}))
    return plan.model_copy(update={"steps": steps}), replaced


def unresolved(ctx: ToolContext, plan: Plan) -> list[Unresolved]:
    """Every value the plan's execution will need and the plan does not carry.

    Three kinds, per step, in the order a researcher would be asked:

    - **researcher** — a key the cited input set marks ``required_from_user``, or a field
      the step's tool names in ``RESEARCHER_FIELDS``, that the step's ``values`` do not
      state: which atoms are quantum, the charge, how long. The tool's own declaration
      holds whatever input set the step cites, so a step that cites the file it renders
      still asks for the chemistry it needs;
    - **resources** — the fields a submitting step's cost is computed from (S1);
    - **parameter** — a field the step's tool requires that neither the step's ``values``
      nor a value in its cited input set states. A file parameter is stated by naming the
      artifact role that feeds it (``tpr: tpr_mimic``) or a path: which file goes where is
      part of what makes a plan runnable.

    A figure a step was given for its cost is quoted beside missing resources: a protocol's
    measured cost says what a run like it booked.
    """
    from myrmidon.core.plan import references_in
    from myrmidon.core.tool import REGISTRY, UnknownToolError

    made_by: dict[str, str] = {}  # role -> the step that makes it
    out: list[Unresolved] = []
    for step in plan.steps:
        earlier = dict(made_by)
        made_by.update({role: step.name for role in step.produces})
        if step.reuse:  # carried out already: it needs nothing more
            continue
        inputset = _cited_protocol(ctx, step.cites)[0] if step.cites else None
        asked: set[str] = set()
        for key, spec in (inputset.unsupplied().items() if inputset else ()):
            if key not in step.values:
                out.append(Unresolved(step=step.name, key=key, kind="researcher",
                                      detail=spec.derivable or f"{inputset.id} leaves it "
                                      "to the researcher"))
                asked.add(key)
        try:
            model = REGISTRY.get(step.tool).params_model
        except UnknownToolError:  # PLAN_STEP_TOOL_UNKNOWN says so
            continue
        for key in getattr(model, "RESEARCHER_FIELDS", ()):
            if key not in step.values and key not in asked:
                out.append(Unresolved(step=step.name, key=key, kind="researcher",
                                      detail=model.model_fields[key].description or ""))
                asked.add(key)
        if inputset is not None and "inputset" in model.model_fields:
            asked.add("inputset")  # inputset_apply: the step's `cites` names it
        price = price_step(step, ctx.config.compute)
        if price.missing:
            hint = ""
            if step.estimate_core_h:
                hint = f"; it was given {step.estimate_core_h} core-h" + (
                    f" ({step.estimate_basis})" if step.estimate_basis else "")
            out.append(Unresolved(step=step.name, key=", ".join(price.missing),
                                  kind="resources",
                                  detail="the scheduler resources its cost is computed "
                                  "from" + hint))
        for name, info in model.model_fields.items():
            if not info.is_required() or name in asked or name in step.values:
                continue
            stated = inputset.parameters.get(name) if inputset else None
            if stated is not None and stated.value is not None:
                continue
            if inputset is not None and RENDER_PARAM.get(inputset.renders or "") == name:
                continue  # the file the cited input set renders, fed to this parameter
            if info.annotation is str and step.needs:
                out.append(Unresolved(step=step.name, key=name, kind="file",
                                      detail="one of " + ", ".join(step.needs)))
                continue
            detail = (stated.derivable if stated is not None and stated.derivable
                      else info.description or "")
            out.append(Unresolved(step=step.name, key=name, kind="parameter",
                                  detail=detail))
        # a file an earlier step makes reaches this one only through a parameter: a role
        # fed to none would leave the tool on its own default (a stage's input made with no
        # method silently keeps the annealing method)
        named = {v for v in _strings(step.values) if not references_in(v)}
        open_fields = [n for n, f in model.model_fields.items()
                       if n not in step.values and f.annotation in (str, str | None)]
        for role in step.needs:
            if role in earlier and role not in plan.given and role not in named:
                out.append(Unresolved(
                    step=step.name, key=role, kind="wiring",
                    detail=f"made by {earlier[role]}; name the parameter it feeds"
                    + (", one of " + ", ".join(open_fields[:6]) if open_fields else "")))
    return out


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _jobs_by_event(ctx: ToolContext) -> dict[str, str]:
    """event id -> the job that call submitted."""
    from myrmidon.execution.jobstore import JobStore

    db = ctx.workspace.state_dir / "jobs.sqlite"
    if not db.exists():
        return {}
    return {rec.meta["event_id"]: rec.job_id for rec in JobStore(db).list()
            if rec.meta.get("event_id")}


def reuse_candidates(ctx: ToolContext, plan: Plan) -> list[ReuseCandidate]:
    """For each step that neither runs under this plan yet nor declares reuse, the latest
    earlier call made outside the plan that fits it. Offered so the researcher can
    choose to use it; never declared on their behalf."""
    from myrmidon.core.plan_binding import step_fit

    events = [ev for ev in ctx.workspace.ledger.events()
              if ev.kind == "call" and ev.ok and ev.plan_id != plan.plan_id]
    if not events:
        return []
    fits = step_fit(ctx.workspace, plan)
    jobs = _jobs_by_event(ctx)
    out = []
    for step in plan.steps:
        if step.reuse:
            continue
        match = next((ev for ev in reversed(events)
                      if ev.tool == step.tool and fits(step, ev)), None)
        if match is not None:
            out.append(ReuseCandidate(step=step.name, event=match.event_id, tool=match.tool,
                                      date=match.timestamp.date().isoformat(),
                                      job=jobs.get(match.event_id)))
    return out


def describe_reuse(ctx: ToolContext, event_id: str) -> str:
    """"gmx_mdrun on <date>, event ev-…, job job-…" — the run a reused step names."""
    event = ctx.workspace.ledger.get(event_id)
    if event is None:
        return f"event {event_id}, which this study's ledger does not hold"
    job = _jobs_by_event(ctx).get(event_id)
    return (f"{event.tool} on {event.timestamp.date().isoformat()}, event {event_id}"
            + (f", job {job}" if job else ""))


def _unresolved_lines(items: list[Unresolved]) -> list[str]:
    """What is still open, said once per decision: the researcher's first, a value asked
    of several steps once with the steps named; the files each step reads last, one line
    per step, since those are the plan author's wiring and not a question for anyone."""
    lines: list[str] = []
    asked: dict[tuple[str, str], list[str]] = {}
    for u in items:
        if u.kind == "researcher":
            asked.setdefault((u.key, u.detail), []).append(u.step)
    if asked:
        lines.append("from the researcher:")
        lines += [f"  {key} ({', '.join(steps)}): {detail}"
                  for (key, detail), steps in asked.items()]
    booked = [u for u in items if u.kind == "resources"]
    if booked:
        lines.append("scheduler resources, the researcher's to agree:")
        lines += [f"  {u.step}: {u.key}" + (f" — {u.detail.split('; ', 1)[1]}"
                                            if "; " in u.detail else "") for u in booked]
    values = [u for u in items if u.kind == "parameter"]
    if values:
        lines.append("values no input set states:")
        lines += [f"  {u.step}.{u.key}: {u.detail}" for u in values]
    files: dict[str, list[str]] = {}
    for u in items:
        if u.kind == "file":
            files.setdefault(u.step, []).append(u.key)
    if files:
        lines.append("files each step reads, named by the role that feeds it:")
        lines += [f"  {step}: {', '.join(keys)}" for step, keys in files.items()]
    wiring = [u for u in items if u.kind == "wiring"]
    if wiring:
        lines.append("files an earlier step makes, fed to no parameter of the step that "
                     "needs them:")
        lines += [f"  {u.step} ← {u.key}: {u.detail}" for u in wiring]
    return lines


def _floor_note(plan: Plan) -> str:
    """How much of the total is real, said the same way everywhere.

    A total that silently omits unestimated steps is worse than no total: it is a number
    the user would approve as if it were the cost.
    """
    if not plan.unestimated:
        return ""
    if len(plan.unestimated) == len(plan.steps):
        return " (a floor of zero: no step states its resources)"
    return " (a floor: " + ", ".join(plan.unestimated) + " state no resources)"


def plan_write_gate(ctx: ToolContext, params: PlanWriteParams):
    warnings: list[ValidationWarning] = []
    checked = ["plan_shape"]

    protocol = None
    if params.from_protocol:
        protocol, unknown = _protocol(ctx, params.from_protocol)
        if unknown is not None:
            return [unknown], checked

    plan, malformed = _build(params, protocol)
    if malformed is not None:
        return [malformed], checked
    still_open = unresolved(ctx, plan)
    plan, replaced = _priced(plan, ctx.config.compute)

    if protocol is not None:
        checked.append("protocol_followed")
        theirs = {s.name for s in plan.steps}
        published = {s.name for s in protocol.steps}
        added, dropped = sorted(theirs - published), sorted(published - theirs)
        if added or dropped:
            # adapting a published pipeline is the work, so this is info and never a gate
            parts = []
            if added:
                parts.append("added " + ", ".join(added))
            if dropped:
                parts.append("left out " + ", ".join(dropped))
            warnings.append(
                ValidationWarning(
                    id="PLAN_ADAPTS_PROTOCOL",
                    severity="info",
                    message=f"this study starts from {protocol.id} and {'; '.join(parts)}",
                )
            )
        else:
            warnings.append(
                ValidationWarning(
                    id="PLAN_FOLLOWS_PROTOCOL",
                    severity="info",
                    message=f"this study follows {protocol.id} v{protocol.version} as "
                    "published",
                )
            )

    root = getattr(ctx.workspace, "root", None)
    if root is not None and (root / PLAN_FILE).exists() and not params.overwrite:
        warnings.append(
            ValidationWarning(
                id="PLAN_ALREADY_WRITTEN",
                severity="block",
                message=f"this study already has a {PLAN_FILE}",
                suggestion="read it first; overwrite=true replaces it, and the new digest "
                "means any approval recorded against the previous plan does not cover it",
            )
        )

    checked.append("tools_exist")
    from myrmidon.core.tool import REGISTRY

    known_tools = {spec.name for spec in REGISTRY.specs()}
    unknown = sorted({s.tool for s in plan.steps} - known_tools)
    if unknown:
        warnings.append(
            ValidationWarning(
                id="PLAN_STEP_TOOL_UNKNOWN",
                severity="block",
                message=f"no such tool(s): {', '.join(unknown)}",
                suggestion="tool_lookup lists what this session can call",
            )
        )

    checked.append("protocols_exist")
    cited = {}
    for step in plan.steps:
        if not step.cites:
            continue
        inputset, finding = _cited_protocol(ctx, step.cites)
        if inputset is not None:
            cited[step.name] = inputset
        if finding is not None:
            warnings.append(finding)

    checked.append("references_resolvable")
    from myrmidon.core.plan import references_in
    from myrmidon.core.report import FACT_TOOLS, RUN_FACTS

    order = {s.name: i for i, s in enumerate(plan.steps)}
    for index, step in enumerate(plan.steps):
        for ref_step, fact in references_in(step.values):
            target = plan.step(ref_step) if ref_step in order else None
            why = ("no step of this plan has that name" if target is None
                   else "it does not come before this step" if order[ref_step] >= index
                   else f"a {target.tool} step's run is not one the harness parses (only "
                        + ", ".join(FACT_TOOLS) + ")" if target.tool not in FACT_TOOLS
                   else f"its run states no {fact!r}" if fact not in RUN_FACTS else "")
            if why:
                warnings.append(
                    ValidationWarning(
                        id="PLAN_REFERENCE_INVALID",
                        severity="block",
                        message=f"step {step.name!r} names ${{{ref_step}.{fact}}}, but {why}",
                        suggestion="a reference names an earlier step's run and one of: "
                        + ", ".join(RUN_FACTS),
                    )
                )

    checked.append("reuse_declared")
    ledger = getattr(ctx.workspace, "ledger", None)
    for step in plan.steps:
        if not step.reuse or ledger is None:
            continue
        event = ledger.get(step.reuse)
        if event is None or not event.ok or event.kind != "call" or event.tool != step.tool:
            what = ("no such event in this study" if event is None else
                    f"a {event.tool} call" + ("" if event.ok else " that failed"))
            warnings.append(
                ValidationWarning(
                    id="PLAN_REUSE_INVALID",
                    severity="block",
                    message=f"step {step.name!r} reuses {step.reuse}, which is {what}; a "
                    f"reused step names a successful {step.tool} call whose result it uses",
                    suggestion="take one from plan_write's reuse_candidates, or drop "
                    "`reuse` and run the step",
                )
            )

    checked.append("inputs_satisfiable")
    for name, roles in plan.unsatisfied().items():
        warnings.append(
            ValidationWarning(
                id="PLAN_STEP_INPUT_UNSATISFIED",
                severity="block",
                message=f"step {name!r} needs {', '.join(roles)}, which no earlier step "
                "produces and the plan does not list under `given`",
                assumption="the step would run on a file that does not exist yet",
                suggestion="add the step that produces it, or list the role in `given` if "
                "the user brings that file",
            )
        )

    checked.append("declared_axis")
    if plan.arms is not None:
        mentions = any(
            plan.arms.varies in inputset.parameters for inputset in cited.values()
        ) or any(plan.arms.varies in step.values for step in plan.steps)
        if cited and not mentions:
            warnings.append(
                ValidationWarning(
                    id="PLAN_AXIS_NAMES_NOTHING",
                    severity="warn",
                    message=f"no cited inputset sets {plan.arms.varies!r}, so declaring it "
                    "as the axis exempts nothing",
                    assumption="the study would claim to vary a parameter none of its "
                    "steps writes",
                    suggestion="name the key as the inputset spells it, or drop `arms`",
                )
            )

    checked.append("resources_stated")
    if replaced:
        warnings.append(
            ValidationWarning(
                id="PLAN_ESTIMATE_COMPUTED",
                severity="info",
                message="a step's cost is computed from the resources it states, never "
                "given: " + "; ".join(replaced),
            )
        )

    checked.append("executable")
    if still_open:
        # writing is free, so this is info; approving is where it blocks
        warnings.append(
            ValidationWarning(
                id="PLAN_STEP_UNRESOLVED",
                severity="info",
                message=f"{len(still_open)} value(s) execution will need are not in the "
                "plan, so it cannot be approved yet:\n" + "\n".join(_unresolved_lines(still_open)),
                assumption="a plan approved without them stops after the approval to ask",
                suggestion="ask the researcher now, while they are here, then write the plan "
                "again with each value in its step's `values` (overwrite=true)",
            )
        )

    checked.append("sentinels_known")
    from myrmidon.core.sentinels import CHECKS

    unknown_checks = sorted({s.check for step in plan.steps for s in step.sentinels
                             if s.check not in CHECKS})
    if unknown_checks:
        warnings.append(
            ValidationWarning(
                id="PLAN_SENTINEL_UNKNOWN",
                severity="block",
                message=f"no such sentinel check(s): {', '.join(unknown_checks)}; the loop "
                "evaluates " + ", ".join(sorted(CHECKS)),
            )
        )
    if plan.until is not None:
        repeats = plan.step(plan.until.repeats)
        if repeats.tool not in CONTINUATION_TOOL:
            warnings.append(
                ValidationWarning(
                    id="PLAN_UNTIL_NOT_CONTINUABLE",
                    severity="block",
                    message=f"until repeats {repeats.name!r}, a {repeats.tool} step; the loop "
                    "continues " + ", ".join(sorted(CONTINUATION_TOOL)) + " steps only",
                )
            )
        elif repeats.tool == "gmx_mdrun" and not plan.until.extend.get("extend_ps"):
            warnings.append(
                ValidationWarning(
                    id="PLAN_UNTIL_EXTENSION_UNSTATED",
                    severity="block",
                    message="until.extend states no extend_ps: how long each further "
                    "segment runs is part of what is approved",
                    suggestion="until.extend = {extend_ps: <ps>, walltime_min: <min>, ...}",
                )
            )

    checked.append("compute_estimate")
    compute = ctx.config.compute
    total = plan.total_core_h
    ceiling = ceiling_core_h(plan, compute)
    floor = _floor_note(plan)

    # Writing a plan runs nothing, so writing one is never gated. Proposing and approving
    # are different acts by different parties; fused, an unattended session could not
    # even propose the plan a human would approve.
    over = ceiling > compute.warn_core_h
    warnings.append(
        ValidationWarning(
            id="COMPUTE_ESTIMATE",
            severity="info",
            message=f"the plan totals {total} core-h{floor}"
            + (f", at most {ceiling} core-h at its ceiling" if ceiling != total else "")
            + (
                f", above the {compute.warn_core_h} core-h approval threshold — nothing "
                "runs until the user approves it with plan_approve"
                if over
                else ""
            ),
        )
    )

    # the per-submission limit is per submission: a step that alone exceeds it can never
    # be run on this machine, which is worth saying before the study is built around it
    if compute.block_core_h is not None:
        too_big = [
            s.name for s in plan.steps
            if (s.estimate_core_h or 0.0) > compute.block_core_h
        ]
        if too_big:
            warnings.append(
                ValidationWarning(
                    id="PLAN_STEP_EXCEEDS_MACHINE_LIMIT",
                    severity="block",
                    message=f"step(s) {', '.join(too_big)} each cost more than this "
                    f"machine's per-submission limit of {compute.block_core_h} core-h and "
                    "could never be submitted here",
                    suggestion="split the step, or ask the user to raise the limit",
                )
            )
    return warnings, checked


class ProtocolLookupParams(BaseModel):
    query: str = Field(
        default="",
        description="words to match against ids, titles, tools and applicability. Empty "
        "lists every published pipeline this machine can see.",
    )


@tool(
    "protocol_lookup",
    navigation=True,
    description="Published pipelines to start a study from: the sequence, the inputset "
    "each step's values come from, what you must bring, and what it cost where it was "
    "measured. A STARTING POINT, never a rule — adapt one with `plan_write "
    "from_protocol=`, or give your own steps and cite none.",
    returns=ProtocolObservation,
    tags={"knowledge", "plan"},
    needs_workspace=False,
)
def protocol_lookup(ctx: ToolContext, params: ProtocolLookupParams) -> ProtocolObservation:
    from myrmidon.knowledge.protocols import ProtocolStore

    store = getattr(ctx, "protocols", None) or ProtocolStore.default()
    found = store.search(params.query) if params.query.strip() else sorted(
        store.all(), key=lambda w: w.id
    )
    hits = [
        ProtocolHit(
            id=w.id,
            title=w.title,
            applicability=w.applicability,
            system_class=w.system_class,
            scope=w.scope,
            layer=w.layer,
            version=w.version,
            given=w.given,
            steps=[s.model_dump(mode="json", exclude_none=True) for s in w.steps],
            total_core_h=round(sum(s.estimate_core_h or 0.0 for s in w.steps), 3),
            citations=[s.text for s in w.sources],
            pitfalls=w.pitfalls,
            notes=w.notes,
        )
        for w in found
    ]
    what = f"matching {params.query!r}" if params.query.strip() else "the whole store"
    first = hits[0].id if hits else "<id>"
    return ProtocolObservation(
        ok=True,
        # naming the next call, in the imperative, at the point of need: the tool that
        # finds the input says which tool makes the thing, or a session reads every input
        # set and then writes prose
        summary=f"{len(hits)} published pipeline(s), {what}. To turn one into this "
        f"study's plan: plan_write from_protocol={first}. Each is a starting point — "
        "adapt it, or give plan_write your own steps and cite none.",
        protocols=hits,
    )


@tool(
    "plan_write",
    description="Write this study's plan: ordered steps, the inputset each cites, the "
    "artifact roles each needs and produces, and what it costs. Assemble the steps "
    "yourself (or pass from_protocol); this checks them and refuses a step whose inputs "
    "nothing produces. Writing runs nothing and is never gated on cost — approving is "
    "`plan_approve`.",
    returns=PlanObservation,
    tags={"knowledge", "plan"},
    gate=plan_write_gate,
)
def plan_write(ctx: ToolContext, params: PlanWriteParams) -> PlanObservation:
    protocol = None
    if params.from_protocol:
        protocol, _ = _protocol(ctx, params.from_protocol)
    plan, malformed = _build(params, protocol)
    if malformed is not None:  # the gate already refused; belt and braces
        return PlanObservation(ok=False, summary=malformed.message)
    still_open = unresolved(ctx, plan)
    plan, _ = _priced(plan, ctx.config.compute)
    # one question, one plan: a revision of the same question keeps its id, and with
    # it the calls already made for it; a new question starts with none behind it
    previous = read_plan(ctx)
    keeps = previous is not None and bool(previous.plan_id) and (
        params.revises or previous.question.strip() == plan.question.strip())
    plan = plan.model_copy(update={
        "plan_id": previous.plan_id if keeps else "pl-" + uuid.uuid4().hex[:8]})
    candidates = reuse_candidates(ctx, plan)
    scope = ""
    if previous is not None:
        scope = (f". Revises {plan.plan_id}: the calls made for it still count" if keeps
                 else f". A new question: what was done for "
                 f"{previous.plan_id or 'the earlier plan'} carries out none of these steps")

    path = ctx.workspace.root / PLAN_FILE
    body = plan.model_dump(mode="json", exclude_none=True, exclude_defaults=False)
    path.write_text(
        "# this study's plan, written by myrmidon plan_write\n"
        "# steps are ordered by what they need and produce, not by an 'after' field\n"
        + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=96)
    )

    from myrmidon.core.artifacts import ArtifactRef

    digest = plan.digest()
    axis = f"; varying {plan.arms.varies} over {plan.arms.over}" if plan.arms else ""
    origin = f" from {plan.cites}" if plan.cites else ""
    return PlanObservation(
        ok=True,
        summary=f"{plan.shape} plan {plan.plan_id} of {len(plan.steps)} step(s){origin}, "
        f"{plan.total_core_h} core-h{_floor_note(plan)}{axis}. digest {digest[:12]}{scope}"
        + (f". Not yet approvable: {len(still_open)} value(s) to settle with the "
           "researcher first — see `unresolved`" if still_open else "")
        + (f". {len(candidates)} step(s) could reuse an earlier call instead of running — "
           "see `reuse_candidates`; reuse is declared in the step, never assumed"
           if candidates else ""),
        path=ctx.workspace.relpath(path),
        artifacts=[ArtifactRef.from_path(path, "plan", ctx.workspace.root)],
        digest=digest,
        plan_id=plan.plan_id,
        cites=plan.cites,
        shape=plan.shape,
        question=plan.question,
        total_core_h=plan.total_core_h,
        unestimated=plan.unestimated,
        unresolved=still_open,
        reuse_candidates=candidates,
        steps=[s.model_dump(mode="json", exclude_none=True) for s in plan.steps],
        given=plan.given,
        arms=plan.arms.model_dump(mode="json") if plan.arms else None,
        until=plan.until.model_dump(mode="json") if plan.until else None,
        envelope=plan.envelope.model_dump(mode="json"),
        ceiling_core_h=ceiling_core_h(plan, ctx.config.compute),
    )


def read_plan(ctx: ToolContext) -> Plan | None:
    """This study's plan, if it has one. The one reader every caller shares."""
    root = getattr(ctx.workspace, "root", None)
    if root is None:
        return None
    path = root / PLAN_FILE
    if not path.exists():
        return None
    try:
        return Plan(**yaml.safe_load(path.read_text()))
    except (OSError, ValueError, TypeError):
        return None


def render_plan(plan: Plan, digest: str | None = None,
                still_open: list[Unresolved] | None = None, ctx: ToolContext | None = None
                ) -> str:
    """The plan as a person reads it — and as a client's plan mode shows it.

    One object, several renderings: this is the first. The harness never implements the
    approval itself; it renders what a client's own plan mode already knows how to show.
    """
    lines = [plan.question, ""]
    if plan.system:
        lines += [f"system: {plan.system}", ""]
    if plan.arms:
        lines += [
            f"varying {plan.arms.varies} over "
            + ", ".join(str(v) for v in plan.arms.over)
            + (f" — {plan.arms.metric}" if plan.arms.metric else ""),
            f"  because {plan.arms.because}",
            "",
        ]
    if plan.until:
        values = ", ".join(f"{k}={v}" for k, v in plan.until.values.items())
        extend = ", ".join(f"{k}={v}" for k, v in plan.until.extend.items())
        lines += [
            f"repeating {plan.until.repeats} until {plan.until.criterion} "
            f"(at most {plan.until.ceiling} segments, decided by {plan.until.check}"
            + (f" with {values}" if values else "") + ")"
            + (f"; each further segment: {extend}" if extend else ""),
            "",
        ]
    env = plan.envelope
    lines += [
        "the loop may, without asking: "
        f"rerun a failed step {env.rerun_on_failure}× with identical inputs; "
        f"extend a stage whose sentinel fails by {env.extend_segments} segment(s); "
        f"drop {env.drop_fraction:.0%} of a sweep's arms on a known signature; "
        f"{'keep waiting' if env.on_deadline == 'wait' else 'park'} past a deadline",
        "",
    ]
    if plan.given:
        lines += ["you bring: " + ", ".join(plan.given), ""]

    width = max((len(s.name) for s in plan.steps), default=4)
    lines.append(f"{'step'.ljust(width)}  {'tool':<18} {'core-h':>7}  values from")
    for step in plan.steps:
        cost = f"{step.estimate_core_h:.1f}" if step.estimate_core_h is not None else "—"
        unpriced = step.estimate_core_h is None
        source = step.cites or "the tool's own"
        lines.append(f"{step.name.ljust(width)}  {step.tool:<18} {cost:>7}  {source}")
        if step.produces:
            lines.append(f"{' ' * width}  -> " + ", ".join(step.produces))
        if step.reuse:
            lines.append(f"{' ' * width}     REUSED, not run: "
                         + (describe_reuse(ctx, step.reuse) if ctx is not None
                            else f"event {step.reuse}"))
        # a cost with no visible basis is the kind of number this harness exists to refuse
        if step.estimate_basis:
            lines.append(f"{' ' * width}     {step.estimate_core_h} core-h: "
                         f"{step.estimate_basis}")
        elif unpriced:
            lines.append(f"{' ' * width}     no cost: its scheduler resources are not stated")
        for sentinel in step.sentinels:
            args = ", ".join(f"{k}={v}" for k, v in sentinel.args.items())
            lines.append(f"{' ' * width}     sentinel {sentinel.check}({args})")
        if step.deadline_min:
            lines.append(f"{' ' * width}     deadline {step.deadline_min} min")
        if step.note:
            lines.append(f"{' ' * width}     {step.note}")

    total = plan.total_core_h
    lines += ["", f"{'total'.ljust(width)}  {'':<18} {total:>7.1f}{_floor_note(plan)}"]
    if ctx is not None:
        ceiling = ceiling_core_h(plan, ctx.config.compute)
        if ceiling != total:
            lines.append(f"{'at most'.ljust(width)}  {'':<18} {ceiling:>7.1f}  at the plan's "
                         "ceiling (further segments, further arms)")
    missing = plan.unsatisfied()
    if missing:
        lines.append("")
        for name, roles in missing.items():
            lines.append(f"UNSATISFIED  {name} needs {', '.join(roles)}")
    if still_open:
        lines += ["", "still needed before this plan can be approved:"]
        lines += [f"  {line}" for line in _unresolved_lines(still_open)]
    if digest:
        lines += ["", f"digest {digest[:12]} — an approval is recorded against this"]
    return "\n".join(lines)


# --- plan against ledger: did we do what we said ------------------------------------


def _call_events(ctx: ToolContext, plan: Plan | None = None) -> list:
    """The study's successful actions, oldest first — with a plan, only the ones made under
    it: a study holds many questions, and a call made for one carries out nothing of
    another's. Looks are not actions."""
    scope = plan.plan_id if plan is not None else ""
    return [
        ev
        for ev in ctx.workspace.ledger.events()
        if ev.kind == "call" and ev.ok and ev.tool not in ("plan_write",)
        and (not scope or ev.plan_id == scope)
    ]


def _match_steps(plan: Plan, events: list, carried_out=None
                 ) -> tuple[list[dict], list, bool]:
    """(per-step state, events no step claimed, whether the match was exact).

    A call an approved plan bound names its step (``plan_step``) and carries out that step
    and no other. The rest is a sequence alignment, and stated as one: walk the steps in
    order, and for each take the next call of the same tool that ``carried_out(step,
    event)`` accepts (any, without one; ``plan_binding.step_fit`` is the one binding and
    ``plan_check`` use). It cannot distinguish two untagged steps that call the same tool
    with nothing between them, and it never guesses in the direction of "done". The match
    is exact when every step found done was found by its name.
    """
    states: list[dict] = []
    claimed: set[str] = set()
    by_name = 0
    cursor = 0
    for step in plan.steps:
        if step.reuse:  # carried out by the earlier call it names, not by one of this plan's
            states.append({"name": step.name, "tool": step.tool, "state": "reused",
                           "events": [step.reuse], "cites": step.cites})
            continue
        hits = []
        # a call that names its step carries it out wherever it sits in the ledger: the
        # loop makes calls in data-flow order, not in the plan's written order
        named_hit = next((ev for ev in events if getattr(ev, "plan_step", "") == step.name
                          and ev.tool == step.tool and ev.event_id not in claimed), None)
        if named_hit is not None:
            hits = [named_hit]
            claimed.add(named_hit.event_id)
            by_name += 1
        for index in range(cursor if not hits else len(events), len(events)):
            event = events[index]
            named = getattr(event, "plan_step", "")
            if event.tool != step.tool or named or event.event_id in claimed:
                continue
            if carried_out is None or carried_out(step, event):
                hits = [event]
                claimed.add(event.event_id)
                cursor = index + 1
                break
        states.append(
            {
                "name": step.name,
                "tool": step.tool,
                "state": "done" if hits else "pending",
                "events": [ev.event_id for ev in hits],
                "cites": step.cites,
            }
        )
    unclaimed = [ev for ev in events if ev.event_id not in claimed]
    return states, unclaimed, bool(claimed) and by_name == len(claimed)


def _booked_for(ctx: ToolContext, plan_id: str) -> float:
    """Core-hours booked by the jobs this plan's calls submitted — never another
    question's."""
    from myrmidon.execution.accounting import spec_core_h
    from myrmidon.execution.jobstore import JobStore

    db = ctx.workspace.state_dir / "jobs.sqlite"
    if not db.exists():
        return 0.0
    total = 0.0
    for rec in JobStore(db).list():
        if rec.meta.get("plan_id") != plan_id:
            continue
        est = rec.meta.get("core_h_estimate")
        if est is None:
            est = spec_core_h(rec.spec, ctx.config.compute)
        total += est or 0.0
    return round(total, 3)


def _protocol_ids(value: str | None) -> str | None:
    if not value:
        return None
    return value.split("@")[0].removeprefix("inputset:")


class PlanCheckParams(BaseModel):
    pass


@tool(
    "plan_check",
    description="Did this study do what its plan said? Per step: run or not. Plus "
    "inputsets cited but never applied and applied but never cited, planned against "
    "booked against consumed core-hours, and whether the plan is still the approved one. "
    "A look, not a step.",
    returns=PlanCheckObservation,
    tags={"knowledge", "plan"},
)
def plan_check(ctx: ToolContext, params: PlanCheckParams) -> PlanCheckObservation:
    from myrmidon.core.artifacts import sha256_cached
    from myrmidon.core.summary import summarize

    plan = read_plan(ctx)
    if plan is None:
        return PlanCheckObservation(
            ok=False,
            summary="this study has no plan — write one with plan_write, or this is a "
            "study that never said what it would do",
            record=False,
        )

    events = _call_events(ctx, plan)
    from myrmidon.core.plan_binding import step_fit

    states, unclaimed, exact = _match_steps(plan, events, step_fit(ctx.workspace, plan))
    done = [s["name"] for s in states if s["state"] == "done"]
    reused = [s["name"] for s in states if s["state"] == "reused"]
    pending = [s["name"] for s in states if s["state"] == "pending"]

    # inputsets: exact, because both sides name an id
    cited = {_protocol_ids(step.cites) for step in plan.steps} - {None}
    applied = {
        ev.params.get("inputset")
        for ev in events
        if ev.tool == "inputset_apply" and ev.params.get("inputset")
    }
    never_applied = sorted(cited - applied)
    uncited = sorted(applied - cited)

    # the approval: a plan_approve naming this exact digest. Because the digest is
    # computed from the plan's content, one comparison answers both "was it approved" and
    # "is it still the plan that was approved" — an edit changes the digest.
    digest = plan.digest()
    path = ctx.workspace.root / PLAN_FILE
    current_sha = sha256_cached(path) if path.exists() else None
    approved_digest, acknowledged, ever_approved = None, [], False
    for event in ctx.workspace.ledger.events():
        if event.tool == "plan_approve" and event.ok:
            approved_sha = next((a.sha256 for a in event.inputs if a.role == "plan"), None)
            if approved_sha is None:
                continue  # a standing grant (core/approvals.py), not an approval of a plan
            ever_approved = True
            if approved_sha is not None and approved_sha == current_sha:
                approved_digest = digest
                acknowledged = list(event.warnings_acknowledged)
            else:
                approved_digest, acknowledged = None, []
    unchanged = approved_digest is not None

    summary_obs = summarize(ctx.workspace, since="all", backend=ctx.backend,
                            compute=getattr(ctx.config, "compute", None), poll_usage=False)
    booked = summary_obs.core_h_submitted
    if plan.plan_id:
        booked = _booked_for(ctx, plan.plan_id)

    lines = [f"{len(done)}/{len(states)} step(s) done"]
    lines += [f"{step.name} reused, not run: {describe_reuse(ctx, step.reuse)}"
              for step in plan.steps if step.reuse]
    if pending:
        lines.append("not yet: " + ", ".join(pending))
    if never_applied:
        lines.append("cited but never applied: " + ", ".join(never_applied))
    if uncited:
        lines.append("applied but not in the plan: " + ", ".join(uncited))
    if unclaimed:
        tools = sorted({ev.tool for ev in unclaimed})
        lines.append(f"{len(unclaimed)} call(s) no step accounts for: " + ", ".join(tools))
    lines.append(
        f"{plan.total_core_h} core-h planned, {booked} booked for it"
        + (f", {summary_obs.core_h_consumed} consumed"
           if summary_obs.core_h_consumed is not None else "")
    )
    if not ever_approved:
        lines.append("not approved: nothing runs against this plan's budget until "
                     "plan_approve records that a person saw the cost")
    elif not unchanged:
        lines.append("the plan on disk is NOT the one that was approved — it changed "
                     "after approval, so the approval does not cover it")
    if not exact and len(states) > 1:
        lines.append("matched by tool and order: no call names the step it belongs to, so "
                     "two steps calling the same tool cannot be told apart")

    return PlanCheckObservation(
        ok=True,
        summary=". ".join(lines),
        record=False,  # a look at the study, not a step of it
        digest=digest,
        plan_id=plan.plan_id or None,
        approved_digest=approved_digest,
        approval_holds=unchanged,
        acknowledged=acknowledged,
        exact=exact,
        steps=states,
        done=done,
        reused=reused,
        pending=pending,
        protocols_never_applied=never_applied,
        protocols_not_in_plan=uncited,
        unaccounted_calls=sorted({ev.tool for ev in unclaimed}),
        planned_core_h=plan.total_core_h,
        booked_core_h=booked,
        consumed_core_h=summary_obs.core_h_consumed,
    )


# --- approving a plan, which is a different act by a different party -----------------


class PlanApproveParams(BaseModel):
    digest: str = Field(
        default="",
        description="the plan digest you are approving, from plan_write or plan_check. "
        "Empty means the study's current plan. Giving it explicitly is how you approve "
        "the plan you actually read.",
    )
    acknowledge_warnings: list[str] = Field(default_factory=list)


def plan_approve_gate(ctx: ToolContext, params: PlanApproveParams):
    warnings: list[ValidationWarning] = []
    checked = ["plan_exists"]

    plan = read_plan(ctx)
    if plan is None:
        return [
            ValidationWarning(
                id="PLAN_ABSENT",
                severity="block",
                message="this study has no plan to approve",
                suggestion="write one with plan_write",
            )
        ], checked

    digest = plan.digest()
    checked.append("digest_matches")
    if params.digest and params.digest != digest:
        warnings.append(
            ValidationWarning(
                id="PLAN_CHANGED_SINCE_YOU_READ_IT",
                severity="block",
                message=f"you are approving {params.digest[:12]} but the study's plan is "
                f"now {digest[:12]} — it changed after you read it",
                assumption="the user would be approving a plan nobody showed them",
                suggestion="read it again with plan_check or `myrmidon plan show`",
            )
        )

    checked.append("executable")
    compute = ctx.config.compute
    priced, _ = _priced(plan, compute)
    still_open = unresolved(ctx, plan)
    if still_open:
        warnings.append(
            ValidationWarning(
                id="PLAN_NOT_EXECUTABLE",
                severity="block",
                message=f"{len(still_open)} value(s) execution will need are not in the "
                "plan; approved without them, the run stops after the approval to ask:\n"
                + "\n".join(_unresolved_lines(still_open)),
                suggestion="ask the researcher for each, put it in its step's `values`, and "
                "write the plan again with overwrite=true",
            )
        )
    stale = [f"{a.name} carries {a.estimate_core_h}, costs {b.estimate_core_h}"
             for a, b in zip(plan.steps, priced.steps, strict=True)
             if a.estimate_core_h != b.estimate_core_h]
    if stale:
        warnings.append(
            ValidationWarning(
                id="PLAN_COST_STALE",
                severity="block",
                message="the plan's costs are not what its resources cost on this machine "
                "— it was written on another machine or edited by hand: " + "; ".join(stale),
                suggestion="write it again with plan_write overwrite=true, which computes "
                "the costs here",
            )
        )

    checked.append("compute_budget")
    total = priced.total_core_h
    ceiling = ceiling_core_h(priced, compute)
    at_most = f" (at most {ceiling} core-h at its ceiling)" if ceiling != total else ""
    if ceiling > compute.warn_core_h:
        warnings.append(
            ValidationWarning(
                id="COMPUTE_BUDGET",
                severity="warn",
                needs_human=True,
                message=f"approving {total} core-h{_floor_note(plan)}{at_most} for "
                f"{plan.question!r} — above the {compute.warn_core_h} core-h threshold. "
                "This is the user's approval of the whole study, its envelope and its "
                f"ceiling, recorded against the plan digest {digest[:12]}",
            )
        )
    else:
        warnings.append(
            ValidationWarning(
                id="COMPUTE_ESTIMATE",
                severity="info",
                message=f"approving {total} core-h{at_most}, under the "
                f"{compute.warn_core_h} core-h threshold",
            )
        )
    return warnings, checked


@tool(
    "plan_approve",
    description="Record the user's approval of this study's plan and its core-hour "
    "total. Approving is not proposing: above the machine's threshold this needs a "
    "human, so an unattended session proposes and reports the cost instead. Recorded "
    "against the plan, so a plan edited afterwards is not approved.",
    returns=PlanObservation,
    tags={"knowledge", "plan"},
    gate=plan_approve_gate,
)
def plan_approve(ctx: ToolContext, params: PlanApproveParams) -> PlanObservation:
    plan = read_plan(ctx)
    if plan is None:  # the gate already refused; belt and braces
        return PlanObservation(ok=False, summary="this study has no plan to approve")
    from myrmidon.core.artifacts import ArtifactRef

    digest = plan.digest()
    return PlanObservation(
        ok=True,
        summary=f"approved: {plan.total_core_h} core-h for {plan.question!r}, "
        f"digest {digest[:12]}",
        # the approval must record WHAT it approved, not merely that it happened: the
        # caller need not pass a digest, so the plan file itself is the record
        inputs=[
            ArtifactRef.from_path(
                ctx.workspace.root / PLAN_FILE, "plan", ctx.workspace.root
            )
        ],
        path=PLAN_FILE,
        digest=digest,
        cites=plan.cites,
        shape=plan.shape,
        question=plan.question,
        total_core_h=plan.total_core_h,
        unestimated=plan.unestimated,
        steps=[s.model_dump(mode="json", exclude_none=True) for s in plan.steps],
        given=plan.given,
        arms=plan.arms.model_dump(mode="json") if plan.arms else None,
        until=plan.until.model_dump(mode="json") if plan.until else None,
    )


# --- the journal: why, left where the loop reads it ------------------------------------


class JournalWriteParams(BaseModel):
    kind: str = Field(
        description="decision (with its reason), ruled_out (with the evidence), suspicion, "
        "or intention")
    text: str = Field(description="one to three sentences; numbers stay in the ledger, "
                      "the journal holds why")
    cites: list[str] = Field(default_factory=list,
                             description="ledger event ids (ev-…) or spool event ids (sp-…) "
                             "the entry rests on")
    subject: str = Field(default="", description="the plan step it is about, if one")
    closes: str | None = Field(default=None, description="an earlier entry (jn-…) this one "
                               "settles, which then stops being shown to a wake")


@tool(
    "journal_write",
    description="Leave a decision, a ruled-out cause, a suspicion or an intention in the "
    "study's journal, citing the events it rests on. Open entries are what the loop shows a "
    "wake-up, so nothing is re-litigated; a closing entry names the one it settles.",
    returns=JournalObservation,
    tags={"plan"},
)
def journal_write(ctx: ToolContext, params: JournalWriteParams) -> JournalObservation:
    from myrmidon.core.spool import Journal

    kinds = ("decision", "ruled_out", "suspicion", "intention")
    if params.kind not in kinds:
        return JournalObservation(ok=False, summary=f"kind is one of {', '.join(kinds)}")
    journal = Journal(ctx.workspace.root)
    by = "model" if ctx.unattended else "human"
    if params.closes:
        entry = journal.close(params.closes, by, params.text)
    else:
        entry = journal.write(params.kind, by, params.text, cites=params.cites,
                              subject=params.subject)
    return JournalObservation(ok=True, summary=f"{entry.kind} {entry.entry_id} written",
                              entry_id=entry.entry_id, kind=entry.kind, record=False)


# --- the report: the product of a study -----------------------------------------------


class StudyReportParams(BaseModel):
    narrative: dict[str, str] = Field(
        default_factory=dict,
        description="your prose for each section, keyed question, methods, ran, checks, "
        "results, deviations, reproduce — placed after the values the harness writes "
        "there. A number in it must be one that section states: say what the values mean, "
        "never a value of your own.",
    )
    output: str = Field(default="report.md",
                        description="a file name in the study's directory")


def study_report_gate(ctx: ToolContext, params: StudyReportParams):
    from myrmidon.core.report import SECTIONS, plan_report, ungrounded

    checked = ["plan_exists"]
    plan = read_plan(ctx)
    if plan is None:
        return [ValidationWarning(
            id="REPORT_NO_PLAN", severity="block",
            message="this study has no plan, so there is nothing to report against",
            suggestion="a report states what a plan set out to do and what it did; write "
            "the plan with plan_write")], checked
    checked.append("inside_the_study")
    from pathlib import PurePosixPath

    if PurePosixPath(params.output).name != params.output or not params.output.endswith(".md"):
        return [ValidationWarning(
            id="REPORT_OUTPUT_NOT_IN_STUDY", severity="block",
            message=f"{params.output!r}: the report is a .md file in the study's own directory",
            suggestion="give a file name such as report.md")], checked
    checked.append("sections_known")
    unknown = sorted(set(params.narrative) - set(SECTIONS))
    if unknown:
        return [ValidationWarning(
            id="REPORT_SECTION_UNKNOWN", severity="block",
            message=f"no section(s) {', '.join(unknown)}; the report's are "
            + ", ".join(SECTIONS))], checked
    checked.append("narrative_grounded")
    sections, _ = plan_report(ctx, plan)
    loose = ungrounded(sections, params.narrative)
    if loose:
        return [ValidationWarning(
            id="REPORT_NUMBER_UNGROUNDED", severity="block",
            message="the prose states numbers its section does not: "
            + "; ".join(f"{k}: {', '.join(v)}" for k, v in loose.items()),
            assumption="a number in the report that no engine produced",
            suggestion="write around the values the section shows; a value it does not "
            "show is not in the study")], checked
    return [], checked


@tool(
    "study_report",
    description="Write the study's report as a file: question, system and methods as "
    "applied, what ran and what was reused, checks, results, deviations, how to reproduce. "
    "The harness writes every value, parsed from the runs' outputs with its source; you "
    "add prose per section, and a number in it that the section does not show is refused.",
    returns=ReportObservation,
    tags={"plan"},
    gate=study_report_gate,
)
def study_report(ctx: ToolContext, params: StudyReportParams) -> ReportObservation:
    from myrmidon.core.artifacts import ArtifactRef
    from myrmidon.core.report import plan_report, render_plan_report

    plan = read_plan(ctx)
    sections, values = plan_report(ctx, plan)
    path = ctx.workspace.root / params.output
    path.write_text(render_plan_report(sections, params.narrative))
    pending = [line.split(":")[0].removeprefix("- ") for line in sections["ran"]
               if line.endswith(": not run")]
    return ReportObservation(
        ok=True,
        summary=f"report written to {ctx.workspace.relpath(path)}: {len(values)} parsed "
        f"value(s)" + (f"; {len(pending)} step(s) of the plan not yet run" if pending else ""),
        path=ctx.workspace.relpath(path),
        artifacts=[ArtifactRef.from_path(path, "report", ctx.workspace.root)],
        plan_id=plan.plan_id or None,
        values=values,
        pending=pending,
    )
