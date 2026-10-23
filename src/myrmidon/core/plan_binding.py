"""An approved plan binds the calls that carry it out.

A researcher approves a plan: its steps and each step's ``values``, the parameters its tool
will be called with. Without a comparison of each call with the step it carries out,
approving a plan would approve a document, not what is submitted: a model may call
``gmx_mdrun`` with the tool's defaults instead of the plan's resources, and then only the
budget gate stands between them.

The registry asks :func:`bind` about every call while the study's plan approval holds:

- **which step** this call carries out — the next not-yet-done step of the same tool, by the
  alignment ``plan_check`` uses (``plan_tools._match_steps``); a call no step names is
  unbound, and so is every call when no plan approval holds;
- **a value the call leaves unset** is taken from the step, since that is the value the
  researcher approved — said in the observation as ``PLAN_VALUES_APPLIED``, and recorded
  in the ledger as the call's parameter;
- **a value the call sets differently** is ``PLAN_STEP_DIFFERS``: warn-level and a human's,
  because the plan was a person's decision and so is departing from it.

Paths are compared by file name, since a plan names ``nvt.tpr`` where the call that
carries it out names the stage directory it landed in. A plan is written before the files
exist, so some of its values only stand for one: an artifact role (``structure_processed``)
or a name no file in the study or the project has (``topol.top`` where pdb2gmx wrote
``processed.top``). Such a value is *guessed*: a call naming a real file instead is
carrying the step out, not departing from it (``PLAN_VALUE_GUESSED``, info), and a role is
never filled in as if it were a path. Treating a guessed path as a departure would refuse
correct calls and put them to a person for nothing.

An ``inputset_apply`` call is bound through the step that cites the input set it renders,
when no step calls ``inputset_apply`` itself: the step's values for that input set's keys —
what it leaves to the researcher above all, the charge, the length, the QM selection — are
the call's ``overrides``. That is what makes approvable mean executable: a
decision collected in the planning conversation reaches the file the run reads, instead of
being asked for again after the approval.

A call is bound to the first pending step of its tool that it *fits* — every value it sets
agrees with the step, or the step only guessed it — and to the first pending one only when
it fits none. ``plan_check``'s alignment counts a step done only by a call that fits it, or
whose departure a person approved; an extra grompp that builds the genion input would
otherwise use up the energy-minimisation step and bind the next call to the wrong one.
"""

from __future__ import annotations

import os
import re
from pathlib import PurePosixPath

import yaml

from myrmidon.core.approvals import PLAN_FILE, holding
from myrmidon.core.validation import ValidationWarning


def _same(planned, given) -> bool:
    if isinstance(planned, str) and isinstance(given, str):
        return PurePosixPath(planned).name == PurePosixPath(given).name
    return planned == given


def _roles(plan) -> set[str]:
    return {r for step in plan.steps for r in (*step.needs, *step.produces)}


_SUFFIX = re.compile(r"\.[A-Za-z][A-Za-z0-9_]{0,7}")


def _names_a_file(ws, value: str) -> bool:
    """Whether a file by this path, or by this name anywhere in the study, exists."""
    try:
        if ws.resolve(value).exists():
            return True
        name = PurePosixPath(value).name
        for _, dirs, files in os.walk(ws.root):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if name in files:
                return True
    except (OSError, ValueError):  # a value too long or odd to be a path is not one
        pass
    return False


def guessed(ws, plan, value) -> bool:
    """A plan value that stands for a file the plan could not yet name: an artifact role,
    or a path-like name no file in the study or the project has."""
    if not isinstance(value, str):
        return False
    if value in _roles(plan):
        return True
    if "/" not in value and not _SUFFIX.fullmatch(PurePosixPath(value).suffix or ""):
        return False
    return not _names_a_file(ws, value)


def fits(ws, plan, step, params: dict) -> bool:
    """Every value the call sets agrees with the step, or the step only guessed it — or
    names an earlier run's fact, which only that run can settle."""
    from myrmidon.core.plan import references_in

    return all(key not in params or _same(planned, params[key])
               or guessed(ws, plan, planned) or bool(references_in(planned))
               for key, planned in step.values.items())


def _number(value) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _substitute(value, facts: dict):
    from myrmidon.core.plan import REFERENCE_RE

    if isinstance(value, str):
        whole = REFERENCE_RE.fullmatch(value)
        if whole:  # the value is the fact itself: keep its type
            return facts[whole.groups()]
        return REFERENCE_RE.sub(lambda m: _number(facts[m.groups()]), value)
    if isinstance(value, dict):
        return {k: _substitute(v, facts) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, facts) for v in value]
    return value


def resolve(ctx, states, value) -> tuple[object, list[str], str]:
    """(the value with each ``${step.fact}`` filled in from that step's run, where each
    number came from, why it cannot be filled in yet). A fact is read only from a run that
    completed — the plan's own call for that step, or the run it declared reused — by
    ``core/report.run_facts``, never typed."""
    from myrmidon.core.plan import references_in
    from myrmidon.core.report import run_facts, run_output
    from myrmidon.execution.backend import JobState

    refs = sorted(set(references_in(value)))
    if not refs:
        return value, [], ""
    by_name = {state["name"]: state for state in states}
    facts, sources = {}, []
    for step_name, fact in refs:
        state = by_name.get(step_name)
        if state is None or state["state"] == "pending" or not state["events"]:
            return value, [], f"{step_name} has not run yet"
        out, rec = run_output(ctx, state["events"][0])
        if rec is None:
            return value, [], f"{step_name}'s call submitted no run"
        if rec.state != JobState.COMPLETED:
            return value, [], (f"{step_name}'s run {rec.job_id} is {rec.state.value}, not "
                               "completed (job_status brings the record up to date)")
        found = run_facts(out.read_text(errors="replace")) if out.is_file() else {}
        if fact not in found:
            return value, [], f"{step_name}'s run states no {fact}"
        facts[(step_name, fact)] = found[fact]
        sources.append(f"${{{step_name}.{fact}}} = {_number(found[fact])}, parsed from "
                       f"{ctx.workspace.relpath(out)} (job {rec.job_id})")
    return _substitute(value, facts), sources, ""


def _reference_findings(step, sources: list[str], waiting: list[tuple[str, str]]
                        ) -> list[ValidationWarning]:
    findings = []
    if sources:
        findings.append(ValidationWarning(
            id="PLAN_REFERENCE_RESOLVED", severity="info",
            message=f"step {step.name!r} takes from earlier runs: " + "; ".join(sources),
        ))
    for key, why in waiting:
        findings.append(ValidationWarning(
            id="PLAN_REFERENCE_UNRESOLVED", severity="block",
            message=f"step {step.name!r} sets {key} from an earlier run, but {why}",
            suggestion="run that step to completion first; the value is filled in from its "
            "output then",
        ))
    return findings


def step_fit(ws, plan):
    """The predicate ``plan_tools._match_steps`` aligns ledger calls with: a call carried a
    step out if it fits it, or a person approved its departure from it."""
    def carried_out(step, event) -> bool:
        return "PLAN_STEP_DIFFERS" in event.warnings_acknowledged or \
            fits(ws, plan, step, event.params)
    return carried_out


def bind(ctx, tool: str, params_model, raw: dict, fields_set: set[str]
         ) -> tuple[dict, list[ValidationWarning], str]:
    """(the call's parameters with the plan's values filled in, findings, the step it
    carries out). Unchanged, no findings and no step when the study has no approved plan or
    no pending step of this tool."""
    ws = ctx.workspace
    if ws is None or not (ws.root / PLAN_FILE).is_file():
        return raw, [], ""
    if not any(a.plan_sha256 for a in holding(ws)):
        return raw, [], ""
    from myrmidon.core.plan import Plan
    from myrmidon.plan_tools import _call_events, _match_steps

    try:
        plan = Plan(**yaml.safe_load((ws.root / PLAN_FILE).read_text()))
    except (OSError, ValueError, TypeError):
        return raw, [], ""
    states, _, _ = _match_steps(plan, _call_events(ctx, plan), step_fit(ws, plan))
    forced = getattr(ctx, "bound_step", None)
    if forced is not None:
        # the loop says which step this call carries out (core/loop.py): a continuation or
        # an extension is recorded under its step and runs with the envelope's values
        if forced == "" or forced not in {s.name for s in plan.steps}:
            return raw, [], ""
        if getattr(ctx, "bound_tag_only", False):
            return raw, [], forced
        pending = [plan.step(forced)]
    else:
        pending = [s for s, st in zip(plan.steps, states, strict=True)
                   if s.tool == tool and st["state"] == "pending"]
    if not pending and tool == "inputset_apply":
        return _bind_inputset(ctx, plan, states, raw)
    if not pending:
        return raw, [], ""
    given_now = {k: v for k, v in raw.items() if k in fields_set}
    step = next((s for s in pending if fits(ws, plan, s, given_now)), pending[0])

    fields = set(params_model.model_fields) - {"acknowledge_warnings"}
    roles = _roles(plan)
    filled = dict(raw)
    applied, differs, kept = [], [], []
    sources, waiting, unnamed = [], [], []
    for key, planned in step.values.items():
        if key not in fields:
            continue
        planned, found, why = resolve(ctx, states, planned)
        if why:
            waiting.append((key, why))
            continue
        sources += found
        if key not in fields_set:
            if isinstance(planned, str) and planned in roles:
                # a role is not a path: the call has to name the file. One an earlier step
                # made is a file the plan wired here, and leaving it out would run the
                # tool on its own default instead
                if planned not in plan.given and any(planned in s.produces for s in plan.steps):
                    unnamed.append((key, planned))
                continue
            filled[key] = planned
            applied.append(key)
        elif not _same(planned, raw.get(key)):
            if guessed(ws, plan, planned):
                kept.append((key, planned, raw.get(key)))
            elif plan.arms is not None and plan.arms.covers(key):
                continue  # the plan declared this key as its axis, and said why
            else:
                differs.append((key, planned, raw.get(key)))
    if tool == "inputset_apply" and "inputset" not in fields_set and step.cites:
        # the step renders the input set it cites: its `cites` names it
        filled["inputset"] = step.cites.removeprefix("inputset:").partition("@")[0]
        applied.append("inputset")

    findings = []
    if applied:
        findings.append(ValidationWarning(
            id="PLAN_VALUES_APPLIED", severity="info",
            message=f"{tool} carries out step {step.name!r} of the approved plan; taken from "
            "it: " + ", ".join(f"{k} = {filled[k]}" for k in applied),
        ))
    if kept:
        findings.append(ValidationWarning(
            id="PLAN_VALUE_GUESSED", severity="info",
            message=f"{tool} carries out step {step.name!r} of the approved plan, which named "
            + ", ".join(f"{k} = {p}" for k, p, _ in kept) + " before the file existed; no "
            "file by that name is in the study, so this call's "
            + ", ".join(f"{k} = {g}" for k, _, g in kept) + " is used",
        ))
    for key, planned, given in differs:
        findings.append(ValidationWarning(
            id="PLAN_STEP_DIFFERS", severity="warn", needs_human=True,
            message=f"{tool} carries out step {step.name!r} of the approved plan, which says "
            f"{key} = {planned}; this call has {key} = {given}",
            assumption="the researcher approved the plan's value, not this one",
            suggestion=f"leave {key} out (the plan's value is used), or ask the researcher "
            "whether to depart from the plan",
        ))
    if tool == "inputset_apply":
        inputset = _inputset(ctx, filled.get("inputset"))
        if inputset is not None:
            filled, more = _overrides_from(ctx, states, inputset, step, filled)
            findings += more
    for key, role in unnamed:
        findings.append(ValidationWarning(
            id="PLAN_ROLE_UNNAMED", severity="block",
            message=f"{tool} carries out step {step.name!r}, whose plan feeds {key} the file "
            f"{role} an earlier step made; this call names no {key}",
            suggestion=f"give {key} the path of that {role} file",
        ))
    return filled, findings + _reference_findings(step, sources, waiting), step.name


def _inputset(ctx, ref):
    store = getattr(ctx, "inputsets", None)
    if store is None or not ref:
        return None
    try:
        return store.get(ref)
    except (KeyError, AttributeError):
        return None


def _planned(inputset, step) -> dict:
    """The step's values for the input set's own keys: what it decided for that set."""
    return {k: v for k, v in step.values.items() if k in inputset.parameters}


def _overrides_from(ctx, states, inputset, step, raw: dict
                    ) -> tuple[dict, list[ValidationWarning]]:
    """The call's ``overrides`` with the step's values for the input set's keys filled in,
    references to earlier runs resolved, and the findings: applied (info), set otherwise (a
    departure, a person's), or waiting on a run that has not completed (a block)."""
    values, sources, waiting = {}, [], []
    for key, planned in _planned(inputset, step).items():
        planned, found, why = resolve(ctx, states, planned)
        if why:
            waiting.append((key, why))
        else:
            values[key] = planned
            sources += found
    given = dict(raw.get("overrides") or {})
    applied = [k for k in values if k not in given]
    differs = [(k, values[k], given[k]) for k in values if k in given and given[k] != values[k]]
    findings = []
    if applied:
        findings.append(ValidationWarning(
            id="PLAN_VALUES_APPLIED", severity="info",
            message=f"{inputset.id} is rendered for step {step.name!r} of the approved plan; "
            "taken from it: " + ", ".join(f"{k} = {values[k]}" for k in applied),
        ))
    for key, planned_value, given_value in differs:
        findings.append(ValidationWarning(
            id="PLAN_STEP_DIFFERS", severity="warn", needs_human=True,
            message=f"{inputset.id} is rendered for step {step.name!r} of the approved plan, "
            f"which says {key} = {planned_value}; this call has {key} = {given_value}",
            assumption="the researcher approved the plan's value, not this one",
            suggestion=f"leave {key} out of overrides (the plan's value is used), or ask the "
            "researcher whether to depart from the plan",
        ))
    findings += _reference_findings(step, sources, waiting)
    if not applied:
        return raw, findings
    return dict(raw, overrides={**given, **{k: values[k] for k in applied}}), findings


def _bind_inputset(ctx, plan, states, raw: dict
                   ) -> tuple[dict, list[ValidationWarning], str]:
    """An input set rendered for a pending step of another tool that cites it: the step's
    values for the input set's keys become the call's ``overrides``."""
    inputset = _inputset(ctx, raw.get("inputset"))
    if inputset is None:
        return raw, [], ""

    def cites(step) -> bool:
        ref = (step.cites or "").removeprefix("inputset:").partition("@")[0]
        cited = _inputset(ctx, ref)
        return cited is not None and cited.id == inputset.id

    pending = [s for s, st in zip(plan.steps, states, strict=True)
               if st["state"] == "pending" and cites(s) and _planned(inputset, s)]
    if not pending:
        return raw, [], ""
    given = dict(raw.get("overrides") or {})
    step = next((s for s in pending
                 if all(k not in given or given[k] == v
                        for k, v in _planned(inputset, s).items())), pending[0])
    filled, findings = _overrides_from(ctx, states, inputset, step, raw)
    return filled, findings, step.name


def current_plan_id(ws) -> str:
    """The id of the plan on disk, which a call made now is made under; "" when the study
    has none, or has one written before plans had ids."""
    path = ws.root / PLAN_FILE if ws is not None else None
    if path is None or not path.is_file():
        return ""
    try:
        return str((yaml.safe_load(path.read_text()) or {}).get("plan_id") or "")
    except (OSError, yaml.YAMLError):
        return ""


def approved_steps(ws, tool: str) -> list:
    """The steps of this tool in the plan a person approved and that still holds; [] when
    no plan approval holds."""
    if ws is None or not (ws.root / PLAN_FILE).is_file():
        return []
    if not any(a.plan_sha256 for a in holding(ws)):
        return []
    from myrmidon.core.plan import Plan

    try:
        plan = Plan(**yaml.safe_load((ws.root / PLAN_FILE).read_text()))
    except (OSError, ValueError, TypeError):
        return []
    return [step for step in plan.steps if step.tool == tool]
