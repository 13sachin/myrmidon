"""Tool registry: the single choke point where rigor is enforced.

Tools are pure functions ``(ToolContext, params) -> Observation`` registered
once with a pydantic params model. The registry — not the tool — handles:

- parameter validation against the schema,
- the validation-gate / acknowledge-by-id inputset for expensive tools,
- provenance recording for every call, inputs included, with the ids of the findings
  it surfaced (knowledge that lives in gates, parsers and diagnostics, not in a store),
- the typed next actions a result makes reachable (`NextAction`): what a refusal, an
  invalid call, a missing study or a running job leaves to do, where that is certain.

The same registry backs both the MCP server and the native agent loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

import myrmidon
from myrmidon.core.approvals import approved_by, call_identity
from myrmidon.core.artifacts import ArtifactRef, consumed_inputs
from myrmidon.core.observation import JobObservation, NextAction, Observation, ValidationReport
from myrmidon.core.provenance import CALL_EVENT, CALL_PLAN, ProvenanceEvent, new_event_id
from myrmidon.core.validation import ValidationWarning, decide
from myrmidon.errors import MyrmidonError, UnknownToolError

# A gate returns (warnings, names_of_checks_that_ran).
GateFn = Callable[["ToolContext", BaseModel], tuple[list[ValidationWarning], list[str]]]
ToolFn = Callable[["ToolContext", BaseModel], Observation]

_SCHEMA_HASHES: dict[type[BaseModel], str] = {}


@dataclass(frozen=True)
class FileReader:
    """A kind of file a tool parses: ``matches`` recognises it, ``param`` takes its path.
    A directory listing offers the call for every file some tool reads, so a model that
    sees a run directory also sees the parser for what is in it."""

    param: str
    matches: Callable[[Path], bool]
    what: str  # "a CPMD output": how the listing names the file


def params_schema_hash(model: type[BaseModel]) -> str:
    """A short hash of a tool's params schema, recorded with every call.

    A ledger is only replayable while the tools still take what they took. When a
    params model changes, every event written before it carries the old hash and the
    replay says so, by tool, instead of failing somewhere downstream.
    """
    if model not in _SCHEMA_HASHES:
        import hashlib
        import json

        blob = json.dumps(model.model_json_schema(), sort_keys=True).encode()
        _SCHEMA_HASHES[model] = hashlib.sha256(blob).hexdigest()[:12]
    return _SCHEMA_HASHES[model]


@dataclass
class ToolContext:
    """Injected capabilities — never model-visible."""

    config: object = None  # MyrmidonConfig
    workspace: object = None  # Workspace | None
    backend: object = None  # ExecutionBackend | None
    engines: dict = field(default_factory=dict)  # name -> EngineInfo
    inputsets: object = None  # InputSetStore | None
    protocols: object = None  # ProtocolStore | None — the published pipelines
    # True when no human sees the tool calls (the native agent loop): findings
    # that need a human decision cannot be acknowledged by the model
    unattended: bool = False
    # how config.env_script went (core/envscript.EnvScriptStatus.as_dict()) — doctor shows it
    env_script: dict = field(default_factory=dict)
    # the loop names the plan step a call carries out instead of leaving the alignment to
    # guess: None lets plan_binding choose as for a session's call; a step name binds to
    # that step; "" binds to none. With bound_tag_only the call is recorded under the step
    # but its values are not compared with the step's — a continuation or an extension runs
    # with what the envelope approved, not with the step's own values
    bound_step: str | None = None
    bound_tag_only: bool = False

    def engine_versions(self) -> dict[str, str]:
        return {name: info.version for name, info in self.engines.items() if info.available}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params_model: type[BaseModel]
    returns: type[Observation]
    tags: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()  # engine names that must be available
    expensive: bool = False  # gate + acknowledge-by-id inputset applies
    needs_workspace: bool = True
    gate: GateFn | None = None
    reads: tuple[FileReader, ...] = ()
    # finds something to act on rather than acts: a lookup, a listing, a status report.
    # Never re-run by a replay.
    navigation: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolFn]] = {}

    def register(self, spec: ToolSpec, fn: ToolFn) -> None:
        if spec.name in self._tools:
            raise MyrmidonError(f"tool '{spec.name}' registered twice")
        if spec.expensive and "acknowledge_warnings" not in spec.params_model.model_fields:
            raise MyrmidonError(
                f"expensive tool '{spec.name}' must accept acknowledge_warnings"
            )
        unknown = set(getattr(spec.params_model, "RESEARCHER_FIELDS", ())) - set(
            spec.params_model.model_fields)
        if unknown:
            raise MyrmidonError(
                f"tool '{spec.name}' names RESEARCHER_FIELDS it does not have: "
                f"{', '.join(sorted(unknown))}"
            )
        # a plan prices a submitting step from these fields (plan_tools.price_step); a tool
        # that books resources without naming them would be priced as free
        if callable(getattr(spec.params_model, "resources", None)):
            fields = getattr(spec.params_model, "RESOURCE_FIELDS", None)
            if not fields or set(fields) - set(spec.params_model.model_fields):
                raise MyrmidonError(
                    f"tool '{spec.name}' books resources and must name the fields its "
                    "cost is computed from in RESOURCE_FIELDS"
                )
        self._tools[spec.name] = (spec, fn)

    def specs(self) -> list[ToolSpec]:
        return [spec for spec, _ in self._tools.values()]

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise UnknownToolError(f"unknown tool '{name}'")
        return self._tools[name][0]

    def available(self, engines: dict) -> list[ToolSpec]:
        """Specs whose engine requirements are satisfied on this machine."""
        up = {name for name, info in engines.items() if getattr(info, "available", False)}
        return [spec for spec, _ in self._tools.values() if spec.requires <= up]

    def readers(self, path: Path) -> list[tuple[str, FileReader]]:
        """The tools that parse this file, with how; empty for most files."""
        out = []
        for spec, _ in self._tools.values():
            for reader in spec.reads:
                try:
                    if reader.matches(path):
                        out.append((spec.name, reader))
                except OSError:
                    continue
        return sorted(out, key=lambda pair: pair[0])

    def call_identity(self, name: str, raw_params: dict, ctx: ToolContext | None = None
                      ) -> str | None:
        """The identity a call with these parameters has (core/approvals.call_identity),
        or None when they do not validate — what a researcher's yes is scoped to. Given the
        context, the identity is of the call as it will run: with the approved plan's
        values filled in."""
        spec, _ = self._tools[name]
        try:
            params, from_plan = _validated(ctx, name, spec, raw_params)
        except ValidationError:
            return None
        if ctx is not None:
            params, _, _ = _bound(ctx, name, spec, params, from_plan)
        return call_identity(name, params.model_dump(mode="json", exclude_none=True))

    def call(self, name: str, ctx: ToolContext, raw_params: dict) -> Observation:
        token = CALL_PLAN.set(("", ""))
        event = CALL_EVENT.set(new_event_id())
        try:
            return self._call(name, ctx, raw_params)
        finally:
            CALL_EVENT.reset(event)
            CALL_PLAN.reset(token)

    def _call(self, name: str, ctx: ToolContext, raw_params: dict) -> Observation:
        if name not in self._tools:
            raise UnknownToolError(f"unknown tool '{name}'")
        spec, fn = self._tools[name]

        try:
            params, from_plan = _validated(ctx, name, spec, raw_params)
        except ValidationError as exc:
            missing = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()
                              if e["type"] == "missing"})
            return Observation(
                ok=False, summary=f"invalid parameters for {name}: {exc}",
                next_actions=[NextAction(
                    kind="retry", tool=name, missing=missing,
                    reason="the parameters did not validate: correct the ones the summary "
                    "names" + (", and give the missing ones" if missing else ""))],
                failure="invalid_params")

        if spec.needs_workspace and ctx.workspace is None:
            return Observation(
                ok=False,
                summary=f"{name} needs an active workspace — call study_create or "
                "study_open first",
                next_actions=[NextAction(
                    kind="call", tool="study_open",
                    reason="no study is open: study_open with no path opens the active "
                    "one or names this project's studies; study_create makes a new one, "
                    "from which the project's files are readable; then call this again")],
                failure="no_workspace",
            )

        # an approved plan binds the call that carries out its step: a value left unset is
        # the plan's, a value set otherwise is a departure to put to a person
        params, bound, step = _bound(ctx, name, spec, params, from_plan)
        # the plan this call is made under and the step it carries out: its ledger event
        # and any job it submits carry both, so a plan counts its own calls
        from myrmidon.core.plan_binding import current_plan_id

        CALL_PLAN.set((current_plan_id(ctx.workspace), step))

        # what this call consumes, hashed before the tool can touch it: the registry
        # records the data-flow edges so no tool has to remember to
        dumped = params.model_dump(mode="json", exclude_none=True)
        inputs = (consumed_inputs(dumped, ctx.workspace.root, ctx.workspace.origin)
                  if ctx.workspace else [])

        # once each, in the order given: a model that resends its whole history of ids
        # would otherwise grow one list to hundreds of repeats
        acknowledged: list[str] = list(dict.fromkeys(
            getattr(params, "acknowledge_warnings", []) or []))
        advisory: list[ValidationWarning] = []
        gate_checked: list[str] = []
        approval = ""  # the approval event an acknowledgement relied on, if any

        # Validation gate: expensive tools must pass (or have every warn-level
        # finding acknowledged by id) before the underlying operation runs.
        gated = spec.gate is not None or bool(bound)
        if gated:
            try:
                warnings, checked = spec.gate(ctx, params) if spec.gate is not None \
                    else ([], [])
            except Exception as exc:  # a gate bug is a refusal the model is told, never a
                # crash of the session it runs in (an OSError from a gate's stat, say)
                obs = Observation(ok=False, failure="raised",
                                  summary=f"{name}'s gate raised {type(exc).__name__}: "
                                  f"{exc}; the call was not made")
                self._record(ctx, spec, dumped, obs, acknowledged, inputs, approval,
                             duration_s=0.0, kind="gate_refused")
                return obs
            warnings, checked = [*warnings, *bound], [*checked, *(["plan_binding"] if bound
                                                                  else [])]
            # what a person has approved for this study, as its ledger records it
            # (core/approvals.py): an unattended session may acknowledge exactly those
            approved = approved_by(ctx.workspace, call=call_identity(name, dumped))
            decision = decide(
                warnings, acknowledged, checked,
                unattended=ctx.unattended, approved=list(approved),
            )
            approval = next((approved[w] for w in decision.acknowledged if w in approved), "")
            # what the call carries into the ledger: its own ids and the approval's
            acknowledged = list(dict.fromkeys([*acknowledged, *decision.acknowledged]))
            advisory, gate_checked = decision.advisory, decision.checked
            if not decision.allowed:
                human = ""
                if decision.human_required:
                    human = (
                        f" {len(decision.human_required)} finding(s) "
                        f"{[w.id for w in decision.human_required]} need a human decision "
                        "and this session is unattended: they cannot be acknowledged here — "
                        "stop and report them to the user with the numbers."
                    )
                stopping = decision.blocking + decision.unacknowledged + decision.human_required
                earlier = _earlier_refusals(ctx, name, stopping)
                refused = ValidationReport(
                    ok=False,
                    next_actions=_refusal_actions(
                        name, dumped, acknowledged, decision, approved,
                        repeats=_same_call(name, dumped, earlier)),
                    summary=(
                        _repeated(name, stopping, earlier)
                        + f"{name} refused by validation gate: "
                        f"{len(decision.blocking)} blocking, "
                        f"{len(decision.unacknowledged)} unacknowledged warning(s). "
                        "Blocking findings cannot be overridden; warn-level findings "
                        "may be accepted by listing their ids in acknowledge_warnings."
                        + human
                    ),
                    blocking=decision.blocking,
                    unacknowledged=decision.unacknowledged,
                    human_required=decision.human_required,
                    acknowledged=decision.acknowledged,
                    advisory=decision.advisory,
                    checked=decision.checked,
                    # every finding is in exactly one group above; a second, flat copy in
                    # `warnings` doubled what the model read on every refusal
                    failure="gate_refused",
                )
                self._record(
                    ctx, spec, dumped, refused, acknowledged, inputs, approval, duration_s=0.0,
                    kind="gate_refused",
                )
                return refused

        start = time.monotonic()
        try:
            obs = fn(ctx, params)
        except MyrmidonError as exc:
            obs = Observation(ok=False, summary=f"{name} failed: {exc}", failure="raised")
        except Exception as exc:  # tool bug or dependency failure: stay structured
            import traceback

            tb = "".join(traceback.format_exception(exc)).splitlines()
            obs = Observation(
                ok=False,
                summary=f"{name} raised {type(exc).__name__}: {exc}",
                log_tail="\n".join(tb[-15:]),
                failure="raised",
            )
        duration = time.monotonic() - start

        if gated:
            # an info-severity finding that only surfaces on a refusal informs nobody
            obs.advisory = advisory
            obs.checked = gate_checked
        waiting = _job_wait(obs)
        if waiting is not None:
            obs.next_actions = [*(obs.next_actions or []), waiting]
        self._record(
            ctx, spec, dumped, obs, acknowledged, inputs, approval,
            duration_s=duration, kind=obs.kind,
        )
        return obs

    def _record(
        self,
        ctx: ToolContext,
        spec: ToolSpec,
        params: dict,
        obs: Observation,
        acknowledged: list[str],
        inputs: list[ArtifactRef],
        approval: str,
        duration_s: float,
        kind: str = "call",
    ) -> None:
        workspace = ctx.workspace
        if workspace is None:
            return
        if not obs.record and obs.ok:
            return  # e.g. an unchanged job poll: nothing happened worth a ledger line
        event = ProvenanceEvent(
            event_id=CALL_EVENT.get() or new_event_id(),
            tool=spec.name,
            params=params,
            command=obs.command,
            engine_versions=ctx.engine_versions(),
            myrmidon_version=myrmidon.__version__,
            inputs=_merge_inputs(inputs, obs.inputs),
            outputs=obs.artifacts,
            warnings_acknowledged=acknowledged,
            ok=obs.ok,
            duration_s=round(duration_s, 3),
            kind=kind,
            knowledge_refs=knowledge_refs(obs),
            approval=approval,
            params_schema=params_schema_hash(spec.params_model),
            plan_id=CALL_PLAN.get()[0],
            plan_step=CALL_PLAN.get()[1],
        )
        obs.provenance_id = workspace.ledger.append(event)


def _validated(ctx: ToolContext | None, name: str, spec: ToolSpec, raw: dict
               ) -> tuple[BaseModel, frozenset[str]]:
    """(the call's parameters, the required ones the approved plan supplied). A required
    parameter left out is the plan's when the step it carries out names it: the binding's
    "leave it out, the plan's value is used" holds for required parameters too, so the
    plan is consulted before validation refuses the call. Raises the original
    ValidationError when the plan cannot make the call valid."""
    try:
        return spec.params_model.model_validate(raw), frozenset()
    except ValidationError as exc:
        missing = {e["loc"][0] for e in exc.errors()
                   if e["type"] == "missing" and len(e["loc"]) == 1}
        if ctx is None or ctx.workspace is None or not missing:
            raise
        from myrmidon.core.plan_binding import bind

        filled, _, _ = bind(ctx, name, spec.params_model, dict(raw), set(raw))
        supplied = frozenset(k for k in missing if k in filled)
        if supplied != missing:
            raise
        try:
            return spec.params_model.model_validate(filled), supplied
        except ValidationError:
            raise exc from None


def _bound(ctx: ToolContext, name: str, spec: ToolSpec, params: BaseModel,
           from_plan: frozenset[str] = frozenset()
           ) -> tuple[BaseModel, list[ValidationWarning], str]:
    """The call's parameters with the approved plan's values filled in, the findings
    binding raised (core/plan_binding.py), and the plan step the call carries out ("" for
    none). ``from_plan``: required values validation already took from the plan, which the
    call itself left unset."""
    if ctx.workspace is None:
        return params, [], ""
    from myrmidon.core.plan_binding import bind

    given = {k: v for k, v in params.model_dump(mode="json", exclude_unset=True).items()
             if k not in from_plan}
    filled, bound, step = bind(ctx, name, spec.params_model, given,
                               params.model_fields_set - from_plan)
    if filled == given:
        return params, bound, step
    try:
        return spec.params_model.model_validate(filled), bound, step
    except ValidationError:  # the plan's value does not fit: the call stands as given
        return params, [w for w in bound if w.id != "PLAN_VALUES_APPLIED"], step


def _earlier_refusals(ctx: ToolContext, name: str, stopping: list[ValidationWarning]) -> list:
    """The ledger's earlier refusals of this tool on the same finding(s)."""
    if ctx.workspace is None or not stopping:
        return []
    ids = {w.id for w in stopping}
    return [ev for ev in ctx.workspace.ledger.events()
            if ev.kind == "gate_refused" and ev.tool == name
            and ids <= set(ev.knowledge_refs)]


def _repeated(name: str, stopping: list[ValidationWarning], earlier: list) -> str:
    """A line for a refusal this study has already met: the same tool, stopped by the same
    finding(s). A finding whose suggestion says what to change is right every time it is
    raised; what a repeated refusal adds is the fact that it has been said. Empty for a
    first refusal."""
    if not earlier:
        return ""
    ids = {w.id for w in stopping}
    return (f"REPEATED: {name} has been refused {len(earlier)} time(s) before in this study "
            f"on the same finding(s) {sorted(ids)} (last: {earlier[-1].event_id}). Calling "
            "it again unchanged will be refused again: change the call as this refusal "
            "says, or stop and report it. ")


def _same_call(name: str, params: dict, earlier: list) -> int:
    """How many of the earlier refusals were this very call (acknowledgements aside)."""
    this = call_identity(name, params)
    return sum(1 for ev in earlier if call_identity(name, ev.params) == this)


def _refusal_actions(name: str, params: dict, acknowledged: list[str], decision,
                     approved: dict, repeats: int) -> list[NextAction]:
    """What a refusal leaves to do. One thing, in this order: a person's decision; a
    blocking finding's missing input; the exact call the study's approval already allows;
    a warning to resolve or knowingly accept. A retry carries the refused call, so the model
    changes what the finding names rather than rebuilding it. The harness never fills in an
    acknowledgement the study has not approved, and once this exact call has been refused
    before, the action says so: unchanged it cannot succeed."""
    call = {k: v for k, v in params.items() if k != "acknowledge_warnings"}
    if acknowledged:
        call["acknowledge_warnings"] = acknowledged  # once each
    if decision.human_required:  # unattended: nobody here may accept them
        return [NextAction(
            kind="ask_user", tool=name, needs_human=True,
            finding_ids=[w.id for w in decision.human_required],
            reason="only a researcher may accept these findings: stop, and report them with "
            "their numbers")]
    asked = [w.id for w in decision.unacknowledged if w.needs_human and w.id not in approved]
    if asked and not decision.blocking:  # attended: the person decides, then the call
        return [NextAction(
            kind="ask_user", tool=name, needs_human=True, finding_ids=asked,
            reason="ask the researcher whether to accept these findings, with their numbers; "
            "acknowledge them only on their yes")]
    again = (" This exact call was refused before: unchanged it cannot succeed — change "
             "what the finding names, or stop and report it.") if repeats else ""
    if decision.blocking:
        missing = sorted({n for w in decision.blocking for n in (w.needs or [])})
        return [NextAction(
            kind="escalate" if repeats else "retry", tool=name, known_parameters=call,
            finding_ids=[w.id for w in decision.blocking], missing=missing,
            reason="a blocking finding cannot be acknowledged: change the call as its "
            "suggestion says" + (f" — give or correct {', '.join(missing)}" if missing else "")
            + "." + again)]
    # an approval's findings never reach here: decide() counts them acknowledged
    return [NextAction(
        kind="escalate" if repeats else "retry", tool=name,
        finding_ids=[w.id for w in decision.unacknowledged],
        known_parameters=call,
        reason="change the call so these do not apply, or list their ids in "
        "acknowledge_warnings if the assumption each one names is the one intended." + again)]


def _job_wait(obs: Observation) -> NextAction | None:
    """A job still queued or running: the next step is to wait for it, by id."""
    from myrmidon.execution.backend import JobState

    if not isinstance(obs, JobObservation) or not obs.job_id or not obs.ok:
        return None
    try:
        terminal = JobState(obs.state).terminal
    except ValueError:
        return None
    if terminal:
        return None
    return NextAction(kind="wait", tool="job_status", known_parameters={"job_id": obs.job_id},
                      reason=f"job {obs.job_id} is {obs.state}: job_status with wait_seconds "
                      "blocks until it moves")


def knowledge_refs(obs: Observation) -> list[str]:
    """The ids of the findings this observation carries: gate warnings, advisories and a
    matched failure signature. They name durable knowledge — a gate, a parser field, a
    diagnostic pattern — so a ledger event says which of it was in play."""
    ids: list[str] = []
    for group in ("warnings", "advisory", "blocking", "unacknowledged", "human_required"):
        for finding in getattr(obs, group, None) or []:
            if finding.id not in ids:
                ids.append(finding.id)
    diagnosis_id = getattr(obs, "diagnosis_id", None)
    if diagnosis_id and diagnosis_id not in ids:
        ids.append(diagnosis_id)
    # an input set's pitfalls: the findings it warns of when it is chosen
    ids += [ref for ref in getattr(obs, "pitfalls", None) or [] if ref not in ids]
    return ids


def _merge_inputs(
    from_params: list[ArtifactRef], from_tool: list[ArtifactRef]
) -> list[ArtifactRef]:
    """The registry's view of what was consumed, with the tool's own the authority.

    A tool that records an input names it better than a parameter walk can — it knows
    the file it opened, and it knows a file it consumed that no parameter named.
    """
    by_path = {ref.path: ref for ref in from_params}
    for ref in from_tool:
        by_path[ref.path] = ref
    return list(by_path.values())


REGISTRY = ToolRegistry()


def tool(
    name: str,
    *,
    description: str,
    returns: type[Observation] = Observation,
    tags: frozenset[str] | set[str] = frozenset(),
    requires: frozenset[str] | set[str] = frozenset(),
    expensive: bool = False,
    needs_workspace: bool = True,
    gate: GateFn | None = None,
    reads: tuple[FileReader, ...] = (),
    navigation: bool = False,
    registry: ToolRegistry = REGISTRY,
):
    """Register a tool. The decorated function's second argument must be
    annotated with its pydantic params model."""

    def decorate(fn: ToolFn) -> ToolFn:
        import inspect

        sig = inspect.signature(fn)
        param_names = list(sig.parameters)
        if len(param_names) != 2:
            raise MyrmidonError(f"tool '{name}' must take (ctx, params)")
        annotation = sig.parameters[param_names[1]].annotation
        if isinstance(annotation, str):  # from __future__ annotations
            annotation = inspect.get_annotations(fn, eval_str=True)[param_names[1]]
        if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
            raise MyrmidonError(f"tool '{name}': params must be a pydantic model")
        spec = ToolSpec(
            name=name,
            description=description,
            params_model=annotation,
            returns=returns,
            tags=frozenset(tags),
            requires=frozenset(requires),
            expensive=expensive,
            needs_workspace=needs_workspace,
            gate=gate,
            reads=tuple(reads),
            navigation=navigation,
        )
        registry.register(spec, fn)
        return fn

    return decorate
