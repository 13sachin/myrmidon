"""Tool registry: the single choke point where rigor is enforced.

Tools are pure functions ``(ToolContext, params) -> Observation`` registered
once with a pydantic params model. The registry — not the tool — handles:

- parameter validation against the schema,
- the one gate: a call that books resources is held to the researcher's approved mandate
  (execution/gate.py); a tool checks its own preconditions inline (`precheck`),
- provenance recording for every call, inputs included, with the ids of the findings
  it surfaced (knowledge that lives in gates, parsers and diagnostics, not in a store).

The same registry backs both the MCP server and the native agent loop.
"""

from __future__ import annotations

import json
import time
import types
import typing
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

import myrmidon
from myrmidon.core.artifacts import ArtifactRef, consumed_inputs
from myrmidon.core.contracts import Contract, check_outputs
from myrmidon.core.mandate import deliverable_findings
from myrmidon.core.mandate import holding as holding_mandate
from myrmidon.core.observation import Observation, ToolError
from myrmidon.core.provenance import (
    CALL_EVENT,
    CALL_FINDINGS,
    CALL_INPUTS,
    CALL_MANDATE,
    ProvenanceEvent,
    new_event_id,
)
from myrmidon.core.validation import Finding, decide
from myrmidon.errors import MyrmidonError, UnknownToolError

ToolFn = Callable[["ToolContext", BaseModel], Observation]


class Refused(MyrmidonError):
    """A tool's own check stopped the call: the findings, the stopping ones first."""

    def __init__(self, findings: list[Finding]):
        super().__init__("; ".join(f.message for f in findings))
        self.findings = findings


class Collected(Exception):
    """What a booking tool's own checks found, handed to the registry before anything
    runs, so it judges them with the submission gate in one decision."""

    def __init__(self, findings: list[Finding]):
        self.findings = findings


#: how ``precheck`` acts in this call: None, it judges; "collect", it hands its findings
#: to the registry (a booking tool, before the gate); "judged", the registry already has
CALL_CHECKS: ContextVar[str | None] = ContextVar("CALL_CHECKS", default=None)


def precheck(ctx: ToolContext, found: list[Finding]) -> None:
    """A tool's own preconditions, checked inline before it acts: a block stops the call
    (raised, and recorded by the registry as refused); a warn is a person's to accept and
    stops a model's call; everything else rides on the result as a finding. A tool that
    books calls it first: the registry collects its findings and judges them with the
    submission gate."""
    mode = CALL_CHECKS.get()
    if mode == "collect":
        raise Collected(found)
    if mode == "judged":
        return
    stopping = [f for f in found if f.severity == "block"
                or (f.severity == "warn" and not getattr(ctx, "person", None))]
    if stopping:
        raise Refused([*stopping, *(f for f in found if f not in stopping)])
    noticed = CALL_FINDINGS.get()
    if noticed is not None:
        noticed.extend(found)

_SCHEMA_HASHES: dict[type[BaseModel], str] = {}
#: the longest description a tool may carry
MAX_DESCRIPTION = 300
#: every tool that books resources takes ``preview``
PREVIEW_HELP = ("true: say what this would cost and whether the approved mandate carries "
                "it (its ceiling, its system, the researcher's decisions); nothing is "
                "submitted")


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

    When a params model changes, every event written before it carries the old hash, so
    a reader of an old ledger can tell which calls were made against a tool that has since
    changed what it takes.
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
    # True when no human sees the tool calls (the unattended loop, the runner)
    unattended: bool = False
    # how config.env_script went (core/envscript.EnvScriptStatus.as_dict()) — doctor shows it
    env_script: dict = field(default_factory=dict)
    # the person at the terminal, by name: set only by a command a person runs
    # (`myrmidon approve`), never by a server or a loop. The one thing a gate can ask for
    # that an agent cannot supply: a warn-level finding is theirs to accept
    person: str | None = None
    # True when a model makes the calls (the MCP server, the native loop); a person's call
    # from the shell leaves it False, so a journal entry knows whose it is
    model_driven: bool = False

    def engine_versions(self) -> dict[str, str]:
        return {name: info.version for name, info in self.engines.items() if info.available}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params_model: type[BaseModel]
    tags: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()  # engine names that must be available
    expensive: bool = False  # it runs an engine or books: the submission gate applies
    needs_workspace: bool = True
    reads: tuple[FileReader, ...] = ()
    # finds something to act on rather than acts: a lookup, a listing, a status report.
    # The report does not count it as work.
    navigation: bool = False
    # what the tool consumes and produces, by artifact kind (core/contracts.py): the call
    # is checked against it after it runs, and the ledger's edges follow it
    contract: Contract | None = None


def books(spec: ToolSpec) -> bool:
    """Whether a call of this tool books resources: its params state ``resources()``."""
    return callable(getattr(spec.params_model, "resources", None))


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolFn]] = {}
        # where a call that books resources runs, what it costs and whether the approved
        # mandate carries it, applied beside the tool's own gate to every such tool
        # (execution/gate.py, installed by register_all); it is told the tool's name
        self.submission_gate: Callable[..., tuple[list[Finding], list[str]]] \
            | None = None

    def register(self, spec: ToolSpec, fn: ToolFn) -> None:
        if spec.name in self._tools:
            raise MyrmidonError(f"tool '{spec.name}' registered twice")
        # a description is read on every turn a client shows the tool: what it is for, not
        # how it works
        if len(spec.description) > MAX_DESCRIPTION:
            raise MyrmidonError(
                f"tool '{spec.name}': its description is {len(spec.description)} characters; "
                f"at most {MAX_DESCRIPTION}")
        # the gate prices a booking from these fields; a tool that books resources without
        # naming them would be priced as free
        if books(spec):
            fields = getattr(spec.params_model, "RESOURCE_FIELDS", None)
            if not fields or set(fields) - set(spec.params_model.model_fields):
                raise MyrmidonError(
                    f"tool '{spec.name}' books resources and must name the fields its "
                    "cost is computed from in RESOURCE_FIELDS"
                )
            if "preview" not in spec.params_model.model_fields:
                raise MyrmidonError(
                    f"tool '{spec.name}' books resources and must take preview"
                )
            # its own checks are collected before anything books (`_collected`)
            if not _checks_first(fn):
                raise MyrmidonError(
                    f"tool '{spec.name}' books resources and must call precheck first"
                )
            # the cost gate prices every booking before it exists
            if not spec.expensive:
                raise MyrmidonError(
                    f"tool '{spec.name}' books resources and must be expensive=True"
                )
        # a contract over parameters the tool does not take would check nothing
        if spec.contract is not None:
            unknown = spec.contract.params_named() - set(spec.params_model.model_fields)
            if unknown:
                raise MyrmidonError(
                    f"tool '{spec.name}': its contract names parameters it does not have: "
                    f"{', '.join(sorted(unknown))}"
                )
            unread = {out.when.split(".", 1)[0] for out in spec.contract.produces
                      if out.when} - set(spec.contract.consumes)
            if unread:
                raise MyrmidonError(
                    f"tool '{spec.name}': its contract conditions an output on parameters "
                    f"it does not consume: {', '.join(sorted(unread))}"
                )
        self._tools[spec.name] = (spec, fn)

    def specs(self) -> list[ToolSpec]:
        return [spec for spec, _ in self._tools.values()]

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise UnknownToolError(f"unknown tool '{name}'")
        return self._tools[name][0]

    def function(self, name: str) -> ToolFn:
        """The registered function behind a tool, for reading what it checks."""
        if name not in self._tools:
            raise UnknownToolError(f"unknown tool '{name}'")
        return self._tools[name][1]

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

    def call(self, name: str, ctx: ToolContext, raw_params: dict) -> Observation:
        event = CALL_EVENT.set(new_event_id())
        mandate = CALL_MANDATE.set("")
        try:
            return self._call(name, ctx, raw_params)
        finally:
            CALL_MANDATE.reset(mandate)
            CALL_EVENT.reset(event)

    def _call(self, name: str, ctx: ToolContext, raw_params: dict) -> Observation:
        if name not in self._tools:
            raise UnknownToolError(f"unknown tool '{name}'")
        spec, fn = self._tools[name]

        try:
            params = spec.params_model.model_validate(lenient(spec.params_model, raw_params))
        except ValidationError as exc:
            missing = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()
                              if e["type"] == "missing"})
            return Observation(
                ok=False, summary=f"invalid parameters for {name}: {exc}",
                error=ToolError(code="INVALID_PARAMS", message=str(exc),
                                hint="correct the parameters the message names"
                                + (f"; give {', '.join(missing)}" if missing else "")),
                failure="invalid_params")

        if spec.needs_workspace and ctx.workspace is None:
            return Observation(
                ok=False, summary=f"{name} needs an open study",
                error=ToolError(code="NO_STUDY", message=f"{name} needs an open study",
                                hint="study_open with no path opens the active one or names "
                                "this project's studies; study_create makes a new one"),
                failure="no_workspace")

        unsized: list[Finding] = []
        sized = getattr(params, "sized", None)
        if callable(sized) and ctx.workspace is not None:
            # what the call leaves unset that the study can fill (a walltime from the run's
            # length, a deliverable's measurement): filled before anything judges it or the
            # ledger records it, so both see the call that is made
            try:
                params = sized(ctx)
            except Exception as exc:  # noqa: BLE001 — the call runs as given, and says so
                unsized.append(Finding(
                    id="CALL_NOT_SIZED",
                    message=f"{name} could not fill what the call left unset "
                    f"({type(exc).__name__}: {exc}); it runs as given"))

        if books(spec):
            # the approved mandate a submission is made under: its event and its job say so
            held = holding_mandate(ctx.workspace)
            CALL_MANDATE.set(held.digest if held is not None else "")

        # what this call consumes, hashed before the tool can touch it: the registry
        # records the data-flow edges so no tool has to remember to
        dumped = params.model_dump(mode="json", exclude_none=True)
        inputs = (consumed_inputs(dumped, ctx.workspace.root, ctx.workspace.origin)
                  if ctx.workspace else [])
        inputs += _included(name, inputs, ctx)

        approved_ids: list[str] = []  # warn-severity ids a person's own call accepted
        findings: list[Finding] = [*unsized]

        # the one gate: a booking against the mandate, with the tool's own checks. A block
        # refuses; a warn refuses unless a person makes the call
        submission = self.submission_gate if books(spec) else None
        judged = submission is not None
        if submission is not None:
            try:
                found = [*_collected(fn, ctx, params), *submission(ctx, params, name)[0]]
            except Exception as exc:  # a gate bug is a refusal the model is told, never a
                # crash of the session it runs in (an OSError from a gate's stat, say)
                obs = Observation(ok=False, failure="raised",
                                  summary=f"{name}'s gate raised {type(exc).__name__}: "
                                  f"{exc}; the call was not made",
                                  error=ToolError(code="GATE_RAISED", message=str(exc)))
                self._record(ctx, spec, dumped, obs, approved_ids, inputs,
                             duration_s=0.0, kind="gate_refused")
                return obs
            decision = decide(found, person=bool(ctx.person))
            approved_ids = decision.approved
            findings = decision.advisory
            if books(spec) and getattr(params, "preview", False):
                previewed = _preview(name, decision, params)
                self._record(ctx, spec, dumped, previewed, approved_ids, inputs,
                             duration_s=0.0, kind="observe")
                return previewed
            if not decision.allowed:
                refused = _refusal(name, [*decision.blocking, *decision.unapproved],
                                   decision.advisory)
                self._record(ctx, spec, dumped, refused, approved_ids, inputs,
                             duration_s=0.0, kind="gate_refused")
                return refused

        start = time.monotonic()
        consumed = CALL_INPUTS.set(inputs)
        raised = CALL_FINDINGS.set([])
        checks = CALL_CHECKS.set("judged" if judged else None)
        try:
            obs = fn(ctx, params)
        except Refused as exc:
            obs = _refusal(name, exc.findings, [])
        except MyrmidonError as exc:
            obs = Observation(ok=False, summary=f"{name} failed: {exc}", failure="raised",
                              error=ToolError(code="FAILED", message=str(exc)))
        except Exception as exc:  # tool bug or dependency failure: stay structured
            import traceback

            tb = "".join(traceback.format_exception(exc)).splitlines()
            obs = Observation(
                ok=False,
                summary=f"{name} raised {type(exc).__name__}: {exc}",
                log_tail="\n".join(tb[-15:]),
                failure="raised",
                error=ToolError(code="RAISED", message=f"{type(exc).__name__}: {exc}"),
            )
        finally:
            noticed = CALL_FINDINGS.get()
            CALL_FINDINGS.reset(raised)
            CALL_INPUTS.reset(consumed)
            CALL_CHECKS.reset(checks)
        duration = time.monotonic() - start

        if obs.failure != "gate_refused":
            obs.findings = [*findings, *noticed, *obs.findings]
            # a warn the tool's own check let through is one the person calling accepted
            approved_ids = [*approved_ids, *(f.id for f in noticed if f.severity == "warn"
                                             and f.id not in approved_ids)]
        if obs.ok and ctx.workspace is not None:
            # a number the approved mandate names as a deliverable, measured otherwise; the
            # call has run, so a fault here is said and the call is still recorded
            try:
                obs.findings = [*obs.findings,
                                *deliverable_findings(ctx.workspace, name, dumped)]
            except Exception as exc:  # noqa: BLE001
                obs.findings = [*obs.findings, Finding(
                    id="DELIVERABLE_CHECK_FAILED",
                    message=f"the mandate's deliverables could not be compared: {exc}")]
        if obs.ok and spec.contract is not None and ctx.workspace is not None:
            # the tool held to its contract: a kind it declared it would write under
            # these inputs and did not record is said, never passed over
            obs.findings = [*obs.findings, *(
                Finding(id="CONTRACT_OUTPUT_MISSING",
                        message=f"{name}: {note}")
                for note in check_outputs(spec.contract, ctx, dumped, obs.artifacts))]
        self._record(
            ctx, spec, dumped, obs, approved_ids, inputs,
            duration_s=duration, kind="gate_refused" if obs.failure == "gate_refused"
            else obs.kind,
        )
        return obs

    def _record(
        self,
        ctx: ToolContext,
        spec: ToolSpec,
        params: dict,
        obs: Observation,
        approved: list[str],
        inputs: list[ArtifactRef],
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
            warnings_acknowledged=approved,
            ok=obs.ok,
            duration_s=round(duration_s, 3),
            exit_code=getattr(obs, "exit_code", None),
            log_tail=obs.log_tail,
            kind=kind,
            knowledge_refs=knowledge_refs(obs),
            params_schema=params_schema_hash(spec.params_model),
            mandate=CALL_MANDATE.get(),
        )
        obs.provenance_id = workspace.ledger.append(event)


def _checks_first(fn: ToolFn) -> bool:
    """Whether the function's first statement (after its docstring) calls `precheck`."""
    import ast
    import inspect
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        return False
    body = next((n.body for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)), [])
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    first = body[0] if body else None
    return (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
            and getattr(first.value.func, "id", "") == "precheck")


def _collected(fn: ToolFn, ctx: ToolContext, params: BaseModel) -> list[Finding]:
    """A booking tool's own findings, collected without running it: its first statement is
    `precheck`, which hands them over in this mode."""
    token = CALL_CHECKS.set("collect")
    try:
        fn(ctx, params)
    except Collected as found:
        return found.findings
    finally:
        CALL_CHECKS.reset(token)
    raise MyrmidonError("a tool that books must call precheck before it acts")


def _refusal(name: str, stopping: list[Finding],
             rest: list[Finding]) -> Observation:
    """A call that did not run: the stopping finding as the error, and every finding."""
    first = stopping[0]
    theirs = first.severity == "warn"
    hint = first.suggestion or ("the researcher's to accept: say what it is and what it "
                                "costs, and end the turn" if theirs else "")
    return Observation(
        ok=False, summary=f"{name} refused: " + " ".join(f.message for f in stopping),
        error=ToolError(code=first.id, message=first.message, hint=hint),
        findings=[*stopping, *rest], failure="gate_refused")


def _preview(name: str, decision, params) -> Observation:
    """What a booking would meet, said without making it: the booking (its walltime and
    ranks as the call is sized), the price and the mandate's verdict, every finding."""
    said = [w.message for w in (*decision.blocking, *decision.unapproved, *decision.advisory)]
    verdict = "would be submitted" if decision.allowed else "would not be submitted"
    stopping = [*decision.blocking, *decision.unapproved]
    booking = params.resources()  # a booking tool's, as the gate priced it
    return Observation(
        ok=decision.allowed, summary=f"preview of {name}: {verdict}; nothing was submitted. "
        + " ".join(said), findings=[*stopping, *decision.advisory],
        error=ToolError(code=stopping[0].id, message=stopping[0].message,
                        hint=stopping[0].suggestion or "") if stopping else None,
        walltime_min=booking.walltime_min, ntasks=booking.ntasks)


def _included(tool: str, inputs: list[ArtifactRef], ctx: ToolContext) -> list[ArtifactRef]:
    """The files the call's inputs include (a topology's ``#include``d .itp files): the
    engine reads them with the file that names them, so they are inputs too, under its
    role."""
    from myrmidon.core.contracts import includes_of
    from myrmidon.core.lineage import kind_read

    if ctx.workspace is None:
        return []
    root = ctx.workspace.root.resolve()
    origin = Path(ctx.workspace.origin).resolve() if ctx.workspace.origin else None
    seen = {ref.path for ref in inputs}
    out = []
    for ref in inputs:
        kind = kind_read(tool, ref.role.split(".", 1)[0].split("[", 1)[0]) or ""
        for path in includes_of(kind, ctx.workspace.resolve(ref.path)):
            real = path.resolve()
            inside = real.is_relative_to(root)
            if not inside and not (origin is not None and real.is_relative_to(origin)):
                continue  # a library file, not the study's nor the researcher's
            rel = ctx.workspace.relpath(path) if inside else str(real)
            if rel in seen:
                continue
            seen.add(rel)
            out.append(ArtifactRef.from_path(path, f"{ref.role}.include",
                                             root=root if inside else None))
    return out


#: how a value with no value is spelled by a model writing JSON as text
NONE_SPELLINGS = ("none", "null")


def takes_none(annotation) -> bool:
    return type(None) in typing.get_args(annotation)


def takes_list(annotation) -> bool:
    """A list, or a union one of whose members is a list (never a list inside a dict)."""
    if typing.get_origin(annotation) is list:
        return True
    union = typing.get_origin(annotation) in (typing.Union, types.UnionType)
    return union and any(typing.get_origin(a) is list for a in typing.get_args(annotation))


def lenient(model: type[BaseModel], raw: dict) -> dict:
    """A call's parameters with the spellings whose meaning is not in doubt read as meant:
    ``"None"`` or ``"null"`` for a field that takes no value, a bare string for a list of
    one, a string holding a JSON list for that list."""
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    for name, value in raw.items():
        field = model.model_fields.get(name)
        if field is None or not isinstance(value, str):
            continue
        annotation = field.annotation
        if takes_none(annotation) and (value.strip().lower() in NONE_SPELLINGS or (
                not value.strip() and str not in typing.get_args(annotation))):
            out[name] = None  # spelled as no value, or left empty where words are no value
        elif takes_list(annotation):
            text = value.strip()
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                except ValueError:
                    parsed = None
                if isinstance(parsed, list):
                    out[name] = parsed
                    continue
            out[name] = [value]
    return out


def contract_of(tool: str) -> Contract | None:
    """The registered tool's contract; None for an unknown tool or one without."""
    try:
        return REGISTRY.get(tool).contract
    except UnknownToolError:
        return None


def knowledge_refs(obs: Observation) -> list[str]:
    """The ids of the findings this observation carries: gate warnings and advisories.
    They name durable knowledge — a gate, a parser field — so a ledger event says which
    of it was in play."""
    ids: list[str] = []
    for finding in obs.findings:
        if finding.id not in ids:
            ids.append(finding.id)
    # an input set's pitfalls: the findings it warns of when it is chosen
    ids += [ref for ref in getattr(obs, "pitfalls", None) or [] if ref not in ids]
    return ids + [ref for ref in obs.refs if ref not in ids]


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
    tags: frozenset[str] | set[str] = frozenset(),
    requires: frozenset[str] | set[str] = frozenset(),
    expensive: bool = False,
    needs_workspace: bool = True,
    reads: tuple[FileReader, ...] = (),
    navigation: bool = False,
    contract: Contract | None = None,
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
            tags=frozenset(tags),
            requires=frozenset(requires),
            expensive=expensive,
            needs_workspace=needs_workspace,
            reads=tuple(reads),
            navigation=navigation,
            contract=contract,
        )
        registry.register(spec, fn)
        return fn

    return decorate
