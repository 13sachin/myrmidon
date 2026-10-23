"""The command line for the registry: every tool is a command a person can type.

One rule binds the agent layer to the tool layer: an agent never has a capability the
terminal lacks. ``myrmidon call TOOL key=value ...`` makes that true by construction:
the parameters are the tool's own params model, the call goes through ``REGISTRY.call``
exactly as an agent's does, and a tool added later has a command line without another
line of CLI.

This module is the parsing and the rendering; ``cli.py`` holds the Typer commands.
"""

from __future__ import annotations

import difflib
import json
import textwrap
import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from myrmidon.core.observation import InputSetHit, InputSetObservation, Observation
from myrmidon.core.tool import ToolSpec
from myrmidon.core.validation import ValidationWarning
from myrmidon.errors import MyrmidonError

WIDTH = 88


class UsageError(MyrmidonError):
    """The command line itself was wrong: an unknown tool, a malformed ``key=value``,
    a parameter the tool does not take. Nothing is called; the exit code is 2."""


# --- reading a value off the command line ------------------------------------------
#
# A params model is flat and typed, so the type decides how the text is read. The one
# rule that matters: a ``str`` field takes the literal text, never JSON, so a path like
# ``01_input/mini.mdp`` and a name like ``true`` survive unchanged.


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """``str | None`` -> ``(str, True)``; a real union is left as it is."""
    if get_origin(annotation) in (Union, types.UnionType):
        args = get_args(annotation)
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1:
            return rest[0], len(rest) != len(args)
        return annotation, len(rest) != len(args)
    return annotation, False


def _kind(annotation: Any) -> str:
    """How a command-line value for this field is read: str | list | dict | other."""
    inner, _ = _unwrap_optional(annotation)
    origin = get_origin(inner)
    if origin is Literal:
        return "str" if all(isinstance(a, str) for a in get_args(inner)) else "other"
    if origin in (list, tuple, set) or inner in (list, tuple, set):
        return "list"
    if origin is dict or inner is dict:
        return "dict"
    if inner is str:
        return "str"
    return "other"


def type_name(annotation: Any) -> str:
    """The type as a person should read it in ``call TOOL --help``."""
    inner, optional = _unwrap_optional(annotation)
    origin = get_origin(inner)
    if origin is Literal:
        name = "one of " + " | ".join(str(a) for a in get_args(inner))
    elif origin in (list, tuple, set):
        args = get_args(inner)
        inside = ", ".join(type_name(a) for a in args) if args else ""
        name = f"{origin.__name__}[{inside}]" if inside else origin.__name__
    elif origin is dict:
        name = "dict (JSON)"
    elif isinstance(inner, type):
        name = inner.__name__
    else:
        name = str(inner).replace("typing.", "")
    return f"{name} or none" if optional else name


def _list_items(value: str) -> list[Any]:
    text = value.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise UsageError(f"not valid JSON: {value!r} ({exc})") from None
        if not isinstance(parsed, list):
            raise UsageError(f"expected a JSON list: {value!r}")
        return parsed
    return [item.strip() for item in text.split(",") if item.strip()]


def _coerce(key: str, values: list[str], field: FieldInfo, tool: str) -> Any:
    kind = _kind(field.annotation)
    if len(values) > 1 and kind != "list":
        raise UsageError(
            f"'{key}' given {len(values)} times, but {tool} takes one value for it"
        )
    if kind == "list":
        items: list[Any] = []
        for value in values:
            items.extend(_list_items(value))
        return items
    value = values[0]
    _, optional = _unwrap_optional(field.annotation)
    if optional and value == "null":
        return None
    if kind == "str":
        return value
    if kind == "dict":
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise UsageError(
                f"'{key}' takes JSON, e.g. {key}='{{\"gen_seed\": 20260913}}' ({exc})"
            ) from None
        if not isinstance(parsed, dict):
            raise UsageError(f"'{key}' takes a JSON object, not {value!r}")
        return parsed
    # numbers, booleans, anything else: JSON where it parses, text otherwise — the
    # params model does the coercion and refuses in the tool's own words
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def nested_models(annotation: Any) -> list[type[BaseModel]]:
    """The models inside a field's type, so the help can say what a JSON value holds
    (``mimic_prepare``'s distance restraints are the case)."""
    inner, _ = _unwrap_optional(annotation)
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return [inner]
    found: list[type[BaseModel]] = []
    for arg in get_args(inner):
        found.extend(nested_models(arg))
    return found


def parse_assignments(
    pairs: list[str], model: type[BaseModel], tool: str = "the tool"
) -> dict:
    """``["distance_nm=1.0384", "terms=T,P"]`` -> the dict ``REGISTRY.call`` takes.

    A key the model does not declare is a usage error rather than a silent drop:
    pydantic ignores extra keys, and a typo that goes unnoticed is exactly the kind of
    silent assumption the harness exists to refuse.
    """
    fields = model.model_fields
    collected: dict[str, list[str]] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise UsageError(
                f"{pair!r} is not a parameter — write key=value "
                f"(myrmidon call {tool} --help lists them)"
            )
        if key not in fields:
            near = difflib.get_close_matches(key, list(fields), n=1)
            hint = f" — did you mean '{near[0]}'?" if near else ""
            raise UsageError(
                f"{tool} takes no parameter '{key}'{hint}\n"
                f"  myrmidon call {tool} --help"
            )
        collected.setdefault(key, []).append(value)
    return {
        key: _coerce(key, values, fields[key], tool) for key, values in collected.items()
    }


# --- what a person reads back -------------------------------------------------------


def _wrap(text: str, indent: str = "", width: int = WIDTH) -> list[str]:
    out: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            continue
        out.extend(
            textwrap.wrap(
                para.strip(), width=width, initial_indent=indent, subsequent_indent=indent
            )
        )
    return out


def tool_help(spec: ToolSpec) -> str:
    """``myrmidon call TOOL --help``: the params model is the help."""
    gate = "  [gated: findings must be accepted by id before it runs]" if spec.gate else ""
    lines = [f"myrmidon call {spec.name} [key=value ...]{gate}", ""]
    lines += _wrap(spec.description)
    lines += ["", "parameters:"]
    for name, field in spec.params_model.model_fields.items():
        if field.is_required():
            state = "required"
        else:
            default = field.get_default(call_default_factory=True)
            state = f"default {default!r}"
        lines.append(f"  {name:<22} {type_name(field.annotation):<22} {state}")
        if field.description:
            lines += _wrap(field.description, indent=" " * 6)
        for model in nested_models(field.annotation):
            inside = ", ".join(
                f"{n} ({type_name(f.annotation)})" for n, f in model.model_fields.items()
            )
            lines += _wrap(f"JSON object(s) with: {inside}", indent=" " * 6)
    lines += [
        "",
        "lists take a repeated key or commas (terms=Temperature,Density); dicts take "
        "JSON.",
    ]
    return "\n".join(lines)


def findings_of(obs: Observation) -> list[ValidationWarning]:
    """Every finding the call reported, once each, worst first."""
    groups = [
        getattr(obs, "blocking", []),
        getattr(obs, "unacknowledged", []),
        getattr(obs, "human_required", []),
        obs.warnings,
        obs.advisory,
    ]
    seen: dict[str, ValidationWarning] = {}
    for group in groups:
        for finding in group:
            seen.setdefault(finding.id, finding)
    order = {"block": 0, "warn": 1, "info": 2}
    return sorted(seen.values(), key=lambda w: order.get(w.severity, 3))


def parsed_values(obs: Observation) -> dict[str, Any]:
    """What this observation's own type carries beyond the base: the values the harness
    parsed out of the engine. They are the answer to any question about numbers, so the
    terminal shows them rather than the summary sentence alone."""
    base = set(Observation.model_fields)
    out: dict[str, Any] = {}
    for name in type(obs).model_fields:
        if name in base:
            continue
        value = getattr(obs, name, None)
        if value is None or value == [] or value == {}:
            continue
        out[name] = value
    return out


def _as_text(value: Any, width: int = 160) -> str:
    """One line. A big dict is named by its keys rather than dumped: the values are in
    the file the call wrote and in the summary sentence, and a wrapped JSON blob in the
    middle of a terminal reads as noise."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (str, int, float)) for item in value):
            text = ", ".join(str(item) for item in value)
            return text if len(text) <= width else text[: width - 3] + "..."
        value = [item.model_dump(mode="json", exclude_none=True)
                 if isinstance(item, BaseModel) else item for item in value]
        text = json.dumps(value, default=str)
        return text if len(text) <= width else f"{len(value)} entries"
    if isinstance(value, dict):
        text = json.dumps(value, default=str)
        if len(text) <= width:
            return text
        keys = list(value)
        more = f", +{len(keys) - 6} more" if len(keys) > 6 else ""
        return f"{len(keys)} entries: {', '.join(keys[:6])}{more}"
    return str(value)


def render_observation(obs: Observation) -> str:
    """The observation as a person reads it: what happened, what the harness parsed out
    of the engine, what it was checked for, what it refuses to assume, what it made."""
    lines = [obs.summary.rstrip()]
    values = parsed_values(obs)
    if values:
        lines.append("parsed:")
        width = max(len(name) for name in values)
        for name, value in values.items():
            lines.append(f"  {name:<{width}}  {_as_text(value)}")
    if obs.checked:
        lines.append("checked: " + ", ".join(obs.checked))
    findings = findings_of(obs)
    if findings:
        lines.append("findings:")
        for finding in findings:
            lines.append(f"  {finding.severity:<5} {finding.id}")
            lines += _wrap(finding.message, indent=" " * 8)
            if finding.assumption:
                lines += _wrap(f"would assume: {finding.assumption}", indent=" " * 8)
            if finding.suggestion:
                lines += _wrap(f"suggestion: {finding.suggestion}", indent=" " * 8)
    acceptable = [f.id for f in findings if f.severity == "warn"]
    if acceptable and not obs.ok:
        accept = " ".join(f"--acknowledge {name}" for name in acceptable)
        lines.append(f"accept by id and run it again: {accept}")
    if obs.artifacts:
        lines.append("produced:")
        for ref in obs.artifacts:
            lines.append(f"  {ref.role:<14} {ref.path}")
    if obs.log_tail:
        lines.append("log tail:")
        lines += [f"  {line}" for line in obs.log_tail.splitlines()[-15:]]
    return "\n".join(lines)


def render_protocol_list(obs: InputSetObservation) -> str:
    """``myrmidon inputset list``: what the three layers hold."""
    if not obs.inputsets:
        return "no inputsets found"
    lines = []
    for hit in sorted(obs.inputsets, key=lambda p: p.id):
        layer = hit.layer or hit.scope
        lines.append(f"{hit.id}  v{hit.version}  [{layer}]")
        lines += _wrap(hit.title, indent="    ")
        lines.append(f"    renders: {hit.renders or 'no file'}")
    lines.append("")
    lines.append("myrmidon inputset show ID  — parameters, sources, pitfalls")
    return "\n".join(lines)


def _param_line(key: str, value: Any, hit: InputSetHit) -> list[str]:
    lines = [f"    {key:<28} {value}"]
    why = hit.rationale.get(key)
    if why:
        lines += _wrap(why, indent=" " * 8)
    return lines


def render_protocol(hit: InputSetHit) -> str:
    """One input set in full: the values, who says so, and what it refuses to supply."""
    layer = hit.layer or hit.scope
    lines = [f"{hit.id}  v{hit.version}  [{layer}]", ""]
    lines += _wrap(hit.title)
    lines += _wrap(hit.applicability, indent="  ")
    lines.append("")
    lines.append(f"renders: {hit.renders or 'no file'}")
    if hit.parameters:
        lines.append("")
        lines.append("parameters:")
        for key, value in hit.parameters.items():
            lines += _param_line(key, value, hit)
    if hit.required_from_user:
        lines.append("")
        lines.append("yours to supply:")
        for key, how in hit.required_from_user.items():
            lines.append(f"    {key:<28} YOURS TO SUPPLY")
            lines += _wrap(how, indent=" " * 8)
    if hit.sources:
        lines.append("")
        lines.append("sources:")
        for index, source in enumerate(hit.sources):
            where = source.locator or source.sha256 or ", ".join(source.evidence)
            lines.append(f"  [{index}] {source.kind}: {source.text}")
            if where:
                lines += _wrap(where, indent=" " * 6)
    if hit.pitfalls:
        lines.append("")
        lines.append("pitfalls (finding ids, attached when you apply it): "
                     + ", ".join(hit.pitfalls))
    if hit.notes:
        lines.append("")
        lines += _wrap(hit.notes)
    lines += [
        "",
        f"myrmidon call inputset_apply inputset={hit.id} output=01_input/<file>",
    ]
    return "\n".join(lines)
