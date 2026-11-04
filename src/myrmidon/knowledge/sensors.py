"""Sensors: what a careful researcher notices in a file, as versioned records.

A sensor record says, for one kind of subject (an mdp, a CPMD input, a system's profile),
when it applies (``when``, a condition over the subject's parsed values), what it sees
(``sees``, with the values filled in), what that means for the physics (``means``), the
usual move (``usual_move``), the engine versions it holds for, whether it is ``verified`` by
a person or still a ``candidate``, and its sources. Records live in ``knowledge/rules/``
under ``sensors``; code keeps only the parsers and predicates that build a subject.

Almost no record stops a call: it is information the model reads at the moment it acts,
decides on and records the reason for. ``stops_the_call`` is for the few where the engine
would run silently and wrongly.

A condition maps a subject key to what it must be: a scalar (equal: numbers as numbers,
text case aside), a list (one of; ``null`` in it admits an absent key), ``null`` (absent),
or an operator map — ``gt``/``ge``/``lt``/``le`` (the value's first number), ``not`` (absent,
or none of these), ``differs_from`` (another key, both present and unequal). A boolean
matches the engines' spellings: true as yes/true/on, false as no/false/off — and YAML reads
an unquoted ``yes``, ``no``, ``on`` or ``off`` in a record as a boolean. Conditions are
checked when the records load, so a record that cannot be read fails there, not inside a
tool call.
"""

from __future__ import annotations

from fnmatch import fnmatch
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from myrmidon.core.validation import Finding, Precondition
from myrmidon.knowledge.sources import Source

RULES_DIR = Path(__file__).parent / "rules"
_NUMERIC_OPS = ("gt", "ge", "lt", "le")


class SensorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    subject: str  # the kind of thing it reads: mdp, cpmd_input, system, ...
    stage: str = ""  # when in a study it applies, for a system record
    engine: str = ""
    versions: list[str] = Field(default_factory=list)  # fnmatch patterns; empty: any
    each: Literal["key"] | None = None  # evaluated once per key of the file
    when: dict[str, Any] = Field(default_factory=dict)
    sees: str
    means: str = ""
    usual_move: str = ""
    assumption: str = ""
    stops_the_call: bool = False
    status: Literal["candidate", "verified"] = "candidate"
    sources: list[Source] = Field(default_factory=list)

    @field_validator("when")
    @classmethod
    def _readable(cls, when: dict[str, Any]) -> dict[str, Any]:
        for key, spec in when.items():
            if not isinstance(spec, dict):
                continue
            for op, arg in spec.items():
                if op in _NUMERIC_OPS:
                    if isinstance(arg, bool) or not isinstance(arg, int | float):
                        raise ValueError(f"{key}: {op} takes a number, not {arg!r}")
                elif op == "differs_from":
                    if not isinstance(arg, str):
                        raise ValueError(f"{key}: differs_from takes a key, not {arg!r}")
                elif op != "not":
                    raise ValueError(f"{key}: unknown condition {op!r}")
        return when


@cache
def load_sensors(directory: Path = RULES_DIR) -> tuple[SensorRecord, ...]:
    out: list[SensorRecord] = []
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        for item in data.get("sensors", []):
            try:
                out.append(SensorRecord.model_validate(item))
            except ValueError as exc:  # a broken record names itself
                raise ValueError(f"{path}: {item.get('id', '?')}: {exc}") from exc
    return tuple(out)


def _absent(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _number(value) -> float | None:
    try:
        return float(str(value).split()[0])
    except (ValueError, IndexError):
        return None


def _equal(value, wanted) -> bool:
    if isinstance(wanted, bool):
        word = str(value).strip().lower()
        return word in ({"yes", "true", "on"} if wanted else {"no", "false", "off"})
    a, b = _number(value), _number(wanted)
    if (a is not None and b is not None and len(str(value).split()) == 1
            and len(str(wanted).split()) == 1):
        return a == b
    return str(value).strip().lower() == str(wanted).strip().lower()


def _meets(value, spec, subject: dict) -> bool:
    if spec is None:
        return _absent(value)
    if isinstance(spec, list):
        return (_absent(value) and None in spec) or (
            not _absent(value) and any(_equal(value, s) for s in spec if s is not None))
    if isinstance(spec, dict):
        for op, arg in spec.items():
            n = _number(value)
            if op in ("gt", "ge", "lt", "le"):
                if n is None or not {"gt": n > arg, "ge": n >= arg, "lt": n < arg,
                                     "le": n <= arg}[op]:
                    return False
            elif op == "not":
                options = arg if isinstance(arg, list) else [arg]
                if not _absent(value) and any(_equal(value, o) for o in options
                                              if o is not None):
                    return False
                if _absent(value) and None in options:
                    return False
            elif op == "differs_from":
                other = subject.get(arg)
                if _absent(value) or _absent(other) or _equal(value, other):
                    return False
        return True
    return not _absent(value) and _equal(value, spec)


class _Filled(dict):
    def __missing__(self, key):
        return "unset"


def _finding(record: SensorRecord, subject: dict) -> Finding:
    filled = _Filled({k: v for k, v in subject.items() if not _absent(v)})
    return (Precondition if record.stops_the_call else Finding)(
        id=record.id,
        message=record.sees.format_map(filled),
        means=record.means.format_map(filled) or None,
        suggestion=record.usual_move.format_map(filled) or None,
        assumption=record.assumption.format_map(filled) or None,
        sources=[s.locator or s.text for s in record.sources])


def evaluate(subject_kind: str, subject: dict, *, version: str = "",
             keys: list[dict] | None = None,
             records: tuple[SensorRecord, ...] | None = None) -> list[Finding]:
    """The findings the records for this kind of subject make of it: a whole-file record
    once over ``subject``, an ``each: key`` record once over each of ``keys`` (one
    subject per key of the file, as its predicate described it)."""
    out: list[Finding] = []
    for record in records if records is not None else load_sensors():
        if record.subject != subject_kind:
            continue
        if record.versions and version and not any(fnmatch(version, v)
                                                   for v in record.versions):
            continue
        for one in (keys or []) if record.each == "key" else [subject]:
            if all(_meets(one.get(k), spec, one) for k, spec in record.when.items()):
                out.append(_finding(record, one))
    return out
