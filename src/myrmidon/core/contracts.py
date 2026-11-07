"""Tool contracts: what a tool reads and writes, by artifact kind, and under what
condition.

A kind is the role the ledger records on an artifact (``ArtifactRef.role``): ``structure``,
``topology``, ``mdp``, ``tpr``, ``trajectory``, ``energy``, ``checkpoint``, ``log``,
``index``, ``data``, and those an engine module declares. A
property is a fact about an artifact that decides what a tool downstream gets: an mdp's
``frames`` (coordinates are written) and ``energies``; a tpr inherits its mdp's; a run
writes a trajectory only from a tpr whose ``frames`` holds. A property is read from a file
on disk (``define_kind``: the kind's parser and properties, declared by the engine module
that knows the format).

``check_outputs`` holds a tool that ran to what it declared, against the artifacts it
recorded; the consumed and produced kinds are also the ledger's edges from a run's files to
its inputs. A tool with no contract is not judged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Props = dict[str, bool]

#: roles some tools record under a name of their own, read as the kind they are; an engine
#: module declares its own
KIND_ALIASES: dict[str, str] = {}


def declare_alias(alias: str, kind: str) -> None:
    """Read artifacts recorded under ``alias`` as the kind they are."""
    KIND_ALIASES[alias] = kind


def canonical(kind: str) -> str:
    return KIND_ALIASES.get(kind, kind)


@dataclass(frozen=True)
class Property:
    """A fact about an artifact of one kind, decided over the file's parsed parameters."""

    name: str
    holds: Callable[[dict], bool]


@dataclass(frozen=True)
class Kind:
    parser: Callable[[Path], dict]  # the file -> its parameters
    properties: tuple[Property, ...]
    normalize: Callable[[dict], dict] = lambda d: d  # one key spelling for file and values
    # (parsed file, a key in the engine's own spelling) -> its value as written, None when
    # the file does not set it. A kind with a lookup is a parameter file: a researcher's
    # decision is checked against it, and descent never follows it to its parents
    lookup: Callable[[dict, str], str | None] | None = None
    # (values) -> (the run's simulated length in ps, its length in steps): what a
    # run-parameter file says about how long its run goes, None where it says nothing
    length: Callable[[dict], tuple[float | None, float | None]] | None = None


KINDS: dict[str, Kind] = {}
#: kinds that carry a system's identity: a job's inputs of these kinds descend from the
#: approved mandate's system files (core/lineage.py). An engine module declares its own.
IDENTITY: set[str] = set()
#: tools whose outputs, per stage, stand in for the system's files of the kinds they write (a
#: selection that rewrites a system's input for the region it picks). An engine module
#: declares its own.
RESELECTS: set[str] = set()


def define_kind(kind: str, parser: Callable[[Path], dict], *properties: Property,
                normalize: Callable[[dict], dict] | None = None,
                lookup: Callable[[dict, str], str | None] | None = None,
                length: Callable[[dict], tuple[float | None, float | None]] | None = None
                ) -> None:
    """Register a kind's parser and properties; an engine module calls this once."""
    KINDS[kind] = Kind(parser, properties, normalize or (lambda d: d), lookup, length)


#: kind -> the files a file of that kind includes, which its engine reads with it
INCLUDES: dict[str, Callable[[Path], list[Path]]] = {}


def declare_includes(kind: str, reader: Callable[[Path], list[Path]]) -> None:
    """Say how a file of ``kind`` names the files it includes (a topology's ``#include``s):
    a call that reads it reads them too, and the registry hashes them as its inputs."""
    INCLUDES[canonical(kind)] = reader


def includes_of(kind: str, path: Path) -> list[Path]:
    """The files a file of ``kind`` includes; none for a kind that includes nothing or a
    file that cannot be read."""
    reader = INCLUDES.get(canonical(kind))
    if reader is None:
        return []
    try:
        return reader(path)
    except (OSError, ValueError):
        return []


def declare_identity(*kinds: str, replaced_by: str | None = None) -> None:
    """Mark kinds as carrying a system's identity; an engine module calls this once.
    ``replaced_by`` names a tool whose outputs replace, for the stage it was called for, the
    system's files of the kinds it writes (``core.lineage.system_file``)."""
    IDENTITY.update(canonical(k) for k in kinds)
    if replaced_by:
        RESELECTS.add(replaced_by)


def parameter_kinds() -> set[str]:
    """The kinds a decision is checked against: those with a lookup."""
    return {name for name, k in KINDS.items() if k.lookup is not None}


def lookup(kind: str, path: Path, key: str) -> str | None:
    """The value a parameter file on disk sets for ``key``; None when it sets none. Raises
    OSError or ValueError when the file cannot be read."""
    k = KINDS[canonical(kind)]
    return k.lookup(k.parser(path), key) if k.lookup is not None else None


def run_length(kind: str, values: dict) -> tuple[float | None, float | None]:
    """(ps, steps) a run-parameter file of this kind with these values runs for; (None,
    None) for a kind that says nothing about it."""
    k = KINDS.get(canonical(kind))
    if k is None or k.length is None:
        return None, None
    return k.length(k.normalize(values))


def props_of_values(kind: str, values: dict) -> Props | None:
    """The properties a file of this kind rendered from these values would hold; None for
    a kind with no properties."""
    k = KINDS.get(canonical(kind))
    if k is None:
        return None
    values = k.normalize(values)
    return {p.name: bool(p.holds(values)) for p in k.properties}


@dataclass(frozen=True)
class Produces:
    """One artifact a tool writes.

    ``kind`` is fixed, or ``kind_of`` decides it from the parameters (a rendered input
    set's). ``when`` names a property of a consumed artifact (``"tpr.frames"``) without
    which the file is not written. ``collected`` marks an output that appears when the job
    is collected, not in the submitting call's observation."""

    kind: str | None = None
    kind_of: Callable[[Any, dict], str | None] | None = None
    when: str | None = None
    collected: bool = False

    def params_named(self) -> set[str]:
        return {self.when.split(".", 1)[0]} if self.when else set()


@dataclass(frozen=True)
class Contract:
    """What a tool consumes (parameter -> the kind, or kinds, it accepts) and produces."""

    consumes: Mapping[str, str | tuple[str, ...]] = field(default_factory=dict)
    produces: tuple[Produces, ...] = ()
    #: the parameter naming the file this tool's outputs take their scientific content from
    upstream: str | None = None
    #: what it reads by itself, from the study, under a role no parameter names (role -> kind)
    reads: Mapping[str, str] = field(default_factory=dict)
    #: a parameter mapping artifact kinds to files: each entry is read as the kind it is under
    by_kind: str | None = None

    def accepts(self, param: str) -> tuple[str, ...]:
        kinds = self.consumes[param]
        return (kinds,) if isinstance(kinds, str) else tuple(kinds)

    def params_named(self) -> set[str]:
        out = set(self.consumes) | ({self.by_kind} if self.by_kind else set())
        for p in self.produces:
            out |= p.params_named()
        return out

    def yields(self, p: Produces, ctx, params: dict, inputs: dict[str, Props | None]
               ) -> tuple[str | None, bool | None]:
        """(kind, whether it is written — None when that cannot be known yet) for one
        output under these inputs."""
        kind = p.kind or (p.kind_of(ctx, params) if p.kind_of else None)
        if kind is None:
            return None, None
        written: bool | None = True
        if p.when:
            param, prop = p.when.split(".", 1)
            props = inputs.get(param)
            written = None if props is None or prop not in props else props[prop]
        return canonical(kind), written


def props_of_file(kind: str, path: Path) -> Props | None:
    """The properties this file holds; None when the kind has no parser or the file cannot
    be read."""
    k = KINDS.get(canonical(kind))
    if k is None:
        return None
    try:
        return props_of_values(kind, k.parser(path))
    except (OSError, ValueError):
        return None


def _file_props(ws, kind: str, value: str) -> Props | None:
    if ws is None:
        return None
    try:
        path = ws.resolve(value)
        return props_of_file(kind, path) if path.is_file() else None
    except (OSError, ValueError):
        return None


def check_outputs(contract: Contract, ctx, params: dict, artifacts) -> list[str]:
    """The kinds a tool that just ran declared it would write under its actual inputs but
    did not record: each as "no trajectory recorded, though the tpr writes frames"."""
    ws = getattr(ctx, "workspace", None)
    inputs: dict[str, Props | None] = {}
    for param in contract.consumes:
        value = params.get(param)
        if isinstance(value, str) and value:
            inputs[param] = _file_props(ws, contract.accepts(param)[0], value)
    recorded = {canonical(a.role) for a in artifacts}
    missing = []
    for p in contract.produces:
        if p.collected:
            continue
        kind, written = contract.yields(p, ctx, params, inputs)
        if kind and written and kind not in recorded:
            missing.append(f"no {kind} recorded" + (
                f", though {p.when.replace('.', ' has ')}" if p.when else ""))
    return missing
