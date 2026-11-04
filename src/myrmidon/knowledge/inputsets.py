"""InputSetStore: versioned, source-bearing parameter recipes for runs outside a mandate's
stages, and the resolver every engine input is compiled through.

A mandate stage's values come from its stage template and the methods (knowledge/stage.py),
which compile through this module's ``Resolution``; an input set is a recipe a machine or a
project keeps for a run of its own (a force-matching input, say), resolved the same way.
Input sets are human-merged: the harness never edits this store itself.

Three layers. A recipe is true for anyone running that engine (``engine``: a pack shipped in
the package, none today), for this machine only (``site``:
``~/.config/myrmidon/inputsets/``, next to the machine config), or for this project
(``project``: ``./inputsets/``). Later layers override earlier ones by inputset id.

Schema v2. A v1 ``citations: [str]`` list loads and is migrated to typed
``sources``; a source that resolves to nothing is refused, because a citation nobody
can follow is not a source. Every parameter may carry its own rationale and point at
the source that justifies it; ``required_from_user`` marks the values no inputset may
supply (charge, multiplicity, the QM region — chemistry is never a default).

An engine input is resolved from two input sets and the system: a **purpose** set says
what the run is for (an NPT equilibration, an annealing stage), a **method** set how its
interactions are computed (a force-field family, a QM method), and the keys a purpose
reads ``from_system`` are facts a tool reads from the system's own files. Each set states
keys the engine reads; a value that decides a key another owner usually decides (a coupled
run turning the force field's constraints off) says so with ``overrides:``, a rationale and
a source. So the same NPT purpose serves every system, and a new molecule needs no new
input set.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from myrmidon.core.invariants import (  # re-exported: inputsets are where invariants are declared
    Invariant,
    InvariantResult,
    evaluate_invariants,
)
from myrmidon.knowledge.catalog import Catalog, EngineKey, SystemFact, canonical, load
from myrmidon.knowledge.layered import LayeredStore, Scope
from myrmidon.knowledge.sources import Source

#: whose decision a value is: what the run is for, how interactions are computed, a fact
#: of the system's files, or the science the researcher states
Owner = Literal["purpose", "method", "system", "researcher"]
_OWNER_OF_ORIGIN: dict[str, Owner] = {
    "template": "purpose", "stage": "researcher", "method": "method", "mandate": "method",
    "system": "system", "user": "researcher"}


def owner_of(origin: str, axes: dict[str, str] | None = None) -> Owner:
    """Whose decision a value is, by where it came from: an input set's by its axis, a
    stage template's the purpose's, a mandate stage's terms and the caller's the
    researcher's, a method's the method's, a fact the system's."""
    kind, _, ref = origin.partition(":")
    if kind == "inputset":
        return (axes or {}).get(ref, "purpose")  # type: ignore[return-value]
    return _OWNER_OF_ORIGIN.get(kind, "purpose")


__all__ = [
    "FACTS",
    "Invariant",
    "InvariantResult",
    "ValueSpec",
    "InputSet",
    "InputSetStore",
    "Resolution",
    "Scope",
    "Source",
    "evaluate_invariants",
    "resolve",
]


#: what an input set's ``renders`` writes: the artifact kind and the file's suffix
RENDERED_FILES = {"mdp": ("mdp", ".mdp"), "cpmd_inp": ("cpmd_input", ".inp"),
                  "fminput": ("fm_input", ".dat")}

class ValueSpec(BaseModel):
    """One parameter and why it has the value it has.

    A bare scalar in YAML is shorthand for ``{value: <scalar>}``.
    """

    value: Any = None
    required_from_user: bool = False  # chemistry: propose, never supply
    derivable: str | None = None  # how it *could* be derived, for confirmation
    rationale: str | None = None
    source: int | None = Field(default=None, ge=0)  # index into InputSet.sources
    # the owner of a key this set decides anyway (a coupled run turning the force field's
    # constraints off): it needs a rationale and a source
    overrides: Owner | None = None
    from_system: str | None = None  # the fact (FACTS) that reads this key from the system
    same_as: str | None = None  # another key whose resolved value this one takes

    @model_validator(mode="before")
    @classmethod
    def _accept_scalar(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return data
        return {"value": data}

    @model_validator(mode="after")
    def _says_something(self) -> ValueSpec:
        if (self.value is None and not self.required_from_user and not self.derivable
                and not self.from_system and not self.same_as):
            raise ValueError(
                "a parameter must have a value, be required_from_user, be derivable, be read "
                "from_system, or be the same_as another"
            )
        if self.from_system is not None and self.from_system not in FACTS:
            raise ValueError(f"from_system names {self.from_system!r}; the facts a tool reads "
                             f"are {', '.join(FACTS)}")
        if self.overrides is not None and (self.rationale is None or self.source is None):
            raise ValueError("a value that overrides another owner's key needs a rationale "
                             "and a source")
        return self


#: the facts a purpose may read from the system (engines/gromacs/system.py)
FACTS = ("coupling_groups", "restraint_defines", "qm_group", "starting_state")
#: keys with one value per temperature-coupling group
PER_COUPLING_GROUP = ("tau_t", "ref_t")
#: the engine that reads the file an input set renders
ENGINE_OF_RENDERS = {"mdp": "gromacs", "cpmd_inp": "cpmd"}


class InputSet(BaseModel):
    id: str
    title: str
    applicability: str  # when this input set is appropriate
    # the class of system its values hold for ("a small closed-shell solute in explicit
    # water"), so a value tuned on one molecule is not carried to a protein unseen
    system_class: str = ""
    aliases: list[str] = Field(default_factory=list)  # ids it was published under before
    scope: Scope = "project"  # what it is claimed to be true for
    parameters: dict[str, ValueSpec] = Field(default_factory=dict)
    # "none": this input set states values and renders no file. An omitted `renders`
    # is not the same thing — that one is rendered by the output file's suffix.
    renders: Literal["mdp", "cpmd_inp", "fminput", "ndx", "sbatch", "none"] | None = None
    sources: list[Source] = Field(min_length=1)  # without sources it is opinion
    invariants: list[Invariant] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)  # finding ids that bite this
    validated_by: list[str] = Field(default_factory=list)  # tests / workspaces
    notes: str = ""
    version: int = 1
    # a purpose set (what the run is for) or a method set (how interactions are computed);
    # neither: a set whose values are its own, rendered alone
    axis: Literal["purpose", "method"] | None = None
    engine: Literal["gromacs", "cpmd"] | None = None
    # a force-field method set: the force-field directories it is for, by prefix
    forcefields: list[str] = Field(default_factory=list)

    # where this file was actually found, which is not always what it claims
    layer: Scope | None = Field(default=None, exclude=True)
    # the sha256 of the file it was read from: a rendered file names the bytes it came from
    sha256: str = Field(default="", exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _migrate_v1(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "sources" in data:
            return data
        citations = data.get("citations")
        if citations:
            data = {k: v for k, v in data.items() if k != "citations"}
            data["sources"] = [Source.from_citation(c).model_dump() for c in citations]
        return data

    @model_validator(mode="after")
    def _sources_resolve(self) -> InputSet:
        n = len(self.sources)
        for key, spec in self.parameters.items():
            if spec.source is not None and spec.source >= n:
                raise ValueError(
                    f"{self.id}: parameters.{key} points at source {spec.source}, "
                    f"but only {n} source(s) are declared"
                )
        for invariant in self.invariants:
            if invariant.crosses_scopes:
                raise ValueError(
                    f"{self.id}: invariant {invariant.id} relates values from different "
                    "steps, which is procedure — declare it on the protocol, not the input set"
                )
        if self.axis is not None:
            self.engine = self.engine or ENGINE_OF_RENDERS.get(self.renders or "")
            if self.engine is None:
                raise ValueError(f"{self.id}: a {self.axis} set names its engine")
            self._check_keys(load(self.engine))
        return self

    def _check_keys(self, catalog: Catalog) -> None:
        """Every key is one the engine reads."""
        for key in self.parameters:
            if engine_key(catalog, key) is None:
                raise ValueError(f"{self.id}: {key!r} is not a key {catalog.engine} "
                                 f"{catalog.version} reads")

    @property
    def citations(self) -> list[str]:
        """The v1 view: the source lines as text."""
        return [s.text for s in self.sources]

    def values(self) -> dict[str, Any]:
        """The plain parameter mapping, for callers that only want the numbers.

        Parameters the input set refuses to supply are left out: asking for them is
        the point at which the caller has to go and confirm one.
        """
        return {
            key: spec.value
            for key, spec in self.parameters.items()
            if spec.value is not None and not spec.required_from_user
        }

    def unsupplied(self) -> dict[str, ValueSpec]:
        """Parameters this input set deliberately does not answer."""
        return {k: v for k, v in self.parameters.items() if v.required_from_user}


class InputSetStore(LayeredStore[InputSet]):
    """Input sets, layered (knowledge/layered.py): ``path`` is the project directory,
    ``site_path`` the machine's, ``builtin_path`` a pack shipped in the package (empty: a
    stage's values are its template's)."""

    model = InputSet
    kind = "inputset"
    dirname = "inputsets"

    def primary_text(self, item: InputSet) -> str:
        return f"{item.id} {item.title} {item.renders or ''}"


def engine_key(catalog: Catalog, key: str) -> EngineKey | None:
    """The catalog entry an input-set key sets: an mdp key however spelled, or a CPMD
    `&SECTION/KEYWORD LINE` by the keyword its line matches."""
    if catalog.engine == "cpmd":
        section, _, line = key.lstrip("&").partition("/")
        return catalog.keyword(section, line)
    return catalog.get(key)


def _identity(catalog: Catalog | None, key: str) -> str:
    """One name for a key across a purpose set, a method set and the caller: two CPMD lines
    of one keyword (TEMPERATURE, TEMPERATURE RAMP) are the same key."""
    if catalog is None:
        return key
    entry = engine_key(catalog, key)
    if entry is None:
        return canonical(key) if catalog.engine == "gromacs" else key
    return f"{entry.section}/{entry.name}" if catalog.engine == "cpmd" else canonical(key)


class Resolution(BaseModel):
    """What the input sets say, plus where each value came from.

    The point of this object is ``origin``: after the fact, every key is one of
    ``inputset:<id>@v<N>`` (the purpose or the method set that stated it),
    ``system:<rule>`` (a fact read from the system's files) or ``user``, and the ledger says
    which; ``owner`` says whose decision each key is. A key with no entry here is one the
    harness never wrote: the engine's default of ``engine_defaults``.
    """

    inputset: str
    version: int
    renders: str | None = None
    method: str | None = None  # the method set's id, for a purpose set
    method_version: int | None = None
    values: dict[str, Any] = Field(default_factory=dict)
    origin: dict[str, str] = Field(default_factory=dict)
    owner: dict[str, str] = Field(default_factory=dict)  # key -> purpose | method | …
    # key -> the owner a set overrode and why: a coupled run's constraints, say
    overridden: dict[str, dict] = Field(default_factory=dict)
    overrides: dict[str, dict] = Field(default_factory=dict)  # key -> {inputset, user}
    additions: list[str] = Field(default_factory=list)  # keys the inputset never mentions
    missing: dict[str, str] = Field(default_factory=dict)  # required_from_user, unsupplied
    read: dict[str, dict] = Field(default_factory=dict)  # key -> the files a fact read
    engine_defaults: str | None = None  # the catalog whose defaults the rest of the keys take
    sources: list[Source] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing


def resolve(
    inputset: InputSet,
    overrides: dict[str, Any] | None = None,
    method: InputSet | None = None,
    facts: dict[str, SystemFact] | None = None,
) -> Resolution:
    """Merge a purpose set with its method set, the facts read from the system and the
    caller's overrides, keeping the origin of every key. Overrides are allowed and loud:
    what is prevented is silent drift, not deviation: every override is recorded."""
    engine = inputset.engine or ENGINE_OF_RENDERS.get(inputset.renders or "")
    catalog = load(engine) if engine else None
    layers = [x for x in (method, inputset) if x is not None]
    facts = {_identity(catalog, k): v for k, v in (facts or {}).items()}

    # one spelling per key: the one the sets write it in
    spelled: dict[str, str] = {}
    for layer in layers:
        for key in layer.parameters:
            spelled.setdefault(_identity(catalog, key), key)
    given: dict[str, Any] = {}
    for key, value in (overrides or {}).items():
        given[spelled.get(_identity(catalog, key), key)] = value

    resolution = Resolution(
        inputset=inputset.id,
        version=inputset.version,
        renders=inputset.renders,
        method=method.id if method else None,
        method_version=method.version if method else None,
        engine_defaults=f"{catalog.engine}-{catalog.version}" if catalog else None,
        sources=[src for layer in layers for src in layer.sources],
    )
    written: dict[str, str] = {}  # identity -> the key the value is written under
    deferred: list[tuple[str, ValueSpec]] = []

    axes = {f"{x.id}@v{x.version}": x.axis or "purpose" for x in layers}
    overriding = {_identity(catalog, key): spec.overrides for x in layers
                  for key, spec in x.parameters.items() if spec.overrides is not None}

    def place(ident: str, key: str, value: Any, origin: str) -> None:
        old = written.pop(ident, None)
        if old is not None:
            resolution.values.pop(old, None)
            resolution.origin.pop(old, None)
        written[ident] = key
        resolution.values[key] = value
        resolution.origin[key] = origin
        if catalog is not None and engine_key(catalog, key) is not None:
            # a set's value on a key it overrides stays that owner's key, as `overridden` says
            resolution.owner[key] = (overriding[ident] if origin.startswith("inputset:")
                                     and ident in overriding else owner_of(origin, axes))

    stated: dict[str, Any] = {}  # key -> what the sets (or the system) would have said
    for layer in layers:
        tag = f"inputset:{layer.id}@v{layer.version}"
        for key, spec in layer.parameters.items():
            ident = _identity(catalog, key)
            if spec.overrides is not None:
                resolution.overridden[key] = {"owner": spec.overrides, "by": layer.id,
                                              "why": spec.rationale}
            if spec.from_system is not None and ident in facts:
                stated[key] = facts[ident].value
            elif spec.value is not None:
                stated[key] = spec.value
            if key in given:
                # the caller's, in the place the set gives the key
                resolution.missing.pop(key, None)
                place(ident, key, given[key], "user")
                continue
            if spec.same_as is not None:
                deferred.append((key, spec))
            elif spec.from_system is not None:
                fact = facts.get(ident)
                if fact is None:
                    resolution.missing[key] = (
                        f"read from the system by {spec.from_system}: "
                        + (spec.derivable or "give the tool the system's files"))
                elif fact.value:
                    place(ident, key, fact.value, fact.origin)
                    resolution.read[key] = {"rule": fact.rule, "files": fact.read,
                                            "sha256": fact.sha256}
            elif spec.required_from_user:
                # chemistry: the input set names it and refuses to answer it
                resolution.missing[key] = spec.derivable or "ask the user"
            elif spec.value is None:
                # a fact about the system in hand: the input set says how to obtain it
                resolution.missing[key] = f"derive for this system: {spec.derivable}"
            else:
                resolution.missing.pop(key, None)
                place(ident, key, spec.value, tag)

    declared = {key: spec for layer in layers for key, spec in layer.parameters.items()}
    for key in given:
        if key in declared:
            was = stated.get(key)
            if was is not None and given[key] != was and declared[key].same_as is None:
                resolution.overrides[key] = {"inputset": was, "user": given[key]}
    for key in sorted(k for k in given if k not in declared):  # keys no set mentions
        resolution.additions.append(key)
        place(_identity(catalog, key), key, given[key], "user")

    for key, spec in deferred:
        if key in given:
            continue
        target = spelled.get(_identity(catalog, spec.same_as), spec.same_as)
        if target in resolution.values:
            first = str(resolution.values[target]).split()[0]
            place(_identity(catalog, key), key, first, resolution.origin[target])
            if target in resolution.owner:  # the same value, the same decision
                resolution.owner[key] = resolution.owner[target]
        else:
            resolution.missing[key] = f"the same as {spec.same_as}, which is not set"

    _per_coupling_group(resolution)
    return resolution


def _per_coupling_group(resolution: Resolution) -> None:
    """One tau-t and one ref-t per temperature-coupling group: a single value stated for a
    run whose groups come from the system is that value for every group."""
    spelled = {canonical(k): k for k in resolution.values}
    groups = spelled.get("tc_grps")
    if groups is None:
        return
    n = len(str(resolution.values[groups]).split())
    for name in PER_COUPLING_GROUP:
        key = spelled.get(name)
        if key is None:
            continue
        tokens = str(resolution.values[key]).split()
        if len(tokens) == 1 and n > 1:
            resolution.values[key] = " ".join(tokens * n)
