"""InputSetStore: curated, versioned, source-bearing parameter recipes.

Tier 2 of the knowledge system. Protocols are YAML files. They are the ONLY
sanctioned source of "default" parameters: an agent that did not get a parameter
from the user should get it from an inputset and cite it, or ask.

Input sets are human-merged: the harness never edits this store itself.

Three layers. A recipe is true for
anyone running that engine (``engine``: the pack shipped in the package), for this
machine only (``site``: ``~/.config/myrmidon/inputsets/``, next to the machine
config), or for this project (``project``: ``./inputsets/``). Later layers override
earlier ones by inputset id.

Schema v2. A v1 ``citations: [str]`` list loads and is migrated to typed
``sources``; a source that resolves to nothing is refused, because a citation nobody
can follow is not a source. Every parameter may carry its own rationale and point at
the source that justifies it; ``required_from_user`` marks the values no inputset may
supply (charge, multiplicity, the QM region — chemistry is never a default).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from myrmidon.core.invariants import (  # re-exported: inputsets are where invariants are declared
    Invariant,
    InvariantResult,
    evaluate_invariants,
)
from myrmidon.knowledge.layered import LayeredStore, Scope

__all__ = [
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


SourceKind = Literal[
    "doi",  # a published method — the strongest
    "url",  # a resolvable page
    "tutorial",  # a project-published worked example
    "personal_communication",  # a colleague's working setup
    "in_house",  # validated here: ledger events are the evidence
    "engine_manual",  # the engine's own documented default
]

_DOI = re.compile(r"\b(?:doi:\s*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/\S+)", re.I)
_URL = re.compile(r"https?://\S+")


class Source(BaseModel):
    """Where a parameter's authority comes from.

    ``kind`` matters because the three bodies of material an input set is distilled
    from carry three different kinds of authority, and flattening them to one
    citation line loses exactly what a later reader needs in order to decide how
    much to trust a number.
    """

    kind: SourceKind = "doi"
    text: str  # human-readable citation line
    locator: str | None = None  # doi, url, or a path on this filesystem
    sha256: str | None = None  # the file this was distilled from
    evidence: list[str] = Field(default_factory=list)  # ledger events / workspace ids

    @model_validator(mode="after")
    def _must_resolve(self) -> Source:
        if not (self.locator or self.sha256 or self.evidence):
            raise ValueError(
                f"source {self.text[:60]!r} has no locator, sha256 or evidence — "
                "a citation nobody can follow is not a source"
            )
        return self

    @classmethod
    def from_citation(cls, citation: str) -> Source:
        """Migrate a v1 citation string. Refuses one that resolves to nothing."""
        doi = _DOI.search(citation)
        if doi:
            return cls(kind="doi", text=citation, locator=f"doi:{doi.group(1)}")
        url = _URL.search(citation)
        if url:
            return cls(kind="url", text=citation, locator=url.group(0))
        raise ValueError(
            f"citation {citation[:60]!r} carries no DOI or URL — give it a locator, "
            "or declare it as a typed source (tutorial / personal_communication / in_house)"
        )


class ValueSpec(BaseModel):
    """One parameter and why it has the value it has.

    A bare scalar in YAML is shorthand for ``{value: <scalar>}``.
    """

    value: Any = None
    required_from_user: bool = False  # chemistry: propose, never supply
    derivable: str | None = None  # how it *could* be derived, for confirmation
    rationale: str | None = None
    source: int | None = Field(default=None, ge=0)  # index into InputSet.sources

    @model_validator(mode="before")
    @classmethod
    def _accept_scalar(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return data
        return {"value": data}

    @model_validator(mode="after")
    def _says_something(self) -> ValueSpec:
        if self.value is None and not self.required_from_user and not self.derivable:
            raise ValueError(
                "a parameter must have a value, be required_from_user, or be derivable"
            )
        return self


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

    # where this file was actually found, which is not always what it claims
    layer: Scope | None = Field(default=None, exclude=True)

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
                    "steps, which is procedure — declare it on the plan, not the input set"
                )
        return self

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
    ``site_path`` the machine's, ``builtin_path`` the pack shipped in the package."""

    model = InputSet
    kind = "inputset"
    dirname = "inputsets"

    def primary_text(self, item: InputSet) -> str:
        return f"{item.id} {item.title} {item.renders or ''}"


class Resolution(BaseModel):
    """What an input set says, plus where each value came from.

    The point of this object is ``origin``: after the fact, every key is one of
    ``inputset:<id>@v<N>`` or ``user``, and the ledger says which. A parameter with no
    entry here is one the harness never wrote.
    """

    inputset: str
    version: int
    renders: str | None = None
    declared_axis: str | None = None  # the plan's Arms.varies, if this study declared one
    values: dict[str, Any] = Field(default_factory=dict)
    origin: dict[str, str] = Field(default_factory=dict)
    overrides: dict[str, dict] = Field(default_factory=dict)  # key -> {inputset, user}
    on_axis: list[str] = Field(default_factory=list)  # overrides the plan declared as its axis
    additions: list[str] = Field(default_factory=list)  # keys the inputset never mentions
    missing: dict[str, str] = Field(default_factory=dict)  # required_from_user, unsupplied
    sources: list[Source] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing


def resolve(
    inputset: InputSet,
    overrides: dict[str, Any] | None = None,
    declared_axis: str | None = None,
) -> Resolution:
    """Merge an input set stage with the caller's overrides, keeping the origin of
    every key. Overrides are allowed and loud: what is prevented is silent drift,
    not deviation.

    ``declared_axis`` is the parameter the study's plan says it varies (``Arms.varies``).
    An override of that key is the experiment the researcher came to run and raises no
    deviation: the plan already said so, and said why. Every other override is a
    deviation to be acknowledged — which is the right way round, because it is the thing
    they did not say they were doing. Nothing here is a permission the input set's author
    granted; an author who has never seen the study cannot know which parameter is the
    axis."""
    overrides = dict(overrides or {})
    declared = inputset.parameters
    tag = f"inputset:{inputset.id}@v{inputset.version}"

    resolution = Resolution(
        inputset=inputset.id,
        version=inputset.version,
        renders=inputset.renders,
        declared_axis=declared_axis,
        sources=list(inputset.sources),
    )
    for key, spec in declared.items():
        if key in overrides:
            if spec.value is not None and overrides[key] != spec.value:
                resolution.overrides[key] = {"inputset": spec.value, "user": overrides[key]}
                if declared_axis is not None and key == declared_axis:
                    resolution.on_axis.append(key)
            resolution.values[key] = overrides.pop(key)
            resolution.origin[key] = "user"
        elif spec.required_from_user:
            # chemistry: the input set names it and refuses to answer it
            resolution.missing[key] = spec.derivable or "ask the user"
        elif spec.value is None:
            # a fact about the system in hand: the input set says how to obtain it
            resolution.missing[key] = f"derive for this system: {spec.derivable}"
        elif spec.value is not None:
            resolution.values[key] = spec.value
            resolution.origin[key] = tag
    for key, value in overrides.items():  # keys the input set never mentions
        resolution.values[key] = value
        resolution.origin[key] = "user"
        resolution.additions.append(key)
    return resolution
