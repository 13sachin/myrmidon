"""Where a value's authority comes from: a typed source someone can follow."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SourceKind = Literal[
    "doi",  # a published method — the strongest
    "url",  # a resolvable page
    "tutorial",  # a project-published worked example
    "personal_communication",  # a colleague's working setup
    "in_house",  # validated here: ledger events are the evidence
    "engine_manual",  # the engine's own documented default
    "engine_source",  # the engine's source code, read where the manual is silent
    "engine_output",  # what the engine itself wrote, recorded as a fixture
]

_DOI = re.compile(r"\b(?:doi:\s*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/\S+)", re.I)
_URL = re.compile(r"https?://\S+")


class Source(BaseModel):
    """Where a parameter's authority comes from.

    ``kind`` matters because the bodies of material a value is distilled from carry
    different kinds of authority, and flattening them to one citation line loses exactly
    what a later reader needs in order to decide how much to trust a number.
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
        """A citation string as a source. Refuses one that resolves to nothing."""
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
