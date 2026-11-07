"""Findings, and the gate's decision over them.

A finding is one of three kinds, by its class:

- ``Finding``: information — what an engine's check noticed, what the harness assumed, what
  it did, what the model decided on its own knowledge. Recorded on the observation and in the
  ledger, shown in the report and the approval view; the call runs.
- ``Precondition``: the call cannot run as made — a missing input, a value it cannot read,
  the rare physics a silent engine would get wrong. It says what is missing or wrong and
  ``needs`` names the parameters any one of which resolves it; it is the call's error.
- ``Theirs``: a person's decision — the spend outside an approved mandate, an act that is
  theirs. A model's call is refused; a person's at the terminal (``ToolContext.person``)
  accepts it. A caller never acknowledges one.

``severity`` (``info``, ``block``, ``warn``) names the kind in a record a reader matches on.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

Severity = Literal["block", "warn", "info"]


class Finding(BaseModel):
    # a field it does not define — a `severity` among them — is an error, not dropped
    model_config = ConfigDict(extra="forbid")
    #: the kind, by class; `severity` names it in a record
    kind: ClassVar[Severity] = "info"

    id: str  # stable, SCREAMING_SNAKE, e.g. "QM_CELL_NO_VACUUM"
    message: str  # what it sees, with the values
    means: str | None = None  # what that means for the physics or the run
    assumption: str | None = None  # what would be silently assumed, spelled out
    suggestion: str | None = None  # the usual move
    sources: list[str] = Field(default_factory=list)  # where the knowledge comes from
    # parameters any one of which, given, resolves the finding
    needs: list[str] | None = None

    @computed_field
    @property
    def severity(self) -> Severity:
        return self.kind

    @classmethod
    def read(cls, data):
        """A finding read back from a record, as the kind its ``severity`` names."""
        if isinstance(data, Finding):
            return data
        data = dict(data)
        kinds = {"block": Precondition, "warn": Theirs, "info": Finding, "": Finding}
        return kinds[data.pop("severity", "")].model_validate(data)


class Precondition(Finding):
    """The call cannot run as made."""

    kind: ClassVar[Severity] = "block"


class Theirs(Finding):
    """A person's to accept."""

    kind: ClassVar[Severity] = "warn"


class Check(BaseModel):
    """One finding a check can raise, as knowledge a reviewer audits: its id, where it was
    learned (a fixture of what an engine did, or the manual that says so), the fixture's
    SHA-256 when its first source is one, and whether the call stops on it."""

    model_config = ConfigDict(frozen=True)

    id: str
    sources: tuple[str, ...]
    sha256: str = ""
    stops: bool = False

    def finding(self, message: str, *, means: str = "", suggestion: str = "",
                assumption: str = "") -> Finding:
        return (Precondition if self.stops else Finding)(
            id=self.id, message=message, means=means or None, suggestion=suggestion or None,
            assumption=assumption or None, sources=list(self.sources))


class GateDecision(BaseModel):
    """Outcome of running a gate against the study's approvals."""

    allowed: bool
    blocking: list[Finding]  # block-severity: the call is wrong as made
    unapproved: list[Finding]  # warn-severity with no approval and no person
    approved: list[str]  # warn-severity ids an approval or the person accepted
    advisory: list[Finding]  # info-severity, never refuse


def decide(warnings: list[Finding], person: bool = False) -> GateDecision:
    blocking = [w for w in warnings if w.severity == "block"]
    warns = [w for w in warnings if w.severity == "warn"]
    advisory = [w for w in warnings if w.severity == "info"]
    accepted = {w.id for w in warns} if person else set()
    unapproved = [w for w in warns if w.id not in accepted]
    return GateDecision(
        allowed=not blocking and not unapproved,
        blocking=blocking,
        unapproved=unapproved,
        approved=sorted({w.id for w in warns if w.id in accepted}),
        advisory=advisory,
    )
