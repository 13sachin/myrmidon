"""ProtocolStore: published pipelines — the sequences a field has agreed are worth running.

Beside input sets in the knowledge system. An **input set** says what a
value should be and who says so; a **protocol** says what sequence of steps a published
pipeline runs, and which inputset supplies each step's values. Both are knowledge: true
before any study exists, citable, shared. A **plan** (`core/plan.py`) is one study's
instance — its system, its costs, its deviations, its approval.

**A protocol is a starting point, never a rule.** "Equilibrate classically, then run QM/MM"
is one published pipeline; it is not a law, and a researcher building a new sequence is
doing the work rather than making a mistake. Nothing here gates a plan that departs from a
protocol: `plan_write` records which protocol a plan started from and says, at info level,
what it changed. Adapting a published pipeline costs no acknowledgement, because adapting
it is the point.

The three layers are the InputSetStore's: the pack shipped in the package, the machine's
`~/.config/myrmidon/protocols/`, and `protocols/` in the project directory; the nearest
definition of an id wins.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from myrmidon.core.invariants import Invariant, InvariantResult, evaluate_invariants
from myrmidon.core.plan import unsatisfied_roles
from myrmidon.knowledge.inputsets import Source
from myrmidon.knowledge.layered import LayeredStore, Scope


class ProtocolStep(BaseModel):
    """One step of a published pipeline.

    It carries no values: those belong to the input set it `cites`. What it may carry is
    its wiring — which of the roles it needs feeds which of its tool's parameters
    (``method: cpmd_method_heating``) — because that is part of the published sequence, not
    a choice. Its estimate is a default for a machine like the one the protocol was measured
    on — a plan's own estimate is what that study will be approved against.
    """

    name: str
    tool: str  # the registered tool this step calls
    cites: str | None = None  # "inputset:<id>@v<N>" — where this step's values come from
    needs: list[str] = Field(default_factory=list)  # artifact roles consumed
    produces: list[str] = Field(default_factory=list)  # artifact roles produced
    values: dict[str, str] = Field(default_factory=dict)  # parameter -> a role it needs
    estimate_core_h: float | None = None
    estimate_basis: str | None = None
    note: str = ""


class Protocol(BaseModel):
    """A published sequence, with the source that published it."""

    id: str
    title: str
    applicability: str  # when this pipeline is the right one to start from
    system_class: str = ""  # the class of system its sequence and estimates hold for
    aliases: list[str] = Field(default_factory=list)  # ids it was published under before
    scope: Scope = "project"
    given: list[str] = Field(default_factory=list)  # roles the researcher brings
    steps: list[ProtocolStep] = Field(min_length=1)
    invariants: list[Invariant] = Field(default_factory=list)  # rules across its steps
    sources: list[Source] = Field(min_length=1)  # without a source it is somebody's habit
    pitfalls: list[str] = Field(default_factory=list)  # finding ids that bite this
    validated_by: list[str] = Field(default_factory=list)
    notes: str = ""
    version: int = 1

    # where this file was actually found, which is not always what it claims
    layer: Scope | None = Field(default=None, exclude=True)

    @property
    def shape(self) -> Literal["one-shot", "chain"]:
        """A protocol has no arms: which experiment to run is the study's, not the
        pipeline's. It is a sequence, and a plan makes it a comparison or a sweep."""
        return "one-shot"

    def unsatisfied(self) -> dict[str, list[str]]:
        return unsatisfied_roles(self.given, self.steps)

    def check_invariants(self, rendered: dict[str, dict]) -> list[InvariantResult]:
        """The rules relating values two of its steps render, against what is rendered.

        A protocol holds them for the same reason a plan does: they are constraints
        between steps, which no single inputset can see. A plan that starts from this
        protocol inherits them.
        """
        return evaluate_invariants(self.invariants, rendered)

    @model_validator(mode="after")
    def _is_runnable_in_order(self) -> Protocol:
        names = [s.name for s in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.id}: duplicate step names {names}")
        missing = self.unsatisfied()
        if missing:
            detail = "; ".join(f"{k} needs {', '.join(v)}" for k, v in missing.items())
            raise ValueError(
                f"{self.id}: published as a pipeline but cannot run in its own order — "
                f"{detail}. Add the step that produces it, or list the role in `given`."
            )
        for step in self.steps:
            loose = sorted(f"{k}: {v}" for k, v in step.values.items() if v not in step.needs)
            if loose:
                raise ValueError(
                    f"{self.id}: step {step.name} states values that are not roles it needs "
                    f"({', '.join(loose)}); a protocol wires roles to parameters, and its "
                    "values belong to the input sets it cites"
                )
        for invariant in self.invariants:
            unknown = sorted(invariant.scopes - {""} - set(names))
            if unknown:
                raise ValueError(
                    f"{self.id}: invariant {invariant.id} names step(s) not in this "
                    f"protocol: {unknown}"
                )
        return self


class ProtocolStore(LayeredStore[Protocol]):
    """Published pipelines, layered like the input sets (knowledge/layered.py). An empty
    query lists them all; a step's tool counts as what the protocol is."""

    model = Protocol
    kind = "protocol"
    dirname = "protocols"
    empty_query_lists_all = True

    def primary_text(self, item: Protocol) -> str:
        return f"{item.id} {item.title} {' '.join(s.tool for s in item.steps)}"
