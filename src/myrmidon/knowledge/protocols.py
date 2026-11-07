"""ProtocolStore: published pipelines — the sequences of stages a field has agreed are worth
running.

Beside the stage templates and methods in the knowledge system. A **template** says how an
engine is told one kind of stage, and a **method** what a Hamiltonian's constants are and who
says so; a **protocol** is what a mandate is written from: its stages in physical terms, the
deliverables they are measured for, the Hamiltonian they are computed with, the rules across
its stages no single template can see, and the class of system it holds for. It names no
tool and no core-hour: how a stage is run is the harness's, and what it costs is this
machine's. A **mandate** (`core/mandate.py`) is one study's instance — its system, its
decisions, its ceiling, its approval — and names the protocol it starts from, if one.

**A protocol is a starting point, never a rule.** "Equilibrate classically, then run QM/MM"
is one published pipeline; it is not a law, and a researcher building a new sequence is
doing the work rather than making a mistake. Nothing here gates a study that departs from a
protocol: `protocol_preview` prices it on the built system, and adapting it raises no
finding, because adapting it is the point. Building the system it starts from (a box, its
solvent, a ligand's parameters) is what the skills guide; the mandate hashes what that made.

The three layers are the InputSetStore's: the pack shipped in the package, the machine's
`~/.config/myrmidon/protocols/`, and `protocols/` in the project directory; the nearest
definition of an id wins.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from myrmidon.core.invariants import Invariant, InvariantResult, evaluate_invariants
from myrmidon.core.mandate import Deliverable, Hamiltonian, Method, Stage
from myrmidon.knowledge.layered import LayeredStore, Scope
from myrmidon.knowledge.sources import Source


class Protocol(BaseModel):
    """A published sequence of stages, with the source that published it."""

    id: str
    title: str
    applicability: str  # when this pipeline is the right one to start from
    system_class: str = ""  # the class of system its sequence holds for
    aliases: list[str] = Field(default_factory=list)  # ids it was published under before
    scope: Scope = "project"
    given: list[str] = Field(default_factory=list)  # the file kinds the researcher brings
    # how the energy is computed beyond the topology's force field: a QM method and its
    # coupling, for a protocol whose stages are QM/MM
    hamiltonian: Hamiltonian | None = None
    stages: list[Stage] = Field(min_length=1)
    deliverables: list[Deliverable] = Field(default_factory=list)
    # rules across its stages, each value as ``$<stage>.<key>`` of what the stage renders
    invariants: list[Invariant] = Field(default_factory=list)
    sources: list[Source] = Field(min_length=1)  # without a source it is somebody's habit
    pitfalls: list[str] = Field(default_factory=list)  # finding ids that bite this
    validated_by: list[str] = Field(default_factory=list)
    notes: str = ""
    version: int = 1

    # where this file was actually found, which is not always what it claims
    layer: Scope | None = Field(default=None, exclude=True)

    @property
    def ref(self) -> str:
        return f"{self.id}@v{self.version}"

    def check_invariants(self, values: dict[str, dict]) -> list[InvariantResult]:
        """The rules relating values its stages render, against what is known of them:
        ``values`` holds each stage's values by name, every file of it at once. A rule on a
        stage not rendered yet is deferred."""
        return evaluate_invariants(self.invariants, values)

    def priced(self, compute, paces: list) -> dict[str, object]:
        """Each stage, priced on this machine (``knowledge.stage.Price``) for a system the
        protocol does not hold: its price wherever a length and a template price it, at the
        machine's measured paces or its unmeasured floor."""
        from myrmidon.core.mandate import Mandate, System
        from myrmidon.knowledge.stage import MethodStore, TemplateStore, methods_for, price_stage

        mandate = Mandate(question=self.title, system=System(
            hamiltonian=self.hamiltonian or Hamiltonian()), method=Method(
            protocol=self.ref, stages=self.stages))
        methods = methods_for(mandate, MethodStore.default(), None)
        out = {stage.name: price_stage(mandate, stage, templates=TemplateStore.default(),
                                       methods=methods, atoms=None, paces=paces,
                                       compute=compute) for stage in self.stages}
        for stage in self.stages:  # a classical step is the force field's, which a system names
            if out[stage.name].core_h is None and methods.step_ps(stage.regime) is None:
                out[stage.name].basis = ("unpriced: its time step is its force field's, which "
                                         "the built system names (protocol_preview prices it)")
        return out

    @model_validator(mode="after")
    def _holds_together(self) -> Protocol:
        """Its stages as a mandate's method would hold them (each named once, each starting
        from an earlier one), its deliverables over its stages, its rules on them."""
        try:
            Method(stages=self.stages)
        except ValueError as exc:
            raise ValueError(f"{self.id}: {exc}") from exc
        names = {s.name for s in self.stages}
        for d in self.deliverables:
            if d.over and d.over not in names:
                raise ValueError(f"{self.id}: deliverable {d.name} is measured over "
                                 f"{d.over!r}, no stage of the protocol")
        for invariant in self.invariants:
            unknown = sorted(invariant.scopes - {""} - names)
            if unknown:
                raise ValueError(f"{self.id}: invariant {invariant.id} names stage(s) not in "
                                 f"this protocol: {unknown}")
        return self


class ProtocolStore(LayeredStore[Protocol]):
    """Published pipelines, layered like the input sets (knowledge/layered.py). An empty
    query lists them all; its stages' kinds and regimes count as what the protocol is."""

    model = Protocol
    kind = "protocol"
    dirname = "protocols"
    empty_query_lists_all = True

    def primary_text(self, item: Protocol) -> str:
        return f"{item.id} {item.title} " + " ".join(
            f"{s.name} {s.kind} {s.regime}" for s in item.stages)
