"""ProtocolStore: published pipelines — the sequences a field has agreed are worth running.

Beside the stage templates and methods in the knowledge system. A **template** says how an
engine is told one kind of stage, and a **method** what a Hamiltonian's constants are and who
says so; a **protocol** says what sequence of steps a published pipeline runs, and which of
its stages each step renders. Both are knowledge: true
before any study exists, citable, shared. A **mandate** (`core/mandate.py`) is one study's
instance — its system, its decisions, its ceiling, its approval — and names the protocol it
starts from, if one.

**A protocol is a starting point, never a rule.** "Equilibrate classically, then run QM/MM"
is one published pipeline; it is not a law, and a researcher building a new sequence is
doing the work rather than making a mistake. Nothing here gates a study that departs from a
protocol: `protocol_preview` prices it on the built system, and adapting it raises no
finding, because adapting it is the point.

The three layers are the InputSetStore's: the pack shipped in the package, the machine's
`~/.config/myrmidon/protocols/`, and `protocols/` in the project directory; the nearest
definition of an id wins.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from myrmidon.core.invariants import Invariant, InvariantResult, evaluate_invariants
from myrmidon.core.mandate import Hamiltonian, Stage
from myrmidon.knowledge.inputsets import RENDERED_FILES
from myrmidon.knowledge.layered import LayeredStore, Scope
from myrmidon.knowledge.sources import Source


def unsatisfied_roles(given: list[str], steps) -> dict[str, list[str]]:
    """Step name -> the artifact roles it needs that nothing before it supplies: walking the
    steps in order, is every role a step reads either brought by the user (``given``) or
    produced by something earlier? Referential integrity and nothing more — a researcher
    who skips a published step by bringing that artifact themselves passes."""
    available = set(given)
    missing: dict[str, list[str]] = {}
    for step in steps:
        absent = [role for role in step.needs if role not in available]
        if absent:
            missing[step.name] = absent
        available.update(step.produces)
    return missing


class ProtocolStep(BaseModel):
    """One step of a published pipeline.

    It carries no values: those belong to the stage it `cites`, compiled from the stage's
    physical terms, its templates and the methods. What it may carry is
    its wiring — which of the roles it needs feeds which of its tool's parameters
    (``method: cpmd_method_heating``) — because that is part of the published sequence, not
    a choice. It carries no price: a stage is priced on the machine that runs it
    (:meth:`Protocol.priced`, and a study's mandate).
    """

    name: str
    tool: str  # the registered tool this step calls
    # where this step's values come from: "stage:<name>", every file a stage some step of the
    # protocol carries is rendered as (a coupled stage's mdp and CPMD input both), or
    # "stage:<name>/<renders>", one of them (mdp, cpmd_inp)
    cites: str | None = None
    needs: list[str] = Field(default_factory=list)  # artifact roles consumed
    produces: list[str] = Field(default_factory=list)  # artifact roles produced
    values: dict[str, str] = Field(default_factory=dict)  # parameter -> a role it needs
    # the physical stage a run step carries out, as a mandate's method states it
    stage: Stage | None = None
    note: str = ""

    def cited(self) -> tuple[str, str]:
        """(the stage it cites, the file it is rendered as; "" for every file of it)."""
        name, _, renders = (self.cites or "").removeprefix("stage:").partition("/")
        return name, renders


class Protocol(BaseModel):
    """A published sequence, with the source that published it."""

    id: str
    title: str
    applicability: str  # when this pipeline is the right one to start from
    system_class: str = ""  # the class of system its sequence and estimates hold for
    aliases: list[str] = Field(default_factory=list)  # ids it was published under before
    scope: Scope = "project"
    given: list[str] = Field(default_factory=list)  # roles the researcher brings
    # how the energy is computed beyond the topology's force field: a QM method and its
    # coupling, for a protocol whose stages are QM/MM
    hamiltonian: Hamiltonian | None = None
    steps: list[ProtocolStep] = Field(min_length=1)
    invariants: list[Invariant] = Field(default_factory=list)  # rules across its steps
    sources: list[Source] = Field(min_length=1)  # without a source it is somebody's habit
    pitfalls: list[str] = Field(default_factory=list)  # finding ids that bite this
    validated_by: list[str] = Field(default_factory=list)
    notes: str = ""
    version: int = 1

    # where this file was actually found, which is not always what it claims
    layer: Scope | None = Field(default=None, exclude=True)


    def unsatisfied(self) -> dict[str, list[str]]:
        return unsatisfied_roles(self.given, self.steps)

    def check_invariants(self, rendered: dict[str, dict]) -> list[InvariantResult]:
        """The rules relating values two of its steps render, against what is rendered.

        A protocol holds them because they are constraints between steps, which no single
        template can see; a study whose mandate names the protocol, and its stages by the
        protocol's names, is checked against them as each step renders. ``rendered`` is
        each render's values by ``stage:<name>/<renders>``; a step citing a whole stage
        reads all of that stage's at once.
        """
        by_step = {}
        for step in self.steps:
            name, renders = step.cited()
            found = [values for key, values in rendered.items()
                     if key == step.cites or (not renders and key.startswith(f"stage:{name}/"))]
            if found:
                by_step[step.name] = {k: v for values in found for k, v in values.items()}
        return evaluate_invariants(self.invariants, by_step)

    def priced(self, compute, paces: list) -> dict[str, object]:
        """Each stage its steps carry, priced on this machine (``knowledge.stage.Price``) for
        a system the protocol does not hold: its price wherever a length and a template
        price it, at the machine's measured paces or its unmeasured floor."""
        from myrmidon.core.mandate import Mandate, Method, System
        from myrmidon.knowledge.stage import MethodStore, TemplateStore, methods_for, price_stage

        stages = [s.stage for s in self.steps if s.stage is not None]
        mandate = Mandate(question=self.title, system=System(
            hamiltonian=self.hamiltonian or Hamiltonian()), method=Method(
            protocol=f"{self.id}@v{self.version}", stages=stages))
        methods = methods_for(mandate, MethodStore.default(), None)
        out = {stage.name: price_stage(mandate, stage, templates=TemplateStore.default(),
                                       methods=methods, atoms=None, paces=paces,
                                       compute=compute) for stage in stages}
        for stage in stages:  # a classical step is the force field's, which a system names
            if out[stage.name].core_h is None and methods.step_ps(stage.regime) is None:
                out[stage.name].basis = ("unpriced: its time step is its force field's, which "
                                         "the built system names (protocol_preview prices it)")
        return out

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
                    "values belong to the stages it cites"
                )
        stages = {s.stage.name for s in self.steps if s.stage is not None}
        for step in self.steps:
            if not step.cites:
                continue
            name, renders = step.cited()
            if (not step.cites.startswith("stage:") or name not in stages
                    or (renders and renders not in RENDERED_FILES)):
                raise ValueError(
                    f"{self.id}: step {step.name} cites {step.cites!r}; a step cites the "
                    f"stage it renders as stage:<name>, or stage:<name>/<renders> for one of "
                    f"its files, a stage among {sorted(stages)} and renders among "
                    f"{sorted(RENDERED_FILES)}; an input set is cited by the stage it renders")
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
