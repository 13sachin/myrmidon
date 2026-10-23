"""The plan: what a study is doing, in what order, for how much.

An input set answers *what should this value be, and who says so* — knowledge, true
before any study exists. This module answers the other question: *what are we doing,
in what order, for how many core-hours* — procedure, true for exactly one study.

The two are separate objects: an input set knows nothing of order, and a plan holds no
value an input set already states.

**A plan is a prospective ledger.** The ledger records, for every call, its ``inputs``
and ``outputs`` as artifact references carrying a role; a plan step declares the same
thing before the call is made. That is why a step has no ``after``: the order of a
simulation is data flow, and a step that needs a role no earlier step produces is a
finding rather than a comment.

A plan is **scoped to its question**. A study holds many questions over its
life; ``plan_id`` names the one this plan answers, every call records the plan it was made
under, and a step is carried out only by this plan's calls — a classical grompp from last
week's question never satisfies this question's QM/MM grompp. Work from another question is
used only where a step declares it.

Two things a plan deliberately does **not** hold:

- **an approval.** ``total_core_h`` goes through the submission gate like any other
  estimate; acknowledging that finding is the approval, and the ledger records it
  against :meth:`Plan.digest`. Edit the plan and the digest changes, so the
  acknowledgement stops covering it and has to be asked again.
- **progress.** A step is done when the ledger holds a call that produced its roles.
  A second record of where execution has got to is the thing that drifts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from myrmidon.core.invariants import Invariant, InvariantResult, evaluate_invariants

Shape = Literal["one-shot", "comparison", "sweep", "chain"]

#: a plan value that is a fact about an earlier step's run: ``${annealing.final_temperature_k}``
#: — approved as a rule, and filled in from that run's parsed output when it has completed
REFERENCE_RE = re.compile(r"\$\{([A-Za-z0-9_-]+)\.([A-Za-z0-9_]+)\}")


def references_in(value) -> list[tuple[str, str]]:
    """(step, fact) for every reference in a plan value, nested ones included."""
    if isinstance(value, str):
        return REFERENCE_RE.findall(value)
    if isinstance(value, dict):
        return [ref for v in value.values() for ref in references_in(v)]
    if isinstance(value, list):
        return [ref for v in value for ref in references_in(v)]
    return []


def _takes_a_list(annotation) -> bool:
    """Whether a parameter's annotation is a list (``list[str]``, ``list[str] | None``)."""
    import typing

    origin = typing.get_origin(annotation)
    if origin is list:
        return True
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        return any(_takes_a_list(a) for a in typing.get_args(annotation))
    return False


def unsatisfied_roles(given: list[str], steps: Sequence[Any]) -> dict[str, list[str]]:
    """Step name -> the artifact roles it needs that nothing before it supplies.

    Shared by a plan and by the protocol a plan starts from, because the question is the
    same for both: walking the steps in order, is every role a step reads either brought by
    the user (``given``) or produced by something earlier?

    This is referential integrity and nothing more. It has no opinion about which sequence
    is correct — a researcher who skips a published step by bringing that artifact
    themselves passes, and inventing a new sequence is the work, not an error.
    """
    available = set(given)
    missing: dict[str, list[str]] = {}
    for step in steps:
        absent = [role for role in step.needs if role not in available]
        if absent:
            missing[step.name] = absent
        available.update(step.produces)
    return missing


class Sentinel(BaseModel):
    """A cheap predicate over a finished run's parsed output, declared on the step whose
    run it guards (``core/sentinels.py`` holds the checks). A violation stops the move and
    raises an event; it never fixes anything, because fixing is a decision and decisions
    belong to the loop."""

    check: str  # a name in core/sentinels.CHECKS
    args: dict[str, Any] = Field(default_factory=dict)


class Envelope(BaseModel):
    """What the loop may do without asking, approved with the plan.

    Routine cluster failures are pre-approved here so that the study never stalls on them;
    anything outside is a question for a person (or, when a wake model is configured, for
    one typed model call first).
    """

    # reruns of identical inputs a failed step may have before it parks — a machine fault
    # does not reproduce, so a rerun is a diagnostic as much as a recovery
    rerun_on_failure: int = Field(default=1, ge=0)
    # segments a stage may gain, each from its own restart, when its sentinel fails
    extend_segments: int = Field(default=0, ge=0)
    # the fraction of a sweep's arms that may fail on a known signature and be dropped
    drop_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    # a job past its step's deadline: park the branch with a question, or keep waiting
    on_deadline: Literal["park", "wait"] = "park"


class PlanStep(BaseModel):
    """One call the plan intends to make.

    ``needs`` and ``produces`` are artifact *roles* — "structure", "topology", "tpr",
    "trajectory" — the same vocabulary ``core/artifacts.py`` gives a recorded
    artifact. They are what orders the plan.
    """

    name: str
    tool: str  # the registered tool this step calls
    cites: str | None = None  # "inputset:<id>@v<N>" — where the values came from
    values: dict[str, Any] = Field(default_factory=dict)
    origin: dict[str, str] = Field(default_factory=dict)  # key -> inputset:… | user
    needs: list[str] = Field(default_factory=list)  # artifact roles consumed
    produces: list[str] = Field(default_factory=list)  # artifact roles produced
    # computed by plan_write from the resources in ``values`` (plan_tools.price_step), never
    # given: None for a submitting step that states none, which cannot be approved
    estimate_core_h: float | None = None
    estimate_basis: str | None = None  # how the estimate was arrived at
    note: str = ""
    # the event id of an earlier call whose result this step uses instead of running: done
    # by declaration, costing nothing, and named wherever the step is shown (reuse is
    # encouraged, silent reuse is forbidden)
    reuse: str | None = None
    # predicates the loop evaluates on this step's finished run; a violation is an event
    sentinels: list[Sentinel] = Field(default_factory=list)
    # how long the step's job is expected to take, submission to end; overrun is an event
    deadline_min: int | None = Field(default=None, ge=1)


class Arms(BaseModel):
    """The declared experiment: the axis this study varies, and why.

    This is what replaces ``ValueSpec.tunable``. An input set author cannot know which
    parameter a later researcher came to vary, so permission to change a value cannot
    live in the input set. It lives here, stated once for the whole study, by the
    person who knows the experiment — and :meth:`covers` is what the override gate
    consults instead of a permission bit.

    ``comparison`` and ``sweep`` are one mechanism and differ only in what the arms
    mean: distinct alternatives, or points on an axis.
    """

    varies: str  # a parameter key, or "topology" / "inputset" / "force_field"
    over: list[Any] = Field(min_length=2)
    metric: str | None = None  # what the resulting table compares
    because: str  # the researcher's reason, in their own words

    def covers(self, key: str) -> bool:
        """Is this key the study's declared axis?

        An override of a covered key raises no finding: the plan already said that is
        the experiment. Anything else is still a deviation to be acknowledged, which
        is the right way round — it is what the researcher did not say they were doing.
        """
        return key == self.varies


class Until(BaseModel):
    """The `chain` shape: repeat a step until a criterion holds.

    ``ceiling`` is not optional. A chain with no ceiling is the failure every agent
    loop eventually finds; the harness refuses to write one. It counts segments, the first
    run included: a ceiling of 5 on a 200 ps step is at most 1 ns.

    The loop decides the criterion with ``check`` — the tool named here, called with
    ``values`` (the researcher's tolerance) on the repeating step's output so far — and,
    while it does not hold, continues the step from its own checkpoint or restart with
    ``extend`` (the continuation's length and resources). Neither involves a model: the
    criterion is a predicate.
    """

    criterion: str  # "the temperature has stopped drifting"
    check: str  # the tool call that decides it
    repeats: str  # the step name that repeats
    ceiling: int = Field(ge=1)  # maximum segments, the first included
    values: dict[str, Any] = Field(default_factory=dict)  # the check's parameters
    extend: dict[str, Any] = Field(default_factory=dict)  # the continuation's parameters


class Plan(BaseModel):
    """What this study is doing. Serialisable, diffable, re-runnable; no state."""

    question: str  # the user's sentence
    system: str = ""
    cites: str | None = None  # "protocol:<id>@v<N>" — the published sequence it started from
    steps: list[PlanStep] = Field(min_length=1)
    given: list[str] = Field(default_factory=list)  # roles supplied from outside
    arms: Arms | None = None
    until: Until | None = None
    invariants: list[Invariant] = Field(default_factory=list)
    # what the loop may do without asking, approved as part of the digest
    envelope: Envelope = Field(default_factory=Envelope)
    notes: str = ""
    version: int = 1
    # which question this plan answers: minted when a plan for a new question is written,
    # kept by a revision of the same one. A call is made under the plan on disk and says so
    # in the ledger, and only this plan's calls carry out its steps. "" is a
    # plan written before plans had ids, which every call in the study is matched against
    plan_id: str = ""

    @property
    def shape(self) -> Shape:
        if self.arms is not None:
            return "sweep" if self.arms.metric is None else "comparison"
        return "chain" if self.until is not None else "one-shot"

    @property
    def total_core_h(self) -> float:
        """What the plan costs if every step runs once: the sum of the costs ``plan_write``
        computed from each step's resources on this machine.

        An ``arms`` plan multiplies the steps after the arms diverge; that is left to
        the caller who knows which steps those are, because guessing it here would be
        a number nobody parsed.
        """
        return sum(s.estimate_core_h or 0.0 for s in self.steps)

    def arm_steps(self) -> list[str]:
        """The steps that run once per arm: from the first step whose values carry the
        varied key up to the join — the first later step that takes a role an arm step
        produces into a parameter that is a list — exclusive. Empty without arms. The
        loop runs these per arm and the pricing counts them per arm."""
        if self.arms is None:
            return []
        first = next((i for i, s in enumerate(self.steps)
                      if self.arms.varies in s.values), None)
        if first is None:
            return []
        from myrmidon.core.tool import REGISTRY, UnknownToolError

        out: list[str] = []
        produced: set[str] = set()
        for step in self.steps[first:]:
            if produced & set(step.needs):
                try:
                    fields = REGISTRY.get(step.tool).params_model.model_fields
                except UnknownToolError:
                    fields = {}
                joins = any(
                    isinstance(v, str) and v in produced and k in fields
                    and _takes_a_list(fields[k].annotation)
                    for k, v in step.values.items())
                if joins:
                    break
            out.append(step.name)
            produced |= set(step.produces)
        return out

    def ceiling_core_h(self, extension_core_h: float = 0.0,
                       arm_core_h: float | None = None) -> float:
        """What the plan may cost at most under its own ceilings: one pass, plus every
        further segment an ``until`` may add (``ceiling - 1`` continuations at
        ``extension_core_h`` each), plus every further arm (``arm_core_h`` is one arm's
        share of the pass; given, the arm steps are counted once per extra arm). This is
        the figure an approval covers; the loop refuses to book past it."""
        total = self.total_core_h
        if self.until is not None:
            total += (self.until.ceiling - 1) * extension_core_h
        if self.arms is not None and arm_core_h is not None:
            total += (len(self.arms.over) - 1) * arm_core_h
        return round(total, 3)

    @property
    def unestimated(self) -> list[str]:
        """Steps carrying no cost — submitting steps that state no resources. The plan's
        total is a floor, not a total, and the plan cannot be approved."""
        return [s.name for s in self.steps if s.estimate_core_h is None]

    def step(self, name: str) -> PlanStep:
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(f"no step {name!r} in this plan")

    def unsatisfied(self) -> dict[str, list[str]]:
        """Step name -> the roles it needs that nothing earlier supplies.

        This is referential integrity, **not an opinion about pipeline order.** The
        harness does not know that equilibration precedes QM/MM, and must not: which
        sequence is right is the researcher's, and a new one is what experimenting looks
        like. What it knows is that a step declaring it reads a file nothing makes cannot
        run. A study that brings its equilibrated box from another package lists the role
        in ``given`` and is accepted; one that skips equilibration entirely and runs on
        what it brought is accepted too.

        What it does catch is a step deleted from an edited plan: the next step's inputs
        stop being satisfiable rather than the omission passing silently.
        """
        return unsatisfied_roles(self.given, self.steps)

    def check_invariants(self, rendered: dict[str, dict[str, Any]]) -> list[InvariantResult]:
        """Evaluate the cross-step rules against the steps rendered so far.

        A plan's invariants are the ones that relate values two *different* steps write —
        the GROMACS output pace against the CPMD force-matching pace, the inert mdp ``dt``
        against the CPMD ``TIMESTEP`` that actually advances time. A rule inside one
        parameter set belongs to that inputset instead, and ``InputSet`` refuses it.
        """
        return evaluate_invariants(self.invariants, rendered)

    def digest(self) -> str:
        """The plan's content hash — what an approval is recorded against.

        Excludes nothing: a changed value, a changed order and a changed reason all
        produce a different digest, so an approval never silently covers a plan the
        user has since edited.
        """
        payload = self.model_dump(mode="json", exclude_none=True)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @model_validator(mode="after")
    def _steps_are_named_once(self) -> Plan:
        names = [s.name for s in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate step names in plan: {names}")
        if self.until is not None and self.until.repeats not in names:
            raise ValueError(f"until.repeats names no step in this plan: {self.until.repeats!r}")
        for invariant in self.invariants:
            unknown = sorted(invariant.scopes - {""} - set(names))
            if unknown:
                raise ValueError(
                    f"invariant {invariant.id} refers to step(s) not in this plan: {unknown}"
                )
        return self
