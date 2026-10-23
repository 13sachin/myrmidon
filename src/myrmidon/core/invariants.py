"""Invariants: the relations a set of values has to satisfy, checked rather than commented.

Some of what a recipe knows is not a value but a relation between two values — the
GROMACS output pace must equal the CPMD force-matching pace, or the frames desynchronise
and the fit silently pairs QM forces with the wrong geometry. Left in a comment, such a
relation is not checked; here it is.

The same machinery serves both objects that hold values, and which one an invariant
belongs to follows one rule:

- **within one scope** — ``$tcoupl`` against a literal, or two keys of the same parameter
  set — is a constraint between parameters, which is knowledge, and it lives on the
  inputset.
- **across scopes** — ``$mimic_mdp.dt`` against ``$production.&CPMD/TIMESTEP`` — is a
  constraint between steps, which is procedure, and it lives on the plan.

An invariant whose references are not all known yet is **deferred, not failed**: steps
render one at a time, and a rule about a value that has not been written is not a
violation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Invariant(BaseModel):
    """One rule. Operands are literals, or references ``$scope.key`` / ``$key``."""

    id: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")  # stable, SCREAMING_SNAKE
    equal: list[str] = Field(default_factory=list)
    not_equal: list[str] = Field(default_factory=list)
    message: str

    @model_validator(mode="after")
    def _one_comparison(self) -> Invariant:
        filled = [name for name in ("equal", "not_equal") if getattr(self, name)]
        if len(filled) != 1:
            raise ValueError(f"{self.id}: give exactly one of equal / not_equal")
        if len(getattr(self, filled[0])) < 2:
            raise ValueError(f"{self.id}: {filled[0]} needs at least two operands")
        return self

    @property
    def operands(self) -> list[str]:
        return self.equal or self.not_equal

    @property
    def references(self) -> list[str]:
        return [o[1:] for o in self.operands if isinstance(o, str) and o.startswith("$")]

    @property
    def scopes(self) -> set[str]:
        """The scopes this invariant reaches into; ``""`` for an unqualified ``$key``."""
        found = set()
        for ref in self.references:
            scope, _, key = ref.partition(".")
            found.add(scope if key else "")
        return found

    @property
    def crosses_scopes(self) -> bool:
        """True when this rule relates values that different steps render.

        The test that decides whether a rule is knowledge or procedure.
        """
        return len(self.scopes - {""}) > 1


class InvariantResult(BaseModel):
    id: str
    status: Literal["satisfied", "violated", "deferred"]
    message: str
    detail: str = ""


def _same(a: object, b: object) -> bool:
    try:
        return float(a) == float(b)  # 1 and "1" are the same pace
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def evaluate_invariants(
    invariants: list[Invariant], known: dict[str, dict[str, Any]]
) -> list[InvariantResult]:
    """Check invariants against the values known so far.

    ``known`` maps scope to its resolved values — a step name for a plan, ``""`` for a
    inputset's own single scope.
    """

    def lookup(operand: str) -> tuple[bool, Any]:
        if not (isinstance(operand, str) and operand.startswith("$")):
            return True, operand
        ref = operand[1:]
        scope, _, key = ref.partition(".")
        if not key:  # unqualified: "$nstxout"
            scope, key = "", ref
        values = known.get(scope)
        if values is None or key not in values:
            return False, None
        return True, values[key]

    results = []
    for invariant in invariants:
        resolved = [lookup(o) for o in invariant.operands]
        if not all(ok for ok, _ in resolved):
            missing = [
                o for o, (ok, _) in zip(invariant.operands, resolved, strict=True) if not ok
            ]
            results.append(
                InvariantResult(
                    id=invariant.id,
                    status="deferred",
                    message=invariant.message,
                    detail="not rendered yet: " + ", ".join(missing),
                )
            )
            continue
        values = [v for _, v in resolved]
        first = values[0]
        if invariant.equal:
            holds = all(_same(first, v) for v in values[1:])
        else:
            holds = all(not _same(first, v) for v in values[1:])
        results.append(
            InvariantResult(
                id=invariant.id,
                status="satisfied" if holds else "violated",
                message=invariant.message,
                detail=" vs ".join(
                    f"{o}={v!r}" for o, v in zip(invariant.operands, values, strict=True)
                ),
            )
        )
    return results
