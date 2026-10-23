"""A skill names ids; it does not hold science.

A `SKILL.md` is prose with no schema, no source field, no store and no parser, so a box
clearance, a QM padding, an NVE drift criterion, a system's temperatures and force errors,
an id nothing can resolve, or an instruction to read `cpmd.out` by hand would be invisible
there: unsourced, unversioned and untested. These tests walk the shipped skills, collect
offenders, one assert per rule, and each docstring says why the rule exists. The rules are
conservative on purpose — the number lint catches the value that leaked, not the science
that did. The rule is one line: if a sentence can change the physical system, the sequence,
an acceptance decision, or compute expenditure, it is not skill prose, and the skill names
the id instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import skills_ref  # noqa: F401
except ImportError as exc:  # pragma: no cover - the environment, not the code, is wrong
    raise ImportError(
        "skills-ref is a dev dependency: without it read_skill_properties() falls back to a "
        "parser that ignores licence and compatibility, and these contracts judge the wrong "
        'thing. Install it with `pip install -e ".[dev]"`.'
    ) from exc

from myrmidon.core.tool import REGISTRY
from myrmidon.knowledge.lookup import read_skill_properties

REPO = Path(__file__).resolve().parent.parent.parent
SKILLS = REPO / "src" / "myrmidon" / "knowledge" / "skills"

MAX_LINES = 120  # a skill is read in full, every time, by a model paying for the context

# The sections, in order. 1-6 are required; 7 is optional.
SECTIONS = [
    "## Scope",
    "## The researcher decides",
    "## Start from",
    "## Drive",
    "## Stop",
    "## Report",
]
OPTIONAL_LAST = "## What goes wrong"

# A value that leaked into prose: a number with a physical unit, or an assignment.
UNITS = r"nm|ps|fs|ns|K\b|bar|core-h|a\.u\.|Å|°|kg/m|steps?|iterations?|frames?|%|Hartree"
NUMBER_WITH_UNIT = re.compile(rf"\d(?:\.\d+)?(?:e-?\d+)?\s*(?:{UNITS})")
ASSIGNMENT = re.compile(r"=\s*-?\d")

# A skill that tells the reader to do the harness's work by hand, or cites an issue number.
WORKAROUND = re.compile(r"\bI[0-9]{1,3}\b|(?:read|check|verify)\b[^.\n]*\byourself\b", re.I)

BACKTICKED = re.compile(r"`([^`\n]+)`")
SNAKE = re.compile(r"[a-z][a-z0-9_]*$")
DOTTED = re.compile(r"([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)[a-z0-9_.]*$")
SHOUTY = re.compile(r"[A-Z][A-Z0-9_]{3,}$")


def _vocabulary() -> set[str]:
    """The field names of the objects a skill drives are ids it may name.

    `required_from_user`, `renders`, `arms`, `cites` are what the input set, the protocol
    and the plan call themselves; a skill that names the concept should spell it the way
    the object does.
    """
    from myrmidon.core.plan import Plan, PlanStep
    from myrmidon.knowledge.inputsets import InputSet, ValueSpec
    from myrmidon.knowledge.protocols import Protocol, ProtocolStep

    names: set[str] = set()
    for model in (InputSet, ValueSpec, Protocol, ProtocolStep, Plan, PlanStep):
        names.update(model.model_fields)
    return names


VOCABULARY = _vocabulary()


def _skills() -> list[tuple[str, Path, dict[str, str], str]]:
    """Every shipped skill: name, directory, frontmatter properties, body."""
    out = []
    for directory in sorted(d for d in SKILLS.iterdir() if d.is_dir()):
        path = directory / "SKILL.md"
        if not path.exists():
            continue
        text = path.read_text()
        body = text.split("---", 2)[2] if text.startswith("---") else text
        out.append((directory.name, directory, read_skill_properties(path), body))
    return out


def _declared(props: dict[str, str], key: str) -> list[str]:
    return props.get(key, "").split()


@pytest.fixture(scope="module")
def skills():
    found = _skills()
    assert found, "no skills ship"
    return found


@pytest.fixture(scope="module")
def stores():
    from myrmidon.server.mcp_server import build_context

    return build_context()


@pytest.fixture(scope="module")
def specs():
    import myrmidon.basic_tools as basic

    basic.register_all()
    return {spec.name: spec for spec in REGISTRY.specs()}


@pytest.fixture(scope="module")
def source_literals() -> str:
    """Every line of shipped Python, for the finding ids a skill is allowed to name."""
    return "\n".join(
        p.read_text() for p in (REPO / "src" / "myrmidon").rglob("*.py")
    )


@pytest.fixture(scope="module")
def finding_ids() -> set[str]:
    """Every finding id the shipped code can raise: what a skill's `myrmidon-findings`
    may name. An id that resolves on one machine and nowhere else is a pointer to nothing;
    a finding id is in the code the reader receives, or it is not."""
    from tests.unit.findings_catalog import known_finding_ids

    return known_finding_ids()


def test_no_value_leaks_into_skill_prose(skills):
    """Lint, not proof.

    A number with a unit, or an assignment, is a value — and a value has a source, an
    applicability and a version, which a skill has nowhere to put: a `distance_nm=1.0384`
    or a `padding_nm=0.35` in prose travels with no source at all. What this cannot catch
    is science stated without a number; that is what review is for.
    """
    offenders = []
    for name, _directory, _props, body in skills:
        for line in body.splitlines():
            hit = NUMBER_WITH_UNIT.search(line) or ASSIGNMENT.search(line)
            if hit:
                offenders.append(f"{name}: {line.strip()[:90]}")
    assert not offenders, (
        "a value is stated in skill prose; it belongs to an input set, a plan or a "
        "gate:\n  " + "\n  ".join(offenders)
    )


def test_a_skill_never_ships_a_workaround_for_a_missing_tool(skills):
    """"`mimic_diagnose` cannot read a single-point log — verify convergence yourself" is
    not guidance. A missing sensor is a stop condition, never an instruction to parse an
    engine log by hand: what the model reads by eye is not in the ledger and is not
    grounded."""
    offenders = []
    for name, _directory, _props, body in skills:
        for line in body.splitlines():
            if WORKAROUND.search(line):
                offenders.append(f"{name}: {line.strip()[:90]}")
    assert not offenders, (
        "a skill tells the reader to do the harness's work by hand:\n  "
        + "\n  ".join(offenders)
    )

