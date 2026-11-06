"""A skill says the moves and the decisions; it does not hold science.

A `SKILL.md` is prose with no schema, no source field, no store and no parser, so a box
clearance, a QM padding, a convergence criterion, a finding's rule or an instruction to read
`cpmd.out` by hand would be invisible there: unsourced, unversioned and untested. A physical
rule is a sensor record in `knowledge/rules/`, read at the moment a call acts; a value is an
input set's. These tests walk the shipped skills, collect offenders, one assert per rule,
and each docstring says why the rule exists. The number lint catches the value that leaked,
not the science that did: that is what review is for.
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

# The sections, in order, and nothing else at that level.
SECTIONS = ["## Scope", "## The researcher decides", "## Drive", "## Stop", "## Report"]

# A value that leaked into prose: a number with a physical unit, or an assignment.
UNITS = r"nm|ps|fs|ns|K\b|bar|core-h|a\.u\.|Å|°|kg/m|steps?|iterations?|frames?|%|Hartree"
NUMBER_WITH_UNIT = re.compile(rf"\d(?:\.\d+)?(?:e-?\d+)?\s*(?:{UNITS})")
ASSIGNMENT = re.compile(r"=\s*-?\d")

# A skill that tells the reader to do the harness's work by hand, or cites an issue number.
WORKAROUND = re.compile(r"\bI[0-9]{1,3}\b|(?:read|check|verify)\b[^.\n]*\byourself\b", re.I)


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
def finding_ids() -> set[str]:
    """Every id a finding can carry: those the code raises and the sensor records'."""
    from tests.unit.findings_catalog import known_finding_ids

    from myrmidon.knowledge.sensors import load_sensors

    return known_finding_ids() | {record.id for record in load_sensors()}


def test_a_skill_has_the_five_sections_and_names_registered_tools(skills):
    """Every skill is read the same way: when it is the one, what the researcher decides,
    the moves, when the turn ends, what is reported. The tools it declares are the
    registry's, so a renamed tool fails here rather than in a researcher's session."""
    import myrmidon.tools as basic
    from myrmidon.knowledge.stage import TemplateStore

    basic.register_all()
    registered = {spec.name for spec in REGISTRY.specs()}
    templates = {t.id for t in TemplateStore.builtin().all()}
    offenders = []
    for name, _directory, props, body in skills:
        headings = [line for line in body.splitlines() if line.startswith("## ")]
        if headings != SECTIONS:
            offenders.append(f"{name}: sections {headings}")
        if len(body.splitlines()) > MAX_LINES:
            offenders.append(f"{name}: over {MAX_LINES} lines")
        offenders += [f"{name}: no tool {t}" for t in _declared(props, "tools")
                      if t not in registered]
        offenders += [f"{name}: no stage template {t}" for t in _declared(props, "stages")
                      if t not in templates]
    assert not offenders, "\n  ".join(offenders)


def test_no_skill_or_guidance_names_a_finding(skills, finding_ids):
    """A finding arrives with what it sees, what it means and the usual move, from its
    record; a skill that names one restates a rule that has a home and goes stale when the
    record changes."""
    guidance = (REPO / "src" / "myrmidon" / "server" / "guidance.md").read_text()
    texts = [("guidance", guidance),
             *((name, (d / "SKILL.md").read_text()) for name, d, _p, _b in skills)]
    offenders = [f"{name}: {word}" for name, text in texts
                 for word in set(re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", text)) & finding_ids]
    offenders += [f"{name}: myrmidon-findings" for name, text in texts
                  if "myrmidon-findings" in text]
    assert not offenders, "\n  ".join(sorted(offenders))


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
    """"`run_diagnose` cannot read a single-point log — verify convergence yourself" is
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


PLAN_WORDS = re.compile(r"\bplan_(write|check|revise|approve)\b|`reuse`|\borigin\b")
APPROVAL_FORMS = ("/approve", "dialog", "myrmidon approve")


def test_the_guidance_and_every_skill_lead_to_the_mandate(skills):
    """No skill or guidance line names the plan's tools; the guidance and each skill that
    books compute name `mandate_write` and the three ways the researcher approves."""
    import myrmidon.tools as basic

    basic.register_all()
    books = {spec.name for spec in REGISTRY.specs() if spec.expensive}
    guidance = (REPO / "src" / "myrmidon" / "server" / "guidance.md").read_text()
    offenders = [f"guidance: {m.group(0)}" for m in PLAN_WORDS.finditer(guidance)]
    offenders += [f"guidance: does not name {w}" for w in ("mandate_write", *APPROVAL_FORMS)
                  if w not in guidance]
    for name, _directory, props, body in skills:
        text = body + " " + " ".join(props.values())
        offenders += [f"{name}: {m.group(0)}" for m in PLAN_WORDS.finditer(text)]
        if books & set(_declared(props, "tools")):
            missing = [w for w in ("mandate_write", *APPROVAL_FORMS) if w not in body]
            offenders += [f"{name}: does not name {w}" for w in missing]
    assert not offenders, "\n  ".join(offenders)
