"""The shipped skills: their properties, how they are found, and the native loop's opening
turn that offers them."""

from __future__ import annotations

from pathlib import Path

from myrmidon.core.observation import SkillHit

# --- skills: the worked recipes, for every driver rather than two ---------------------



def skills_dir() -> Path:
    """The skills this install ships (packaged with the module, like the input sets)."""
    return Path(__file__).resolve().parent / "skills"

def read_skill_properties(skill_md: Path) -> dict[str, str]:
    """A SKILL.md's frontmatter, as the listing and the contract test both need it.

    One parser, two readers: `skill_lookup` and `tests/unit/test_skill_contract.py` must
    agree about what a skill declares, or the contract checks a different file from the one
    the agent is served. `skills-ref` is the spec's own parser and is what the contract
    validates with; it is a dev dependency, so a release environment falls back to a
    regex over the frontmatter. Only its absence falls back: a SKILL.md it
    cannot parse is an error, not a reason to read the file with a weaker parser.
    """
    text = skill_md.read_text()
    meta: dict = {}
    try:
        import skills_ref
    except ImportError:
        skills_ref = None
    if skills_ref is not None:
        props = skills_ref.read_properties(skill_md.parent)
        meta = {
            "name": props.name,
            "description": props.description,
            "license": props.license or "",
            "compatibility": props.compatibility or "",
            **(props.metadata or {}),
        }
    else:
        import re

        head = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        if head:
            m = re.search(
                r"^description:\s*>-?\s*\n((?:\s+.*\n)+)|^description:\s*(.+)$",
                head.group(1),
                re.M,
            )
            if m:
                meta["description"] = " ".join((m.group(1) or m.group(2) or "").split())
            n = re.search(r"^name:\s*(.+)$", head.group(1), re.M)
            if n:
                meta["name"] = n.group(1).strip()
            for key in ("question", "tools", "protocols", "stages"):
                v = re.search(rf"^\s+myrmidon-{key}:\s*(.+)$", head.group(1), re.M)
                if v:
                    meta[f"myrmidon-{key}"] = v.group(1).strip().strip("\"'")
    return {
        "name": meta.get("name", skill_md.parent.name),
        "description": meta.get("description", ""),
        "license": meta.get("license", ""),
        "compatibility": meta.get("compatibility", ""),
        "question": meta.get("myrmidon-question", ""),
        "tools": meta.get("myrmidon-tools", ""),
        "protocols": meta.get("myrmidon-protocols", ""),
        "stages": meta.get("myrmidon-stages", ""),
    }


def skill_hit(name: str, props: dict[str, str]) -> SkillHit:
    return SkillHit(
        name=name,
        description=props["description"],
        question=props["question"],
    )


def skill_catalogue() -> list[tuple[str, str]]:
    """(name, description) of every shipped skill, in name order: what a model chooses from."""

    out = []
    for directory in sorted(d for d in skills_dir().iterdir() if d.is_dir()):
        path = directory / "SKILL.md"
        if path.exists():
            out.append((directory.name, " ".join(
                read_skill_properties(path).get("description", "").split())))
    return out


def skill_opening() -> dict | None:
    """The tool a native session's first turn must call: `skill_lookup`, with the skills as
    a closed choice and each one's "use when" in front of the model.

    Smaller models do not call `skill_lookup` on their own, and a study driven without its
    skill fails the checks the skill names. The choice stays the model's; the runtime only
    makes choosing the first step. None when no skill ships."""
    skills = skill_catalogue()
    if not skills:
        return None
    menu = "\n".join(f"- {name}: {description}" for name, description in skills)
    return {"type": "function", "function": {
        "name": "skill_lookup",
        "description": "Before anything else, choose the skill for this request and read it: "
                       "how a request of this kind becomes a mandate, and what decides each "
                       "step. Pick the one whose 'use when' fits best:\n" + menu,
        "parameters": {"type": "object", "properties": {
            "skill": {"type": "string", "enum": [name for name, _ in skills]}},
            "required": ["skill"]},
    }}
