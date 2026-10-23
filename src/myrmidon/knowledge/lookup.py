"""``lookup``: one tool for the external fact sources.

RCSB PDB search, PubChem and the literature (arXiv + Crossref) are rarely called and an
agent client has web tools of its own, so they are one tool rather than three;
``fetch_pdb`` is a tool of its own because planting a file with provenance is harness
work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation, SkillHit, SkillObservation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.knowledge.chemicals import PubchemParams, lookup_pubchem
from myrmidon.knowledge.literature import LitSearchParams, Paper, search_literature
from myrmidon.knowledge.structures import PdbHit, PdbSearchParams, search_pdb


class LookupObservation(Observation):
    source: str = ""
    # pdb
    hits: list[PdbHit] = Field(default_factory=list)
    # literature
    papers: list[Paper] = Field(default_factory=list)
    # pubchem
    cid: int | None = None
    name: str | None = None
    formula: str | None = None
    molecular_weight: float | None = None
    charge: int | None = None
    smiles: str | None = None
    iupac_name: str | None = None


class LookupParams(BaseModel):
    source: Literal["pdb", "pubchem", "literature"] = Field(
        description="pdb: full-text search of the RCSB PDB (entry ids for fetch_pdb); "
        "pubchem: a compound's formula, weight, net charge, SMILES and IUPAC name; "
        "literature: citable records from arXiv and Crossref"
    )
    query: str = Field(description="free text, or for pubchem a name, SMILES or CID")
    max_results: int = Field(default=10, ge=1, le=50)
    query_type: str = Field(default="name", pattern="^(name|smiles|cid)$",
                            description="pubchem only: how to read the query")
    literature_source: str = Field(default="both", pattern="^(arxiv|crossref|both)$",
                                   description="literature only: which index")


@tool(
    "lookup",
    navigation=True,
    description="Look a fact up outside the workspace, citably: a PDB entry by full-text "
    "search (source=pdb; then fetch_pdb plants it), a compound's formula, molecular weight, "
    "net charge, SMILES and IUPAC name from PubChem (source=pubchem — e.g. to confirm a QM "
    "region's charge), or literature records with DOI or arXiv id (source=literature). "
    "Every answer names its source so it can be cited.",
    returns=LookupObservation,
    tags={"knowledge", "web"},
    needs_workspace=False,
)
def lookup(ctx: ToolContext, params: LookupParams) -> LookupObservation:
    if params.source == "pdb":
        found = search_pdb(ctx, PdbSearchParams(query=params.query,
                                                max_results=params.max_results))
        return LookupObservation(ok=found.ok, summary=found.summary, source="rcsb-pdb",
                                 hits=found.hits)
    if params.source == "pubchem":
        found = lookup_pubchem(ctx, PubchemParams(query=params.query,
                                                  query_type=params.query_type))
        fields = ("cid", "name", "formula", "molecular_weight", "charge", "smiles", "iupac_name")
        return LookupObservation(ok=found.ok, summary=found.summary, source="pubchem",
                                 **{k: getattr(found, k) for k in fields})
    found = search_literature(ctx, LitSearchParams(
        query=params.query, max_results=min(params.max_results, 20),
        source=params.literature_source))
    return LookupObservation(ok=found.ok, summary=found.summary, source="literature",
                             papers=found.papers)


# --- skills: the worked recipes, for every driver rather than two ---------------------


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
            for key in ("question", "tools", "protocols", "inputsets", "findings"):
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
        "inputsets": meta.get("myrmidon-inputsets", ""),
        "findings": meta.get("myrmidon-findings", ""),
    }


def _skill_hit(name: str, props: dict[str, str]) -> SkillHit:
    return SkillHit(
        name=name,
        description=props["description"],
        question=props["question"],
    )



def skill_catalogue() -> list[tuple[str, str]]:
    """(name, description) of every shipped skill, in name order: what a model chooses from."""
    from myrmidon.server.install import skills_dir

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
                       "how a request of this kind becomes a plan, and what decides each "
                       "step. Pick the one whose 'use when' fits best:\n" + menu,
        "parameters": {"type": "object", "properties": {
            "skill": {"type": "string", "enum": [name for name, _ in skills]}},
            "required": ["skill"]},
    }}


class SkillLookupParams(BaseModel):
    skill: str = Field(
        default="",
        description="the skill to read in full. Empty lists what there is, with what each "
        "one is for.",
    )


@tool(
    "skill_lookup",
    navigation=True,
    description="How the agent turns a class of request into a plan and drives it: what to "
    "decide, the protocol to start from, the observation that justifies each step, when to "
    "stop. Ids only. Empty lists them, a name returns one.",
    returns=SkillObservation,
    tags={"knowledge"},
    needs_workspace=False,
)
def skill_lookup(ctx: ToolContext, params: SkillLookupParams) -> SkillObservation:
    """Skills reach every driver through a tool, not through a client's private directory.

    `init-project` copies these into `.claude/skills/` and `.qwen/skills/`, which reaches
    only the clients that read those directories and neither agent loop. A session that
    cannot read the recipe for the pipeline it is running is the same failure as a hidden
    tool.

    A skill names ids and holds no science: the values are the input set's and
    the sequence the protocol's, which is why the description above promises neither.
    """
    from myrmidon.server.install import skills_dir

    root = skills_dir()
    found = []
    for directory in sorted(d for d in root.iterdir() if d.is_dir()):
        path = directory / "SKILL.md"
        if not path.exists():
            continue
        found.append((directory.name, read_skill_properties(path), path))

    if params.skill:
        hit = next((f for f in found if f[0] == params.skill), None)
        if hit is None:
            known = ", ".join(f[0] for f in found) or "none"
            return SkillObservation(
                ok=False, summary=f"no skill {params.skill!r}; there is: {known}"
            )
        name, props, path = hit
        ref = ArtifactRef.from_path(
            path, "skill", ctx.workspace.root if ctx.workspace else None
        )
        return SkillObservation(
            ok=True,
            summary=f"{name}: {props['description'][:160]}",
            skills=[_skill_hit(name, props)],
            content=path.read_text(),
            skill_id=name,
            sha256=ref.sha256,
            # the SKILL.md is a file this call consumed: hashed into the event's inputs,
            # so the ledger says which skill *text* drove the study
            inputs=[ref],
        )

    listing = ", ".join(
        f"{n} ({p['question']})" if p["question"] else n for n, p, _ in found
    )
    return SkillObservation(
        ok=True,
        summary=f"{len(found)} skill(s): {listing}"
        + ". Read one in full with skill_lookup skill=<name> before assembling that kind "
        "of study.",
        skills=[_skill_hit(n, p) for n, p, _ in found],
    )
