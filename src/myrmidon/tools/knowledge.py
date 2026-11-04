"""Knowledge tools: the shipped skills, one tool for the external fact sources, and the
engines' own documentation.

``lookup``: RCSB PDB search, PubChem and the literature (arXiv + Crossref) are rarely called
and an agent client has web tools of its own, so they are one tool rather than three;
``fetch_pdb`` is a tool of its own because planting a file with provenance is harness work.

``knowledge_search`` is what the agent asks when it is not sure: verbatim, cited passages
of the installed engines' documentation, filtered to the installed version. Local, so it
needs no engine and every machine shows it; a ledger event, so `report` shows what was
consulted before a decision."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.knowledge.chemicals import PubchemParams, lookup_pubchem
from myrmidon.knowledge.docs import DocStore, Passage, excerpt
from myrmidon.knowledge.literature import LitSearchParams, search_literature
from myrmidon.knowledge.lookup import read_skill_properties, skill_hit
from myrmidon.knowledge.search import params_query, release_series
from myrmidon.knowledge.structures import PdbSearchParams, chem_component, search_pdb


class LookupParams(BaseModel):
    source: Literal["pdb", "ligand", "pubchem", "literature"] = Field(
        description="pdb: full-text search of the RCSB PDB (entry ids for fetch_pdb); "
        "ligand: what a deposited residue code (JZ4, ATP) is — name, formula, SMILES and "
        "the charge of the dictionary's reference form, not its protonation at the study's "
        "pH — from the PDB's chemical component dictionary; "
        "pubchem: a compound's formula, weight, net charge, SMILES and IUPAC name; "
        "literature: citable records from arXiv and Crossref"
    )
    query: str = Field(description="free text; for ligand a residue code; for pubchem a "
                       "name, SMILES or CID")
    max_results: int = Field(default=10, ge=1, le=50)
    query_type: str = Field(default="name", pattern="^(name|smiles|cid)$",
                            description="pubchem only: how to read the query")
    literature_source: str = Field(default="both", pattern="^(arxiv|crossref|both)$",
                                   description="literature only: which index")


@tool(
    "lookup",
    navigation=True,
    description="Look a fact up outside the study, citably: a PDB entry by full-text search "
                "(source=pdb), what a deposited ligand's residue code is (source=ligand), a "
                "compound's formula, weight, charge, SMILES and name from PubChem "
                "(source=pubchem), or literature with DOI or arXiv id (source=literature).",
    tags={"knowledge", "web"},
    needs_workspace=False,
)
def lookup(ctx: ToolContext, params: LookupParams) -> Observation:
    if params.source == "pdb":
        found = search_pdb(ctx, PdbSearchParams(query=params.query,
                                                max_results=params.max_results))
        return Observation(ok=found.ok, summary=found.summary, source="rcsb-pdb",
                                 hits=found.hits)
    if params.source == "ligand":
        return chem_component(params.query)
    if params.source == "pubchem":
        found = lookup_pubchem(ctx, PubchemParams(query=params.query,
                                                  query_type=params.query_type))
        fields = ("cid", "name", "formula", "molecular_weight", "charge", "smiles", "iupac_name")
        return Observation(ok=found.ok, summary=found.summary, source="pubchem",
                                 **{k: getattr(found, k) for k in fields})
    found = search_literature(ctx, LitSearchParams(
        query=params.query, max_results=min(params.max_results, 20),
        source=params.literature_source))
    return Observation(ok=found.ok, summary=found.summary, source="literature",
                             papers=found.papers)


class SkillLookupParams(BaseModel):
    skill: str = Field(
        default="",
        description="the skill to read in full. Empty lists what there is, with what each "
        "one is for.",
    )


@tool(
    "skill_lookup",
    navigation=True,
    description="How the agent turns a class of request into a study and drives it: what to "
    "decide, the protocol to start from, the observation that justifies each step, when to "
    "stop. Ids only. Empty lists them, a name returns one.",
    tags={"knowledge"},
    needs_workspace=False,
)
def skill_lookup(ctx: ToolContext, params: SkillLookupParams) -> Observation:
    """Skills reach every driver through a tool, not through a client's private directory.

    `init-project` copies these into `.claude/skills/` and `.qwen/skills/`, which reaches
    only the clients that read those directories and neither agent loop. A session that
    cannot read the recipe for the pipeline it is running is the same failure as a hidden
    tool.

    A skill names ids and holds no science: the values are the input set's and
    the sequence the protocol's, which is why the description above promises neither.
    """
    from myrmidon.knowledge.lookup import skills_dir

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
            return Observation(
                ok=False, summary=f"no skill {params.skill!r}; there is: {known}"
            )
        name, props, path = hit
        ref = ArtifactRef.from_path(
            path, "skill", ctx.workspace.root if ctx.workspace else None
        )
        return Observation(
            ok=True,
            summary=f"{name}: {props['description'][:160]}",
            skills=[skill_hit(name, props)],
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
    return Observation(
        ok=True,
        summary=f"{len(found)} skill(s): {listing}"
        + ". Read one in full with skill_lookup skill=<name> before assembling that kind "
        "of study.",
        skills=[skill_hit(n, p) for n, p, _ in found],
    )


class KnowledgeOption(BaseModel):
    """One route the agent can put to the user: what it is, where it comes from."""
    title: str
    how: str  # one or two sentences, assembled from the sources named in `cites`
    kind: str  # "inputset" (an input set sets it) | "documented" (the manual's)
    cites: list[str] = Field(default_factory=list)
    inputset: str | None = None  # inputset id, for kind inputset
    value: str | None = None  # what that inputset sets


class KnowledgeSearchParams(BaseModel):
    query: str = Field(description="the question or the option name, e.g. 'refcoord-scaling "
                       "with position restraints under pressure coupling' or 'tcoupl'")
    mode: str = Field(default="passages", pattern="^(passages|options)$",
                      description="passages: cited excerpts of the documentation. options: "
                      "two or three routes for a decision, assembled from what an input set "
                      "sets and what the documentation offers — to "
                      "put to the user as named, cited alternatives")
    source: str | None = Field(default=None, pattern="^(gromacs|cpmd|mimic|mimicpy|tutorial)$",
                               description="restrict to one engine's documentation; default: "
                               "the engines this machine has, at their installed versions")
    max_results: int = Field(default=3, ge=1, le=5)


@tool(
    "knowledge_search",
    navigation=True,
    description="Search the installed engines' own documentation: verbatim passages for an option,"
                " keyword or method, each with page, section and version. Quote and cite the "
                "locator; a value it does not return is not documented.",
    tags={"knowledge"},
    needs_workspace=False,
)
def knowledge_search(ctx: ToolContext, params: KnowledgeSearchParams) -> Observation:
    store = _store(ctx)
    installed = {name: info.version for name, info in ctx.engines.items()
                 if getattr(info, "available", False)}
    found = store.search(params.query, limit=params.max_results, source=params.source,
                         installed=installed)
    if not found:
        known = ", ".join(f"{s} {'/'.join(sorted(v))}" for s, v in store.versions().items())
        return Observation(
            ok=False,
            summary=f"nothing documented for {params.query!r} in the corpus "
                    f"({known or 'empty'}); say so rather than guessing, or ask the user",
            versions=installed)
    shown = [p.model_copy(update={"text": excerpt(p.text, params.query)}) for p in found]
    if params.mode == "options":
        return _options(ctx, params.query, shown, installed, store)
    # an engine newer than the corpus reads the nearest documentation, and is told so:
    # a default may have changed between the two, and the citation names the series read
    mismatch = [f"{src}: installed {have}, documentation read is the {use} series"
                for src, (have, use) in store.series_for(installed).items()
                if use and have and use != release_series(have)
                and any(p.source == src for p in shown)]
    return Observation(
        ok=True,
        # the passages are in `passages` and their quotable lines in `citations`: the summary
        # only counts them, so no excerpt is carried three times
        summary=f"{len(shown)} passage(s) for {params.query!r}; quote a passage by its "
                "line in citations"
                + ("\nNote — " + "; ".join(mismatch) + ". Check the option in the installed "
                   "version's own manual before relying on a default." if mismatch else ""),
        passages=shown, citations=[p.citation() for p in shown], versions=installed)


_STORE: DocStore | None = None


def _store(ctx: ToolContext) -> DocStore:
    """One store per process: the corpus is read once and is the same for every call."""
    global _STORE
    if _STORE is None:
        _STORE = DocStore()
    return _STORE


def _options(ctx: ToolContext, query: str, passages: list[Passage], installed: dict,
             store: DocStore) -> Observation:
    """Routes for a decision, from the stage templates, any input sets and the
    documentation. Nothing here is ranked or recommended: a template's or set's setting is
    one route, the documentation's alternatives another — the user chooses."""
    from myrmidon.knowledge.inputsets import InputSetStore

    options: list[KnowledgeOption] = []
    # the option the question is about: the best passage's, when it is an mdp option
    primary = (passages[0].section.replace("mdp option ", "")
               if passages and passages[0].section.startswith("mdp option ") else None)
    inputsets = ctx.inputsets if ctx.inputsets is not None else InputSetStore.default()
    if primary:
        want = primary.replace("-", "_")
        for inputset in inputsets.all():
            spec = next(
                (s for k, s in inputset.parameters.items() if k.replace("-", "_") == want),
                None,
            )
            if spec is None:
                continue
            src = (inputset.sources[spec.source].text if spec.source is not None
                   and spec.source < len(inputset.sources) else None)
            options.append(KnowledgeOption(
                title=f"{primary} = {spec.value} — what inputset {inputset.id} does",
                how=spec.rationale or f"The inputset sets {primary} = {spec.value}.",
                kind="inputset", inputset=inputset.id,
                value=str(spec.value), cites=[c for c in [src] if c]))
        # what a stage template does with it: a value, or one per state the stage chooses
        from myrmidon.knowledge.stage import MethodStore, TemplateStore, flattened

        templates, methods = TemplateStore.default(), MethodStore.default()
        for template in templates.all():
            if not template.kinds:
                continue
            forms, sources = flattened(template, templates)
            form = next((f for k, f in forms.items() if k.replace("-", "_") == want), None)
            if form is None:
                continue
            src = (sources[form.source].text if form.source is not None
                   and form.source < len(sources) else None)
            when = ("; when " + ", ".join(f"{k} is {v}" for k, v in form.when.items())
                    if form.when else "")
            chosen = ([(form.value, when)] if form.value is not None else
                      [(v, f" when the stage's {form.by_stage} is {k}")
                       for k, v in form.map.items()] if form.by_stage else
                      [(m.parameters[form.from_method].value,
                        f" under the {m.id} method ({form.from_method})")
                       for m in methods.all() if m.kind == (form.of or template.reads)
                       and form.from_method in m.parameters] if form.from_method else [])
            for value, when in chosen:
                options.append(KnowledgeOption(
                    title=f"{primary} = {value}{when} — what stage template {template.id} does",
                    how=form.rationale or f"The template sets {primary} = {value}{when}.",
                    kind="inputset", inputset=template.id,
                    value=str(value), cites=[c for c in [src] if c]))
        # one route per distinct value and reason: two agreeing are one alternative
        seen: set[str] = set()
        options = [o for o in options
                   if not (o.title in seen or seen.add(o.title))][:5]
    if passages:
        p0 = passages[0]
        options.append(KnowledgeOption(
            title=f"as documented: {p0.section} ({p0.source} {p0.version})",
            how=p0.text, kind="documented", cites=[p0.citation()]))
    lines = [f"{i + 1}. [{o.kind}] {o.title}" for i, o in enumerate(options)]
    return Observation(
        ok=True,
        summary=f"{len(options)} route(s) for {params_query(query)}; put them to the user by "
                f"name and cite each:\n" + "\n".join(lines),
        passages=passages, citations=[p.citation() for p in passages], versions=installed,
        options=options)
