"""``knowledge_search``: what the agent asks when it is not sure.

Between "never invent a number" and "ask the user two things at most" the agent needs
somewhere to turn when it does not know how to proceed. This tool is that place: verbatim,
cited passages of the installed engines' documentation, filtered to the installed
version. Local, so it needs no engine and every machine shows it; a ledger event, so
`report` shows what was consulted before a decision.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.knowledge.docs import DocStore, Passage, excerpt


class KnowledgeOption(BaseModel):
    """One route the agent can put to the user: what it is, where it comes from."""
    title: str
    how: str  # one or two sentences, assembled from the sources named in `cites`
    kind: str  # "inputset" (an input set sets it) | "documented" (the manual's)
    cites: list[str] = Field(default_factory=list)
    inputset: str | None = None  # inputset id, for kind inputset
    value: str | None = None  # what that inputset sets


class KnowledgeObservation(Observation):
    passages: list[Passage] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)  # one line per passage, to quote
    versions: dict[str, str] = Field(default_factory=dict)  # what the corpus was filtered to
    options: list[KnowledgeOption] = Field(default_factory=list)  # mode=options


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
    description="Ask the engines' own documentation before asking the user or guessing: "
    "verbatim passages for an mdp option, a keyword or a method, each with the page, "
    "section and version it came from, filtered to the installed engine versions. Quote "
    "what it returns and cite the locator; a value it does not return is not documented.",
    returns=KnowledgeObservation,
    tags={"knowledge"},
    needs_workspace=False,
)
def knowledge_search(ctx: ToolContext, params: KnowledgeSearchParams) -> KnowledgeObservation:
    store = _store(ctx)
    installed = {name: info.version for name, info in ctx.engines.items()
                 if getattr(info, "available", False)}
    found = store.search(params.query, limit=params.max_results, source=params.source,
                         installed=installed)
    if not found:
        known = ", ".join(f"{s} {'/'.join(sorted(v))}" for s, v in store.versions().items())
        return KnowledgeObservation(
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
                if use and have and use != _series(have)
                and any(p.source == src for p in shown)]
    return KnowledgeObservation(
        ok=True,
        # the passages are in `passages` and their quotable lines in `citations`: the summary
        # only counts them, so no excerpt is carried three times
        summary=f"{len(shown)} passage(s) for {params.query!r}; quote a passage by its "
                "line in citations"
                + ("\nNote — " + "; ".join(mismatch) + ". Check the option in the installed "
                   "version's own manual before relying on a default." if mismatch else ""),
        passages=shown, citations=[p.citation() for p in shown], versions=installed)


def _series(version: str) -> str:
    from myrmidon.knowledge.docs import _major

    return _major(version)


_STORE: DocStore | None = None


def _store(ctx: ToolContext) -> DocStore:
    """One store per process: the corpus is read once and is the same for every call."""
    global _STORE
    if _STORE is None:
        _STORE = DocStore()
    return _STORE


def _options(ctx: ToolContext, query: str, passages: list[Passage], installed: dict,
             store: DocStore) -> KnowledgeObservation:
    """Routes for a decision, from the input sets and the documentation. Nothing here is
    ranked or recommended: the input set's setting is one route, the documentation's
    alternatives another — the user chooses."""
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
        # one route per distinct value: two inputsets agreeing are one alternative
        seen: set[str] = set()
        options = [o for o in options if not (o.value in seen or seen.add(o.value))][:3]
    if passages:
        p0 = passages[0]
        options.append(KnowledgeOption(
            title=f"as documented: {p0.section} ({p0.source} {p0.version})",
            how=p0.text, kind="documented", cites=[p0.citation()]))
    lines = [f"{i + 1}. [{o.kind}] {o.title}" for i, o in enumerate(options)]
    return KnowledgeObservation(
        ok=True,
        summary=f"{len(options)} route(s) for {params_query(query)}; put them to the user by "
                f"name and cite each:\n" + "\n".join(lines),
        passages=passages, citations=[p.citation() for p in passages], versions=installed,
        options=options)


def params_query(query: str) -> str:
    return repr(query if len(query) <= 60 else query[:57] + "…")
