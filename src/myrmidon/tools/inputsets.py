"""Input-set tools: look one up, and apply one to render an engine file."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import (
    InputSetHit,
    InputSetSource,
    Observation,
    inputset_ref,
)
from myrmidon.core.tool import ToolContext, precheck, tool
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.inputs import materialize_symlink, symlink_component
from myrmidon.knowledge.catalog import canonical
from myrmidon.knowledge.inputsets import RENDERED_FILES


def get_protocol_store(ctx: ToolContext):
    from myrmidon.knowledge.inputsets import InputSetStore

    if ctx.inputsets is not None:
        return ctx.inputsets
    return InputSetStore.default()


class ProtocolLookupParams(BaseModel):
    query: str = Field(
        default="",
        description="what you need parameters for, e.g. 'NVT protein'. Leave empty to "
        "list the whole store — the answer to 'what inputsets do I have?'",
    )


def _adjustable(inputset) -> tuple[dict, dict]:
    """(rationale, derivable) — why each value is what it is, and how to obtain the ones
    the input set refuses to supply.

    There is no third map of pre-approved "tunable" keys: an input set's author, who has
    never seen the study, cannot know which parameter the researcher came to vary, so a
    value set otherwise is said each time, with the why.
    """
    rationale, derivable = {}, {}
    for key, spec in inputset.parameters.items():
        if spec.rationale:
            rationale[key] = spec.rationale
        if spec.derivable:
            derivable[key] = spec.derivable
    return rationale, derivable


def _protocol_hit(inputset, *, full: bool = False, values: bool = False) -> InputSetHit:
    """Project an InputSet onto the wire shape.

    A listing returns the compact summary a mandate needs (`required_from_user`, `derivable`,
    `pitfalls`) so the result stays well within the client's output cap; several sets named
    by id add their parameter maps (``values``); one set by id returns its full parameter
    map, rationale and sources.
    """
    rationale, derivable = _adjustable(inputset)
    required = {
        key: spec.derivable or "ask the user"
        for key, spec in inputset.unsupplied().items()
    }
    if not full:
        # a listing names what each set leaves open in a clause; one set by id says why
        def clause(text: str) -> str:
            first = text.split(". ")[0].split("; ")[0]
            return first if len(first) <= 90 else first[:89] + "…"

        return InputSetHit(
            id=inputset.id,
            title=inputset.title,
            applicability="",
            system_class=inputset.system_class,
            scope=inputset.scope,
            layer=inputset.layer,
            version=inputset.version,
            renders=inputset.renders,
            required_from_user={k: clause(v) for k, v in required.items()},
            derivable={k: clause(v) for k, v in derivable.items()},
            parameters=inputset.values() if values else {},
        )
    return InputSetHit(
        id=inputset.id,
        title=inputset.title,
        applicability=inputset.applicability,
        system_class=inputset.system_class,
        scope=inputset.scope,
        layer=inputset.layer,
        version=inputset.version,
        renders=inputset.renders,
        parameters=inputset.values(),
        required_from_user=required,
        rationale=rationale,
        derivable=derivable,
        sources=[InputSetSource(**s.model_dump()) for s in inputset.sources],
        citations=inputset.citations,
        pitfalls=inputset.pitfalls,
        validated_by=inputset.validated_by,
        notes=inputset.notes,
    )


#: where a key read from the run before comes from, in words
FROM_RUN_WORDS = ("the final ionic temperature of the collected run the stage starts from, "
                  "read from its log at render, unless temperature_k.start_k is stated")


def _said(form, reads: str):
    """Where a template's key takes its value from, in words: its source, then what shapes
    it — the states it is written in, a default, its unit, the numbers beside it, how often
    it repeats."""
    from myrmidon.knowledge.stage import FROM_RUN

    if form.value is not None and not (form.when or form.repeat):
        return form.value
    if form.value is not None:
        out = f"{form.value}"
    elif form.from_stage in FROM_RUN:
        out = FROM_RUN_WORDS
    elif form.from_stage:
        out = f"the stage's {form.from_stage}"
    elif form.from_method:
        out = f"the {form.of or reads} method's {form.from_method}"
    elif form.from_system:
        out = f"read from the system ({form.from_system})"
    else:
        out = f"by the stage's {form.by_stage}: " + ", ".join(
            f"{k} -> {v}" for k, v in form.map.items())
    if form.with_:
        out += " then " + " ".join(str(v) for v in form.with_)
    if form.unit:
        out += f", in {form.unit}"
    if form.default is not None:
        out += f", else {form.default}"
    if form.repeat:
        out += f", once per {form.repeat.replace('_', ' ')}"
    if form.when:
        out += "; when " + ", ".join(f"{k} is {v}" for k, v in form.when.items())
    return out


def _template_hit(template, store, *, full: bool = False, values: bool = False) -> InputSetHit:
    """A stage template on the wire, as an input set is: what a stage of its kinds renders,
    each key with where its value comes from."""
    from myrmidon.knowledge.stage import FROM_RUN, flattened, with_inherited

    template = with_inherited(template, store)
    forms, sources = flattened(template, store)
    return InputSetHit(
        id=template.id, title=template.title,
        applicability=template.applicability if full else "",
        system_class=f"{', '.join(template.kinds)} stages, {', '.join(template.regimes)}",
        scope=template.scope, layer=template.layer, version=template.version,
        renders=template.renders,
        parameters={k: _said(f, template.reads) for k, f in forms.items()}
        if full or values else {},
        # what the stage must state for the file to be written; what the render reads from
        # the run before (a ramp's start) is derivable, not the researcher's
        required_from_user={k: f"the stage's {f.from_stage}" for k, f in forms.items()
                            if f.required and f.from_stage and f.from_stage not in FROM_RUN},
        derivable={k: FROM_RUN_WORDS for k, f in forms.items() if f.from_stage in FROM_RUN},
        rationale={k: f.rationale for k, f in forms.items() if f.rationale} if full else {},
        sources=[InputSetSource(**x.model_dump()) for x in sources] if full else [],
        pitfalls=template.pitfalls,
    )


@tool(
    "inputset_lookup",
    navigation=True,
    description="Search the stage templates (what a mandate stage of each kind renders) and "
                "any site or project input sets, or list them with an empty query. Name ids "
                "for each one's keys and where each value comes from; one id alone adds its "
                "rationale and sources. required_from_user values are the researcher's.",
    tags={"knowledge", "inputsets"},
    needs_workspace=False,
)
def inputset_lookup(ctx: ToolContext, params: ProtocolLookupParams):
    from myrmidon.knowledge.stage import Template, TemplateStore

    store = get_protocol_store(ctx)
    templates = TemplateStore.default()
    q = params.query.strip()
    # the sets and templates the query names by id: one alone in full, several with values
    sets = store.all()
    chosen = [t for t in templates.all() if t.kinds]  # the ones a stage is told with
    by_name = {**{t.id: t for t in chosen},
               **{a: s for s in sets for a in getattr(s, "aliases", ())},
               **{s.id: s for s in sets}}  # an id before any alias
    named = []
    for word in q.replace(",", " ").split():
        name = word.strip(".;:()[]'\"").removeprefix("inputset:").partition("@")[0]
        found = by_name.get(name)
        if found is not None and found not in named:
            named.append(found)

    def hit(item, *, full: bool = False, values: bool = False) -> InputSetHit:
        if isinstance(item, Template):
            return _template_hit(item, templates, full=full, values=values)
        return _protocol_hit(item, full=full, values=values)

    if named:
        alone = len(named) == 1
        return Observation(
            ok=True,
            summary=f"{len(named)} named: " + ", ".join(
                f"{p.id}@v{p.version} ({p.layer})" for p in named)
            + ("" if alone else "; each with its values — name one alone for its rationale "
               "and sources"),
            inputsets=[hit(p, full=alone, values=True) for p in named],
        )
    listing = not q
    leaves = {t.id for t in chosen}
    hits = (sorted([*sets, *chosen], key=lambda p: p.id) if listing else
            [*store.search(q), *(t for t in templates.search(q) if t.id in leaves)])
    full = len(hits) == 1
    if listing:
        summary = (f"{len(chosen)} stage template(s) and {len(sets)} input set(s); pass "
                   "query='<id>' for one's keys, where each value comes from, and sources")
    else:
        summary = f"{len(hits)} matched" + (
            f"; best: {hits[0].id}@v{hits[0].version} ({hits[0].layer})" if hits else
            " — call with no query to list them all")
    return Observation(ok=True, summary=summary, inputsets=[hit(p, full=full) for p in hits])


class InputSetApplyParams(BaseModel):
    inputset: str = Field(default="", description="a site or project input set's id, from "
                          "inputset_lookup, for a run outside the mandate's stages; omit it "
                          "to render a mandate stage (stage=), which every shipped template "
                          "serves")
    method: str | None = Field(
        default=None,
        description="for a purpose input set, its method set: how interactions are computed. "
        "Default for GROMACS: the one covering the force field the topology includes")
    topology: str | None = Field(
        default=None, description="the topology the run uses: the force field, the solute's "
        "restraints and the coupling groups are read from it")
    index: str | None = Field(default=None, description="the index file the run uses: "
                              "coupling groups an index splits, the QM atoms' group")
    structure: str | None = Field(default=None, description="the structure the run starts "
                                  "from: whether it continues a run (carries velocities)")
    output: str | None = Field(
        default=None,
        description="workspace-relative path to write, e.g. '02_run/nvt.mdp'. Omit it to "
        "resolve the values only: you get every value with its origin and its sources, and "
        "nothing is written. That is how to read an input set that renders no file.",
    )
    stage: str | None = Field(
        default=None,
        description="a stage the study's mandate names (method.stages): its file is "
        "compiled from the stage's physical terms, the force field's method and the "
        "system's facts, whatever inputset the call names, and the ledger records it as "
        "that stage's. The output's suffix names the engine (.mdp, .inp); topology and "
        "index default to the mandate's",
    )
    overrides: dict = Field(
        default_factory=dict,
        description="parameters to set yourself. Each is recorded as origin 'user'. "
        "Changing a value the input set states is recorded (INPUTSET_VALUE_OVERRIDDEN) "
        "and shown in the report; adding a key the input set does not mention is free. "
        "This is also how you supply what the input set lists as required_from_user.",
    )
    materialize: bool = Field(
        default=False,
        description="if the output path goes through a symlink, replace that link with a "
        "private real copy before writing",
    )

    def sized(self, ctx: ToolContext) -> InputSetApplyParams:
        """A stage the study's mandate names is rendered from the mandate, whatever input
        set the call also names: its physical terms, not a set's, are what is approved."""
        from myrmidon.core.mandate import read_mandate

        if not (self.stage and self.inputset):
            return self
        mandate = read_mandate(ctx.workspace)
        if mandate is None or mandate.method.stage(self.stage) is None:
            return self
        return self.model_copy(update={"inputset": ""})


def _file(ctx: ToolContext, path: str | None) -> Path | None:
    from myrmidon.engines.inputs import resolve_in_ws

    if not path:
        return None
    real = resolve_in_ws(ctx, path)
    return real if real.is_file() else None


def _covers(method, forcefield: str) -> bool:
    return any(forcefield.lower().startswith(p.lower()) for p in method.forcefields)


def _method_for(ctx: ToolContext, params: InputSetApplyParams, inputset, store
                ) -> tuple[object | None, list[Finding]]:
    """The method set a purpose set is rendered with: the call's, or for an engine whose
    topology includes a force field, the one covering it."""
    from myrmidon.engines.registry import adapter

    if inputset.axis != "purpose":
        return None, []
    ref = params.method
    try:
        reads_forcefield = adapter(inputset.engine).forcefield
    except KeyError:
        reads_forcefield = None
    top = _file(ctx, params.topology) if reads_forcefield else None
    ff = reads_forcefield(top) if top is not None else None
    candidates = sorted(m.id for m in store.all()
                        if m.axis == "method" and m.engine == inputset.engine)
    if ref:
        try:
            method, wanted_ver = store.get_versioned(ref)
        except KeyError:
            method, wanted_ver = None, None
        if method is None or method.axis != "method" or method.engine != inputset.engine:
            return None, [Precondition(
                id="INPUTSET_METHOD_UNKNOWN",
                message=f"{ref} is not a {inputset.engine} method set; known: "
                + (", ".join(candidates) or "none"),
                suggestion="name one of them as method", needs=["method"])]
        if wanted_ver is not None and str(method.version) != wanted_ver:
            return None, [Precondition(
                id="INPUTSET_OTHER_VERSION",
                message=f"requested {method.id} at v{wanted_ver}; "
                        f"this machine has v{method.version}",
                suggestion=f"use {method.id}@v{method.version}", needs=["method"])]
        if ff is not None and method.forcefields and not _covers(method, ff.value):
            return None, [Precondition(
                id="INPUTSET_METHOD_MISMATCH",
                message=f"the topology includes {ff.value}; {method.id} is for "
                f"{', '.join(method.forcefields)}*",
                suggestion="name the method set for the topology's force field, or the "
                "topology the method is for", needs=["method"])]
        return method, []
    if ff is not None:
        matches = [m for m in store.all() if m.axis == "method"
                   and m.engine == inputset.engine and _covers(m, ff.value)]
        if len(matches) == 1:
            return matches[0], [Finding(
                id="INPUTSET_METHOD_READ",
                message=f"the topology includes {ff.value}: {matches[0].id} "
                f"v{matches[0].version} computes its interactions")]
        return None, [Precondition(
            id="INPUTSET_METHOD_MISSING",
            message=f"no method set covers {ff.value}, the force field the topology includes; "
            "known: " + (", ".join(candidates) or "none"),
            suggestion="a method set for it (cutoffs, electrostatics, constraints, each with "
            "its source) goes in the site layer, ~/.config/myrmidon/inputsets/",
            needs=["method"])]
    return None, [Precondition(
        id="INPUTSET_METHOD_MISSING",
        message=f"{inputset.id} says what the run is for, not how its interactions are "
        "computed: name the method set — one of " + (", ".join(candidates) or "none")
        + (", or pass the topology, whose force field names it"
           if inputset.engine == "gromacs" else ""),
        suggestion="method = the method set the study uses", needs=["method", "topology"])]


def _facts(ctx: ToolContext, params: InputSetApplyParams, inputset, method) -> dict:
    """The keys the purpose reads from the system, read from the files the call names."""
    rules = {spec.from_system for layer in (inputset, method) if layer is not None
             for spec in layer.parameters.values() if spec.from_system}
    if not rules:
        return {}
    return read_facts(ctx, params.topology, params.index, params.structure, rules)


def read_facts(ctx: ToolContext, topology: str | None, index: str | None,
               structure: str | None, rules: set[str] | None = None) -> dict:
    """The facts the named system files hold, each naming the files it read as the study
    names them, so the record holds no machine path."""
    from myrmidon.engines.facts import data_dir_of, system_facts

    data = data_dir_of(ctx.engines)
    files = {"topology": _file(ctx, topology), "index": _file(ctx, index),
             "structure": _file(ctx, structure)}
    facts = system_facts({k: v for k, v in files.items() if v is not None}, data, rules)
    rel = ctx.workspace.relpath
    return {k: f.model_copy(update={"read": [rel(Path(x)) for x in f.read]})
            for k, f in facts.items()}


def _resolve_protocol(ctx: ToolContext, params: InputSetApplyParams):
    """(inputset, method, resolution, findings) — no resolution means do not render; the
    findings may also be advisory."""
    from myrmidon.knowledge.inputsets import resolve

    if params.stage and not params.inputset:
        return _resolve_stage(ctx, params)
    if not params.inputset:
        return None, None, None, [Precondition(
            id="INPUTSET_NOT_FOUND", message="name the input set, or the mandate stage",
            needs=["inputset", "stage"])]
    store = get_protocol_store(ctx)
    try:
        inputset, wanted_ver = store.get_versioned(params.inputset)
    except KeyError:
        known = ", ".join(sorted(p.id for p in store.all())) or "none"
        return None, None, None, [
            Precondition(
                id="INPUTSET_NOT_FOUND",
                message=f"no inputset {params.inputset!r} in any layer; known: {known}",
                suggestion="find one with inputset_lookup",
            )
        ]
    if wanted_ver is not None and str(inputset.version) != wanted_ver:
        return inputset, None, None, [
            Precondition(
                id="INPUTSET_OTHER_VERSION",
                message=f"requested {inputset.id} at v{wanted_ver}; this machine has "
                f"v{inputset.version}",
                suggestion=f"use {inputset.id}@v{inputset.version}",
                needs=["inputset"],
            )
        ]
    method, findings = _method_for(ctx, params, inputset, store)
    if any(f.severity == "block" for f in findings):
        return inputset, None, None, findings
    resolution = resolve(inputset, params.overrides, method=method,
                         facts=_facts(ctx, params, inputset, method))
    return inputset, method, resolution, findings


_RENDERERS = {suffix: renders for renders, (_, suffix) in RENDERED_FILES.items()}


def _resolve_stage(ctx: ToolContext, params: InputSetApplyParams):
    """(template, None, resolution, findings) for a mandate stage: the template of the
    engine the output's suffix names, compiled over the mandate and the system's facts."""
    from myrmidon.core.mandate import read_mandate
    from myrmidon.knowledge.stage import (
        MethodStore,
        TemplateStore,
        compile_stage,
        expected_keys,
        methods_for,
        templates_for,
        with_inherited,
    )

    mandate = read_mandate(ctx.workspace)
    stage = mandate.method.stage(params.stage) if mandate is not None else None
    if stage is None:
        names = ", ".join(s.name for s in mandate.method.stages) if mandate else ""
        return None, None, None, [Precondition(
            id="MANDATE_STAGE_UNKNOWN",
            message=f"the mandate names no stage {params.stage!r}"
            + (f"; its stages: {names}" if names else "; write the mandate's stages first"),
            suggestion="render a stage the mandate names, or name the input set for a run "
            "outside them", needs=["stage"])]
    renders = _RENDERERS.get(Path(params.output).suffix) if params.output else None
    store = TemplateStore.default()
    view = stage.model_dump(mode="json")
    found = [t for t in templates_for(view, store) if renders is None or t.renders == renders]
    if len(found) != 1:
        engines = ", ".join(sorted({t.engine for t in templates_for(view, store)})) or "none"
        return None, None, None, [Precondition(
            id="MANDATE_STAGE_TEMPLATE",
            message=f"stage {stage.name} ({stage.kind}, {stage.regime}) is told to: {engines}"
            + ("; name the file it renders: its suffix (.mdp, .inp, .dat) says which" if found
               else "; no template tells an engine this stage — run it with its own tool")
            + "; or omit stage to render a named input set outside the mandate",
            needs=["output"])]
    template = with_inherited(found[0], store)

    def system(kind: str, given: str | None) -> str | None:
        return given or system_file(ctx, mandate, kind)

    facts = read_facts(ctx, system("topology", params.topology), system("index", params.index),
                       params.structure)
    facts |= _starting_facts(ctx, mandate, stage, template)
    ff = facts.get("forcefield")
    methods = methods_for(mandate, MethodStore.default(), ff.value if ff else None)
    if methods.missing:
        return template, None, None, [Precondition(
            id="INPUTSET_METHOD_MISSING", message="; ".join(methods.missing),
            suggestion="a method for it goes in the site layer, ~/.config/myrmidon/methods/",
            needs=["topology"])]
    begins = mandate.method.starts_from(stage)
    stage, read, started = _ramp_from_run(ctx, stage, begins.name if begins else None, methods,
                                          template.ramp_hold_au)
    resolution = compile_stage(mandate, stage, template, templates=store, methods=methods,
                               facts=facts, overrides=params.overrides)
    if read is not None:  # the ramp's keys trace to the log they were read from
        for key in _from_run_keys(template, store):
            if key in resolution.values and resolution.origin.get(key) != "user":
                resolution.read[key] = read
    findings: list[Finding] = [Finding(id="STAGE_RAMP_FROM_RUN", message=started)] \
        if started else []
    if params.overrides:
        from myrmidon.core.lineage import same_words, satisfies

        locked, _ = expected_keys(mandate, stage.name, template.engine, templates=store,
                                  methods=methods, facts=facts)
        broken = [f"{k} = {resolution.values[k]} (\"{said}\")"
                  for k, (constraint, said, _) in locked.items()
                  if resolution.origin.get(k) == "user"
                  and not satisfies(resolution.values[k], constraint)
                  and not same_words(resolution.values[k], constraint)]
        if broken:
            findings.append(Finding(
                id="MANDATE_LOCK_OVERRIDDEN",
                message=f"stage {stage.name}: {', '.join(broken)} breaks what the researcher "
                "locked; a job on this file is refused at submission",
                suggestion="drop the override, or revise the mandate with the reason"))
    return template, None, resolution, findings


def _from_run_keys(template, store) -> set[str]:
    """The template's keys a render supplies from the run before (knowledge/stage.FROM_RUN)."""
    from myrmidon.knowledge.stage import FROM_RUN, flattened

    forms, _ = flattened(template, store)
    return {key for key, form in forms.items() if form.from_stage in FROM_RUN}


def system_file(ctx: ToolContext, mandate, kind: str, stage: str | None = None) -> str | None:
    """The study's file of a system kind (``core.lineage.system_file``)."""
    from myrmidon.core.lineage import system_file as study_file

    return study_file(ctx.workspace, mandate, kind, stage)


def _starting_facts(ctx: ToolContext, mandate, stage, template) -> dict:
    """What a render reads from the run its stage starts from: that its run left no state to
    restart from, and what the engine takes from the file that run's stage collected."""
    from myrmidon.core.lineage import stage_run
    from myrmidon.engines.registry import adapter
    from myrmidon.knowledge.catalog import SystemFact
    from myrmidon.knowledge.stage import FRESH_START

    begins = mandate.method.starts_from(stage)
    run = stage_run(ctx.workspace, begins.name) if begins is not None else None
    facts: dict = {}
    own_first = run is None and begins is not None and begins.name == stage.name
    if own_first or (run is not None and not any(a.role == "restart" for a in run[0])):
        facts[FRESH_START] = SystemFact(key=FRESH_START, value="yes", rule="starting_run")
    engine = adapter(template.engine)
    if engine.input_facts is not None:
        from myrmidon.core.lineage import latest_dataset

        dataset = latest_dataset(ctx.workspace, begins.name if begins is not None else None)
        if dataset is not None:
            facts |= engine.input_facts(ctx, ctx.workspace.resolve(dataset))
    return facts


def _ramp_from_run(ctx: ToolContext, stage, begins: str | None, methods,
                   hold_au: float | None = None):
    """(the stage with its ramp started where the stage it starts from ended, the record of
    what was read, what to say): the final ionic temperature the engine log of that stage's
    latest collected run states, read only when the log is the one collected. The stage as
    it is, with why, while that stage has not run."""
    from myrmidon.core.artifacts import sha256_file
    from myrmidon.core.lineage import stage_run
    from myrmidon.core.mandate import Ramp
    from myrmidon.engines.facts import run_facts
    from myrmidon.knowledge.stage import ramp_started

    ramp = stage.temperature_k
    if not isinstance(ramp, Ramp) or (ramp.start_k is not None
                                      and ramp.rate_k_per_au is not None):
        return stage, None, ""
    step = methods.step_ps(stage.regime)
    if ramp.start_k is not None:
        return ramp_started(stage, ramp.start_k, step, hold_au), None, ""
    run = stage_run(ctx.workspace, begins) if begins else None
    log = next((a for a in (run[0] if run else []) if a.role == "qm_log"), None)
    if run is None:
        return stage, None, (f"stage {stage.name} ramps from where {begins} ended, "
                             f"and no run of {begins} has been collected: run and "
                             f"collect it first, or state the ramp's start_k")
    if log is None:
        return stage, None, (f"{begins}'s collected run wrote no log to read "
                             "its final ionic temperature from: state the ramp's start_k")
    path = ctx.workspace.root / log.path
    if not path.is_file() or sha256_file(path) != log.sha256:
        return stage, None, (f"{log.path} is not the log {begins}'s collected run "
                             "wrote: the ramp's start is not read from it")
    start = run_facts(path.read_text(errors="replace")).get("final_temperature_k")
    if start is None:
        return stage, None, f"{log.path} states no final ionic temperature to ramp from"
    started = ramp_started(stage, start, step, hold_au)
    read = {"rule": f"final ionic temperature of {begins}'s run",
            "files": [log.path], "sha256": log.sha256}
    return started, read, (
        f"stage {stage.name} ramps from {start:g} K, where {begins} ended "
        f"({log.path}), to {ramp.to_k:g} K at {started.temperature_k.rate_k_per_au:g} K per "
        "a.u. at the method's time step"
        + (f", reaching it {hold_au:g} a.u. before the stage ends and holding it" if hold_au
           else ", reaching it at the stage's last step"))


def _render_kind(resolution, output: str | None) -> str | None:
    """What to render: the input set says so, else the output suffix does.

    `None` means nothing is rendered — either because no output was asked for (the values
    are the answer) or because the input set declares `renders: none`.
    """
    if output is None or resolution.renders == "none":
        return None
    kind = resolution.renders or _RENDERERS.get(Path(output).suffix)
    return kind if kind in RENDERED_FILES else None


def _sidecar_values(ctx: ToolContext) -> dict[str, dict]:
    """``stage:<name>/<renders>`` -> the values already rendered for that stage as that
    file in this workspace, read back from the sidecars: how a protocol's rule about two
    different files sees the other one. Two renders of one stage: the later path wins."""
    known: dict[str, dict] = {}
    root = getattr(ctx.workspace, "root", None)
    if root is None:
        return known
    for path in sorted(Path(root).rglob("*.myrmidon.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if data.get("stage") and "values" in data:  # a stage's render, as a step cites it
            known[f"stage:{data['stage']}/{data.get('renders')}"] = data["values"]
    return known


def _study_protocol(ctx: ToolContext):
    """The protocol the study's mandate names as its method, if it names one.

    A protocol carries the rules that relate values *different steps* render — the GROMACS
    output pace against the CPMD force-matching pace. They cannot live on an input set,
    which knows only its own parameter set, so without a protocol they are not checked.
    """
    from myrmidon.core.mandate import read_mandate
    from myrmidon.knowledge.protocols import ProtocolStore

    mandate = read_mandate(ctx.workspace) if ctx.workspace is not None else None
    if mandate is None or not mandate.method.protocol:
        return None
    store = ctx.protocols or ProtocolStore.default()
    try:
        return store.get_versioned(mandate.method.protocol)[0]
    except (KeyError, ValueError):  # unknown, or a layer's file that does not load
        return None


def _check_invariants(ctx: ToolContext, inputset, resolution, method=None,
                      stage: str | None = None):
    """The input set's or template's own rules, plus any the study's protocol declares
    across steps; ``stage`` names the mandate stage a template render is for."""
    from myrmidon.knowledge.inputsets import evaluate_invariants

    results = []
    own = [*inputset.invariants, *(method.invariants if method is not None else [])]
    if own:
        results += evaluate_invariants(own, {"": resolution.values})

    protocol = _study_protocol(ctx)
    if protocol is not None and protocol.invariants:
        rendered = _sidecar_values(ctx)
        if stage:  # the one about to be written
            rendered[f"stage:{stage}/{inputset.renders}"] = resolution.values
        results += protocol.check_invariants(rendered)
    return results


def inputset_apply_gate(ctx: ToolContext, params: InputSetApplyParams):
    warnings: list[Finding] = []
    checked = ["protocol_exists", "method"]
    inputset, method, resolution, findings = _resolve_protocol(ctx, params)
    if resolution is None:
        return findings, checked
    warnings.extend(findings)
    checked.append("required_parameters")
    # a ramp's keys come from the run it starts from: why they did not is what to act on
    ramp = next((f.message for f in findings if f.id == "STAGE_RAMP_FROM_RUN"), None)
    from_run_keys = set()
    if ramp is not None:
        from myrmidon.knowledge.stage import TemplateStore

        from_run_keys = _from_run_keys(inputset, TemplateStore.default())
    for key, how in resolution.missing.items():
        from_run = key in from_run_keys
        warnings.append(
            Precondition(
                id="INPUTSET_REQUIRED_PARAM_MISSING",
                message=f"{inputset.id} does not supply {key!r} and it was not given: "
                f"{how}",
                assumption=f"rendering without {key} would leave the engine to default it",
                suggestion=ramp if from_run else f"confirm {key} and pass it in overrides",
            )
        )
    checked.append("renderer")
    kind = _render_kind(resolution, params.output)
    if kind is None and params.output is not None:
        # An input set that renders no file cannot be written into one, and the message says
        # what to do instead rather than what is missing.
        if resolution.renders == "none":
            message = (
                f"{inputset.id} renders no file — it states values. Call inputset_apply "
                "without `output` to resolve them, with their origins and sources."
            )
        else:
            message = (
                "inputset_apply renders .mdp, CPMD input and fit input files; declare the "
                "inputset's `renders` or give the output an .mdp/.inp/.dat suffix."
            )
        warnings.append(
            Precondition(
                id="INPUTSET_RENDERER_UNSUPPORTED", message=message
            )
        )
    checked.append("invariants")
    for result in _check_invariants(ctx, inputset, resolution, method,
                                    params.stage if not params.inputset else None):
        if result.status == "violated":
            warnings.append(
                Precondition(
                    id=result.id,
                    message=f"{result.message} ({result.detail})",
                )
            )
        elif result.status == "deferred":
            warnings.append(
                Finding(
                    id=result.id,
                    message=f"{result.message} — cannot be checked yet: {result.detail}",
                )
            )
    checked.append("overrides")
    deviations = resolution.overrides
    if deviations:
        detail = "; ".join(f"{k}: {v['inputset']} -> {v['user']}" for k, v in deviations.items())
        warnings.append(
            Finding(
                id="INPUTSET_VALUE_OVERRIDDEN",
                message=f"{len(deviations)} validated value(s) of {inputset.id} replaced: "
                + detail,
                assumption="the inputset's validated value would be silently replaced",
                suggestion="state the reason in the journal",
            )
        )
    return warnings, checked


#: the artifact kind each rendered file is
_RENDERED_KIND = {renders: kind for renders, (kind, _) in RENDERED_FILES.items()}


def _rendered_kind(ctx, params: dict) -> str | None:
    """The kind of file the call would render; None when nothing would be written or the
    set is unknown."""
    from myrmidon.knowledge.inputsets import resolve

    if params.get("stage") and not params.get("inputset"):
        output = params.get("output")
        renders = _RENDERERS.get(Path(output).suffix) if output else None
        return _RENDERED_KIND.get(renders) if renders else None
    try:
        inputset = get_protocol_store(ctx).get(str(params.get("inputset", "")))
    except (KeyError, AttributeError):
        return None
    kind = _render_kind(resolve(inputset, dict(params.get("overrides") or {})),
                        params.get("output"))
    return _RENDERED_KIND.get(kind) if kind else None


@tool(
    "inputset_apply",
    description="Render a run-parameter file, recording where every value came from (the set, "
                "the system, or you); a stage's input complete, for a run outside the stages an "
                "input set. stage_prepare renders every file of a stage in one call. Never write "
                "an .mdp by hand.",
    tags={"knowledge", "inputsets", "files"},
    contract=Contract(produces=(Produces(kind_of=_rendered_kind),)),
)
def inputset_apply(ctx: ToolContext, params: InputSetApplyParams) -> Observation:
    precheck(ctx, inputset_apply_gate(ctx, params)[0])
    from myrmidon.core.mandate import read_mandate
    from myrmidon.engines.registry import StageRender, adapter_rendering

    inputset, method, resolution, findings = _resolve_protocol(ctx, params)
    if resolution is None:  # the gate already refused; belt and braces
        return Observation(ok=False, summary=findings[0].message)
    method_tag = f"{method.id}@v{method.version}" if method is not None else None

    root = ctx.workspace.root
    rendered_stage = bool(params.stage and not params.inputset)
    origins = list(resolution.origin.values())
    from_system = sum(1 for v in origins if v.startswith("system:"))
    from_user = sum(1 for v in origins if v == "user")
    # the rest is the knowledge's: an input set's, or a stage's template, method and terms
    from_protocol = len(origins) - from_system - from_user
    source = ("the mandate's stage, its templates and method" if rendered_stage
              else "the input sets")
    if params.output is None:
        # Resolve only: the values with their origins are the answer, and nothing is
        # written. An input set whose `renders` is "none" has no other way to be read —
        # a box clearance or a QM padding is a value a tool takes as a parameter, not a
        # line in a file — and reading one is still a call the ledger records.
        invariants = _check_invariants(ctx, inputset, resolution, method,
                                       params.stage if rendered_stage else None)
        return Observation(
            ok=True,
            summary=f"{inputset.id} v{inputset.version} resolves to "
            f"{len(resolution.values)} value(s), {from_protocol} from {source}, "
            f"{from_system} read from the system, {from_user} from you; nothing written.",
            pitfalls=inputset.pitfalls,
            inputset=inputset.id,
            protocol_version=inputset.version,
            method=method_tag,
            renders=inputset.renders,
            invariants=[r.model_dump() for r in invariants],
            values=resolution.values,
            origin=resolution.origin,
            owner=resolution.owner,
            overrides=resolution.overrides,
            additions=resolution.additions,
            sources=[InputSetSource(**s.model_dump()) for s in resolution.sources],
        )

    target = (root / params.output).resolve()
    if not str(target).startswith(str(root) + "/"):
        return Observation(
            ok=False, summary=f"refused: '{params.output}' escapes the workspace"
        )
    link = symlink_component(root, params.output)
    materialized = None
    if link is not None:
        if not params.materialize:
            return Observation(
                ok=False,
                summary=f"refused: '{params.output}' goes through the symlink "
                f"{ctx.workspace.relpath(link)} -> {link.resolve()}; writing would change "
                "the shared target. materialize=true replaces the link with a private copy.",
            )
        materialized = materialize_symlink(link)
        target = (root / params.output).resolve()

    label = f"{inputset.id} v{inputset.version}" + (
        f" with {method.id} v{method.version}" if method is not None else "") + (
        f" for stage {params.stage}" if rendered_stage else "")
    header = [
        "written by myrmidon inputset_apply — do not edit by hand",
        f"inputset: {label}",
        "each value names its origin; the machine-readable map is in "
        f"{target.name}.myrmidon.json",
    ]
    kind = _render_kind(resolution, params.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    # an input set supplies the method, never the system: a stage's file takes the rest from
    # what the system holds, as its engine says
    engine = adapter_rendering(kind)
    mandate = read_mandate(ctx.workspace) if rendered_stage else None
    extra: dict = {}
    if mandate is not None and engine.stage_text is not None:
        files = {k: ctx.workspace.resolve(path) for k in ("cpmd_input",)
                 if (path := system_file(ctx, mandate, k, params.stage))}
        text = engine.stage_text(StageRender(
            mandate.method.stage(params.stage), resolution.values, resolution.origin, header,
            files))
        target.write_text(text.text)
        extra = text.sidecar
        findings = [*findings, *text.findings]
    else:
        target.write_text(engine.render(resolution.values, resolution.origin, header))

    invariants = _check_invariants(ctx, inputset, resolution, method,
                                   params.stage if rendered_stage else None)
    sidecar = target.with_name(target.name + ".myrmidon.json")
    sidecar.write_text(
        json.dumps(
            {
                "inputset": inputset.id,
                "protocol_version": inputset.version,
                "stage": params.stage if rendered_stage else None,
                "method": method_tag or resolution.method,
                "renders": kind,
                "values": resolution.values,
                "origin": resolution.origin,
                "owner": resolution.owner,
                "overridden": resolution.overridden,
                "read": resolution.read,
                "engine_defaults": resolution.engine_defaults,
                "overrides": resolution.overrides,
                "invariants": [r.model_dump() for r in invariants],
                "sources": [s.model_dump(exclude_none=True) for s in resolution.sources],
                "written_by": "inputset_apply",
                **extra,
            },
            indent=1,
            sort_keys=True,
        )
    )

    note = (
        f" (replaced symlink {ctx.workspace.relpath(link)} with a private copy of "
        f"{materialized})" if materialized else ""
    )
    # An input set names the failures that shaped it, so they arrive when the recipe is
    # chosen rather than after the run has crashed. Each is a finding id a gate or
    # diagnostic raises with its full text when it fires.
    traps = (
        f" Known ways this recipe fails: {', '.join(inputset.pitfalls)}."
        if inputset.pitfalls else ""
    )
    # how the run starts, read from the structure it starts from: said, so nobody counts
    # the file's columns to learn it
    key = next((k for k in resolution.read if canonical(k) == "continuation"
                and str(resolution.origin.get(k, "")).startswith("system:")), None)
    begins = ""
    if key is not None:
        start = resolution.read[key]
        moving = str(resolution.values.get(key, "")).lower() == "yes"
        where = ", ".join(start.get("files") or []) or "the structure"
        begins = (f" The run continues from the velocities {where} carries." if moving else
                  f" {where} carries no velocities: the run draws them at its temperature.")
    return Observation(
        ok=True,
        summary=f"Wrote {ctx.workspace.relpath(target)} from {label}: "
        f"{from_protocol} value(s) from {source}, {from_system} read from the system, "
        f"{from_user} from you{note}.{begins}{traps}",
        pitfalls=inputset.pitfalls,
        inputset=inputset.id,
        protocol_version=inputset.version,
        method=method_tag,
        renders=kind,
        invariants=[r.model_dump() for r in invariants],
        values=resolution.values,
        origin=resolution.origin,
        owner=resolution.owner,
        overrides=resolution.overrides,
        additions=resolution.additions,
        sources=[InputSetSource(**s.model_dump()) for s in resolution.sources],
        sidecar=ctx.workspace.relpath(sidecar),
        findings=[f for f in findings if f.severity != "block"],
        artifacts=[
            ArtifactRef.from_path(target, kind, root=root),
            ArtifactRef.from_path(sidecar, "provenance", root=root),
        ],
        # the rendered file's parent in the ledger: the input set's bytes it came from
        refs=[inputset_ref(inputset.id, inputset.version, inputset.sha256),
              *([f"stage:{params.stage}"] if rendered_stage else [])],
        # where each value came from and why is in the sidecar beside the file
        quiet={"origin", "owner", "sources"},
    )
