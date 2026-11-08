"""The mandate tools: write the study's mandate for the researcher, and show it.

There is no approve tool: a `ready` return is the signal. The model tells the researcher the
stages and their cost in core-hours and ends its turn; the researcher approves with
`/approve` in Hermes, the dialog in Pi, or `myrmidon approve` at a terminal
(core/mandate.py records it).
"""

from __future__ import annotations

from difflib import get_close_matches

from pydantic import BaseModel, Field, ValidationError

from myrmidon.core import mandate as mandates
from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation, ToolError
from myrmidon.core.tool import REGISTRY, ToolContext, books, tool
from myrmidon.core.validation import Finding
from myrmidon.execution.jobs import booked_under_mandates

#: what the turn does after a ready mandate, in every client
APPROVAL_FORMS = ("tell the researcher its stages, what they measure and the estimated "
                  "core-hours against the ceiling, and that their approval opens the runs; "
                  "then end the turn: they approve with `/approve` (Hermes), in the approval "
                  "dialog (Pi), or with `myrmidon approve` at a terminal")


class SystemFileIn(BaseModel):
    path: str = Field(description="a file that is the system, as the study resolves it")
    kind: str = Field(description="its artifact kind: structure, topology, index, "
                      "qm_selection, cpmd_restart, ...")


class MandateWriteParams(BaseModel):
    question: str = Field(default="", description="the researcher's question, verbatim; a "
                          "revision keeps the revised mandate's")
    system_description: str = Field(default="", description="what will be simulated, in "
                                    "words: molecules, force field, solvent, box, size")
    system_files: list[SystemFileIn] = Field(default_factory=list, description="the files "
                                             "that are the system, built in this session")
    subsystems: dict[str, mandates.Subsystem] = Field(
        default_factory=dict, description="the system's named parts, as the topology names "
        "them: name -> {kind: protein|ligand|nucleic|membrane|cofactor|solvent|ions, "
        "molecules: [moleculetype names], resnames, chains, resid, charge}; resid and chains "
        "only to pick part of a molecule, as the built structure numbers it, never the "
        "deposit. Left out, they are read from the topology. A stage's restraints and "
        "qm_region and a deliverable's target name them")
    hamiltonian: mandates.Hamiltonian | None = Field(
        default=None, description="how the energy is computed: mm (a method id; read from "
        "the topology's force field when omitted), qm {method, cutoff_ry}, coupling "
        "{scheme, multipole_order, cutoff_bohr}")
    protocol: str = Field(default="", description="the published sequence followed, as "
                          "<id>@v<N>")
    stages: list[mandates.Stage] = Field(
        default_factory=list, description="what physically happens, stage by stage, in "
        "order: {name, kind: minimize|dynamics|single_point|parameter_fit, regime: "
        "mm|qmmm|qm, starts_from: <stage it continues, default the one before>, ensemble: "
        "NVE|NVT|NPT, temperature_k, pressure_bar, duration_ps or "
        "steps, restraints: [subsystems held], sampling_ps, qm_region {subsystems, charge, "
        "multiplicity, padding_nm}, scan {parameter, values}, converge}. Each stage's "
        "engine file is compiled from these; stage_prepare stage=<name> writes them")
    decisions_yours: list[mandates.Decision] = Field(
        default_factory=list, description="what the researcher decided in this "
        "conversation: their words in `said`, and in `locks` the stage fields it fixes by "
        "path: {\"production.duration_ps\": 1000, \"nvt.restraints\": [protein, ligand], "
        "\"*.temperature_k\": 300}; a value, a list (one of) or {min, max}. A choice the "
        "system files, the ceiling or the approval already hold (force field, budget, 'ask "
        "me first') keeps its `said` with no locks. `keys` on `target` locks a booking "
        "tool's parameter or the backend's")
    decisions_proposed: list[mandates.Proposal] = Field(
        default_factory=list, description="what you propose on their behalf, each with its "
        "source")
    deliverables: list[mandates.Deliverable] = Field(
        default_factory=list, description="what the report will contain, in physical "
        "terms: {name, property: rmsd|rmsf|radius_of_gyration|min_distance|contacts|"
        "pair_distance|thermodynamics|energy_drift|dipole|step_timing|qm_forces|force_rmse|"
        "boundary_bonds, target: <subsystem>, reference: <subsystem> for a relation, atoms: "
        "heavy|backbone|ca|all (on every side alike), align_to: {subsystem, atoms}, "
        "reference_frame: stage_start|minimised|<path>, over: <stage>, statistics: "
        "[mean, max, min, sd, final]}. The harness compiles the call that measures it")
    ceiling_core_h: float | None = Field(default=None, description="the most the researcher "
                                         "lets the study book: theirs to state")
    reruns: int = Field(default=1, ge=0, description="identical reruns of a failed job the "
                        "harness may make on its own")
    model_fixes: int = Field(default=2, ge=0, description="changes outside the researcher's "
                             "decisions the model may make on its own after a failure")
    deadline_h: float | None = Field(default=None, description="hours after which the "
                                     "harness stops and asks")
    questions: list[str] = Field(default_factory=list, max_length=mandates.MAX_QUESTIONS,
                                 description="what only the researcher can still answer "
                                 "(at most 3); the mandate waits on them")
    revise: bool = Field(default=False, description="replace an approved mandate with a new "
                         "revision, which the researcher approves again; every field the "
                         "call leaves out but questions and reason is the revised mandate's "
                         "(new system files: the subsystems and Hamiltonian are read again)")
    reason: str = Field(default="", description="why the approved mandate is revised")


def _unknown_params(deliverables: list[mandates.Deliverable]) -> list[str]:
    """Each deliverable param its tool does not take, as "name: tool takes no key"."""
    out = []
    for d in deliverables:
        tool = (d.tool.split() or [""])[0]
        if not d.params or tool not in {s.name for s in REGISTRY.specs()}:
            continue
        fields = REGISTRY.get(tool).params_model.model_fields
        out += [f"{d.name}: {tool} takes no {key!r}" for key in d.params if key not in fields]
    return out


def _unread_targets(ctx: ToolContext, decisions: list[mandates.Decision]
                    ) -> list[Finding]:
    """A decision whose target names no input set, no tool that books, no backend and is
    not ``*``, or a key nothing on its target reads — the tool's or the booking's
    parameters, the engine keys of the set's engine, any of these for ``*`` — is checked
    against nothing at submission: the researcher would approve a lock that holds nothing."""
    from myrmidon.execution.backend import Resources
    from myrmidon.knowledge.catalog import load
    from myrmidon.knowledge.inputsets import ENGINE_OF_RENDERS, engine_key
    from myrmidon.knowledge.stage import TemplateStore

    booking = {spec.name: spec for spec in REGISTRY.specs() if books(spec)}
    backends = {"slurm", "local", str(getattr(ctx.config, "backend", ""))}
    known = {"*", *backends, *booking}
    sets = {s.id: s for s in ctx.inputsets.all()} if ctx.inputsets is not None else {}
    # a template a stage is told with renders files the gate reads by its id, as a purpose
    # set's are; one only extended (kinds: []) is never what a render cites
    sets |= {t.id: t for t in TemplateStore.default().all() if t.kinds and t.id not in sets}
    known |= set(sets)
    catalogs = {}
    for engine in ("gromacs", "cpmd"):
        try:
            catalogs[engine] = load(engine)
        except Exception:  # noqa: BLE001 — an engine with no shipped catalog reads nothing
            continue

    def known_to(catalog, key: str) -> bool:
        # the gate reads a CPMD keyword by its &SECTION/KEYWORD LINE spelling only
        if catalog.engine == "cpmd" and not key.startswith("&"):
            return False
        return engine_key(catalog, key) is not None

    def read_on(target: str, key: str) -> bool:
        if target in booking:
            return key in booking[target].params_model.model_fields
        if target in backends:
            return key in Resources.model_fields
        if target in sets:
            inputset = sets[target]
            # the gate reads a rendered file by the purpose set it came from: a method set
            # and a set that renders nothing are read by nothing at submission
            if getattr(inputset, "axis", None) == "method" or inputset.renders == "none":
                return False
            catalog = catalogs.get(inputset.engine
                                   or ENGINE_OF_RENDERS.get(inputset.renders or "") or "")
            return key in inputset.parameters or (
                catalog is not None and known_to(catalog, key))
        return (any(read_on(t, key) for t in (*booking, *backends))
                or any(known_to(c, key) for c in catalogs.values()))

    def readable(target: str) -> list[str]:
        """The keys a target reads, for the nearest-spelling suggestion."""
        if target in booking:
            return list(booking[target].params_model.model_fields)
        if target in backends:
            return list(Resources.model_fields)
        if target in sets:
            inputset = sets[target]
            catalog = catalogs.get(inputset.engine
                                   or ENGINE_OF_RENDERS.get(inputset.renders or "") or "")
            return [*inputset.parameters, *(catalog.keys if catalog is not None else [])]
        return []

    out = []
    for d in decisions:
        if d.target.split("@")[0] not in known:
            out.append(f"the decision \"{d.said}\" targets {d.target!r}, which no submission "
                       "reads: name a template a stage is told with (not one only "
                       "extended) or an input set id, a tool that books a job, the backend, "
                       "or *. A choice the system files, the ceiling or the approval already "
                       "hold keeps its words with no keys")
            continue
        target = d.target.split("@")[0]
        unread = sorted(k for k in d.keys if not read_on(target, k))
        if not unread:
            continue
        inputset = sets.get(target)
        if inputset is not None and getattr(inputset, "axis", None) == "method":
            why = "a method set's values are read through the purpose set rendered with it: " \
                  "lock them on that purpose set"
        elif inputset is not None and inputset.renders == "none":
            why = f"{target} renders no file: lock the values on the tool that takes them"
        else:
            near = [] if target == "*" else sorted({m for k in unread for m in
                                                     get_close_matches(k, readable(target),
                                                                       n=3, cutoff=0.6)})
            why = f"{target} reads no such key" + (
                f" (nearest it reads: {', '.join(near)})" if near else "")
        out.append(f"the decision \"{d.said}\" locks {', '.join(unread)} on {target}: "
                   f"{why}. A choice the system files, the ceiling or the approval already "
                   "hold keeps its words with no keys")
    return [Finding(id="MANDATE_TARGET_UNREAD", message=m)
            for m in out]


def open_chemistry(ws, stages: list, decisions: list, selected_for: str | None = None
                   ) -> list[str]:
    """What the latest QM partition (for a stage, if named) left open of the region's chemistry
    and no stage or decision settles, as the engine that partitions a region asks it."""
    from myrmidon.engines.registry import adapter_with

    engine = adapter_with("region_questions")
    return engine.region_questions(ws, stages, decisions, selected_for) if engine else []


def _selection_files(ws, stated: set[str]) -> list[mandates.SystemFile]:
    """What the QM selection wrote is the system too, for a stage with a QM region: the kinds
    the call does not state, as the selection left them."""
    from myrmidon.core.lineage import latest_outputs

    wrote = latest_outputs(ws, "qm_region")
    return [mandates.SystemFile(path=wrote[kind], kind=kind)
            for kind in ("cpmd_input", "index", "qm_selection")
            if kind in wrote and kind not in stated]


def _spent(ws) -> float:
    return booked_under_mandates(ws, mandates.approved_digests(ws))


def compiled(ctx: ToolContext, mandate: mandates.Mandate, facts: dict | None = None):
    """The mandate's stages compiled and priced (knowledge/stage.py) over the facts its
    system files hold (read here unless given) and the paces measured on this machine."""
    from myrmidon.engines.structure import atom_count
    from myrmidon.execution.gate import mandate_facts
    from myrmidon.execution.pace import read_paces, study_paces
    from myrmidon.knowledge.stage import (
        MandateCompiled,
        MethodStore,
        TemplateStore,
        compile_mandate,
        methods_for,
    )

    if not mandate.method.stages and not mandate.deliverables:
        return MandateCompiled()
    ws = ctx.workspace
    facts = mandate_facts(ctx, mandate) if facts is None else facts
    ff = facts.get("forcefield")
    structure = next((f for f in mandate.system.files if f.kind == "structure"), None)
    return compile_mandate(
        mandate, templates=TemplateStore.default(),
        methods=methods_for(mandate, MethodStore.default(), ff.value if ff else None),
        facts=facts, atoms=atom_count(ws.resolve(structure.path)) if structure else None,
        paces=[*study_paces(ws), *read_paces(ctx.config.studies_root)],
        compute=ctx.config.compute, invariants=_protocol_invariants(ctx, mandate))


def _region_charges(mandate: mandates.Mandate, parts: dict[str, dict]) -> list[str]:
    """Each coupled stage's QM region charge the stage leaves unstated, as the sum of its
    subsystems' topology charges (a subsystem's own, else that of the parts of the topology
    it names by molecule type or residue), proposed with the topology as its source; the
    questions for a region whose sum is unknown or no integer (a region that cuts a
    molecule's charge distribution)."""
    subs = mandate.system.subsystems
    topology = next((f.path for f in mandate.system.files if f.kind == "topology"), "")

    def one(sub) -> float | None:
        """The subsystem's charge in the topology as it is now, else the one it states; a
        subsystem of some copies of its residues, the one it states alone."""
        laid = [p for p in parts.values()
                if set(p.get("molecules") or ()) & set(sub.molecules)
                or set(p.get("resnames") or ()) & set(sub.resnames)]
        if sub.resid or sub.chains:
            return sub.charge  # some copies of its residues: the topology's sum is not theirs
        if laid and all(p.get("charge") is not None for p in laid):
            return sum(p["charge"] for p in laid)
        return sub.charge

    def charge(names: list[str]) -> float | None:
        """The charge of the subsystems a region names, each once: a subsystem by its name, or
        every subsystem of that kind, as the region's selection resolves the names."""
        found = {name: ([(name, subs[name])] if name in subs else
                        [(k, s) for k, s in subs.items() if s.kind == name]) for name in names}
        if not all(found.values()):
            return None  # a name no subsystem answers to: the question is asked
        matched = {key: sub for pairs in found.values() for key, sub in pairs}
        charges = [one(sub) for sub in matched.values()]
        return sum(charges) if charges and None not in charges else None

    # a charge this derivation proposed and the researcher has not confirmed is derived again,
    # from the topology as it is now: a revision keeps no charge an older topology gave
    proposed = {path: value for p in mandate.decisions.proposed for path, value in p.locks.items()
                if path.endswith(".qm_region.charge")}
    asked = []
    for stage in mandate.method.stages:
        region = stage.qm_region
        path = f"{stage.name}.qm_region.charge"
        if region is None:
            continue
        locked = mandates.charge_lock(mandate.decisions.yours, stage.name)
        if path in proposed:
            # a proposal is read again when the stage still holds it, and gone when the stage
            # states a charge of its own or the researcher decides one
            if proposed[path] == region.charge or locked is not None:
                region.charge = None
            mandate.decisions.proposed = [p for p in mandate.decisions.proposed
                                          if path not in p.locks]
        if region.charge is None and isinstance(locked, int):
            region.charge = locked  # the researcher's
        if region.charge is not None:
            continue
        total = charge(list(region.subsystems))
        if total is None or abs(total - round(total)) > 1e-3:
            asked.append(f"What is stage {stage.name}'s QM region charge? (its subsystems' "
                         "topology charges " + ("sum to no integer" if total is not None
                                                else "are not all read") + ")")
            continue
        region.charge = round(total)
        mandate.decisions.proposed.append(mandates.Proposal(
            what=f"stage {stage.name}'s QM region carries charge {region.charge}, the sum of "
            f"the topology charges of {', '.join(region.subsystems)}",
            locks={f"{stage.name}.qm_region.charge": region.charge}, source=topology))
    return asked


def _protocol_invariants(ctx: ToolContext, mandate: mandates.Mandate) -> list:
    from myrmidon.tools.inputsets import protocol_of

    protocol = protocol_of(ctx, mandate)
    return list(protocol.invariants) if protocol is not None else []


def _from_protocol(ctx: ToolContext, params: MandateWriteParams) -> MandateWriteParams | None:
    """The call with what its protocol states as its defaults: its stages and their
    deliverables, and its Hamiltonian, where the call states none of its own; the
    researcher's edits are the mandate's. None when this machine has no protocol of that
    name."""
    from myrmidon.knowledge.protocols import ProtocolStore

    if not params.protocol:
        return params
    store = ctx.protocols or ProtocolStore.default()
    try:
        protocol = store.get_versioned(params.protocol)[0]
    except (KeyError, ValueError):
        return None
    update: dict = {"protocol": protocol.ref}
    if not params.stages:  # its deliverables are measured over its own stages
        update["stages"] = [s.model_copy(deep=True) for s in protocol.stages]
        if not params.deliverables and "deliverables" not in params.model_fields_set:
            update["deliverables"] = [d.model_copy(deep=True) for d in protocol.deliverables]
    if params.hamiltonian is None and protocol.hamiltonian is not None:
        update["hamiltonian"] = protocol.hamiltonian.model_copy(deep=True)
    return params.model_copy(update=update)


def _topology_parts(ctx: ToolContext, mandate: mandates.Mandate) -> dict[str, dict]:
    """The system's parts as its topology lays them out (engines/facts.system_parts)."""
    from myrmidon.engines.facts import data_dir_of, system_parts

    files = {f.kind: ctx.workspace.resolve(f.path) for f in mandate.system.files
             if ctx.workspace.resolve(f.path).is_file()}
    return system_parts(files, data_dir_of(ctx.engines))


def _read_system(ctx: ToolContext, mandate: mandates.Mandate, params: MandateWriteParams,
                 facts: dict, parts: dict[str, dict]) -> str:
    """What the mandate leaves unstated of the system, read from its files into it: the
    subsystems from the topology, the force field's method from the topology's include.
    Returns the sentence that asks the model to confirm them; "" when nothing was read."""
    from myrmidon.knowledge.stage import MethodStore, methods_for

    said = []
    system = mandate.system
    if not params.subsystems and parts:
        system.subsystems = {k: mandates.Subsystem.model_validate(v)
                             for k, v in parts.items()}
        said.append("subsystems " + ", ".join(
            f"{k} ({v.kind}: {', '.join(v.names())})"
            for k, v in system.subsystems.items()))
    ff = facts.get("forcefield")
    found = methods_for(mandate, MethodStore.default(), ff.value if ff else None)
    if "mm" in found.specs and system.hamiltonian.mm != found.specs["mm"].id:
        # left out, or named by the force field rather than the method that computes it
        named = system.hamiltonian.mm or (ff.value if ff else "")
        system.hamiltonian.mm = found.specs["mm"].id
        said.append(f"force field {named}, computed as {system.hamiltonian.mm}")
    return ("Read from the topology: " + "; ".join(said) + ". Confirm them, or write them "
            "as they are.") if said else ""


# the kinds a deposit numbers apart from its chains, and a build renumbers but never renames
NUMBERED_APART = {"ligand", "cofactor", "ions"}


def _subsystem_reading(ctx: ToolContext, mandate: mandates.Mandate, parts: dict[str, dict]
                       ) -> tuple[list[str], list[str]]:
    """(what keeps a deliverable from being measured, what the model should know): each
    subsystem's selection read on the system's own structure. A ligand's, cofactor's or ion's
    residue numbers that select nothing where its names (on its chains) select one residue
    are that residue's in another numbering (a deposit's, carried into a built structure):
    they are dropped, and said. A polymer's are not, since a build renames its residues. One
    a deliverable measures that still selects no atom cannot be measured; residue numbers or
    chains that select fewer atoms than the subsystem's names do are said — a pocket means
    to, a range written as two numbers does not. A chain the file does not carry is dropped
    and the rest judged; a molecule type, which only the run's .tpr carries, is left to the
    measurement."""
    from difflib import get_close_matches

    from myrmidon.analysis.trajectory import selected_atoms
    from myrmidon.knowledge.observables import subsystem_selection

    subs = mandate.system.subsystems
    measured = {name for d in mandate.deliverables
                for name in (d.target, d.reference, d.align_to.subsystem if d.align_to else "")
                if name}
    # the subsystems something reads: a deliverable's sides, a stage's restraints or QM region
    read = measured | {name for s in mandate.method.stages
                       for name in (*s.restraints,
                                    *(s.qm_region.subsystems if s.qm_region else ()))}
    problems: list[str] = []
    notes: list[str] = []
    # a molecule type the topology does not lay out is named nowhere a stage reads it
    laid_out = sorted({m for part in parts.values() for m in part["molecules"]})
    for name, sub in subs.items():
        unknown = [m for m in sub.molecules if laid_out and m not in laid_out]
        if unknown:
            near = sorted({n for m in unknown for n in get_close_matches(m, laid_out, n=1)})
            said = (f"subsystem {name} names the molecule type(s) {', '.join(unknown)}, which "
                    f"the topology's [ molecules ] does not list (it lists "
                    f"{', '.join(laid_out)}" + (f"; nearest: {', '.join(near)}" if near else "")
                    + ")")
            (problems if name in read else notes).append(said)
    structure = next((f for f in mandate.system.files if f.kind == "structure"), None)
    if structure is None or not subs:
        return problems, notes
    # each subsystem as stated, without its chains (which a structure file may not carry,
    # while the run's .tpr does), without its numbers, and by its names alone
    drops = ({}, {"chains": []}, {"resid": []}, {"resid": [], "chains": []})
    rows = [(name, [subsystem_selection(sub.model_copy(update=drop), mandate)
                    for drop in drops]) for name, sub in subs.items()]
    counts = selected_atoms(ctx.workspace.resolve(structure.path),
                            [s for _, sels in rows for s in sels])
    if not counts:
        return problems, notes
    for i, (name, (full, unchained, unnumbered, named)) in enumerate(rows):
        c_full, c_unchained, c_unnumbered, c_named = counts[4 * i:4 * i + 4]
        # judged as far as the file can: a chain it does not carry is left to the .tpr
        (sel, c), (bare, c_bare) = ((full, c_full), (unnumbered, c_unnumbered)) \
            if c_full is not None else ((unchained, c_unchained), (named, c_named))
        if (c and c[0] == 0 and subs[name].resid and subs[name].kind in NUMBERED_APART
                and c_bare and c_bare[1] == 1):
            notes.append(f"subsystem {name}: residue {subs[name].resid} selects nothing in "
                         f"{structure.path}, where {bare!r} is one residue; that residue "
                         "is read as the one meant, by its name")
            subs[name] = subs[name].model_copy(update={"resid": []})
            sel, c = bare, c_bare
        n, n_named = c and c[0], c_named and c_named[0]
        if n == 0 and name in measured:
            problems.append(f"subsystem {name} ({sel!r}) selects no atom of "
                            f"{structure.path}, so a deliverable on it measures nothing: "
                            "state it as the built structure numbers and names it")
        elif n is not None and n_named is not None and n != n_named:
            notes.append(f"subsystem {name}: {sel!r} selects {n} of the {n_named} atoms "
                         f"{named!r} selects in {structure.path}, which numbers residues as "
                         "the system was built; a range is written \"first-last\"")
    return problems, notes


def _unbound(said: list[str]) -> Observation:
    """The refusal of a mandate whose parts do not hold together: each reason a finding."""
    return Observation(
        ok=False, summary="the mandate does not hold together as written: " + " ".join(said),
        findings=[Finding(id="MANDATE_STAGE", message=m) for m in said],
        error=ToolError(code="MANDATE_STAGE", message=said[0],
                        hint="make the stages, the subsystems and the locks agree, or build "
                        "what the system lacks (a restrained subsystem's position "
                        "restraints) before writing the mandate"))


def _observe(ctx: ToolContext, mandate: mandates.Mandate, summary: str) -> Observation:
    ws = ctx.workspace
    held = mandates.holding(ws)
    spent = _spent(ws)
    comp = compiled(ctx, mandate)
    return Observation(
        ok=True, summary=summary, path=mandates.MANDATE_FILE, status=mandate.status,
        digest=mandate.digest, revision=mandate.revision, questions=mandate.questions,
        estimate_core_h=mandate.cost.estimate_core_h,
        stage_prices={name: p.model_dump() for name, p in comp.prices.items()},
        ceiling_core_h=mandate.cost.ceiling_core_h, spent_core_h=spent,
        approved=held is not None and held.digest == mandate.digest,
        why_not=mandates.why_not_holding(ws),
        text=mandates.render(mandate, spent, compiled=comp.lines, priced=comp.priced,
                             measured=comp.measured, detail=True))


def _carried(params: MandateWriteParams, held: mandates.Mandate) -> MandateWriteParams:
    """A revision: the fields the call states, and the held mandate's for the rest; the
    parts read from the system files only while the files stay."""
    s = held.system
    stated = {k for k in params.model_fields_set if k != "question" or params.question.strip()}
    held_fields = {
        "question": held.question, "system_description": s.description,
        "system_files": [SystemFileIn(path=f.path, kind=f.kind) for f in s.files],
        **({} if "system_files" in stated else {"subsystems": s.subsystems,
                                                 "hamiltonian": s.hamiltonian}),
        "protocol": held.method.protocol, "stages": held.method.stages,
        "decisions_yours": held.decisions.yours, "decisions_proposed": held.decisions.proposed,
        "deliverables": held.deliverables, "ceiling_core_h": held.cost.ceiling_core_h or None,
        "reruns": held.recovery.reruns, "model_fixes": held.recovery.model_fixes,
        "deadline_h": held.recovery.deadline_h,
    }
    return params.model_copy(update={k: v for k, v in held_fields.items() if k not in stated})


#: what a lock may name that no stage holds, and what holds it instead
HELD_ELSEWHERE = {"ceiling_core_h": "ceiling_core_h, which the researcher approves,",
                  "cost": "ceiling_core_h, which the researcher approves,",
                  "subsystems": "the system's subsystems", "system": "the system",
                  "hamiltonian": "the system's Hamiltonian",
                  "deliverables": "the deliverables"}


def _held_elsewhere_dropped(mandate: mandates.Mandate) -> list[str]:
    """Drops each lock on a field of the mandate outside its stages (a path whose head is
    no stage's name), keeping the decision's words, and says so: the approval or the
    system already holds what it names."""
    stages = {s.name for s in mandate.method.stages}
    said = []
    for decision in mandate.decisions.yours:
        for path in list(decision.locks):
            head = path.split(".")[0]
            held = HELD_ELSEWHERE.get(head) if head not in stages else None
            if held:
                del decision.locks[path]
                said.append(f"\"{decision.said}\": its lock on {path} is dropped, since "
                            f"{held} holds it; the decision stands as its words")
    return said


def _unread_dropped(mandate: mandates.Mandate) -> list[str]:
    """Drops each lock, or item of a locked mapping, that nothing the stages compile to
    reads, keeping the decision's words, and says so. Run once the locks are known to agree
    with the stages: a lock that contradicts its stage is refused, never dropped."""
    from myrmidon.knowledge.stage import TemplateStore, unread_locks

    said = []
    for at, path, item, _ in unread_locks(mandate, TemplateStore.default()):
        decision = mandate.decisions.yours[at]
        held = decision.locks.get(path)
        if item is not None and isinstance(held, dict):
            held.pop(item, None)
            if held:
                said.append(f"\"{decision.said}\": its lock on {path}.{item} is dropped, "
                            "since nothing the stages compile to reads it; the rest binds")
                continue
        decision.locks.pop(path, None)
        said.append(f"\"{decision.said}\": its lock on {path} is dropped, since nothing the "
                    "stages compile to reads it; the decision stands as its words")
    return said


def _built(params: MandateWriteParams, files: list, questions: list[str],
           existing: mandates.Mandate | None, approved: bool) -> mandates.Mandate:
    """The mandate the call states, with the revision it is."""
    return mandates.Mandate(
        question=params.question,
        # each file by its path in the study, which a later file of the same name elsewhere
        # cannot take over
        system=mandates.System(description=params.system_description, files=files,
                               subsystems=params.subsystems,
            hamiltonian=params.hamiltonian or mandates.Hamiltonian()),
        method=mandates.Method(protocol=params.protocol, stages=params.stages),
        decisions=mandates.Decisions(yours=params.decisions_yours,
                                     proposed=params.decisions_proposed),
        deliverables=params.deliverables,
        cost=mandates.Cost(ceiling_core_h=params.ceiling_core_h or 0.0),
        recovery=mandates.Recovery(reruns=params.reruns, model_fixes=params.model_fixes,
                                   deadline_h=params.deadline_h),
        status="needs_input" if questions else "ready",
        questions=questions[:mandates.MAX_QUESTIONS],
        revision=existing.revision + 1 if approved else
        (existing.revision if existing is not None else 1),
        # a draft revision written again keeps why it exists
        reason=params.reason or (existing.reason if existing is not None and not approved
                                 else ""),
    )


@tool(
    "mandate_write",
    description="Write the mandate for the researcher to approve: question, system and "
                "subsystems, stages in physical terms, their decisions with the stage fields "
                "they lock, deliverables, their ceiling. Compiles and prices each stage. "
                "Returns `ready` with a digest, or `needs_input` with their questions.",
    tags={"mandate"},
)
def mandate_write(ctx: ToolContext, params: MandateWriteParams) -> Observation:
    from myrmidon.knowledge.observables import property_named

    ws = ctx.workspace
    existing = mandates.read_mandate(ws)
    named_protocol = "protocol" in params.model_fields_set
    if params.revise and existing is not None:
        params = _carried(params, existing)
    stated = params
    params = _from_protocol(ctx, params)
    if params is None and not named_protocol:
        params = stated  # a held mandate's protocol this machine lacks: its stages stand
    if params is None:
        from myrmidon.knowledge.protocols import ProtocolStore

        known = ", ".join(sorted(p.id for p in (ctx.protocols or ProtocolStore.default()).all()))
        why = f"no protocol {stated.protocol!r} on this machine; published: {known}"
        return Observation(ok=False, summary=why, error=ToolError(
            code="PROTOCOL_UNKNOWN", message=why, hint="name one of them, or none"))
    if not params.question.strip():
        return Observation(ok=False, summary="a mandate states the researcher's question, "
                           "verbatim: give `question`")
    params = params.model_copy(update={"deliverables": [
        d.model_copy(update={"property": property_named(d.property)}) if d.property else d
        for d in params.deliverables]})
    missing = [f.path for f in params.system_files if not ws.resolve(f.path).is_file()]
    if missing:
        return Observation(ok=False, summary="no such system file(s): "
                                  + ", ".join(missing) + " — name files the study holds")
    approved = existing is not None and existing.digest in mandates.approved_digests(ws)
    if approved and not params.revise:
        return Observation(
            ok=False, summary=f"mandate revision {existing.revision} is approved; a change "
            "to it is a revision: mandate_write with revise=true and the reason, which the "
            "researcher approves")
    if params.revise and approved and not params.reason.strip():
        return Observation(ok=False, summary="a revision says why: give the reason")
    unread = _unread_targets(ctx, params.decisions_yours)
    if unread:  # a lock that holds nothing is not put to the researcher as one
        return Observation(ok=False, summary="a decision binds nothing as written: "
                           + " ".join(f.message for f in unread), findings=unread,
                           error=ToolError(code="MANDATE_TARGET_UNREAD",
                                           message=unread[0].message,
                                           hint="lock the researcher's words on keys the "
                                           "target reads; a choice the system files, the "
                                           "ceiling or the approval hold keeps `said` with no "
                                           "keys; how a number is measured goes in its "
                                           "deliverable's params"))
    unknown = _unknown_params(params.deliverables)
    if unknown:
        return Observation(ok=False, summary="a deliverable's params are its tool's "
                           "parameters: " + "; ".join(unknown))

    # the harness's own questions first: the model's cannot crowd them out
    questions = []
    if params.ceiling_core_h is None:
        questions.append("How many core-hours may this study book at most?")
    if not params.system_files:
        questions.append("Which files are the system to simulate?")
    questions += open_chemistry(ws, params.stages, params.decisions_yours)
    questions += params.questions
    from myrmidon.core.contracts import includes_of

    files = [mandates.SystemFile(path=ws.relpath(ws.resolve(f.path)), kind=f.kind)
             for f in params.system_files]
    if any(s.qm_region is not None for s in params.stages):
        files += _selection_files(ws, {f.kind for f in files})
    for f in list(files):  # what a topology includes is the system too
        for path in includes_of(f.kind, ws.resolve(f.path)):
            if ws.relpath(path) not in {x.path for x in files}:
                files.append(mandates.SystemFile(path=ws.relpath(path), kind="itp"))
    try:
        mandate = _built(params, files, questions, existing, approved)
    except ValidationError as exc:  # the parts do not hold together: a stage order, say
        said = [e["msg"].removeprefix("Value error, ") for e in exc.errors()
                if e["type"] == "value_error"]
        if not said:
            raise
        return _unbound(said)
    from myrmidon.execution.gate import mandate_facts

    facts = mandate_facts(ctx, mandate)
    parts = _topology_parts(ctx, mandate)
    read = _read_system(ctx, mandate, params, facts, parts)
    unmeasured, subsystem_notes = _subsystem_reading(ctx, mandate, parts)
    asked = _region_charges(mandate, parts)
    if asked:
        mandate.questions = [*mandate.questions, *asked][:mandates.MAX_QUESTIONS]
        mandate.status = "needs_input"
    dropped = _held_elsewhere_dropped(mandate)
    unbound = [*mandates.lock_findings(mandate), *unmeasured]
    if not unbound:
        dropped += _unread_dropped(mandate)
    comp = compiled(ctx, mandate, facts)
    if unbound or comp.problems:  # what the researcher would approve cannot bind as written
        return _unbound([*unbound, *comp.problems])
    comp.notes += dropped
    mandate.cost.estimate_core_h = comp.estimate_core_h
    mandate.cost.unpriced = list(comp.unpriced)
    mandate = mandates.write_mandate(ws, mandate)
    if mandate.status == "needs_input":
        summary = (f"mandate revision {mandate.revision} waits on the researcher: ask them "
                   + " ".join(mandate.questions) + " Then write it again with the answers.")
    else:
        summary = ("ready for the researcher: stages "
                   + ", ".join(s.name for s in mandate.method.stages)
                   + (", measuring " + ", ".join(d.name for d in mandate.deliverables)
                      if mandate.deliverables else "")
                   + f"; estimated {mandate.cost.estimate_core_h:g} of "
                   f"the ceiling's {mandate.cost.ceiling_core_h:g} core-h"
                   + (f" ({', '.join(comp.unpriced)} unpriced)" if comp.unpriced else "")
                   + ": " + APPROVAL_FORMS + ".")
    if read:
        summary += " " + read
    if comp.notes or subsystem_notes:
        summary += " Note: " + " ".join([*comp.notes, *subsystem_notes])
    obs = _observe(ctx, mandate, summary)
    obs.findings = [*(Finding(id="MANDATE_NOTE", message=m) for m in comp.notes),
                    *(Finding(id="MANDATE_SUBSYSTEM_NOTE", message=m) for m in subsystem_notes)]
    obs.facts = [read] if read else []  # what was read from the system, to confirm
    obs.quiet = {"text"}  # the model wrote it; the researcher reads it rendered
    obs.artifacts = [ArtifactRef.from_path(ws.root / mandates.MANDATE_FILE,
                                           mandates.MANDATE_ROLE, ws.root)]
    return obs


class MandateShowParams(BaseModel):
    pass


@tool(
    "mandate_show",
    description="The study's mandate as the researcher reads it, whether an approval of it "
    "holds, and what its jobs have booked against the ceiling.",
    tags={"mandate"},
    navigation=True,
)
def mandate_show(ctx: ToolContext, params: MandateShowParams) -> Observation:
    mandate = mandates.read_mandate(ctx.workspace)
    if mandate is None:
        return Observation(ok=False, summary="this study has no mandate: write one "
                                  "with mandate_write once the system is built")
    obs = _observe(ctx, mandate, "")
    obs.summary = (f"mandate revision {mandate.revision}, "
                   + ("approved" if obs.approved else f"not approved: {obs.why_not}")
                   + f"; {obs.spent_core_h:g} of {obs.ceiling_core_h:g} core-h booked")
    obs.kind = "observe"
    return obs


class ProtocolPreviewParams(BaseModel):
    protocol: str = Field(description="a published protocol, as <id> or <id>@v<N>")
    structure: str | None = Field(default=None, description="the built system's structure: "
                                  "with the topology, each stage is rendered and built on it")
    topology: str | None = Field(default=None, description="the built system's topology")
    index: str | None = Field(default=None, description="the built system's index, when the "
                              "stages' groups need one")
    overrides: dict[str, dict] = Field(default_factory=dict, description="step name (a "
                                       "_prep step) -> the engine keys the study sets "
                                       "differently from its stage, e.g. a length: priced "
                                       "and built as set")


@tool(
    "protocol_preview",
    description="Price a published protocol's stages for the mandate and, given the built "
    "system, render and build each on it (grompp, nothing run, nothing recorded): what "
    "mandate_write takes as the stages and the estimate.",
    tags={"mandate"},
    navigation=True,
    needs_workspace=False,
)
def protocol_preview(ctx: ToolContext, params: ProtocolPreviewParams) -> Observation:
    store = ctx.protocols
    try:
        protocol, wanted = store.get_versioned(params.protocol)
    except (KeyError, AttributeError):
        known = ", ".join(sorted(p.id for p in store.all())) if store else "none"
        return Observation(ok=False, summary=f"no protocol {params.protocol!r}; "
                                  f"published: {known}")
    if wanted is not None and str(protocol.version) != wanted:
        return Observation(ok=False, summary=f"this machine has {protocol.id} at "
                                  f"v{protocol.version}, not v{wanted}")
    from myrmidon.orchestration.preview import stages

    system = {"structure": params.structure, "topology": params.topology,
              "index": params.index}
    tried = stages(ctx, protocol, system, params.overrides)
    priced = [s for s in tried if s.core_h]
    unpriced = [s.name for s in tried if s.core_h is None and s.basis.startswith("unpriced")]
    total = round(sum(s.core_h for s in priced), 3)
    broken = [s.name for s in tried if s.rendered is False or s.built is False]
    built = [s for s in tried if s.built]
    return Observation(
        ok=not broken, estimate_core_h=total, broken=broken,
        summary=f"{protocol.id}@v{protocol.version}: {len(priced)} stage(s) book jobs, "
        f"{total:g} core-h on this machine"
        + ("" if params.structure and params.topology else " for no system in particular")
        + (f", not counting {', '.join(unpriced)}, which nothing here prices yet"
           if unpriced else "")
        + (f"; {len(built)} stage(s) render and build on this system" if built else "")
        + (f"; {', '.join(broken)} do NOT: read `text`" if broken else "")
        + f". mandate_write with protocol={protocol.id}@v{protocol.version} prices them "
        "for the study's own system.",
        text="\n".join(s.line() for s in tried if s.core_h or s.rendered is not None or s.said))
