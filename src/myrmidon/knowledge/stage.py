"""A mandate's stage compiled into what an engine reads, and priced.

A stage says what physically happens (``minimize``; NPT dynamics at 300 K and 1 bar for
100 ps with the protein and the ligand restrained). A **stage template**
(``knowledge/stages/<engine>.<kind>.yaml``) says how one engine is told so, key by key, each
key in one of five tabular forms — no expression language:

- ``value``: the template's own, with its source;
- ``from_stage``: a stage field or a quantity derived from it (``steps`` from the length
  and the method's step, ``sampling_steps``, ``qm_region.charge``);
- ``from_method``: a constant of the Hamiltonian's method (``knowledge/methods/``),
  converted to the ``unit`` the engine takes;
- ``from_system``: a fact read from the system's files, ``for: restraints`` the part of it
  belonging to the subsystems the stage restrains;
- ``by_stage``: a derived quantity (``thermostat``, ``barostat``, ``geometry``, ``kind``)
  mapped to the engine's word; a quantity the map has no word for leaves the key out, or,
  for a ``required`` key, says the engine cannot run the stage so.

``when`` keeps a key to the stages it applies to, ``with`` appends values on the key's line,
``repeat`` writes the value once per axis a derived count names, ``in_key`` writes the value
into the key's own line. A template may ``extends`` another, whose keys it shares, and
names the ensembles it can run. Every compiled key keeps its origin (``template:``,
``stage:``, ``method:``, ``mandate:``, ``system:``, ``user``) in the :class:`Resolution`, so
the sidecar beside a rendered file says where each value came from.

Nothing here reads a file: the facts a template takes from the system are read by the
engine's own reader (engines/) and passed in.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any, Literal, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from myrmidon.core.mandate import Mandate, Stage, field_value, lock_path, locks_on
from myrmidon.knowledge.catalog import SystemFact, canonical, load
from myrmidon.knowledge.inputsets import (
    Invariant,
    Resolution,
    _identity,
    _per_coupling_group,
    engine_key,
    owner_of,
)
from myrmidon.knowledge.layered import LayeredStore, Scope
from myrmidon.knowledge.observables import compile_deliverable
from myrmidon.knowledge.sources import Source

#: a picosecond in each time unit a method states its step in
PS_PER = {"ps": 1.0, "fs": 1e-3, "au": 2.418884326585747e-5}
#: the stage fields a derived quantity stands for: a lock on one binds the keys the
#: quantity sets (a thermostat is the ensemble's, not the temperature's)
DERIVED = {
    "steps": ("steps", "duration_ps"),
    "sampling_steps": ("sampling_ps",),
    "thermostat": ("kind", "ensemble"),
    "barostat": ("ensemble",),
    "geometry": (),
    "pressure_axes": (),
    "open_shell": ("qm_region.multiplicity",),
    "temperature_k": ("temperature_k",),
    "temperature_ramp": ("temperature_k",),
    "ramp_start_k": ("temperature_k",),
}
#: the stage fields the integration step stands for: a length is steps times the step
STEP_BOUND = ("duration_ps", "sampling_ps")
#: derived quantities a render supplies from the run before (a ramp starts where it ended)
FROM_RUN = ("temperature_ramp", "ramp_start_k")


class Form(BaseModel):
    """Where one engine key's value comes from."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    value: Any = None
    from_stage: str | None = None
    from_method: str | None = None
    of: Literal["mm", "qm", "coupling"] | None = None  # whose method; the template's
    from_system: str | None = None
    for_: Literal["restraints"] | None = Field(default=None, alias="for")
    by_stage: str | None = None
    map: dict[str, Any] = Field(default_factory=dict)
    when: dict[str, Any] = Field(default_factory=dict)
    with_: list[Any] = Field(default_factory=list, alias="with")
    repeat: str | None = None  # a derived count: the value written once per axis
    unit: str | None = None
    in_key: bool = False
    default: Any = None
    required: bool = False
    rationale: str | None = None
    source: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _scalar(cls, data: Any) -> Any:
        return data if isinstance(data, dict) else {"value": data}

    @model_validator(mode="after")
    def _one_source(self) -> Form:
        named = [n for n in ("value", "from_stage", "from_method", "from_system", "by_stage")
                 if getattr(self, n) is not None]
        if len(named) != 1:
            raise ValueError("a key has exactly one of value, from_stage, from_method, "
                             f"from_system, by_stage; this one has {named or 'none'}")
        return self

    def depends(self) -> set[str]:
        """The stage fields this key's value follows, by their paths."""
        names = [n for n in (self.from_stage, self.by_stage) if n]
        out = {f for n in names for f in DERIVED.get(n, (n,))}
        if self.for_ == "restraints":
            out.add("restraints")
        if self.from_method and self.from_method.startswith("timestep_"):
            out |= set(STEP_BOUND)
        return out


class Template(BaseModel):
    """How one engine is told one kind of stage, key by key."""

    id: str
    title: str
    engine: str
    renders: str
    kinds: list[str] = Field(default_factory=list)  # none: only extended, never chosen
    regimes: list[str] = Field(default_factory=list)
    ensembles: list[str] = Field(default_factory=list)  # none: any the stage has
    reads: Literal["mm", "qm"] = "mm"  # the Hamiltonian part ``from_method`` reads
    extends: str | None = None
    applicability: str = ""
    pricing: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Form] = Field(default_factory=dict)
    sources: list[Source] = Field(min_length=1)
    pitfalls: list[str] = Field(default_factory=list)  # finding ids that bite this
    invariants: list[Invariant] = Field(default_factory=list)
    notes: str = ""
    version: int = 1
    scope: Scope = "project"
    layer: Scope | None = Field(default=None, exclude=True)
    sha256: str = Field(default="", exclude=True)

    @model_validator(mode="after")
    def _keys_read(self) -> Template:
        try:
            catalog = load(self.engine)
        except Exception:  # noqa: BLE001 — an engine with no catalog is checked by its runs
            catalog = None
        for key, form in self.parameters.items():
            if catalog is not None and engine_key(catalog, key) is None:
                raise ValueError(f"{self.id}: {key!r} is not a key {self.engine} reads")
            if form.source is not None and form.source >= len(self.sources):
                raise ValueError(f"{self.id}: {key} points at source {form.source} of "
                                 f"{len(self.sources)}")
        return self

    @property
    def ref(self) -> str:
        return f"{self.id}@v{self.version}"


class MethodValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Any
    rationale: str | None = None
    source: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _scalar(cls, data: Any) -> Any:
        return data if isinstance(data, dict) else {"value": data}


class MethodSpec(BaseModel):
    """A Hamiltonian's constants, from its papers, in no engine's words; a name's suffix
    is its unit (``timestep_fs``, ``cutoff_nm``, ``cutoff_ry``)."""

    id: str
    title: str
    kind: Literal["mm", "qm", "coupling"]
    forcefields: list[str] = Field(default_factory=list)  # force-field directory prefixes
    applicability: str = ""
    parameters: dict[str, MethodValue] = Field(default_factory=dict)
    sources: list[Source] = Field(min_length=1)
    notes: str = ""
    version: int = 1
    scope: Scope = "project"
    layer: Scope | None = Field(default=None, exclude=True)
    sha256: str = Field(default="", exclude=True)

    @property
    def ref(self) -> str:
        return f"{self.id}@v{self.version}"


class TemplateStore(LayeredStore[Template]):
    model = Template
    kind = "stage template"
    dirname = "stages"


class MethodStore(LayeredStore[MethodSpec]):
    model = MethodSpec
    kind = "method"
    dirname = "methods"


def extends_chain(template: Template, store: TemplateStore) -> list[Template]:
    """The template and every template it extends, itself first."""
    chain, seen = [template], {template.id}
    while chain[-1].extends:
        base = store.get(chain[-1].extends)
        if base.id in seen:
            raise ValueError(f"{template.id}: extends itself through {base.id}")
        seen.add(base.id)
        chain.append(base)
    return chain


def with_inherited(template: Template, store: TemplateStore) -> Template:
    """The template with the pitfalls and invariants of every template it extends: a rule
    on the keys an extender shares holds for the extender too."""
    chain = extends_chain(template, store)
    pitfalls = list(dict.fromkeys(p for t in reversed(chain) for p in t.pitfalls))
    invariants = list({i.id: i for t in reversed(chain) for i in t.invariants}.values())
    return template.model_copy(update={"pitfalls": pitfalls, "invariants": invariants})


def flattened(template: Template, store: TemplateStore) -> tuple[dict[str, Form],
                                                                 list[Source]]:
    """A template's keys with those of the templates it extends, its own last, and the
    sources they point at, renumbered into one list."""
    chain = extends_chain(template, store)
    params: dict[str, Form] = {}
    sources: list[Source] = []
    for layer in reversed(chain):
        offset = len(sources)
        sources += layer.sources
        for key, form in layer.parameters.items():
            if form.source is not None:
                form = form.model_copy(update={"source": form.source + offset})
            params.pop(key, None)  # the extending template's place and value
            params[key] = form
    return params, sources


def templates_for(stage: dict, store: TemplateStore, engine: str | None = None,
                  any_ensemble: bool = False) -> list[Template]:
    """The templates that tell an engine this kind of stage in this regime and ensemble."""
    return [t for t in store.all()
            if stage.get("kind") in t.kinds and stage.get("regime", "mm") in t.regimes
            and (any_ensemble or not t.ensembles or stage.get("ensemble") is None
                 or stage.get("ensemble") in t.ensembles)
            and (engine is None or t.engine == engine)]


class Methods(BaseModel):
    """The Hamiltonian a stage is computed with: each part's method and its values, the
    mandate's calibration over the method's own, and what has no method here."""

    specs: dict[str, MethodSpec] = Field(default_factory=dict)
    values: dict[str, dict[str, tuple[Any, str]]] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)

    def step_ps(self, regime: str) -> float | None:
        """The integration step in ps of the part that integrates a stage of ``regime``,
        from the ``timestep_<unit>`` its method states."""
        for name, (value, _) in self.values.get(
                "qm" if regime in ("qmmm", "qm") else "mm", {}).items():
            unit = name.removeprefix("timestep_")
            if name.startswith("timestep_") and unit in PS_PER:
                return float(value) * PS_PER[unit]
        return None


def methods_for(mandate: Mandate, store: MethodStore, forcefield: str | None = None
                ) -> Methods:
    """The mandate's Hamiltonian as methods: ``mm`` by its id, else by the force field the
    topology includes; ``qm`` and ``coupling`` by id, with the values the mandate states
    over the method's own."""
    ham = mandate.system.hamiltonian
    out = Methods()
    known = {m.id: m for m in store.all()}

    def take(role: str, ref: str, overlay: dict[str, Any]) -> None:
        spec = known.get(ref)
        if spec is None:
            out.missing.append(f"no {role} method {ref!r}; known: "
                               + ", ".join(sorted(k for k, m in known.items()
                                                  if m.kind == role)))
            return
        out.specs[role] = spec
        values = {k: (v.value, f"method:{spec.ref}") for k, v in spec.parameters.items()}
        values.update({k: (v, f"mandate:hamiltonian.{role}.{k}")
                       for k, v in overlay.items() if v is not None})
        out.values[role] = values

    def covering(name: str) -> MethodSpec | None:
        """The MM method whose force-field prefix the name starts with, the longest prefix
        winning (a site's amber14sb over the shipped amber); None when none or a tie."""
        hits = sorted(((len(p), m) for m in known.values() if m.kind == "mm"
                       for p in m.forcefields if name.lower().startswith(p.lower())),
                      key=lambda hit: -hit[0])
        if not hits or (len(hits) > 1 and hits[0][0] == hits[1][0]
                        and hits[0][1].id != hits[1][1].id):
            return None
        return hits[0][1]

    # the force field by its method's id, or by its own name (amber99sb-ildn), which must
    # agree with the force field the topology includes
    if ham.mm in known:
        by_topology = covering(forcefield) if forcefield else None
        if by_topology is not None and by_topology.id != ham.mm:
            out.missing.append(f"system.hamiltonian.mm is {ham.mm}, and the topology "
                               f"includes {forcefield} ({by_topology.id}): name the one the "
                               "topology is")
        else:
            take("mm", ham.mm, {})
    elif ham.mm or forcefield:
        by_name = covering(ham.mm) if ham.mm else None
        by_topology = covering(forcefield) if forcefield else None
        if by_name and by_topology and by_name.id != by_topology.id:
            out.missing.append(f"system.hamiltonian.mm names {ham.mm} ({by_name.id}), and the "
                               f"topology includes {forcefield} ({by_topology.id}): name the "
                               "one the topology is")
        elif by_name or by_topology:
            take("mm", (by_name or by_topology).id, {})
        else:
            out.missing.append(
                f"no method here covers the force field {ham.mm or forcefield} (known: "
                + ", ".join(sorted(k for k, m in known.items() if m.kind == "mm"))
                + "): its constants (step, cutoffs, electrostatics) go in "
                "~/.config/myrmidon/methods/")
    if ham.qm is not None:
        take("qm", ham.qm.method, {"cutoff_ry": ham.qm.cutoff_ry})
    if ham.coupling is not None:
        take("coupling", ham.coupling.scheme, {
            name: getattr(ham.coupling, name) for name in ("multipole_order", "cutoff_bohr")})
    return out


def derived(stage: dict, mandate: Mandate, step_ps: float | None,
            facts: dict[str, SystemFact] | None = None) -> dict[str, Any]:
    """The quantities a template reads that the stage implies rather than states; the
    system's facts say whether the structure it starts from carries velocities."""
    temp = stage.get("temperature_k")
    ramp = temp if isinstance(temp, dict) else None
    out: dict[str, Any] = {"kind": stage.get("kind")}
    steps = stage.get("steps")
    if steps is None and stage.get("duration_ps") and step_ps:
        steps = round(stage["duration_ps"] / step_ps)
    out["steps"] = steps
    sampling = stage.get("sampling_ps")
    out["sampling_steps"] = round(sampling / step_ps) if sampling and step_ps else None
    if stage.get("kind") != "dynamics" or stage.get("ensemble") in (None, "NVE"):
        out["thermostat"] = "none"
    else:
        out["thermostat"] = "ramp" if ramp is not None else "constant"
    # a free box continues earlier dynamics, or starts from rest — after a minimisation, as
    # the first stage, or from a structure that carries no velocities — and draws them afresh
    names = [s.name for s in mandate.method.stages]
    at = names.index(stage["name"]) if stage.get("name") in names else 0
    if facts and "gen-vel" in facts:  # the structure says it
        continues = str(facts["gen-vel"].value).lower() != "yes"
    else:
        continues = at > 0 and mandate.method.stages[at - 1].kind == "dynamics"
    out["barostat"] = ("none" if stage.get("ensemble") != "NPT"
                       else "NPT_restrained" if stage.get("restraints")
                       else "NPT_free" if continues else "NPT_from_rest")
    membrane = any(s.kind == "membrane" for s in mandate.system.subsystems.values())
    out["geometry"] = "membrane" if membrane else "bulk"
    out["pressure_axes"] = 2 if membrane else 1  # in the plane, along the normal
    region = stage.get("qm_region") or {}
    out["open_shell"] = bool(region) and region.get("multiplicity", 1) > 1
    out["temperature_k"] = ramp.get("to_k") if ramp is not None else temp
    out["ramp_start_k"] = ramp.get("start_k") if ramp is not None else None
    out["temperature_ramp"] = (
        " ".join(f"{v:g}" for v in (ramp["start_k"], ramp["to_k"], ramp["rate_k_per_au"]))
        if ramp is not None and ramp.get("start_k") is not None
        and ramp.get("rate_k_per_au") is not None else None)
    return out


def _stage_value(stage: dict, quantities: dict, name: str):
    return quantities[name] if name in quantities else field_value(stage, name)


def _number(value: Any) -> Any:
    """A computed number as an engine input states it: integral floats as integers, the
    rest without float noise."""
    if isinstance(value, float):
        value = float(f"{value:.10g}")
        return int(value) if value.is_integer() and abs(value) >= 1 else value
    return value


def _applies(form: Form, quantities: dict) -> bool:
    for name, wanted in form.when.items():
        have = quantities.get(name)
        options = wanted if isinstance(wanted, list) else [wanted]
        if str(have) not in {str(o) for o in options}:
            return False
    return True


class Compiled(Resolution):
    """A stage's resolution, and the keys no render can supply because the stage, as
    stated, cannot be told to this engine (a subsystem without restraints, an ensemble the
    engine has no word for); each is also in ``missing``."""

    unbuildable: dict[str, str] = Field(default_factory=dict)


def compile_stage(mandate: Mandate, stage: Stage | dict | str, template: Template, *,
                  templates: TemplateStore, methods: Methods,
                  facts: dict[str, SystemFact] | None = None,
                  overrides: dict[str, Any] | None = None) -> Compiled:
    """One stage told one engine: every key the template states, from the stage, the
    method, the system's facts and the template itself, each with its origin; a key the
    template requires and nothing supplies is in ``missing`` with how to supply it."""
    if isinstance(stage, str):
        found = mandate.method.stage(stage)
        if found is None:
            raise KeyError(f"no stage {stage!r} in the mandate's method")
        stage = found
    view = stage.model_dump(mode="json") if isinstance(stage, Stage) else dict(stage)
    params, sources = flattened(template, templates)
    facts = facts or {}
    quantities = derived(view, mandate, methods.step_ps(view.get("regime", "mm")), facts)
    res = Compiled(inputset=template.id, version=template.version, renders=template.renders,
                   method=", ".join(m.ref for m in methods.specs.values()) or None,
                   sources=sources)
    try:
        catalog = load(template.engine)
        res.engine_defaults = f"{catalog.engine}-{catalog.version}"
    except Exception:  # noqa: BLE001
        catalog = None
    restrained = {name: mandate.system.subsystems.get(name)
                  for name in view.get("restraints") or []}
    name = view.get("name", "*")

    def place(key: str, value: Any, origin: str) -> None:
        res.values[key] = _number(value) if not isinstance(value, bool) else value
        res.origin[key] = origin
        if catalog is not None and engine_key(catalog, key) is not None:
            res.owner[key] = owner_of(origin)

    for key, form in params.items():
        if not _applies(form, quantities):
            continue
        value, origin, why, stated = None, "", "", False
        if form.value is not None:
            value, origin = form.value, f"template:{template.ref}"
        elif form.from_stage:
            value = _stage_value(view, quantities, form.from_stage)
            origin, why = f"stage:{name}.{form.from_stage}", f"the stage's {form.from_stage}"
        elif form.by_stage:
            have = _stage_value(view, quantities, form.by_stage)
            value = form.map.get(str(have))
            origin = f"stage:{name}.{form.by_stage}"
            why = (f"{template.engine} has no {key} for a stage whose {form.by_stage} is "
                   f"{have}")
            stated = True
        elif form.from_method:
            role = form.of or template.reads
            found = methods.values.get(role, {}).get(form.from_method)
            if found is not None:
                value, origin = found
                if form.unit:
                    unit = form.from_method.rsplit("_", 1)[-1]
                    value = float(value) * PS_PER[unit] / PS_PER[form.unit]
            why = f"the {role} method's {form.from_method}"
        elif form.from_system:
            fact = facts.get(key)
            why = f"read from the system by {form.from_system}: pass its files"
            if form.for_ == "restraints" and restrained and fact is None:
                why = (f"no topology among the system files to read the restraints of "
                       f"{', '.join(restrained)} from")
                stated = True
            elif fact is not None and form.for_ == "restraints":
                value, why = _restraints_of(fact, restrained)
                origin, stated = fact.origin, True
            elif fact is not None:
                value, origin = fact.value, fact.origin
            if fact is not None:
                res.read[key] = {"rule": fact.rule, "files": fact.read,
                                 "sha256": fact.sha256}
        if value is None and form.default is not None:
            value, origin = form.default, f"template:{template.ref}"
        if value is None or value == "":
            # a fact the system was not read for is missing; a method silent on a key
            # leaves it to the engine; restraints on nothing are no restraints
            if form.required or (value is None and form.from_system):
                res.missing[key] = why or "the template requires it"
                if stated and value is None:  # the stage itself cannot be told so
                    res.unbuildable[key] = why
            continue
        if form.repeat:
            value = " ".join([str(_number(value))] * int(quantities.get(form.repeat) or 1))
        if form.with_:
            value = " ".join(str(_number(v)) for v in (value, *form.with_))
        if form.in_key:
            key, value = f"{key} {value}", True
        place(key, value, origin)

    # an override by the spelling the template writes its key in: one key, one line
    spelled = {_identity(catalog, k): k for k in [*params, *res.values]}
    for given, value in (overrides or {}).items():
        key = spelled.get(_identity(catalog, given), given)
        was = res.values.get(key)
        if key not in res.values:
            res.additions.append(key)
        elif str(was) != str(value):
            res.overrides[key] = {"inputset": was, "user": value}
        res.missing.pop(key, None)
        place(key, value, "user")
    _per_coupling_group(res)
    return res


def _restraints_of(fact: SystemFact, restrained: dict) -> tuple[str | None, str]:
    """The restraint switches of the restrained subsystems only, or why there are none."""
    if not restrained:
        return "", ""
    tokens: list[str] = []
    for name, sub in restrained.items():
        if sub is None:
            return None, (f"the stage restrains {name!r}, which system.subsystems does not "
                          "define")
        parts = [fact.parts[n] for n in sub.names() if n in fact.parts]
        if not parts:
            return None, (f"the topology defines no position restraints for {name} "
                          f"({', '.join(sub.names()) or 'no molecules named'})")
        tokens += [t for p in parts for t in p.split() if t not in tokens]
    return " ".join(tokens), ""


# --- what a lock binds ------------------------------------------------------------------


def _numbers(value) -> list[float] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return [float(value)]
    try:
        return [float(t) for t in str(value).split()] or None
    except ValueError:
        return None


def _uniform(value):
    """The one number a per-group value repeats (``"300 300"``), else the value."""
    numbers = _numbers(value)
    if isinstance(value, str) and numbers and len(numbers) > 1 and len(set(numbers)) == 1:
        return _number(numbers[0])
    return value


def _list_field(field: str) -> bool:
    info = Stage.model_fields.get(field)
    return info is not None and get_origin(info.annotation) is list


def _options(stage: dict, field: str, constraint) -> tuple[str, list[dict]]:
    """(how the options combine, the stage as each option of a lock has it): ``one`` for
    an equal lock, ``any`` for one of a list, ``range`` for ``{min, max}``."""
    current = field_value(stage, field)

    def at(value) -> dict:
        out = {**stage}
        parts = field.split(".")
        node = out
        for part in parts[:-1]:
            node[part] = dict(node.get(part) or {})
            node = node[part]
        node[parts[-1]] = value
        return out

    if isinstance(constraint, dict) and constraint and set(constraint) <= {"min", "max"}:
        ends = [constraint.get("min"), constraint.get("max")]
        return "range", [at(v) for v in ends if v is not None]
    if isinstance(constraint, dict):  # some items of a mapping field
        return "one", [at({**(current or {}), **constraint})]
    if isinstance(constraint, list) and not (isinstance(current, list) or _list_field(field)):
        return "any", [at(v) for v in constraint]
    return "one", [at(constraint)]


def _alternatives(form: Form, option: dict, value, mandate: Mandate,
                  step_ps: float | None) -> list:
    """The values a file may hold for a key of this option: a word the template chooses
    among for a physical state (a barostat for NPT, whichever the restraints call for) is
    any of the words for that state; a number repeated per group is that number."""
    if not form.by_stage:
        return [_uniform(value)]
    if form.by_stage == "barostat":  # any barostat the ensemble allows, wherever it starts
        states = (("NPT_restrained", "NPT_from_rest", "NPT_free")
                  if option.get("ensemble") == "NPT" else ("none",))
        return list(dict.fromkeys(form.map[s] for s in states if s in form.map)) or [value]
    words = []
    for restraints in ([], ["any"]):
        quantities = derived({**option, "restraints": restraints}, mandate, step_ps)
        word = form.map.get(str(_stage_value(option, quantities, form.by_stage)))
        if word is not None and word not in words:
            words.append(word)
    return words or [value]


def _constraint(form: Form, how: str, values: list):
    """One constraint over what each option allows: on a line whose later numbers the
    template appends (a thermostat's frequency after its temperature), the first number
    only; a range of numbers as ``{min, max}``; one value equal; several one of."""
    if form.with_:
        firsts = [(_numbers(v) or [None])[0] for v in values]
        if None not in firsts:
            free = [None] * len(form.with_)
            ends = sorted({_number(f) for f in firsts})
            if how == "any" and len(ends) > 1:
                return [{"min": [e, *free], "max": [e, *free]} for e in ends]
            return {"min": [ends[0], *free], "max": [ends[-1], *free]}
    if how == "range":
        numbers = [_numbers(v) for v in values]
        if all(n is not None and len(n) == 1 for n in numbers):
            ends = [n[0] for n in numbers]
            return {"min": _number(min(ends)), "max": _number(max(ends))}
    kept = list(dict.fromkeys(values))
    return kept[0] if len(kept) == 1 else kept


def binds(form: Form, field: str) -> bool:
    """Whether a lock on ``field`` binds a key: the key follows that field, a part of it
    (a lock on ``qm_region`` binds its charge), or the whole it is part of."""
    return not form.in_key and any(
        d == field or d.startswith(field + ".") or field.startswith(d + ".")
        for d in form.depends())


def _bound(template: Template, templates: TemplateStore, field: str) -> dict[str, Form]:
    params, _ = flattened(template, templates)
    return {k: f for k, f in params.items() if binds(f, field)}


def expected_keys(mandate: Mandate, stage_name: str | None, engine: str, *,
                  templates: TemplateStore, methods: Methods,
                  facts: dict[str, SystemFact] | None = None
                  ) -> tuple[dict[str, tuple[Any, str]], list[str]]:
    """What a parameter file of ``engine`` written for ``stage_name`` must hold for the
    researcher's locks: key -> (constraint, their words, whether leaving it unset breaks
    the lock), each constraint in the form a decision has (equal, one of, ``{min, max}``),
    and what cannot be compiled. A key only
    a render supplies (a ramp's start, read from the run before) is the render's. A stage
    the method does not name is held to every stage's locks, as any template of the engine
    would compile them."""
    stage = mandate.method.stage(stage_name)
    out: dict[str, tuple[Any, str]] = {}
    unresolved: list[str] = []
    for field, constraint, said in locks_on(mandate, stage_name):
        if stage is not None:
            view = stage.model_dump(mode="json")
            if field_value(view, field) is None:
                continue  # every stage's lock binds the stages that have the field
            bases = [(view, t) for t in templates_for(view, templates, engine)]
        else:
            bases = [({"name": "*", "kind": kind, "regime": regime, "ensemble": ens}, t)
                     for t in templates.all() if t.engine == engine
                     for kind in t.kinds for regime in t.regimes
                     for ens in ((None, "NVE", "NVT", "NPT") if kind == "dynamics"
                                 else (None,))]
        per_key: dict[str, tuple[Form, list]] = {}
        how = "one"
        for view, template in bases:
            how, options = _options(view, field, constraint)
            bound = _bound(template, templates, field)
            for option in options:
                res = compile_stage(mandate, option, template, templates=templates,
                                    methods=methods, facts=facts)
                step = methods.step_ps(option.get("regime", "mm"))
                for key, form in bound.items():
                    if key in res.unbuildable:
                        unresolved.append(f"{key} for \"{said}\": {res.unbuildable[key]}")
                    elif key in res.missing:
                        if form.from_stage not in FROM_RUN:
                            unresolved.append(f"{key} for \"{said}\": {res.missing[key]}")
                    elif key in res.values:
                        per_key.setdefault(key, (form, []))[1].extend(
                            _alternatives(form, option, res.values[key], mandate, step))
                    elif form.from_method:
                        unresolved.append(f"{key} for \"{said}\": no method here states "
                                          f"{form.from_method}")
                    else:
                        per_key.setdefault(key, (form, []))[1].append("")
        for key, (form, values) in per_key.items():
            if stage is None and form.for_ != "restraints":
                # any template's, so a key it leaves unset binds nothing; restraints left
                # out are no restraints, which the lock may say
                values = [v for v in values if v != ""]
                if not values:
                    continue
            # a run of no named stage need not set what it does not use; restraints it
            # leaves out are restraints it does not have
            out[key] = (_constraint(form, how, values), said,
                        stage is not None or form.for_ == "restraints")
    return out, list(dict.fromkeys(unresolved))


def lock_problems(mandate: Mandate, templates: TemplateStore) -> list[str]:
    """Each lock nothing compiled for its stages reads: it would hold nothing the
    researcher approved."""
    out = []
    for decision in mandate.decisions.yours:
        for path, constraint in decision.locks.items():
            on, field = lock_path(path)
            stages = [s for s in mandate.method.stages if (on == "*" or s.name == on)
                      and field_value(s, field) is not None]
            # a mapping is locked item by item: each item must bind on its own
            items = ([f"{field}.{k}" for k in constraint] if isinstance(constraint, dict)
                     and not set(constraint) <= {"min", "max"} else [field])
            for item in items:
                if stages and not any(_bound(t, templates, item) for s in stages
                                      for t in templates_for(s.model_dump(mode="json"),
                                                             templates)):
                    out.append(f"\"{decision.said}\" locks {on}.{item}, which nothing "
                               f"compiled for {', '.join(s.name for s in stages)} reads: "
                               "keep their words with no lock, and the stage as they said it")
    return out


# --- what a stage costs -----------------------------------------------------------------


class Pace(BaseModel):
    """A measured rate on this machine: a dynamics stage's ns/day or a QM step's seconds,
    at a system size and a booking."""

    engine: str
    atoms: int = Field(gt=0)
    cores: int = Field(default=0, ge=0)
    partition: str = ""
    ns_per_day: float | None = None
    s_per_step: float | None = None
    cutoff_ry: float | None = None  # a QM step's plane-wave cutoff
    where: str = ""  # the study and the run it was measured on


#: how much longer than the measured pace a booking allows
MARGIN = 1.5


class Price(BaseModel):
    stage: str
    core_h: float | None = None
    walltime_min: int | None = None
    basis: str = ""
    measured: bool = False


def pace_for(paces: list[Pace], rate: str, cores: int, *, booking: bool = False
             ) -> tuple[Pace | None, float]:
    """The measurement for a run on ``cores``: one on as many cores first, else the
    nearest, and the factor its rate scales by to that many — linear in cores for a price;
    for a booking, never faster than measured, since runs do not scale perfectly."""
    found = [p for p in paces if getattr(p, rate)]
    same = next((p for p in found if p.cores == cores), None)
    if same is not None or not found:
        return same, 1.0
    near = min(found, key=lambda p: abs(math.log((p.cores or 1) / cores)))
    factor = cores / (near.cores or 1)
    return near, min(factor, 1.0) if booking else factor


def walltime(*, ns: float, atoms: int | None, cores: int, paces: list[Pace], compute,
             margin: float = MARGIN, booking: bool = False) -> tuple[float, str, bool]:
    """(minutes, why, measured) a dynamics run of ``ns`` on ``cores`` needs here, before
    the scheduler's rounding: at the pace measured on this machine, scaled to its atoms
    and cores, with a margin; else at ``compute.plausible_ns_per_day``, the fastest a run
    plausibly goes, so a floor."""
    pace, factor = pace_for(paces, "ns_per_day", cores, booking=booking)
    if pace is not None and atoms:
        rate = pace.ns_per_day * pace.atoms / atoms * factor
        why = (f"{ns:g} ns at {rate:.3g} ns/day: measured {pace.ns_per_day:g} ns/day on "
               f"{pace.atoms} atoms and {pace.cores} core(s) ({pace.where}), scaled to "
               f"{atoms} atoms" + (f" and {cores} cores" if factor != 1 else ""))
        measured = True
    else:
        rate = compute.plausible_ns_per_day
        why = (f"at least: {ns:g} ns at {rate:g} ns/day, the fastest plausible here; "
               "unmeasured on this machine, re-priced after the first stage")
        measured = False
    return ns / rate * 24 * 60 * margin, why, measured


def qm_walltime(*, steps: int, qm_atoms: int | None, cutoff_ry: float | None, cores: int,
                paces: list[Pace]) -> tuple[float | None, str]:
    """(minutes, why) a coupled run of ``steps`` needs here, at the seconds per step
    measured on the same QM region size and cutoff — a step's cost grows steeply with
    both, so no other measurement stands in; (None, why) without one."""
    same = [p for p in paces if p.s_per_step and p.atoms == qm_atoms
            and p.cutoff_ry == cutoff_ry]
    pace, factor = pace_for(same, "s_per_step", cores, booking=True)
    if pace is None:
        return None, (f"no coupled step measured here on {qm_atoms} QM atoms at "
                      f"{cutoff_ry} Ry")
    seconds = pace.s_per_step / factor
    return (steps * seconds / 60 * MARGIN,
            f"{steps} steps at {seconds:.3g} s/step, measured on {qm_atoms} QM atoms at "
            f"{cutoff_ry} Ry ({pace.where})")


def booked_minutes(minutes: float, start: float = 2.0) -> int:
    """A walltime as a scheduler is asked for it: ``start`` minutes to start, in 5-min
    steps; a template's fixed walltime is booked as it states it (``start=0``)."""
    return max(5, 5 * math.ceil((minutes + start) / 5))


def price_stage(mandate: Mandate, stage: Stage, *, templates: TemplateStore,
                methods: Methods, atoms: int | None, paces: list[Pace], compute,
                margin: float = MARGIN) -> Price:
    """A stage's booking, one node of the default partition: its walltime from the
    template's fixed walltime, or from its length at the pace measured on this machine
    (the study's own runs first, then the machine's cache), scaled to the system's atoms
    and the node's cores; unmeasured, at ``compute.plausible_ns_per_day``, the fastest a
    run plausibly goes here, so the price is a floor and says so."""
    view = stage.model_dump(mode="json")
    chosen = templates_for(view, templates)
    pricing = next((t.pricing for t in chosen if t.pricing.get("throughput")),
                   next((t.pricing for t in chosen if t.pricing), {}))
    _, billing = compute.billing()
    price = Price(stage=stage.name)
    step_ps = methods.step_ps(stage.regime)
    steps = derived(view, mandate, step_ps)["steps"]
    if steps is None and stage.kind == "single_point":
        steps = 1
    start = 2.0
    if "walltime_min" in pricing:
        minutes, start = float(pricing["walltime_min"]), 0.0
        price.basis = f"{minutes:g} min, the template's walltime for a {stage.kind}"
    elif pricing.get("throughput") == "ns_per_day":
        if not steps or not step_ps:
            price.basis = "unpriced: no length"
            return price
        minutes, price.basis, price.measured = walltime(
            ns=steps * step_ps / 1000.0, atoms=atoms, cores=billing.cores_per_node,
            paces=paces, compute=compute, margin=margin)
    elif pricing.get("throughput") == "s_per_step":
        pace, factor = pace_for(paces, "s_per_step", billing.cores_per_node)
        if pace is None or not steps:
            price.basis = ("unpriced: no QM step measured on this machine yet; priced after "
                           "the first coupled stage")
            return price
        minutes = steps * pace.s_per_step / factor / 60 * margin
        price.basis = f"{steps} steps at {pace.s_per_step / factor:.3g} s/step ({pace.where})"
        price.measured = True
    else:
        price.basis = "unpriced: no template prices this stage"
        return price
    if stage.scan is not None:
        minutes *= len(stage.scan.values)
        price.basis += f", × {len(stage.scan.values)} scan points"
    price.walltime_min = booked_minutes(minutes, start)
    price.core_h = round(price.walltime_min / 60 * billing.billed_per_node_h, 3)
    return price


# --- the mandate as a whole -------------------------------------------------------------


class MandateCompiled(BaseModel):
    """The mandate's stages as the engines are told them, and priced."""

    problems: list[str] = Field(default_factory=list)  # what keeps a stage from binding
    notes: list[str] = Field(default_factory=list)  # what the researcher should know
    lines: dict[str, list[str]] = Field(default_factory=dict)  # stage -> what it sets
    priced: dict[str, str] = Field(default_factory=dict)  # stage -> its price and basis
    unpriced: list[str] = Field(default_factory=list)
    measured: dict[str, str] = Field(default_factory=dict)  # deliverable -> its call
    estimate_core_h: float = 0.0


#: the facts a structure gives about how a run starting from it starts
STARTING_STATE = frozenset({"continuation", "gen-vel"})


def compile_mandate(mandate: Mandate, *, templates: TemplateStore, methods: Methods,
                    facts: dict[str, SystemFact], atoms: int | None, paces: list[Pace],
                    compute) -> MandateCompiled:
    """Every stage compiled by each template that tells an engine it: the keys its
    physical terms set, what only a render supplies, what the stage as stated cannot be
    told at all; and its price on this machine."""
    out = MandateCompiled(notes=list(methods.missing),
                          problems=lock_problems(mandate, templates))
    # the system's structure is where the first stage starts; a later one starts from what
    # the stage before it leaves, which its render reads
    later_facts = {k: v for k, v in facts.items() if k not in STARTING_STATE}
    for at, stage in enumerate(mandate.method.stages):
        view = stage.model_dump(mode="json")
        lines = out.lines.setdefault(stage.name, [])
        found = templates_for(view, templates)
        if not found:
            if templates_for(view, templates, any_ensemble=True):
                out.problems.append(f"stage {stage.name}: no engine here runs a "
                                    f"{stage.regime} {stage.kind} in {stage.ensemble}")
            else:
                lines.append("no engine template tells this stage: its own tool runs it")
            continue
        for template in found:
            res = compile_stage(mandate, stage, template, templates=templates,
                                methods=methods, facts=facts if at == 0 else later_facts)
            forms, _ = flattened(template, templates)
            out.problems += [f"stage {stage.name}: {why}" for why in res.unbuildable.values()]
            later = [k for k in res.missing if k not in res.unbuildable]
            shown = [f"{k} {v}" for k, v in res.values.items()
                     if k in forms and forms[k].depends()]
            lines.append(f"{template.id}: " + (", ".join(shown) or "nothing of the stage's")
                         + (f"; set at render: {', '.join(later)}" if later else ""))
        price = price_stage(mandate, stage, templates=templates, methods=methods, atoms=atoms,
                            paces=paces, compute=compute)
        out.priced[stage.name] = (f"{price.core_h:g} core-h ({price.walltime_min} min): "
                                  if price.core_h is not None else "") + price.basis
        if price.core_h is None:
            out.unpriced.append(stage.name)
        out.estimate_core_h += price.core_h or 0.0
    stages = {s.name for s in mandate.method.stages}
    for d in mandate.deliverables:
        if d.over and stages and d.over not in stages:
            out.problems.append(f"deliverable {d.name} is measured over {d.over!r}, no stage "
                                f"of the method ({', '.join(sorted(stages))})")
        if d.property:
            measured = compile_deliverable(mandate, d)
            out.problems += measured.problems
            out.notes += measured.notes
            out.measured[d.name] = measured.line()
    out.problems = list(dict.fromkeys(out.problems))
    out.estimate_core_h = round(out.estimate_core_h, 3)
    return out


@cache
def decided_keys(engine: str) -> frozenset[str]:
    """The engine keys the harness reasons about: those a shipped stage template states, by
    the name the catalog gives them (an mdp key canonically, a CPMD keyword as
    SECTION/NAME). A file setting any other key away from the engine's default has stepped
    outside what the harness checks, and is told so."""
    catalog = load(engine)
    out: set[str] = set()
    for template in TemplateStore.builtin().all():
        if template.engine != engine:
            continue
        for key in template.parameters:
            entry = engine_key(catalog, key)
            if entry is not None:
                out.add(f"{entry.section}/{entry.name}" if engine == "cpmd"
                        else canonical(entry.name))
    return frozenset(out)
