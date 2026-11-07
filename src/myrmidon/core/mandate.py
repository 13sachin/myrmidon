"""The mandate: the researcher's authority to the model, bounded.

One document per study, `mandate.yaml`, in a research proposal's shape: the question, the
system and the files that are it, the method, what the researcher decided and what the model
proposes on their behalf, the deliverables, the cost and its ceiling, and what the harness may
do without them. The researcher reads it rendered as sentences (:func:`render`) and approves
it once; the submission gate reads the same file afterwards.

Its digest covers the file without the digest and the current contents of every file in
``system.files``: an edited mandate, or a replaced system file, is a different mandate, and
an approval given for the old digest does not hold for it. An approval is a ledger event
(``mandate_approve``) recorded by the researcher's own act — `myrmidon approve`, a client's
dialog, Hermes's `/approve` — never by a tool the model is handed.

Nothing here knows an engine: a file's kind is the contracts' (core/contracts.py), and a
decision's keys are spelled as the target that reads them spells them.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, WithJsonSchema, model_validator

from myrmidon.core.artifacts import ArtifactRef, sha256_cached
from myrmidon.core.provenance import ProvenanceEvent
from myrmidon.core.validation import Finding

MANDATE_FILE = "mandate.yaml"
#: the ledger's name for a researcher's approval of a mandate; not a registered tool
APPROVE_EVENT = "mandate_approve"
MANDATE_ROLE = "mandate"
#: where every revision written is kept, by revision and digest
ARCHIVE_DIR = "mandates"
MAX_QUESTIONS = 3


def _as_list(value: Any) -> Any:
    """A list as a model may write it: one item, null, or a JSON list as text."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
        # one item, or several written in a line: names and numbers hold no space or comma
        return [w for w in text.replace(",", " ").split() if w]
    if isinstance(value, int | float) and not isinstance(value, bool):
        return [value]
    return value


def _listed(items: dict) -> WithJsonSchema:
    """A list's schema that also admits the spellings ``_as_list`` reads."""
    return WithJsonSchema({"anyOf": [{"type": "array", "items": items}, items,
                                     {"type": "null"}]})


#: names: a list of words, or one word
Words = Annotated[list[str], BeforeValidator(_as_list), _listed({"type": "string"})]


def _as_residues(value: Any) -> Any:
    """Residue numbers as a model may write them: a range as "first-last", or as a pair of
    numbers inside the list ([[1, 163]]; a flat [1, 163] is two residues)."""
    return [f"{item[0]}-{item[1]}" if isinstance(item, list) and len(item) == 2
            and all(isinstance(i, int) and not isinstance(i, bool) for i in item) else item
            for item in _as_list(value)]


#: residue numbers: a list of numbers or ranges ("first-last", or [first, last]), or one
Residues = Annotated[list[int | str], BeforeValidator(_as_residues), WithJsonSchema(
    {"anyOf": [{"type": "array", "items": {"anyOf": [
        {"type": "integer"}, {"type": "string"},
        {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}]}},
        {"type": "integer"}, {"type": "string"}, {"type": "null"}]})]


class _Strict(BaseModel):
    """A field the file does not define is an edit nobody made through the harness."""

    model_config = ConfigDict(extra="forbid")


class SystemFile(_Strict):
    path: str
    kind: str
    sha256: str = ""


SubsystemKind = Literal["protein", "ligand", "nucleic", "membrane", "cofactor", "solvent",
                        "ions"]


class Subsystem(_Strict):
    """A named physical part of the system, in the topology's and the structure's own
    vocabulary. ``kind`` is chemical only: which part is treated quantum mechanically is a
    role a stage gives it (``qm_region``)."""

    kind: SubsystemKind
    molecules: Words = Field(default_factory=list)  # moleculetype names
    resnames: Words = Field(default_factory=list)
    chains: Words = Field(default_factory=list)
    # residue numbers as the built structure numbers them; a range is "first-last"
    resid: Residues = Field(default_factory=list)
    charge: float | None = None

    def names(self) -> list[str]:
        """What the topology may call it: its molecule types, then its residue names."""
        return list(dict.fromkeys([*self.molecules, *self.resnames]))


class QM(_Strict):
    method: str  # a method id (knowledge/methods/)
    cutoff_ry: float | None = None


class Coupling(_Strict):
    scheme: str  # a coupling method id (knowledge/methods/)
    multipole_order: int | None = None
    cutoff_bohr: float | None = None


class Hamiltonian(_Strict):
    """How the system's energy is computed: ``mm`` a force-field method id read from the
    topology, ``qm`` and ``coupling`` a QM method and its coupling with the values this
    study calibrates."""

    mm: str = ""
    qm: QM | None = None
    coupling: Coupling | None = None

    @model_validator(mode="before")
    @classmethod
    def _method(cls, data: Any) -> Any:
        """A Hamiltonian named by one word: the force field's method id, or ``mm`` for a
        classical one read from the topology."""
        if isinstance(data, str):
            word = data.strip()
            return {} if word.lower() in ("mm", "classical", "") else {"mm": word}
        return data


class System(_Strict):
    description: str = ""
    files: list[SystemFile] = Field(default_factory=list)
    subsystems: dict[str, Subsystem] = Field(default_factory=dict)
    hamiltonian: Hamiltonian = Field(default_factory=Hamiltonian)


StageKind = Literal["minimize", "dynamics", "single_point", "parameter_fit"]
Regime = Literal["mm", "qmmm", "qm"]
Ensemble = Literal["NVE", "NVT", "NPT"]


class QMRegion(_Strict):
    """The subsystems a stage treats quantum mechanically, with the region's chemistry."""

    subsystems: Words = Field(min_length=1)
    charge: int
    #: unset, what the region's electrons decide, as the QM selection read them
    multiplicity: int | None = Field(default=None, ge=1)
    padding_nm: float | None = None


class Ramp(_Strict):
    """A temperature carried from where the stage the ramping stage starts from ended to
    ``to_k``: the start is a fact of that stage's run, known once it has run."""

    to_k: float
    start_k: float | None = None
    rate_k_per_au: float | None = None


class Scan(_Strict):
    """One value of ``parameter`` per point (``hamiltonian.qm.cutoff_ry``,
    ``<stage>.qm_region.padding_nm``, ``system.topology``)."""

    parameter: str
    values: list[Any] = Field(min_length=1)


class Converge(_Strict):
    observable: str
    tolerance: float
    reference: Any = None


class Stage(_Strict):
    """One physical transition of the system, in physical terms only; what an engine reads
    for it is compiled from these and the system (knowledge/stage.py)."""

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    kind: StageKind
    regime: Regime = "mm"
    #: the stage whose collected run this one continues from: unset, the stage before it; an
    #: earlier one when stages branch; itself to extend a finished run
    starts_from: str = ""
    ensemble: Ensemble | None = None
    temperature_k: float | Ramp | None = None
    pressure_bar: float | None = None
    duration_ps: float | None = Field(default=None, gt=0)
    steps: int | None = Field(default=None, ge=0)
    restraints: Words = Field(default_factory=list)  # subsystems held in place
    sampling_ps: float | None = Field(default=None, gt=0)
    qm_region: QMRegion | None = None
    scan: Scan | None = None
    converge: Converge | None = None

    @model_validator(mode="after")
    def _physical(self) -> Stage:
        said = []
        if self.name in LOCKABLE:
            said.append(f"a stage is not named after a field ({self.name})")
        if self.kind == "dynamics":
            if self.ensemble is None:
                said.append("dynamics runs in an ensemble: NVE, NVT or NPT")
            if self.duration_ps is None and self.steps is None:
                said.append("dynamics has a length: duration_ps or steps")
        if self.duration_ps is not None and self.steps is not None:
            said.append("a length is duration_ps or steps, not both")
        if self.ensemble in ("NVT", "NPT") and self.temperature_k is None:
            said.append(f"{self.ensemble} holds a temperature: temperature_k")
        if self.ensemble == "NPT" and self.pressure_bar is None:
            said.append("NPT holds a pressure: pressure_bar")
        if self.regime in ("qmmm", "qm") and self.qm_region is None:
            said.append(f"a {self.regime} stage names its QM region: qm_region")
        if said:
            raise ValueError(f"stage {self.name}: " + "; ".join(said))
        return self


#: the stage fields a decision may lock
LOCKABLE = ("kind", "regime", "ensemble", "temperature_k", "pressure_bar", "duration_ps",
            "steps", "restraints", "sampling_ps", "qm_region", "scan", "converge")


class Method(_Strict):
    protocol: str = ""
    stages: list[Stage] = Field(default_factory=list)
    #: a schema 1 file's stages, in the words they were written in
    described: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _words(cls, data: Any) -> Any:
        """A schema 1 file says its stages in words: they are kept as words."""
        if isinstance(data, dict) and any(isinstance(x, str) for x in data.get("stages", [])):
            data = {**data, "described": [*data.get("described", []), *data["stages"]],
                    "stages": []}
        return data

    @model_validator(mode="after")
    def _named_once(self) -> Method:
        names = [s.name for s in self.stages]
        twice = sorted({n for n in names if names.count(n) > 1})
        if twice:
            raise ValueError(f"stage name(s) {', '.join(twice)} used twice: name each "
                             "stage once and run it as often as needed")
        for i, stage in enumerate(self.stages):
            if stage.starts_from and stage.starts_from not in [*names[:i], stage.name]:
                raise ValueError(f"stage {stage.name} starts from {stage.starts_from!r}, which "
                                 "is no earlier stage")
            ramp = stage.temperature_k
            if isinstance(ramp, Ramp) and ramp.start_k is None and i == 0:
                raise ValueError(f"stage {stage.name} ramps from where the stage before it "
                                 "ended, and no stage comes before it: state the ramp's start_k")
        return self

    def stage(self, name: str | None) -> Stage | None:
        return next((s for s in self.stages if s.name == name), None)

    def starts_from(self, stage: Stage | str) -> Stage | None:
        """The stage whose collected run this one continues from: the one it names, else the
        one before it; None for the first stage."""
        name = stage if isinstance(stage, str) else stage.name
        names = [s.name for s in self.stages]
        if name not in names:
            return None
        found = self.stages[names.index(name)]
        if found.starts_from:
            return self.stage(found.starts_from)
        at = names.index(name)
        return self.stages[at - 1] if at else None


#: the stage fields whose value is a list: a lock of nothing on one is the empty list
LIST_FIELDS = ("restraints",)


def _locks(value: Any) -> Any:
    """Locks as written, a lock of nothing (None) on a list field read as the empty list."""
    if not isinstance(value, dict):
        return value
    return {path: [] if c is None and lock_path(path)[1] in LIST_FIELDS else c
            for path, c in value.items()}


#: what a decision or proposal locks, by stage-field path
Locks = Annotated[dict[str, Any], BeforeValidator(_locks)]


class Decision(_Strict):
    """What the researcher decided, in their words (``said``), and what it locks: stage
    fields by path in ``locks`` (``production.duration_ps``, ``nvt.restraints``, or
    ``*.temperature_k`` / ``temperature_k`` for every stage), or engine keys on a target in
    ``keys`` (a tool's parameter, the backend's booking, ``*`` for every parameter file).
    A constraint is a scalar (equal), a list (one of; a list field's value as a whole) or
    ``{min, max}`` (within)."""

    said: str
    locks: Locks = Field(default_factory=dict)
    target: str = "*"
    keys: dict[str, Any] = Field(default_factory=dict)


class Proposal(_Strict):
    """What the model proposes on the researcher's behalf, with where it comes from."""

    what: str
    # what the proposal would fix, as a decision would lock it: stage fields by path, or
    # keys on a target
    locks: Locks = Field(default_factory=dict)
    target: str = "*"
    keys: dict[str, Any] = Field(default_factory=dict)
    source: str = ""


class Decisions(_Strict):
    yours: list[Decision] = Field(default_factory=list)
    proposed: list[Proposal] = Field(default_factory=list)


AtomFilter = Literal["heavy", "backbone", "ca", "all"]
Statistic = Literal["mean", "max", "min", "sd", "final"]


class Align(_Strict):
    """The atoms every frame is superposed on before a fluctuation is measured."""

    subsystem: str
    atoms: AtomFilter = "backbone"


class Deliverable(_Strict):
    """A number the report will contain: in physical terms, an observable (``property``)
    of a ``target`` subsystem — against a ``reference`` subsystem for a relation between
    two —, its atoms filtered alike on every side, superposed on ``align_to`` from
    ``reference_frame``, summarised over the run of stage ``over`` by ``statistics``; or, as
    a schema 1 file states it, the measuring ``tool`` and the ``params`` that define it."""

    name: str
    property: str = ""
    target: str = ""
    reference: str = ""
    atoms: AtomFilter | None = None  # heavy, unless stated
    align_to: Align | None = None
    reference_frame: str | None = None  # stage_start, unless stated; "minimised"; a path
    over: str = ""
    statistics: list[Statistic] = Field(default_factory=list)
    tool: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class Cost(_Strict):
    estimate_core_h: float = 0.0
    ceiling_core_h: float = 0.0
    #: stages the estimate leaves out: not priced yet (each one's reason is in the render)
    unpriced: list[str] = Field(default_factory=list)


def estimate_words(cost: Cost) -> str:
    """The estimate as every approval says it, with the stages it leaves out; "" for a
    mandate nothing prices at all."""
    left = ", ".join(cost.unpriced)
    if cost.estimate_core_h:
        return (f"estimated {cost.estimate_core_h:g} core-h"
                + (f" without {left}, which nothing prices yet" if left else ""))
    return f"not estimated: {left} not priced yet" if left else ""


class Recovery(_Strict):
    reruns: int = 1
    model_fixes: int = 2
    deadline_h: float | None = None


class Mandate(_Strict):
    model_config = ConfigDict(populate_by_name=True)

    schema_version: int = Field(default=2, alias="schema")
    question: str
    system: System = Field(default_factory=System)
    method: Method = Field(default_factory=Method)
    decisions: Decisions = Field(default_factory=Decisions)
    deliverables: list[Deliverable] = Field(default_factory=list)
    cost: Cost = Field(default_factory=Cost)
    recovery: Recovery = Field(default_factory=Recovery)
    status: Literal["ready", "needs_input"] = "needs_input"
    questions: list[str] = Field(default_factory=list, max_length=MAX_QUESTIONS)
    revision: int = 1
    #: why this revision was written, when it revises an earlier one
    reason: str = ""
    digest: str = ""

    def body(self) -> dict:
        """The file's fields without the digest, in the order the file has them. What a
        field leaves unset is left out, so the file says only what was stated, and a
        mandate written before a field existed holds the digest it was approved with."""
        body = self.model_dump(mode="json", by_alias=True, exclude={"digest"})
        if self.schema_version >= 2:
            return _pruned(body)
        # a schema 1 file is hashed as it was written then: every field, none of the later
        for deliverable in body.get("deliverables", []):
            if not deliverable.get("params"):
                deliverable.pop("params", None)
            for later in ("property", "target", "reference", "atoms", "align_to",
                          "reference_frame", "statistics"):
                deliverable.pop(later, None)
        for decision in [*body["decisions"]["yours"], *body["decisions"]["proposed"]]:
            decision.pop("locks", None)
        body["system"].pop("subsystems", None)
        body["system"].pop("hamiltonian", None)
        body["cost"].pop("unpriced", None)
        body["method"]["stages"] = body["method"].pop("described")
        return body


#: fields a mandate keeps even when empty: the file's own shape
_KEPT = {"questions", "files", "stages", "deliverables", "yours", "proposed", "restraints",
         "reason", "protocol", "description"}


def _pruned(value, key: str = ""):
    """``value`` without the None and empty fields nobody stated; what a decision locks is
    stated, empty or not."""
    if key in ("locks", "keys"):
        return value
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            v = _pruned(v, k)
            if v is None or (v in ({}, [], "") and k not in _KEPT):
                continue
            out[k] = v
        return out
    if isinstance(value, list):
        return [_pruned(v) for v in value]
    return value


def lock_path(path: str) -> tuple[str, str]:
    """(stage name or ``*``, field path) of a lock's path: ``production.duration_ps``,
    ``*.temperature_k``, ``temperature_k``, ``nvt.qm_region.charge``."""
    head, _, rest = path.partition(".")
    if head in LOCKABLE:
        return "*", path
    return head, rest


def field_value(stage: Stage | dict, field: str):
    """A stage's value at a field path, as JSON: ``qm_region.charge``, ``restraints``."""
    value: Any = stage.model_dump(mode="json") if isinstance(stage, Stage) else stage
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def holds(value, constraint) -> bool:
    """Whether a stage's value meets a lock: a range on a number, the same items for a list
    field, every stated item for a mapping, one of a list for a scalar, else equal."""
    if isinstance(constraint, dict) and constraint and set(constraint) <= {"min", "max"}:
        if not isinstance(value, int | float) or isinstance(value, bool):
            return False
        lo, hi = constraint.get("min"), constraint.get("max")
        return (lo is None or value >= lo) and (hi is None or value <= hi)
    if isinstance(constraint, dict):
        return isinstance(value, dict) and all(holds(value.get(k), c)
                                               for k, c in constraint.items())
    if isinstance(value, list):
        return isinstance(constraint, list) and sorted(map(str, value)) == sorted(
            map(str, constraint))
    if isinstance(constraint, list):
        return any(holds(value, c) for c in constraint)
    if isinstance(value, int | float) and isinstance(constraint, int | float):
        return abs(value - constraint) <= 1e-9 * max(1.0, abs(constraint))
    return value == constraint


def locks_on(mandate: Mandate, stage: str | None) -> list[tuple[str, Any, str]]:
    """(field, constraint, said) of every lock on ``stage``: its own and every stage's
    (``*``); for a stage the method does not name, every stage's only."""
    named = mandate.method.stage(stage) is not None
    out = []
    for decision in mandate.decisions.yours:
        for path, constraint in decision.locks.items():
            on, field = lock_path(path)
            if on == "*" or (named and on == stage):
                out.append((field, constraint, decision.said))
    return out


def lock_findings(mandate: Mandate) -> list[str]:
    """What keeps the decisions' locks from binding the method as written: a stage the
    method does not name, a field a stage does not have, a stage whose value breaks a lock
    on it."""
    stages = {s.name: s for s in mandate.method.stages}
    out = []
    for decision in mandate.decisions.yours:
        for path, constraint in decision.locks.items():
            on, field = lock_path(path)
            root = field.split(".")[0]
            if on != "*" and on not in stages:
                out.append(f"\"{decision.said}\" locks {path}: no stage {on!r} in the "
                           f"method (stages: {', '.join(stages) or 'none'})")
                continue
            if root not in LOCKABLE:
                out.append(f"\"{decision.said}\" locks {path}: a stage has no field "
                           f"{root!r} (fields: {', '.join(LOCKABLE)})")
                continue
            for stage in ([stages[on]] if on != "*" else stages.values()):
                value = field_value(stage, field)
                if on == "*" and value is None:
                    continue  # every stage's lock binds the stages that have the field
                if not holds(value, constraint):
                    out.append(f"\"{decision.said}\" locks {path} to {_constraint(constraint)}"
                               f", and stage {stage.name} has {value}: make them agree")
    return out


def _file(ws, path: str) -> Path:
    return ws.resolve(path)


def file_hash(ws, path: str) -> str | None:
    """A system file's sha256 now; None when it is gone."""
    try:
        return sha256_cached(_file(ws, path))
    except OSError:
        return None


def compute_digest(ws, mandate: Mandate) -> str:
    """sha256 over the file without its digest and the current contents of every system
    file, by their own hashes, in the order the file lists them."""
    h = hashlib.sha256(json.dumps(mandate.body(), sort_keys=True).encode())
    for f in mandate.system.files:
        h.update(f"\n{f.path}\0{file_hash(ws, f.path) or 'missing'}".encode())
    return h.hexdigest()


def read_mandate(ws) -> Mandate | None:
    """The mandate on disk; None when the study has none or it does not parse."""
    if ws is None:
        return None
    path = ws.root / MANDATE_FILE
    if not path.is_file():
        return None
    try:
        return Mandate.model_validate(yaml.safe_load(path.read_text()) or {})
    except (OSError, ValueError, yaml.YAMLError):
        return None


def write_mandate(ws, mandate: Mandate) -> Mandate:
    """Hash the system files, compute the digest, write `mandate.yaml`, and keep a copy of
    the revision under the study's state directory."""
    files = [f.model_copy(update={"sha256": file_hash(ws, f.path) or ""})
             for f in mandate.system.files]
    mandate = mandate.model_copy(update={
        "system": mandate.system.model_copy(update={"files": files})})
    mandate = mandate.model_copy(update={"digest": compute_digest(ws, mandate)})
    text = yaml.safe_dump({**mandate.body(), "digest": mandate.digest}, sort_keys=False,
                          allow_unicode=True)
    (ws.root / MANDATE_FILE).write_text(text)
    archive = ws.state_dir / ARCHIVE_DIR
    archive.mkdir(exist_ok=True)
    (archive / f"r{mandate.revision}-{mandate.digest[:12]}.yaml").write_text(text)
    return mandate


class MandateApproval(BaseModel):
    event_id: str
    digest: str
    approver: str
    via: str = ""
    timestamp: datetime | None = None


def approvals(ws) -> list[MandateApproval]:
    """Every approval of a mandate this study's ledger records, oldest first."""
    if ws is None:
        return []
    return [MandateApproval(event_id=ev.event_id, digest=str(ev.params.get("digest", "")),
                            approver=str(ev.params.get("approver", "")),
                            via=str(ev.params.get("via", "")), timestamp=ev.timestamp)
            for ev in ws.ledger.events()
            if ev.tool == APPROVE_EVENT and ev.ok and ev.kind == "call"]


def approved_digests(ws) -> set[str]:
    return {a.digest for a in approvals(ws)}


def holding(ws) -> Mandate | None:
    """The mandate on disk when an approval of exactly it holds: it is ready, its digest is
    the one computed now from the file and the system files, and that digest was approved."""
    mandate = read_mandate(ws)
    if mandate is None or mandate.status != "ready" or not mandate.digest:
        return None
    if compute_digest(ws, mandate) != mandate.digest:
        return None
    return mandate if mandate.digest in approved_digests(ws) else None


def why_not_holding(ws) -> str:
    """Why no mandate holds, in one sentence; "" when one does."""
    mandate = read_mandate(ws)
    if mandate is None:
        if ws is not None and (ws.root / MANDATE_FILE).is_file():
            return f"{MANDATE_FILE} does not parse: write it again with mandate_write"
        return "this study has no mandate: mandate_write, then the researcher approves it"
    if mandate.status != "ready":
        return "the mandate waits on the researcher's answers to its questions"
    # an approved mandate changes by a revision; a draft is simply written again
    rewrite = ("mandate_write with revise=true and the reason, which the researcher "
               "approves" if mandate.digest in approved_digests(ws) else "mandate_write")
    changed = [f.path for f in mandate.system.files if file_hash(ws, f.path) != f.sha256]
    if changed:
        return (f"the system file(s) {', '.join(changed)} changed since the mandate was "
                f"written: write it again over the system as it is ({rewrite})")
    if compute_digest(ws, mandate) != mandate.digest:
        return f"{MANDATE_FILE} was edited by hand: write it with {rewrite}"
    if mandate.digest not in approved_digests(ws):
        return (f"mandate revision {mandate.revision} ({mandate.digest[:12]}) is not "
                "approved yet: the researcher approves it")
    return ""


def record_approval(ws, mandate: Mandate, *, approver: str, via: str, origin: str = "",
                    myrmidon_version: str = "") -> str:
    """The researcher's yes, in the ledger: the mandate file and every system file, hashed,
    under the digest they read. Returns the event id."""
    inputs = [ArtifactRef.from_path(ws.root / MANDATE_FILE, MANDATE_ROLE, ws.root)]
    for f in mandate.system.files:
        path = _file(ws, f.path)
        inputs.append(ArtifactRef(path=f.path, role=f.kind, sha256=f.sha256,
                                  size_bytes=path.stat().st_size if path.is_file() else 0))
    event = ProvenanceEvent(
        tool=APPROVE_EVENT,
        params={"digest": mandate.digest, "approver": approver, "via": via,
                "origin": origin, "revision": mandate.revision,
                "ceiling_core_h": mandate.cost.ceiling_core_h},
        inputs=inputs, myrmidon_version=myrmidon_version, kind="call", ok=True,
        timestamp=datetime.now(UTC), mandate=mandate.digest)
    return ws.ledger.append(event)


def _constraint(value: Any) -> str:
    if isinstance(value, dict) and set(value) <= {"min", "max"}:
        lo, hi = value.get("min"), value.get("max")
        return (f"between {lo} and {hi}" if lo is not None and hi is not None
                else f"at least {lo}" if lo is not None else f"at most {hi}")
    if value == []:
        return "none"
    if isinstance(value, list):
        return "one of " + ", ".join(str(v) for v in value)
    return str(value)


def _keys(keys: dict[str, Any]) -> str:
    return ", ".join(f"{k} {_constraint(v)}" for k, v in keys.items())


def _on(target: str) -> str:
    return "" if target in ("", "*") else f" ({target})"


def deliverable_findings(ws, tool: str, params: dict) -> list[Finding]:
    """What a call may measure otherwise than a deliverable of the holding mandate states:
    the deliverables of this tool whose first parameter the call shares and whose other
    parameters it sets otherwise — said only when the call matches none of them wholly,
    since a call can measure one deliverable exactly and another one never."""
    mandate = read_mandate(ws) if ws is not None else None
    if mandate is None:
        return []
    return [*_measured_otherwise(ws, mandate, tool, params),
            *_measured_on(ws, mandate, tool, params)]


def _matches(d: Deliverable, tool: str, params: dict) -> bool:
    """Whether a call measures ``d``: its tool, with every parameter that defines it."""
    return (d.tool.split()[0:1] == [tool] and bool(d.params)
            and all(params.get(k) == v for k, v in d.params.items()))


def _measured_on(ws, mandate: Mandate, tool: str, params: dict) -> list[Finding]:
    """A deliverable over a stage the method names, measured on a run's output that no run
    of that stage wrote: an ad-hoc run, or another stage's, does not stand in for it."""
    from myrmidon.core.artifacts import consumed_inputs
    from myrmidon.core.lineage import lineage

    mine = [d for d in mandate.deliverables
            if d.over and mandate.method.stage(d.over) is not None and _matches(d, tool, params)]
    if not mine or holding(ws) is None:
        return []
    lin = lineage(ws)
    runs = [(ref.path, lin.stages_behind(ref.sha256))
            for ref in consumed_inputs(params, ws.root, ws.origin)
            if any(ev.event_id in lin.submissions for ev in lin.writers.get(ref.sha256, ()))]
    out = []
    for d in mine:
        for path, stages in runs:
            if d.over in stages:
                continue
            made = (f"the run of stage {', '.join(sorted(stages))}" if stages
                    else "a run of no stage the mandate names")
            out.append(Finding(
                id="DELIVERABLE_STAGE_MISMATCH",
                message=f"if this call measures {d.name!r}, it measures it on {path}, which "
                f"{made} wrote, not stage {d.over}",
                suggestion=f"render {d.over}'s file with inputset_apply stage={d.over}, run "
                "it, and measure its output"))
    return out


def _measured_otherwise(ws, mandate: Mandate, tool: str, params: dict) -> list[Finding]:
    mine = [d for d in mandate.deliverables if d.params and d.tool.split()[0:1] == [tool]]
    if not mine:
        return []
    same = [d for d in mine if params.get(next(iter(d.params))) == next(iter(d.params.values()))]
    if not same or any(all(params.get(k) == v for k, v in d.params.items()) for d in same):
        return []
    if holding(ws) is None:  # what an unapproved mandate states binds nothing yet
        return []
    out: list[Finding] = []
    for d in same:
        differ = [f"{k} {params.get(k)!r}, not {v!r}" for k, v in d.params.items()
                  if params.get(k) != v]
        out.append(Finding(
            id="DELIVERABLE_MEASURED_OTHERWISE",
            message=f"if this call measures {d.name!r}, it measures it otherwise than the "
            "approved mandate states: " + "; ".join(differ),
            suggestion="measure it as the mandate states, or revise the mandate with the "
            "reason; a different measurement needs nothing"))
    return out


def stage_words(stage: Stage) -> str:
    """A stage in physical words: ``dynamics NPT, 300 K, 1 bar, 1000 ps, restrained:
    protein, ligand``."""
    bits = [f"{stage.kind}" + (f" {stage.ensemble}" if stage.ensemble else "")
            + (f" ({stage.regime})" if stage.regime != "mm" else "")]
    temp = stage.temperature_k
    if isinstance(temp, Ramp):
        begins = stage.starts_from or "the stage before it"
        bits.append(f"from where {begins} ends to {temp.to_k:g} K")
    elif temp is not None:
        bits.append(f"{temp:g} K")
    if stage.pressure_bar is not None:
        bits.append(f"{stage.pressure_bar:g} bar")
    if stage.duration_ps is not None:
        bits.append(f"{stage.duration_ps:g} ps")
    if stage.steps is not None:
        bits.append(f"{stage.steps} steps")
    if stage.sampling_ps is not None:
        bits.append(f"a frame every {stage.sampling_ps:g} ps")
    bits.append("restrained: " + ", ".join(stage.restraints) if stage.restraints
                else "unrestrained")
    if stage.qm_region is not None:
        q = stage.qm_region
        bits.append(f"QM: {', '.join(q.subsystems)}, charge {q.charge}"
                    + (f", multiplicity {q.multiplicity}" if q.multiplicity is not None else "")
                    + (f", padding {q.padding_nm:g} nm" if q.padding_nm is not None else ""))
    if stage.scan is not None:
        bits.append(f"scanning {stage.scan.parameter} over "
                    + ", ".join(str(v) for v in stage.scan.values))
    if stage.converge is not None:
        bits.append(f"converged on {stage.converge.observable} to "
                    f"{stage.converge.tolerance:g}")
    return ", ".join(bits)


def deliverable_words(d: Deliverable) -> str:
    """A deliverable in physical words: ``rmsd of ligand (heavy atoms), superposed on
    protein (backbone) from the stage's first frame, over production: mean, max``."""
    if not d.target:  # a property of the run as a whole
        bits = [d.property]
    else:
        atoms = d.atoms or "heavy"
        what = f"{d.property} of {d.target}" + (f" to {d.reference}" if d.reference else "")
        bits = [f"{what} ({atoms} atoms)" if atoms != "all" else what]
    start = d.reference_frame or "stage_start"
    if d.align_to is not None:
        bits.append(f"superposed on {d.align_to.subsystem} ({d.align_to.atoms} atoms)")
    if d.align_to is not None or start != "stage_start":
        bits.append("from " + ("the stage's first frame" if start == "stage_start"
                               else start))
    return ", ".join(bits) + (f", over {d.over}" if d.over else "") + (
        f": {', '.join(d.statistics)}" if d.statistics else "")


def render(mandate: Mandate, spent_core_h: float = 0.0,
           compiled: dict[str, list[str]] | None = None,
           priced: dict[str, str] | None = None,
           measured: dict[str, str] | None = None) -> str:
    """The mandate as the researcher reads it: seven sections, every machine field said,
    nothing said that the file does not hold. ``compiled`` says, per stage, what the
    engines are told for it, ``priced`` what it costs and why, and ``measured``, per
    deliverable, the call that measures it, as the caller compiled them."""
    pad = " " * 14
    lines = [f"QUESTION      {mandate.question}", ""]
    lines.append(f"SYSTEM        {mandate.system.description or '(not described)'}")
    if mandate.system.files:
        lines.append(pad + "files: " + "  ".join(
            f"{f.path} [{f.kind}]" for f in mandate.system.files))
    for name, sub in mandate.system.subsystems.items():
        lines.append(f"{pad}{name}: {sub.kind}, " + (", ".join(sub.names()) or "unnamed")
                     + (f", charge {sub.charge:g}" if sub.charge is not None else ""))
    ham = mandate.system.hamiltonian
    if ham.mm or ham.qm or ham.coupling:
        lines.append(pad + "energy: " + "; ".join(x for x in (
            f"MM {ham.mm}" if ham.mm else "",
            f"QM {ham.qm.method}" + (f" at {ham.qm.cutoff_ry:g} Ry" if ham.qm and
                                    ham.qm.cutoff_ry else "") if ham.qm else "",
            f"coupled by {ham.coupling.scheme}" if ham.coupling else "") if x))
    lines.append("")
    method = mandate.method.protocol or "(no published protocol named)"
    lines.append(f"METHOD        {method}")
    if mandate.method.described:
        lines.append(pad + "  →  ".join(mandate.method.described))
    for stage in mandate.method.stages:
        lines.append(f"{pad}{stage.name}: {stage_words(stage)}")
        for line in (compiled or {}).get(stage.name, []):
            lines.append(f"{pad}    → {line}")
    lines.append("")
    yours = mandate.decisions.yours
    proposed = mandate.decisions.proposed
    lines.append("DECISIONS     yours" + ("" if yours else "      (none recorded)"))
    for d in yours:
        locked = _keys({**d.locks, **d.keys})
        lines.append(f"{pad}  \"{d.said}\"  →  {locked or 'nothing locked'}"
                     + (_on(d.target) if d.keys else ""))
    lines.append(pad + "proposed" + ("" if proposed else "   (none)"))
    for p in proposed:
        detail = "".join(x for x in (f"  →  would lock {_keys(p.locks)}" if p.locks else "",
                                     f"  →  {_keys(p.keys)}" if p.keys else ""))
        source = f"  ({p.source})" if p.source else ""
        lines.append(f"{pad}  {p.what}{detail}{_on(p.target)}{source}")
    lines.append("")
    lines.append("DELIVERABLES  " + ("" if mandate.deliverables else "(none named)"))
    for d in mandate.deliverables:
        if d.property:
            lines.append(f"{pad}{d.name}: {deliverable_words(d)}")
        else:
            how = ", ".join(x for x in (f"by {d.tool}" if d.tool else "",
                                        f"over {d.over}" if d.over else "") if x)
            lines.append(f"{pad}{d.name}" + (f" ({how})" if how else ""))
        if d.params:
            lines.append(f"{pad}  measured as "
                         + ", ".join(f"{k} {json.dumps(v)}" for k, v in d.params.items()))
        if (measured or {}).get(d.name):
            lines.append(f"{pad}    → {measured[d.name]}")
    lines.append("")
    est = estimate_words(mandate.cost) or "not priced"
    lines.append(f"COST          {est}; your ceiling {mandate.cost.ceiling_core_h:g} core-h"
                 + (f"; {spent_core_h:g} core-h already booked in this study"
                    if spent_core_h else "") + ".")
    for name, basis in (priced or {}).items():
        lines.append(f"{pad}{name}: {basis}")
    lines.append("")
    r = mandate.recovery
    deadline = f"stop and ask after {r.deadline_h:g} h or at the ceiling" \
        if r.deadline_h else "stop and ask at the ceiling"
    lines.append(f"RECOVERY      {r.reruns} identical rerun(s) of a failed job; up to "
                 f"{r.model_fixes} change(s) by the model outside your decisions; {deadline}.")
    if mandate.status == "needs_input":
        lines += ["", "QUESTIONS     " + (mandate.questions[0] if mandate.questions else "")]
        lines += [pad + q for q in mandate.questions[1:]]
    lines += ["", f"revision {mandate.revision} · digest {mandate.digest}"
              + (f" · {mandate.reason}" if mandate.reason else "")]
    return "\n".join(lines)


def view(ws, mandate: Mandate | None, spent_core_h: float = 0.0,
         compiled: dict[str, list[str]] | None = None,
         priced: dict[str, str] | None = None,
         measured: dict[str, str] | None = None) -> dict:
    """What an approval surface shows and checks: the rendering (with the stages as the
    engines are told them, and their prices, when given), the numbers, the digest, and
    what keeps it from being approved now ("" when nothing does)."""
    blocked = ""
    if mandate is None:
        why = why_not_holding(ws)
        return {"kind": "mandate", "study": str(ws.root), "text": why, "digest": "",
                "blocked": why}
    if mandate.status != "ready":
        blocked = "it waits on the researcher's answers: " + " ".join(mandate.questions)
    elif compute_digest(ws, mandate) != mandate.digest:
        blocked = why_not_holding(ws)
    return {
        "kind": "mandate",
        "study": str(ws.root),
        "question": mandate.question,
        "text": render(mandate, spent_core_h, compiled, priced, measured),
        "digest": mandate.digest,
        "revision": mandate.revision,
        "status": mandate.status,
        "questions": mandate.questions,
        "estimate_core_h": mandate.cost.estimate_core_h,
        "estimate": estimate_words(mandate.cost),
        "ceiling_core_h": mandate.cost.ceiling_core_h,
        "spent_core_h": spent_core_h,
        "approved": mandate.digest in approved_digests(ws),
        "blocked": blocked,
    }
