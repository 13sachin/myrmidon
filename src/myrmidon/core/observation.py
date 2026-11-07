"""The observation: the feedback contract between harness and model.

Every tool returns one Observation — never raw stdout. Beside the fields every result has,
a tool states its own data as keyword fields (``Observation(ok=True, summary=..., n_atoms=
33497)``); the model is shown them under ``data``. Values in it are computed by harness code
parsing engine output files; the model composes interpretation around them but is never
their source.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.validation import Finding


class ToolError(BaseModel):
    """Why a call did not do what it was asked, in the words of what refused it: a code a
    reader can match, the message, and the one move that would change the outcome."""

    code: str
    message: str
    hint: str = ""


class Observation(BaseModel):
    # a tool's own data rides as extra fields; one it did not state reads as None
    model_config = ConfigDict(extra="allow")

    ok: bool
    summary: str  # 1-3 lines, model-facing
    error: ToolError | None = None  # set when the call was refused or could not run
    # what the engines' checks and the harness noticed, each with what it means: read,
    # decide, and record the reason; a refusal carries its stopping finding first
    findings: list[Finding] = Field(default_factory=list)
    # values the harness derived from the system's files instead of asking for them
    facts: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)  # produced (outputs)
    inputs: list[ArtifactRef] = Field(default_factory=list)  # consumed, hashed
    provenance_id: str | None = None
    command: list[str] | None = None  # exact argv when a binary ran
    log_tail: str | None = None  # truncated, only on failure
    # Harness-only hint, never model-visible: False = do not write a ledger
    # event for this call (unchanged job polls). Failures are always recorded.
    record: bool = Field(default=True, exclude=True)
    # Harness-only: the ledger event kind this call writes. "call" is an action;
    # "observe" is a look at something already running, which a summary counts
    # apart from the work.
    kind: str = Field(default="call", exclude=True)
    # Harness-only: why the registry could not carry the call out — "invalid_params",
    # "no_workspace", "gate_refused" or "raised" — and None when the tool ran, whatever it
    # reported. A diagnosis of a failed run is ok=False and ran: it is the right behaviour,
    # not a failed call.
    failure: str | None = Field(default=None, exclude=True)
    # Harness-only: durable knowledge this call drew on beyond its findings, recorded in
    # the event's knowledge_refs (an input set rendered, as `inputset:<id>@v<N>#<sha256>`)
    refs: list[str] = Field(default_factory=list, exclude=True)
    # Harness-only: data fields the model is not shown — kept for the code and the record,
    # and on disk beside the file the call wrote (a sidecar, the report)
    quiet: set[str] = Field(default_factory=set, exclude=True)

    @field_validator("findings", mode="before")
    @classmethod
    def _kinds(cls, value):
        # read back from a record, each finding keeps the kind its severity names
        return [Finding.read(f) for f in value] if isinstance(value, list) else value

    def model_post_init(self, _context) -> None:
        for name, value in (self.model_extra or {}).items():
            self.model_extra[name] = _plain(value)

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, _plain(value) if name not in type(self).model_fields
                            else value)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith("_"):
                raise
            return None  # a data field this tool did not state


def _plain(value):
    """A numpy value as the Python number or list it is, anywhere in a data field: a typed
    field once coerced it, and the JSON a model is shown cannot carry it."""
    if type(value).__module__ == "numpy" and hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def inputset_ref(set_id: str, version: int, sha256: str) -> str:
    """How a ledger event names the input set a file was rendered from."""
    return f"inputset:{set_id}@v{version}#{sha256}"


def parse_inputset_ref(ref: str) -> tuple[str, int, str] | None:
    """(id, version, sha256) of an ``inputset:`` ref; None for any other."""
    if not ref.startswith("inputset:"):
        return None
    body, _, sha = ref.removeprefix("inputset:").partition("#")
    set_id, _, version = body.partition("@v")
    try:
        return set_id, int(version), sha
    except ValueError:
        return None


def _strip_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_empty(v) for k, v in value.items()
                if v is not None and v != [] and v != {} and v != ""}
    if isinstance(value, list):
        return [_strip_empty(x) for x in value]
    return value


def model_view(obs: Observation) -> dict:
    """What a model is shown of an observation: the fields every result has, the tool's own
    under ``data``, without the empty lists and maps a result carries by default. Presence
    still means something — an empty list a tool sets on purpose is indistinguishable from
    the default, and was never information. The transcript keeps the full dump."""
    dump = obs.model_dump(mode="json", exclude_none=True)
    data = {k: dump.pop(k) for k in list(dump) if k not in type(obs).model_fields
            and k not in obs.quiet}
    for k in obs.quiet:
        dump.pop(k, None)
    if data:
        dump["data"] = data
    return _strip_empty(dump)


class RunProgress(BaseModel):
    step: int | None = None
    time_ps: float | None = None
    ns_per_day: float | None = None


class JobState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    # a scheduler kill at the walltime: a failure whose cause is already known, and the
    # one state that says so without reading the log
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"

    @property
    def terminal(self) -> bool:
        return self in (
            JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT,
        )


class ToolTally(BaseModel):
    tool: str
    calls: int
    failed: int = 0


class JobTally(BaseModel):
    job_id: str
    label: str | None = None
    state: str
    submitted_at: str
    core_h_estimate: float | None = None
    core_h_actual: float | None = None


class EnergyTermStats(BaseModel):
    unit: str
    mean: float
    std: float
    drift_per_ns: float | None = None  # slope of linear fit
    n_points: int


class InputSetSource(BaseModel):
    kind: str
    text: str
    locator: str | None = None
    sha256: str | None = None
    evidence: list[str] = Field(default_factory=list)


class InputSetHit(BaseModel):
    id: str
    title: str
    applicability: str
    system_class: str = ""  # the class of system its values hold for: read it before using
    scope: str = "project"  # what the inputset claims to be true for
    layer: str | None = None  # which layer it was actually found in
    version: int = 1
    renders: str | None = None  # the file kind these values are written into
    parameters: dict = Field(default_factory=dict)
    # What the input set refuses to supply and how to obtain it, and why each value it does
    # supply is what it is. Without these a caller can only discover a value's reason by
    # overriding it and reading the response — that is, by trying it.
    required_from_user: dict[str, str] = Field(default_factory=dict)
    rationale: dict[str, str] = Field(default_factory=dict)
    derivable: dict[str, str] = Field(default_factory=dict)
    sources: list[InputSetSource] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)  # the v1 text view
    pitfalls: list[str] = Field(default_factory=list)  # finding ids that bite this recipe
    validated_by: list[str] = Field(default_factory=list)
    notes: str = ""


class ProtocolHit(BaseModel):
    """One published pipeline, on the wire."""

    id: str
    title: str
    applicability: str
    system_class: str = ""  # the class of system its values hold for: read it before using
    scope: str = "project"
    layer: str | None = None
    version: int = 1
    given: list[str] = Field(default_factory=list)  # the file kinds the researcher brings
    # its stages as a mandate's method states them, each with its price here
    stages: list[dict] = Field(default_factory=list)
    deliverables: list[dict] = Field(default_factory=list)
    # the QM method and coupling its QM/MM stages were measured with, as a mandate's
    # hamiltonian states them
    hamiltonian: dict | None = None
    total_core_h: float = 0.0  # its stages priced on this machine, for no system in particular
    unpriced: list[str] = Field(default_factory=list)  # stages that total leaves out
    citations: list[str] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    notes: str = ""


class ReportValue(BaseModel):
    """One parsed value in a study report, with where it came from."""

    step: str
    label: str
    value: float | int | bool | str
    unit: str = ""
    event: str  # the ledger event of the call that produced it
    file: str  # the output it was parsed from


class SkillHit(BaseModel):
    name: str
    description: str = ""
    question: str = ""  # the researcher's sentence this skill is for


