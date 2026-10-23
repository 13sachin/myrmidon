"""Typed observations: the feedback contract between harness and model.

Every tool returns an Observation — never raw stdout. Values inside an
Observation are computed by harness code parsing engine output files; the
model composes interpretation around them but is never their source.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.validation import ValidationWarning


class NextAction(BaseModel):
    """A step the harness knows is reachable from this observation, typed so the model need
    not dig it out of prose. Generated only where reachability is deterministic — a gate
    refusal, a job that is still running, a missing input, a file a registered parser
    reads — never a choice of physics and never an approval: an acknowledgement appears in
    ``known_parameters`` only when the study's recorded approval already accepts it."""

    kind: Literal["call", "wait", "ask_user", "retry", "escalate"]
    reason: str
    tool: str | None = None
    # the call's parameters as far as the harness knows them; a retry carries the whole call
    known_parameters: dict = Field(default_factory=dict)
    # parameters (or artifact roles) still to be given; for a blocking finding, any one of
    # them resolves it
    missing: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    needs_human: bool = False  # only a researcher may take this step


class Observation(BaseModel):
    ok: bool
    summary: str  # 1-3 lines, model-facing
    # what can be done next, where the harness knows it (NextAction); first after the
    # summary so a truncated result still carries it
    next_actions: list[NextAction] | None = None
    warnings: list[ValidationWarning] = Field(default_factory=list)
    # info-severity gate findings, carried even when the gate let the call through:
    # a finding that only appears on a refusal cannot inform anything
    advisory: list[ValidationWarning] = Field(default_factory=list)
    checked: list[str] = Field(default_factory=list)  # gate checks that ran — no silent skips
    artifacts: list[ArtifactRef] = Field(default_factory=list)  # produced (outputs)
    inputs: list[ArtifactRef] = Field(default_factory=list)  # consumed, hashed
    provenance_id: str | None = None
    command: list[str] | None = None  # exact argv when a binary ran
    log_tail: str | None = None  # truncated, only on failure
    # Harness-only hint, never model-visible: False = do not write a ledger
    # event for this call (unchanged job polls). Failures are always recorded.
    record: bool = Field(default=True, exclude=True)
    # Harness-only: the ledger event kind this call writes. "call" is an action;
    # "observe" is a look at something already running, which a replay re-does by
    # waiting and a summary counts apart.
    kind: str = Field(default="call", exclude=True)
    # Harness-only: why the registry could not carry the call out — "invalid_params",
    # "no_workspace", "gate_refused" or "raised" — and None when the tool ran, whatever it
    # reported. A diagnosis of a failed run is ok=False and ran: it is the right behaviour,
    # not a failed call.
    failure: str | None = Field(default=None, exclude=True)

def model_view(obs: Observation) -> dict:
    """What a model is shown of an observation: its fields, without the empty lists and maps
    every result carries by default (`warnings: []`, `artifacts: []`, ...). Presence still
    means something — an empty list a tool sets on purpose is indistinguishable from the
    default, and was never information. The transcript keeps the full dump."""
    dump = obs.model_dump(mode="json", exclude_none=True)
    return {k: v for k, v in dump.items() if v != [] and v != {}}


# Every field a subclass adds is optional, with a default that reads as "not known".
# A tool must be able to build its own failure: `Observation(ok=False, summary=...)` on
# the error path of a tool whose `returns` model had a required field raised a pydantic
# error *instead of* the sentence the user needed.


class WorkspaceObservation(Observation):
    workspace: str = ""
    stages: list[str] = Field(default_factory=list)
    n_artifacts: int = 0
    n_events: int = 0
    # one sentence per job the workspace is still waiting on, each after a poll of the
    # backend: what it is, where it got to, and — when it has ended — the call that
    # collects it. A list of ids would say none of that.
    jobs: list[str] = Field(default_factory=list)


class StructureObservation(Observation):
    """A fetched or inspected molecular structure."""

    source: str = ""  # e.g. "rcsb:1L2Y" or a local path
    n_atoms: int | None = None
    chains: list[str] = Field(default_factory=list)
    het_residues: list[str] = Field(default_factory=list)


class PrepObservation(Observation):
    n_atoms: int | None = None
    n_solvent: int | None = None
    n_ions_positive: int | None = None
    n_ions_negative: int | None = None
    box_nm: tuple[float, float, float] | None = None
    net_charge: float | None = None
    forcefield: str | None = None
    water_model: str | None = None


class ValidationReport(Observation):
    """Result of a validation gate run (standalone or pre-run).

    ``advisory`` and ``checked`` live on Observation, so a call the gate let through
    reports them too.
    """

    blocking: list[ValidationWarning] = Field(default_factory=list)
    unacknowledged: list[ValidationWarning] = Field(default_factory=list)
    human_required: list[ValidationWarning] = Field(default_factory=list)  # unattended
    acknowledged: list[str] = Field(default_factory=list)


class RunProgress(BaseModel):
    step: int | None = None
    time_ps: float | None = None
    ns_per_day: float | None = None


class JobObservation(Observation):
    job_id: str = ""
    state: str = ""  # JobState value
    exit_code: int | None = None
    progress: RunProgress | None = None
    diagnosis: str | None = None  # crash-pattern diagnosis on failure
    diagnosis_id: str | None = None  # its stable id (mimic_diagnose's signature table)
    core_h_estimate: float | None = None  # walltime × cores booked at submission


class ScriptObservation(JobObservation):
    """A research script submitted (or run to completion) through the harness."""

    script: str = ""  # path as given
    interpreter: str | None = None
    git_commit: str | None = None  # of the repo the script lives in, if any
    git_dirty: bool | None = None  # uncommitted changes in that repo
    outputs_expected: list[str] = Field(default_factory=list)
    outputs_missing: list[str] = Field(default_factory=list)  # after completion


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


class ToolListObservation(Observation):
    """What tools exist for a job: ``shown`` are callable here, ``hidden`` exist but need
    an engine this machine lacks."""

    shown: list[str] = Field(default_factory=list)
    hidden: list[str] = Field(default_factory=list)


class SessionSummary(Observation):
    """What went through the harness in a time window — the visibility report.

    Everything here is counted from the workspace ledger and job store. Work
    that bypassed the tools leaves no trace and is, by construction, absent.
    """

    workspace: str = ""
    window_start: str = ""
    window_end: str = ""
    n_events: int = 0
    tools: list[ToolTally] = Field(default_factory=list)
    observations: int = 0  # job polls and the like: looks, not actions
    gate_refusals: int = 0
    approvals: int = 0  # plan approvals (successful plan_approve calls) in the window
    warnings_acknowledged: list[str] = Field(default_factory=list)
    # of those, the ones a human accepted in an approval; the rest are the agent's own
    acknowledged_under_approval: list[str] = Field(default_factory=list)
    # finding ids the window's calls surfaced (ProvenanceEvent.knowledge_refs)
    knowledge_refs: list[str] = Field(default_factory=list)
    jobs: list[JobTally] = Field(default_factory=list)
    core_h_submitted: float = 0.0  # sum of estimates (walltime upper bound)
    core_h_consumed: float | None = None  # backend accounting where available
    n_artifacts: int = 0
    artifacts_by_role: dict[str, int] = Field(default_factory=dict)
    scripts: list[str] = Field(default_factory=list)  # "path @ commit[+dirty]"
    protocols_used: list[str] = Field(default_factory=list)  # "<id> v<N>" rendered in the window
    # engine runs whose parameter file has a recorded source, and those that do not
    runs_sourced: list[str] = Field(default_factory=list)
    runs_unsourced: list[str] = Field(default_factory=list)



class FileCheckObservation(ValidationReport):
    """One engine input file, checked and explained: what the gate for its kind says
    about it, and where every value came from and what was edited since."""

    path: str = ""
    file_kind: str = "other"  # mdp | cpmd | other
    provenance: str = "unknown"
    sourced: bool = False
    inputset: str | None = None
    keys: list[dict] = Field(default_factory=list)
    yours: list[str] = Field(default_factory=list)  # keys the user set or changed
    edited: bool = False


class EnergyTermStats(BaseModel):
    unit: str
    mean: float
    std: float
    drift_per_ns: float | None = None  # slope of linear fit
    n_points: int


class EnergyObservation(Observation):
    terms: dict[str, EnergyTermStats] = Field(default_factory=dict)
    time_span_ps: float | None = None
    available_terms: list[str] = Field(
        default_factory=list,
        description="on a failed selection: the labels this .edr offers, exactly as GROMACS "
        "spells them ('Total-Energy', 'Kinetic-En.') — request these strings",
    )


class TrajectoryObservation(Observation):
    metric: str = ""
    unit: str | None = None
    mean: float | None = None
    final: float | None = None
    n_frames: int | None = None


class VizObservation(Observation):
    """State of the live visualization session after a command batch.

    Counts and identifiers are parsed from ChimeraX ``info`` output — the
    model composes selections from these, never from its own recollection
    of the structure.
    """

    session_started: bool = False  # True when this call launched ChimeraX
    n_models: int = 0
    n_atoms: int | None = None
    chains: dict[str, int] = Field(default_factory=dict)  # chain id -> n residues
    ligands: list[str] = Field(default_factory=list)  # non-polymer residue specs
    image: str | None = None  # workspace-relative path of the rendered snapshot
    view_matrix: str | None = None  # camera state, for reproducing the exact view
    executed: list[str] = Field(default_factory=list)  # commands that actually ran


class EngineReport(Observation):
    """Doctor output: what engines were found, versions, what's missing and why."""

    engines: dict[str, str] = Field(default_factory=dict)  # name -> version
    missing: dict[str, str] = Field(default_factory=dict)  # name -> reason/hint
    backend: str = "local"
    # entries under studies_root that are neither workspaces nor _scratch
    workspace_strays: list[str] = Field(default_factory=list)
    # config.env_script: {path, state (none|applied|already|missing|failed), changed, error}
    env_script: dict = Field(default_factory=dict)
    # agent clients with a myrmidon MCP entry: {client, scope, file, command, command_found}
    clients: list[dict] = Field(default_factory=list)


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



class InputSetObservation(Observation):
    inputsets: list[InputSetHit] = Field(default_factory=list)


class ProtocolHit(BaseModel):
    """One published pipeline, on the wire."""

    id: str
    title: str
    applicability: str
    system_class: str = ""  # the class of system its values hold for: read it before using
    scope: str = "project"
    layer: str | None = None
    version: int = 1
    given: list[str] = Field(default_factory=list)  # roles the researcher brings
    steps: list[dict] = Field(default_factory=list)
    total_core_h: float = 0.0  # where it was measured, not a promise about here
    citations: list[str] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    notes: str = ""


class ProtocolObservation(Observation):
    protocols: list[ProtocolHit] = Field(default_factory=list)


class Unresolved(BaseModel):
    """A value the plan's execution will need and the plan does not carry.

    ``kind`` says who has to supply it: ``researcher`` — a value a cited input set refuses
    to supply (chemistry, length, charge); ``resources`` — the scheduler resources a
    submitting step's cost is computed from; ``parameter`` — a value the step's tool
    requires that neither the step nor its input set states; ``file`` — a file the tool
    requires, stated by the artifact role that feeds it or a path; ``wiring`` — a role an
    earlier step makes and this step needs, fed to none of its parameters.
    """

    step: str
    key: str
    kind: Literal["researcher", "resources", "parameter", "file", "wiring"]
    detail: str = ""


class ReuseCandidate(BaseModel):
    """An earlier call, made outside this plan, that fits one of its steps: what the step
    could declare as ``reuse`` instead of running. Offered, never filled in."""

    step: str
    event: str
    tool: str
    date: str  # when it was made, YYYY-MM-DD
    job: str | None = None  # the job it submitted, when it submitted one


class PlanObservation(Observation):
    """A study's plan, written and checked.

    ``digest`` is what an approval is recorded against: it content-addresses the plan, so
    an acknowledgement stops covering a plan the user has since edited.
    """

    path: str | None = None
    digest: str | None = None
    plan_id: str | None = None  # the question this plan answers; kept by a revision
    cites: str | None = None  # "protocol:<id>@v<N>" — the published sequence it started from
    shape: str | None = None  # one-shot | comparison | sweep | chain
    question: str | None = None
    total_core_h: float = 0.0
    unestimated: list[str] = Field(default_factory=list)  # steps with no estimate
    # what must be settled before the plan can be approved; empty means executable
    unresolved: list[Unresolved] = Field(default_factory=list)
    reuse_candidates: list[ReuseCandidate] = Field(default_factory=list)
    steps: list[dict] = Field(default_factory=list)
    given: list[str] = Field(default_factory=list)
    arms: dict | None = None
    until: dict | None = None
    envelope: dict | None = None  # what the loop may do without asking
    ceiling_core_h: float = 0.0  # the most the plan may cost under its ceilings


class JournalObservation(Observation):
    entry_id: str = ""
    kind: str = ""


class LoopObservation(Observation):
    """The study's loop: what a tick did, or where the loop stands."""

    plan_id: str = ""
    follower_pid: int | None = None  # the detached `tick --follow` process, when one runs
    follower_alive: bool = False
    done: bool = False
    steps: list[dict] = Field(default_factory=list)  # step, tool, state, segments
    running: list[dict] = Field(default_factory=list)  # step, job_id, state
    questions: list[dict] = Field(default_factory=list)  # what waits on a person
    journal: list[dict] = Field(default_factory=list)  # open entries
    events: list[dict] = Field(default_factory=list)  # the last spool events
    submitted: list[str] = Field(default_factory=list)  # a tick's submissions
    collected: list[str] = Field(default_factory=list)
    booked_core_h: float = 0.0
    ceiling_core_h: float = 0.0


class ReportValue(BaseModel):
    """One parsed value in a study report, with where it came from."""

    step: str
    label: str
    value: float | int | bool | str
    unit: str = ""
    event: str  # the ledger event of the call that produced it
    file: str  # the output it was parsed from


class ReportObservation(Observation):
    """The study's report, written as a file: the harness's sections with the model's
    prose around them, and every parsed value with its source."""

    path: str | None = None
    plan_id: str | None = None
    values: list[ReportValue] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)  # steps of the plan not yet run


class SkillHit(BaseModel):
    name: str
    description: str = ""
    question: str = ""  # the researcher's sentence this skill is for


class SkillObservation(Observation):
    """The skills, listed or read in full.

    When one was read, the observation says *which text* was read: `skill_id` and the
    sha256 of the SKILL.md served. A skill is edited like anything else, so "this study
    followed the qmmm-preflight skill" is only a fact if the version is recorded — and the
    file itself goes into the event's `inputs` as a consumed artifact, the way every file a
    tool reads does.
    """

    skills: list[SkillHit] = Field(default_factory=list)
    content: str | None = None  # the whole SKILL.md, when one was named
    skill_id: str | None = None  # the skill that was read in full, if one was
    sha256: str | None = None  # of that SKILL.md, so replay can tell two versions apart


class PlanCheckObservation(Observation):
    """The plan against the ledger: what was said, and what was done.

    A plan is a prospective ledger and a ledger is an executed plan, so this is a diff
    between the same list in two tenses. ``exact`` says which kind of answer it is: true
    when the calls name the steps they belong to, false when the match is a sequence
    alignment on tool names.
    """

    digest: str | None = None
    plan_id: str | None = None  # only calls made under this plan carry out its steps
    approved_digest: str | None = None  # None when the file is not what was approved
    approval_holds: bool = False
    acknowledged: list[str] = Field(default_factory=list)  # findings accepted at write
    exact: bool = False
    steps: list[dict] = Field(default_factory=list)  # name / tool / state / events
    done: list[str] = Field(default_factory=list)
    reused: list[str] = Field(default_factory=list)  # done by an earlier call, declared
    pending: list[str] = Field(default_factory=list)
    protocols_never_applied: list[str] = Field(default_factory=list)
    protocols_not_in_plan: list[str] = Field(default_factory=list)
    unaccounted_calls: list[str] = Field(default_factory=list)
    planned_core_h: float = 0.0
    booked_core_h: float = 0.0
    consumed_core_h: float | None = None


class InputSetApplication(Observation):
    """An input set rendered into a real file, with the origin of every key.

    ``origin`` is the answer to "who said so" for every value in the artifact:
    ``inputset:<id>@v<N>`` or ``user``. It is written to a sidecar next to the file
    and lands in the ledger with the artifact hash.
    """

    inputset: str | None = None
    protocol_version: int | None = None
    renders: str | None = None
    values: dict = Field(default_factory=dict)
    origin: dict[str, str] = Field(default_factory=dict)
    overrides: dict = Field(default_factory=dict)  # key -> {inputset, user}
    on_axis: list[str] = Field(default_factory=list)  # overrides the plan declared as its axis
    additions: list[str] = Field(default_factory=list)
    sources: list[InputSetSource] = Field(default_factory=list)
    invariants: list[dict] = Field(default_factory=list)  # id / status / message / detail
    # finding ids (gate, advisory, diagnosis) the input set names as the ways it fails
    pitfalls: list[str] = Field(default_factory=list)
    sidecar: str | None = None
