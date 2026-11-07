"""Descent: whether the files a job runs on come from the approved system, by the ledger.

Every event records what it consumed and what it wrote, hashed; a tool's contract says what
kind of artifact each parameter takes (core/contracts.py). A file descends from the
mandate's system when it is one of the system's files by hash, or when some call that wrote
those bytes read only files that descend, and read the system through at least one of them.
A parameter file is never followed (its parent is the input set it was rendered from); a
file of a kind that carries identity must reach the system; any other kind (a run input, a
checkpoint, a trajectory) is followed to what it was made from, and one no recorded call
wrote reaches nothing. A job's collection stands for its submission: its outputs were made
from what the submission read.
"""

from __future__ import annotations

from dataclasses import dataclass

#: how a collection names the submission it collects, in its event's knowledge_refs
COLLECTS = "collects:"
#: how a parameter file's render names the mandate stage it was compiled for
STAGE = "stage:"
#: a collection whose node read other inputs than were submitted: its files are no one's
RECEIPT_CHANGED = "RECEIPT_INPUT_CHANGED"


@dataclass(frozen=True)
class Node:
    """What the ledger says of one file's bytes: ``ok`` when nothing on the way down is
    outside the system, ``reached`` when some path ends in a system file, and ``chain``,
    from the file down to what breaks it, each (path, kind)."""

    ok: bool = True
    reached: bool = False
    chain: tuple[tuple[str, str], ...] = ()


def _param(role: str) -> str:
    """The parameter a recorded input sat under: ``extra_files.index`` -> ``extra_files``."""
    return role.split(".", 1)[0].split("[", 1)[0]


class Lineage:
    """The ledger read once: every call that wrote each file's bytes, and what it read."""

    def __init__(self, ws, kind_of, identity: set[str], parameters: set[str]):
        self.kind_of = kind_of  # (tool, parameter) -> the kind its contract reads, or None
        self.identity, self.parameters = identity, parameters
        # a failed or timed-out run's files are still its submission's — the checkpoint a
        # continuation reads — unless the node read other inputs than were submitted
        calls = [ev for ev in ws.ledger.events() if ev.kind == "call"]
        by_id = {ev.event_id: ev for ev in calls if ev.ok}
        events = [ev for ev in calls if ev.ok or (
            any(ref.startswith(COLLECTS) for ref in ev.knowledge_refs)
            and RECEIPT_CHANGED not in ev.knowledge_refs)]
        self.writers: dict[str, list] = {}
        #: workspace path -> the latest input set render written there
        self.rendered_at: dict[str, object] = {}
        #: the submissions a collection stands for: their files are another job's
        self.submissions: set[str] = set()
        #: the submissions a successful collection stands for: runs that ran
        self.completed: set[str] = set()
        for ev in events:
            source = self._submission(ev, by_id)
            if source is not ev:
                self.submissions.add(source.event_id)
                if ev.ok:
                    self.completed.add(source.event_id)
            read = {inp.sha256 for inp in source.inputs}
            renders = any(ref.startswith("inputset:") for ref in ev.knowledge_refs)
            for out in ev.outputs:
                if out.sha256 not in read:  # a copy of its own input says nothing new
                    self.writers.setdefault(out.sha256, []).append(source)
                if renders:
                    self.rendered_at[out.path] = ev
        self._memo: dict[str, Node] = {}

    @staticmethod
    def _submission(ev, by_id):
        """The call a collection collects, for an event that is one; else the event."""
        for ref in ev.knowledge_refs:
            if ref.startswith(COLLECTS) and ref[len(COLLECTS):] in by_id:
                return by_id[ref[len(COLLECTS):]]
        return ev

    def kind(self, ev, role: str) -> str:
        return self.kind_of(ev.tool, role) or ""

    def node(self, path: str, sha: str, system: set[str]) -> Node:
        """The verdict on one file's bytes, memoized: fine when some call that wrote them
        made them from the system; a file met again on its own way down is neutral."""
        if sha in system:
            return Node(reached=True)
        if sha in self._memo:
            return self._memo[sha]
        self._memo[sha] = Node()  # in progress
        verdicts = [self._made_by(ev, system) for ev in self.writers.get(sha, ())]
        good = [v for v in verdicts if v.ok]
        if good or not verdicts:
            found = Node(reached=any(v.reached for v in good))
        else:
            found = Node(ok=False, chain=((path, ""), *verdicts[-1].chain))
        self._memo[sha] = found
        return found

    def _made_by(self, ev, system: set[str]) -> Node:
        """Whether one call made its outputs from the system: every input it read outside
        its parameter files is fine, and an input of an identity kind reaches it."""
        reached = False
        for inp in ev.inputs:
            kind = self.kind(ev, inp.role)
            if kind in self.parameters:
                continue
            sub = self.fits(inp.path, inp.sha256, kind, system)
            if not sub.ok:
                return Node(ok=False, chain=sub.chain)
            reached = reached or sub.reached
        return Node(ok=True, reached=reached)

    def fits(self, path: str, sha: str, kind: str, system: set[str]) -> Node:
        """``node`` held to its kind: a file of an identity kind must reach the system."""
        found = self.node(path, sha, system)
        if found.ok and kind in self.identity and not found.reached:
            return Node(ok=False, chain=((path, kind),))
        if not found.ok:
            head, *rest = found.chain
            return Node(ok=False, chain=((head[0], kind or head[1]), *rest))
        return found

    def parameter_files(self, tool: str, inputs) -> list[tuple[str, str, str]]:
        """The parameter files a job runs on, each (path, sha256, kind): its own, and those
        the calls that wrote its run inputs read. Earlier stages' are theirs, not its."""
        out: dict[tuple[str, str], str] = {}
        for ref in inputs:
            kind = self.kind_of(tool, ref.role) or ""
            if kind in self.parameters:
                out[(ref.path, ref.sha256)] = kind
                continue
            for ev in self.writers.get(ref.sha256, ())[-1:]:
                if ev.event_id in self.submissions:
                    continue  # a previous job's run input: its parameter files are its own
                for inp in ev.inputs:
                    k = self.kind(ev, inp.role)
                    if k in self.parameters:
                        out[(inp.path, inp.sha256)] = k
        return [(path, sha, kind) for (path, sha), kind in out.items()]

    def stage_of(self, path: str, sha: str) -> str | None:
        """The mandate stage a parameter file was compiled for: by the render that wrote
        these bytes, else — bytes no render wrote, the file edited since — by the latest
        render at this path, whose stage an edit does not leave; None when none names one."""
        writer = [*self.writers.get(sha, ())[-1:]]
        rendered = [ev for ev in writer
                    if any(ref.startswith("inputset:") for ref in ev.knowledge_refs)]
        candidates = rendered or ([self.rendered_at[path]] if path in self.rendered_at
                                  else [])
        for ev in candidates:
            for ref in ev.knowledge_refs:
                if ref.startswith(STAGE):
                    return ref[len(STAGE):]
        return None

    def stages_behind(self, sha: str) -> set[str]:
        """The stages of the parameter files the run that wrote these bytes ran on."""
        out: set[str] = set()
        for ev in self.writers.get(sha, ())[-1:]:
            for path, psha, _ in self.parameter_files(ev.tool, ev.inputs):
                stage = self.stage_of(path, psha)
                if stage:
                    out.add(stage)
        return out

    def rendered_from(self, path: str, sha: str, depth: int = 3) -> str:
        """The input set a parameter file comes from, as ``<id>@v<N>``: by the render that
        wrote these bytes, else the latest render at this path (a file edited since), else
        through the parameter files its writer read (a run input made from a rendered
        template); "" when none says."""
        from myrmidon.core.observation import parse_inputset_ref

        candidates = [*self.writers.get(sha, ())[-1:]]
        if path in self.rendered_at:
            candidates.append(self.rendered_at[path])
        for ev in candidates:
            for ref in ev.knowledge_refs:
                parsed = parse_inputset_ref(ref)
                if parsed is not None:
                    return f"{parsed[0]}@v{parsed[1]}"
        if depth:
            for ev in self.writers.get(sha, ())[-1:]:
                for inp in ev.inputs:
                    if self.kind(ev, inp.role) in self.parameters:
                        found = self.rendered_from(inp.path, inp.sha256, depth - 1)
                        if found:
                            return found
        return ""


def kind_read(tool: str, role: str) -> str | None:
    """The kind a registered tool's contract reads under this recorded role; None for a tool
    or a parameter the contracts do not name. An entry of a mapping of kinds to files is the
    kind it sits under (``inputs.tpr``)."""
    from myrmidon.core.contracts import canonical
    from myrmidon.core.tool import contract_of

    contract = contract_of(tool)
    if contract is None:
        return None
    param = _param(role)
    if contract.by_kind == param and "." in role:
        return canonical(role.split(".", 1)[1].split(".", 1)[0])
    if param in contract.consumes:
        return canonical(contract.accepts(param)[0])
    return canonical(contract.reads[param]) if param in contract.reads else None


def lineage(ws) -> Lineage:
    """The study's ledger read for descent, with the kinds the engines declared."""
    from myrmidon.core.contracts import IDENTITY, parameter_kinds

    return Lineage(ws, kind_read, set(IDENTITY), parameter_kinds())


TRUE, FALSE = {"yes", "true", "on"}, {"no", "false", "off"}


def _numbers(text) -> list[float] | None:
    try:
        return [float(t) for t in str(text).split()]
    except ValueError:
        return None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


def _meets(value: str, option) -> bool:
    if isinstance(option, bool):
        word = str(value).strip().lower()
        return word in (TRUE if option else FALSE)
    have, want = _numbers(value), _numbers(option)
    if have and want:
        if len(want) == len(have):  # position by position: a CPMD value line
            return all(_close(a, b) for a, b in zip(have, want, strict=True))
        if len(want) == 1:  # one value per coupling group, all the same
            return all(_close(a, want[0]) for a in have)
        return False
    return str(value).strip().lower() == str(option).strip().lower()


def same_words(value, constraint) -> bool:
    """Whether a value states a constraint's words in any order: a define's switches."""
    return (isinstance(constraint, str)
            and sorted(str(value).lower().split()) == sorted(constraint.lower().split()))


def satisfies(value, constraint) -> bool:
    """Whether a value as written meets a decision's constraint: a scalar (equal), a list
    (one of) or ``{min, max}`` (within). Numbers compare as numbers, position by position or
    each against one; a boolean matches the engine's yes/no spellings; text compares case
    aside."""
    if isinstance(constraint, dict) and constraint and set(constraint) <= {"min", "max"}:
        numbers = _numbers(value)
        lo, hi = constraint.get("min"), constraint.get("max")
        if isinstance(lo, list) or isinstance(hi, list):
            # a line, position by position; a position bounded by neither end is free
            return bool(numbers) and len(numbers) == len(lo or hi) and all(
                (lo is None or lo[i] is None or n >= float(lo[i]))
                and (hi is None or hi[i] is None or n <= float(hi[i]))
                for i, n in enumerate(numbers))
        return bool(numbers) and all((lo is None or n >= float(lo))
                                     and (hi is None or n <= float(hi)) for n in numbers)
    options = constraint if isinstance(constraint, list) else [constraint]
    return any(satisfies(value, option) if isinstance(option, dict) else _meets(value, option)
               for option in options)


def _said(constraint) -> str:
    if isinstance(constraint, dict):
        return " to ".join(str(constraint.get(k, "…")) for k in ("min", "max"))
    if isinstance(constraint, list):
        return "one of " + ", ".join(str(c) for c in constraint)
    return str(constraint)


def stage_run(ws, stage: str):
    """(what the latest collected run of ``stage`` wrote, what it was submitted on, how
    many collected runs of it there are): the files a measurement over that stage reads,
    from the ledger; None when none was collected."""
    found = stage_runs(ws, stage)
    return (*found[-1], len(found)) if found else None


def stage_runs(ws, stage: str) -> list[tuple]:
    """(what each collected run of ``stage`` wrote, what it was submitted on), oldest first. A
    rerun of a stage's run input over a trajectory recomputes it, and is no run of the stage
    unless submitted as one."""
    lin = lineage(ws)
    events = [ev for ev in ws.ledger.events() if ev.ok and ev.kind == "call"]
    by_id = {ev.event_id: ev for ev in events}
    found = []
    for ev in events:
        submission = Lineage._submission(ev, by_id)
        if submission is ev or not ev.outputs:
            continue
        if f"{STAGE}{stage}" not in submission.knowledge_refs:
            if any(lin.kind(submission, inp.role) == "trajectory"
                   for inp in submission.inputs):
                continue
            if stage not in {lin.stage_of(path, sha) for path, sha, _ in
                             lin.parameter_files(submission.tool, submission.inputs)}:
                continue
        found.append((ev.outputs, submission.inputs))
    return found


def stage_left(ws, stage: str) -> list:
    """What the latest run of ``stage`` left, whether or not it succeeded, what its submission
    recorded first: a run stopped at its walltime is continued from the checkpoint it wrote and
    the run input it ran. None collected: empty."""
    events = list(ws.ledger.events())
    by_id = {ev.event_id: ev for ev in events if ev.kind == "call"}
    for ev in reversed(events):
        if ev.kind != "call" or not ev.outputs:
            continue
        submission = Lineage._submission(ev, by_id)
        if submission is not ev and f"{STAGE}{stage}" in submission.knowledge_refs:
            return [*submission.outputs, *ev.outputs]
    return []


def stage_files(ws, stage: str) -> dict[str, str]:
    """What the calls that prepared ``stage`` wrote, by kind, the latest of each."""
    from myrmidon.core.contracts import canonical

    out: dict[str, str] = {}
    for ev in ws.ledger.events():
        if (ev.ok and ev.kind == "call" and f"{STAGE}{stage}" in ev.knowledge_refs
                and not any(ref.startswith(COLLECTS) for ref in ev.knowledge_refs)
                and not _runs(ev)):
            out |= {canonical(a.role): a.path for a in ev.outputs}
    return out


def _runs(ev) -> bool:
    """Whether a call submitted a job: what it wrote is what the job read, not its output."""
    from myrmidon.core.tool import REGISTRY, books
    from myrmidon.errors import UnknownToolError

    try:
        return books(REGISTRY.get(ev.tool))
    except UnknownToolError:
        return False


def latest_outputs(ws, tool: str, stage: str | None = None) -> dict[str, str]:
    """Role -> path of what the latest successful call of ``tool`` wrote, for ``stage`` when one
    was made for it; empty when it has not run."""
    found = [ev for ev in ws.ledger.events()
             if ev.tool == tool and ev.ok and ev.kind == "call" and ev.outputs]
    own = [ev for ev in found if stage and ev.params.get("stage") == stage]
    return {a.role: a.path for a in (own or found)[-1].outputs} if found else {}


def system_file(ws, mandate, kind: str, stage: str | None = None) -> str | None:
    """The study's file of a system kind, for a stage if named: the mandate's, unless a QM
    selection wrote that one, when the latest selection for the stage replaces it."""
    named = next((f.path for f in mandate.system.files if f.kind == kind), None)
    selected = latest_outputs(ws, "qm_region", stage).get(kind)
    if named is None or selected is None:
        return named or selected
    from_selection = any(ev.tool == "qm_region" and ev.ok and any(a.path == named
                                                                  for a in ev.outputs)
                         for ev in ws.ledger.events())
    return selected if from_selection else named


def latest_dataset(ws, stage: str | None = None) -> str | None:
    """The path of the reference dataset a parameter fit reads: the latest collected from the
    stage it starts from, else the latest in the study."""
    from myrmidon.core.contracts import canonical

    found = [(ev, out) for ev in ws.ledger.events() if ev.ok and ev.kind == "call"
             for out in ev.outputs if canonical(out.role) == "reference_dataset"]
    own = [out for ev, out in found if stage and ev.params.get("stage") == stage]
    return own[-1].path if own else (found[-1][1].path if found else None)


def stages_run(ws) -> set[str]:
    """The mandate stages some successfully collected job of this study ran."""
    lin = lineage(ws)
    out: set[str] = set()
    for ev in ws.ledger.events():
        if ev.event_id in lin.completed:
            out |= {stage for path, sha, _ in lin.parameter_files(ev.tool, ev.inputs)
                    if (stage := lin.stage_of(path, sha))}
    return out


def submission_findings(ws, mandate, tool: str, params: dict, resources: dict,
                        backend: str, expect=None) -> list:
    """What stops a job under ``mandate``: an input that does not come from its system
    (``MANDATE_SYSTEM``), or a value that is not what the researcher decided
    (``MANDATE_DECISION``), read from the job's parameter files on disk now, the call's own
    parameters and its booking. ``expect(kind, stage)`` says what a parameter file of a
    kind, written for a stage (None: no stage the method names), must hold for the
    researcher's locks: ``({key: (constraint, said, must_set)}, what cannot be
    compiled)``, ``must_set`` when leaving the key unset breaks the lock too. Every
    other value is the model's. Empty when it may run."""
    from myrmidon.core.artifacts import consumed_inputs
    from myrmidon.core.validation import Finding, Precondition

    lin = lineage(ws)
    system = {f.sha256 for f in mandate.system.files}
    names = ", ".join(f.path for f in mandate.system.files) or "none"
    revise = ("If that is intended: mandate_write with revise=true and the reason, and the "
              "researcher approves the revision.")
    inputs = consumed_inputs(params, ws.root, ws.origin)
    findings: list[Finding] = []
    followed, reached = [], False
    for ref in inputs:
        kind = kind_read(tool, ref.role) or ""
        if not kind or kind in lin.parameters:
            continue  # a script or a file the contract does not name is never followed
        followed.append(ref.path)
        verdict = lin.fits(ref.path, ref.sha256, kind, system)
        reached = reached or verdict.reached
        if not verdict.ok:
            chain = " ← ".join(f"{p} [{k}]" if k else p for p, k in verdict.chain)
            findings.append(Precondition(
                id="MANDATE_SYSTEM",
                message=f"{chain}: not built from the approved system ({names}). {revise}"))
    if followed and not reached and not findings:
        findings.append(Precondition(
            id="MANDATE_SYSTEM",
            message=f"{', '.join(followed)}: no recorded call built it from the approved "
            f"system ({names}); build it with the harness's tools from the system files. "
            + revise))

    files = lin.parameter_files(tool, inputs)
    if expect is not None:
        findings += _locked(lin, ws, files, expect, revise)
    findings += _unattributed(lin, ws, mandate, tool, backend, files)
    for decision in mandate.decisions.yours:
        for key, constraint in decision.keys.items():
            for where, value, problem in _decided(lin, ws, decision.target, key, tool,
                                                  params, resources, backend, files):
                if problem:
                    findings.append(Precondition(
                        id="MANDATE_DECISION",
                        message=f"{where} {problem}, and the researcher decided {key} "
                        f"there (\"{decision.said}\")"))
                    continue
                if value is None:
                    # an empty decision ("no define") is what an unset key already is
                    if decision.target == "*" or constraint in ("", None, []):
                        continue
                    message = f"{where} does not set {key}"
                elif satisfies(value, constraint):
                    continue
                else:
                    message = f"{where} has {key} = {value}"
                findings.append(Precondition(
                    id="MANDATE_DECISION",
                    message=f"{message}; the researcher decided {_said(constraint)} "
                    f"(\"{decision.said}\"). Set it back, or: {revise}"))
    return findings


def _locked(lin: Lineage, ws, files: list[tuple[str, str, str]], expect, revise: str
            ) -> list:
    """Each parameter file held to the keys the researcher's locks compile to for the stage
    it was written for."""
    from myrmidon.core import contracts
    from myrmidon.core.artifacts import sha256_cached
    from myrmidon.core.validation import Precondition

    findings = []
    for path, sha, kind in files:
        stage = lin.stage_of(path, sha)
        keys, unresolved = expect(kind, stage)
        what = f"{path} (stage {stage})" if stage else path
        for why in unresolved:
            findings.append(Precondition(
                id="MANDATE_DECISION",
                message=f"{what}: the researcher's decision cannot be checked, {why}"))
        if not keys:
            continue
        file = ws.resolve(path)
        try:
            if sha256_cached(file) != sha:
                findings.append(Precondition(
                    id="MANDATE_DECISION",
                    message=f"{path} changed since the run input was built from it: build "
                    "the run input again from the file as it is"))
                continue
            for key, (constraint, said, must_set) in keys.items():
                value = contracts.lookup(kind, file, key)
                if value is None:
                    if constraint in ("", None, []) or not must_set:
                        continue  # unset is what "none" is, or what a key not used is
                    message = f"{what} does not set {key}"
                elif constraint in ("", None, []):
                    if not str(value).strip():
                        continue
                    message = f"{what} has {key} = {value}"
                    constraint = "unset"
                elif satisfies(value, constraint) or same_words(value, constraint):
                    continue
                else:
                    message = f"{what} has {key} = {value}"
                findings.append(Precondition(
                    id="MANDATE_DECISION",
                    message=f"{message}; the researcher's \"{said}\" compiles to "
                    f"{_said(constraint)} there. Render it again from the mandate "
                    f"(stage_prepare stage={stage or '<name>'}), or: {revise}"))
        except (OSError, ValueError):
            findings.append(Precondition(id="MANDATE_DECISION",
                                         message=f"{path} cannot be read"))
    return findings


def _decided(lin: Lineage, ws, target: str, key: str, tool: str, params: dict,
             resources: dict, backend: str, files: list[tuple[str, str, str]]):
    """(where, the value written there, what keeps it from being read) for each place a
    decision on ``target`` reads ``key``: this call's parameters (a tool target), its
    booking (a backend target), the parameter files it runs on (an input set target, or
    ``*`` for all of them)."""
    from myrmidon.core import contracts
    from myrmidon.core.artifacts import sha256_cached

    # a key the call or its booking cannot hold is no reading of it (mandate_write says so)
    if key in params and target in (tool, "*"):
        yield tool, str(params[key]), ""
    if key in resources and target in (backend, "*"):
        yield f"the {backend} booking", str(resources[key]), ""
    if target in (tool, backend):
        return
    for path, sha, kind in files:
        rendered = lin.rendered_from(path, sha)
        named = target == "*" or target in (rendered, rendered.split("@")[0])
        if not named:
            continue  # another set's; a file no set is known for is _unattributed's
        file = ws.resolve(path)
        try:
            if sha256_cached(file) != sha:
                yield path, None, ("changed since the run input was built from it: build "
                                   "the run input again from the file as it is")
                continue
            value = contracts.lookup(kind, file, key)
        except (OSError, ValueError):
            yield path, None, "cannot be read"
            continue
        yield path, value, ""


def _unattributed(lin: Lineage, ws, mandate, tool: str, backend: str,
                  files: list[tuple[str, str, str]]) -> list:
    """A parameter file no input set is known for, held to the decisions aimed at sets: a
    key it sets must meet at least one of the constraints decided for that key, since the
    file may be any stage's; two stages' decisions on one key never contradict it. One
    whose bytes changed since the run input was built from it cannot be read as it was, and
    is passed over."""
    from myrmidon.core import contracts
    from myrmidon.core.artifacts import sha256_cached
    from myrmidon.core.validation import Precondition

    aimed = [d for d in mandate.decisions.yours if d.target not in ("*", tool, backend)]
    findings = []
    for path, sha, kind in files:
        if not aimed or lin.rendered_from(path, sha):
            continue
        file = ws.resolve(path)
        try:
            if sha256_cached(file) != sha:
                continue
            for key in dict.fromkeys(k for d in aimed for k in d.keys):
                value = contracts.lookup(kind, file, key)
                deciding = [d for d in aimed if key in d.keys]
                if value is None or any(satisfies(value, d.keys[key]) for d in deciding):
                    continue
                said = "; ".join(f"\"{d.said}\" ({_said(d.keys[key])})" for d in deciding)
                findings.append(Precondition(
                    id="MANDATE_DECISION",
                    message=f"{path} has {key} = {value}, which none of the researcher's "
                    f"decisions on it allows: {said}"))
        except (OSError, ValueError):
            continue
    return findings
