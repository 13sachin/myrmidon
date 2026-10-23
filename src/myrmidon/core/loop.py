"""``myrmidon tick``: one crash-only pass of the study loop.

A study is a loop over enumerable events. One tick, under an exclusive lock on the study:

1. **reconcile** the action a killed tick left in flight, against the ledger and the job
   store — found, it is adopted; not found, it never happened;
2. **collect** every job of the plan that has ended, through ``job_status`` (the one call
   that hashes a run's outputs into the ledger), and raise a ``job`` event for each;
3. **watch**: a live job past its step's deadline raises a ``deadline`` event; the plan's
   booked core-hours against its ceiling raise a ``budget`` event;
4. **handle** every event not yet handled, in order, through the ladder: a finished run
   is checked by its step's sentinels and, for the repeating step, by the until-check; a
   failed run is matched against the failure signatures, rerun once with identical inputs
   when the envelope allows, else put as a question; a sentinel that fails is answered by
   one more segment while the envelope has them, else put as a question; a question goes
   to a wake model when one is configured and to the inbox otherwise; a person's answer is
   carried out;
5. **continue**: every pending step whose inputs exist and whose branch is not parked is
   called through the registry with the plan's values — roles resolved to the files the
   earlier steps recorded, the plan's approval acknowledging what it approved and nothing
   else. A sweep runs its first arm alone until that run has passed its sentinels, then
   the rest; a step whose parameter takes a list joins them.

Nothing is spent without a holding plan approval. Between ticks only the engines move.
The loop's own record (``.myrmidon/loop.json``) holds what the ledger and the job store do
not say on their own — which event was handled, which call is which arm — and every
action is written there before it is taken.
"""

from __future__ import annotations

import fcntl
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from myrmidon.core.plan import Plan, PlanStep
from myrmidon.core.spool import (
    Inbox,
    InboxEntry,
    InFlight,
    Journal,
    LoopState,
    Option,
    Spool,
    SpoolEvent,
    now_iso,
)
from myrmidon.core.tool import REGISTRY, ToolContext, ToolRegistry
from myrmidon.execution.backend import JobState

LOCK_FILENAME = "loop.lock"
WAKES_DIRNAME = "wakes"

#: a plan's artifact-role vocabulary against the tools' own: the prefix of the role a plan
#: step produces, and the roles a tool records that stand for it. Longest prefix wins.
ROLE_KINDS: list[tuple[str, tuple[str, ...]]] = [
    ("cpmd_template", ("cpmd_inp",)),
    ("cpmd_method", ("cpmd_inp",)),
    ("cpmd_input", ("cpmd_input",)),
    ("cpmd_restart", ("restart",)),
    ("reference_forces", ("fm_reference",)),
    ("fm_reference", ("fm_reference",)),
    ("fm_dataset", ("fm_dataset",)),
    ("checkpoint", ("checkpoint",)),
    ("trajectory", ("trajectory",)),
    ("structure", ("structure",)),
    ("topology", ("topology",)),
    ("restart", ("restart",)),
    ("energy", ("energy",)),
    ("frames", ("structure",)),
    ("index", ("index",)),
    ("qm_log", ("qm_log",)),
    ("tpr", ("tpr",)),
    ("mdp", ("mdp",)),
    ("log", ("log",)),
]
#: what an input set's `renders` becomes on disk when the loop names the file
_RENDER_EXT = {"mdp": "mdp", "cpmd_inp": "inp", "fminput": "dat", "ndx": "ndx", "sbatch": "sh"}
TERMINAL_OK = (JobState.COMPLETED,)


class TickReport(BaseModel):
    """What one tick did — the answer to "and now?"."""

    plan_id: str = ""
    blocked: str = ""  # why nothing could be done: not approved, locked, no plan
    collected: list[str] = Field(default_factory=list)
    submitted: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)  # spool events raised, "kind subject"
    handled: list[str] = Field(default_factory=list)
    parked: list[str] = Field(default_factory=list)  # inbox entry ids opened
    running: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    done: bool = False
    stopped: bool = False

    @property
    def idle(self) -> bool:
        """Nothing to do until something outside changes: a job ends or a person answers."""
        return not (self.collected or self.submitted or self.events or self.handled)


class WakeRequest(BaseModel):
    """A prepared question put to a wake model: the question as the inbox would show it,
    and the context composed from the study (`core/wake.py`)."""

    question: InboxEntry
    context: str
    context_hash: str


class WakeReply(BaseModel):
    option: str
    reason: str = ""
    model: str = ""


@dataclass
class _Tick:
    ctx: ToolContext
    plan: Plan
    registry: ToolRegistry
    wake: Any
    state: LoopState
    spool: Spool
    journal: Journal
    inbox: Inbox
    report: TickReport
    by_id: dict = field(default_factory=dict)
    jobs: dict = field(default_factory=dict)  # event id -> job id

    # ---------------------------------------------------------------- study readers
    @property
    def ws(self):
        return self.ctx.workspace

    def refresh(self) -> None:
        from myrmidon.plan_tools import _jobs_by_event

        self.by_id = {ev.event_id: ev for ev in self.ws.ledger.events()}
        self.jobs = _jobs_by_event(self.ctx)

    @property
    def store(self):
        from myrmidon.execution.jobstore import open_jobstore

        return open_jobstore(self.ws)

    def record(self, event_id: str):
        job = self.jobs.get(event_id)
        return self.store.get(job) if job else None

    def collection_outputs(self, job_id: str) -> list:
        """The artifacts ``job_status`` hashed when it collected this job."""
        for ev in self.by_id.values():
            if ev.tool == "job_status" and ev.kind == "call" and ev.ok \
                    and ev.params.get("job_id") == job_id:
                return list(ev.outputs)
        return []

    def outputs_of(self, key: str) -> list:
        """Every artifact the runs of this step (and arm) produced, oldest first."""
        out = []
        for event_id in self.state.runs.get(key, []):
            ev = self.by_id.get(event_id)
            if ev is None or not ev.ok:
                continue
            out += list(ev.outputs)
            job = self.jobs.get(event_id)
            if job:
                out += self.collection_outputs(job)
        return out

    def status(self, key: str) -> str:
        """pending | running | ended (not yet collected) | done | failed | refused."""
        events = self.state.runs.get(key, [])
        if not events:
            return "pending"
        ev = self.by_id.get(events[-1])
        if ev is None:
            return "pending"
        if ev.kind == "gate_refused":
            return "refused"
        if not ev.ok:
            return "failed"
        rec = self.record(ev.event_id)
        if rec is None:
            return "done"
        if not rec.state.terminal:
            return "running"
        if rec.state in TERMINAL_OK:
            return "done" if rec.meta.get("collected") else "ended"
        return "failed"

    # ---------------------------------------------------------------- the plan's shape
    def key(self, step: PlanStep, arm: str = "") -> str:
        return f"{step.name}[{arm}]" if arm else step.name

    def arm_labels(self) -> list[str]:
        if self.plan.arms is None:
            return []
        return [_label(v) for v in self.plan.arms.over]

    def arm_steps(self) -> list[str]:
        return self.plan.arm_steps()

    def arms_of(self, step: PlanStep) -> list[str]:
        """The arm labels this step runs under: one per arm for an arm step, [""] else."""
        return self.arm_labels() if step.name in self.arm_steps() else [""]

    def producer(self, role: str) -> PlanStep | None:
        return next((s for s in self.plan.steps if role in s.produces), None)

    def parked(self) -> set[str]:
        return {q.subject for q in self.inbox.open_questions()} | set(self.state.stopped)

    def upstream_parked(self, step: PlanStep, arm: str) -> str:
        """The parked key this step waits on, or ""."""
        parked = self.parked()
        seen: set[str] = set()
        stack = [step]
        while stack:
            s = stack.pop()
            for role in s.needs:
                p = self.producer(role)
                if p is None or p.name in seen:
                    continue
                seen.add(p.name)
                for k in (self.key(p, arm), p.name):
                    if k in parked:
                        return k
                stack.append(p)
        return ""

    # ---------------------------------------------------------------- roles -> files
    def resolve_role(self, role: str, arm: str) -> str | list[str] | None:
        """The file (ws-relative) an earlier step recorded for this role, for this arm —
        or, for a join, one per arm. None when it is not there yet."""
        p = self.producer(role)
        if p is None:
            return None
        if p.name in self.arm_steps() and not arm:
            paths = [self.latest_artifact(self.key(p, label), role)
                     for label in self.arm_labels()
                     if self.key(p, label) not in self.state.dropped]
            return None if any(x is None for x in paths) else paths
        if p.reuse:
            ev = self.by_id.get(p.reuse)
            arts = list(ev.outputs) if ev else []
            job = self.jobs.get(p.reuse)
            if job:
                arts += self.collection_outputs(job)
            return _pick(arts, role)
        return self.latest_artifact(self.key(p, arm) if p.name in self.arm_steps()
                                    else p.name, role)

    def latest_artifact(self, key: str, role: str) -> str | None:
        return _pick(self.outputs_of(key), role)

    def params_for(self, step: PlanStep, arm: str) -> tuple[dict, str]:
        """(the call's parameters, why they cannot be made yet). Roles become the files
        the earlier steps recorded; a value that names an earlier run's fact is left to
        the binding, which parses it; the varied key takes the arm's value."""
        from myrmidon.core.plan import references_in

        roles = {r for s in self.plan.steps for r in (*s.needs, *s.produces)}
        spec = self.registry.get(step.tool)
        fields = spec.params_model.model_fields
        params: dict = {}
        for k, v in step.values.items():
            if isinstance(v, str) and references_in(v):
                continue  # binding fills it from the run it names
            if isinstance(v, str) and v in roles and v not in self.plan.given:
                found = self.resolve_role(v, arm)
                if found is None:
                    return {}, f"{v} is not produced yet"
                params[k] = found
            elif isinstance(v, str) and v in self.plan.given and not self.ws.resolve(v).exists():
                return {}, (f"{v} is a role the researcher brings, and no file by that name "
                            "exists: the plan must name the file in the step's values")
            else:
                params[k] = v
        if self.plan.arms is not None and arm and step.name in self.arm_steps() \
                and self.plan.arms.varies in fields:
            params[self.plan.arms.varies] = self.plan.arms.over[self.arm_labels().index(arm)]
        # a value the cited input set states for one of the tool's own parameters is the
        # call's (a `renders: none` set such as the box's cell type): what the plan check
        # counted as stated, the call carries
        for key, spec_value in self.cited_values(step).items():
            if key in fields and key not in params:
                params[key] = spec_value
        stage = step.name + (f"_{arm}" if arm else "")
        if "stage" in fields and "stage" not in params:
            params["stage"] = stage
        if step.tool == "inputset_apply" and "output" not in params:
            ext = self.render_ext(step)
            if ext:
                params["output"] = f"{stage}/{step.name}.{ext}"
        if step.tool == "mimic_prepare" and "output" not in params:
            params["output"] = f"{stage}/{step.name}.inp"
        return params, ""

    def render_cited(self, step: PlanStep, arm: str, params: dict) -> dict:
        """A step that cites an input set rendering a file, without a step that renders it:
        the loop renders it first — bound through the citing step, so the step's values
        for the set's keys are its overrides — and feeds the file to the parameter."""
        from myrmidon.plan_tools import RENDER_PARAM

        if step.tool == "inputset_apply" or not step.cites:
            return params
        store = getattr(self.ctx, "inputsets", None)
        ref = step.cites.removeprefix("inputset:").partition("@")[0]
        try:
            kind = store.get(ref).renders if store is not None else None
        except (KeyError, AttributeError):
            return params
        param = RENDER_PARAM.get(kind or "")
        if param is None or param not in self.registry.get(step.tool).params_model.model_fields:
            return params
        given = params.get(param)
        if isinstance(given, str) and self.ws.resolve(given).exists():
            return params
        stage = params.get("stage", step.name + (f"_{arm}" if arm else ""))
        output = f"{stage}/{step.name}.{_RENDER_EXT[kind]}"
        rendered = self.call(step, arm, "render", {"inputset": ref, "output": output},
                             tool="inputset_apply")
        if rendered.ok:
            art = next((a.path for a in rendered.artifacts if a.role == kind), None)
            if art:
                params[param] = art
        return params

    def cited_values(self, step: PlanStep) -> dict:
        store = getattr(self.ctx, "inputsets", None)
        ref = (step.cites or "").removeprefix("inputset:").partition("@")[0]
        if store is None or not ref:
            return {}
        try:
            inputset = store.get(ref)
        except (KeyError, AttributeError):
            return {}
        return {k: spec.value for k, spec in inputset.parameters.items()
                if getattr(spec, "value", None) is not None}

    def render_ext(self, step: PlanStep) -> str | None:
        store = getattr(self.ctx, "inputsets", None)
        ref = (step.cites or "").removeprefix("inputset:").partition("@")[0]
        if store is None or not ref:
            return None
        try:
            kind = store.get(ref).renders
        except (KeyError, AttributeError):
            return None
        return _RENDER_EXT.get(kind or "")

    # ---------------------------------------------------------------- actions
    def call(self, step: PlanStep, arm: str, action: str, params: dict,
             tool: str | None = None, event: str = "") -> Any:
        """One registry call on behalf of a step, recorded in the loop state before and
        after, so a tick killed in between is reconciled rather than repeated."""
        key = self.key(step, arm)
        tool = tool or step.tool
        self.state.in_flight = InFlight(action=action, step=step.name, arm=arm, tool=tool,
                                        started=now_iso(), event=event)
        self.state.save(self.ws.state_dir)
        # the call is recorded under its step; a continuation or an extension carries the
        # envelope's values, so it is tagged and not compared with the step's own
        self.ctx.bound_step = step.name
        self.ctx.bound_tag_only = action in ("extend", "continue")
        try:
            obs = self.registry.call(tool, self.ctx, params)
        finally:
            self.ctx.bound_step, self.ctx.bound_tag_only = None, False
        self.state.in_flight = None
        if obs.provenance_id:
            self.state.runs.setdefault(key, []).append(obs.provenance_id)
        job_id = getattr(obs, "job_id", "") if obs.ok else ""
        if job_id:
            self.store.set_meta(job_id, loop_key=key, loop_action=action)
            self.state.deadline_from[key] = now_iso()
            self.report.submitted.append(f"{key} ({action}) job {job_id}")
        elif obs.ok:
            self.report.submitted.append(f"{key} ({action}) in the session")
        self.state.save(self.ws.state_dir)
        self.refresh()
        if not obs.ok:
            if getattr(obs, "failure", None) == "gate_refused":
                ids = sorted({w.id for g in ("blocking", "unacknowledged", "human_required")
                              for w in getattr(obs, g, [])})
                self.state.refused[key] = ids
                self.raise_event("job", key, refused=True, findings=ids,
                                 summary=obs.summary.splitlines()[0][:300],
                                 event=obs.provenance_id or "")
            else:
                self.raise_event("job", key, state="FAILED", ok=False,
                                 summary=obs.summary.splitlines()[0][:300],
                                 event=obs.provenance_id or "", in_session=True)
        return obs

    def raise_event(self, kind: str, subject: str, **detail) -> SpoolEvent:
        ev = self.spool.raise_event(kind, subject, plan_id=self.plan.plan_id, **detail)
        self.report.events.append(f"{kind} {subject}")
        return ev

    def ask(self, subject: str, question: str, evidence: list[str], options: list[Option],
            cites: list[str] | None = None) -> None:
        """Park the branch with a prepared question — or, with a wake model configured,
        put the question to it first and carry out a typed answer as a person's."""
        if self.wake is not None:
            from myrmidon.core.wake import compose, save_wake

            entry = InboxEntry(subject=subject, question=question, evidence=evidence,
                               options=options, cites=list(cites or []))
            context, digest = compose(self, entry)
            reply = None
            try:
                reply = self.wake(WakeRequest(question=entry, context=context,
                                              context_hash=digest))
            except Exception as exc:  # a wake that fails is a wake that parked
                self.report.messages.append(f"wake failed ({type(exc).__name__}: {exc}); "
                                            "the question goes to the inbox")
            save_wake(self.ws.state_dir / WAKES_DIRNAME, entry, context, digest, reply)
            self.inbox.append(entry)
            if reply is not None and reply.option in {o.id for o in options} \
                    and reply.option != "park":
                who = f"model:{reply.model or 'wake'}"
                self.inbox.answer(entry.entry_id, reply.option, by=who)
                option = next(o for o in options if o.id == reply.option)
                self.journal.write("decision", "model",
                                   f"{subject}: {option.text} — {reply.reason}",
                                   cites=[entry.entry_id, *entry.cites], subject=subject,
                                   status="closed")
                self.raise_event("human", subject, inbox=entry.entry_id, option=option.id,
                                 action=option.action, args=option.args, by=who)
                return
            self.report.parked.append(entry.entry_id)
            return
        entry = self.inbox.ask(subject, question, evidence, options, cites)
        self.report.parked.append(entry.entry_id)

    # ---------------------------------------------------------------- 1. reconcile
    def reconcile(self) -> None:
        flight = self.state.in_flight
        if flight is None:
            return
        key = f"{flight.step}[{flight.arm}]" if flight.arm else flight.step
        since = datetime.fromisoformat(flight.started)
        known = {e for events in self.state.runs.values() for e in events}
        found = [ev for ev in self.by_id.values()
                 if ev.tool == flight.tool and ev.plan_id == self.plan.plan_id
                 and ev.event_id not in known and ev.timestamp >= since]
        if found:
            ev = found[-1]
            self.state.runs.setdefault(key, []).append(ev.event_id)
            job = self.jobs.get(ev.event_id)
            if job:
                self.store.set_meta(job, loop_key=key, loop_action=flight.action)
                self.state.deadline_from[key] = now_iso()
            self.report.messages.append(
                f"{key}: a tick died after calling {flight.tool} ({flight.action}); "
                f"recovered event {ev.event_id}, not called again")
        else:
            self.report.messages.append(
                f"{key}: a tick died before {flight.tool} ({flight.action}) ran; "
                "it is called again")
        self.state.in_flight = None
        self.state.save(self.ws.state_dir)

    def adopt(self) -> None:
        """Steps carried out outside the loop — by a session before the loop took over —
        count, by the plan's own alignment."""
        from myrmidon.core.plan_binding import step_fit
        from myrmidon.plan_tools import _call_events, _match_steps

        states, _, _ = _match_steps(self.plan, _call_events(self.ctx, self.plan),
                                    step_fit(self.ws, self.plan))
        known = {e for events in self.state.runs.values() for e in events}
        for step, st in zip(self.plan.steps, states, strict=True):
            if st["state"] == "done" and step.name not in self.state.runs \
                    and step.name not in self.arm_steps() \
                    and not any(e in known for e in st["events"]):
                self.state.runs[step.name] = list(st["events"])

    # ---------------------------------------------------------------- 2. collect
    def collect(self) -> None:
        for rec in self.store.list():
            if rec.meta.get("plan_id") != self.plan.plan_id:
                continue
            if rec.state.terminal and rec.meta.get("collected"):
                continue
            key = rec.meta.get("loop_key") or self.key_of_job(rec)
            if key is None:
                continue
            obs = self.registry.call("job_status", self.ctx, {"job_id": rec.job_id})
            state = getattr(obs, "state", "")
            if state in [s.value for s in JobState if s.terminal]:
                self.refresh()
                self.report.collected.append(f"{key} job {rec.job_id} {state}")
                self.raise_event(
                    "job", key, job_id=rec.job_id, state=state, ok=bool(obs.ok),
                    exit_code=getattr(obs, "exit_code", None),
                    diagnosis_id=getattr(obs, "diagnosis_id", None),
                    diagnosis=getattr(obs, "diagnosis", None),
                    summary=(obs.summary or "").splitlines()[0][:300],
                    event=obs.provenance_id or "", action=rec.meta.get("loop_action", "run"))
            else:
                self.report.running.append(f"{key} job {rec.job_id} {state}")

    def key_of_job(self, rec) -> str | None:
        submitted = rec.meta.get("event_id")
        for key, events in self.state.runs.items():
            if submitted in events:
                return key
        return None

    # ---------------------------------------------------------------- 3. watch
    def watch(self) -> None:
        now = datetime.now(UTC)
        for rec in self.store.list():
            if rec.meta.get("plan_id") != self.plan.plan_id or rec.state.terminal:
                continue
            key = rec.meta.get("loop_key")
            step = self.step_of(key)
            if step is None or step.deadline_min is None:
                continue
            start = datetime.fromisoformat(self.state.deadline_from.get(key, rec.submitted_at))
            if start.tzinfo is None:
                start = start.replace(tzinfo=UTC)
            waited = (now - start).total_seconds() / 60.0
            if waited > step.deadline_min and not self.spool.already(
                    "deadline", key, job_id=rec.job_id, since=start.isoformat()):
                self.raise_event("deadline", key, job_id=rec.job_id, state=rec.state.value,
                                 waited_min=round(waited, 1), deadline_min=step.deadline_min,
                                 since=start.isoformat())
        booked, ceiling = self.booked(), self.ceiling()
        if ceiling and booked >= 0.8 * ceiling and not self.state.budget_notified:
            self.state.budget_notified = True
            self.raise_event("budget", "plan", booked_core_h=booked, ceiling_core_h=ceiling,
                             share=round(booked / ceiling, 2))
            self.inbox.notify("plan", f"{booked:g} of {ceiling:g} core-h booked "
                              f"({booked / ceiling:.0%} of the approved ceiling); at 100 % "
                              "the loop stops submitting and asks")

    def step_of(self, key: str | None) -> PlanStep | None:
        if not key:
            return None
        name = key.split("[", 1)[0]
        return next((s for s in self.plan.steps if s.name == name), None)

    def booked(self) -> float:
        from myrmidon.plan_tools import _booked_for

        return _booked_for(self.ctx, self.plan.plan_id)

    def ceiling(self) -> float:
        from myrmidon.plan_tools import ceiling_core_h

        return ceiling_core_h(self.plan, self.ctx.config.compute)

    # ---------------------------------------------------------------- 4. handle
    def handle_all(self) -> None:
        while True:
            pending = [ev for ev in self.spool.all()
                       if ev.plan_id == self.plan.plan_id
                       and ev.event_id not in self.state.handled]
            if not pending:
                return
            ev = pending[0]
            self.state.handled.append(ev.event_id)
            self.state.save(self.ws.state_dir)
            self.report.handled.append(f"{ev.kind} {ev.subject}")
            getattr(self, f"handle_{ev.kind}")(ev)
            self.state.save(self.ws.state_dir)

    def handle_job(self, ev: SpoolEvent) -> None:
        key = ev.subject
        step = self.step_of(key)
        arm = _arm_of(key)
        if step is None:
            return
        d = ev.detail
        if d.get("refused"):
            ids = d.get("findings", [])
            self.ask(key, f"{step.tool} for step {step.name} was refused by its gate on "
                     + ", ".join(ids) + ". Approve the findings a person may accept "
                     "(`myrmidon approve`), then answer `retry`; or stop the branch.",
                     [d.get("summary", "")],
                     [Option(id="retry", action="rerun", text="call it again once the "
                             "findings are approved", cost_core_h=step.estimate_core_h or 0),
                      Option(id="stop", action="stop", text="stop this branch")],
                     cites=[ev.event_id, d.get("event", "")])
            return
        if d.get("ok") and d.get("state") == "COMPLETED":
            self.after_success(step, arm, key, ev)
            return
        # a failed run: the ladder, cheapest instrument first
        signature = d.get("diagnosis_id") or ""
        attempts = self.state.attempts.get(key, 0)
        env = self.plan.envelope
        if signature and arm and self.may_drop():
            self.state.dropped.append(key)
            self.journal.write("decision", "machinery",
                               f"{key} dropped on {signature}: within the envelope's "
                               f"drop fraction {env.drop_fraction}", cites=[ev.event_id],
                               subject=key, status="closed")
            self.report.messages.append(f"{key}: dropped on {signature} (envelope)")
            return
        if not signature and attempts < env.rerun_on_failure and not d.get("in_session"):
            # a job that failed with no known signature: a machine fault does not reproduce.
            # A call that failed in the session failed on its parameters, and a rerun of the
            # same ones would too
            self.state.attempts[key] = attempts + 1
            self.journal.write("suspicion", "machinery",
                               f"{key}: no failure signature matched (exit "
                               f"{d.get('exit_code')}); rerun of identical inputs {attempts + 1}"
                               f" of {env.rerun_on_failure} — a machine fault does not "
                               "reproduce", cites=[ev.event_id, d.get("event", "")],
                               subject=key)
            self.rerun(step, arm, key, ev.event_id)
            return
        if not signature and attempts >= env.rerun_on_failure and attempts:
            for entry in self.journal.open_entries():
                if entry.subject == key and entry.kind == "suspicion":
                    self.journal.close(entry.entry_id, "machinery",
                                       f"{key}: the failure reproduced on the rerun — not a "
                                       "machine fault")
        cost = step.estimate_core_h or 0.0
        what = (f"{d.get('diagnosis')} (signature {signature})" if signature
                else "the call failed in the session" if d.get("in_session")
                else f"no failure signature matched; it reproduced on {attempts} rerun(s)")
        self.ask(key, f"{step.name} failed: {what}. What next?",
                 [d.get("summary", ""), f"exit code {d.get('exit_code')}"],
                 [Option(id="rerun", action="rerun", text="rerun with identical inputs",
                         cost_core_h=cost),
                  Option(id="stop", action="stop", text="stop this branch")],
                 cites=[ev.event_id, d.get("event", "")])

    def may_drop(self) -> bool:
        n = len(self.arm_labels())
        return n > 0 and (len(self.state.dropped) + 1) <= self.plan.envelope.drop_fraction * n

    def after_success(self, step: PlanStep, arm: str, key: str, ev: SpoolEvent) -> None:
        rec = self.store.get(ev.detail["job_id"])
        from myrmidon.core.sentinels import evaluate

        for sentinel in step.sentinels:
            result = evaluate(sentinel, rec)
            if not result.holds:
                self.raise_event("sentinel", key, check=result.check, holds=False,
                                 value=result.value, detail=result.detail,
                                 job_id=rec.job_id, event=ev.detail.get("event", ""))
        until = self.plan.until
        if until is not None and until.repeats == step.name:
            self.until_check(step, key, rec, ev)

    def handle_sentinel(self, ev: SpoolEvent) -> None:
        key = ev.subject
        step = self.step_of(key)
        arm = _arm_of(key)
        d = ev.detail
        if step is None:
            return
        if d.get("until"):
            if d.get("met"):
                self.state.until_closed = True
                self.journal.write("decision", "machinery",
                                   f"{key}: the criterion holds after {d.get('segments')} "
                                   f"segment(s): {d.get('detail')}", cites=[ev.event_id],
                                   subject=key, status="closed")
                return
            segments = d.get("segments", 1)
            if segments < self.plan.until.ceiling:
                self.extend_until(step, key, ev)
                return
            self.ask(key, f"{step.name}: the ceiling of {self.plan.until.ceiling} segments is "
                     f"reached and the criterion does not hold: {d.get('detail')}. A larger "
                     "ceiling is a new plan to approve.",
                     [d.get("detail", "")],
                     [Option(id="stop", action="stop", text="stop with what there is"),
                      Option(id="accept", action="continue",
                             text="treat the run as equilibrated and go on")],
                     cites=[ev.event_id])
            return
        if d.get("holds", True):
            return
        extensions = self.state.extensions.get(key, 0)
        if extensions < self.plan.envelope.extend_segments and step.tool == "mimic_run":
            self.extend_stage(step, arm, key, ev)
            return
        cost = step.estimate_core_h or 0.0
        options = [Option(id="stop", action="stop", text="stop this branch"),
                   Option(id="continue", action="continue",
                          text="accept the run as it is and go on")]
        if step.tool == "mimic_run":
            options.insert(0, Option(id="extend", action="extend",
                                     text="one more segment from this run's restart",
                                     cost_core_h=cost))
        self.ask(key, f"{step.name}: sentinel {d.get('check')} does not hold "
                 f"({d.get('detail')}); the envelope allows no further extension.",
                 [d.get("detail", "")], options, cites=[ev.event_id, d.get("event", "")])

    def handle_deadline(self, ev: SpoolEvent) -> None:
        key = ev.subject
        d = ev.detail
        if self.plan.envelope.on_deadline == "wait":
            self.inbox.notify(key, f"{key}: job {d.get('job_id')} is {d.get('state')} after "
                              f"{d.get('waited_min')} min, past its deadline of "
                              f"{d.get('deadline_min')} min; the envelope says wait")
            self.state.deadline_from[key] = now_iso()
            return
        self.ask(key, f"{key}: job {d.get('job_id')} is still {d.get('state')} after "
                 f"{d.get('waited_min')} min, past its deadline of {d.get('deadline_min')} "
                 "min. Keep waiting, or cancel it and stop the branch?",
                 [f"job {d.get('job_id')} {d.get('state')}"],
                 [Option(id="wait", action="wait", text="keep waiting (free)"),
                  Option(id="cancel", action="stop", text="cancel the job and stop the branch",
                         args={"cancel": d.get("job_id")})],
                 cites=[ev.event_id])

    def handle_budget(self, ev: SpoolEvent) -> None:
        return  # the notice was written when the event was raised; nothing waits on it

    def handle_checkpoint(self, ev: SpoolEvent) -> None:
        return

    def handle_human(self, ev: SpoolEvent) -> None:
        key = ev.subject
        step = self.step_of(key)
        arm = _arm_of(key)
        d = ev.detail
        action = d.get("action")
        if step is None and key != "plan":
            return
        if action == "rerun":
            self.state.refused.pop(key, None)
            self.rerun(step, arm, key, ev.event_id)
        elif action == "extend":
            self.extend_stage(step, arm, key, ev)
        elif action == "wait":
            self.state.deadline_from[key] = now_iso()
        elif action == "continue":
            self.journal.write("decision", d.get("by", "human") if d.get("by", "").startswith(
                "model") else "human", f"{key}: continue as it is ({d.get('option')})",
                cites=[ev.event_id], subject=key, status="closed")
            if self.plan.until is not None and self.plan.until.repeats == step.name:
                self.state.until_closed = True
        elif action == "stop":
            job = (d.get("args") or {}).get("cancel")
            if job:
                self.registry.call("job_cancel", self.ctx, {"job_id": job})
            if key not in self.state.stopped:
                self.state.stopped.append(key)
            self.journal.write("decision", "human", f"{key}: stopped ({d.get('option')})",
                               cites=[ev.event_id], subject=key, status="closed")

    # ---------------------------------------------------------------- the moves
    def rerun(self, step: PlanStep, arm: str, key: str, event: str) -> None:
        params, why = self.params_for(step, arm)
        if why:
            self.report.messages.append(f"{key}: cannot rerun, {why}")
            return
        self.call(step, arm, "rerun", params, event=event)

    def extend_stage(self, step: PlanStep, arm: str, key: str, ev: SpoolEvent) -> None:
        """One more segment of a coupled stage from its own restart, in its own stage
        directory; the envelope's extension, or a person's."""
        restart = self.latest_artifact(key, "cpmd_restart")
        if restart is None:
            self.report.messages.append(f"{key}: no restart to extend from")
            return
        params, why = self.params_for(step, arm)
        if why:
            self.report.messages.append(f"{key}: cannot extend, {why}")
            return
        n = self.state.extensions.get(key, 0) + 1
        self.state.extensions[key] = n
        params["restart_from"] = restart
        params["stage"] = f"{params.get('stage', step.name)}_s{n + 1}"
        self.journal.write("decision", "machinery",
                           f"{key}: segment {n + 1} from {restart} — {ev.detail.get('check')} "
                           f"did not hold ({ev.detail.get('detail')}); extension {n} of "
                           f"{self.plan.envelope.extend_segments} inside the envelope",
                           cites=[ev.event_id], subject=key, status="closed")
        self.call(step, arm, "extend", params, event=ev.event_id)

    def until_check(self, step: PlanStep, key: str, rec, ev: SpoolEvent) -> None:
        until = self.plan.until
        segments = 1 + self.state.extensions.get(key, 0)
        if step.tool != "gmx_mdrun" or until.check != "series_convergence":
            self.raise_event("sentinel", key, until=True, met=None, segments=segments,
                             detail=f"the loop evaluates series_convergence on a gmx_mdrun "
                                    f"step only; this plan asks {until.check} on {step.tool}")
            return
        edr = self.latest_artifact(key, "energy")
        values = dict(until.values)
        series = values.pop("series", ["Density"])
        if edr is None:
            self.raise_event("sentinel", key, until=True, met=None, segments=segments,
                             detail="the run recorded no energy file")
            return
        self.ctx.bound_step = ""  # looks at the run, not steps of the plan
        try:
            energy = self.registry.call("gmx_energy", self.ctx, {"edr": edr, "terms": series})
            xvg = next((a.path for a in energy.artifacts if a.path.endswith(".xvg")), None) \
                if energy.ok else None
            if xvg is None:
                self.raise_event("sentinel", key, until=True, met=None, segments=segments,
                                 detail=f"gmx_energy: {energy.summary.splitlines()[0][:200]}")
                return
            obs = self.registry.call("series_convergence", self.ctx,
                                     {"data": xvg, "series": series, **values})
        finally:
            self.ctx.bound_step = None
        verdicts = [s.criterion_met for s in getattr(obs, "series", [])]
        met = all(verdicts) if verdicts and all(v is not None for v in verdicts) else None
        findings = [w.id for w in obs.advisory]
        self.raise_event("sentinel", key, until=True, met=met, segments=segments,
                         detail=obs.summary.splitlines()[0][:300], findings=findings,
                         event=obs.provenance_id or "", job_id=rec.job_id)

    def extend_until(self, step: PlanStep, key: str, ev: SpoolEvent) -> None:
        until = self.plan.until
        tpr = self.latest_artifact(key, "tpr")
        cpt = self.latest_artifact(key, "checkpoint")
        if tpr is None or cpt is None:
            self.report.messages.append(f"{key}: no tpr/checkpoint to continue from")
            return
        extend = dict(until.extend)
        params = {"tpr": tpr, "checkpoint": cpt,
                  "deffnm": step.values.get("deffnm", "md"),
                  "stage": step.values.get("stage", step.name)}
        for k in ("ntasks", "cpus_per_task", "walltime_min", "account"):
            if k in extend:
                params[k] = extend[k]
            elif k in step.values:
                params[k] = step.values[k]
        params["extend_ps"] = extend.get("extend_ps")
        n = self.state.extensions.get(key, 0) + 1
        self.state.extensions[key] = n
        self.journal.write("decision", "machinery",
                           f"{key}: segment {n + 1} of at most {until.ceiling} — the "
                           f"criterion does not hold yet ({ev.detail.get('detail')})",
                           cites=[ev.event_id], subject=key, status="closed")
        self.call(step, "", "continue", params, tool="checkpoint_continue",
                  event=ev.event_id)

    # ---------------------------------------------------------------- 5. continue
    def proceed(self) -> None:
        if self.state.in_flight is not None:
            return
        booked, ceiling = self.booked(), self.ceiling()
        # the plan's alignment pairs a call with the first pending step of its tool, in plan
        # order; so a step waits while an earlier step of the same tool is still pending,
        # whatever the data flow allows
        waiting: set[str] = set()
        for step in self.plan.steps:
            if step.reuse:
                continue
            for arm in self.arms_of(step):
                key = self.key(step, arm)
                if self.status(key) != "pending" or key in self.state.dropped:
                    continue
                if step.tool in waiting:
                    continue
                if (key in self.parked() or self.upstream_parked(step, arm)
                        or self.state.refused.get(key)  # waits for an approval, an answer
                        or not self.ready(step, arm)
                        or (arm and not self.probe_passed(step, arm))):
                    waiting.add(step.tool)
                    continue
                cost = step.estimate_core_h or 0.0
                if ceiling and cost and booked + cost > ceiling + 1e-9:
                    self.ask("plan", f"the approved ceiling of {ceiling:g} core-h would be "
                             f"passed by {key} ({cost:g} core-h on {booked:g} booked). A larger "
                             "ceiling is a new plan to approve.", [],
                             [Option(id="stop", action="stop", text="stop here")])
                    return
                params, why = self.params_for(step, arm)
                if why:
                    continue
                params = self.render_cited(step, arm, params)
                if self.status(key) == "failed":  # the render failed: the event says so
                    continue
                obs = self.call(step, arm, "run", params)
                if obs.ok and getattr(obs, "job_id", ""):
                    booked += cost
                if self.state.in_flight is not None:
                    return

    def ready(self, step: PlanStep, arm: str) -> bool:
        from myrmidon.core.plan import references_in

        until = self.plan.until
        # a value that names an earlier run's fact waits for that run
        for ref_step, _ in references_in(step.values):
            p = self.step_of(ref_step)
            if p is None or self.status(self.key(p, arm) if p.name in self.arm_steps()
                                        else p.name) != "done":
                return False
        for role in step.needs:
            if role in self.plan.given:
                continue
            p = self.producer(role)
            if p is None:
                return False
            if p.reuse:
                continue
            if until is not None and p.name == until.repeats and not self.state.until_closed:
                return False
            if p.name in self.arm_steps():
                keys = ([self.key(p, arm)] if arm else
                        [self.key(p, label) for label in self.arm_labels()
                         if self.key(p, label) not in self.state.dropped])
            else:
                keys = [p.name]
            if not keys or any(self.status(k) != "done" for k in keys):
                return False
        return True

    def probe_step(self) -> str | None:
        """The first arm step that submits a job: the point checked before the rest are
        spent. None when no arm step books anything."""
        for name in self.arm_steps():
            step = self.step_of(name)
            model = self.registry.get(step.tool).params_model
            if callable(getattr(model, "resources", None)):
                return name
        return None

    def probe_passed(self, step: PlanStep, arm: str) -> bool:
        """The first arm runs alone up to and through its first submitted job; the other
        arms' spending steps wait until that run has ended well and raised nothing (size
        follows demonstrated validity). Session steps before the probe run for every arm."""
        labels = self.arm_labels()
        probe = self.probe_step()
        if arm == labels[0] or probe is None:
            return True
        order = self.arm_steps()
        if order.index(step.name) < order.index(probe):
            return True
        key = f"{probe}[{labels[0]}]"
        if self.status(key) != "done" or key in self.parked():
            return False
        return not any(ev.kind == "sentinel" and ev.subject == key
                       and not ev.detail.get("holds", True) for ev in self.spool.all())

    # ---------------------------------------------------------------- the verdict
    def finish(self) -> None:
        running = [rec for rec in self.store.list()
                   if rec.meta.get("plan_id") == self.plan.plan_id and not rec.state.terminal]
        self.report.running = [f"{r.meta.get('loop_key', '?')} job {r.job_id} {r.state.value}"
                               for r in running]
        open_questions = self.inbox.open_questions()
        self.report.parked = sorted({*self.report.parked, *(q.entry_id for q in open_questions)})
        left = []
        for step in self.plan.steps:
            if step.reuse:
                continue
            for arm in self.arms_of(step):
                key = self.key(step, arm)
                if key in self.state.dropped or key in self.state.stopped:
                    continue
                if self.status(key) != "done":
                    left.append(key)
        until = self.plan.until
        if until is not None and not self.state.until_closed \
                and until.repeats not in self.state.stopped:
            left.append(f"{until.repeats} (until)")
        self.report.stopped = bool(self.state.stopped)
        self.report.done = not left and not running and not open_questions


def _arm_of(key: str) -> str:
    m = re.search(r"\[(.*)\]$", key)
    return m.group(1) if m else ""


def _label(value) -> str:
    text = Path(str(value)).stem if isinstance(value, str) and "/" in value else str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "arm"


def _pick(artifacts: list, role: str) -> str | None:
    """The latest artifact that stands for a plan role, by the tools' own role names."""
    kinds = next((k for prefix, k in ROLE_KINDS if role.startswith(prefix)), ())
    by_kind = [a for a in artifacts if a.role in kinds]
    if by_kind:
        return by_kind[-1].path
    same = [a for a in artifacts if a.role == role]
    if same:
        return same[-1].path
    return artifacts[-1].path if len(artifacts) == 1 else None


def _try_lock(path: Path):
    handle = open(path, "a+")  # noqa: SIM115 — closed by the caller
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def tick(ctx: ToolContext, *, wake=None, registry: ToolRegistry = REGISTRY) -> TickReport:
    """One pass. Nothing is spent without a holding plan approval; a second tick that finds
    the study locked returns at once saying so."""
    from myrmidon.core.approvals import PLAN_FILE, holding
    from myrmidon.core.artifacts import sha256_file
    from myrmidon.plan_tools import read_plan

    ws = ctx.workspace
    plan = read_plan(ctx)
    if plan is None:
        return TickReport(blocked="this study has no plan")
    report = TickReport(plan_id=plan.plan_id)
    current = sha256_file(ws.root / PLAN_FILE)
    if not any(a.plan_sha256 == current for a in holding(ws)):
        report.blocked = ("the plan on disk is not approved: nothing runs until plan_approve "
                          "records that a person saw it")
        return report
    lock = _try_lock(ws.state_dir / LOCK_FILENAME)
    if lock is None:
        report.blocked = "another tick holds this study"
        return report
    was_unattended = ctx.unattended
    ctx.unattended = True
    t = _Tick(ctx=ctx, plan=plan, registry=registry, wake=wake,
              state=LoopState.load(ws.state_dir), spool=Spool(ws.root),
              journal=Journal(ws.root), inbox=Inbox(ws.root), report=report)
    try:
        t.refresh()
        t.reconcile()
        t.adopt()
        t.collect()
        t.watch()
        t.handle_all()
        t.proceed()
        t.handle_all()  # what the moves just made raised (a refusal, an in-session failure)
        t.finish()
    finally:
        t.state.save(ws.state_dir)
        ctx.unattended = was_unattended
        lock.close()
    return report


def snapshot(ctx: ToolContext, registry: ToolRegistry = REGISTRY) -> dict:
    """Where the study's loop stands, from its files alone: every step (and arm) with its
    state, the live jobs, the open questions with their options, the open journal, the
    last events, and whether the plan is done. What `myrmidon status` and `loop_status`
    show; no model and no scheduler call is involved."""
    from myrmidon.plan_tools import read_plan

    ws = ctx.workspace
    plan = read_plan(ctx)
    if plan is None:
        return {"plan_id": "", "steps": [], "done": False, "note": "this study has no plan"}
    t = _Tick(ctx=ctx, plan=plan, registry=registry, wake=None,
              state=LoopState.load(ws.state_dir), spool=Spool(ws.root),
              journal=Journal(ws.root), inbox=Inbox(ws.root), report=TickReport())
    t.refresh()
    steps = []
    for step in plan.steps:
        for arm in t.arms_of(step):
            key = t.key(step, arm)
            state = ("reused" if step.reuse else "dropped" if key in t.state.dropped
                     else "stopped" if key in t.state.stopped else t.status(key))
            steps.append({"step": key, "tool": step.tool, "state": state,
                          "segments": 1 + t.state.extensions.get(key, 0)})
    running = [{"step": rec.meta.get("loop_key", ""), "job_id": rec.job_id,
                "state": rec.state.value}
               for rec in t.store.list()
               if rec.meta.get("plan_id") == plan.plan_id and not rec.state.terminal]
    questions = [{"entry_id": q.entry_id, "step": q.subject, "question": q.question,
                  "evidence": q.evidence,
                  "options": [{"id": o.id, "text": o.text, "cost_core_h": o.cost_core_h}
                              for o in q.options]}
                 for q in t.inbox.open_questions()]
    journal = [{"kind": e.kind, "by": e.by, "text": e.text, "step": e.subject}
               for e in t.journal.open_entries()]
    events = [{"ts": ev.ts, "kind": ev.kind, "step": ev.subject,
               "summary": (ev.detail.get("summary") or ev.detail.get("detail") or "")[:160]}
              for ev in t.spool.all()[-8:]]
    t.finish()
    return {"plan_id": plan.plan_id, "steps": steps, "running": running,
            "questions": questions, "journal": journal, "events": events,
            "done": t.report.done, "stopped": t.report.stopped,
            "booked_core_h": t.booked(), "ceiling_core_h": t.ceiling()}


def render(report: TickReport) -> str:
    if report.blocked:
        return f"tick: {report.blocked}"
    lines = [f"tick {report.plan_id}"]
    for label, items in (("collected", report.collected), ("events", report.events),
                         ("handled", report.handled), ("submitted", report.submitted),
                         ("parked", report.parked), ("running", report.running)):
        for item in items:
            lines.append(f"  {label:<9} {item}")
    lines += [f"  {m}" for m in report.messages]
    if report.done:
        lines.append("the plan is done" + (" (a branch was stopped)" if report.stopped else ""))
    elif report.parked:
        lines.append("waiting on an answer: `myrmidon inbox`")
    elif report.idle:
        lines.append("nothing to do: waiting on the jobs above" if report.running
                     else "nothing to do")
    return "\n".join(lines)
