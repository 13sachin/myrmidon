"""The loop as the agent reaches it: start it, advance it once, see where it stands, carry
a person's answer out.

A conversation is a pipe into the study and never the loop itself: once the plan is
approved the agent starts a detached ``myrmidon tick --follow`` process (`loop_start`),
which outlives the session the way a submitted job does, and the researcher comes back to
`loop_status` and the inbox. `loop_tick` is one pass in the session, for a study whose
follower is not running; `loop_answer` records a person's choice on a parked question — a
decision that needs the person, so an unattended session cannot make it.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.observation import LoopObservation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning

PID_FILENAME = "tick.pid"
LOG_FILENAME = "tick.log"


def _pid_path(ctx: ToolContext) -> Path:
    return ctx.workspace.state_dir / PID_FILENAME


def follower(ctx: ToolContext) -> tuple[int | None, bool]:
    """(the follower's pid, whether it is alive) from the study's pid file."""
    path = _pid_path(ctx)
    if not path.is_file():
        return None, False
    try:
        pid = int(path.read_text().strip())
    except ValueError:
        return None, False
    # a child this process started and never waited for lingers as a zombie, which a
    # signal 0 still reaches: reap it, then ask
    with contextlib.suppress(ChildProcessError, OSError):
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return pid, False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return pid, False
    except PermissionError:
        return pid, True
    status = Path(f"/proc/{pid}/status")
    if status.is_file() and "State:\tZ" in status.read_text(errors="replace"):
        return pid, False
    return pid, True


def _on_mocks() -> bool:
    from myrmidon.engines import mock

    return str(mock.BIN) in os.environ.get("PATH", "").split(os.pathsep)[:1]


def _status(ctx: ToolContext, summary: str, **fields) -> LoopObservation:
    from myrmidon.core.loop import snapshot

    snap = snapshot(ctx)
    pid, alive = follower(ctx)
    return LoopObservation(
        ok=True, summary=summary, plan_id=snap.get("plan_id", ""), follower_pid=pid,
        follower_alive=alive, done=snap.get("done", False), steps=snap.get("steps", []),
        running=snap.get("running", []), questions=snap.get("questions", []),
        journal=snap.get("journal", []), events=snap.get("events", []),
        booked_core_h=snap.get("booked_core_h", 0.0),
        ceiling_core_h=snap.get("ceiling_core_h", 0.0), **fields)


class LoopStartParams(BaseModel):
    every_s: int = Field(default=60, ge=5, description="seconds between passes")
    model: str = Field(default="", description="a wake model: a parked question is put to "
                       "it before it goes to the inbox (default: none, every question "
                       "waits for a person)")
    endpoint: str = Field(default="", description="the named endpoint that serves it")


def loop_start_gate(ctx: ToolContext, params: LoopStartParams):
    from myrmidon.core.approvals import PLAN_FILE, holding
    from myrmidon.core.artifacts import sha256_file

    warnings: list[ValidationWarning] = []
    checked = ["plan_approved"]
    plan = ctx.workspace.root / PLAN_FILE
    approved = plan.is_file() and any(a.plan_sha256 == sha256_file(plan)
                                      for a in holding(ctx.workspace))
    if not approved:
        warnings.append(ValidationWarning(
            id="LOOP_PLAN_NOT_APPROVED", severity="block",
            message="the study's plan is not approved, so the loop would spend nothing: "
            "approve it first (plan_approve, the researcher's step)"))
    return warnings, checked


@tool(
    "loop_start",
    description="Start the study's loop as a detached process that outlives this session: "
    "every few seconds it collects what ended, checks it, continues what is ready and parks "
    "what needs the researcher. Needs an approved plan. Idempotent: a running follower is "
    "reported, not duplicated. loop_status says where it stands; the researcher answers a "
    "parked question with loop_answer or `myrmidon answer`.",
    returns=LoopObservation,
    tags={"plan", "jobs"},
    gate=loop_start_gate,
)
def loop_start(ctx: ToolContext, params: LoopStartParams) -> LoopObservation:
    pid, alive = follower(ctx)
    if alive:
        return _status(ctx, f"the loop already runs as process {pid}")
    root = ctx.workspace.root
    cmd = [sys.executable, "-m", "myrmidon", "tick", "--study", str(root),
           "--follow", str(params.every_s)]
    if params.model:
        cmd += ["--model", params.model]
    if params.endpoint:
        cmd += ["--endpoint", params.endpoint]
    if _on_mocks():
        cmd.append("--mock")
    log = (ctx.workspace.state_dir / LOG_FILENAME).open("a")
    proc = subprocess.Popen(  # noqa: S603
        cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True)
    _pid_path(ctx).write_text(f"{proc.pid}\n")
    return _status(ctx, f"the loop runs as process {proc.pid}, a pass every "
                   f"{params.every_s} s; its log is {ctx.workspace.relpath(Path(log.name))}")


class EmptyParams(BaseModel):
    pass


@tool(
    "loop_stop",
    description="Stop the study's detached loop process. Jobs already submitted keep "
    "running; the next loop_start or `myrmidon tick` picks up where it stopped.",
    returns=LoopObservation,
    tags={"plan", "jobs"},
)
def loop_stop(ctx: ToolContext, params: EmptyParams) -> LoopObservation:
    pid, alive = follower(ctx)
    if not alive:
        return _status(ctx, "no loop process is running for this study")
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    return _status(ctx, f"stopped the loop process {pid}")


@tool(
    "loop_tick",
    description="One pass of the study's loop in this session: collect what ended, check "
    "it, continue what is ready, park what needs the researcher. For a study whose loop "
    "process is not running; nothing is spent without an approved plan.",
    returns=LoopObservation,
    tags={"plan", "jobs"},
)
def loop_tick(ctx: ToolContext, params: EmptyParams) -> LoopObservation:
    from myrmidon.core.loop import render, tick

    report = tick(ctx)
    obs = _status(ctx, render(report), submitted=report.submitted, collected=report.collected)
    obs.ok = not report.blocked
    obs.record = False  # the pass's own calls are the ledger's; the pass is a look
    obs.kind = "observe"
    return obs


@tool(
    "loop_status",
    description="Where the study's loop stands, from its files and no model: each step's "
    "state, the live jobs, the questions waiting on the researcher with their options, the "
    "open journal, the last events, whether the loop process runs.",
    returns=LoopObservation,
    tags={"plan", "jobs"},
    navigation=True,
)
def loop_status(ctx: ToolContext, params: EmptyParams) -> LoopObservation:
    obs = _status(ctx, "")
    waiting = len(obs.questions)
    obs.summary = (
        ("the plan is done" if obs.done else
         f"{sum(1 for s in obs.steps if s['state'] == 'done')}/{len(obs.steps)} step(s) done, "
         f"{len(obs.running)} job(s) live")
        + (f"; {waiting} question(s) wait on the researcher — show them, with the options, "
           "and record their choice with loop_answer" if waiting else "")
        + (f"; the loop process {obs.follower_pid} runs" if obs.follower_alive
           else "; no loop process runs (loop_start, or loop_tick for one pass)"))
    obs.record = False
    return obs


class LoopAnswerParams(BaseModel):
    entry_id: str = Field(description="the inbox question (in-…), from loop_status")
    option: str = Field(description="one of its option ids, as the researcher chose")
    acknowledge_warnings: list[str] = Field(default_factory=list)


def loop_answer_gate(ctx: ToolContext, params: LoopAnswerParams):
    from myrmidon.core.spool import Inbox

    checked = ["question_open", "researcher_present"]
    inbox = Inbox(ctx.workspace.root)
    question = inbox.get(params.entry_id)
    if question is None or inbox.answered(params.entry_id) is not None:
        return [ValidationWarning(
            id="LOOP_QUESTION_UNKNOWN", severity="block",
            message=f"no open question {params.entry_id}; loop_status lists them")], checked
    if params.option not in {o.id for o in question.options}:
        return [ValidationWarning(
            id="LOOP_OPTION_UNKNOWN", severity="block",
            message=f"{params.option!r} is not an option of {params.entry_id}: "
            + ", ".join(o.id for o in question.options))], checked
    chosen = next(o for o in question.options if o.id == params.option)
    return [ValidationWarning(
        id="LOOP_ANSWER_IS_THE_RESEARCHERS", severity="warn", needs_human=True,
        message=f"answering {question.subject}: {chosen.text}"
        + (f" ({chosen.cost_core_h:g} core-h)" if chosen.cost_core_h else "")
        + " — the researcher's decision, recorded in their name",
        assumption="the researcher chose this option after seeing the question")], checked


@tool(
    "loop_answer",
    description="Record the researcher's answer to a parked question, by option id; the "
    "next pass of the loop carries it out. The choice is theirs: show the question and the "
    "options first, and call this only on their word.",
    returns=LoopObservation,
    tags={"plan", "jobs"},
    expensive=True,
    gate=loop_answer_gate,
)
def loop_answer(ctx: ToolContext, params: LoopAnswerParams) -> LoopObservation:
    import getpass

    from myrmidon.core.plan_binding import current_plan_id
    from myrmidon.core.spool import Inbox, Spool

    inbox = Inbox(ctx.workspace.root)
    question = inbox.get(params.entry_id)
    who = getpass.getuser()
    inbox.answer(params.entry_id, params.option, by=who)
    chosen = next(o for o in question.options if o.id == params.option)
    Spool(ctx.workspace.root).raise_event(
        "human", question.subject, plan_id=current_plan_id(ctx.workspace),
        inbox=params.entry_id, option=params.option, action=chosen.action, args=chosen.args,
        by=who)
    return _status(ctx, f"{params.entry_id}: {params.option} — {chosen.text}; the next pass "
                   "carries it out")
