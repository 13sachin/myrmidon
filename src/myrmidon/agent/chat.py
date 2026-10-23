"""`myrmidon`: the native agent at a terminal, with the researcher at the keyboard.

Three parts, kept apart:

- ``Renderer`` turns the loop's events into lines on the terminal. By default it shows the
  study — PLAN, DECISION, PROGRESS, EVIDENCE, RESULT — and not the agent's working;
  ``--debug`` shows every call and result. Presentation only: it reads the transcript
  records as they are written, decides nothing, and hides nothing from the transcript.
- ``ask_researcher`` is the loop's ``on_decision``: a finding that needs a human is shown
  with its numbers and the researcher answers yes or no. A yes becomes an approval of that
  one call in the ledger (core/approvals.py); the model never acknowledges it.
- ``repl`` reads questions and slash commands. Slash commands are views of the study
  (status, jobs, plan); none of them calls a tool, so none can change what the ledger
  holds.

Ctrl-C while the agent works stops it and keeps the session; at the prompt it clears the
line. Ctrl-D or ``/exit`` saves and leaves. Leaving never cancels a scheduler job.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from myrmidon.agent.session import ChatSession

HELP = """\
[bold]Ask in your own words[/]; the agent works through myrmidon's tools and answers.
Enter sends; Esc then Enter starts a new line.

  /model    pick another model (or /model NAME); remembered for next time
  /endpoint pick another endpoint (or /endpoint NAME)
  /status   session, model, study, tokens
  /jobs     this study's scheduler jobs
  /plan     this study's plan, if it has one
  /loop     where the study's loop stands: steps, live jobs, what waits on you
  /help     this text
  /exit     save and leave (Ctrl-D too) — jobs keep running

Ctrl-C while the agent works stops it; the session is kept and you can ask again."""

_ARG_CHARS = 140


def _short(value, limit: int = _ARG_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


LEVELS = ("research", "verbose", "debug")

# What a tool result is to a researcher, decided by its Observation type (the transcript's
# `observation_type`) — never by the summary's prose and never by a tool's name. A type
# named nowhere here is EVIDENCE: a new parser is shown until someone decides otherwise.
_LOOKUP = frozenset({
    "ToolListObservation", "SkillObservation", "ProtocolObservation", "KnowledgeObservation",
    "LiteratureObservation", "LookupObservation", "PdbSearchObservation",
    "ChemicalObservation", "InputSetObservation", "SessionSummary",
    "EngineReport", "PlanCheckObservation", "VizObservation",
})
_STEP = frozenset({
    "WorkspaceObservation", "StructureObservation", "PrepObservation",
    "QMMMPrepObservation", "QMRegionObservation", "InputSetApplication",
    "FMPlanObservation", "FMDisplacePlanObservation", "FMDatasetObservation",
    "FMMergeObservation",
})
_JOB = frozenset({"JobObservation", "ScriptObservation"})
# a bare Observation is a file read or write, or an engine step: its tags tell which
_LOOKUP_TAGS = frozenset({"files", "workspace"})

_STYLE = {"PLAN": "bold blue", "DECISION": "bold yellow", "PROGRESS": "magenta",
          "EVIDENCE": "bold green", "ERROR": "bold red"}


def classify(event: dict) -> str | None:
    """PLAN, DECISION, PROGRESS, EVIDENCE or ERROR for a transcript record a researcher
    should see; None for the agent's own working (lookups, retries, recoveries), which the
    transcript keeps and ``--debug`` shows."""
    kind = event.get("kind")
    if kind == "researcher_decision":
        return "DECISION"
    if kind == "error":
        return "ERROR"
    if kind != "tool_result":
        return None
    obs = event.get("obs") or {}
    failure = event.get("failure")
    if failure == "gate_refused":
        # a finding the model may acknowledge is its to handle (shown when it is accepted);
        # one that needs a person is asked at the keyboard (ask_researcher)
        return "DECISION" if obs.get("blocking") else None
    if failure:
        return None  # bad parameters, no study, a raise: told to the model, which recovers
    otype = event.get("observation_type") or "Observation"
    if otype == "PlanObservation":
        return "PLAN" if event.get("ok") else None
    if otype in _JOB:
        return "PROGRESS" if obs.get("job_id") else None
    if otype in _LOOKUP or (otype == "Observation"
                            and _LOOKUP_TAGS & set(event.get("tags") or ())):
        return None
    if not event.get("ok"):
        return "ERROR"
    return "PROGRESS" if otype in _STEP else "EVIDENCE"


def _first_line(text: str | None) -> str:
    text = (text or "").strip()
    return text.splitlines()[0] if text else ""


class Renderer:
    """The loop's events as terminal lines, at one of three levels:

    - ``research`` (default): the study — PLAN, DECISION, PROGRESS, EVIDENCE, ERROR — and
      nothing of the agent's own working;
    - ``verbose``: the same, with provenance ids, artifacts, accepted findings' detail and
      a note when an answer was corrected;
    - ``debug``: every call, result, bounce and per-step token count.

    Presentation only: what is hidden here is still in the transcript.
    """

    def __init__(self, console: Console, level: str = "research"):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        self.console = console
        self.level = level
        self._status = None
        # presentation state of this process only; losing it loses nothing
        self.study_work = False  # this question changed or showed the study
        self._acks: list[str] = []  # acknowledged by the last call
        self._findings: dict[str, str] = {}  # finding id -> message, from refusals seen
        self._approved: set[str] = set()  # accepted by the researcher, not the agent
        self._jobs: dict[str, tuple] = {}  # job id -> last (state, step, time_ps) shown

    def begin_question(self) -> None:
        self.study_work = False

    # the loop calls this for every transcript record, and "thinking" before model calls
    def __call__(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == "thinking":
            self._spin(f"thinking (step {event.get('turn')})" if self.level == "debug"
                       else "working")
            return
        if self.level == "debug":
            self.stop_spinner()
            handler = getattr(self, f"_debug_{kind}", None)
            if handler is not None:
                handler(event)
            return
        self._remember(event)
        label = classify(event)
        if label is None:
            self._quiet(event)
            return
        self.stop_spinner()
        if kind == "tool_result" and event.get("ok"):
            self._accepted()  # the assumption first, then what it let through
        getattr(self, f"_show_{label.lower()}")(event)
        if label != "ERROR":
            self.study_work = True

    def _remember(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == "tool_call":
            self._acks = list((event.get("args") or {}).get("acknowledge_warnings") or [])
        elif kind == "tool_result" and event.get("failure") == "gate_refused":
            obs = event.get("obs") or {}
            for group in ("blocking", "human_required", "unacknowledged"):
                for f in obs.get(group) or []:
                    self._findings[f.get("id")] = f.get("message", "")
        elif kind == "researcher_decision" and event.get("approved"):
            self._approved.update(event.get("findings") or [])

    def _quiet(self, event: dict) -> None:
        """What research mode leaves out, said once where it matters."""
        kind = event.get("kind")
        if kind == "interrupted":
            self.stop_spinner()
            self.console.print(Text("  stopped — the session is kept; ask again or /exit",
                                    style="yellow"))
        elif kind == "truncated":
            self.stop_spinner()
            self.console.print(Text("  the reply hit the output limit and was cut off",
                                    style="yellow"))
        elif kind == "grounding_bounce" and self.level == "verbose":
            self.stop_spinner()
            self.console.print(Text("  (an answer cited a number no tool gave; it was sent "
                                    "back)", style="dim"))

    def _line(self, label: str, text: str, source: str | None = None) -> None:
        line = Text(f"{label:<10}", style=_STYLE[label])
        line.append(_short(text, 200))
        if source:
            line.append(f"  · {source}", style="dim")
        self.console.print(line)

    def _detail(self, event: dict) -> None:
        if self.level != "verbose":
            return
        obs = event.get("obs") or {}
        if event.get("provenance_id"):
            self.console.print(Text(f"          provenance {event['provenance_id']}",
                                    style="dim"))
        for art in obs.get("artifacts") or []:
            self.console.print(Text(f"          {art.get('role')}: {art.get('path')}",
                                    style="dim"))
        for w in obs.get("warnings") or []:
            self.console.print(Text(f"          {w.get('id')}: {_short(w.get('message'), 150)}",
                                    style="yellow"))

    def _show_plan(self, ev: dict) -> None:
        obs = ev.get("obs") or {}
        parts = [obs.get("question") or _first_line(obs.get("summary"))]
        if obs.get("shape"):
            parts.append(obs["shape"])
        if obs.get("steps"):
            parts.append(f"{len(obs['steps'])} steps")
        parts.append(f"{obs.get('total_core_h', 0):g} core-h")
        if obs.get("unestimated"):
            parts.append(f"{len(obs['unestimated'])} unestimated")
        if obs.get("digest"):
            parts.append(f"digest {obs['digest'][:12]}")
        self._line("PLAN", " · ".join(p for p in parts if p), "/plan shows it")
        self._detail(ev)

    def _show_decision(self, ev: dict) -> None:
        if ev.get("kind") == "researcher_decision":
            ids = ", ".join(ev.get("findings") or [])
            self._line("DECISION", f"approved by you: {ids}" if ev.get("approved")
                       else f"declined by you: {ids}")
            return
        for f in (ev.get("obs") or {}).get("blocking") or []:
            self._line("DECISION", f"refused {ev.get('tool')}: {f.get('id')} — "
                       f"{f.get('message', '')}")

    def _accepted(self) -> None:
        """Findings the agent acknowledged for a call that then ran: the assumptions made
        on the researcher's behalf, which must never be silent."""
        for fid in self._acks:
            if fid in self._approved:
                continue
            message = self._findings.get(fid)
            self._line("DECISION", f"accepted {fid}" + (f" — {message}" if message else ""),
                       "by the agent")
        self._acks = []

    def _show_progress(self, ev: dict) -> None:
        obs = ev.get("obs") or {}
        if obs.get("job_id"):
            progress = obs.get("progress") or {}
            key = (obs.get("state"), progress.get("step"), progress.get("time_ps"))
            if self._jobs.get(obs["job_id"]) == key:
                return  # an unchanged poll
            self._jobs[obs["job_id"]] = key
        text = _first_line(obs.get("summary"))
        if obs.get("diagnosis_id"):
            text += f"  [{obs['diagnosis_id']}]"
        self._line("PROGRESS", text, ev.get("tool"))
        self._detail(ev)

    def _show_evidence(self, ev: dict) -> None:
        self._line("EVIDENCE", _first_line((ev.get("obs") or {}).get("summary")),
                   ev.get("tool"))
        self._detail(ev)

    def _show_error(self, ev: dict) -> None:
        if ev.get("kind") == "error":
            self._line("ERROR", f"model error: {ev.get('error', '')}")
        else:
            self._line("ERROR", _first_line((ev.get("obs") or {}).get("summary")),
                       ev.get("tool"))
        self._detail(ev)

    def _spin(self, text: str) -> None:
        if not self.console.is_terminal:
            return
        if self._status is None:
            self._status = self.console.status(text, spinner="dots")
            self._status.start()
        else:
            self._status.update(text)

    def stop_spinner(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    # --- debug: every record, as the agent works --------------------------------------

    def _debug_tool_call(self, ev: dict) -> None:
        args = ev.get("args") or {}
        shown = ", ".join(f"{k}={_short(v, 60)}" for k, v in args.items()
                          if k != "acknowledge_warnings" and v not in (None, "", [], {}))
        acks = args.get("acknowledge_warnings")
        line = Text("  ▸ ", style="cyan")
        line.append(ev.get("tool", "?"), style="bold cyan")
        if shown:
            line.append(f"  {_short(shown)}", style="dim")
        if acks:
            line.append(f"  acknowledging {acks}", style="yellow")
        if ev.get("error") == "unknown-tool":
            line.append("  (no such tool)", style="red")
        elif ev.get("error") == "bad-args":
            line.append("  (unparsable arguments)", style="red")
        self.console.print(line)
        if ev.get("tool") not in (None, "") and not ev.get("error"):
            self._spin(f"running {ev.get('tool')}")

    def _debug_tool_result(self, ev: dict) -> None:
        obs = ev.get("obs") or {}
        summary = (obs.get("summary") or "").strip()
        first = summary.splitlines()[0] if summary else ""
        ok = ev.get("ok")
        mark, style = ("✓", "green") if ok else ("✗", "red")
        line = Text(f"    {mark} ", style=style)
        line.append(_short(first, 200), style="" if ok else style)
        if obs.get("job_id"):
            line.append(f"  [{obs['job_id']}]", style="magenta")
        self.console.print(line)
        if ev.get("failure") == "gate_refused":
            for group in ("blocking", "human_required", "unacknowledged"):
                for finding in obs.get(group) or []:
                    self.console.print(Text(f"      {finding.get('id')}: ", style="yellow")
                                       + Text(_short(finding.get("message", ""), 160)))

    def _debug_grounding_bounce(self, ev: dict) -> None:
        self.console.print(Text(
            f"  ⟲ answer sent back: {', '.join(ev.get('ungrounded') or [])} came from no "
            "tool — asking for one that cites observations", style="yellow"))

    def _debug_researcher_decision(self, ev: dict) -> None:
        if ev.get("approved"):
            self.console.print(Text(f"  ✓ approved by you for this call ({ev.get('approval')})",
                                    style="green"))
        else:
            self.console.print(Text("  ✗ declined by you", style="red"))

    def _debug_error(self, ev: dict) -> None:
        self.console.print(Text(f"  model error: {_short(ev.get('error', ''), 300)}",
                                style="bold red"))

    def _debug_truncated(self, ev: dict) -> None:
        self.console.print(Text("  the reply hit the output limit and was cut off",
                                style="yellow"))

    def _debug_interrupted(self, ev: dict) -> None:
        self.console.print(Text("  stopped — the session is kept; ask again or /exit",
                                style="yellow"))

    def _debug_turn(self, ev: dict) -> None:
        self.console.print(Text(
            f"    · step {ev.get('turn')}: {ev.get('prompt_tokens')}+"
            f"{ev.get('completion_tokens')} tokens, {ev.get('model_s')}s", style="dim"))


def make_decider(console: Console, renderer: Renderer, interactive: bool):
    """The loop's ``on_decision``. Without a terminal to ask at, the answer is no."""
    from rich.prompt import Confirm

    def ask_researcher(tool: str, args: dict, findings) -> bool:
        renderer.stop_spinner()
        body = Text()
        for f in findings:
            body.append(f"{f.id}\n", style="bold yellow")
            body.append(f"{f.message}\n")
            if f.assumption:
                body.append(f"assumes: {f.assumption}\n", style="dim")
            if f.suggestion:
                body.append(f"suggestion: {f.suggestion}\n", style="dim")
        console.print(Panel(body, title=f"{tool} needs your decision",
                            border_style="yellow", expand=False))
        if not interactive:
            console.print("  (no terminal to ask at: declined)")
            return False
        return Confirm.ask("  Approve this call?", default=False, console=console)

    return ask_researcher


def _status(session: ChatSession) -> str:
    s, ctx = session.state, session.loop.ctx
    study = ctx.workspace.root if ctx.workspace is not None else "(none — the agent can "\
        "create or open one)"
    return (f"session   {s.session_id}\n"
            f"model     {s.model} via {s.runner}"
            + (f" @ {s.endpoint}" if s.endpoint else "") + "\n"
            f"study     {study}\n"
            f"asked     {s.asks} question(s); tokens {s.prompt_tokens}+{s.completion_tokens}\n"
            f"last stop {s.last_stop or '—'}\n"
            f"saved in  {session.root / s.session_id}")


def _jobs(session: ChatSession) -> str:
    ctx = session.loop.ctx
    if ctx.workspace is None:
        return "no study open, so no jobs"
    from myrmidon.execution.jobstore import open_jobstore

    jobs = open_jobstore(ctx.workspace).list()
    if not jobs:
        return "no jobs in this study"
    lines = [f"{j.job_id:<14} {j.state.value:<10} {j.label or '':<28} "
             f"submitted {j.submitted_at[:16]}" for j in jobs]
    return "\n".join(lines) + "\n(the stored state; the agent's job_status asks the scheduler)"


def _plan(session: ChatSession) -> str:
    ctx = session.loop.ctx
    if ctx.workspace is None:
        return "no study open"
    from myrmidon.plan_tools import read_plan, render_plan, unresolved

    plan = read_plan(ctx)
    return render_plan(plan, digest=plan.digest(), still_open=unresolved(ctx, plan), ctx=ctx) \
        if plan is not None else \
        "this study has no plan"


def _loop(session: ChatSession) -> str:
    """Where the study's loop stands: the same files `myrmidon status` reads."""
    ctx = session.loop.ctx
    if ctx.workspace is None:
        return "no study open"
    from myrmidon.core.loop import snapshot
    from myrmidon.loop_tools import follower

    snap = snapshot(ctx)
    if not snap.get("plan_id"):
        return snap.get("note", "this study has no plan")
    pid, alive = follower(ctx)
    lines = [f"loop process: {pid} (alive)" if alive else "loop process: none"]
    lines += [f"  {s['state']:<8} {s['step']:<28} {s['tool']}"
              + (f"  ({s['segments']} segments)" if s["segments"] > 1 else "")
              for s in snap["steps"]]
    for job in snap["running"]:
        lines.append(f"  job {job['job_id']} {job['state']} for {job['step']}")
    for q in snap["questions"]:
        lines.append(f"  waiting on you: {q['entry_id']} {q['step']}: {q['question'][:100]}")
        lines += [f"    {o['id']:<10} {o['text']}" for o in q["options"]]
    lines.append("the plan is done" if snap["done"] else
                 f"{snap['booked_core_h']:g} of {snap['ceiling_core_h']:g} core-h booked")
    return "\n".join(lines)


def pick(console: Console, title: str, options: list[str], current: str | None) -> str | None:
    """A numbered list and a number typed back; None when there is nothing to pick or
    nothing was picked."""
    from rich.prompt import Prompt

    if not options:
        return None
    console.print(f"[bold]{title}[/]")
    for i, option in enumerate(options, 1):
        mark = "  [green]← current[/]" if option == current else ""
        console.print(f"  {i:>2}  {option}{mark}")
    choice = Prompt.ask("  number (Enter keeps the current)", default="", console=console,
                        show_default=False).strip()
    if not choice:
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return options[int(choice) - 1]
    return choice if choice in options else None


def _switch_model(arg: str, session: ChatSession, console: Console) -> None:
    from myrmidon.agent.session import served_models

    served = served_models(session.state.endpoint)
    if arg:
        if served and arg not in served:
            console.print(f"{arg!r} is not served by {session.state.endpoint}; /model lists "
                          "what is", style="yellow")
            return
        chosen = arg
    else:
        if not served:
            console.print("the endpoint did not list its models; use /model NAME",
                          style="yellow")
            return
        chosen = pick(console, f"models on {session.state.endpoint}", served,
                      session.state.model)
    if chosen and chosen != session.state.model:
        session.switch(model=chosen)
        console.print(f"now {chosen} — from the next question on", style="green")


def _switch_endpoint(arg: str, session: ChatSession, console: Console) -> None:
    from myrmidon.endpoints import EndpointError, get_endpoint, load_endpoints

    names = sorted(load_endpoints())
    chosen = arg or pick(console, "endpoints", names, session.state.endpoint)
    if not chosen or chosen == session.state.endpoint:
        return
    try:
        get_endpoint(chosen).key()
    except EndpointError as exc:
        console.print(str(exc), style="red")
        return
    session.switch(endpoint=chosen)
    console.print(f"endpoint {chosen}; pick its model:", style="green")
    _switch_model("", session, console)


def slash(command: str, session: ChatSession, console: Console) -> bool:
    """Run a slash command; False means leave."""
    name, _, arg = command.strip().partition(" ")
    name, arg = name.lower(), arg.strip()
    if name in ("/exit", "/quit"):
        return False
    views = {"/status": _status, "/jobs": _jobs, "/plan": _plan, "/loop": _loop}
    if name == "/help":
        console.print(HELP)
    elif name == "/model":
        _switch_model(arg, session, console)
    elif name == "/endpoint":
        _switch_endpoint(arg, session, console)
    elif name in views:
        try:
            console.print(views[name](session), markup=False, highlight=False)
        except Exception as exc:  # a view must never end the session
            console.print(f"{name} failed: {exc}", style="red")
    else:
        console.print(f"unknown command {name} — /help lists them", style="yellow")
    return True


def _prompt_session(history: Path):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings

    keys = KeyBindings()

    @keys.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    history.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(history=FileHistory(str(history)), key_bindings=keys)


def show_answer(console: Console, episode, study_work: bool = False) -> None:
    """The answer; framed as the RESULT when the question did study work, plain prose when
    it was only conversation."""
    if episode.final_answer:
        console.print()
        body = Markdown(episode.final_answer)
        console.print(Panel(body, title="RESULT", title_align="left", border_style="green")
                      if study_work else body)
        if episode.grounding is not None and not episode.grounding.ok:
            console.print(Text(f"  note: {', '.join(episode.grounding.ungrounded)} in this "
                               "answer came from no tool", style="yellow"))
    elif episode.stop_reason.value not in ("INTERRUPTED", "MODEL_ERROR"):
        console.print(Text(f"  stopped without an answer: {episode.stop_reason.value}",
                           style="yellow"))
    console.print()


def repl(session: ChatSession, console: Console, renderer: Renderer,
         first: str | None = None) -> None:
    """Questions until /exit or Ctrl-D; the session is saved after every answer."""
    interactive = console.is_terminal
    prompt = _prompt_session(session.root / "history") if interactive else None
    pending = first
    try:
        while True:
            if pending is not None:
                text, pending = pending, None
            elif prompt is not None:
                try:
                    text = prompt.prompt("› ", bottom_toolbar=lambda: _toolbar(session))
                except KeyboardInterrupt:
                    continue  # Ctrl-C at the prompt clears the line
                except EOFError:
                    break
            else:
                break  # not a terminal: one question (the argument) and done
            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                if not slash(text, session, console):
                    break
                continue
            renderer.begin_question()
            try:
                episode = session.ask(text)
            except KeyboardInterrupt:  # between the loop's own handling and the save
                renderer.stop_spinner()
                session.checkpoint()
                console.print("  stopped — the session is kept", style="yellow")
                continue
            finally:
                renderer.stop_spinner()
            show_answer(console, episode, renderer.study_work)
    finally:
        renderer.stop_spinner()
        session.checkpoint()
        console.print(Text(f"saved — continue with: myrmidon -r {session.state.session_id}",
                           style="dim"))


def _toolbar(session: ChatSession) -> str:
    ctx, s = session.loop.ctx, session.state
    study = ctx.workspace.root.name if ctx.workspace is not None else "no study"
    return (f" {s.model}" + (f" @ {s.endpoint}" if s.endpoint else "")
            + f"  ·  {study}  ·  {s.prompt_tokens + s.completion_tokens} tokens  ·  /help")
