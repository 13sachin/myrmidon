"""Model-agnostic agent loop: the native way to drive the registry.

Same tools, same gates, same provenance as the MCP path — plus hard stops
(max turns, token budget, wall clock), a full episode transcript, and the
numeric grounding check on the final answer.
"""

from __future__ import annotations

import hashlib
import json
import time
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.agent.grounding import (
    BOUNCE,
    GROUNDING_RETRIES,
    SCRIPT_TOOLS,
    GroundingReport,
    NumericIndex,
    citable,
)
from myrmidon.agent.prompts import default_system_prompt
from myrmidon.agent.transcript import TRANSCRIPT_FORMAT, Transcript, digest, trace_history
from myrmidon.core.observation import model_view
from myrmidon.core.tool import REGISTRY, ToolContext, ToolRegistry
from myrmidon.errors import UnknownToolError

_MAX_TOOL_RESULT_CHARS = 8000
#: a model request that timed out, or that the server failed (5xx, 429, a dropped
#: connection), is asked again after each of these waits: a slow endpoint answers most
#: calls in a few minutes, and one timeout must not cost the episode. A refusal the request
#: itself earned (400 credit, 401/403 key, a bad parameter) is not retried: it would fail
#: the same way
MODEL_RETRY_WAITS_S = (10.0, 30.0)


def transient(exc: BaseException) -> bool:
    """A model-request failure that asking again may cure: a timeout, a lost connection,
    a server error or rate limit; never a 4xx the request itself caused."""
    seen = exc
    while seen is not None:
        status = getattr(seen, "status_code", None)
        if status is None:
            status = getattr(getattr(seen, "response", None), "status_code", None)
        if isinstance(status, int):
            return status >= 500 or status in (408, 409, 429)
        if isinstance(seen, (TimeoutError, ConnectionError)) or type(seen).__name__ in (
                "APITimeoutError", "APIConnectionError", "TimeoutException", "ConnectError",
                "ReadTimeout", "RemoteProtocolError"):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


class StopReason(StrEnum):
    COMPLETED = "COMPLETED"
    MAX_TURNS = "MAX_TURNS"
    TOKEN_BUDGET = "TOKEN_BUDGET"
    WALL_CLOCK = "WALL_CLOCK"
    MODEL_ERROR = "MODEL_ERROR"
    # the reply hit the output-token limit: what was cut off is not an answer
    TRUNCATED = "TRUNCATED"
    # the researcher pressed Ctrl-C (chat): the session is kept, the question unanswered
    INTERRUPTED = "INTERRUPTED"


class Episode(BaseModel):
    episode_id: str
    model: str
    task: str
    stop_reason: StopReason
    final_answer: str | None = None
    error: str | None = None  # model/transport failure text (never graded as an answer)
    # the sampling parameters every model call carried (empty: provider defaults)
    sampling: dict = Field(default_factory=dict)
    # the named endpoint that served it (endpoints.py); None: the provider's own server
    endpoint: str | None = None
    # False when the driver reports no usage (Cursor): the totals are unknown, not zero
    tokens_known: bool = True
    n_turns: int = 0
    n_tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    grounding: GroundingReport | None = None
    transcript_path: str | None = None
    workspace: str | None = None
    wall_s: float = 0.0  # from the first model call to the answer, or the stop


def tool_schemas(registry: ToolRegistry, engines: dict) -> list[dict]:
    """OpenAI-style tool definitions from the same specs the MCP server shows: every
    registered tool whose engines are present on this machine."""
    specs = registry.available(engines)
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.params_model.model_json_schema(),
            },
        }
        for spec in specs
    ]


def _dump_message(msg) -> dict:
    # reasoning is recorded with the response (`model_call`), never sent back
    return msg.model_dump(exclude_none=True, exclude={"reasoning"})


def _version(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None


class AgentLoop:
    def __init__(
        self,
        model: str,
        ctx: ToolContext,
        registry: ToolRegistry = REGISTRY,
        max_turns: int = 40,
        token_budget: int = 1_500_000,
        wall_clock_s: float = 3600.0,
        transcript_dir: Path = Path("episodes"),
        completion_fn=None,
        enforce_grounding: bool = True,
        unattended: bool = True,
        sampling: dict | None = None,
        connection: dict | None = None,
        endpoint: str | None = None,
        on_event=None,
        on_decision=None,
        opening: dict | None = None,
        retry_waits: tuple[float, ...] | None = None,
    ):
        self.model = model
        self.retry_waits = tuple(MODEL_RETRY_WAITS_S if retry_waits is None else retry_waits)
        # where the calls go (api_base, api_key): passed on every call, never recorded —
        # unlike `sampling`, which the episode and the report carry
        self.connection = dict(connection or {})
        self.endpoint = endpoint
        # passed on every call and recorded on the episode; a provider that rejects one
        # fails the call visibly rather than running with its own default
        self.sampling = dict(sampling or {})
        self.ctx = ctx
        # nobody sees these tool calls as they happen: findings that need a human
        # decision refuse instead of accepting the model's own acknowledgment
        ctx.unattended = unattended
        self.registry = registry
        self.max_turns = max_turns
        self.token_budget = token_budget
        self.wall_clock_s = wall_clock_s
        self.transcript_dir = transcript_dir
        # False = measure grounding but never bounce (the no-harness baseline)
        self.enforce_grounding = enforce_grounding
        if completion_fn is None:
            from myrmidon.agent.pydantic_adapter import PydanticCompletion

            completion_fn = PydanticCompletion()
        self.completion_fn = completion_fn
        # sees every transcript event as it is written, plus "thinking" before each model
        # call (never written): what a terminal renders. Presentation only.
        self.on_event = on_event
        # (tool, args, findings) -> bool: a researcher at the keyboard deciding a finding
        # that needs a human. None: nobody is there, and the refusal stands.
        self.on_decision = on_decision
        # calls the researcher declined since their last message, by identity: not put
        # to them twice in one reply
        self._declined: set[str] = set()
        self.messages: list[dict] = []
        self.transcript: Transcript | None = None
        self.index: NumericIndex | None = None
        self._schemas: list[dict] | None = None
        # a tool schema the first turn of a session is offered alone, with a call required
        # (session.build_loop gives the skill choice); skipped when the session's tools do
        # not include it, and on a resumed session, which made its choice already
        self.opening = opening
        self._opening_due = False
        # the trace (`model_call` events): how much of self.messages is recorded already,
        # the schema sets written to this transcript, and the requests made so far
        self._sent = 0
        self._schema_hashes: set[str] = set()
        self._calls = 0
        # a registry's observations are measurements; a baseline's tools are a shell
        self._source = "observation" if registry is REGISTRY else "shell"

    def run(self, task_prompt: str, system_prompt: str | None = None) -> Episode:
        """One question, one episode: what `myrmidon run` asks."""
        self.start(system_prompt)
        return self.ask(task_prompt)

    def start(self, system_prompt: str | None = None, transcript: Transcript | None = None,
              messages: list[dict] | None = None) -> None:
        """Begin a session, or — given the transcript and messages of one — resume it, with
        the grounding index rebuilt from what its tools returned."""
        self.transcript = transcript or Transcript(self.transcript_dir)
        self.transcript.listener = self.on_event
        if messages is not None:
            self.messages = list(messages)
            events = self.transcript.events()
            self.index = rebuild_index(events, self._source)
            self._schemas = tool_schemas(self.registry, self.ctx.engines)
            self._schema_hashes = {e["sha256"] for e in events
                                   if e.get("kind") == "tool_schemas"}
            self._calls = sum(1 for e in events if e.get("kind") == "model_call")
            self._emit_trace(resumed=True)
            self._record_resumed(events)
            return
        self._sent, self._schema_hashes, self._calls = 0, set(), 0
        self._emit_trace(resumed=False)
        system = system_prompt or default_system_prompt()
        self.messages = [{"role": "system", "content": system}]
        self.transcript.emit("system", content=system)
        self.index = NumericIndex()
        self._schemas = None
        self._opening_due = self.opening is not None

    def ask(self, prompt: str) -> Episode:
        """The researcher's next message, worked until an answer or a stop."""
        transcript = self.transcript
        self.messages.append({"role": "user", "content": prompt})
        transcript.emit("user", content=prompt)
        self._declined.clear()  # a new message from the researcher may change their answer
        # numbers the user stated (300 K, 1.0 nm) may be restated, and are reported as such
        self.index.add_tree(prompt, source="prompt")
        if self._schemas is None:
            self._schemas = tool_schemas(self.registry, self.ctx.engines)
            transcript.emit("tools", n_tools=len(self._schemas),
                            schema_chars=len(json.dumps(self._schemas, sort_keys=True)))
        if self._opening_due:
            offered = {s["function"]["name"] for s in self._schemas}
            self._opening_due = self.opening["function"]["name"] in offered
        started = time.monotonic()
        episode = Episode(
            episode_id=transcript.episode_id,
            model=self.model,
            task=prompt,
            stop_reason=StopReason.MAX_TURNS,
            transcript_path=str(transcript.path),
            sampling=self.sampling,
            endpoint=self.endpoint,
        )
        try:
            self._work(episode, started)
        except KeyboardInterrupt:
            self._close_open_calls()
            episode.stop_reason = StopReason.INTERRUPTED
            transcript.emit("interrupted", n_turns=episode.n_turns)

        if self.ctx.workspace is not None:
            episode.workspace = str(self.ctx.workspace.root)
        transcript.emit(
            "episode_end",
            stop_reason=episode.stop_reason.value,
            n_turns=episode.n_turns,
            grounding_ok=episode.grounding.ok if episode.grounding else None,
        )
        episode.wall_s = round(time.monotonic() - started, 1)
        return episode

    def _emit_trace(self, resumed: bool) -> None:
        """What served this session's calls, once at its start (and again on a resume)."""
        import myrmidon

        self.transcript.emit(
            "trace", format=TRANSCRIPT_FORMAT, resumed=resumed, model=self.model,
            endpoint=self.endpoint,
            adapter=type(self.completion_fn).__name__,
            versions={"myrmidon": myrmidon.__version__,
                      "pydantic-ai-slim": _version("pydantic-ai-slim")},
        )

    def _record_resumed(self, events: list[dict]) -> None:
        """A resumed session's history, as far as the transcript does not have it yet: the
        last answer (appended after its call) and any calls closed by an interrupt. A
        history that is not the one the trace implies is written out whole."""
        recorded = trace_history(events, check=False)
        n = len(recorded)
        if self.messages[:n] == recorded:
            payload = {"append": self.messages[n:]}
        else:
            payload = {"replace": self.messages}
        self.transcript.emit("resumed", n_messages=len(self.messages),
                             sha256=digest(self.messages), **payload)
        self._sent = len(self.messages)

    def _schema_ref(self, tools: list[dict]) -> str:
        """The schema set's hash; the set itself the first time this transcript sees it."""
        sha = hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()
        if sha not in self._schema_hashes:
            self._schema_hashes.add(sha)
            self.transcript.emit("tool_schemas", sha256=sha, schemas=tools)
        return sha

    def _record_call(self, request: dict, turn: int, model_s: float, response=None,
                     error: str | None = None) -> None:
        """One request to the model, verbatim: what was added to the history since the
        previous one, the schema set, the settings (never the connection), and the reply
        or the error. Every text the loop sends reaches the model through self.messages,
        so it is in some call's ``append`` without its sender having to record it."""
        self._calls += 1
        self._sent += len(request["append"])
        record = dict(call=self._calls, turn=turn, model=self.model,
                      model_s=round(model_s, 3), **request)
        if response is not None:
            choice = response.choices[0]
            msg = choice.message
            usage = getattr(response, "usage", None)
            record["response"] = {
                "content": getattr(msg, "content", None),
                "tool_calls": [{"id": tc.id, "name": tc.function.name,
                                "arguments": tc.function.arguments}
                               for tc in getattr(msg, "tool_calls", None) or []],
                "reasoning": getattr(msg, "reasoning", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "provider_finish_reason": getattr(response, "provider_finish_reason", None),
                "usage": None if usage is None else {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None)},
            }
        else:
            record["error"] = error
        self.transcript.emit("model_call", **record)

    def _notify(self, kind: str, **payload) -> None:
        if self.on_event is not None:
            self.on_event({"kind": kind, **payload})

    def _work(self, episode: Episode, started: float) -> None:
        transcript, messages, index = self.transcript, self.messages, self.index
        grounding_bounces = 0

        while True:
            if episode.n_turns >= self.max_turns:
                episode.stop_reason = StopReason.MAX_TURNS
                break
            if episode.prompt_tokens + episode.completion_tokens >= self.token_budget:
                episode.stop_reason = StopReason.TOKEN_BUDGET
                break
            if time.monotonic() - started >= self.wall_clock_s:
                episode.stop_reason = StopReason.WALL_CLOCK
                break

            history_chars = len(json.dumps(messages, default=str))
            self._notify("thinking", turn=episode.n_turns + 1)
            opening, self._opening_due = self._opening_due, False
            tools, forced = ([self.opening], {"tool_choice": "required"}) if opening \
                else (self._schemas, {})
            request = dict(append=messages[self._sent:], tools=self._schema_ref(tools),
                           settings={**self.sampling, **forced})
            waits = list(self.retry_waits)
            while True:
                called = time.monotonic()
                try:
                    response = self.completion_fn(
                        model=self.model, messages=messages, tools=tools,
                        **self.sampling, **self.connection, **forced,
                    )
                    error = None
                except Exception as exc:
                    error, text = exc, str(exc)
                    key = self.connection.get("api_key")
                    if key:  # a provider error that echoes the key must not record it
                        text = text.replace(key, "[api_key]")
                    self._record_call(request, episode.n_turns + 1,
                                      time.monotonic() - called, error=text)
                    # the history was sent with the failed request: a retry adds nothing
                    request = dict(request, append=[])
                    if waits and transient(exc) and \
                            time.monotonic() - started + waits[0] < self.wall_clock_s:
                        time.sleep(waits.pop(0))
                        continue
                break
            if error is not None:
                episode.stop_reason = StopReason.MODEL_ERROR
                episode.error = f"model call failed: {text}"
                transcript.emit("error", error=text)
                break

            model_s = time.monotonic() - called
            self._record_call(request, episode.n_turns + 1, model_s, response=response)
            episode.n_turns += 1
            turn = episode.n_turns
            usage = getattr(response, "usage", None)
            call_prompt = call_completion = None
            if usage is not None:
                call_prompt = getattr(usage, "prompt_tokens", 0) or 0
                call_completion = getattr(usage, "completion_tokens", 0) or 0
                episode.prompt_tokens += call_prompt
                episode.completion_tokens += call_completion
                transcript.emit(
                    "usage",
                    prompt_tokens=episode.prompt_tokens,
                    completion_tokens=episode.completion_tokens,
                )

            msg = response.choices[0].message
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            tool_calls = getattr(msg, "tool_calls", None)
            # the tools' time is added when they have run: one record per model call
            turn_record = dict(turn=turn, prompt_tokens=call_prompt,
                               completion_tokens=call_completion, model_s=round(model_s, 3),
                               history_chars=history_chars,
                               tool_calls=len(tool_calls or []),
                               finish_reason=finish_reason,
                               **({"opening": True} if opening else {}))
            if finish_reason == "length":
                transcript.emit("turn", **turn_record)
                # the model's outcome, not a transport failure: graded (as a failure, with
                # no answer), never counted as an error
                episode.stop_reason = StopReason.TRUNCATED
                transcript.emit("truncated", content=msg.content or "")
                break

            if tool_calls:
                tools_started = time.monotonic()
                messages.append(_dump_message(msg))
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError as exc:
                        result_text = json.dumps(
                            {"ok": False, "summary": f"unparsable arguments: {exc}"}
                        )
                        messages.append(
                            {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                        )
                        transcript.emit("tool_call", tool=name, error="bad-args", turn=turn)
                        continue
                    try:
                        self.registry.get(name)
                    except UnknownToolError:
                        # a name the model invented is its mistake to recover from, told
                        # like unparsable arguments; raising would end the episode as a
                        # harness error and excuse the model
                        result_text = json.dumps(
                            {"ok": False, "summary": f"unknown tool '{name}': call one of "
                             "the tools you were given"})
                        messages.append(
                            {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                        )
                        transcript.emit("tool_call", tool=name, args=args,
                                        error="unknown-tool", turn=turn)
                        continue
                    obs, result_text = self._execute(name, args, turn, started, episode)
                    if (self.on_decision is not None and getattr(obs, "human_required", None)
                            and not getattr(obs, "blocking", None)
                            and self.ctx.workspace is not None):
                        obs, result_text = self._researcher_decides(
                            name, args, obs, result_text, turn, started, episode)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                    )
                transcript.emit("turn", **turn_record,
                                tools_s=round(time.monotonic() - tools_started, 3))
                continue

            # No tool calls: this is a final-answer attempt.
            transcript.emit("turn", **turn_record)
            answer = msg.content or ""
            transcript.emit("assistant", content=answer)
            report = index.check(answer)
            if (
                self.enforce_grounding
                and not report.ok
                and grounding_bounces < GROUNDING_RETRIES
            ):
                grounding_bounces += 1
                messages.append(_dump_message(msg))
                bounce = BOUNCE.format(bad=", ".join(report.ungrounded))
                messages.append({"role": "user", "content": bounce})
                transcript.emit("grounding_bounce", ungrounded=report.ungrounded)
                continue

            # the answer stays in the history: the researcher's next message follows it
            messages.append(_dump_message(msg))
            episode.final_answer = answer
            episode.grounding = report
            episode.stop_reason = StopReason.COMPLETED
            break

    def _execute(self, name: str, args: dict, turn: int, started: float,
                 episode: Episode) -> tuple:
        """One registry call, recorded in the transcript; (observation, what the model is
        shown)."""
        transcript = self.transcript
        transcript.emit("tool_call", tool=name, args=args, turn=turn)
        call_started = time.monotonic()
        try:
            obs = self.registry.call(name, self.ctx, args)
        except Exception as exc:  # the registry's own failure is the model's to hear about,
            # not the end of the researcher's chat
            from myrmidon.core.observation import Observation

            obs = Observation(ok=False, failure="raised",
                              summary=f"{name} raised {type(exc).__name__}: {exc}")
        call_s = time.monotonic() - call_started
        episode.n_tool_calls += 1
        dump = obs.model_dump(mode="json", exclude_none=True)
        self.index.add_tree(citable(dump),
                            source="script" if name in SCRIPT_TOOLS else self._source)
        if obs.ok:  # a choice the registry accepted, not free text
            self.index.add_parameters(args)
        result_text = json.dumps(model_view(obs))
        result_chars = len(result_text)
        if result_chars > _MAX_TOOL_RESULT_CHARS:
            result_text = result_text[:_MAX_TOOL_RESULT_CHARS] + "…[truncated]"
        transcript.emit(
            "tool_result",
            tool=name,
            ok=obs.ok,
            provenance_id=obs.provenance_id,
            obs=dump,  # full observation: graders inspect values here
            turn=turn,
            wall_s=round(call_s, 3),
            # since the episode started: time from the question to this result
            elapsed_s=round(time.monotonic() - started, 3),
            result_chars=result_chars,  # before truncation
            truncated=result_chars > _MAX_TOOL_RESULT_CHARS,
            # why the registry did not run it; None: it ran
            failure=obs.failure,
            # what kind of result this is, for the terminal to decide what a researcher
            # sees without reading the summary's prose (agent/chat.py `classify`)
            observation_type=type(obs).__name__,
            tags=sorted(self.registry.get(name).tags),
        )
        return obs, result_text

    def _researcher_decides(self, name, args, obs, result_text, turn, started, episode):
        """A finding only a person may accept, and a person at the keyboard: ask. A yes is
        an approval of exactly this call (core/approvals.py), recorded before the call is
        made again with the finding acknowledged; a no is told to the model as such."""
        from myrmidon.core.approvals import grant

        ids = [w.id for w in obs.human_required]
        identity = self.registry.call_identity(name, args, self.ctx)
        if identity in self._declined:
            # the same call again: the researcher has answered it, so the model is told,
            # not the researcher asked again (a model may resend a declined call unchanged)
            self.transcript.emit("researcher_decision", tool=name, findings=ids,
                                 approved=False, repeated=True, turn=turn)
            note = (f"THE RESEARCHER ALREADY DECLINED THIS EXACT CALL ({ids}); they were not "
                    "asked again. Sent unchanged it cannot run: change the parameters as "
                    "they said, or stop and say what you need from them. ")
            return obs, note + result_text
        yes = bool(self.on_decision(name, args, obs.human_required))
        if not yes:
            self._declined.add(identity)
            self.transcript.emit("researcher_decision", tool=name, findings=ids,
                                 approved=False, turn=turn)
            note = (f"THE RESEARCHER DECLINED {ids} for this call. Do not repeat it; "
                    "say what was declined and what it would have cost. ")
            return obs, note + result_text
        approval = grant(self.ctx.workspace, ids, approver="researcher", call=identity)
        self.transcript.emit("researcher_decision", tool=name, findings=ids, approved=True,
                             approval=approval, turn=turn)
        acked = sorted(set(args.get("acknowledge_warnings") or []) | set(ids))
        return self._execute(name, {**args, "acknowledge_warnings": acked}, turn, started,
                             episode)

    def _close_open_calls(self) -> None:
        """After Ctrl-C: answer every tool call of the last model turn that has no result,
        without claiming whether it ran, and close the turn with a note that the question
        was dropped — or the model answers it along with the next one."""
        last = next((i for i in range(len(self.messages) - 1, -1, -1)
                     if self.messages[i].get("role") == "assistant"), None)
        if last is not None and self.messages[last].get("tool_calls"):
            answered = {m.get("tool_call_id") for m in self.messages[last + 1:]
                        if m.get("role") == "tool"}
            for tc in self.messages[last]["tool_calls"]:
                if tc["id"] not in answered:
                    self.messages.append({"role": "tool", "tool_call_id": tc["id"],
                        "content": json.dumps({"ok": False, "summary": "interrupted by the "
                            "researcher before a result came back: the call may or may "
                            "not have run — check the study (study_open) or job_status "
                            "before repeating it"})})
        self.messages.append({"role": "assistant", "content": "[Stopped by the researcher "
                              "before I finished. That request is dropped unless they ask "
                              "again.]"})

def rebuild_index(events: list[dict], source: str) -> NumericIndex:
    """The grounding index a session's transcript implies: its prompts, what its tools
    returned and the parameters the registry accepted — what a resumed session may cite."""
    index = NumericIndex()
    calls: dict[tuple, dict] = {}
    for ev in events:
        kind = ev.get("kind")
        if kind == "user":
            index.add_tree(ev.get("content") or "", source="prompt")
        elif kind == "tool_call" and not ev.get("error"):
            calls[(ev.get("turn"), ev.get("tool"))] = ev.get("args") or {}
        elif kind == "tool_result":
            name = ev.get("tool")
            index.add_tree(citable(ev.get("obs") or {}),
                           source="script" if name in SCRIPT_TOOLS else source)
            if ev.get("ok"):
                index.add_parameters(calls.get((ev.get("turn"), name), {}))
    return index
