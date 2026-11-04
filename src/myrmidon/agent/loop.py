"""Model-agnostic agent loop: the native way to drive the registry.

Same tools, same gates, same provenance as the MCP path — plus hard stops
(max turns, token budget, wall clock) and a full episode transcript.
"""

from __future__ import annotations

import hashlib
import json
import time
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.agent.prompts import default_system_prompt
from myrmidon.agent.transcript import TRANSCRIPT_FORMAT, Transcript, digest, trace_history
from myrmidon.core.observation import model_view
from myrmidon.core.tool import REGISTRY, ToolContext, ToolRegistry
from myrmidon.errors import UnknownToolError

_MAX_TOOL_RESULT_CHARS = 8000
#: tool results this many model calls old are sent as their outcome and summary line: the
#: whole stays in the history and the transcript, and the ledger has every number. The
#: boundary moves in steps of this many calls, so the history a provider caches as a
#: prefix changes once a step rather than on every call
STUB_AFTER_CALLS = 10


def brief(ok, summary: str) -> str:
    """A tool result as it is sent once it is old."""
    line = str(summary or "").splitlines()[0][:200] if summary else ""
    return json.dumps({"ok": ok, "summary": line,
                       "stubbed": "an earlier result in brief; the ledger and the study's "
                       "files have the rest"})


def stubbed(messages: list[dict], briefs: dict[int, tuple[int, str]], call: int
            ) -> tuple[list[dict], int]:
    """The history as sent at model call ``call``: each tool result in ``briefs`` (its
    index -> the call it answered and its brief) from before the current step's boundary
    replaced by its brief. Returns it and how many were."""
    boundary = (call // STUB_AFTER_CALLS - 1) * STUB_AFTER_CALLS
    out, n = [], 0
    for i, msg in enumerate(messages):
        made, short = briefs.get(i, (call, ""))
        if made >= boundary:
            out.append(msg)
            continue
        out.append({**msg, "content": short})
        n += 1
    return out, n
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


def _resumed_brief(content) -> str:
    try:
        whole = json.loads(content or "{}")
    except ValueError:  # a result cut at the size cap
        return brief(None, str(content)[:200])
    if not isinstance(whole, dict):
        return brief(None, str(content)[:200])
    return brief(whole.get("ok"), whole.get("summary", ""))


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
        unattended: bool = True,
        sampling: dict | None = None,
        connection: dict | None = None,
        endpoint: str | None = None,
        on_event=None,
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
        # the model makes these calls; nobody sees them as they happen
        ctx.model_driven = True
        ctx.unattended = unattended
        self.registry = registry
        self.max_turns = max_turns
        self.token_budget = token_budget
        self.wall_clock_s = wall_clock_s
        self.transcript_dir = transcript_dir
        if completion_fn is None:
            from myrmidon.agent.pydantic_adapter import PydanticCompletion

            completion_fn = PydanticCompletion()
        self.completion_fn = completion_fn
        # sees every transcript event as it is written, plus "thinking" before each model
        # call (never written): what a terminal renders. Presentation only.
        self.on_event = on_event
        self.messages: list[dict] = []
        # index in self.messages of each tool result -> the model call it answered, and
        # the result in brief
        self._briefs: dict[int, tuple[int, str]] = {}
        self.transcript: Transcript | None = None
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

    def run(self, task_prompt: str, system_prompt: str | None = None) -> Episode:
        """One question, one episode: what `myrmidon run` asks."""
        self.start(system_prompt)
        return self.ask(task_prompt)

    def start(self, system_prompt: str | None = None, transcript: Transcript | None = None,
              messages: list[dict] | None = None) -> None:
        """Begin a session, or — given the transcript and messages of one — resume it."""
        self.transcript = transcript or Transcript(self.transcript_dir)
        self.transcript.listener = self.on_event
        if messages is not None:
            self.messages = list(messages)
            events = self.transcript.events()
            self._schemas = tool_schemas(self.registry, self.ctx.engines)
            self._schema_hashes = {e["sha256"] for e in events
                                   if e.get("kind") == "tool_schemas"}
            self._calls = sum(1 for e in events if e.get("kind") == "model_call")
            # a resumed session's results age from here
            self._briefs = {i: (self._calls, _resumed_brief(m["content"]))
                            for i, m in enumerate(self.messages) if m.get("role") == "tool"}
            self._emit_trace(resumed=True)
            self._record_resumed(events)
            return
        self._sent, self._schema_hashes, self._calls, self._briefs = 0, set(), 0, {}
        self._emit_trace(resumed=False)
        system = system_prompt or default_system_prompt()
        self.messages = [{"role": "system", "content": system}]
        self.transcript.emit("system", content=system)
        self._schemas = None
        self._opening_due = self.opening is not None

    def ask(self, prompt: str) -> Episode:
        """The researcher's next message, worked until an answer or a stop."""
        transcript = self.transcript
        self.messages.append({"role": "user", "content": prompt})
        transcript.emit("user", content=prompt)
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
        transcript.emit("episode_end", stop_reason=episode.stop_reason.value,
                        n_turns=episode.n_turns)
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
        transcript, messages = self.transcript, self.messages

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

            history_chars = len(json.dumps(messages, default=str))  # the whole history
            self._notify("thinking", turn=episode.n_turns + 1)
            opening, self._opening_due = self._opening_due, False
            tools, forced = ([self.opening], {"tool_choice": "required"}) if opening \
                else (self._schemas, {})
            request = dict(append=messages[self._sent:], tools=self._schema_ref(tools),
                           settings={**self.sampling, **forced})
            sent, n_stubbed = stubbed(messages, self._briefs, self._calls + 1)
            if n_stubbed:
                request["stubbed"] = n_stubbed
            waits = list(self.retry_waits)
            while True:
                called = time.monotonic()
                try:
                    response = self.completion_fn(
                        model=self.model, messages=sent, tools=tools,
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
                    self._briefs[len(messages)] = (self._calls, brief(obs.ok, obs.summary))
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                    )
                transcript.emit("turn", **turn_record,
                                tools_s=round(time.monotonic() - tools_started, 3))
                continue

            # No tool calls: the answer. It stays in the history, so the researcher's next
            # message follows it
            transcript.emit("turn", **turn_record)
            answer = msg.content or ""
            transcript.emit("assistant", content=answer)
            messages.append(_dump_message(msg))
            episode.final_answer = answer
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
            tags=sorted(self.registry.get(name).tags),
        )
        return obs, result_text

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
