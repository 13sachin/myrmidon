"""PydanticAI as the native loop's model adapter: messages in, one model call, a reply out.

`AgentLoop` owns the episode — limits, the transcript, tool calls through the registry,
the grounding bounce — and calls its ``completion_fn`` once per turn with OpenAI-shaped
messages and tool schemas. ``PydanticCompletion`` is that function: it rebuilds the
history as PydanticAI messages, makes the request through ``pydantic_ai.direct``, and
answers in the shape the loop reads (``choices[0].message``, ``finish_reason``,
``usage``). The loop is unaware of the framework: an episode is graded, measured and
replayed the same whatever answers it.

No domain logic lives here: no tool is executed, nothing is validated, no number is
judged. A named endpoint (``endpoints.py``) becomes PydanticAI's OpenAI-compatible
provider; without one, the model string is PydanticAI's own (``anthropic:claude-sonnet-5-5``,
``openai:gpt-4o``; ``openai/gpt-4o`` is accepted).
"""

from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

#: how long one model request may take before it is abandoned; without it a hung request
#: holds a session for the OpenAI client's own default (ten minutes, then retries).
#: Long enough for a reasoning model to write a full 8192-token reply.
CALL_TIMEOUT_S = 180.0
#: the OpenAI client's own retries after a failed request (its default is 2): one rides out
#: a dropped connection; more only multiply a hang
CALL_RETRIES = 1

#: PydanticAI's normalized finish reasons, in the OpenAI words the transcript records
_FINISH = {"tool_call": "tool_calls"}
#: sampling keys PydanticAI's ModelSettings takes; anything else is refused, not dropped
_SETTINGS = {"max_tokens", "temperature", "top_p", "seed", "timeout", "presence_penalty",
             "frequency_penalty", "stop_sequences", "parallel_tool_calls"}


class _Function(BaseModel):
    name: str
    arguments: str


class _ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: _Function


class _Message(BaseModel):
    """The assistant message as the loop reads it and puts back into its history."""

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None
    # what a reasoning model wrote before answering: recorded, never sent back
    reasoning: str | None = None


def pydantic_model_name(model: str) -> str:
    """PydanticAI's ``provider:name`` for a model given either way."""
    if ":" not in model and model.startswith("openai/"):
        return "openai:" + model.split("/", 1)[1]
    return model


def to_pydantic_messages(messages: list[dict]) -> list:
    """OpenAI-shaped history -> PydanticAI messages. Consecutive system, user and tool
    messages form one request, as PydanticAI expects; a tool result is matched to its
    call's tool name by id."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    out: list = []
    pending: list = []
    names: dict[str, str] = {}

    def flush():
        if pending:
            out.append(ModelRequest(parts=list(pending)))
            pending.clear()

    for msg in messages:
        role = msg["role"]
        if role == "system":
            pending.append(SystemPromptPart(content=msg["content"]))
        elif role == "user":
            pending.append(UserPromptPart(content=msg["content"]))
        elif role == "tool":
            call_id = msg["tool_call_id"]
            pending.append(ToolReturnPart(tool_name=names.get(call_id, ""),
                                          content=msg["content"], tool_call_id=call_id))
        elif role == "assistant":
            flush()
            parts: list = []
            if msg.get("content"):
                parts.append(TextPart(content=msg["content"]))
            for tc in msg.get("tool_calls") or []:
                fn = tc["function"]
                names[tc["id"]] = fn["name"]
                parts.append(ToolCallPart(tool_name=fn["name"], args=fn["arguments"],
                                          tool_call_id=tc["id"]))
            out.append(ModelResponse(parts=parts or [TextPart(content="")]))
        else:
            raise ValueError(f"unknown message role {role!r}")
    flush()
    return out


def tool_definitions(tools: list[dict]) -> list:
    """The loop's OpenAI-style tool schemas as PydanticAI tool definitions."""
    from pydantic_ai.tools import ToolDefinition

    return [ToolDefinition(name=t["function"]["name"],
                           description=t["function"].get("description"),
                           parameters_json_schema=t["function"]["parameters"])
            for t in tools]


def from_pydantic_response(response) -> SimpleNamespace:
    """A PydanticAI ModelResponse in the shape ``AgentLoop`` reads."""
    from pydantic_ai.messages import TextPart, ThinkingPart, ToolCallPart

    text = "".join(p.content for p in response.parts if isinstance(p, TextPart))
    reasoning = "".join(p.content for p in response.parts if isinstance(p, ThinkingPart))
    calls = [_ToolCall(id=p.tool_call_id, function=_Function(
                 name=p.tool_name, arguments=p.args_as_json_str()))
             for p in response.parts if isinstance(p, ToolCallPart)]
    message = _Message(content=text or None, tool_calls=calls or None,
                       reasoning=reasoning or None)
    finish = response.finish_reason
    usage = response.usage
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=_FINISH.get(finish, finish))],
        # the server's own word for it, before PydanticAI normalised it
        provider_finish_reason=(response.provider_details or {}).get("finish_reason"),
        usage=SimpleNamespace(prompt_tokens=usage.input_tokens,
                              completion_tokens=usage.output_tokens),
    )


class PydanticCompletion:
    """``completion_fn`` for ``AgentLoop``, backed by PydanticAI.

    ``model`` overrides the model the loop names (tests pass a ``FunctionModel``). The
    endpoint's ``api_base``/``api_key`` arrive with every call; the
    model object is built once for them and reused. Every request carries
    ``call_timeout_s``; one that runs out raises ``TimeoutError``, which the loop records
    as a model error (`run` exits 3, a chat question stops).
    """

    def __init__(self, model=None, call_timeout_s: float | None = CALL_TIMEOUT_S):
        self._model = model
        self.call_timeout_s = call_timeout_s
        self._built: dict[tuple, object] = {}
        # models whose server refuses a required tool call ("only 'auto' tool choice is
        # supported"): asked with auto choice from then on
        self._auto_only: set[tuple] = set()

    def _resolve(self, name: str, api_base: str | None, api_key: str | None):
        if self._model is not None:
            return self._model
        if api_base is None:
            return pydantic_model_name(name)
        key = (name, api_base)
        if key not in self._built:
            from openai import AsyncOpenAI
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            from myrmidon.endpoints import served_model_id

            client = AsyncOpenAI(base_url=api_base, api_key=api_key, max_retries=CALL_RETRIES)
            self._built[key] = OpenAIChatModel(
                served_model_id(name), provider=OpenAIProvider(openai_client=client))
        return self._built[key]

    def __call__(self, *, model: str, messages: list[dict], tools: list[dict],
                 api_base: str | None = None, api_key: str | None = None, **sampling):
        from pydantic_ai.direct import model_request_sync
        from pydantic_ai.models import ModelRequestParameters

        # "required": the reply must be a tool call (the loop's opening turn) — where the
        # server allows it to be required; otherwise the turn's single tool is the only ask
        must_call = sampling.pop("tool_choice", None) == "required" \
            and (model, api_base) not in self._auto_only
        unknown = set(sampling) - _SETTINGS
        if unknown:
            raise ValueError(f"sampling parameters PydanticAI does not take: {sorted(unknown)}")
        settings = dict(sampling)
        if self.call_timeout_s and "timeout" not in settings:
            settings["timeout"] = self.call_timeout_s
        def request(required: bool):
            return model_request_sync(
                self._resolve(model, api_base, api_key),
                to_pydantic_messages(messages),
                model_settings=settings or None,
                model_request_parameters=ModelRequestParameters(
                    function_tools=tool_definitions(tools), allow_text_output=not required),
            )

        try:
            try:
                response = request(must_call)
            except Exception as exc:
                if not must_call or _is_timeout(exc):
                    raise
                # the server would not require a call: ask again as it allows, and remember
                self._auto_only.add((model, api_base))
                response = request(False)
        except Exception as exc:
            if _is_timeout(exc):
                raise TimeoutError(f"no reply from the model within "
                                   f"{settings.get('timeout')} s") from exc
            raise
        return from_pydantic_response(response)


def _is_timeout(exc: BaseException) -> bool:
    """A request that ran out of time, however the client layers report it."""
    import httpx

    seen = exc
    while seen is not None:
        if isinstance(seen, (TimeoutError, httpx.TimeoutException)) or \
                type(seen).__name__ == "APITimeoutError":
            return True
        seen = seen.__cause__ or seen.__context__
    return False

