"""Agent loop mechanics with a scripted fake model — no network, no LLM."""

import json

import pytest

from myrmidon.agent.loop import AgentLoop, StopReason


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, args, call_id="call_1"):
        self.id = call_id
        self.function = FakeFunction(name, json.dumps(args))


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, **kw):
        out = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return out


class FakeUsage:
    prompt_tokens = 100
    completion_tokens = 50


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]
        self.usage = FakeUsage()


def scripted(*messages):
    """A completion_fn yielding the given messages in order."""
    queue = list(messages)

    def fn(model, messages, tools, **kw):
        return FakeResponse(queue.pop(0))

    return fn


def test_the_first_reply_without_tool_calls_is_the_answer(ctx, tmp_path):
    """The model's answer stands as it gave it: the harness records what the tools
    returned and never sends a reply back over the numbers it states."""
    answer = "The RMSD converged to 0.4527 nm."
    fn = scripted(FakeMessage(tool_calls=[FakeToolCall("doctor", {})]),
                  FakeMessage(content=answer),
                  FakeMessage(content="never asked for"))
    loop = AgentLoop("fake", ctx, transcript_dir=tmp_path / "eps", completion_fn=fn)
    episode = loop.run("Report results.")
    assert episode.stop_reason is StopReason.COMPLETED and episode.n_tool_calls == 1
    assert episode.final_answer == answer and episode.n_turns == 2
    assert [e["kind"] for e in _events(episode)][-3:] == ["turn", "assistant", "episode_end"]


def _events(episode):
    from pathlib import Path

    from myrmidon.agent.transcript import Transcript

    return Transcript(
        Path(episode.transcript_path).parent, episode_id=episode.episode_id
    ).events()



class _Status(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


# (what the endpoint raises before it answers; how the question ends; requests made). A
# timeout or a server error is asked again; a refusal the request earned is not; the
# retries are bounded
RETRIES = [
    ([TimeoutError("no reply from the model within 180.0 s")], StopReason.COMPLETED, 2),
    ([_Status(503), ConnectionError("reset")], StopReason.COMPLETED, 3),
    ([_Status(400)], StopReason.MODEL_ERROR, 1),  # "credit balance is too low"
    ([TimeoutError("t")] * 3, StopReason.MODEL_ERROR, 3),
]


@pytest.mark.parametrize("failures,stop,requests", RETRIES)
def test_a_transient_model_failure_is_retried_and_recorded(ctx, tmp_path, failures, stop,
                                                           requests):
    queue = [*failures, FakeMessage(content="One engine is available.")]
    made = []

    def fn(model, messages, tools, **kw):
        made.append(len(messages))
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return FakeResponse(item)

    loop = AgentLoop("fake", ctx, transcript_dir=tmp_path / "eps", completion_fn=fn)
    episode = loop.run("Which engines are there?")
    assert episode.stop_reason is stop and len(made) == requests
    calls = [e for e in _events(episode) if e["kind"] == "model_call"]
    assert ["error" in c for c in calls] == [True] * min(requests, len(failures)) \
        + [False] * (requests - len(failures))
    assert all(c["append"] == [] for c in calls[1:]), "a retry re-sends, it adds nothing"


@pytest.mark.parametrize("call,stubbed_now", [(19, False), (20, True)])
def test_an_old_tool_result_is_sent_in_brief(call, stubbed_now):
    """The history keeps every result whole; what is sent replaces one from before the
    current ten-call step by its outcome and summary line — read from the observation, so
    a result cut at the size cap keeps them too."""
    import json

    from myrmidon.agent.loop import brief, stubbed

    whole = json.dumps({"ok": True, "summary": "Wrote nvt.mdp", "data": "x" * 9000})[:8000]
    messages = [{"role": "user", "content": "q"},
                {"role": "tool", "tool_call_id": "t1", "content": whole}]
    sent, n = stubbed(messages, {1: (1, brief(True, "Wrote nvt.mdp\nand more"))}, call)
    assert (n == 1) is stubbed_now and messages[1]["content"] == whole
    if stubbed_now:
        shown = json.loads(sent[1]["content"])
        assert (shown["ok"], shown["summary"]) == (True, "Wrote nvt.mdp")
    assert sent[0] == messages[0] and sent[1]["tool_call_id"] == "t1"
