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


_LIE = FakeMessage(content="The binding free energy is -42.7 kcal/mol.")

# (what the model answers after one doctor call; the answer stands grounded; the numbers
# each bounce sent back). An ungrounded number is bounced once with the number named; a
# model that keeps it ends the episode ungrounded, which is what a grader reads.
EPISODES = [
    ([FakeMessage(content="One engine is available; the backend is local.")], True, []),
    ([FakeMessage(content="The RMSD converged to 0.4527 nm."),
      FakeMessage(content="No RMSD was computed; only engine discovery ran.")],
     True, [["0.4527"]]),
    ([_LIE, _LIE, _LIE], False, None),
]


@pytest.mark.parametrize("answers,grounded,bounced", EPISODES)
def test_an_answer_with_a_number_no_tool_gave_is_bounced(ctx, tmp_path, answers, grounded,
                                                        bounced):
    fn = scripted(FakeMessage(tool_calls=[FakeToolCall("doctor", {})]), *answers)
    loop = AgentLoop("fake", ctx, transcript_dir=tmp_path / "eps", completion_fn=fn)
    episode = loop.run("Report results.")
    assert episode.stop_reason is StopReason.COMPLETED and episode.n_tool_calls == 1
    assert episode.grounding.ok is grounded
    if bounced is not None:
        assert [e["ungrounded"] for e in _events(episode)
                if e["kind"] == "grounding_bounce"] == bounced
    else:
        assert "-42.7" in episode.grounding.ungrounded


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
