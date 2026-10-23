"""Every request to the model is in the transcript, and rebuilds exactly."""

import copy
import json
from types import SimpleNamespace

import pytest

from myrmidon.agent.loop import AgentLoop
from myrmidon.agent.pydantic_adapter import _Function, _Message, _ToolCall
from myrmidon.agent.session import ChatSession, build_loop, load, new_state
from myrmidon.agent.transcript import TraceError, model_calls


class Recorder:
    """A model that keeps an exact copy of every request and answers from a script."""

    def __init__(self, replies, seen=None):
        self.replies = list(replies)
        self.seen = [] if seen is None else seen

    def __call__(self, model, messages, tools, **kw):
        settings = {k: v for k, v in kw.items() if k not in ("api_base", "api_key")}
        self.seen.append({"model": model, "messages": copy.deepcopy(messages),
                          "tools": copy.deepcopy(tools), "settings": settings})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


_ids = iter(range(1000))


def reply(content=None, calls=(), reasoning=None):
    msg = _Message(content=content, reasoning=reasoning, tool_calls=[
        _ToolCall(id=f"call_{next(_ids)}", function=_Function(name=n, arguments=a))
        for n, a in calls] or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if calls else "stop")],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
        provider_finish_reason="tool_calls" if calls else "stop")


def _sent(seen):
    """What the model was given, through JSON as the transcript holds it."""
    return json.loads(json.dumps(seen, sort_keys=True))


def _rebuilt(path):
    calls = model_calls([json.loads(line) for line in path.read_text().splitlines()])
    return calls, _sent([{k: c[k] for k in ("model", "messages", "tools", "settings")}
                         for c in calls])


def test_a_saved_and_resumed_session_rebuilds_call_for_call(ws_ctx, tmp_path):
    root = tmp_path / "sessions"
    seen: list = []
    first = Recorder([
        None,  # the opening's reply, filled in below once the skill names are known
        reply("Checking the engines first.", reasoning="The user wants engines.", calls=[
            ("doctor", "{}"), ("doctor", "{not json"), ("no_such_tool", "{}")]),
        reply("The temperature was 412.7 K."),  # nothing measured it: bounced
        reply("The engines are listed above.", reasoning="Answer without numbers."),
    ], seen)
    loop = build_loop(ws_ctx, "fake-model", completion_fn=first,
                      sampling={"temperature": 0.0, "max_tokens": 512})
    skill = loop.opening["function"]["parameters"]["properties"]["skill"]["enum"][0]
    first.replies[0] = reply(calls=[("skill_lookup", json.dumps({"skill": skill}))])
    state = new_state("fake-model", "pydantic", None)
    session = ChatSession.begin(loop, state, root, "You are a test agent.")
    episode = session.ask("Which engines are there?")
    assert episode.stop_reason == "COMPLETED"
    session.checkpoint()

    second = Recorder([reply("Nothing has changed."), RuntimeError("endpoint down")], seen)
    loop2 = build_loop(ws_ctx, "fake-model", completion_fn=second,
                       sampling={"temperature": 0.0, "max_tokens": 512})
    resumed = ChatSession.resume(loop2, load(root, state.session_id), root)
    resumed.ask("And now?")
    assert resumed.ask("Once more?").stop_reason == "MODEL_ERROR"

    calls, rebuilt = _rebuilt(resumed.transcript_path)
    assert rebuilt == _sent(seen), "the transcript must rebuild every request exactly"
    assert len(calls) == 6
    # the opening was a forced choice over one tool, and its setting is recorded
    assert seen[0]["settings"]["tool_choice"] == "required"
    assert [t["function"]["name"] for t in calls[0]["tools"]] == ["skill_lookup"]
    # reasoning is kept with the reply, and never sent back
    assert calls[1]["response"]["reasoning"] == "The user wants engines."
    assert "reasoning" not in json.dumps(seen[2]["messages"])
    # the texts the loop sent on its own reached the model and the trace alike
    sent_back = json.dumps(calls[2]["messages"])
    assert "unparsable arguments" in sent_back and "unknown tool" in sent_back
    assert "412.7" in json.dumps(calls[3]["messages"][-1])  # the bounce, after the answer
    # the answer given before the session was left is in the request after the resume
    assert calls[4]["messages"][-2]["content"] == "The engines are listed above."
    assert calls[5]["error"] == "endpoint down"


def test_the_trace_is_deltas_not_copies(ws_ctx, tmp_path):
    """Linear in the number of calls: each message is written once."""
    loop = AgentLoop("m", ws_ctx, transcript_dir=tmp_path, completion_fn=Recorder([
        reply(calls=[("doctor", "{}")]), reply(calls=[("doctor", "{}")]),
        reply("Done.")]))
    loop.run("q")
    events = loop.transcript.events()
    appended = sum(len(e["append"]) for e in events if e["kind"] == "model_call")
    assert appended == len(loop.messages) - 1  # all but the final answer, sent to no one
    assert sum(e["kind"] == "tool_schemas" for e in events) == 1


def test_a_restored_history_that_differs_is_written_whole(ws_ctx, tmp_path):
    root = tmp_path / "sessions"
    seen: list = []
    loop = AgentLoop("m", ws_ctx, transcript_dir=tmp_path,
                     completion_fn=Recorder([reply("First.")], seen))
    state = new_state("m", "pydantic", None)
    session = ChatSession.begin(loop, state, root, "sys")
    session.ask("one")
    session.checkpoint()
    state = load(root, state.session_id)
    state.messages[0]["content"] = "a system prompt edited by hand"
    loop2 = AgentLoop("m", ws_ctx, transcript_dir=tmp_path,
                      completion_fn=Recorder([reply("Second.")], seen))
    resumed = ChatSession.resume(loop2, state, root)
    resumed.ask("two")
    _, rebuilt = _rebuilt(resumed.transcript_path)
    assert rebuilt == _sent(seen)
    assert rebuilt[1]["messages"][0]["content"] == "a system prompt edited by hand"


def test_a_trace_that_does_not_add_up_is_refused(ws_ctx, tmp_path):
    loop = AgentLoop("m", ws_ctx, transcript_dir=tmp_path,
                     completion_fn=Recorder([reply("Done.")]))
    loop.run("q")
    events = [e for e in loop.transcript.events() if e["kind"] != "tool_schemas"]
    with pytest.raises(TraceError, match="never wrote"):
        model_calls(events)


def test_the_key_never_reaches_the_transcript(ws_ctx, tmp_path):
    key = "sk-test-0123456789"
    loop = AgentLoop("m", ws_ctx, transcript_dir=tmp_path,
                     connection={"api_base": "https://x.invalid/v1", "api_key": key},
                     completion_fn=Recorder([RuntimeError(f"401: bad key {key}")]))
    loop.run("q")
    text = loop.transcript.path.read_text()
    assert key not in text and "x.invalid" not in text
    (call,) = model_calls(loop.transcript.events())
    assert call["error"] == "401: bad key [api_key]"


def test_the_trace_says_what_served_it(ws_ctx, tmp_path):
    loop = AgentLoop("m", ws_ctx, transcript_dir=tmp_path, endpoint="site-a",
                     completion_fn=Recorder([reply("Done.")]))
    loop.run("q")
    (trace,) = [e for e in loop.transcript.events() if e["kind"] == "trace"]
    assert trace["format"] == 1 and trace["endpoint"] == "site-a"
    assert trace["adapter"] == "Recorder" and trace["versions"]["myrmidon"]
