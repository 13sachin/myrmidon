"""docs/FORMATS.md describes what the harness writes, all of it and nothing else."""

import ast
import json
import re
from pathlib import Path

from myrmidon.agent.transcript import TRANSCRIPT_FORMAT
from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.provenance import ProvenanceEvent
from myrmidon.core.workspace import FORMAT, META_FILENAME, Workspace

ROOT = Path(__file__).resolve().parents[2]
DOC = (ROOT / "docs" / "FORMATS.md").read_text()
SRC = ROOT / "src" / "myrmidon"


def _section(title: str) -> str:
    start = DOC.index(f"## {title}")
    end = DOC.find("\n## ", start + 1)
    return DOC[start:end if end != -1 else len(DOC)]


def documented_events() -> dict[str, set[str]]:
    """Each `#### \\`kind\\`` heading and the fields its "Fields:" line names."""
    events: dict[str, set[str]] = {}
    kind = None
    for line in _section("Transcript").splitlines():
        if m := re.match(r"#### `(\w+)`", line):
            kind = m.group(1)
            events[kind] = set()
        elif kind and line.startswith("Fields:"):
            events[kind] = set(re.findall(r"`(\w+)`", line))
    return events


def emitted_kinds() -> set[str]:
    """Every event kind the shipped harness writes: the first argument of each
    `.emit("kind", ...)` call in the package."""
    kinds = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "emit" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                kinds.add(node.args[0].value)
    return kinds


def test_every_event_written_is_documented_and_nothing_else():
    documented, written = set(documented_events()), emitted_kinds()
    assert written - documented == set(), "written but not in docs/FORMATS.md"
    assert documented - written == set(), "documented but never written"


def test_every_documented_event_names_its_fields():
    assert all(documented_events().values()), "an event without a Fields: line"


def test_the_versions_are_the_codes():
    assert f"(`format`, **{FORMAT}**)" in DOC
    assert f"(`format`, **{TRANSCRIPT_FORMAT}**)" in DOC
    assert f"## Study (workspace format {FORMAT})" in DOC
    assert f"## Transcript (format {TRANSCRIPT_FORMAT})" in DOC


def test_every_ledger_field_is_listed():
    ledger = _section("Ledger")
    rows = set(re.findall(r"^\| `(\w+)` \|", ledger, flags=re.M))
    assert rows == set(ProvenanceEvent.model_fields)
    artifact = ledger[ledger.index("An artifact is"):].split("\n\n")[0]
    assert set(ArtifactRef.model_fields) <= set(re.findall(r"`(\w+)`", artifact))


def test_the_manifest_fields_are_the_ones_written(tmp_path):
    ws = Workspace.create(tmp_path / "s", "s", "d", origin=tmp_path)
    written = set(json.loads((ws.root / META_FILENAME).read_text()))
    line = next(x for x in _section("Study").splitlines() if x.startswith("The manifest is"))
    assert set(re.findall(r"`(\w+)`", line.split("Fields:")[1])) == written


def test_a_session_writes_only_documented_fields(ws_ctx, tmp_path):
    """A chat session through every common event, saved, resumed and ended by a model
    error: each record's keys are the ones its event documents."""
    from tests.unit.test_model_trace import Recorder, reply

    from myrmidon.agent.session import ChatSession, build_loop, load, new_state

    root = tmp_path / "sessions"
    first = Recorder([None, reply("Looking.", calls=[
        ("doctor", "{}"), ("doctor", "{bad"), ("nope", "{}")]),
        reply("It is 412.7 K."), reply("Done.")])
    loop = build_loop(ws_ctx, "m", completion_fn=first, sampling={"temperature": 0.0})
    skill = loop.opening["function"]["parameters"]["properties"]["skill"]["enum"][0]
    first.replies[0] = reply(calls=[("skill_lookup", json.dumps({"skill": skill}))])
    state = new_state("m", "pydantic", None)
    session = ChatSession.begin(loop, state, root, "sys")
    session.ask("q")
    session.checkpoint()
    loop2 = build_loop(ws_ctx, "m", completion_fn=Recorder([RuntimeError("down")]))
    resumed = ChatSession.resume(loop2, load(root, state.session_id), root)
    resumed.ask("again")

    documented = documented_events()
    seen = set()
    for line in resumed.transcript_path.read_text().splitlines():
        event = json.loads(line)
        kind = event.pop("kind")
        event.pop("ts")
        seen.add(kind)
        assert set(event) <= documented[kind], f"{kind}: {set(event) - documented[kind]}"
    # the walk reached the events most sessions write
    assert {"trace", "tool_schemas", "model_call", "resumed", "turn", "tool_call",
            "tool_result", "grounding_bounce", "error", "episode_end"} <= seen
