"""A native agent session: one conversation with a researcher, kept so it can be resumed.

A session is the loop's history, the transcript it wrote and the study it was working in —
nothing more. It never claims that a tool ran or that a job finished: on resume, the ledger
and the job store are asked again (the study is reopened, and the model reads job state
through the registry like any other call). Sessions live under the studies root, in
``.sessions/<id>/`` (``state.json`` and the transcript ``<id>.jsonl``), because a session
may move between studies and may begin before any study exists.

``state.json`` is replaced atomically after every answer and on exit; the transcript is
append-only.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.agent.loop import AgentLoop, Episode
from myrmidon.agent.transcript import Transcript
from myrmidon.basic_tools import SESSIONS_DIRNAME as SESSIONS_DIR

STATE_FILE = "state.json"
#: limits per question: a chat waits on jobs, so it gets longer than `run`'s defaults
CHAT_LIMITS = {"max_turns": 60, "token_budget": 3_000_000, "wall_clock_s": 4 * 3600.0}


class SessionError(ValueError):
    """A session that does not exist, or a state file this version cannot read."""


class SessionState(BaseModel):
    schema_version: int = 1
    session_id: str
    created: str
    updated: str
    model: str
    runner: str  # the model adapter: "pydantic"
    endpoint: str | None = None  # a named endpoint (endpoints.py); never a key
    workspace: str | None = None  # the study it was last working in
    origin: str = ""  # the directory it was started from
    asks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_stop: str | None = None
    last_question: str = ""
    messages: list[dict] = Field(default_factory=list)


def sessions_root(studies_root: Path) -> Path:
    root = Path(studies_root).expanduser()
    return (root if root.is_absolute() else Path.cwd() / root) / SESSIONS_DIR


def run_transcripts(ctx) -> Path:
    """Where `myrmidon run` writes its transcript: in the study it starts in, beside the
    job store (``.myrmidon/transcripts/``), so the record stays with what it did; with no
    study open yet, under the studies root beside the chat sessions (``.sessions/runs/``).
    Never the working directory."""
    from myrmidon.core.workspace import STATE_DIRNAME

    if ctx.workspace is not None:
        return ctx.workspace.root / STATE_DIRNAME / "transcripts"
    return sessions_root(ctx.config.studies_root) / "runs"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_state(model: str, runner: str, endpoint: str | None) -> SessionState:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    return SessionState(session_id=f"s-{stamp}-{uuid.uuid4().hex[:4]}", created=_now(),
                        updated=_now(), model=model, runner=runner, endpoint=endpoint,
                        origin=str(Path.cwd()))


def save(root: Path, state: SessionState) -> Path:
    """Write ``state.json`` atomically: a crash leaves the old state or the new one."""
    directory = root / state.session_id
    directory.mkdir(parents=True, exist_ok=True)
    state.updated = _now()
    tmp = directory / (STATE_FILE + ".tmp")
    tmp.write_text(state.model_dump_json(indent=1))
    os.replace(tmp, directory / STATE_FILE)
    return directory / STATE_FILE


def list_sessions(root: Path) -> list[SessionState]:
    """Every readable session under ``root``, most recently updated first."""
    out = []
    if not root.is_dir():
        return out
    for path in root.glob(f"*/{STATE_FILE}"):
        try:
            out.append(SessionState.model_validate_json(path.read_text()))
        except ValueError:
            continue  # a state this version cannot read is not offered for resume
    return sorted(out, key=lambda s: s.updated, reverse=True)


def load(root: Path, session_id: str | None = None) -> SessionState:
    """A session by id or unique id prefix; without one, the most recent."""
    sessions = list_sessions(root)
    if not sessions:
        raise SessionError(f"no sessions under {root}")
    if not session_id:
        return sessions[0]
    hits = [s for s in sessions if s.session_id.startswith(session_id)]
    if len(hits) != 1:
        known = ", ".join(s.session_id for s in sessions[:5])
        raise SessionError(f"{'no' if not hits else 'more than one'} session matches "
                           f"{session_id!r} (recent: {known})")
    return hits[0]


def transcript_for(root: Path, state: SessionState) -> Transcript:
    return Transcript(root / state.session_id, episode_id=state.session_id)


class ChatSession:
    """A loop, its state, and the bookkeeping between them."""

    def __init__(self, loop: AgentLoop, state: SessionState, root: Path):
        self.loop, self.state, self.root = loop, state, root

    @classmethod
    def begin(cls, loop: AgentLoop, state: SessionState, root: Path,
              system_prompt: str) -> ChatSession:
        loop.start(system_prompt, transcript=transcript_for(root, state))
        session = cls(loop, state, root)
        session.checkpoint()
        return session

    @classmethod
    def resume(cls, loop: AgentLoop, state: SessionState, root: Path) -> ChatSession:
        loop.start(transcript=transcript_for(root, state), messages=state.messages)
        return cls(loop, state, root)

    def ask(self, text: str) -> Episode:
        episode = self.loop.ask(text)
        self.state.asks += 1
        self.state.prompt_tokens += episode.prompt_tokens
        self.state.completion_tokens += episode.completion_tokens
        self.state.last_stop = episode.stop_reason.value
        self.state.last_question = text[:200]
        self.checkpoint()
        return episode

    def switch(self, model: str | None = None, endpoint: str | None = None) -> None:
        """Use another model (or endpoint) from the next question on. Recorded in the
        transcript, so the session says which model gave which answer, and remembered as
        what `myrmidon` starts with."""
        if endpoint is not None and endpoint != self.state.endpoint:
            from myrmidon.endpoints import get_endpoint

            self.loop.connection = get_endpoint(endpoint).call_kwargs()
            self.loop.endpoint = self.state.endpoint = endpoint
        if model:
            self.loop.model = self.state.model = model
        self.loop.transcript.emit("model_change", model=self.state.model,
                                  endpoint=self.state.endpoint)
        save_prefs(self.state.endpoint, self.state.model)
        self.checkpoint()

    def checkpoint(self) -> None:
        ws = self.loop.ctx.workspace
        self.state.workspace = str(ws.root) if ws is not None else self.state.workspace
        self.state.messages = self.loop.messages
        save(self.root, self.state)

    @property
    def transcript_path(self) -> Path:
        return self.loop.transcript.path


def describe(state: SessionState) -> str:
    """One line for a session listing."""
    study = Path(state.workspace).name if state.workspace else "(no study)"
    where = f"{state.model}" + (f" @ {state.endpoint}" if state.endpoint else "")
    return (f"{state.session_id}  {state.updated[:16].replace('T', ' ')}  {study:<24} "
            f"{where:<32} {state.asks} question(s)  {json.dumps(state.last_question[:50])}")


def build_loop(ctx, model: str, endpoint=None, registry=None, completion_fn=None,
               call_timeout_s: float | None = None, **loop_kwargs) -> AgentLoop:
    """The native loop on the PydanticAI adapter, reaching ``endpoint`` (a name from
    endpoints.toml, or an ``Endpoint``) when one is given. Chat and `run` are built
    here; ``loop_kwargs`` are the caller's hooks and limits
    (``on_event``, ``on_decision``, ``max_turns``, ``enforce_grounding``, ...)."""
    from myrmidon.core.tool import REGISTRY

    connection = name = None
    if endpoint:
        if isinstance(endpoint, str):
            from myrmidon.endpoints import get_endpoint

            endpoint = get_endpoint(endpoint)
        connection, name = endpoint.call_kwargs(), endpoint.name
    if completion_fn is None:
        from myrmidon.agent.pydantic_adapter import CALL_TIMEOUT_S, PydanticCompletion

        completion_fn = PydanticCompletion(call_timeout_s=call_timeout_s or CALL_TIMEOUT_S)
    if "opening" not in loop_kwargs and (registry or REGISTRY) is REGISTRY:
        # the harness's sessions start by choosing a skill; a baseline has none to choose
        from myrmidon.knowledge.lookup import skill_opening

        loop_kwargs["opening"] = skill_opening()
    return AgentLoop(model=model, ctx=ctx, registry=registry or REGISTRY,
                     completion_fn=completion_fn, connection=connection, endpoint=name,
                     **loop_kwargs)


def default_endpoint() -> str | None:
    """The only endpoint defined, when there is exactly one."""
    from myrmidon.endpoints import load_endpoints

    endpoints = load_endpoints()
    return next(iter(endpoints)) if len(endpoints) == 1 else None


# --- the model you used last: so `myrmidon` needs no flags ------------------------------

PREFS_NAME = "agent.json"


class AgentPrefs(BaseModel):
    """What `myrmidon` starts with: the endpoint and model chosen last (by the picker,
    `/model`, `/endpoint` or a flag). A preference, not a setting: a `--model` given on the
    command line always wins. Beside the machine config; it holds no secret."""

    endpoint: str | None = None
    model: str | None = None


def prefs_path() -> Path:
    from myrmidon.bootstrap import config_path

    return config_path().parent / PREFS_NAME


def load_prefs() -> AgentPrefs:
    path = prefs_path()
    try:
        return AgentPrefs.model_validate_json(path.read_text())
    except (OSError, ValueError):
        return AgentPrefs()


def save_prefs(endpoint: str | None, model: str) -> None:
    path = prefs_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(AgentPrefs(endpoint=endpoint, model=model).model_dump_json(indent=1))
        os.replace(tmp, path)
    except OSError:
        pass  # a preference that cannot be saved is asked for again next time


def served_models(endpoint: str | None, timeout: float = 15.0) -> list[str]:
    """The model ids an endpoint serves (``GET /models``), sorted; empty when it cannot be
    asked."""
    if not endpoint:
        return []
    import httpx

    from myrmidon.endpoints import EndpointError, get_endpoint, list_models

    try:
        data = list_models(get_endpoint(endpoint), timeout)
    except (EndpointError, httpx.HTTPError, ValueError):
        return []
    return sorted(str(m["id"]) for m in data if m.get("id"))
