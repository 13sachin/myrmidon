"""What `myrmidon run` is built from: where its transcript goes, the loop on the model
adapter, the endpoint and model used last."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from myrmidon.agent.loop import AgentLoop
from myrmidon.core.studies import SESSIONS_DIRNAME as SESSIONS_DIR


def sessions_root(studies_root: Path) -> Path:
    root = Path(studies_root).expanduser()
    return (root if root.is_absolute() else Path.cwd() / root) / SESSIONS_DIR


def run_transcripts(ctx) -> Path:
    """Where `myrmidon run` writes its transcript: in the study it starts in, beside the
    job store (``.myrmidon/transcripts/``), so the record stays with what it did; with no
    study open yet, under the studies root (``.sessions/runs/``). Never the working
    directory."""
    from myrmidon.core.workspace import STATE_DIRNAME

    if ctx.workspace is not None:
        return ctx.workspace.root / STATE_DIRNAME / "transcripts"
    return sessions_root(ctx.config.studies_root) / "runs"


def build_loop(ctx, model: str, endpoint=None, registry=None, completion_fn=None,
               call_timeout_s: float | None = None, **loop_kwargs) -> AgentLoop:
    """The unattended loop on the PydanticAI adapter, reaching ``endpoint`` (a name from
    endpoints.toml, or an ``Endpoint``) when one is given; ``loop_kwargs`` are the
    caller's hooks and limits (``on_event``, ``max_turns``, ``token_budget``, ...)."""
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
        # the harness's runs start by choosing a skill; a baseline has none to choose
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


# --- the model used last: so `run` and `tick --model` need no flags ---------------------

PREFS_NAME = "agent.json"


class AgentPrefs(BaseModel):
    """The endpoint and model `run` used last. A preference, not a setting: a `--model`
    given on the command line always wins. Beside the machine config; it holds no secret."""

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
        pass  # a preference that cannot be saved is given again next time
