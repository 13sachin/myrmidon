"""The agent commands: an agent client started wired to the harness, one question run
unattended, and the model endpoints."""

from __future__ import annotations

import typer

import myrmidon
from myrmidon.cli._app import (
    CLIENT_HELP,
    CONFIG_HELP,
    ENDPOINT_HELP,
    LAUNCHABLE,
    MODEL_HELP,
    TIMEOUT_HELP,
    agent_context,
    agent_endpoint,
    app,
    config_file,
    mock_state_dir,
)
from myrmidon.client_registry import REGISTRY


def launch_client(prompt: str = "", model: str = "", continue_: bool = False,
                  resume: str = "", client: str = "") -> None:
    """Bare `myrmidon`, `myrmidon "QUESTION"` and `myrmidon client`: the configured agent
    client, in this directory, wired to this harness."""
    from myrmidon.server.launch import LaunchError, launch

    if client and client not in REGISTRY:
        typer.echo(f"error: unknown client {client!r}; one of {', '.join(REGISTRY)} "
                   f"(started by myrmidon: {', '.join(LAUNCHABLE)}).")
        raise typer.Exit(2)
    try:
        launch(model=model, continue_=continue_, resume=resume, prompt=prompt, client=client)
    except LaunchError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(2) from None


_QUESTION_ARG = typer.Argument(..., help="the question, in your own words")


@app.command("ask", hidden=True)
def ask(
    question: list[str] = _QUESTION_ARG,
    client: str = typer.Option("", "--client", help=CLIENT_HELP),
    model: str = typer.Option("", "--model", help="model for the client to use"),
) -> None:
    """Start your agent here with the question typed — what `myrmidon "QUESTION"` does."""
    launch_client(prompt=" ".join(question), model=model, client=client)


@app.command("client", hidden=True)
def client_command(
    prompt: str = typer.Argument("", help="what you want done; asked in the session "
                                 "when you leave it out"),
    model: str = typer.Option("", "--model", help="model for the client to use"),
    continue_: bool = typer.Option(False, "--continue", "-c",
                                   help="continue this directory's last conversation"),
    resume: str = typer.Option("", "--resume", help="resume a conversation by id"),
    client: str = typer.Option("", "--client", help=CLIENT_HELP),
) -> None:
    """Start an agent client (Claude Code, Qwen Code, Goose) wired to this harness.

    The directory is made a myrmidon project first if it is not one already (the same
    thing `myrmidon init-project` does), so the client finds the server, the guidance,
    the status line, the session-start hook and the skills where it looks for them.
    """
    launch_client(prompt=prompt, model=model, continue_=continue_, resume=resume,
                  client=client)


@app.command("version", hidden=True)
def version_command() -> None:
    """Print the myrmidon version."""
    typer.echo(myrmidon.__version__)


@app.command("run", hidden=True)
def agent_run(
    prompt: str,
    model: str = typer.Option("", "--model", "-m", help=MODEL_HELP),
    endpoint: str = typer.Option("", "--endpoint", "-e", help=ENDPOINT_HELP),
    max_turns: int = typer.Option(40),
    call_timeout: float = typer.Option(0, "--call-timeout", help=TIMEOUT_HELP,
                                      show_default=False),
    study: str = typer.Option("", "--study", help="study to work in (default: active)"),
    config: str = typer.Option("", "--config", help=CONFIG_HELP),
    mock: bool = typer.Option(False, "--mock", help="fake engines and a fake scheduler: "
                              "nothing real runs"),
) -> None:
    """One question, unattended (for scripts).

    The agent works until it answers or stops; a finding that needs a human is refused,
    and the run reports it instead of accepting it.

    Exit code: 0 answered, with every number traced to a tool's output; 1 no such answer
    (out of turns, tokens or time, a reply cut off, or numbers that could not be traced);
    3 the model or its endpoint failed. 2 is a usage error, as for every command.

    Every call goes through the same registry a person types at, so the gates and the
    ledger apply unchanged — and `myrmidon watch` shows what it is doing.
    """
    from myrmidon.agent.session import build_loop, run_transcripts

    chosen = agent_endpoint(endpoint)
    model = agent_model(model, chosen)
    chosen_config = config_file(config)
    if mock:
        from myrmidon.engines import mock as mocks

        # before the context is built, so discovery finds the mock binaries
        mocks.enter(mock_state_dir(study, chosen_config))
    ctx = agent_context(study, config=chosen_config)
    if mock:
        ctx = mocks.mock_context(ctx)
    loop = build_loop(ctx, model, endpoint=chosen,
                      call_timeout_s=call_timeout or None, max_turns=max_turns,
                      transcript_dir=run_transcripts(ctx))
    episode = loop.run(prompt)
    typer.echo(f"episode {episode.episode_id}: {episode.stop_reason}")
    typer.echo(f"transcript: {loop.transcript.path}")
    if episode.error:
        typer.echo(episode.error)
    typer.echo(f"turns={episode.n_turns} tool_calls={episode.n_tool_calls} "
               f"tokens={episode.prompt_tokens}+{episode.completion_tokens}")
    typer.echo("")
    typer.echo(episode.final_answer or "(no final answer)")
    raise typer.Exit(run_exit_code(episode))


def run_exit_code(episode) -> int:
    """What `myrmidon run` exits with, so a script can tell an answer from a run that
    stopped short, and both from a model that could not be reached."""
    from myrmidon.agent.loop import StopReason

    if episode.stop_reason == StopReason.MODEL_ERROR:
        return 3
    return 0 if episode.stop_reason == StopReason.COMPLETED else 1


def agent_model(model: str, endpoint: str | None, remember: bool = True) -> str:
    """The model asked for, else the one used last on this endpoint. Remembered for next
    time unless ``remember`` is False (a tick's wake chooses nothing for the user)."""
    from myrmidon.agent.session import load_prefs, save_prefs

    prefs = load_prefs()
    chosen = model or (prefs.model if prefs.endpoint == endpoint else None)
    if not chosen:
        typer.echo("no model chosen: pass --model (`myrmidon endpoints --json` lists what each "
                   "endpoint serves)")
        raise typer.Exit(2)
    if remember and (chosen != prefs.model or endpoint != prefs.endpoint):
        save_prefs(endpoint, chosen)
    return chosen


@app.command("endpoints", hidden=True)
def endpoints_cmd(
    as_json: bool = typer.Option(False, "--json", help="machine-readable, for a client "
                                 "that registers the endpoints as model providers: each "
                                 "one's base URL, key variable and the models it serves "
                                 "(cached a day), and the secrets file's path. Never a key"),
    refresh: bool = typer.Option(False, "--refresh", help="with --json: ask each endpoint "
                                 "for its models again"),
) -> None:
    """The named model endpoints (endpoints.toml beside the machine config), and whether
    each one's key variable is set in this shell. Never prints a key."""
    import os

    from myrmidon.endpoints import EndpointError, endpoints_path, load_endpoints

    path = endpoints_path()
    try:
        endpoints = load_endpoints(path)
    except EndpointError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from None
    if as_json:
        import json

        from myrmidon.endpoints import apply_secrets, cached_models, secrets_path

        apply_secrets()  # this process only: the models list needs the key
        typer.echo(json.dumps({
            "secrets": str(secrets_path()),
            "endpoints": [
                {"name": name, "api_base": ep.api_base, "key_env": ep.key_env,
                 "key_set": bool(os.environ.get(ep.key_env)),
                 "context_window": ep.context_window,
                 "models": cached_models(ep, refresh=refresh)}
                for name, ep in sorted(endpoints.items())],
        }))
        return
    if not endpoints:
        typer.echo(f"no endpoints defined ({path} absent or empty). One per server:\n\n"
                   "  [endpoints.NAME]\n  api_base = \"https://host/v1\"\n"
                   "  key_env = \"NAME_API_KEY\"   # the variable, not the key")
        return
    typer.echo(f"{path}:")
    for name, ep in sorted(endpoints.items()):
        state = "set" if os.environ.get(ep.key_env) else "NOT SET"
        typer.echo(f"  {name:<16} {ep.api_base}   ${ep.key_env} {state}")
