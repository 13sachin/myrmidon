"""Myrmidon CLI: with no command, Myrmidon's own agent; the commands set up the machine,
run the study loop and read back what went through the harness."""

from __future__ import annotations

import typer

from myrmidon.cli._app import CLIENT_HELP, app, show_version
from myrmidon.cli.agent import launch_client


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", callback=show_version,
                                 is_eager=True, help="print the version and exit"),
    client: str = typer.Option("", "--client", help=CLIENT_HELP),
    model: str = typer.Option("", "--model", "-m", help="the model for your agent to use"),
    continue_: bool = typer.Option(False, "--continue", "-c",
                                   help="continue this directory's last conversation"),
    resume: str = typer.Option("", "--resume", "-r", help="resume a conversation by id"),
) -> None:
    """With no command: set this machine up the first time, make this directory a project,
    and start your agent in it, wired to the harness."""
    if ctx.invoked_subcommand is None:
        import sys

        from myrmidon.bootstrap import config_path

        if not config_path().is_file():
            if not sys.stdin.isatty():
                typer.echo("myrmidon is not set up on this machine: run `myrmidon setup` at a "
                           "terminal (or `myrmidon setup --yes` for the defaults)")
                raise typer.Exit(2)
            from myrmidon.cli.setup import run_setup

            typer.echo("myrmidon is not set up on this machine.")
            run_setup(client=client)
            typer.echo("")
        launch_client(model=model, continue_=continue_, resume=resume, client=client)


@app.command(hidden=True)
def workshop() -> None:
    """The commands the front page does not show: the registry from the shell, the runner
    and the readback piece by piece, and what `myrmidon` now does for you."""
    import typer.main

    group = typer.main.get_command(app)
    rows = [(name, cmd.get_short_help_str(limit=72))
            for name, cmd in sorted(group.commands.items()) if cmd.hidden]
    width = max(len(name) for name, _ in rows)
    for name, short in rows:
        typer.echo(f"  {name.ljust(width)}  {short}")


# the commands register on import, in the order --help lists them
from myrmidon.cli import loop, readback, setup, agent, study  # noqa: E402, F401, I001

__all__ = ["app"]
