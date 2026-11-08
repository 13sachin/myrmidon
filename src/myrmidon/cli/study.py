"""Named commands that are one tool call: study create and open, plan show (the mandate
and the trace), input sets listed and shown."""

from __future__ import annotations

import typer

from myrmidon.cli._app import app, call_and_exit, tool_context

study_app = typer.Typer(
    help="Studies: one question each, with its directory, its ledger, and which is active.",
    no_args_is_help=True,
)


app.add_typer(study_app, name="study", hidden=True)


# `workspace` is an alias for `study`, so existing scripts keep working; hidden so the
# help lists one word for it
app.add_typer(study_app, name="workspace", hidden=True)


@study_app.command("create")
def study_create(
    name: str = typer.Argument(..., help="short study name; it becomes the directory "
                               "under studies_root, and may name subdirectories "
                               "('_scratch/try-3' — where smoke runs belong)"),
    description: str = typer.Option(..., "--description", help="one line: what this "
                                    "study is for. It is the first thing in the ledger."),
) -> None:
    """Create a study under studies_root and make it the active one.

    NAME may contain slashes: `_scratch/tests/acetone` puts the workspace where
    `myrmidon doctor` expects a throwaway run to be.
    """
    call_and_exit("study_create", {"name": name, "description": description})


@study_app.command("open")
def study_open(
    path: str = typer.Argument("", help="path to an existing workspace; empty: report "
                               "the active one"),
) -> None:
    """Make a workspace active — or, with no path, say where the active one stands."""
    call_and_exit("study_open", {"path": path})


protocol_app = typer.Typer(
    help="Input sets: source-bearing parameter recipes for runs outside a mandate's "
         "stages, layered engine < site < project; a stage's values are its template's.",
    no_args_is_help=True,
)


app.add_typer(protocol_app, name="inputset", hidden=True)


plan_app = typer.Typer(
    help="What this study set out to do and what it did: the mandate and the trace.",
    no_args_is_help=True,
)


app.add_typer(plan_app, name="plan", hidden=True)


@plan_app.command("show")
def plan_show(
    workspace: str = typer.Option("", "--study", "-s", help="study directory"),
    detail: bool = typer.Option(False, "--detail", help="with what the engines are told per "
                                "stage, the locks, the measuring calls, the recovery terms "
                                "and the full digest"),
) -> None:
    """The study's mandate as the researcher approved it, with what its jobs booked, then
    the trace: every call the ledger records, with its outcome, its files and its job."""
    from myrmidon.core import mandate as mandates
    from myrmidon.execution.jobs import booked_under_mandates
    from myrmidon.orchestration.report import trace_lines

    ctx = tool_context(workspace)
    ws = ctx.workspace
    mandate = mandates.read_mandate(ws)
    if mandate is None:
        typer.echo("this study has no mandate — the agent writes one with mandate_write")
    else:
        typer.echo(mandates.render(mandate,
                                   booked_under_mandates(ws, mandates.approved_digests(ws)),
                                   detail=detail))
    typer.echo("")
    typer.echo("TRACE")
    for line in trace_lines(ws.ledger.events()) or ["(nothing has run)"]:
        typer.echo(line)


@protocol_app.command("list")
def protocol_list() -> None:
    """Every input set this machine can see, with the layer it was found in."""
    from myrmidon.cli_call import render_protocol_list
    from myrmidon.core.tool import REGISTRY

    ctx = tool_context("")
    obs = REGISTRY.call("inputset_lookup", ctx, {"query": ""})
    typer.echo(render_protocol_list(obs) if obs.ok else obs.summary)
    raise typer.Exit(0 if obs.ok else 1)


@protocol_app.command("show")
def protocol_show(
    inputset: str = typer.Argument(..., help="inputset id, from `myrmidon inputset list`"),
) -> None:
    """One input set: every value, why it is that value, what it refuses to supply, and
    the sources behind it. Rendering it into a real file is `call inputset_apply`."""
    from myrmidon.cli_call import render_protocol
    from myrmidon.core.tool import REGISTRY

    ctx = tool_context("")
    obs = REGISTRY.call("inputset_lookup", ctx, {"query": inputset})
    hit = next((p for p in obs.inputsets if p.id == inputset), None)
    if hit is None:
        typer.echo(f"no inputset '{inputset}' — myrmidon inputset list")
        raise typer.Exit(2)
    typer.echo(render_protocol(hit))
