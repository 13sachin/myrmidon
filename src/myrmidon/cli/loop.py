"""A study from a terminal: a person's approval of its mandate, a tick, the inbox and its
answers, and where the study stands."""

from __future__ import annotations

import typer

from myrmidon.cli._app import (
    CONFIG_HELP,
    ENDPOINT_HELP,
    PANEL_WORK,
    agent_endpoint,
    app,
    config_file,
    loop_context,
    resolve_workspace,
)
from myrmidon.cli.agent import agent_model


def inside_agent() -> str:
    """The client whose shell this process runs in, by the variable it sets, or ""."""
    import os

    from myrmidon.client_registry import REGISTRY as CLIENTS

    for desc in CLIENTS.values():
        if desc.nested_env and os.environ.get(desc.nested_env):
            return desc.display
    return ""


def _origin() -> str:
    """Where the yes was given, as the record keeps it: the controlling terminal and the
    processes above this one. Not a boundary — attribution, for the record a study is
    judged by."""
    import os
    import sys
    from pathlib import Path

    try:
        tty = os.ttyname(sys.stdin.fileno())
    except (OSError, ValueError):
        tty = "none"
    chain, pid = [], os.getppid()
    for _ in range(5):
        try:
            comm = Path(f"/proc/{pid}/comm").read_text().strip()
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            break
        chain.append(comm)
        pid = int(stat.rsplit(")", 1)[1].split()[1])
        if pid <= 1:
            break
    return f"tty={tty} parents={','.join(chain) or 'unknown'}"


def _approve_mandate(ctx, *, digest: str, show: bool, as_json: bool, yes: bool, via: str,
                     approver: str, detail: bool = False) -> None:
    """The mandate on disk, shown; a person's yes, bound to the digest they read; the
    approval recorded in the ledger in their name."""
    import getpass
    import json

    import myrmidon
    from myrmidon.core import mandate as mandates
    from myrmidon.execution.jobs import booked_under_mandates

    ws = ctx.workspace
    mandate = mandates.read_mandate(ws)
    spent = booked_under_mandates(ws, mandates.approved_digests(ws))
    from myrmidon.tools.mandate import compiled

    comp = compiled(ctx, mandate) if mandate is not None else None
    prices = {name: p.core_h for name, p in comp.prices.items()} if comp else None
    view = mandates.view(ws, mandate, spent, comp.lines if comp else None,
                         comp.priced if comp else None, comp.measured if comp else None,
                         detail=detail, core_h=prices)
    if mandate is None:
        typer.echo(f"✗ {view['blocked']}")
        raise typer.Exit(1)
    if show:
        typer.echo(json.dumps(view, indent=2) if as_json else view["text"])
        raise typer.Exit(1 if view["blocked"] else 0)
    if view["blocked"]:
        typer.echo(f"✗ the mandate cannot be approved: {view['blocked']}")
        raise typer.Exit(1)
    if digest and not mandate.digest.startswith(digest.lower()):
        typer.echo(f"✗ that digest is not this mandate's: the mandate on disk is "
                   f"{mandate.digest[:12]}. Read it again (`myrmidon approve --show`).")
        raise typer.Exit(1)
    who = approver or getpass.getuser()
    if not yes:
        typer.echo(view["text"])
        typer.echo("")
        if not typer.confirm(f"Approve mandate revision {mandate.revision}, at most "
                             f"{mandate.cost.ceiling_core_h:g} core-h?", default=False):
            typer.echo("not approved; nothing recorded")
            raise typer.Exit(1)
    event = next((a.event_id for a in mandates.approvals(ws) if a.digest == mandate.digest),
                 "") or mandates.record_approval(ws, mandate, approver=who, via=via,
                                                 origin=_origin(),
                                                 myrmidon_version=myrmidon.__version__)
    summary = (f"mandate revision {mandate.revision} approved: its stages may book at most "
               f"{mandate.cost.ceiling_core_h:g} core-h")
    if as_json:
        typer.echo(json.dumps({"ok": True, "event": event, "approver": who, "via": via,
                               "digest": mandate.digest, "summary": summary}))
        return
    typer.echo(f"{event}: approved by {who} — {summary} (digest {mandate.digest[:12]}). The "
               "session carries the study on under it; `myrmidon status` shows where it stands")


@app.command("approve", rich_help_panel=PANEL_WORK)
def approve(
    study: str = typer.Option("", "--study", help="the study (default: the active one)"),
    approver: str = typer.Option("", "--approver", help="who is approving (default: your "
                                 "user name)"),
    config: str = typer.Option("", "--config", help=CONFIG_HELP),
    show: bool = typer.Option(False, "--show", help="print the mandate as the researcher "
                              "approves it — the question, the system, the stages and their "
                              "prices, the decisions, the deliverables, the cost and ceiling, "
                              "the revision — and stop"),
    detail: bool = typer.Option(False, "--detail", help="with the mandate, what the engines "
                                "are told per stage, the locks, the measuring calls, the "
                                "recovery terms and the full digest"),
    as_json: bool = typer.Option(False, "--json", help="machine-readable: with --show the "
                                 "mandate view, after a --yes the approval event"),
    digest: str = typer.Option("", "--digest", help="approve only while the mandate on disk "
                               "still has this digest (the one you read)"),
    yes: bool = typer.Option(False, "--yes", help="approve without asking: the path a "
                             "client's approval dialog takes after the person said yes"),
    via: str = typer.Option("terminal", "--via", help="how the yes reached myrmidon: "
                            "terminal | hermes-plugin | pi-extension | claude-hook"),
    mock: bool = typer.Option(False, "--mock", help="the study runs on the mock engines"),
) -> None:
    """Approve this study's mandate, as the person responsible for it.

    The mandate is shown as you sign it — the question, the system and its files, the
    method, what you decided and what the agent proposes, the deliverables, the cost and
    your ceiling — and one yes records the approval in your name, bound to its digest: edit
    the mandate or a system file and it stops holding.
    Refused from inside an agent's shell: approving is a person's act, at a terminal (in a
    Pi session, the approval dialog does this).
    """
    if not show and via == "terminal" and (client := inside_agent()):
        typer.echo(f"refused: this shell is inside {client}. Approving is a person's act at "
                   "a terminal — in the session, the approval dialog; outside it, another "
                   "terminal")
        raise typer.Exit(2)
    _approve_mandate(loop_context(study, config, mock), digest=digest, show=show,
                     as_json=as_json, yes=yes, via=via, approver=approver, detail=detail)


def _agent_wake(ctx, model: str, endpoint: str):
    """The wake of a study under a mandate: the agent itself, on the native loop, with the
    same tools, gate and ledger as a session; None without a model, so a tick that needs
    the agent tells the researcher instead."""
    if not model and not endpoint:
        return None
    from myrmidon.agent.session import build_loop
    from myrmidon.orchestration.continuation import WakeResult

    chosen = agent_endpoint(endpoint)
    name = agent_model(model, chosen, remember=False)

    def wake(prompt, directory, turns):
        # the prompt names the study and what happened: no skill is chosen first
        loop = build_loop(ctx, name, endpoint=chosen, max_turns=turns,
                          transcript_dir=directory, opening=None)
        episode = loop.run(prompt)
        return WakeResult(loop.transcript.path, str(episode.stop_reason),
                          episode.final_answer or "")

    return wake


@app.command("tick", hidden=True)
def tick_cmd(
    study: str = typer.Option("", "--study", help="the study (default: the active one)"),
    follow: float = typer.Option(0, "--follow", help="seconds between ticks until the study "
                                 "is done or waits on a person; 0 runs one tick and exits. "
                                 "Crash-only: kill it at any instant and start it again"),
    model: str = typer.Option("", "--model", "-m", help="the wake model: the agent woken to "
                              "carry the study on under its mandate"),
    endpoint: str = typer.Option("", "--endpoint", "-e", help=ENDPOINT_HELP),
    config: str = typer.Option("", "--config", help=CONFIG_HELP),
    mock: bool = typer.Option(False, "--mock", help="fake engines and a fake scheduler"),
    cron_line: bool = typer.Option(False, "--cron-line", help="print a cron entry that "
                                   "ticks this study every ten minutes, and exit"),
) -> None:
    """Advance the study by one pass under its approved mandate: collect what ended, rerun
    a failure as the mandate's recovery allows, otherwise wake the agent named by --model
    to carry the study on, and park what needs a person (orchestration/continuation.py).

    Between ticks only the engines move: there is no resident process, and a tick killed
    at any instant is reconciled by the next one rather than repeated. Exit code: 0 when
    something was done or nothing is pending, 1 when no approved mandate holds, 3 when the
    study is locked by another tick.
    """
    import time

    from myrmidon.orchestration.continuation import tick

    ctx = loop_context(study, config, mock)
    if cron_line:
        flags = (f" --study {ctx.workspace.root}" + (f" --config {config}" if config else "")
                 + (f" --endpoint {endpoint}" if endpoint else "")
                 + (f" --model {model}" if model else "") + (" --mock" if mock else ""))
        typer.echo(f"*/10 * * * * myrmidon tick{flags} >> "
                   f"{ctx.workspace.state_dir}/tick.log 2>&1")
        return
    wake = _agent_wake(ctx, model, endpoint)
    while True:
        report = tick(ctx, wake=wake)
        typer.echo(report.render())
        if report.blocked:
            raise typer.Exit(3 if "another tick" in report.blocked else 1)
        if not follow or report.done or (report.parked and not report.running):
            return
        time.sleep(follow)


@app.command("inbox", hidden=True)
def inbox_cmd(
    study: str = typer.Option("", "--study", help="the study (default: the active one)"),
    config: str = typer.Option("", "--config", help=CONFIG_HELP),
    all_entries: bool = typer.Option(False, "--all", help="answered questions and notices too"),
) -> None:
    """What waits on a person: each parked question with its evidence and its options.
    Answer one with `myrmidon answer ID OPTION`."""
    from myrmidon.orchestration.spool import Inbox

    ws = resolve_workspace(study, config_file(config))
    if ws is None:
        typer.echo("no study: give --study PATH, or open one (`myrmidon study open PATH`)")
        raise typer.Exit(2)
    inbox = Inbox(ws.root)
    entries = inbox.all() if all_entries else inbox.open_questions()
    if not entries:
        typer.echo("nothing waits on you" if not all_entries else "the inbox is empty")
        return
    for e in entries:
        if e.answers:
            typer.echo(f"{e.ts}  answer to {e.answers}: {e.option} by {e.by}")
            continue
        typer.echo(f"{e.entry_id}  {e.kind}  {e.subject}  ({e.ts})")
        typer.echo(f"  {e.question}")
        for line in e.evidence:
            if line:
                typer.echo(f"    - {line}")
        for o in e.options:
            typer.echo(f"    {o.id:<10} {o.text}")
        if e.cites:
            typer.echo("    cites " + ", ".join(c for c in e.cites if c))


@app.command("answer", rich_help_panel=PANEL_WORK)
def answer_cmd(
    entry_id: str = typer.Argument(..., help="the inbox question (in-…)"),
    option: str = typer.Argument(..., help="one of its option ids"),
    study: str = typer.Option("", "--study", help="the study (default: the active one)"),
    by: str = typer.Option("", "--by", help="who answers (default: your user name)"),
    config: str = typer.Option("", "--config", help=CONFIG_HELP),
) -> None:
    """Answer a parked question by its option id. The next tick carries it out."""
    import getpass

    from myrmidon.orchestration.spool import Inbox

    ws = resolve_workspace(study, config_file(config))
    if ws is None:
        typer.echo("no study: give --study PATH, or open one (`myrmidon study open PATH`)")
        raise typer.Exit(2)
    inbox = Inbox(ws.root)
    question = inbox.get(entry_id)
    if question is None:
        typer.echo(f"no question {entry_id} in {ws.root}; `myrmidon inbox` lists them")
        raise typer.Exit(2)
    who = by or getpass.getuser()
    try:
        inbox.answer(entry_id, option, by=who)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(2) from None
    chosen = next(o for o in question.options if o.id == option)
    typer.echo(f"{entry_id}: {option} — {chosen.text}. The next tick carries it out.")


@app.command("status", rich_help_panel=PANEL_WORK)
def status_cmd(
    study: str = typer.Option("", "--study", help="the study (default: the active one)"),
    config: str = typer.Option("", "--config", help=CONFIG_HELP),
) -> None:
    """Where the study stands, from its own files and no model: the mandate and what it has
    booked, the continuation, the jobs, what waits on a person, the open journal."""
    from myrmidon.core.tool import REGISTRY
    from myrmidon.execution.collect import pending_lines
    from myrmidon.orchestration.continuation import progress
    from myrmidon.orchestration.spool import Inbox, Journal

    ctx = loop_context(study, config, mock=False)
    ws = ctx.workspace
    typer.echo(f"study {ws.meta.get('name', ws.root.name)} · {ws.root}")
    typer.echo("  " + REGISTRY.call("mandate_show", ctx, {}).summary)
    line = progress(ws)
    if line:
        typer.echo("  " + line)
    jobs = pending_lines(ctx)
    if jobs:
        typer.echo("jobs:")
        for line in jobs:
            typer.echo(f"  {line}")
    questions = Inbox(ws.root).open_questions()
    if questions:
        typer.echo("waiting on you:")
        for q in questions:
            typer.echo(f"  {q.entry_id}  {q.subject}: {q.question[:120]}")
    entries = Journal(ws.root).open_entries()
    if entries:
        typer.echo("open in the journal:")
        for e in entries:
            typer.echo(f"  [{e.kind}, {e.by}] {e.text[:140]}")
