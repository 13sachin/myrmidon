"""Myrmidon's presence inside Hermes: harness code beside the person.

Through MCP the harness only ever hears tool calls, which are the model's. What is the
person's, or the harness's own to check, happens here, in Hermes's process:

- the approval gate: a shell spelling of the researcher's act, `myrmidon approve`, is
  blocked before it runs. The researcher reads the mandate with `/approve` and gives their
  yes by naming the digest they read, or `/approve yes` for the one `/approve` last showed
  in that session, which runs `myrmidon approve` as them. The yes is theirs: it never
  arrives as something the model said;
- the look at the start of a turn: where the study stands, as context for the first run;
- the views: /status, /plan, /inbox, /answer, /approve.

This file runs inside Hermes's own interpreter, not myrmidon's, and holds no logic about
mandates or findings: the CLI renders, the CLI records, the server checks, and all run from
the install named in Hermes's `mcp_servers` entry (`_harness`).
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict, deque
from pathlib import Path

from ._harness import SERVER, cli, run, server_entry, session_cwd

TERMINAL_TOOLS = ("terminal", "execute_code")
#: A shell spelling of the approval, the person's act: `myrmidon approve` however the
#: words are tokenised (a shell line, a list of words, `python -m myrmidon.cli`). Reading
#: the ledger or the source is not the act and passes.
SHELL_APPROVAL = re.compile(r"\bmyrmidon(?:\.cli)?\b[^\w\n]*\bapprove\b")
VIA = ("--via", "hermes-plugin")

TERMINAL_VERB = (
    "Approving is the researcher's act. Say the stages and their cost in core-hours and end "
    "the turn; "
    "they approve with /approve in this session, or at a terminal with `myrmidon approve`."
)
#: how many sessions' last-shown digests the plugin keeps, as it bounds its first look
SHOWN_BOUND = 64
NO_SERVER = f"no {SERVER} server entry in Hermes's config — run `myrmidon mcp install`."


# ---------------------------------------------------------------- the approval


def _mandate_view(entry: dict, cwd: Path) -> tuple[dict | None, str]:
    """The mandate as the approval dialog reads it, or why it cannot be approved."""
    code, stdout, stderr = _show(entry, cwd)
    if code != 0:
        # the view names what blocks it; that, not the view, is the line the researcher reads
        blocks: list[str] = []
        try:
            refused = json.loads(stdout)
            blocks = [str(refused.get("blocked") or "")] + [
                str(f.get("message", "")) for f in refused.get("findings", [])
                if f.get("severity") == "block"]
        except (ValueError, AttributeError, TypeError):
            pass  # not a view: the command's own words say why
        why = "\n".join(b for b in blocks if b) or (stdout + stderr).strip()
        return None, f"it cannot be approved as it stands: {why}"
    try:
        view = json.loads(stdout)
    except ValueError:
        return None, f"`myrmidon approve --show` returned no readable view:\n{stdout[:2000]}"
    return (view, "") if isinstance(view, dict) else (None, "the view was not a mandate")


def _show(entry: dict, cwd: Path) -> tuple[int, str, str]:
    return run([*cli(entry, ["approve"]), "--show", "--json", *VIA], cwd)


def _approval_text(view: dict) -> str:
    """The mandate in full, for the transcript, as the harness renders it."""
    return str(view.get("text", ""))


def _dialog_text(view: dict) -> str:
    """What the researcher is asked: a dozen lines a terminal of any height holds, with
    the full mandate already above it."""
    digest = str(view.get("digest", ""))
    estimate = str(view.get("estimate") or "")
    return "\n".join([
        str(view.get("question", "")), "",
        f"cost       {estimate + '; ' if estimate else ''}"
        f"at most {view.get('ceiling_core_h', 0):g} core-h"
        + (f", {view.get('spent_core_h'):g} already booked" if view.get("spent_core_h")
           else ""),
        f"revision   {view.get('revision')}", "",
        f"digest     {digest[:12]}",
        f"To approve, type `/approve yes` (or `/approve {digest[:12]}`).",
        "Anything else leaves the mandate unapproved."])


#: session -> the digest `/approve` last showed in it, oldest forgotten past the bound
_SHOWN: OrderedDict[str, str] = OrderedDict()


def _remember_shown(session_id: str, digest: str) -> None:
    _SHOWN[session_id] = digest
    _SHOWN.move_to_end(session_id)
    while len(_SHOWN) > SHOWN_BOUND:
        _SHOWN.popitem(last=False)


def _record(entry: dict, cwd: Path, view: dict, digest: str, session_id: str = "") -> str:
    """The researcher's yes, carried to the harness as them. A typed digest must be a prefix
    of the one on disk; `yes` stands for the digest `/approve` last showed in this session,
    and holds only while the file on disk still has it: one rewritten since is shown again,
    and nothing is recorded."""
    full = str(view.get("digest", ""))
    if digest.lower() in ("yes", "y"):
        shown = _SHOWN.get(session_id, "")
        if not shown:
            return "Read it first with `/approve`, then answer `/approve yes`."
        if shown != full:
            _remember_shown(session_id, full)
            return ("It changed since you read it; nothing was recorded. Here it is as it "
                    "stands now:\n\n" + _approval_text(view) + "\n\n" + _dialog_text(view))
        digest = shown
    if not digest or not full.startswith(digest.lower()):
        return (f"that digest is not this one's: the one on disk is {full[:12]}. "
                "Read it again with /approve, then answer with `/approve yes` or its digest.")
    code, stdout, stderr = run([*cli(entry, ["approve"]), "--study", str(view.get("study", "")),
                                "--digest", full, "--yes", *VIA, "--json"], cwd, timeout=300)
    if code != 0:
        return f"the approval was not recorded:\n{(stdout + stderr).strip()}"
    try:
        result = json.loads(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return f"approval recorded, but its result was unreadable:\n{stdout[:2000]}"
    return (f"Approved by {result.get('approver')}: "
            f"{result.get('summary')}. The session carries the study on under it; "
            "/status says where it stands.")


# ---------------------------------------------------------------- the plugin


def _view(command: list[str]) -> str:
    entry = server_entry()
    if entry is None:
        return NO_SERVER
    code, stdout, stderr = run(cli(entry, command), session_cwd())
    return (stdout + ("" if code == 0 else "\n" + stderr)).strip() or "(nothing)"


def _answer(args: str = "", **_kwargs) -> str:
    parts = (args or "").strip().split()
    if len(parts) != 2:
        return "usage: /answer ID OPTION (the ids are in /inbox)"
    return _view(["answer", *parts])


def _approve(args: str = "", session_id: str = "", **_kwargs) -> str:
    entry = server_entry()
    if entry is None:
        return NO_SERVER
    cwd = session_cwd()
    view, why = _mandate_view(entry, cwd)
    if view is None:
        return why
    given = (args or "").strip().split()
    if given:
        return _record(entry, cwd, view, given[0], session_id)
    _remember_shown(session_id, str(view.get("digest", "")))
    return _approval_text(view) + "\n\n" + _dialog_text(view)


def register(ctx):
    """Wire the gate, the look and the views into this Hermes session."""
    #: the sessions already looked at: a bound, because the plugin lives as long as the
    #: gateway and a session that comes back after the bound only gets a second look
    looked: deque[str] = deque(maxlen=64)

    def gate(tool_name: str = "", args: dict | None = None, **_kwargs):
        """The approval is the person's: no shell line the model writes can stand in for
        it. A `pre_tool_call` block is what Hermes honours here;
        it fails closed, so a hung or raising gate still refuses."""
        args = args or {}
        if tool_name in TERMINAL_TOOLS:
            command = str(args.get("command") or args.get("code") or "")
            if SHELL_APPROVAL.search(command):
                return {"action": "block",
                        "message": f"a shell command is not the researcher's yes. {TERMINAL_VERB}"}
        return None

    def look(session_id: str = "", is_first_turn: bool = False, **_kwargs):
        """`study open`: what ended while nobody watched, what waits, where the study stands."""
        if not is_first_turn or session_id in looked:
            return None
        looked.append(session_id)
        entry = server_entry()
        if entry is None:
            return None
        code, stdout, _ = run(cli(entry, ["study", "open"]), session_cwd(), timeout=60)
        text = stdout.strip()
        if code != 0 or not text:
            return None
        return {"context": "Where the study stands, as the harness sees it before this "
                           f"turn:\n{text}"}

    ctx.register_hook("pre_tool_call", gate)
    ctx.register_hook("pre_llm_call", look)
    ctx.register_command(
        "status", lambda *_a, **_k: _view(["status"]),
        "where the study stands: the mandate, the jobs, what waits on you")
    ctx.register_command(
        "plan", lambda *_a, **_k: _view(["plan", "show"]),
        "this study's mandate and its trace: what it set out to do and what it did")
    ctx.register_command(
        "inbox", lambda *_a, **_k: _view(["inbox"]),
        "the questions that wait on you, with their options")
    ctx.register_command("answer", _answer, "answer a parked question: /answer ID OPTION")
    ctx.register_command(
        "approve", _approve,
        "read this study's mandate, then approve it: /approve yes, or the digest you read")
