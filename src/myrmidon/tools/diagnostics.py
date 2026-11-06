"""Tools that report on the harness itself: the session summary, the tool list and the
engines this machine has."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from myrmidon.core.observation import Observation
from myrmidon.core.studies import SCRATCH_DIRNAME, studies_dir, workspace_strays
from myrmidon.core.tool import ToolContext, tool
from myrmidon.knowledge.tool_profiles import shown as shown_by


class EmptyParams(BaseModel):
    pass


class SessionSummaryParams(BaseModel):
    since: str = Field(
        default="session",
        description="'session' (since the workspace was last opened), 'all', a duration "
        "('6h', '90m', '2d'), or an ISO-8601 timestamp",
    )


@tool(
    "session_summary",
    navigation=True,
    description="What went through the harness in a time window: tool calls and failures, "
                "refusals, jobs with core-hours booked and consumed, artifacts, scripts run. "
                "Counted from the ledger and job store; work outside the tools is absent.",
    tags={"workspace", "diagnostics"},
)
def session_summary(ctx: ToolContext, params: SessionSummaryParams) -> Observation:
    from myrmidon.orchestration.summary import summarize

    return summarize(
        ctx.workspace, params.since, backend=ctx.backend, compute=ctx.config.compute,
    )


class ToolLookupParams(BaseModel):
    query: str = Field(
        description="what you are trying to do, in your own words — 'compare candidate "
        "force fields against a QM/MM reference', 'pick frames from a trajectory'")
    limit: int = Field(default=6, description="how many tools to name")


@tool(
    "tool_lookup",
    navigation=True,
    description="Which tool does this job, including tools that exist but need an engine not "
                "installed here. Ask before scripting something an engine computes: a script that "
                "reimplements an engine produces numbers nobody can trace.",
    tags={"knowledge"},
    needs_workspace=False,
)
def tool_lookup(ctx: ToolContext, params: ToolLookupParams) -> Observation:
    """Keyword overlap against every registered tool's name, description and tags.

    The model is shown the tools whose engines this machine has, and a tool whose engine
    is absent is not in its list. An agent that cannot see `observable_measure` and is asked
    which force field agrees best with a QM/MM reference reimplements molecular mechanics
    in a script and reports numbers that are not the engine's. What this tool adds is the
    sentence the agent could not otherwise say: *the tool for this exists, and it needs
    an engine that is not installed here*.
    """
    from myrmidon.core.tool import REGISTRY

    specs = REGISTRY.specs()
    up = {name for name, info in ctx.engines.items() if getattr(info, "available", False)}
    missing = {s.name: sorted(s.requires - up) for s in specs if not s.requires <= up}
    words = {w for w in re.split(r"[^a-z0-9]+", params.query.lower()) if len(w) > 2}

    def score(spec) -> int:
        blob = f"{spec.name} {spec.description} {' '.join(sorted(spec.tags))}".lower()
        return sum(1 for w in words if w in blob)

    ranked = sorted(((score(s), s) for s in specs), key=lambda p: -p[0])
    hits = [s for n, s in ranked if n][:max(1, params.limit)]
    shown = [s.name for s in hits if s.name not in missing]
    hidden = [s.name for s in hits if s.name in missing]

    def gist(spec) -> str:
        text = " ".join(spec.description.split())
        return text if len(text) <= 110 else text[:110].rsplit(" ", 1)[0] + "..."

    def tag(spec) -> str:
        needs = missing.get(spec.name)
        return f" [NOT AVAILABLE HERE: needs {', '.join(needs)}]" if needs else ""

    lines = [f"{s.name}{tag(s)}: {gist(s)}" for s in hits]
    summary = (f"{len(specs) - len(missing)} of {len(specs)} tools are available on this "
               "machine. "
               + ("\n".join(lines) if lines else f"no tool matches {params.query!r}"))
    if hidden:
        summary += (f"\n{len(hidden)} of these need an engine this machine does not have "
                    f"({', '.join(missing[hidden[0]])} for {hidden[0]}). Do not reimplement "
                    "one: name it and stop, so the user can install the engine or move to a "
                    "machine that has it.")
    profiled = shown_by(ctx.config.tool_profile)
    if profiled is not None and any(s.name not in profiled for s in hits):
        summary += ("\nA tool named here that an MCP client's list does not show is called "
                    "with tool_run(name, params).")
    return Observation(ok=bool(hits), summary=summary, shown=shown, hidden=hidden)


@tool(
    "doctor",
    navigation=True,
    description="Report what simulation engines myrmidon finds on this machine, their "
    "versions, what is unavailable and how to fix it, and the active execution backend.",
    tags={"diagnostics"},
    needs_workspace=False,
)
def doctor(ctx: ToolContext, params: EmptyParams) -> Observation:
    engines = {}
    missing = {}
    for name, info in ctx.engines.items():
        if info.available:
            engines[name] = info.version
        else:
            missing[name] = info.hint
    backend = ctx.backend.name if ctx.backend is not None else "none"
    strays = workspace_strays(studies_dir(ctx))
    from myrmidon.client_entries import find_client_entries

    env_status = dict(ctx.env_script or {})
    clients = find_client_entries()
    env_note = ""
    if env_status.get("state") in ("missing", "failed"):
        env_note = (f"; env_script {env_status.get('path')} {env_status['state'].upper()}"
                    + (f" ({env_status.get('error')})" if env_status.get("error") else ""))
    broken = [c for c in clients if not c["command_found"]]
    client_note = (
        f"; {len(clients)} agent client entr{'y' if len(clients) == 1 else 'ies'}"
        + (f", {len(broken)} pointing at a missing command" if broken else "")
        if clients else "; no agent client wired at user scope (a project is wired when "
        "`myrmidon` starts there)"
    )
    return Observation(
        ok=True,
        summary=f"{len(engines)} engine(s) available ({', '.join(engines) or 'none'}), "
        f"backend: {backend}{env_note}{client_note}"
        + (
            f"; {len(strays)} stray entr{'y' if len(strays) == 1 else 'ies'} under "
            f"studies_root (not workspaces, not {SCRATCH_DIRNAME}/): {strays}"
            if strays else ""
        ),
        engines=engines,
        missing=missing,
        backend=backend,
        workspace_strays=strays,
        env_script=env_status,
        clients=clients,
    )
