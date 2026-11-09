"""Study tools: create a study, open one, and say where it stands."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.observation import (
    Observation,
    ToolError,
)
from myrmidon.core.studies import (
    active_workspace,
    free_name,
    mark_active,
    other_project,
    project_dir,
    project_studies,
    resolve_study_path,
    studies_dir,
    study_dir,
    study_lines,
)
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.workspace import META_FILENAME, Workspace
from myrmidon.errors import WorkspaceError
from myrmidon.execution.collect import pending_lines
from myrmidon.execution.jobs import existing_jobstore


def _stage_names(ws) -> list[str]:
    """The stage directories, with the ones that hold nothing marked.

    A call the gate refused and a job that crashed both leave a numbered directory
    behind, and a later session cannot tell those from a stage that produced the file
    it is looking for. They are not removed: their paths are in the ledger, and `report`
    reads them.
    """
    out = []
    for name in ws.stages():
        empty = not any((ws.root / name).iterdir())
        out.append(f"{name} (empty)" if empty else name)
    return out


def _since(events: list) -> str:
    """", last call 3 days ago" — how long this workspace has been quiet, or "" when the
    ledger says nothing useful."""
    from datetime import datetime

    if not events:
        return ""
    last = getattr(events[-1], "timestamp", None)
    if isinstance(last, str):  # a ledger read back from JSON
        try:
            last = datetime.fromisoformat(last)
        except ValueError:
            return ""
    if not isinstance(last, datetime):
        return ""
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    seconds = (datetime.now(UTC) - last).total_seconds()
    if seconds < 90:
        return ", last call just now"
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if seconds >= size:
            n = int(seconds // size)
            return f", last call {n} {unit}{'s' if n > 1 else ''} ago"
    return ""


def _ws_observation(ctx: ToolContext, summary: str, ok: bool = True) -> Observation:
    ws = ctx.workspace
    events = list(ws.ledger.events())
    # what this study was asking. A session that comes back to a directory of files has
    # to reconstruct the question from them; the study recorded it at the start.
    try:
        question = str(ws.meta.get("description", "")).strip()
    except (OSError, ValueError):
        question = ""
    if question and not summary.startswith("Study created"):
        summary += '\n  question: "' + question + '"'
    pending: list[str] = []
    jobs: list[str] = []
    ended: list[str] = []
    store = existing_jobstore(ws)
    if store is not None:
        jobs = pending_lines(ctx)  # polls the backend, so the states below are current
        records = store.list()
        pending = [r.job_id for r in records if not r.state.terminal]
        ended = [r.job_id for r in records
                 if r.state.terminal and not r.meta.get("collected")]
        done = len(records) - len(pending) - len(ended)
    if ended:
        summary += f" — {len(ended)} job(s) ended and are not collected yet"
    elif pending:
        summary += f" — {len(pending)} job(s) still running"
    elif store is not None and done:
        # A study whose jobs have all finished and been collected is not the thing to
        # continue by default; the pointer outlives the work, so the summary says so.
        summary += (f" — nothing running, {done} job(s) done and collected"
                    f"{_since(events)}. Continue here, or study_create a new study?")
    n_artifacts = len({a.sha256 for ev in events for a in ev.outputs})
    return Observation(
        ok=ok,
        summary=summary,
        workspace=str(ws.root),
        stages=_stage_names(ws),
        n_artifacts=n_artifacts,
        n_events=len(events),
        jobs=jobs,
    )


class WorkspaceCreateParams(BaseModel):
    name: str = Field(description="short study name (becomes the directory name)")
    description: str = Field(
        description="the question this study answers, in the user's own words — not a "
        "label. It is what `report` and a later session are shown, and a study whose "
        "question is recorded can be told apart from one that merely has files in it")


@tool(
    "study_create",
    description="Start a study: one question, its own directory and provenance ledger, "
    "made active. Every subsequent tool call is recorded in its ledger. Not needed when "
    "a study is already active for this question (study_open with no path shows it): a "
    "new study starts empty, and paths then resolve inside it.",
    tags={"workspace"},
    needs_workspace=False,
)
def study_create(ctx: ToolContext, params: WorkspaceCreateParams) -> Observation:
    root = study_dir(ctx, params.name)
    if (root / META_FILENAME).is_file():
        return _name_is_taken(ctx, root, params.name)
    previous = ctx.workspace
    ws = Workspace.create(root, params.name, params.description, origin=project_dir())
    ctx.workspace = ws
    mark_active(ctx, ws)
    return _ws_observation(ctx, f"Study created and active: {ws.root}"
                           + _left_behind(previous))


def _left_behind(previous: Workspace | None) -> str:
    """What the study that was active holds, said when a new one replaces it. A study
    created while the one holding the inputs is active makes every later relative path
    resolve in the empty new one, so the caller is told where the files stayed."""
    if previous is None:
        return ""
    held = []
    for entry in sorted(previous.root.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            files = sorted(f.name for f in entry.iterdir() if not f.name.startswith("."))
            if files:
                more = f", +{len(files) - 4} more" if len(files) > 4 else ""
                held.append(f"{entry.name}/ ({', '.join(files[:4])}{more})")
    if not held:
        return ""
    return (f". The previously active study {previous.root} is unchanged and holds "
            f"{'; '.join(held)} — none of it is in this new study, and relative paths now "
            f"resolve here. To work on those files, study_open {previous.root}")


def _name_is_taken(ctx: ToolContext, root: Path, name: str) -> Observation:
    """The answer when the study name is already a directory under studies_root.

    For another project's study a bare path is an invitation: a model told to "pass a
    path" and handed one with its own task's name on it opens it and makes it this
    project's active study. So the free name comes first, and whose study it is comes
    before where it is.
    """
    try:
        origin = Workspace.open(root).origin
    except WorkspaceError:
        origin = ""
    free = free_name(studies_dir(ctx), name)
    theirs = other_project(origin)
    if theirs:
        whose = (f"It belongs to another project ({theirs}), not this one; open it only "
                 f"if you meant that project's study ({root}).")
    elif origin:
        whose = (f"It is this project's own earlier study ({root}) — study_open that "
                 f"path to continue it.")
    else:
        whose = (f"Nothing records which project made it ({root}), so it may be another "
                 f"project's; open it only if you recognise it.")
    return Observation(
        ok=False,
        summary=(f"the name '{name}' is taken under {studies_dir(ctx)}. For a new "
                 f"study here, study_create name='{free}'. {whose}"),
    )


class WorkspaceOpenParams(BaseModel):
    path: str = Field(default="", description="an existing study: its path, or its name "
                      "under the studies root; empty: re-read the active one (its stages, "
                      "artifact and event counts, pending jobs)")


@tool(
    "study_open",
    description="Open an existing study and make it active — or, with no path, report "
    "the active one: its question, stages, artifact and event counts, pending jobs.",
    tags={"workspace"},
    needs_workspace=False,
)
def study_open(ctx: ToolContext, params: WorkspaceOpenParams) -> Observation:
    if not params.path:
        if ctx.workspace is None:
            # A server process starts with no workspace — the MCP server builds its
            # context before anyone has said which study this is — but the workspace the
            # last session worked in is named on disk. Without this, the second session
            # of every study would begin by being told there is nothing.
            active, note = active_workspace(studies_dir(ctx), Path.cwd())
            if active is None:
                obs = _no_study(ctx)
                if note:
                    obs.summary = f"{note}. {obs.summary}"
                return obs
            ctx.workspace = active
            return _with_studies(ctx, _ws_observation(
                ctx, f"Active study: {ctx.workspace.root}{_whose(ctx.workspace)}"))
        return _with_studies(ctx, _ws_observation(ctx, "Study status"))
    ws = Workspace.open(resolve_study_path(studies_dir(ctx), params.path))
    ctx.workspace = ws
    mark_active(ctx, ws)
    return _ws_observation(ctx, f"Study active: {ws.root}{_whose(ws)}")


def _with_studies(ctx: ToolContext, obs: Observation) -> Observation:
    """The observation with what this project's other studies asked and wrote, each output by
    the ``<study>/<path>`` this study reads it by."""
    ws = ctx.workspace
    studies = [s for s in project_studies(ws.root.parent, ws.origin) if s["name"] != ws.root.name]
    obs.studies = studies
    lines = study_lines(studies)
    if lines:
        obs.summary += ("\n  this project's other studies, each output named as this study "
                        "reads it:\n  " + "\n  ".join(lines))
    return obs


def _no_study(ctx: ToolContext) -> Observation:
    """No study is open: what this project already has, and that a new one reads the
    project's files. A bare "no active study: pass a path" leaves a model believing the
    researcher's files live in some workspace only the researcher can name; it stops, or
    guesses paths."""
    project = project_dir()
    earlier = project_studies(studies_dir(ctx), str(project))
    summary = (f"No study is open. A new one reads this project's files: a relative path "
               f"resolves in the study, then in the project directory {project}, then as "
               "<study>/<path> in this project's other studies. study_create starts one for "
               "this question.")
    if earlier:
        summary += (" This project's earlier studies, under " + str(studies_dir(ctx))
                    + " (study_open with a name continues one):\n  "
                    + "\n  ".join(study_lines(earlier)))
    return Observation(
        ok=False, summary=summary, studies=earlier,
        error=ToolError(code="NO_STUDY", message="no study is open",
                        hint="study_create with a name and description for this question"
                        + ("; study_open with one of the paths above continues it"
                           if earlier else "")))


def _whose(ws: Workspace) -> str:
    """" (created from ..., which is not this project)" — or "" when it is this
    project's, or carries no origin. Opening a study by path says where it came from, so
    a session cannot work in another project's study without being told."""
    theirs = other_project(ws.origin)
    return f" (created from {theirs}, which is not this project)" if theirs else ""
