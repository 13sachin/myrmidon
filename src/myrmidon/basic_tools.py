"""Study, structure-fetch, validation, input-set and doctor tools."""

from __future__ import annotations

import json
import re
from datetime import UTC
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import (
    EngineReport,
    FileCheckObservation,
    InputSetApplication,
    InputSetHit,
    InputSetObservation,
    InputSetSource,
    NextAction,
    Observation,
    SessionSummary,
    StructureObservation,
    ToolListObservation,
    WorkspaceObservation,
)
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning, decide
from myrmidon.core.workspace import META_FILENAME, Workspace
from myrmidon.errors import WorkspaceError

ACTIVE_POINTER = ".active"  # under studies_root: path of the last opened workspace
SCRATCH_DIRNAME = "_scratch"  # under studies_root: disposable test/smoke/debug runs
SESSIONS_DIRNAME = ".sessions"  # under studies_root: native agent sessions (agent/session.py)


def workspace_strays(studies_root: Path) -> list[str]:
    """Entries under studies_root that are neither workspaces nor the
    `_scratch` area — the tmp_* dirs and hand-made trees that a long campaign
    accumulates until nobody can tell a study from a scratch directory."""
    if not studies_root.is_dir():
        return []
    from myrmidon.core.workspace import META_FILENAME

    strays = []
    for child in sorted(studies_root.iterdir()):
        if child.name in (ACTIVE_POINTER, SCRATCH_DIRNAME, SESSIONS_DIRNAME):
            continue
        if child.is_dir() and (child / META_FILENAME).is_file():
            continue
        strays.append(child.name)
    return strays


def _studies_root(ctx: ToolContext) -> Path:
    root = Path(ctx.config.studies_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    return root


def _workspace_root(ctx: ToolContext, name: str) -> Path:
    return _studies_root(ctx) / name


def is_project(directory: Path) -> bool:
    """Whether this directory has been made a myrmidon project (`init-project`), for any
    client: its own settings file, a server entry, or the guidance block in the file any
    registered client reads. A goose project has only `AGENTS.md` and `.myrmidon.toml`,
    so checking `CLAUDE.md` alone is not enough."""
    from myrmidon.server.client_registry import REGISTRY

    try:
        if (directory / ".myrmidon.toml").is_file():
            return True
        entry = directory / ".mcp.json"
        if entry.is_file() and '"myrmidon"' in entry.read_text():
            return True
        for guidance in {d.guidance_file for d in REGISTRY.values()}:
            f = directory / guidance
            if f.is_file() and "myrmidon:begin" in f.read_text():
                return True
        return False
    except OSError:
        return False


def project_dir(start: Path | None = None) -> Path:
    """The project a directory belongs to: the nearest directory at or above it that
    `init-project` has touched, else the directory itself. A study is opened from a
    subdirectory as often as from the project root, and both are the same project —
    the config loader walks up for the same reason."""
    start = (start or Path.cwd()).resolve()
    for directory in (start, *start.parents):
        if is_project(directory):
            return directory
    return start


def _other_project(origin: str, cwd: Path | None = None) -> str:
    """The project a study was created from, when that is not the caller's — else "".

    An empty answer means "not somebody else's": either the study is this project's, or
    it carries no origin and nothing can be said.
    """
    if not origin:
        return ""
    try:
        return "" if Path(origin).resolve() == project_dir(cwd) else origin
    except OSError:
        return ""


def _free_name(root: Path, name: str) -> str:
    """The first `<name>-N` that is nobody's."""
    n = 2
    while (root / f"{name}-{n}").exists():
        n += 1
    return f"{name}-{n}"


def _mark_active(ctx: ToolContext, ws: Workspace) -> None:
    """Remember the active workspace so `myrmidon summary` / `statusline` can
    find it from another process (best effort).

    The pointer records *where* it was written as well as what it points at. One
    studies_root serves every project on the machine, so without the origin the
    first session in a brand-new project would be told about whatever study was last
    touched anywhere, and to collect that study's jobs first.
    """
    try:
        root = _studies_root(ctx)
        root.mkdir(parents=True, exist_ok=True)
        (root / ACTIVE_POINTER).write_text(
            json.dumps({"workspace": str(ws.root), "origin": str(Path.cwd())}) + "\n"
        )
    except OSError:
        pass


def _read_pointer(pointer: Path) -> tuple[Path, Path | None] | None:
    """(workspace, origin) from the pointer file. A bare-path pointer carries no origin,
    and says nothing about where it came from."""
    try:
        text = pointer.read_text().strip()
    except OSError:
        return None
    if not text:
        return None
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        origin = data.get("origin")
        return Path(data["workspace"]), Path(origin) if origin else None
    return Path(text), None


def active_workspace_path(studies_root: Path,
                          origin: Path | None = None) -> Path | None:
    """The workspace last opened, or None.

    ``origin`` is the directory the caller is working in. When it is a myrmidon project
    and the pointer was written somewhere else, the answer is None: that workspace is not
    this project's, and inheriting it would have this session work in another study.
    Whether the other directory still exists, or still looks like a project, does not
    enter into it: a deleted origin is still not this project.

    A caller outside any project — someone running `myrmidon summary` from their home
    directory — still gets the last workspace, which is the best answer available there,
    and so does a bare-path pointer that carries no origin.
    """
    read = _read_pointer(studies_root / ACTIVE_POINTER)
    if read is None:
        return None
    path, written_in = read
    if not path.is_dir():
        return None
    if (origin is not None and written_in is not None
            and origin.resolve() != written_in.resolve()
            and is_project(origin)):
        return None
    return path


def _stage_names(ws) -> list[str]:
    """The stage directories, with the ones that hold nothing marked.

    A call the gate refused and a job that crashed both leave a numbered directory
    behind, and a later session cannot tell those from a stage that produced the file
    it is looking for. They are not removed: their paths are in the ledger, and `report`
    and `replay` read them.
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


def _ws_observation(ctx: ToolContext, summary: str, ok: bool = True) -> WorkspaceObservation:
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
    jobs_db = ws.state_dir / "jobs.sqlite"
    if jobs_db.exists():
        from myrmidon.engines.gromacs.results import pending_lines
        from myrmidon.execution.jobstore import JobStore

        jobs = pending_lines(ctx)  # polls the backend, so the states below are current
        records = JobStore(jobs_db).list()
        pending = [r.job_id for r in records if not r.state.terminal]
        ended = [r.job_id for r in records
                 if r.state.terminal and not r.meta.get("collected")]
        done = len(records) - len(pending) - len(ended)
    if ended:
        summary += f" — {len(ended)} job(s) ended and are not collected yet"
    elif pending:
        summary += f" — {len(pending)} job(s) still running"
    elif jobs_db.exists() and done:
        # A study whose jobs have all finished and been collected is not the thing to
        # continue by default; the pointer outlives the work, so the summary says so.
        summary += (f" — nothing running, {done} job(s) done and collected"
                    f"{_since(events)}. Continue here, or study_create a new study?")
    n_artifacts = len({a.sha256 for ev in events for a in ev.outputs})
    return WorkspaceObservation(
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
    returns=WorkspaceObservation,
    tags={"workspace"},
    needs_workspace=False,
)
def study_create(ctx: ToolContext, params: WorkspaceCreateParams) -> WorkspaceObservation:
    root = _workspace_root(ctx, params.name)
    if (root / META_FILENAME).is_file():
        return _name_is_taken(ctx, root, params.name)
    previous = ctx.workspace
    ws = Workspace.create(root, params.name, params.description, origin=project_dir())
    ctx.workspace = ws
    _mark_active(ctx, ws)
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


def _name_is_taken(ctx: ToolContext, root: Path, name: str) -> WorkspaceObservation:
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
    free = _free_name(_studies_root(ctx), name)
    theirs = _other_project(origin)
    if theirs:
        whose = (f"It belongs to another project ({theirs}), not this one; open it only "
                 f"if you meant that project's study ({root}).")
    elif origin:
        whose = (f"It is this project's own earlier study ({root}) — study_open that "
                 f"path to continue it.")
    else:
        whose = (f"Nothing records which project made it ({root}), so it may be another "
                 f"project's; open it only if you recognise it.")
    return WorkspaceObservation(
        ok=False,
        summary=(f"the name '{name}' is taken under {_studies_root(ctx)}. For a new "
                 f"study here, study_create name='{free}'. {whose}"),
    )


class WorkspaceOpenParams(BaseModel):
    path: str = Field(default="", description="path to an existing myrmidon workspace; "
                      "empty: re-read the active one (its stages, artifact and event "
                      "counts, pending jobs)")


@tool(
    "study_open",
    description="Open an existing study and make it active — or, with no path, report "
    "the active one: its question, stages, artifact and event counts, pending jobs.",
    returns=WorkspaceObservation,
    tags={"workspace"},
    needs_workspace=False,
)
def study_open(ctx: ToolContext, params: WorkspaceOpenParams) -> WorkspaceObservation:
    if not params.path:
        if ctx.workspace is None:
            # A server process starts with no workspace — the MCP server builds its
            # context before anyone has said which study this is — but the workspace the
            # last session worked in is named on disk. Without this, the second session
            # of every study would begin by being told there is nothing.
            active = active_workspace_path(_studies_root(ctx), Path.cwd())
            if active is None:
                return _no_study(ctx)
            ctx.workspace = Workspace.open(active)
            return _ws_observation(
                ctx, f"Active study: {ctx.workspace.root}{_whose(ctx.workspace)}")
        return _ws_observation(ctx, "Study status")
    ws = Workspace.open(Path(params.path))
    ctx.workspace = ws
    _mark_active(ctx, ws)
    return _ws_observation(ctx, f"Study active: {ws.root}{_whose(ws)}")


def _no_study(ctx: ToolContext) -> WorkspaceObservation:
    """No study is open: what this project already has, and that a new one reads the
    project's files. A bare "no active study: pass a path" leaves a model believing the
    researcher's files live in some workspace only the researcher can name; it stops, or
    guesses paths."""
    project = project_dir()
    earlier = []
    try:
        for meta in sorted(_studies_root(ctx).glob(f"*/{META_FILENAME}")):
            if not _other_project(json.loads(meta.read_text()).get("origin", "")):
                earlier.append(meta.parent)
    except (OSError, ValueError):
        pass
    summary = (f"No study is open. A new one reads this project's files: a relative path "
               f"resolves in the study, then in the project directory {project}. "
               "study_create starts one for this question.")
    if earlier:
        summary += (" This project's earlier studies: "
                    + ", ".join(str(p) for p in earlier[:5])
                    + " — study_open with a path continues one.")
    return WorkspaceObservation(
        ok=False, summary=summary,
        next_actions=[NextAction(kind="call", tool="study_create",
                                 missing=["name", "description"],
                                 reason="start a study for this question; the project's "
                                 "files stay readable from it")]
        + [NextAction(kind="call", tool="study_open", known_parameters={"path": str(p)},
                      reason="continue this project's earlier study") for p in earlier[:5]],
    )


def _whose(ws: Workspace) -> str:
    """" (created from ..., which is not this project)" — or "" when it is this
    project's, or carries no origin. Opening a study by path says where it came from, so
    a session cannot work in another project's study without being told."""
    theirs = _other_project(ws.origin)
    return f" (created from {theirs}, which is not this project)" if theirs else ""


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
    "gate refusals, acknowledged warnings, jobs with core-hours "
    "booked and consumed, artifacts, scripts run (with git commits). Counted from the "
    "ledger and job store — work done outside the tools is absent. Use it to close a "
    "session or answer 'what did myrmidon actually do'.",
    returns=SessionSummary,
    tags={"workspace", "diagnostics"},
)
def session_summary(ctx: ToolContext, params: SessionSummaryParams) -> SessionSummary:
    from myrmidon.core.summary import summarize

    return summarize(
        ctx.workspace, params.since, backend=ctx.backend, compute=ctx.config.compute,
    )


class FileWriteParams(BaseModel):
    path: str = Field(description="workspace-relative path, e.g. '02_run/em.mdp'")
    content: str = Field(description="full file content")
    materialize: bool = Field(
        default=False,
        description="if the path goes through a symlink, replace that link with a private "
        "real copy before writing (never write through a link into shared files)",
    )


@tool(
    "file_write",
    description="Write a text file inside the active workspace (e.g. an .mdp you "
    "authored). The file is hashed into provenance. Paths outside the workspace are "
    "refused.",
    returns=Observation,
    tags={"workspace", "files"},
)
def file_write(ctx: ToolContext, params: FileWriteParams) -> Observation:
    root = ctx.workspace.root
    target = (root / params.path).resolve()
    if not str(target).startswith(str(root) + "/"):
        return Observation(
            ok=False, summary=f"refused: '{params.path}' escapes the workspace"
        )
    link = symlink_component(root, params.path)
    if link is not None:
        real = link.resolve()
        if not params.materialize:
            return Observation(
                ok=False,
                summary=f"refused: '{params.path}' goes through the symlink "
                f"{ctx.workspace.relpath(link)} -> {real}; writing would change the shared "
                "target for everything that links it (a force-field itp zeroed this way "
                "invalidates every fit that links it). Write to a real copy instead — "
                "materialize=true replaces the link with a private copy first.",
            )
        materialized = materialize_symlink(link)
        target = (root / params.path).resolve()
    else:
        materialized = None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(params.content)
    role = "mdp" if target.suffix == ".mdp" else "file"
    note = (
        f" (replaced symlink {ctx.workspace.relpath(link)} with a private copy of "
        f"{materialized})" if materialized else ""
    )
    return Observation(
        ok=True,
        summary=f"Wrote {ctx.workspace.relpath(target)} "
        f"({len(params.content)} chars){note}",
        artifacts=[ArtifactRef.from_path(target, role, root=root)],
    )


def symlink_component(root: Path, relpath: str) -> Path | None:
    """The first path component under ``root`` that is a symlink, else None —
    the check every write must make before it follows a link somewhere shared."""
    current = root
    for part in Path(relpath).parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def materialize_symlink(link: Path) -> Path:
    """Replace ``link`` with a real copy of what it points to (a directory tree
    is copied with symlinks=False). Returns the former target."""
    import shutil

    real = link.resolve()
    link.unlink()
    if real.is_dir():
        shutil.copytree(real, link, symlinks=False)
    else:
        shutil.copy2(real, link)
    return real


class FileReadParams(BaseModel):
    path: str = Field(
        description="the path as the researcher gave it: relative to the study or to "
        "the project directory it was made from, or absolute inside either; a directory "
        "lists what is in it, and an empty path lists the study")
    max_chars: int = Field(default=8000, ge=1, le=100_000)


@tool(
    "file_read",
    navigation=True,
    description="Read a text file (inputs, mdps, small logs), truncated at max_chars — "
    "or pass a directory to list what is in it, which is how to find files without "
    "guessing their names. A relative path is looked up in the study, then in the "
    "project directory the study was made from, where the researcher's own files are; "
    "an absolute path is read if it is in one of the two. Read-only.",
    returns=Observation,
    tags={"workspace", "files"},
)
def file_read(ctx: ToolContext, params: FileReadParams) -> Observation:
    root = ctx.workspace.root
    # The researcher's files are in the project directory, never in a new study, and the
    # study's own root ('', '.') is readable too. Nothing beyond the two: what is read
    # goes to the model's server, and the machine's secrets are files too.
    target = ctx.workspace.resolve(params.path).resolve()
    allowed = [root.resolve()] + ([Path(ctx.workspace.origin).resolve()]
                                  if ctx.workspace.origin else [])
    inside = next((a for a in allowed if target.is_relative_to(a)), None)
    # a study made from $HOME has the home directory as its project: hidden paths there
    # (~/.config/myrmidon/secrets.env, ~/.ssh) are not the researcher's inputs
    if inside is not None and inside != allowed[0] and any(
            part.startswith(".") for part in target.relative_to(inside).parts):
        inside = None
    if inside is None:
        return Observation(
            ok=False,
            summary=f"refused: '{params.path}' is outside {_searched(ctx)}, or hidden; "
                    "file_read looks only there. Ask the researcher to copy the file into "
                    "the project directory.",
        )
    where = _where(ctx, target)
    if target.is_dir():
        # A directory is the question "what is in here"; an agent with no listing guesses
        # file names, or writes a script instead of calling the tool it needs. Listing is
        # a look, like reading.
        entries = sorted(target.iterdir(), key=lambda q: (q.is_dir(), q.name))
        lines = [f"{q.name}/" if q.is_dir() else f"{q.name}  ({q.stat().st_size} bytes)"
                 for q in entries]
        rel = (params.path.rstrip("/") or ".") + where
        return Observation(
            ok=True, kind="observe",
            summary=(f"{rel}/ holds {len(lines)} entr{'y' if len(lines) == 1 else 'ies'}:\n"
                     + "\n".join(lines) if lines else f"{rel}/ is empty"),
            next_actions=_parsers_for(entries, root.resolve(), ctx.workspace.ledger) or None,
        )
    if not target.is_file():
        return Observation(
            ok=False,
            summary=f"file not found: {params.path} (looked in {_searched(ctx)}) — "
                    "file_read on the directory that should hold it lists what is there",
        )
    text = target.read_text(errors="replace")
    truncated = len(text) > params.max_chars
    return Observation(
        ok=True,
        summary=(f"{target}{where}:\n" if where else "")
        + ("[truncated]\n" if truncated else "") + text[: params.max_chars],
        # a look at a file, like file_check: it consumes nothing and produces nothing,
        # so it is an observation and never a step of the study
        kind="observe",
    )


def _where(ctx: ToolContext, target: Path) -> str:
    """Said beside a path that is not in the study, so the model passes on a path that
    resolves (the absolute one) rather than the one that found it."""
    if target.is_relative_to(ctx.workspace.root.resolve()):
        return ""
    return f" (in the project directory {ctx.workspace.origin}, not in the study)"


def _searched(ctx: ToolContext) -> str:
    origin = ctx.workspace.origin
    return (f"the study {ctx.workspace.root} and the project directory {origin}"
            if origin else f"the study {ctx.workspace.root}")


def _parsers_for(entries: list[Path], root: Path, ledger=None,
                 limit: int = 5) -> list[NextAction]:
    """For each listed file some registered tool parses, the call that parses it — unless
    the ledger shows that tool already read these very bytes. A stale offer is a loop: a
    model that has diagnosed a log and lists its directory again is offered the same call
    again."""
    from myrmidon.core.artifacts import sha256_file
    from myrmidon.core.tool import REGISTRY

    read = set()  # (tool, sha256) of every file a tool has consumed in this study
    if ledger is not None:
        read = {(ev.tool, ref.sha256) for ev in ledger.events() for ref in ev.inputs}
    actions = []
    for path in entries:
        if not path.is_file():
            continue
        for tool_name, reader in REGISTRY.readers(path):
            if read and (tool_name, sha256_file(path)) in read:
                continue
            # a file outside the study is offered by its absolute path, which resolves
            # the same for every tool
            rel = (str(path.relative_to(root)) if path.is_relative_to(root)
                   else str(path))
            actions.append(NextAction(
                kind="call", tool=tool_name, known_parameters={reader.param: rel},
                reason=f"{path.name} is {reader.what}; {tool_name} parses it"))
            if len(actions) >= limit:
                return actions
    return actions


class FetchPdbParams(BaseModel):
    pdb_id: str = Field(description="4-character RCSB PDB id, e.g. 1L2Y")


@tool(
    "fetch_pdb",
    description="Download a structure from RCSB PDB into the workspace; reports chains "
    "and heteroatom residues parsed from the file.",
    returns=StructureObservation,
    tags={"knowledge", "structures"},
)
def fetch_pdb(ctx: ToolContext, params: FetchPdbParams) -> StructureObservation:
    pdb_id = params.pdb_id.strip().lower()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        return StructureObservation(
            ok=False, summary=f"'{params.pdb_id}' is not a valid 4-character PDB id",
            source=f"rcsb:{params.pdb_id}",
        )
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        resp = httpx.get(url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return StructureObservation(
            ok=False, summary=f"RCSB fetch failed for {pdb_id}: {exc}",
            source=f"rcsb:{pdb_id}",
        )
    stage = ctx.workspace.stage("fetch")
    path = stage / f"{pdb_id}.pdb"
    path.write_bytes(resp.content)
    chains: set[str] = set()
    hets: set[str] = set()
    n_atoms = 0
    for line in resp.text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            n_atoms += 1
            chains.add(line[21].strip() or "?")
        if line.startswith("HETATM"):
            res = line[17:20].strip()
            if res != "HOH":
                hets.add(res)
    return StructureObservation(
        ok=True,
        summary=f"Fetched {pdb_id.upper()}: {n_atoms} atoms, chains "
        f"{sorted(chains)}, non-water heteroatoms {sorted(hets) or 'none'}",
        source=f"rcsb:{pdb_id}",
        artifacts=[ArtifactRef.from_path(path, "structure", root=ctx.workspace.root)],
        n_atoms=n_atoms,
        chains=sorted(chains),
        het_residues=sorted(hets),
    )


class FileCheckParams(BaseModel):
    path: str = Field(description="the engine input file (.mdp, a CPMD input, or any file "
                      "an input set rendered): relative to the study or its project "
                      "directory, or absolute")
    tpr: str = Field(default="", description="for a CPMD/MiMiC input: the .tpr it will run "
                     "with, so the QM/MM coupling can be checked against it")


@tool(
    "file_check",
    description="One engine input file, checked and explained, without running anything: "
    "its kind's gate (an .mdp's sanity rules; a CPMD/MiMiC input's coupling, charge and "
    "multiplicity), where every value came from, and what changed by hand since it was "
    "rendered — on the study's declared axis it is yours, otherwise it is a finding to "
    "acknowledge. Use it before running an edited file, and to answer 'where did this "
    "number come from'. A look, not a step.",
    returns=FileCheckObservation,
    tags={"files", "validation", "knowledge"},
)
def file_check(ctx: ToolContext, params: FileCheckParams) -> FileCheckObservation:
    from myrmidon.core.file_origin import file_origin
    from myrmidon.core.file_origin import render as render_origin
    from myrmidon.engines.inputs import resolve_in_ws

    if not ctx.workspace.resolve(params.path).is_file():
        return FileCheckObservation(
            ok=False, summary=f"no file at {params.path} in {_searched(ctx)}",
            path=params.path,
        )
    real = resolve_in_ws(ctx, params.path)
    head = real.read_text(errors="replace")[:4000]
    if real.suffix.lower() == ".mdp":
        kind = "mdp"
    elif real.suffix.lower() == ".inp" or "&CPMD" in head or "&MIMIC" in head:
        kind = "cpmd"
    else:
        kind = "other"

    warnings: list[ValidationWarning] = []
    checked: list[str] = []
    if kind == "mdp":
        from myrmidon.engines.gromacs.mdp import parse_mdp_file, validate_mdp

        warnings, checked = validate_mdp(parse_mdp_file(real))
    elif kind == "cpmd":
        from myrmidon.engines.mimic.launch import MimicRunParams, mimic_gate

        probe = MimicRunParams(cpmd_input=params.path, tpr=params.tpr or params.path)
        warnings, checked = mimic_gate(ctx, probe)
    origin = file_origin(ctx, params.path)
    known = {w.id for w in warnings}
    warnings += [w for w in origin.findings if w.id not in known]
    checked = [*checked, "parameter_source"]
    decision = decide(warnings, [], checked)
    label = {"mdp": "MDP", "cpmd": "MiMiC", "other": "file"}[kind]
    obs = FileCheckObservation(
        ok=not decision.blocking,
        summary=f"{label} check of {params.path}: {len(decision.blocking)} blocking, "
        f"{len(decision.unacknowledged)} warning(s), {len(decision.advisory)} advisory "
        f"across: {', '.join(checked)}\n" + render_origin(origin),
        path=params.path,
        file_kind=kind,
        blocking=decision.blocking,
        unacknowledged=decision.unacknowledged,
        advisory=decision.advisory,
        checked=checked,
        provenance=origin.provenance,
        sourced=origin.sourced,
        inputset=origin.inputset,
        keys=[k.model_dump() for k in origin.keys],
        yours=[k.key for k in origin.yours],
        edited=origin.edited,
    )
    obs.kind = "observe"  # a look at a file: counted apart, never replayed, never a node
    return obs


def get_protocol_store(ctx: ToolContext):
    from myrmidon.knowledge.inputsets import InputSetStore

    if ctx.inputsets is not None:
        return ctx.inputsets
    return InputSetStore.default()


class ProtocolLookupParams(BaseModel):
    query: str = Field(
        default="",
        description="what you need parameters for, e.g. 'NVT protein'. Leave empty to "
        "list the whole store — the answer to 'what inputsets do I have?'",
    )


def _adjustable(inputset) -> tuple[dict, dict]:
    """(rationale, derivable) — why each value is what it is, and how to obtain the ones
    the input set refuses to supply.

    There is no third map of pre-approved "tunable" keys: an input set's author, who has
    never seen the study, cannot know which parameter the researcher came to vary, so that
    permission lives where the experiment is declared — ``Arms.varies`` on the plan.
    """
    rationale, derivable = {}, {}
    for key, spec in inputset.parameters.items():
        if spec.rationale:
            rationale[key] = spec.rationale
        if spec.derivable:
            derivable[key] = spec.derivable
    return rationale, derivable


def _protocol_hit(inputset) -> InputSetHit:
    """Project a InputSet onto the wire shape: the values it supplies, the ones it
    refuses to supply and how to obtain them, and why each is what it is."""
    rationale, derivable = _adjustable(inputset)
    required = {
        key: spec.derivable or "ask the user"
        for key, spec in inputset.unsupplied().items()
    }
    return InputSetHit(
        id=inputset.id,
        title=inputset.title,
        applicability=inputset.applicability,
        system_class=inputset.system_class,
        scope=inputset.scope,
        layer=inputset.layer,
        version=inputset.version,
        renders=inputset.renders,
        parameters=inputset.values(),
        required_from_user=required,
        rationale=rationale,
        derivable=derivable,
        sources=[InputSetSource(**s.model_dump()) for s in inputset.sources],
        citations=inputset.citations,
        pitfalls=inputset.pitfalls,
        validated_by=inputset.validated_by,
        notes=inputset.notes,
    )


@tool(
    "inputset_lookup",
    navigation=True,
    description="Search the curated inputset store, or list all of it with an empty "
    "query: validated, source-bearing parameter recipes, layered engine < site < "
    "project. This is the ONLY sanctioned source of 'default' parameters — cite the "
    "inputset id, version and its sources when you use one. Parameters under "
    "required_from_user are NOT supplied: obtain and confirm them, never fill them in "
    "yourself. Every value carries its rationale; changing one is a deviation the study's "
    "declared experiment either covers or the caller acknowledges.",
    returns=InputSetObservation,
    tags={"knowledge", "inputsets"},
    needs_workspace=False,
)
def inputset_lookup(ctx: ToolContext, params: ProtocolLookupParams):
    store = get_protocol_store(ctx)
    # An empty query lists everything: "what inputsets do I have?" is the first question
    # anyone asks a parameter store.
    listing = not params.query.strip()
    hits = sorted(store.all(), key=lambda p: p.id) if listing else store.search(params.query)
    if listing:
        by_layer = ", ".join(f"{n} {layer}" for layer, n in sorted(store.by_layer().items()))
        summary = f"the whole store: {len(hits)} inputset(s) ({by_layer})"
    else:
        summary = f"{len(hits)} inputset(s) matched" + (
            f"; best: {hits[0].id} ({hits[0].layer})" if hits else
            " — call with no query to list the whole store"
        )
    return InputSetObservation(
        ok=True, summary=summary, inputsets=[_protocol_hit(p) for p in hits]
    )


class InputSetApplyParams(BaseModel):
    inputset: str = Field(description="inputset id, from inputset_lookup")
    output: str | None = Field(
        default=None,
        description="workspace-relative path to write, e.g. '02_run/nvt.mdp'. Omit it to "
        "resolve the values only: you get every value with its origin and its sources, and "
        "nothing is written. That is how to read an input set that renders no file.",
    )
    stage: str | None = Field(
        default=None,
        description="stage name for a multi-stage inputset; omit for a single-stage one",
    )
    overrides: dict = Field(
        default_factory=dict,
        description="parameters to set yourself. Each is recorded as origin 'user'. "
        "Changing a value the input set states requires acknowledging "
        "INPUTSET_VALUE_OVERRIDDEN; adding a key the input set does not mention is free. "
        "This is also how you supply what the input set lists as required_from_user.",
    )
    materialize: bool = Field(
        default=False,
        description="if the output path goes through a symlink, replace that link with a "
        "private real copy before writing",
    )
    acknowledge_warnings: list[str] = Field(default_factory=list)


def _resolve_protocol(ctx: ToolContext, params: InputSetApplyParams):
    """(inputset, resolution, findings) — findings non-empty means do not render."""
    from myrmidon.knowledge.inputsets import resolve

    store = get_protocol_store(ctx)
    try:
        inputset = store.get(params.inputset)
    except KeyError:
        known = ", ".join(sorted(p.id for p in store.all())) or "none"
        return None, None, [
            ValidationWarning(
                id="INPUTSET_NOT_FOUND",
                severity="block",
                message=f"no inputset {params.inputset!r} in any layer; known: {known}",
                suggestion="find one with inputset_lookup",
            )
        ]
    plan = _workspace_plan(ctx)
    axis = plan.arms.varies if plan is not None and plan.arms is not None else None
    return inputset, resolve(inputset, params.overrides, declared_axis=axis), []


_RENDERERS = {".mdp": "mdp", ".inp": "cpmd_inp", ".dat": "fminput"}


def _render_kind(resolution, output: str | None) -> str | None:
    """What to render: the input set says so, else the output suffix does.

    `None` means nothing is rendered — either because no output was asked for (the values
    are the answer) or because the input set declares `renders: none`.
    """
    if output is None or resolution.renders == "none":
        return None
    kind = resolution.renders or _RENDERERS.get(Path(output).suffix)
    return kind if kind in ("mdp", "cpmd_inp", "fminput") else None


def _sidecar_values(ctx: ToolContext) -> dict[str, dict]:
    """InputSet id -> the values already rendered from it in this workspace, read back
    from the sidecars. How a rule about two different files sees the other one."""
    known: dict[str, dict] = {}
    root = getattr(ctx.workspace, "root", None)
    if root is None:
        return known
    for path in sorted(Path(root).rglob("*.myrmidon.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if "inputset" in data and "values" in data:
            known[data["inputset"]] = data["values"]
    return known


def _workspace_plan(ctx: ToolContext):
    """The study's plan, if it has one — written by `plan_write`.

    A plan carries the rules that relate values *different steps* render — the GROMACS
    output pace against the CPMD force-matching pace — and the axis the study declared it
    varies. Neither can live on an input set, which knows only its own parameter set, so
    without a plan neither is checked.
    """
    from myrmidon.plan_tools import read_plan

    return read_plan(ctx)


def _check_invariants(ctx: ToolContext, inputset, resolution):
    """The input set's own rules, plus any the study's plan declares across steps."""
    from myrmidon.knowledge.inputsets import evaluate_invariants

    results = []
    if inputset.invariants:
        results += evaluate_invariants(inputset.invariants, {"": resolution.values})

    plan = _workspace_plan(ctx)
    if plan is not None and plan.invariants:
        rendered = _sidecar_values(ctx)
        rendered[inputset.id] = resolution.values  # the one about to be written
        by_step = {}
        for step in plan.steps:
            if not step.cites:
                continue
            pid = step.cites.split("@")[0].removeprefix("inputset:")
            if pid in rendered:
                by_step[step.name] = rendered[pid]
        results += plan.check_invariants(by_step)
    return results


def inputset_apply_gate(ctx: ToolContext, params: InputSetApplyParams):
    warnings: list[ValidationWarning] = []
    checked = ["protocol_exists"]
    inputset, resolution, findings = _resolve_protocol(ctx, params)
    if findings:
        return findings, checked
    checked.append("required_parameters")
    for key, how in resolution.missing.items():
        warnings.append(
            ValidationWarning(
                id="INPUTSET_REQUIRED_PARAM_MISSING",
                severity="block",
                message=f"{inputset.id} does not supply {key!r} and it was not given: "
                f"{how}",
                assumption=f"rendering without {key} would leave the engine to default it",
                suggestion=f"confirm {key} and pass it in overrides",
            )
        )
    checked.append("renderer")
    kind = _render_kind(resolution, params.output)
    if kind is None and params.output is not None:
        # An input set that renders no file cannot be written into one, and the message says
        # what to do instead rather than what is missing.
        if resolution.renders == "none":
            message = (
                f"{inputset.id} renders no file — it states values. Call inputset_apply "
                "without `output` to resolve them, with their origins and sources."
            )
        else:
            message = (
                "inputset_apply renders .mdp and CPMD input files; declare the inputset's "
                "`renders` or give the output an .mdp/.inp suffix. Force-matching inputs "
                "go through fm_plan."
            )
        warnings.append(
            ValidationWarning(
                id="INPUTSET_RENDERER_UNSUPPORTED", severity="block", message=message
            )
        )
    checked.append("invariants")
    for result in _check_invariants(ctx, inputset, resolution):
        if result.status == "violated":
            warnings.append(
                ValidationWarning(
                    id=result.id,
                    severity="block",
                    message=f"{result.message} ({result.detail})",
                )
            )
        elif result.status == "deferred":
            warnings.append(
                ValidationWarning(
                    id=result.id,
                    severity="info",
                    message=f"{result.message} — cannot be checked yet: {result.detail}",
                )
            )
    checked.append("overrides")
    deviations = {
        k: v for k, v in resolution.overrides.items() if k not in resolution.on_axis
    }
    if deviations:
        detail = "; ".join(f"{k}: {v['inputset']} -> {v['user']}" for k, v in deviations.items())
        warnings.append(
            ValidationWarning(
                id="INPUTSET_VALUE_OVERRIDDEN",
                severity="warn",
                message=f"{len(deviations)} validated value(s) of {inputset.id} replaced: "
                + detail,
                assumption="the inputset's validated value would be silently replaced",
                suggestion="acknowledge only if you have a reason, and state it",
            )
        )
    if resolution.on_axis:
        warnings.append(
            ValidationWarning(
                id="INPUTSET_VALUE_ON_DECLARED_AXIS",
                severity="info",
                message=f"{inputset.id}: changed along the experiment this study declared "
                + "; ".join(
                    f"{k}: {resolution.overrides[k]['inputset']} -> "
                    f"{resolution.overrides[k]['user']}"
                    for k in resolution.on_axis
                ),
            )
        )
    return warnings, checked


@tool(
    "inputset_apply",
    description="Render an inputset into a real run-parameter file in the workspace, "
    "recording where every single value came from (inputset id + version, or you). "
    "Use this instead of writing an .mdp by hand: a file written by hand has no "
    "recorded source for its numbers. The origin map is written to a sidecar next to "
    "the file and lands in the provenance ledger with the artifact hash.",
    returns=InputSetApplication,
    tags={"knowledge", "inputsets", "files"},
    gate=inputset_apply_gate,
)
def inputset_apply(ctx: ToolContext, params: InputSetApplyParams) -> InputSetApplication:
    from myrmidon.engines.gromacs.mdp import render_mdp
    from myrmidon.engines.mimic.cpmd_input import render_cpmd_from_values
    from myrmidon.engines.mimic.fm import render_fminput

    inputset, resolution, findings = _resolve_protocol(ctx, params)
    if findings:  # the gate already refused; belt and braces
        return InputSetApplication(ok=False, summary=findings[0].message)

    root = ctx.workspace.root
    if params.output is None:
        # Resolve only: the values with their origins are the answer, and nothing is
        # written. An input set whose `renders` is "none" has no other way to be read —
        # a box clearance or a QM padding is a value a tool takes as a parameter, not a
        # line in a file — and reading one is still a call the ledger records.
        invariants = _check_invariants(ctx, inputset, resolution)
        from_inputset = sum(
            1 for v in resolution.origin.values() if v.startswith("inputset:")
        )
        return InputSetApplication(
            ok=True,
            summary=f"{inputset.id} v{inputset.version} resolves to "
            f"{len(resolution.values)} value(s), {from_inputset} from the input set; "
            "nothing written.",
            pitfalls=inputset.pitfalls,
            inputset=inputset.id,
            protocol_version=inputset.version,
            renders=inputset.renders,
            invariants=[r.model_dump() for r in invariants],
            values=resolution.values,
            origin=resolution.origin,
            overrides=resolution.overrides,
            additions=resolution.additions,
            on_axis=resolution.on_axis,
            sources=[InputSetSource(**s.model_dump()) for s in resolution.sources],
        )

    target = (root / params.output).resolve()
    if not str(target).startswith(str(root) + "/"):
        return InputSetApplication(
            ok=False, summary=f"refused: '{params.output}' escapes the workspace"
        )
    link = symlink_component(root, params.output)
    materialized = None
    if link is not None:
        if not params.materialize:
            return InputSetApplication(
                ok=False,
                summary=f"refused: '{params.output}' goes through the symlink "
                f"{ctx.workspace.relpath(link)} -> {link.resolve()}; writing would change "
                "the shared target. materialize=true replaces the link with a private copy.",
            )
        materialized = materialize_symlink(link)
        target = (root / params.output).resolve()

    label = f"{inputset.id} v{inputset.version}"
    header = [
        "written by myrmidon inputset_apply — do not edit by hand",
        f"inputset: {label}",
        "each value names its origin; the machine-readable map is in "
        f"{target.name}.myrmidon.json",
    ]
    kind = _render_kind(resolution, params.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "cpmd_inp":
        # an input set supplies the method, never the system: &MIMIC PATHS/BOX/OVERLAPS
        # and &ATOMS are written per system by mimicpy's prepqm, so this is a template
        target.write_text(render_cpmd_from_values(resolution.values, header))
    elif kind == "fminput":
        # a force-matching recipe: the ~40 settings mimicpy fm reads. A campaign's arms
        # are typically one key apart, and the rendered file records which key and why
        target.write_text(render_fminput(resolution.values, resolution.origin, header))
    else:
        target.write_text(render_mdp(resolution.values, resolution.origin, header))

    invariants = _check_invariants(ctx, inputset, resolution)
    sidecar = target.with_name(target.name + ".myrmidon.json")
    sidecar.write_text(
        json.dumps(
            {
                "inputset": inputset.id,
                "protocol_version": inputset.version,
                "renders": kind,
                "on_axis": resolution.on_axis,
                "values": resolution.values,
                "origin": resolution.origin,
                "overrides": resolution.overrides,
                "invariants": [r.model_dump() for r in invariants],
                "sources": [s.model_dump(exclude_none=True) for s in resolution.sources],
                "written_by": "inputset_apply",
            },
            indent=1,
            sort_keys=True,
        )
    )

    from_protocol = sum(1 for v in resolution.origin.values() if v.startswith("inputset:"))
    from_user = len(resolution.origin) - from_protocol
    note = (
        f" (replaced symlink {ctx.workspace.relpath(link)} with a private copy of "
        f"{materialized})" if materialized else ""
    )
    # An input set names the failures that shaped it, so they arrive when the recipe is
    # chosen rather than after the run has crashed. Each is a finding id a gate or
    # diagnostic raises with its full text when it fires.
    traps = (
        f" Known ways this recipe fails: {', '.join(inputset.pitfalls)}."
        if inputset.pitfalls else ""
    )
    return InputSetApplication(
        ok=True,
        summary=f"Wrote {ctx.workspace.relpath(target)} from {label}: "
        f"{from_protocol} value(s) from the input set, {from_user} from you{note}.{traps}",
        pitfalls=inputset.pitfalls,
        inputset=inputset.id,
        protocol_version=inputset.version,
        renders=kind,
        invariants=[r.model_dump() for r in invariants],
        values=resolution.values,
        origin=resolution.origin,
        overrides=resolution.overrides,
        additions=resolution.additions,
        on_axis=resolution.on_axis,
        sources=[InputSetSource(**s.model_dump()) for s in resolution.sources],
        sidecar=ctx.workspace.relpath(sidecar),
        artifacts=[
            ArtifactRef.from_path(target, kind, root=root),
            ArtifactRef.from_path(sidecar, "provenance", root=root),
        ],
    )


class ToolLookupParams(BaseModel):
    query: str = Field(
        description="what you are trying to do, in your own words — 'compare candidate "
        "force fields against a QM/MM reference', 'pick frames from a trajectory'")
    limit: int = Field(default=6, description="how many tools to name")


@tool(
    "tool_lookup",
    navigation=True,
    description="Which tool does this job — including tools that exist but are not "
    "available on this machine because an engine they need is not installed. Ask before "
    "writing a script that computes something an engine could compute: a tool that is "
    "missing here looks exactly like a tool that does not exist, and a script that "
    "reimplements an engine produces numbers nobody can trace.",
    returns=ToolListObservation,
    tags={"knowledge"},
    needs_workspace=False,
)
def tool_lookup(ctx: ToolContext, params: ToolLookupParams) -> ToolListObservation:
    """Keyword overlap against every registered tool's name, description and tags.

    The model is shown the tools whose engines this machine has, and a tool whose engine
    is absent is not in its list. An agent that cannot see `fm_validate` and is asked
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
    return ToolListObservation(ok=bool(hits), summary=summary, shown=shown, hidden=hidden)


@tool(
    "doctor",
    navigation=True,
    description="Report what simulation engines myrmidon finds on this machine, their "
    "versions, what is unavailable and how to fix it, and the active execution backend.",
    returns=EngineReport,
    tags={"diagnostics"},
    needs_workspace=False,
)
def doctor(ctx: ToolContext, params: EmptyParams) -> EngineReport:
    engines = {}
    missing = {}
    for name, info in ctx.engines.items():
        if info.available:
            engines[name] = info.version
        else:
            missing[name] = info.hint
    backend = ctx.backend.name if ctx.backend is not None else "none"
    strays = workspace_strays(_studies_root(ctx))
    from myrmidon.server.clients import find_client_entries

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
        if clients else "; no agent client wired (`myrmidon mcp install`)"
    )
    return EngineReport(
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


# imported for side effects: registers all tools with the default registry
def register_all() -> None:
    """Import every module that defines tools (idempotent per process)."""
    import myrmidon.analysis.convergence  # noqa: F401
    import myrmidon.analysis.energy  # noqa: F401
    import myrmidon.analysis.trajectory  # noqa: F401
    import myrmidon.engines.gromacs.index  # noqa: F401
    import myrmidon.engines.gromacs.prep  # noqa: F401
    import myrmidon.engines.gromacs.results  # noqa: F401
    import myrmidon.engines.gromacs.run  # noqa: F401
    import myrmidon.engines.mimic.cpmd_input  # noqa: F401
    import myrmidon.engines.mimic.diagnostics  # noqa: F401
    import myrmidon.engines.mimic.fm  # noqa: F401
    import myrmidon.engines.mimic.fm_report  # noqa: F401
    import myrmidon.engines.mimic.launch  # noqa: F401
    import myrmidon.engines.mimic.qm_region  # noqa: F401
    import myrmidon.engines.mimic.validate  # noqa: F401
    import myrmidon.knowledge.lookup  # noqa: F401
    import myrmidon.knowledge.search  # noqa: F401
    import myrmidon.loop_tools  # noqa: F401
    import myrmidon.plan_tools  # noqa: F401
    import myrmidon.scripts.tools  # noqa: F401
    import myrmidon.viz.tools  # noqa: F401

