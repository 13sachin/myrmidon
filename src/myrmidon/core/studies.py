"""Where studies live: the studies root, the active-study pointer, and the project a study
was made from."""

from __future__ import annotations

import json
from pathlib import Path

from myrmidon.core.tool import ToolContext
from myrmidon.core.workspace import Workspace

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

    def holds_workspaces(directory: Path) -> bool:
        """A project's own studies root under the machine's: it holds workspaces, or the
        pointer to the one last opened there."""
        try:
            return (directory / ACTIVE_POINTER).is_file() or any(
                (child / META_FILENAME).is_file() for child in directory.iterdir()
                if child.is_dir())
        except OSError:
            return False

    strays = []
    for child in sorted(studies_root.iterdir()):
        if child.name in (ACTIVE_POINTER, SCRATCH_DIRNAME, SESSIONS_DIRNAME):
            continue
        if child.is_dir() and ((child / META_FILENAME).is_file() or holds_workspaces(child)):
            continue
        strays.append(child.name)
    return strays


def resolve_study_path(studies_root: Path, given: str) -> Path:
    """The study a person or a tool names: a path as given; a bare name as the directory of
    that name under the studies root, the way `study_create NAME` made it. A name that is
    also a directory here is the directory. One name means one place."""
    path = Path(given).expanduser()
    if path.is_dir() or "/" in given or path.is_absolute():
        return path
    under = studies_root / given
    return under if under.is_dir() else path


def studies_dir(ctx: ToolContext) -> Path:
    root = Path(ctx.config.studies_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    return root


def study_dir(ctx: ToolContext, name: str) -> Path:
    return studies_dir(ctx) / name


def is_project(directory: Path) -> bool:
    """Whether this directory has been made a myrmidon project (`init-project`), for any
    client: its own settings file, a server entry, or the guidance block in the file any
    registered client reads. A goose project has only `AGENTS.md` and `.myrmidon.toml`,
    so checking `CLAUDE.md` alone is not enough."""
    from myrmidon.client_registry import REGISTRY

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


def other_project(origin: str, cwd: Path | None = None) -> str:
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


def project_studies(studies_root: Path, origin: str, limit: int = 8,
                    exclude: str = "") -> list[dict]:
    """The studies of the project ``origin`` names under ``studies_root`` but ``exclude``,
    newest first, each
    ``{name, question, reported, outputs}``: its mandate's question (else the question it was
    created with), whether it has written its report, and what the latest collected run of
    each of its mandate's stages wrote, each ``{stage, role, path}`` by the role its collection
    recorded and the ``<study>/<path>`` a study of the same project reads it by
    (``Workspace.resolve``)."""
    from myrmidon.core.lineage import stage_run
    from myrmidon.core.mandate import read_mandate
    from myrmidon.core.workspace import META_FILENAME
    from myrmidon.errors import WorkspaceError

    if not origin:
        return []
    found = []
    try:
        metas = sorted(studies_root.glob(f"*/{META_FILENAME}"),
                       key=lambda m: m.stat().st_mtime, reverse=True)
    except OSError:
        return []
    for meta in metas:
        if meta.parent.name == exclude:
            continue
        try:
            theirs = json.loads(meta.read_text()).get("origin", "")
            if not theirs or Path(theirs).resolve() != Path(origin).resolve():
                continue
            ws = Workspace.open(meta.parent)
        except (OSError, ValueError, WorkspaceError):
            continue
        mandate = read_mandate(ws)
        outputs: list[dict] = []
        for stage in (mandate.method.stages if mandate is not None else []):
            run = stage_run(ws, stage.name)
            seen: set[str] = set()
            for art in (run[0] if run else []):
                if art.role in seen:
                    continue
                seen.add(art.role)
                path = art.path if Path(art.path).is_absolute() else f"{ws.root.name}/{art.path}"
                outputs.append({"stage": stage.name, "role": art.role, "path": path})
        found.append({
            "name": ws.root.name,
            "question": mandate.question if mandate is not None
            else str(ws.meta.get("description", "")),
            "reported": (ws.root / "report.json").is_file(),
            "outputs": outputs})
        if len(found) == limit:
            break
    return found


def study_lines(studies: list[dict]) -> list[str]:
    """Each study as one line, and a line per stage below it: the directory its latest run's
    outputs are in, and each output's role and name there."""
    import os

    lines = []
    for s in studies:
        lines.append(f"{s['name']}: \"{s['question']}\" — "
                     + ("reported" if s["reported"] else "no report yet"))
        stages: dict[str, list[dict]] = {}
        for o in s["outputs"]:
            stages.setdefault(o["stage"], []).append(o)
        for stage, outputs in stages.items():
            try:
                where = os.path.commonpath([os.path.dirname(o["path"]) for o in outputs])
            except ValueError:  # a study's file and one outside it: each named in full
                where = ""
            lines.append(f"    {stage}: " + (f"{where}/ — " if where else "") + ", ".join(
                f"{o['role']} {os.path.relpath(o['path'], where) if where else o['path']}"
                for o in outputs))
    return lines


def free_name(root: Path, name: str) -> str:
    """The first `<name>-N` that is nobody's."""
    n = 2
    while (root / f"{name}-{n}").exists():
        n += 1
    return f"{name}-{n}"


def mark_active(ctx: ToolContext, ws: Workspace) -> None:
    """Remember the active workspace so `myrmidon summary` / `statusline` can
    find it from another process (best effort).

    The pointer records *where* it was written as well as what it points at. One
    studies_root serves every project on the machine, so without the origin the
    first session in a brand-new project would be told about whatever study was last
    touched anywhere, and to collect that study's jobs first.
    """
    try:
        root = studies_dir(ctx)
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


def active_workspace(studies_root: Path, origin: Path | None = None
                     ) -> tuple[Workspace | None, str]:
    """The active workspace opened, or None with the reason: a pointer at a directory this
    version cannot read is "no active study" with a sentence, never a stop, so the command
    that would make or point at another one can run."""
    from myrmidon.errors import WorkspaceError

    path = active_workspace_path(studies_root, origin)
    if path is None:
        return None, ""
    try:
        return Workspace.open(path), ""
    except WorkspaceError as exc:
        return None, f"the active study {path} cannot be opened ({exc}); treating it as none"
