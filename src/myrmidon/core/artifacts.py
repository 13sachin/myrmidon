"""Artifact references: every file a tool produces or consumes, hashed."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from pydantic import BaseModel


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


class ArtifactRef(BaseModel):
    """A hashed pointer to a file in (or under) the workspace."""

    path: str  # workspace-relative when possible, else absolute
    role: str  # e.g. "structure", "topology", "mdp", "tpr", "trajectory", "log"
    sha256: str
    size_bytes: int

    @classmethod
    def from_path(cls, path: Path, role: str, root: Path | None = None) -> ArtifactRef:
        path = path.resolve()
        rel: str
        if root is not None:
            try:
                rel = str(path.relative_to(root.resolve()))
            except ValueError:
                rel = str(path)
        else:
            rel = str(path)
        return cls(
            path=rel,
            role=role,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )


#: workspace subdirectories whose files are harness state, not study inputs
_NOT_STUDY_DATA = (".myrmidon",)
_MAX_PATH_CHARS = 4096
#: (abs path, mtime_ns, size) -> sha256. A study hashes the same trajectory once per
#: process however many nodes consume it.
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_cached(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = sha256_file(path)
    return _HASH_CACHE[key]


def _ref_under(root: Path, rel: str, role: str) -> ArtifactRef:
    """An ArtifactRef for a workspace file, keeping the workspace-relative path even
    when it is a symlink into a research tree (``resolve()`` would point outside)."""
    real = root / rel
    return ArtifactRef(
        path=rel, role=role, sha256=sha256_cached(real), size_bytes=real.stat().st_size
    )


def consumed_inputs(params: dict, root: Path, origin: Path | str = "") -> list[ArtifactRef]:
    """Every parameter value that names an existing file under the workspace, hashed.

    The registry calls this before a tool runs, so a tool never has to remember to
    record what it consumed: the data-flow edge "this grompp consumed that
    inputset_apply's mdp" becomes a hashed record instead of a path string inside
    ``params`` that a reader has to match against earlier outputs.

    The role is where the value sat in the params: ``topology``, ``frames[3]``,
    ``extra_files.index``. A file in the project directory the study was made from
    (``origin``) is the researcher's own input and is recorded by its absolute path, as
    tools resolve it (``Workspace.resolve``). Other paths outside the study
    (a GROMACS library file) and harness state are not study inputs and are skipped.
    """
    root = root.resolve()
    project = Path(origin).resolve() if origin else None
    found: dict[str, ArtifactRef] = {}

    def visit(value, role: str) -> None:
        if isinstance(value, str):
            rel = _relative_file(value, root)
            if rel is not None and rel not in found:
                try:
                    found[rel] = _ref_under(root, rel, role)
                except OSError:
                    return
            elif rel is None and project is not None:
                there = _project_file(value, root, project)
                if there is not None and there not in found:
                    try:
                        found[there] = ArtifactRef.from_path(Path(there), role)
                    except OSError:
                        return
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, f"{role}[{i}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{role}.{key}" if role else str(key))

    visit(params, "")
    return list(found.values())


def _project_file(value: str, root: Path, project: Path) -> str | None:
    """The absolute path of the project-directory file this string names, if it names
    one: absolute under the project, or relative and absent from the study."""
    if not value or len(value) > _MAX_PATH_CHARS or "\n" in value or "\x00" in value:
        return None
    try:
        p = Path(value).expanduser()
        if not p.is_absolute():
            if (root / p).exists():
                return None
            p = project / p
        p = Path(os.path.abspath(p))
        if not p.is_file() or not p.resolve().is_relative_to(project):
            return None
    except (OSError, ValueError):
        return None
    return str(p)


def _relative_file(value: str, root: Path) -> str | None:
    """The workspace-relative path this string names, if it names a file in the
    workspace. Relative values are workspace-relative, as every tool resolves them."""
    if not value or len(value) > _MAX_PATH_CHARS or "\n" in value or "\x00" in value:
        return None
    try:
        p = Path(value)
        absolute = p if p.is_absolute() else root / p
        # normalise '..' without resolving symlinks: a planted symlink into a research
        # tree is still a file of this workspace
        absolute = Path(os.path.abspath(absolute))
        if not absolute.is_file():
            return None
        if absolute.is_relative_to(root):
            rel = absolute.relative_to(root)
        else:
            # the value may name the workspace through a symlinked prefix (a site's
            # scratch mount, typically): the resolved path still has to land inside it
            resolved = absolute.resolve()
            if not resolved.is_relative_to(root):
                return None
            rel = resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    if rel.parts and rel.parts[0] in _NOT_STUDY_DATA:
        return None
    return str(rel)
