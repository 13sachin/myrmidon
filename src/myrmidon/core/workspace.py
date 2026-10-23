"""Workspace: one directory per study, with numbered stages and a ledger."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from myrmidon.core.provenance import LEDGER_FILENAME, ProvenanceLedger
from myrmidon.errors import WorkspaceError

META_FILENAME = "myrmidon.workspace.json"
#: the one workspace and ledger format this code reads and writes. A study whose manifest
#: names no format or another one is refused, with a way to read it, never migrated: there
#: is one current format.
FORMAT = 2
STATE_DIRNAME = ".myrmidon"  # job store, run scratch, `run` transcripts — not study artifacts
#: a stage directory: its number (two digits at least, more from the 100th on) and name
STAGE_RE = re.compile(r"^(\d{2,})_(.+)$")


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        if not (self.root / META_FILENAME).is_file():
            raise WorkspaceError(f"{self.root} is not a myrmidon workspace")
        found = self.meta.get("format")
        if found != FORMAT:
            made = "without a format" if found is None else f"in format {found}"
            raise WorkspaceError(
                f"{self.root} was made {made}; this myrmidon reads format {FORMAT} only. "
                "Read it with the myrmidon version it was made by, or start a new study."
            )
        self.ledger = ProvenanceLedger(self.root / LEDGER_FILENAME)

    @classmethod
    def create(cls, root: Path, name: str, description: str = "",
               origin: Path | str | None = None) -> Workspace:
        root = root.resolve()
        if (root / META_FILENAME).exists():
            raise WorkspaceError(f"workspace already exists at {root}")
        root.mkdir(parents=True, exist_ok=True)
        (root / STATE_DIRNAME).mkdir(exist_ok=True)
        meta = {
            "format": FORMAT,
            "name": name,
            "description": description,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        # Which project made it. One studies_root serves every project on the machine, so
        # without this a second project that asks for a name its task suggests is handed
        # the path of a stranger's study. The manifest carries it rather than the ledger,
        # because the question is asked about a study before anything of it has been read.
        if origin is not None:
            meta["origin"] = str(Path(origin).resolve())
        (root / META_FILENAME).write_text(json.dumps(meta, indent=2) + "\n")
        (root / LEDGER_FILENAME).touch()
        return cls(root)

    @classmethod
    def open(cls, root: Path) -> Workspace:
        return cls(root)

    @property
    def meta(self) -> dict:
        return json.loads((self.root / META_FILENAME).read_text())

    @property
    def origin(self) -> str:
        """The directory this study was created from, or "" when the manifest does not
        record one."""
        try:
            return str(self.meta.get("origin", ""))
        except (OSError, ValueError):
            return ""

    def resolve(self, path: str | Path) -> Path:
        """Where a path a tool is given points: an absolute one as given; a relative one
        under the study, or — when nothing is there — under the directory the study was
        made from, which is where the researcher who typed it is; a new study is empty,
        and the researcher's own files are not in it."""
        p = Path(path).expanduser()
        if p.is_absolute():
            return p
        here = self.root / p
        if not here.exists() and self.origin:
            there = Path(self.origin) / p
            if there.exists():
                return there
        return here

    @property
    def state_dir(self) -> Path:
        d = self.root / STATE_DIRNAME
        d.mkdir(exist_ok=True)
        return d

    def stages(self) -> list[str]:
        found = [(int(m.group(1)), child.name) for child in self.root.iterdir()
                 if child.is_dir() and (m := STAGE_RE.match(child.name))]
        return [name for _, name in sorted(found)]

    def stage(self, name: str) -> Path:
        """Return the stage directory for `name`, creating and numbering it if new."""
        # dots are kept: stage names such as "sp1.select" read as they were given
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "stage"
        for existing in self.stages():
            m = STAGE_RE.match(existing)
            # the bare name, or the numbered directory name handed back (agents do)
            if m and (m.group(2) == safe or existing == safe):
                return self.root / existing
        # one past the highest, so a removed stage never hands its number out twice
        n = max((int(STAGE_RE.match(st).group(1)) for st in self.stages()), default=0) + 1
        path = self.root / f"{n:02d}_{safe}"
        path.mkdir()
        return path

    def run_dir(self, stage: Path, label: str) -> Path:
        """A fresh, numbered directory under ``stage`` for one submission: ``01_<label>``,
        ``02_<label>``, … in the order they were made, and never handed out twice (mkdir
        fails rather than reuse one). An engine that writes fixed file names — CPMD's
        ``RESTART.1``, ``LATEST``, ``GEOMETRY``, ``ENERGIES`` and its log — would otherwise
        overwrite the previous run's outputs in a shared stage, and the hashes the ledger
        recorded for them would name files that have been overwritten."""
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "run"
        taken = [int(m.group(1)) for child in stage.iterdir()
                 if child.is_dir() and (m := STAGE_RE.match(child.name))]
        path = stage / f"{max(taken, default=0) + 1:02d}_{safe}"
        path.mkdir()
        return path

    def relpath(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            return str(path)
