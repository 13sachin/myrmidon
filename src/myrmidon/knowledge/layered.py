"""A layered YAML store: the shape input sets and protocols share.

Three layers, read in order and merged by id — the package's own pack (``engine``), the
machine's directory beside its config (``site``), and the project's (``project``); a later
layer overrides an earlier one. Any layer may be absent. A file that does not claim a scope
is claimed by the layer it sits in, and a file that does not parse names itself rather than
vanishing. What differs between the two stores is only the model, the directory names and
which fields a search weighs, so a subclass states those and nothing else.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import ClassVar, Generic, Literal, TypeVar

import yaml
from pydantic import BaseModel

Scope = Literal["engine", "site", "project"]

T = TypeVar("T", bound=BaseModel)


#: path -> (mtime_ns, size, parsed YAML). A lookup reads every layer, and a file check
#: looks up its input set once per key: without the cache one `file_check` re-parses the
#: whole store hundreds of times and takes seconds. An edited file is read again.
_PARSED: dict[Path, tuple[int, int, object]] = {}


def _parsed(path: Path):
    st = path.stat()
    hit = _PARSED.get(path)
    if hit is None or hit[:2] != (st.st_mtime_ns, st.st_size):
        hit = (st.st_mtime_ns, st.st_size, yaml.safe_load(path.read_text()))
        _PARSED[path] = hit
    return copy.deepcopy(hit[2])


class LayeredStore(Generic[T]):
    model: ClassVar[type[BaseModel]]
    kind: ClassVar[str]  # how a missing id is named: "no inputset 'x'"
    dirname: ClassVar[str]  # of the package pack, the site directory and the project's
    #: a query with no usable term lists everything (protocols) or nothing (input sets)
    empty_query_lists_all: ClassVar[bool] = False

    def __init__(self, path: Path | None = None, site_path: Path | None = None,
                 builtin_path: Path | None = None):
        self.builtin_path = Path(builtin_path) if builtin_path else self.builtin_dir()
        self.site_path = Path(site_path).expanduser() if site_path else None
        self.path = Path(path) if path else None

    @classmethod
    def builtin_dir(cls) -> Path:
        return Path(__file__).parent / cls.dirname

    @classmethod
    def site_dir(cls) -> Path:
        return Path("~/.config/myrmidon") / cls.dirname

    @classmethod
    def default(cls, project_dir: Path | None = None):
        """The package's pack, the site directory next to the machine config, and the
        store's directory in the project (cwd)."""
        root = Path(project_dir or Path.cwd())
        return cls(path=root / cls.dirname, site_path=cls.site_dir())

    @classmethod
    def with_layers(cls, layers: list[Scope], project_dir: Path | None = None):
        """:meth:`default` reduced to the layers named; the package's pack is always read."""
        root = Path(project_dir or Path.cwd())
        return cls(path=root / cls.dirname if "project" in layers else None,
                   site_path=cls.site_dir() if "site" in layers else None)

    @classmethod
    def builtin(cls):
        """Only the pack shipped in the package — no site or project layer."""
        return cls()

    @property
    def layers(self) -> list[tuple[Scope, Path]]:
        out: list[tuple[Scope, Path]] = [("engine", self.builtin_path)]
        if self.site_path is not None:
            out.append(("site", self.site_path))
        if self.path is not None:
            out.append(("project", self.path))
        return out

    def _read(self, directory: Path, layer: Scope) -> list[T]:
        if not directory.is_dir():
            return []
        found = []
        for path in sorted(directory.glob("*.yaml")):
            try:
                item = self.model.model_validate(_parsed(path))
            except Exception as exc:  # a broken file must name itself, not vanish
                raise ValueError(f"{path}: {exc}") from exc
            item.layer = layer
            if "scope" not in item.model_fields_set:
                item.scope = layer
            found.append(item)
        return found

    def all(self) -> list[T]:
        """Every item from every layer; later layers override earlier ones by id."""
        merged: dict[str, T] = {}
        for layer, directory in self.layers:
            for item in self._read(directory, layer):
                merged[item.id] = item
        return list(merged.values())

    def by_layer(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.all():
            key = item.layer or "project"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get(self, item_id: str) -> T:
        items = self.all()
        for item in items:
            if item.id == item_id:
                return item
        # an id it was published under before: a study that cited it still resolves, and
        # its version check says whether the values are the ones it was written against
        for item in items:
            if item_id in getattr(item, "aliases", ()):
                return item
        raise KeyError(f"no {self.kind} {item_id!r}")

    # what an item IS outweighs what it mentions
    def primary_text(self, item: T) -> str:
        return f"{item.id} {item.title}"

    def prose_text(self, item: T) -> str:
        return f"{item.applicability} {getattr(item, 'system_class', '')} {item.notes}"

    def search(self, query: str) -> list[T]:
        terms = {t for t in query.lower().split() if len(t) > 2}
        scored = []
        for item in self.all():
            primary, prose = self.primary_text(item).lower(), self.prose_text(item).lower()
            score = (sum(2 for t in terms if t in primary)
                     + sum(1 for t in terms if t in prose))
            if score or (self.empty_query_lists_all and not terms):
                scored.append((score, item))
        scored.sort(key=lambda pair: -pair[0])
        return [item for _, item in scored]
