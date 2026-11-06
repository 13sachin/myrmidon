"""Every key GROMACS and CPMD read and the engine's own default for it.

A catalog is generated from the engine (scripts/build_key_catalog.py) and never edited.
Which keys the harness reasons about is what its stage templates set
(knowledge/stage.py `decided_keys`); every other key is the engine's default, and an input
that sets one has stepped outside what the harness reasons about.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr

from myrmidon.knowledge.sources import Source

CATALOG_DIR = Path(__file__).parent / "catalog"

Engine = Literal["gromacs", "cpmd"]


class EngineKey(BaseModel):
    name: str  # as the engine spells it
    section: str = ""  # CPMD: the section, without the ampersand
    default: str | None = None  # the value the engine takes when the input is silent
    group: str = ""  # the heading it sits under in mdout.mdp
    unit: str = ""
    options: str = ""  # CPMD: the options the manual lists
    aliases: list[str] = Field(default_factory=list)  # CPMD: other spellings it accepts
    written: bool = True  # GROMACS: mdout.mdp writes it whatever the input says
    doc: str = ""


class Catalog(BaseModel):
    engine: Engine
    version: str
    sources: list[Source]
    obsolete: list[str] = Field(default_factory=list)
    renamed: dict[str, str] = Field(default_factory=dict)
    #: how a run of this engine divides among ranks and threads (engines/<engine>/layout.py),
    #: and how a QM step's cost scales: constants of the engine, not of a machine
    decomposition: dict[str, Any] = Field(default_factory=dict)
    keys: dict[str, EngineKey]
    _sections: dict[str, list[EngineKey]] | None = PrivateAttr(default=None)

    def get(self, key: str) -> EngineKey | None:
        return self.keys.get(canonical(key) if self.engine == "gromacs" else key.upper())

    def keyword(self, section: str, line: str) -> EngineKey | None:
        """The CPMD keyword a line of `section` sets, matched the way CPMD's parser
        matches: every word of the keyword (or of one of its spellings) is on the line,
        in any order. The keyword with the most words wins, so TEMPERATURE RAMP is
        TEMPERATURE and TIMESTEP IONS is TIMESTEP IONS."""
        words = set(line.upper().split())
        best, size = None, 0
        if self._sections is None:
            self._sections = {}
            for key in self.keys.values():
                self._sections.setdefault(key.section, []).append(key)
        for key in self._sections.get(section.upper().lstrip("&"), ()):
            for spelling in (key.name, *key.aliases):
                need = set(spelling.split())
                if need <= words and len(need) > size:
                    best, size = key, len(need)
        return best


class SystemFact(BaseModel):
    """A key's value read from the system's own files: what decided it, from what."""

    key: str  # the engine key it decides
    value: str
    rule: str  # the function that read it
    read: list[str] = Field(default_factory=list)  # the files it read
    sha256: str = ""  # of what it read, in order
    # the value split by the molecule each part belongs to, for a fact a stage takes for
    # some molecules only (the restraints of the subsystems it holds)
    parts: dict[str, str] = Field(default_factory=dict)

    @property
    def origin(self) -> str:
        return f"system:{self.rule}"


def canonical(key: str) -> str:
    """One spelling of a GROMACS key: grompp treats '-' and '_' alike, and case too."""
    return str(key).strip().lower().replace("-", "_")


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", version)[:2]) or (0,)


def shipped_versions(engine: Engine) -> list[str]:
    return sorted((p.stem.split("-", 1)[1] for p in CATALOG_DIR.glob(f"{engine}-*.yaml")),
                  key=_version_tuple)


def nearest_version(engine: Engine, version: str | None) -> str:
    """The shipped catalog for `version`: itself, else the newest of the same major
    release, else the newest shipped."""
    shipped = shipped_versions(engine)
    if not shipped:
        raise FileNotFoundError(f"no {engine} key catalog ships in {CATALOG_DIR}")
    if version in shipped:
        return version
    major = _version_tuple(version or "")[0]
    same = [v for v in shipped if _version_tuple(v)[0] == major]
    return (same or shipped)[-1]


@cache
def load(engine: Engine, version: str | None = None) -> Catalog:
    """The catalog nearest `version`."""
    chosen = nearest_version(engine, version)
    data = yaml.safe_load((CATALOG_DIR / f"{engine}-{chosen}.yaml").read_text())
    return Catalog.model_validate(data)



def for_engine(engines: dict, engine: Engine) -> Catalog:
    """The catalog for the engine this machine discovered (its version, or the nearest
    shipped)."""
    info = engines.get(engine)
    return load(engine, getattr(info, "version", None))
