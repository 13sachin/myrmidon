"""The engines the harness drives, one adapter each.

An adapter is everything engine-blind code asks of an engine: what a system's files of its
kinds hold, how many ranks and threads a run of it takes, what its log states, how a stage
becomes a job and what a finished run wrote. A new engine brings its own package under
``engines/``, its data under ``knowledge/``, and one row in :data:`_ADAPTERS`; nothing above
this layer names it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

#: an engine -> the module holding its ``ADAPTER``; a coupled run's main program comes before
#: its partner, so the run's account is read from the main program's log
_ADAPTERS = {
    "cpmd": "myrmidon.engines.mimic.adapter",
    "gromacs": "myrmidon.engines.gromacs.adapter",
}


@dataclass(frozen=True)
class EngineAdapter:
    """One engine as engine-blind code sees it. A capability it does not have stays None."""

    name: str
    #: the file kinds of a system this engine reads
    file_kinds: tuple[str, ...] = ()
    #: (files by kind, the engine's data directory, the rules to keep) -> facts by engine key
    system_facts: Callable[[Mapping[str, Path], Path | None, set[str] | None], dict] | None = None
    #: (files by kind, the engine's data directory) -> the system's parts by name
    system_parts: Callable[[Mapping[str, Path], Path | None], dict] | None = None
    #: a log's text -> whether this engine wrote it
    is_log_text: Callable[[str], bool] | None = None
    #: a file -> whether it is this engine's log, read from its head
    is_log: Callable[[Path], bool] | None = None
    #: a log's text -> what it states, by fact key
    run_facts: Callable[[str], dict] | None = None
    #: fact key -> (how a report labels it, its unit), in the order a report states them
    fact_labels: Mapping[str, tuple[str, str]] | None = None
    #: a job record -> the file its run wrote its account into; None when there is none
    run_log: Callable[[object], Path | None] | None = None
    #: (cores, system facts) -> the ranks and threads a run takes on them
    resolve_layout: Callable[..., object] | None = None
    #: (stage, compiled inputs, layout) -> the job, single, coupled or frame batch
    build_job: Callable[..., object] | None = None
    #: (a run's directory) -> the typed artifacts it wrote
    collect: Callable[..., object] | None = None


@cache
def adapters() -> tuple[EngineAdapter, ...]:
    """Every engine's adapter, in the table's order."""
    return tuple(importlib.import_module(module).ADAPTER for module in _ADAPTERS.values())


def adapter(engine: str) -> EngineAdapter:
    """The adapter of the engine named; KeyError for one this harness does not drive."""
    if engine not in _ADAPTERS:
        raise KeyError(engine)
    return importlib.import_module(_ADAPTERS[engine]).ADAPTER


def adapters_reading(kinds: set[str] | Mapping[str, object]) -> Iterator[EngineAdapter]:
    """The adapters that read at least one of these file kinds."""
    return (a for a in adapters() if set(a.file_kinds) & set(kinds))


def adapter_of_log(text: str) -> EngineAdapter | None:
    """The adapter whose engine wrote this log text; None for a log no engine claims."""
    return next((a for a in adapters() if a.is_log_text and a.is_log_text(text)), None)
