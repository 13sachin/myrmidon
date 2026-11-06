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
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

#: an engine -> the module holding its ``ADAPTER``; a coupled run's main program comes before
#: its partner, so the run's account is read from the main program's log
_ADAPTERS = {
    "cpmd": "myrmidon.engines.mimic.adapter",
    "gromacs": "myrmidon.engines.gromacs.adapter",
    "mimicpy": "myrmidon.engines.mimic.fit_adapter",
}


@dataclass(frozen=True)
class Layout:
    """How a run divides the cores it is given: MPI ranks and threads per rank of its main
    program, ranks of a coupled partner, and why it is so."""

    ntasks: int
    cpus_per_task: int = 1
    partner_ntasks: int = 0
    why: str = ""

    @property
    def cores(self) -> int:
        return (self.ntasks + self.partner_ntasks) * self.cpus_per_task


@dataclass
class StageJob:
    """A stage as its engine runs it: the job, the collector that reads it when it ends, and
    what the call that submits it records."""

    spec: object  # execution/backend.JobSpec
    kind: str
    label: str
    summary: str = ""
    meta: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)


@dataclass
class StageRender:
    """A mandate stage's compiled values on their way into the file its engine reads."""

    stage: object  # the mandate's Stage
    values: dict
    origin: dict
    header: list[str]
    #: what the system holds, by file kind
    system: dict[str, Path] = field(default_factory=dict)


@dataclass
class StageText:
    """What a stage's file holds, what its sidecar adds, and what was noticed writing it."""

    text: str
    sidecar: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)


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
    #: the kinds of input file it renders (knowledge/inputsets.RENDERED_FILES)
    renders: tuple[str, ...] = ()
    #: (values, origin of each, header lines) -> the file's text
    render: Callable[[dict, dict, list[str]], str] | None = None
    #: a stage's values -> its file, where the engine adds to them what the system holds
    stage_text: Callable[[StageRender], StageText] | None = None
    #: an engine key as a caller spells it -> whether this engine reads it
    reads_key: Callable[[str], bool] | None = None
    #: (ctx, the rendered file, the system's files by kind, the stage's directory, the stage's
    #: name, warnings allowed) -> an observation holding the run input it compiled to
    compile_run_input: Callable[..., object] | None = None
    #: (ctx, a file a stage reads) -> facts a render takes from it, by the key each decides
    input_facts: Callable[..., dict] | None = None
    #: the fixed names its logs go by: a run that died before its banner says so nowhere else
    log_names: tuple[str, ...] = ()
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
    #: (a stage's inputs by kind, the stage) -> the facts its layout is resolved from
    layout_facts: Callable[..., dict] | None = None
    #: (ctx, mandate, stage, the inputs prepared for it, a run asked to continue?) -> what else
    #: its run reads, by kind, from the run it starts from and the system
    stage_inputs: Callable[..., dict] | None = None
    #: (ctx, stage, inputs by kind, walltime) -> what stops or is worth saying of the run
    job_findings: Callable[..., list] | None = None
    #: (ctx, stage, inputs, layout, cores) -> (minutes, why) its run is booked for
    size_walltime: Callable[..., tuple] | None = None
    #: (ctx, stage, inputs by kind, layout, run request) -> the job, single, coupled or batch
    build_job: Callable[..., StageJob] | None = None
    #: (ctx, job record, state, exit code) -> the typed artifacts a finished run wrote
    collect: Mapping[str, Callable] | None = None
    #: (ctx, request) -> a validated reference dataset, merged from the runs named
    collect_dataset: Callable[..., object] | None = None
    #: (ctx, a log it wrote) -> its log read into typed facts, with whether the run is clean
    diagnose: Callable[..., object] | None = None


@cache
def adapters() -> tuple[EngineAdapter, ...]:
    """Every engine's adapter, in the table's order."""
    return tuple(importlib.import_module(module).ADAPTER for module in _ADAPTERS.values())


def adapter(engine: str) -> EngineAdapter:
    """The adapter of the engine named; KeyError for one this harness does not drive."""
    if engine not in _ADAPTERS:
        raise KeyError(engine)
    return importlib.import_module(_ADAPTERS[engine]).ADAPTER


def adapter_rendering(renders: str) -> EngineAdapter | None:
    """The adapter that renders this kind of input file; None for a kind none renders."""
    return next((a for a in adapters() if renders in a.renders), None)


def adapter_running(engines: list[str]) -> EngineAdapter | None:
    """The adapter that runs a stage told to these engines: a coupled stage's main program."""
    return next((a for a in adapters() if a.name in engines and a.build_job), None)


def adapters_reading(kinds: set[str] | Mapping[str, object]) -> Iterator[EngineAdapter]:
    """The adapters that read at least one of these file kinds."""
    return (a for a in adapters() if set(a.file_kinds) & set(kinds))


def adapter_of_log(text: str) -> EngineAdapter | None:
    """The adapter whose engine wrote this log text; None for a log no engine claims."""
    return next((a for a in adapters() if a.is_log_text and a.is_log_text(text)), None)


def adapter_of_file(path: Path) -> EngineAdapter | None:
    """The adapter whose engine wrote this file, by its name or what it says; None for a file
    no engine claims."""
    named = next((a for a in adapters() if path.name in a.log_names), None)
    return named or adapter_of_log(path.read_text(errors="replace"))


def fit_adapter() -> EngineAdapter | None:
    """The adapter that collects reference datasets for a fit; None where no engine does."""
    return next((a for a in adapters() if a.collect_dataset), None)
