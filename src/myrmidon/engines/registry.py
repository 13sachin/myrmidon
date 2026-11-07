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
    artifacts: list = field(default_factory=list)


@dataclass
class DatasetRequest:
    """What a dataset is collected from, as the call and the ledger state it."""

    reference_files: list[str]
    trajectory: list[str]
    topology: str
    coords: str
    selection: str
    stage: str = "fm"
    check_frames: int = 20
    align_if_desynced: bool = True
    output_prefix: str = "merged"
    monitor_pairs: dict[str, list[int]] = field(default_factory=dict)
    run_logs: list[str] = field(default_factory=list)
    duplicate_rmsd_a: float = 0.05
    overlaps_input: str | None = None
    reference_file: str = ""
    output: str = "frame_report.csv"


@dataclass(frozen=True)
class FitReaders:
    """How a fitting engine's files read, for a report that names no engine: its log, its
    alignment check, its optimisation and validation results, and the topology it fits."""

    parse_fm_log: Callable[[str], dict]
    parse_fmcheck_report: Callable[[str], dict]
    parse_optimization_results: Callable[[str], dict]
    parse_validation_force_residuals: Callable[..., dict]
    #: the file of per-atom force residuals the fit writes beside its log
    residuals_file: str
    force_type_table: Callable[[dict], str]
    unphysical_force_constants: Callable[[str], list]
    parse_itp_atoms: Callable[[str], dict]
    parse_itp_bonded: Callable[[str], dict]
    #: a bonded section -> how many atoms a term of it names
    bonded_n_atoms: Mapping[str, int]


@dataclass
class InputCheck:
    """What an engine says of one of its input files as it is on disk."""

    kind: str  # the kind file_check reports it as
    label: str  # how a summary names the check
    findings: list
    checked: list[str]


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
    #: (ctx, request) -> matched snapshots cut from a run this engine's stage left
    cut_snapshots: Callable[..., object] | None = None
    #: (ctx, topology, structure, forces, dataset, selection, frames, label) -> a force comparison
    force_comparison: Callable[..., object] | None = None
    #: (a stage's inputs by kind, the stage) -> the facts its layout is resolved from
    layout_facts: Callable[..., dict] | None = None
    #: (ctx, mandate, stage, the inputs prepared for it, a run asked to continue?) -> what else
    #: its run reads, by kind, from the run it starts from and the system
    stage_inputs: Callable[..., dict] | None = None
    #: (ctx, stage, inputs by kind, walltime) -> what stops or is worth saying of the run
    job_findings: Callable[..., list] | None = None
    #: (ctx, stage, inputs, layout, cores, extend_ps, continuing, frames) -> (minutes, why)
    #: its run is booked for
    size_walltime: Callable[..., tuple] | None = None
    #: (ctx, stage, inputs by kind, layout, run request) -> the job, single, coupled or batch
    build_job: Callable[..., StageJob] | None = None
    #: (ctx, request) -> a validated reference dataset, merged from the runs named
    collect_dataset: Callable[..., object] | None = None
    #: (ctx, a log it wrote) -> its log read into typed facts, with whether the run is clean
    diagnose: Callable[..., object] | None = None
    #: (ctx, trajectories in order, the joined file) -> the finished process that joined them
    join_trajectories: Callable[..., object] | None = None
    #: (ctx, a run input, a structure of its atoms, the .pdb to write) -> that structure with
    #: the run input's bonds, for a reader that cannot take the run input; None when not written
    bonded_structure: Callable[..., Path | None] | None = None
    #: (ctx, an energy file, its terms by label, the directory to write in) -> (the series
    #: file, the finished process that wrote it)
    energy_series: Callable[..., tuple] | None = None
    #: (ctx, an input file as the call named it, the file, the run input it pairs with) -> what
    #: the engine says of it as it is on disk; None for a file that is not its own
    input_findings: Callable[..., InputCheck | None] | None = None
    #: (a topology) -> the force field it includes, as a system fact; None when it names none
    forcefield: Callable[[Path], object] | None = None
    #: how the files of a fit it ran read
    fit_readers: FitReaders | None = None
    #: every finding its checks of an input as it is on disk can raise, with its sources
    checks: tuple = ()
    #: the file kinds a stage's run of this engine reads
    run_reads: tuple[str, ...] = ()
    #: (ctx, the partition call's parameters) -> the system split into its QM region and the
    #: rest, with what the selection leaves of the region's chemistry
    partition_qm: Callable[..., object] | None = None
    #: (the system's parts by name, as dicts; the names asked) -> (the selection commands that
    #: take them into a QM region, the names none resolves)
    select_subsystems: Callable[..., tuple] | None = None
    #: (ctx, stage, frames asked) -> the frames a run of a scan stage computes; None for a
    #: stage that is no scan
    scan_frames: Callable[..., list | None] | None = None
    #: (mandate, stage) -> (the padding in nm, the pseudopotential table, "" when the method
    #: names none) the stage's QM region is partitioned with
    region_settings: Callable[..., tuple] | None = None
    #: (ws, stages, decisions, the stage selected for) -> what the latest partition left open
    #: of the region's chemistry, as questions for the researcher
    region_questions: Callable[..., list] | None = None
    #: (ctx, mandate, stage, the structure it starts from) -> what stops the stage's render
    prepare_findings: Callable[..., list] | None = None


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


def adapter_with(capability: str) -> EngineAdapter | None:
    """The first adapter that has this capability (a field of :class:`EngineAdapter`); None
    where no engine here has it."""
    return next((a for a in adapters() if getattr(a, capability)), None)


def run_kinds() -> frozenset[str]:
    """The file kinds a stage's run reads, under any engine: what its preparation wrote
    that no run reads (sidecars, provenance) is no input of it."""
    return frozenset(k for a in adapters() for k in a.run_reads)
