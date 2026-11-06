"""What an engine's files state, for code that names no engine: a system's files by kind and
a run's log, each read by the adapter of the engine that wrote them
(``engines/registry.py``)."""

from __future__ import annotations

from pathlib import Path

from myrmidon.engines.registry import adapter_of_log, adapters, adapters_reading


def data_dir_of(engines) -> Path | None:
    """The data directory (a force-field library, say) of the engines discovered here: where
    a system's files include their libraries from. None when no engine holds one."""
    infos = (engines or {}).values()
    found = next((info.data_dir for info in infos if getattr(info, "data_dir", None)), None)
    return Path(found) if found else None


def run_facts(text: str) -> dict:
    """What a run's log states, by fact key, parsed by the parser of the engine that wrote
    it (told apart by the log itself), never typed. A log no engine claims states nothing."""
    engine = adapter_of_log(text)
    return engine.run_facts(text) if engine is not None else {}


def run_values(text: str) -> list[tuple[str, object, str]]:
    """The run's facts as a report states them: (label, value, unit), in the engine's
    fact order."""
    engine = adapter_of_log(text)
    if engine is None:
        return []
    facts = engine.run_facts(text)
    return [(label, facts[key], unit) for key, (label, unit) in (engine.fact_labels or {}).items()
            if key in facts]


def dipole_values(log: Path) -> list[tuple[str, object, str]]:
    """The QM dipole as a report states it, from the DIPOLE file beside a CPMD log; none
    when there is no such file."""
    from myrmidon.engines.mimic.diagnostics import DIPOLE_FACTS, dipole_facts

    dipole = log.parent / "DIPOLE"
    if not dipole.is_file():
        return []
    facts = dipole_facts(log.read_text(errors="replace"), dipole.read_text(errors="replace"))
    return [(label, facts[key], unit) for key, (label, unit) in DIPOLE_FACTS.items()
            if key in facts]


def engine_log(rec) -> Path:
    """The file a run's engine wrote its account into, the one the facts are parsed from:
    the first any engine claims, else the run's stdout."""
    for engine in adapters():
        found = engine.run_log(rec) if engine.run_log else None
        if found is not None:
            return found
    return Path(rec.spec.cwd) / (rec.spec.stdout or "")


def is_run_log(path: Path) -> bool:
    """Whether a file is a log some engine wrote."""
    return path.is_file() and any(a.is_log(path) for a in adapters() if a.is_log)


def system_facts(files: dict, data_dir: Path | None = None,
                 rules: set[str] | None = None) -> dict:
    """The facts a system's files hold (``files``: kind -> path), by the engine key each
    decides: the union of what the reader of each engine whose files are among them reads.
    ``rules`` keeps only the ones named."""
    facts: dict = {}
    for engine in adapters_reading(files):
        if engine.system_facts is not None:
            facts.update(engine.system_facts(files, data_dir, rules))
    return facts


def system_parts(files: dict, data_dir: Path | None = None) -> dict[str, dict]:
    """The system's parts (subsystems) as its files lay them out; none without a file that
    does."""
    parts: dict[str, dict] = {}
    for engine in adapters_reading(files):
        if engine.system_parts is not None:
            parts.update(engine.system_parts(files, data_dir))
    return parts
