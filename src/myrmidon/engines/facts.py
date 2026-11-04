"""What a finished run's engine log states, by fact key: a coupled run's CPMD log or an
mdrun log, told apart by the log itself and parsed by the engine's own parser."""

from __future__ import annotations

#: what a coupled run's CPMD log states, by its key, and how a report labels it
RUN_FACTS = {
    "ion_temperature_control": ("ionic temperature control", ""),
    "md_steps_completed": ("MD steps completed", ""),
    "md_time_ps": ("simulated time", "ps"),
    "final_temperature_k": ("final ionic temperature", "K"),
    "tail_mean_temperature_k": ("mean ionic temperature over the last 10 % of steps", "K"),
    "mean_temperature_k": ("mean ionic temperature", "K"),
    "sd_temperature_k": ("its standard deviation", "K"),
    "eclassic_drift_ha_per_ps": ("conserved-energy drift", "Ha/ps"),
    "eclassic_fluctuation_ha": ("fluctuation about the fit", "Ha"),
    "finished_normally": ("finished normally", ""),
}
#: what DIPOLE DYNAMICS's file beside a coupled run's log states, with the log's cell volume
DIPOLE_FACTS = {
    "dipole_steps": ("steps of the QM dipole series", ""),
    "dipole_debye_mean": ("mean magnitude of the QM dipole", "D"),
    "dipole_debye_sd": ("its standard deviation", "D"),
    "dipole_debye_last": ("last magnitude of the QM dipole", "D"),
}
#: what a GROMACS mdrun log states (``engines/gromacs/parse.run_facts``): the means are
#: mdrun's own, averaged over the whole run, relaxation included
GROMACS_FACTS = {
    "minimisation_steps": ("minimisation steps", ""),
    "minimisation_converged": ("minimisation converged to its force tolerance", ""),
    "final_potential_kj_mol": ("final potential energy", "kJ/mol"),
    "final_max_force_kj_mol_nm": ("final maximum force", "kJ/mol/nm"),
    "md_steps_completed": ("MD steps completed", ""),
    "md_time_ps": ("simulated time", "ps"),
    "mean_temperature_k": ("mean temperature over the run (mdrun's average)", "K"),
    "mean_pressure_bar": ("mean pressure over the run (mdrun's average)", "bar"),
    "mean_potential_kj_mol": ("mean potential energy over the run (mdrun's average)",
                              "kJ/mol"),
    "mean_total_energy_kj_mol": ("mean total energy over the run (mdrun's average)",
                                 "kJ/mol"),
    "mean_conserved_energy_kj_mol": ("mean conserved energy over the run (mdrun's average)",
                                     "kJ/mol"),
    "averaged_over_frames": ("energy frames the averages are over", ""),
    "atoms": ("atoms", ""),
    "performance_ns_per_day": ("performance", "ns/day"),
    "finished_normally": ("finished normally", ""),
}

_CPMD_MARKS = ("PROGRAM CPMD STARTED", "The CPMD consortium", "PATH TO THE RESTART FILES",
               "MAXIMUM NUMBER OF ITERATIONS FOR SC")


def is_cpmd_text(text: str) -> bool:
    head = text[:16384]
    return any(mark in head for mark in _CPMD_MARKS)


def run_facts(text: str) -> dict:
    """What a run's log states, by fact key — a CPMD log's (RUN_FACTS) or an mdrun log's
    (GROMACS_FACTS), told apart by the log itself. Parsed by the engine's own parser
    (``engines/mimic/diagnostics``, ``engines/gromacs/parse``), never typed."""
    if not is_cpmd_text(text):
        from myrmidon.engines.gromacs.parse import run_facts as gromacs_facts

        return gromacs_facts(text)
    return cpmd_facts(text)


def cpmd_facts(text: str) -> dict:
    """What a coupled run's CPMD log states, by RUN_FACTS key: how the ionic temperature
    was controlled, how far the dynamics got, the temperature it ran at, and — only for a
    run whose header says the ions were not thermostatted — the drift of the conserved
    energy."""
    from myrmidon.engines.mimic.diagnostics import parse_cpmd_output, parse_cpmd_series

    parsed = parse_cpmd_output(text)
    series = parse_cpmd_series(text)
    facts: dict = {}
    if series:
        facts["ion_temperature_control"] = (parsed.get("ion_temperature_control")
                                            or "none (energy conserved)")
    for key in ("md_steps_completed", "md_time_ps", "eclassic_drift_ha_per_ps",
                "eclassic_fluctuation_ha"):
        if parsed.get(key) is not None:
            facts[key] = parsed[key]
    if series:
        temps = [step["temp_k"] for step in series]
        facts["final_temperature_k"] = temps[-1]
        tail = temps[-max(1, -(-len(temps) // 10)):]  # the last 10 %, at least one step
        facts["tail_mean_temperature_k"] = sum(tail) / len(tail)
        if len(temps) > 1:
            mean = sum(temps) / len(temps)
            facts["mean_temperature_k"] = mean
            facts["sd_temperature_k"] = (sum((t - mean) ** 2 for t in temps)
                                         / (len(temps) - 1)) ** 0.5
    facts["finished_normally"] = bool(parsed.get("finished_normally"))
    return facts


def run_values(text: str) -> list[tuple[str, object, str]]:
    """The run's facts as a report states them: (label, value, unit), in the engine's
    fact order."""
    facts = run_facts(text)
    table = RUN_FACTS if is_cpmd_text(text) else GROMACS_FACTS
    return [(label, facts[key], unit) for key, (label, unit) in table.items()
            if key in facts]


def dipole_values(log) -> list[tuple[str, object, str]]:
    """The QM dipole as a report states it, from the DIPOLE file beside a CPMD log; none
    when there is no such file."""
    from myrmidon.engines.mimic.diagnostics import dipole_facts

    dipole = log.parent / "DIPOLE"
    if not dipole.is_file():
        return []
    facts = dipole_facts(log.read_text(errors="replace"), dipole.read_text(errors="replace"))
    return [(label, facts[key], unit) for key, (label, unit) in DIPOLE_FACTS.items()
            if key in facts]


def engine_log(rec):
    """The file a run's engine wrote its account into: CPMD's stdout for a coupled run,
    ``<deffnm>.log`` for an mdrun — the file the facts are parsed from."""
    from pathlib import Path

    from myrmidon.engines.gromacs.collect import spec_deffnm
    from myrmidon.engines.gromacs.parse import is_gromacs_log
    from myrmidon.engines.mimic.diagnostics import is_cpmd_output

    cwd = Path(rec.spec.cwd)
    stdout = cwd / (rec.spec.stdout or "")
    if stdout.is_file() and is_cpmd_output(stdout):
        return stdout
    log = cwd / f"{spec_deffnm(rec)}.log"
    if log.is_file() and is_gromacs_log(log):
        return log
    return stdout


def system_facts(files: dict, data_dir=None, rules: set[str] | None = None) -> dict:
    """The facts a system's files hold (``files``: kind -> path: topology, index,
    structure), by the engine key each decides, read by the reader of the engine whose
    files they are; ``rules`` keeps only the ones named."""
    from myrmidon.engines.gromacs.system import system_facts as gromacs_facts

    return gromacs_facts(files.get("topology"), files.get("index"), files.get("structure"),
                         data_dir, rules)


def system_parts(files: dict, data_dir=None) -> dict[str, dict]:
    """The system's parts (subsystems) as its topology lays them out; none without one."""
    from myrmidon.engines.gromacs.system import subsystems_of

    top = files.get("topology")
    if top is None:
        return {}
    residuetypes = data_dir / "residuetypes.dat" if data_dir is not None else None
    return subsystems_of(top, [data_dir] if data_dir is not None else [], residuetypes)
