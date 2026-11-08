"""What GROMACS writes, read: mdrun logs into progress and run facts, and a walltime kill
into how far the run got."""

from __future__ import annotations

import re
from pathlib import Path

from myrmidon.core.observation import RunProgress

ROLE_BY_SUFFIX = {
    ".log": "log",
    ".edr": "energy",
    ".xtc": "trajectory",
    ".trr": "trajectory",
    ".gro": "structure",
    ".cpt": "checkpoint",
}

#: what SLURM writes into the job's stderr when it kills it at the walltime. The state
#: says so too (JobState.TIMEOUT), but a local run only has the text.
TIME_LIMIT = r"DUE TO TIME LIMIT|CANCELLED AT .*DUE TO TIME"


def parse_progress(log_path: Path) -> RunProgress | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    progress = RunProgress()
    steps = re.findall(r"^\s+Step\s+Time\s*\n\s*(\d+)\s+([\d.eE+-]+)", text, re.MULTILINE)
    if steps:
        progress.step = int(steps[-1][0])
        progress.time_ps = float(steps[-1][1])
    perf = re.findall(r"^Performance:\s+([\d.]+)", text, re.MULTILINE)
    if perf:
        progress.ns_per_day = float(perf[-1])
    if progress.step is None and progress.ns_per_day is None:
        return None
    return progress


#: what the head of an mdrun log says, finished or not (the mock's starts at the
#: parameter dump)
_LOG_MARKS = (b"GROMACS - gmx mdrun", b"Started mdrun on rank", b"Input Parameters:",
              b"Reading checkpoint file")
#: the labels of mdrun's end-of-run averages block a report states, by fact key and unit;
#: the keys are shared with the QM engine's facts where the quantity is one
_AVERAGE_FACTS = {
    "Temperature": ("mean_temperature_k", "K"),
    "Pressure (bar)": ("mean_pressure_bar", "bar"),
    "Potential": ("mean_potential_kj_mol", "kJ/mol"),
    "Total Energy": ("mean_total_energy_kj_mol", "kJ/mol"),
    "Conserved En.": ("mean_conserved_energy_kj_mol", "kJ/mol"),
}
#: what an mdrun log states, how a report labels it and its unit (``run_facts``): the means
#: are mdrun's own, averaged over the whole run, relaxation included
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
#: what says a run of each stage kind did what it asks: a minimisation's last force and whether
#: it reached its tolerance, a run's mean temperature, and an NPT run's pressure beside it
HEADLINE = {
    "minimize": ("final_max_force_kj_mol_nm", "minimisation_converged"),
    "dynamics NVE": ("mean_conserved_energy_kj_mol",),
    "dynamics NPT": ("mean_temperature_k", "mean_pressure_bar"),
    "dynamics": ("mean_temperature_k",),
}
_COLUMN = 15  # mdrun prints its energy tables in %15s / %15.5e columns
_MINIMISERS = ("steep", "cg", "l-bfgs")


def is_gromacs_text(text: str) -> bool:
    """Whether this log text is an mdrun log, finished or not, from its first 16 kB."""
    head = text[:16384]
    return any(mark.decode() in head for mark in _LOG_MARKS)


def is_gromacs_log(path: Path) -> bool:
    """Whether this file is an mdrun log, finished or not, from its first 16 kB."""
    try:
        stat = path.stat()
    except OSError:
        return False
    if stat.st_size == 0 or stat.st_mode & 0o111:  # an output is never a program
        return False
    with path.open("rb") as fh:
        head = fh.read(16384)
    return any(mark in head for mark in _LOG_MARKS)


def _columns(line: str) -> list[str]:
    return [line[k:k + _COLUMN].strip() for k in range(0, len(line), _COLUMN)]


def parse_averages(text: str) -> dict[str, float]:
    """label -> mean over the run, from the ``A V E R A G E S`` block mdrun writes when a
    run completes: label lines and value lines alternate in fixed columns until a blank
    line. Empty while the run is going or when it died."""
    marker = text.rfind("A V E R A G E S")
    if marker < 0:
        return {}
    block = text[marker:]
    start = block.find("Energies (kJ/mol)")
    if start < 0:
        return {}
    lines = block[start:].splitlines()[1:]
    out: dict[str, float] = {}
    i = 0
    while i + 1 < len(lines) and lines[i].strip():
        for label, value in zip(_columns(lines[i]), _columns(lines[i + 1]), strict=False):
            try:
                out[label] = float(value)
            except ValueError:
                continue
        i += 2
    return out


def run_facts(text: str) -> dict:
    """What an mdrun log states: how far the dynamics got, whether mdrun finished, its
    performance, and — for a completed run — the means mdrun itself averaged over the run.
    Parsed from the log, never typed; the keys are the report's (GROMACS_FACTS).
    """
    facts: dict = {}
    integrator = re.search(r"^\s*integrator\s*=\s*(\S+)", text, re.MULTILINE)
    minimiser = integrator is not None and integrator.group(1).lower() in _MINIMISERS
    steps = re.findall(r"^\s+Step\s+Time\s*\n\s*(\d+)\s+([\d.eE+-]+)", text, re.MULTILINE)
    if minimiser:
        # a minimiser's "Time" column counts steps: there is no simulated time, and the
        # averages block averages nothing dynamical. Its verdict is its own closing lines.
        done = re.search(r"converged to Fmax < \S+ in (\d+) steps", text)
        if done:
            facts["minimisation_steps"] = int(done.group(1))
            facts["minimisation_converged"] = True
        elif steps:
            facts["minimisation_steps"] = int(steps[-1][0])
            if re.search(r"reached the maximum number of steps|did not converge", text):
                facts["minimisation_converged"] = False
        for pattern, key in ((r"^Potential Energy\s*=\s*(\S+)", "final_potential_kj_mol"),
                             (r"^Maximum force\s*=\s*(\S+)", "final_max_force_kj_mol_nm")):
            found = re.search(pattern, text, re.MULTILINE)
            if found:
                facts[key] = float(found.group(1))
    elif steps:
        facts["md_steps_completed"] = int(steps[-1][0])
        facts["md_time_ps"] = float(steps[-1][1])
    atoms = re.search(r"^There are:\s+(\d+)\s+Atoms", text, re.MULTILINE)
    if atoms:
        facts["atoms"] = int(atoms.group(1))
    perf = re.findall(r"^Performance:\s+([\d.]+)", text, re.MULTILINE)
    if perf and not minimiser:
        facts["performance_ns_per_day"] = float(perf[-1])
    if not minimiser:
        averages = parse_averages(text)
        for label, (key, _unit) in _AVERAGE_FACTS.items():
            if label in averages:
                facts[key] = averages[label]
        found = re.search(r"Statistics over (\d+) steps using (\d+) frames", text)
        if found:
            facts["averaged_over_frames"] = int(found.group(2))
    facts["finished_normally"] = bool(re.search(r"^Finished mdrun on rank", text, re.MULTILINE))
    return facts


def parse_nsteps(text: str) -> int | None:
    """How many steps the run was asked for, from mdrun's own dump of the input
    parameters into the log — never from the mdp the harness thinks it used."""
    found = re.search(r"^\s*nsteps\s*=\s*(\d+)", text, re.MULTILINE)
    return int(found.group(1)) if found else None


def walltime_diagnosis(log_path: Path, text: str) -> str:
    """A walltime kill, with how far it got: the number that decides whether to continue
    from the checkpoint or to book a bigger job."""
    progress = parse_progress(log_path)
    step = progress.step if progress is not None else None
    reached = f"step {step}" if step is not None else "no completed step"
    asked = parse_nsteps(text)
    of = f" of {asked}" if asked else ""
    share = ""
    if step is not None and asked:
        share = f" ({100 * step / asked:.0f}% of the run)"
    integrator = re.search(r"^\s*integrator\s*=\s*(\S+)", text, re.MULTILINE)
    if integrator and integrator.group(1) in _MINIMISERS:
        # a minimisation writes no checkpoint: it starts again, for longer
        return (f"killed at its walltime after {reached}{of}{share} — the booking was "
                "shorter than the minimisation needed, and a minimisation writes no "
                "checkpoint: run it again with a longer walltime_min (or, on fewer than a "
                "node's cores, more ntasks)")
    return (
        f"killed at its walltime after {reached}{of}{share} — the booking was shorter "
        "than the run needed. Continue from the checkpoint: stage_run with continue_run, or "
        "book more (stage_run walltime_min)."
    )
