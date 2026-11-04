"""QM/MM diagnostics: the CPMD output parsed into typed convergence facts and
time-series statistics. What a failed run's log says is read by the model from its tail;
this module reads the numbers (steps, temperatures, energies, the NVE drift) and the
error message CPMD leaves in its LocalError file."""

from __future__ import annotations

import re
from pathlib import Path


def run_completed(facts: dict) -> bool:
    """Whether the log's own evidence says the run did what it was asked: it ended
    normally and, for dynamics, reached the step count its input set (``md_steps_requested``
    from the MAXIMUM NUMBER OF STEPS echo), or, for a wavefunction optimisation, converged.
    A log that ended without a step count to compare is not known to have completed."""
    if not facts.get("finished_normally"):
        return False
    converged = facts.get("wavefunction_converged")
    if converged is not None:
        return converged is True
    done = facts.get("md_steps_completed")
    requested = facts.get("md_steps_requested")
    if requested is not None:
        return done is not None and done >= requested
    return done is not None


def incomplete_run_note(facts: dict) -> str:
    """How far a run that did not complete got, by its own parsed facts."""
    if facts.get("wavefunction_converged") is False:
        return (f"the wavefunction optimisation stopped after {facts.get('scf_iterations')} "
                f"iteration(s) with GEMAX {facts.get('gemax_last')} above the criterion "
                f"{facts.get('gemax_criterion')}")
    done, asked = facts.get("md_steps_completed"), facts.get("md_steps_requested")
    if asked is not None:
        return f"{done if done is not None else 0} of {asked} MD steps"
    if not facts.get("finished_normally"):
        return "the log does not end normally"
    return "the log states no step count or convergence to judge it by"


def read_local_errors(directory: Path) -> str | None:
    """CPMD writes LocalError-<rank>.log on abort; return its error message(s)."""
    messages: list[str] = []
    for log in sorted(directory.glob("LocalError-*.log")):
        for line in log.read_text(errors="replace").splitlines():
            m = re.match(r"\s*error message:\s*(.+)", line)
            if m and m.group(1).strip() not in messages:
                messages.append(m.group(1).strip())
    return "; ".join(messages) if messages else None


_PLAIN = r"-?\d+\.\d*"  # a decimal without exponent
# BO-MD step line (real CPMD 4.3): NFI TEMPP EKS ECLASSIC DIS TCPU. SCF lines
# (INFR GEMAX ETOT DETOT TCPU) and wavefunction-optimisation lines have an
# exponent in their second column, which is what keeps them out of this match.
_MD_STEP_RE = re.compile(
    rf"^\s*(\d+)\s+(?:{_PLAIN}\s+)*?(-\d+\.\d+)\s+(-\d+\.\d+)(?:\s|$)", re.MULTILINE
)
_SCF_LINE_RE = re.compile(r"^\s*(\d+)\s+\d\.\d+E[-+]\d+\s+-\d+\.\d+", re.MULTILINE)

# A single point is a WAVEFUNCTION OPTIMIZATION run, not dynamics: its table is
# `NFI GEMAX CNORM ETOT DETOT TCPU`, interleaved with LINE SEARCH lines, and it has no MD
# step table at all, so the MD parser above reads nothing from it.
_WFOPT_HEADER_RE = re.compile(r"^\s*NFI\s+GEMAX\s+CNORM\s+ETOT\s+DETOT", re.MULTILINE)
_WFOPT_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d\.\d+E[-+]\d+)\s+(\d\.\d+E[-+]\d+)\s+(-?\d+\.\d+)", re.MULTILINE
)
_WFOPT_CRITERION_RE = re.compile(
    r"CONVERGENCE CRITERIA FOR WAVEFUNCTION OPTIMIZATION:\s*([\d.]+E[-+]\d+)"
)
_WFOPT_MAXSTEP_RE = re.compile(r"MAXIMUM NUMBER OF ITERATIONS FOR SC:\s*(\d+)")
_TOTAL_ENERGY_RE = re.compile(
    r"TOTAL ENERGY\s*=\s*(-?\d+\.\d+)\s*A\.U\.", re.MULTILINE
)


def parse_wavefunction_optimisation(text: str) -> dict:
    """The facts of a single point: how many SCF iterations, how far GEMAX fell, against
    which criterion, and the total energy that came out.

    A wavefunction that stopped at its iteration cap exits 0 and writes a RESTART and a
    reference file that looks exactly like a good one; this table is what tells them apart.
    """
    if not _WFOPT_HEADER_RE.search(text):
        return {}
    rows = _WFOPT_ROW_RE.findall(text)
    if not rows:
        return {}
    facts: dict = {
        "scf_iterations": int(rows[-1][0]),
        "gemax_last": float(rows[-1][1]),
        "energy_last": float(rows[-1][3]),
    }
    criterion = _WFOPT_CRITERION_RE.search(text)
    if criterion:
        facts["gemax_criterion"] = float(criterion.group(1))
    cap = _WFOPT_MAXSTEP_RE.search(text)
    if cap:
        facts["scf_max_iterations"] = int(cap.group(1))
    totals = _TOTAL_ENERGY_RE.findall(text)
    if totals:
        facts["energy_total"] = float(totals[-1])
    if "gemax_criterion" in facts:
        facts["wavefunction_converged"] = facts["gemax_last"] <= facts["gemax_criterion"]
    return facts


def seconds_per_step(text: str) -> float | None:
    """The median of the seconds each Born-Oppenheimer MD step took (the step line's last
    column, TCPU); None for a log with no MD step."""
    times = []
    for m in _MD_STEP_RE.finditer(text):
        line = text[m.start():text.find("\n", m.start())]
        try:
            times.append(float(line.split()[-1]))
        except ValueError:
            continue
    if not times:
        return None
    times.sort()
    return times[len(times) // 2]


def parse_cpmd_series(text: str) -> list[dict]:
    """Per-MD-step series from a CPMD BO-MD log: nfi, temperature (TEMPP), EKS,
    ECLASSIC and the SCF iteration count of the step's INFR block. Subsystem
    temperatures are included when the build prints nonzero values."""
    steps: list[dict] = []
    scf = 0
    t_qm = t_mm = None
    for line in text.splitlines():
        m = _SCF_LINE_RE.match(line)
        if m:
            scf = int(m.group(1))
            continue
        m = re.match(r"\s*T\((QM|MM)\)\s*=\s*([\d.]+)", line)
        if m:
            # the SUBSYSTEM TEMPERATURES block follows the step it was computed for (real
            # CPMD 4.3 + MiMiC logs: the NFI line, the SCF block, then the block), so it
            # belongs to the step already parsed, never to the next one
            value = float(m.group(2))
            key = "t_qm" if m.group(1) == "QM" else "t_mm"
            if steps:
                steps[-1][key] = value or None
            elif key == "t_qm":
                t_qm = value or None
            else:
                t_mm = value or None
            continue
        m = _MD_STEP_RE.match(line)
        if m and "E" not in line.split()[1].upper():
            fields = line.split()
            steps.append(
                {
                    "nfi": int(fields[0]),
                    "temp_k": float(fields[1]),
                    "eks": float(m.group(2)),
                    "eclassic": float(m.group(3)),
                    "scf_iters": scf or None,
                    "t_qm": t_qm,
                    "t_mm": t_mm,
                }
            )
            scf = 0
            t_qm = t_mm = None
    return steps


_TIMESTEP_IONS_RE = re.compile(r"TIME STEP FOR IONS:\s*([\d.]+)")
_MD_TABLE_RE = re.compile(r"^\s*NFI\s+TEMPP\s+EKS\s+ECLASSIC", re.MULTILINE)
#: what the header of a CPMD 4.3 log says about the ionic temperature (as the annealing,
#: NVE, heating and production stages of a solvated solute print it).
#: NVE is said outright; annealing is a line of its own beside "NOT CONTROLLED".
_ION_NOT_CONTROLLED = re.compile(r"ION DYNAMICS:\s+THE TEMPERATURE IS NOT CONTROLLED")
_ION_DYNAMICS_LINE = re.compile(r"^\s*(ION DYNAMICS.*?)\s*$", re.MULTILINE)
_ION_CONTROL = (
    (re.compile(r"SIMULATED ANNEALING OF IONS"), "simulated annealing"),
    (re.compile(r"BERENDSEN-STYLE THERMOSTAT"), "Berendsen-style thermostat"),
    (re.compile(r"TEMPERATURE CONTROL \(NOSE-HOOVER"), "Nose-Hoover thermostats"),
    (re.compile(r"TEMPERATURE RAMP"), "temperature ramp"),
)
#: atomic unit of time in ps (CODATA 2018)
_AU_TIME_PS = 2.4188843265857e-5


def nve_drift(text: str) -> dict:
    """The energy-conservation facts of a BO-MD log: what controls the ionic temperature,
    read from the header before the MD table (an NVE log names Nose only in its end-of-run
    summary), how long the dynamics ran, and, only when the header says the ionic
    temperature is not controlled, the drift of ECLASSIC with its fluctuation about the fit."""
    table = _MD_TABLE_RE.search(text)
    if not table:
        return {}
    header = text[: table.start()]
    control = next((label for pat, label in _ION_CONTROL if pat.search(header)), None)
    if control is None and not _ION_NOT_CONTROLLED.search(header):
        line = _ION_DYNAMICS_LINE.search(header)
        control = f"unrecognised: {line.group(1)}" if line else "not stated in the header"
    facts: dict = {"ion_temperature_control": control}
    dt = _TIMESTEP_IONS_RE.search(header)
    steps = parse_cpmd_series(text)
    if not dt or len(steps) < 3:
        return facts
    import numpy as np

    t = np.array([s["nfi"] for s in steps], dtype=float) * float(dt.group(1)) * _AU_TIME_PS
    e = np.array([s["eclassic"] for s in steps])
    facts["md_time_ps"] = float(t[-1] - t[0])
    if control is None and t[-1] > t[0]:
        slope, icept = np.polyfit(t, e, 1)
        facts["eclassic_drift_ha_per_ps"] = float(slope)
        facts["eclassic_fluctuation_ha"] = float(np.std(e - (slope * t + icept)))
    return facts


def parse_cpmd_output(text: str) -> dict:
    facts: dict = {}
    steps = [
        m for m in _MD_STEP_RE.finditer(text)
        if "E" not in m.group(0).split()[1].upper()  # 2nd column is TEMPP/EKINC, never GEMAX
    ]
    if steps:
        facts["md_steps_completed"] = int(steps[-1].group(1))
        facts["energy_first"] = float(steps[0].group(2))  # EKS
        facts["energy_last"] = float(steps[-1].group(2))
        facts["eclassic_last"] = float(steps[-1].group(3))
    # SCF iterations of the last step: the last INFR block
    blocks = re.split(r"^\s*INFR\s+GEMAX.*$", text, flags=re.MULTILINE)
    if len(blocks) > 1:
        last_iters = [int(m.group(1)) for m in _SCF_LINE_RE.finditer(blocks[-1])]
        if last_iters:
            facts["scf_last_iterations"] = max(last_iters)
    if "scf_last_iterations" not in facts:
        legacy = re.findall(r"^\s*ITERATIONS:\s*(\d+)", text, re.MULTILINE)
        if legacy:
            facts["scf_last_iterations"] = int(legacy[-1])
    t_qm = re.findall(r"T\(QM\)\s*=\s*([-\d.]+)", text)
    t_mm = re.findall(r"T\(MM\)\s*=\s*([-\d.]+)", text)
    if t_qm:
        facts["t_qm_last"] = float(t_qm[-1])
    if t_mm:
        facts["t_mm_last"] = float(t_mm[-1])
    facts.update(parse_wavefunction_optimisation(text))
    facts.update(nve_drift(text))
    facts["finished_normally"] = bool(
        re.search(r"PROGRAM CPMD ENDED|JOB DONE|CPU TIME", text)
    )
    requested = re.search(r"MAXIMUM NUMBER OF STEPS:\s*(\d+)", text)
    if requested:
        facts["md_steps_requested"] = int(requested.group(1))
    return facts


#: phrases only CPMD's stdout prints near its top — the banner, and the input echo every run
#: writes before it can fail (a start-up crash has no banner-free run, but a log trimmed to
#: its tail-end echo still has the echo)
_CPMD_MARKS = (b"PROGRAM CPMD STARTED", b"The CPMD consortium",
               b"PATH TO THE RESTART FILES", b"MAXIMUM NUMBER OF ITERATIONS FOR SC")


def is_cpmd_output(path: Path) -> bool:
    """Whether this file is CPMD's stdout, finished or not, from its first 16 kB."""
    stat = path.stat()
    if stat.st_size == 0 or stat.st_mode & 0o111:  # an output is never a program
        return False
    with path.open("rb") as fh:
        head = fh.read(16384)
    return any(mark in head for mark in _CPMD_MARKS)
