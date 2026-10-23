"""QM/MM diagnostics: parse the CPMD output into typed convergence facts.

The crash/convergence patterns target real CPMD output tokens; they are data,
not logic — extend the pattern tables as real-world outputs surface, with the
real output as a fixture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.observation import Observation
from myrmidon.core.tool import FileReader, ToolContext, tool
from myrmidon.core.validation import ValidationWarning
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.text import tail

# (id, regex, diagnosis) — first match wins. The id is what input sets, protocols and
# skills cite; the sentence is what the model reads.
_CPMD_FAILURE_PATTERNS: list[tuple[str, str, str]] = [
    (
        "CPMD_MULTIPLICITY_WITHOUT_LSD",
        # real CPMD 4.3: MULTIPLICITY in &SYSTEM without LSD in &CPMD
        r"INCOMPATIBLE WITH LDA|REQUIRES LOCAL SPIN DENSITY",
        "MULTIPLICITY given without LSD — CPMD accepts MULTIPLICITY only in "
        "spin-polarized runs. Omit it for closed-shell systems, or add LSD to &CPMD "
        "for open-shell ones (mimic_prepare handles both).",
    ),
    (
        "MIMIC_PATHS_NOT_CWD",
        # real MCL 2.0.2 (MPMDTransport): GROMACS's canonical cwd is not in &MIMIC PATHS
        r"Unidentified client",
        "MiMiC coupling refused: GROMACS's working directory is not one of the &MIMIC PATHS "
        "entries. PATHS must be the canonical absolute path of the directory mdrun runs in "
        "(no symlinks, no '..', matching what mimic_select_qm -path wrote).",
    ),
    (
        "CPMD_SHAKE_WRONG_GEOMETRY",
        # real CPMD 4.3 + MiMiC: a RESTART ... GEOFILE whose GEOMETRY belongs to another
        # solvation of the system. CPMD dies in SHAKE before the MiMiC handshake completes,
        # so GROMACS reports a coupling failure — ahead of the MPI pattern for that reason.
        # The abort line is "NEW_SHAKE| ..." (routine, pipe, message); the bare routine
        # name is also a row of the timing table every finished run prints.
        r"NEW_SHAKE\||PCG DID NOT CONVERGE AFTER SHAKE_CG_ITER",
        "CPMD could not satisfy its bond constraints (SHAKE) at the start of the run. After "
        "a restart with GEOFILE the usual cause is a GEOMETRY file from a different "
        "solvation of the system: its line count differs from the system's atom count, so "
        "CPMD places coordinates on the wrong atoms. Any coupling error GROMACS reports is "
        "a symptom. Use a GEOMETRY file from this box, or drop GEOFILE from RESTART.",
    ),
    (
        "SCHEDULER_TIME_LIMIT",
        r"DUE TO TIME LIMIT",
        "The scheduler stopped the job at its walltime limit; CPMD did not fail. The last "
        "completed step is what the run has; extend the walltime or restart from it.",
    ),
    (
        "MIMIC_UNEVEN_RANK_LAYOUT",
        # real MiMiC, 2 nodes, block distribution: CPMD ranks per node unequal
        r"Message truncated",
        "An MPI collective received a different count than was sent — in a multi-node "
        "MiMiC run, the CPMD ranks are unevenly split across nodes (block distribution "
        "puts the GROMACS ranks on the last node), and a startup collective uses a "
        "node-local rank count. Use a cyclic task distribution with CPMD ranks divisible "
        "by the node count and GROMACS ranks spread evenly; single-node runs are "
        "unaffected.",
    ),
    (
        "CPMD_PROPCAL_SIGFPE",
        # real CPMD 4.3: N-1 ranks trap after the first MD step (propcal MOD(nfi, 0))
        r"integer divide by zero|Caught signal 8",
        "CPMD integer divide-by-zero (SIGFPE). With >1 CPMD rank right after the first "
        "MD step this is propcal's MOD(nfi, ipolarise/iconduct): those &PROP defaults are "
        "set on the I/O rank only and never broadcast (CPMD 4.3 propin_utils). Fix CPMD "
        "(broadcast condpa%/ipolarise) or run CPMD on 1 rank.",
    ),
    (
        "CPMD_SEGFAULT",
        # real CPMD 4.3 + MiMiC: crash in mimic_short_range with a zero-vacuum QM cell
        r"Segmentation fault|SIGSEGV|Caught signal 11",
        "CPMD segfaulted. In MiMiC runs the usual cause is a QM cell without vacuum "
        "(prepqm -pad 0: &SYSTEM CELL equals the QM atoms' bounding box) — check "
        "min_vacuum_bohr from mimic_select_qm; next suspects are &MIMIC BOX/OVERLAPS "
        "inconsistent with the GROMACS system.",
    ),
    (
        "CPMD_WAVEFUNCTION_STEP_CAP",
        # real CPMD 4.3: a wavefunction optimisation that ran out of steps. It exits 0 and
        # writes a RESTART, so without this the run reads as a success. Ahead of
        # the generic SCF pattern below, which would blame the chemistry for a cap.
        r"RWFOPT\|\s*THE MAXIMUM NUMBER OF STEP IS REACHED",
        "The wavefunction optimisation stopped at its iteration cap, not at the "
        "convergence criterion: MAXSTEP bounds SCF iterations under WAVEFUNCTION "
        "OPTIMIZATION (it bounds ionic steps under MOLECULAR DYNAMICS). The forces and "
        "the energy are those of an unconverged wavefunction. Raise the cap or leave it "
        "unset rather than reading the numbers.",
    ),
    (
        "CPMD_KILLED_OUT_OF_MEMORY",
        # real MiMiC single point, a large QM cell (~6 M density plane waves on 42 ranks
        # of a 94 GB node): one rank is OOM-killed, cpmd.out stops after the plane-wave
        # setup with no error text. After the signal patterns, so an explicit crash wins.
        r"oom[-_ ]?kill|Out Of Memory|\btasks? [\d,-]+: Killed",
        "A rank was killed by the system with no error from CPMD. When cpmd.out stops "
        "right after the plane-wave setup (NUMBER OF PLANE WAVES), this is memory: the "
        "QM cell and cutoff need more memory per node than the job had. Use nodes with "
        "more memory, fewer ranks per node, or a smaller cell or cutoff.",
    ),
    (
        "CPMD_SCF_NOT_CONVERGED",
        r"SELF-CONSISTENCY NOT (?:REACHED|ACHIEVED)|NO CONVERGENCE",
        "SCF did not converge — typical causes: wrong QM charge/multiplicity, "
        "too-small wavefunction cutoff, or a bad QM/MM boundary (cut through a "
        "conjugated/polar bond).",
    ),
    (
        "CPMD_BAD_GEOMETRY",
        r"GEOMETRY IS NOT (?:COMPLETE|CONSISTENT)|ATOM .* TOO CLOSE",
        "Bad QM geometry — overlapping atoms or a broken structure entering CPMD.",
    ),
    (
        "MIMIC_MPI_COUPLING",
        r"MPI_|COMMUNICATOR|CONNECT.*(?:FAIL|TIMEOUT)",
        "MPI coupling failure — GROMACS and CPMD did not connect. Check that both "
        "programs started, ranks match the allocation, and &MIMIC PATHS are correct.",
    ),
    (
        "CPMD_OUT_OF_MEMORY",
        r"MEMORY ALLOCATION|INSUFFICIENT MEMORY",
        "CPMD ran out of memory — reduce the plane-wave cutoff, the QM region, or "
        "increase ranks/nodes.",
    ),
]


@dataclass(frozen=True)
class FailureMatch:
    """A known failure signature found in a run's evidence, with the text it matched."""

    id: str
    diagnosis: str
    span: str  # the matched line, so a false match can be read instead of inferred


def find_failure(text: str) -> FailureMatch | None:
    """The first known failure signature in ``text``, in pattern order, with its line."""
    for failure_id, pattern, diagnosis in _CPMD_FAILURE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            start = text.rfind("\n", 0, m.start()) + 1
            end = text.find("\n", m.end())
            span = text[start:end if end >= 0 else len(text)].strip()
            return FailureMatch(failure_id, diagnosis, span[:200])
    return None


def match_failure(text: str) -> tuple[str, str] | None:
    """(id, diagnosis) of the first known failure signature in ``text``."""
    hit = find_failure(text)
    return (hit.id, hit.diagnosis) if hit else None


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


def signature_stands(facts: dict, *, exit_code: int | None = None,
                     cpmd_error: str | None = None, witnessed: bool = False) -> bool:
    """Whether the signature ``parse_cpmd_output`` found is the run's diagnosis.

    It is, unless the run completed by its own log and nothing else witnesses an abort:
    a non-zero exit code, a LocalError file, a crash on stderr (``witnessed``). A run
    that ended normally with every step it asked for is not failed by a pattern — the
    end-of-run timing table of every finished run lists the SHAKE routine by name, and
    under the loop a known signature parks the branch."""
    if not facts.get("signature_id"):
        return False
    if (exit_code not in (0, None)) or cpmd_error or witnessed:
        return True
    return not run_completed(facts)


def outranked_note(facts: dict, *, exit_code: int | None = None) -> str:
    """What to say about a signature that matched in a run that completed."""
    steps = facts.get("md_steps_completed")
    requested = facts.get("md_steps_requested")
    did = (f"{steps} of {requested} MD steps" if requested is not None
           else f"{steps} MD steps" if steps is not None else "its wavefunction optimisation")
    exit_part = f", exit code {exit_code}" if exit_code is not None else ""
    return (f"{facts['signature_id']} matched {facts['signature_span']!r}, but the log ends "
            f"normally after {did}{exit_part}: not taken as a failure. Read the matched "
            "line if the result looks wrong; a real abort leaves a non-zero exit, a "
            "LocalError file or an incomplete log.")


class QMMMDiagnostics(Observation):
    """What a CPMD log says, parsed.

    The field descriptions are where the engine facts live: a skill names this observation
    and the field that decides its next step, and the description says what the column is.
    That is why "CPMD prints four temperatures and only one of them is the one you want"
    is written here and not in prose somewhere a reader has to be told to go and find.
    """

    md_steps_completed: int | None = None
    scf_last_iterations: int | None = Field(
        default=None,
        description="SCF iterations of the last MD step (the last INFR GEMAX block). For a "
        "single point see scf_iterations, which counts the whole optimisation.",
    )
    energy_first: float | None = None  # a.u., EKS (Kohn-Sham energy) at the first MD step
    energy_last: float | None = None
    eclassic_last: float | None = None  # a.u., ECLASSIC (conserved classical energy)
    t_qm_last: float | None = Field(
        default=None,
        description="K. MiMiC SUBSYSTEM TEMPERATURES T(QM): the instantaneous temperature "
        "of the subsystem CPMD integrates, and the one a CPMD thermostat acts on. A CPMD "
        "log prints four temperatures — T(QM), T(MM), a combined TEMPP dominated by "
        "whichever subsystem has more atoms, and an end-of-run averaged IONIC TEMPERATURE. "
        "A heating ramp's starting point is this one, never the averaged one. A small QM "
        "region has few degrees of freedom, so this value swings widely step to step.",
    )
    t_mm_last: float | None = Field(
        default=None,
        description="K. T(MM): the classical subsystem's instantaneous temperature. In a "
        "solvated box it dominates TEMPP and can read as equilibrated while T(QM) is far "
        "colder.",
    )
    scf_iterations: int | None = Field(
        default=None,
        description="Iterations of a WAVEFUNCTION OPTIMIZATION (single-point) run.",
    )
    gemax_last: float | None = Field(
        default=None,
        description="The last GEMAX of a wavefunction optimisation: the largest component "
        "of the electronic gradient. Convergence is GEMAX <= gemax_criterion.",
    )
    gemax_criterion: float | None = Field(
        default=None,
        description="CONVERGENCE CRITERIA FOR WAVEFUNCTION OPTIMIZATION, as the input set "
        "asked for it.",
    )
    scf_max_iterations: int | None = Field(
        default=None,
        description="MAXIMUM NUMBER OF ITERATIONS FOR SC. Under WAVEFUNCTION OPTIMIZATION "
        "CPMD's MAXSTEP bounds SCF iterations; under MOLECULAR DYNAMICS it bounds ionic "
        "steps. Reaching this cap ends the run at exit code 0 with a RESTART written.",
    )
    energy_total: float | None = Field(
        default=None,
        description="a.u., the (K+E1+L+N+X+MM+QM/MM) TOTAL ENERGY of a single point.",
    )
    wavefunction_converged: bool | None = Field(
        default=None,
        description="Whether GEMAX reached the criterion. None when the log is not a "
        "wavefunction optimisation, or carries no table to read. False is the failure an "
        "unconverged reference frame looks like from outside: exit 0, a RESTART, and "
        "forces that are not the forces of the state you asked for.",
    )
    ion_temperature_control: str | None = Field(
        default=None,
        description="What controls the ionic temperature, as the log's header states it "
        "(a Berendsen-style thermostat, simulated annealing, a temperature ramp, Nose-Hoover "
        "thermostats, or what it says when that is none of these). None only when the header "
        "says the ionic temperature is not controlled: NVE, where ECLASSIC is the energy the "
        "dynamics conserves.",
    )
    md_time_ps: float | None = Field(
        default=None, description="ps of dynamics in the log: steps times TIME STEP FOR IONS"
    )
    eclassic_drift_ha_per_ps: float | None = Field(
        default=None,
        description="Hartree per ps, the slope of a straight line through ECLASSIC over the "
        "MD steps: the NVE energy-conservation number. Only for a run with no ionic "
        "temperature control, where ECLASSIC is conserved; whether it is small enough is "
        "the researcher's criterion. Compare it with eclassic_fluctuation_ha: a drift over "
        "the run smaller than the fluctuation is not resolved by this run.",
    )
    eclassic_fluctuation_ha: float | None = Field(
        default=None,
        description="Hartree, the standard deviation of ECLASSIC about that line",
    )
    md_steps_requested: int | None = Field(
        default=None,
        description="MAXIMUM NUMBER OF STEPS as the log echoes it from the input: the step "
        "count a dynamics run is asked for. A run is complete when md_steps_completed "
        "reaches it and the log ends normally.",
    )
    finished_normally: bool = False
    diagnosis: str | None = None
    diagnosis_id: str | None = Field(
        default=None,
        description="stable id of the matched failure signature (e.g. "
        "CPMD_MULTIPLICITY_WITHOUT_LSD); None when no known signature matched, or when "
        "one matched in a run that completed with nothing witnessing an abort (then "
        "warnings carries CPMD_SIGNATURE_IN_FINISHED_LOG with the matched line)",
    )
    signature_span: str | None = Field(
        default=None,
        description="the line a failure signature matched, quoted from the file it was "
        "found in, so a match can be read rather than inferred",
    )
    cpmd_error: str | None = None  # 'error message:' from CPMD's LocalError-*.log


class MimicDiagnoseParams(BaseModel):
    cpmd_output: str = Field(default="", description="CPMD stdout log (default: "
                             "<stage>/cpmd.out)")
    stage: str = Field(default="qmmm")


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
    hit = find_failure(text)
    if hit:
        facts["signature_id"], facts["signature"] = hit.id, hit.diagnosis
        facts["signature_span"] = hit.span
        # the log alone: a signature in a run that completed is not its diagnosis; a
        # reader with the exit code or a LocalError judges again (signature_stands)
        if signature_stands(facts):
            facts["diagnosis_id"], facts["diagnosis"] = hit.id, hit.diagnosis
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


@tool(
    "mimic_diagnose",
    description="Parse a MiMiC/CPMD log into typed facts: MD steps, subsystem "
    "temperatures, SCF iterations, energies (a.u.), and — for a single point — whether "
    "the wavefunction converged. ok=False when the run did not finish normally, a known "
    "failure matched, or the wavefunction did not converge.",
    returns=QMMMDiagnostics,
    tags={"mimic", "qmmm", "analysis"},
    requires={"mimicpy"},
    reads=(FileReader("cpmd_output", is_cpmd_output, "a CPMD output"),),
)
def mimic_diagnose(ctx: ToolContext, params: MimicDiagnoseParams) -> QMMMDiagnostics:
    """The designated reader of a CPMD log, for both run types it can be.

    `ok` is False when the run did not finish normally, when a known failure signature
    matched, when the wavefunction did not converge, and when nothing at all could be read
    from the log. The last two matter because a single point that stops at its iteration
    cap exits 0 and writes a RESTART and a reference file that look exactly like a good
    one's. A tool that cannot tell those apart is worse than one that is absent, because
    it is the one a chain is built to trust.
    """
    located = ""
    if params.cpmd_output:
        path = resolve_in_ws(ctx, params.cpmd_output)
    else:
        stage = Path(ctx.workspace.stage(params.stage))
        path = stage / "cpmd.out"
        if not path.exists():
            # every mimic_run executes in its own numbered directory under the stage;
            # without a path, the latest run there is the one meant
            runs = sorted(p for p in stage.glob("*/cpmd.out") if p.parent.is_dir())
            if not runs:
                return QMMMDiagnostics(
                    ok=False,
                    summary=f"no CPMD output under {ctx.workspace.relpath(stage)} — pass "
                    "cpmd_output explicitly",
                )
            path = runs[-1]
            located = (f"{ctx.workspace.relpath(path)}, the latest of {len(runs)} run(s) "
                       f"in {ctx.workspace.relpath(stage)}. ")
    text = path.read_text(errors="replace")
    facts = parse_cpmd_output(text)
    cpmd_error = read_local_errors(path.parent)
    # crashes (signals, MPI aborts) land on stderr, not in the CPMD log
    err_path = path.with_suffix(".err")
    err_text = err_path.read_text(errors="replace") if err_path.exists() else ""
    # One search over every source, in pattern order: the most specific cause wins
    # wherever it was written, so a generic MPI line in cpmd.out cannot stand in for the
    # SHAKE abort that only LocalError-*.log names.
    evidence = "\n".join(part for part in (text, err_text, cpmd_error or "") if part)
    hit = find_failure(evidence)
    sig = re.search(r"Caught signal \d+ \([^)]*\)", err_text) if err_text else None
    if sig and not cpmd_error:
        frame = re.search(r"^\s*1\s+0x[0-9a-f]+\s+(\S+?)\(\)", err_text, re.MULTILINE)
        cpmd_error = sig.group(0) + (f" in {frame.group(1)}" if frame else "")
    # a signature is the diagnosis when something witnesses the abort — a LocalError, a
    # signal, or the match itself lying outside the log (stderr, LocalError) — or when the
    # log did not complete what it was asked. A completed run is not failed by a pattern
    # alone: the SHAKE routine's name sits in the timing table of every finished run
    diagnosis_id = diagnosis = span = None
    warnings = []
    if hit:
        span = hit.span
        facts.setdefault("signature_id", hit.id)
        facts.setdefault("signature_span", hit.span)
        if signature_stands(facts, cpmd_error=cpmd_error,
                            witnessed=bool(sig) or hit.span not in text):
            diagnosis_id, diagnosis = hit.id, hit.diagnosis
        else:
            warnings.append(ValidationWarning(
                id="CPMD_SIGNATURE_IN_FINISHED_LOG", severity="warn",
                message=outranked_note(facts)))
    if cpmd_error and diagnosis is None:
        diagnosis = f"CPMD aborted: {cpmd_error}"
    converged = facts.get("wavefunction_converged")
    read_something = any(
        facts.get(key) is not None
        for key in ("md_steps_completed", "scf_last_iterations", "scf_iterations")
    )
    ok = (
        facts.get("finished_normally", False)
        and diagnosis is None
        and cpmd_error is None
        and converged is not False
        and read_something
    )
    bits = []
    if facts.get("md_steps_completed") is not None:
        bits.append(f"{facts['md_steps_completed']} MD step(s)")
    if facts.get("scf_last_iterations") is not None:
        bits.append(f"last SCF: {facts['scf_last_iterations']} iterations")
    if facts.get("scf_iterations") is not None:
        bits.append(
            f"wavefunction optimisation: {facts['scf_iterations']} iterations, "
            f"GEMAX {facts.get('gemax_last')} against {facts.get('gemax_criterion')}"
        )
    if facts.get("energy_total") is not None:
        bits.append(f"total energy = {facts['energy_total']} a.u.")
    elif facts.get("energy_last") is not None:
        bits.append(f"EKS = {facts['energy_last']} a.u.")
    if facts.get("t_qm_last") is not None:
        bits.append(f"T(QM) = {facts['t_qm_last']} K, T(MM) = {facts.get('t_mm_last')} K")
    if facts.get("eclassic_drift_ha_per_ps") is not None:
        bits.append(
            f"NVE over {facts['md_time_ps']:.4g} ps: ECLASSIC drift "
            f"{facts['eclassic_drift_ha_per_ps']:.3g} Ha/ps, fluctuation "
            f"{facts['eclassic_fluctuation_ha']:.2g} Ha")
    elif facts.get("ion_temperature_control"):
        bits.append(f"ions under {facts['ion_temperature_control']}: ECLASSIC not conserved")
    if converged is False:
        bits.append("wavefunction NOT converged")
    # A log nothing could be read from is not a log that says the run was fine: a
    # "finished normally, no step data parsed" would let unconverged forces into a fit
    # unremarked.
    head = (
        "Run finished normally: " if ok
        else ("Nothing could be read from this log: " if not read_something
              else "Run NOT clean: ")
    )
    summary = (
        located
        + head
        + (", ".join(bits) if bits else "no step data parsed")
        + (f". {diagnosis}" if diagnosis else "")
        + (f" [matched: {span!r}]" if diagnosis_id and span else "")
        + (f" [CPMD: {cpmd_error}]" if cpmd_error and cpmd_error not in (diagnosis or "") else "")
    )
    tail_text = None
    if not ok:
        tail_text = tail(text, 25)
    return QMMMDiagnostics(
        ok=ok,
        summary=summary,
        warnings=warnings,
        md_steps_completed=facts.get("md_steps_completed"),
        md_steps_requested=facts.get("md_steps_requested"),
        signature_span=span,
        scf_last_iterations=facts.get("scf_last_iterations"),
        energy_first=facts.get("energy_first"),
        energy_last=facts.get("energy_last"),
        eclassic_last=facts.get("eclassic_last"),
        t_qm_last=facts.get("t_qm_last"),
        t_mm_last=facts.get("t_mm_last"),
        scf_iterations=facts.get("scf_iterations"),
        gemax_last=facts.get("gemax_last"),
        gemax_criterion=facts.get("gemax_criterion"),
        scf_max_iterations=facts.get("scf_max_iterations"),
        energy_total=facts.get("energy_total"),
        wavefunction_converged=converged,
        ion_temperature_control=facts.get("ion_temperature_control"),
        md_time_ps=facts.get("md_time_ps"),
        eclassic_drift_ha_per_ps=facts.get("eclassic_drift_ha_per_ps"),
        eclassic_fluctuation_ha=facts.get("eclassic_fluctuation_ha"),
        finished_normally=facts.get("finished_normally", False),
        diagnosis=diagnosis,
        diagnosis_id=diagnosis_id,
        cpmd_error=cpmd_error,
        log_tail=tail_text,
    )
