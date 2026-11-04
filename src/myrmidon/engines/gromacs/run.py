"""What a run's input says about the run it will be: grompp's warnings, and whether the
booked walltime can cover the dynamics or the minimisation it carries."""

from __future__ import annotations

import re

from myrmidon.core.tool import ToolContext
from myrmidon.core.validation import Finding

_WARNING_BLOCK_RE = re.compile(r"WARNING \d+ \[[^\]]*\]:\s*\n((?:\s{2,}.*\n?)+)")
_NET_CHARGE_RE = re.compile(r"System has non-zero total charge:\s*(-?[\d.]+)")


#: grompp's input errors: where, and what
_ERROR_BLOCK_RE = re.compile(
    r"ERROR \d+ \[file ([^,\]]+)(?:, line (\d+))?\]:[ \t]*\n((?:[ \t]{2,}.*\n?)+)")
#: a fatal error's own lines, up to the troubleshooting boilerplate every one ends with
_FATAL_RE = re.compile(r"Fatal error:\n(.*?)(?:\n\s*\n|\nFor more information|\Z)", re.S)


def grompp_errors(stderr: str) -> list[Finding]:
    """What made grompp stop, in its own words without the boilerplate after them: each
    ``ERROR n [file, line]`` block, else the fatal error's lines — except the fatal error
    that only counts the warnings, which the WARNING findings already say."""
    found = [Finding(id="GROMPP_ERROR",
                     message=f"{path}{f', line {line}' if line else ''}: "
                     + " ".join(text.split()))
             for path, line, text in _ERROR_BLOCK_RE.findall(stderr)]
    if not found:
        found = [Finding(id="GROMPP_FATAL", message=" ".join(m.split()))
                 for m in _FATAL_RE.findall(stderr)
                 if m.strip() and not m.lstrip().startswith("Too many warnings")]
    return found


def grompp_warnings(stderr: str) -> list[Finding]:
    """GROMACS' NOTEs, WARNINGs and errors re-parsed into structured findings. A WARNING
    is what -maxwarn would have hidden; the net-charge one carries the number
    and the fix, because 'silence it' is the plausible wrong path."""
    found = [
        Finding(id="GROMPP_NOTE", message=m.strip())
        for m in re.findall(r"NOTE \d+ \[[^\]]*\]:\s*\n((?:\s{2,}.*\n?)+)", stderr)
    ]
    charge = _NET_CHARGE_RE.search(stderr)
    for block in _WARNING_BLOCK_RE.findall(stderr):
        text = " ".join(block.split())
        if "net charge" in text.lower() and charge:
            found.append(
                Finding(
                    id="GROMPP_NET_CHARGE",
                    message=f"system net charge {charge.group(1)} e with Ewald electrostatics: "
                    "GROMACS would add a uniform background charge and the dynamics would be "
                    "of the wrong system. Neutralize with gmx_genion — given this structure "
                    "and topology it builds its own run input — then grompp the neutral "
                    "system cleanly. " + text,
                )
            )
        else:
            found.append(Finding(id="GROMPP_WARNING", message=text))
    return found + grompp_errors(stderr)


def throughput_warning(length_ns: float, walltime_min: int, ctx: ToolContext) -> Finding:
    """Does the booked walltime cover the run? The harness cannot know this
    machine's ns/day, but it can compute what the booking *requires* and compare
    it with the plausible ceiling in the machine config — a job that needs more
    stops at the time limit with a partial trajectory, which is not the run
    that was asked for. The ceiling is the machine config's estimate, so the finding
    informs the caller rather than refusing the call."""
    cap = ctx.config.compute.plausible_ns_per_day
    required = length_ns / (walltime_min / 1440.0)
    reach = round(cap * walltime_min / 1440.0, 3)
    if required > cap:
        return Finding(
            id="WALLTIME_INSUFFICIENT",
            message=f"the run is {length_ns:g} ns; {walltime_min} min of walltime would need "
            f"{required:.0f} ns/day, above this machine's plausible ceiling of {cap:g} ns/day "
            f"(compute.plausible_ns_per_day) — mdrun will stop at the time limit after roughly "
            f"{reach:g} ns, not the run that was asked for. Book enough walltime (and the "
            "core-hours that implies), split into checkpointed segments, or report that the "
            "run does not fit the allowance.",
        )
    return Finding(
        id="RUN_LENGTH",
        message=f"the run is {length_ns:g} ns; {walltime_min} min needs {required:.0f} ns/day",
    )


def minimisation_warning(
    nsteps: int, n_atoms: int, walltime_min: int, ctx: ToolContext
) -> Finding:
    """Can the booked walltime cover a minimisation? Its size is steps × atoms, and the
    machine config says how fast that goes here. A minimisation killed at the time limit
    leaves no converged structure, and a box many times the intended size is the usual
    reason the booking falls short."""
    rate = ctx.config.compute.plausible_em_atom_steps_per_s
    capacity = rate * walltime_min * 60
    reach = int(capacity / max(n_atoms, 1))
    if nsteps * n_atoms > capacity:
        return Finding(
            id="EM_WALLTIME_INSUFFICIENT",
            message=f"minimising {n_atoms} atoms for up to {nsteps} steps: at this "
            f"machine's plausible {rate:g} atom-steps/s "
            f"(compute.plausible_em_atom_steps_per_s) a {walltime_min} min booking "
            f"reaches about {reach} steps, so the job is killed at the time limit "
            "unless it converges first. Book more walltime, or check that the system is "
            "the size you meant — an unexpectedly large box is the usual cause.",
            suggestion="raise walltime_min, or check the atom count against the system "
            "you intended",
        )
    return Finding(
        id="EM_LENGTH",
        message=f"minimising {n_atoms} atoms, at most {nsteps} steps; {walltime_min} min "
        f"covers about {reach} steps here",
    )
