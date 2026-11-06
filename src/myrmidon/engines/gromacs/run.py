"""What a run's input says about the run it will be: grompp's warnings, and whether the
booked walltime can cover the dynamics or the minimisation it carries."""

from __future__ import annotations

import re
from pathlib import Path

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation, ToolError
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


def posres_defines(mdp: Path) -> list[str]:
    """The ``-DPOSRES*`` switches the mdp's ``define`` turns on."""
    from myrmidon.engines.gromacs.mdp import parse_mdp_file

    return [d for d in parse_mdp_file(mdp).get("define", "").split()
            if d.startswith("-DPOSRES")]


def grompp(ctx: ToolContext, mdp: Path, structure: Path, topology: Path, index: Path | None,
           reference: Path | None, tpr: Path, cwd: Path, max_warn: int = 0, *,
           mdp_name: str = "", structure_name: str = "") -> Observation:
    """grompp as one call: the run input it wrote and its own notes as findings, or why it
    stopped. Restraints the mdp switches on hold atoms where the run starts unless told
    otherwise."""
    from myrmidon.engines.gromacs.common import run_gmx, tail

    args = ["grompp", "-f", str(mdp), "-c", str(structure), "-p", str(topology),
            "-o", str(tpr), "-po", str(cwd / "mdout.mdp"), "-maxwarn", str(max_warn)]
    if index is not None:
        args += ["-n", str(index)]
    notes: list[Finding] = []
    defines = posres_defines(mdp)
    if defines and reference is None:
        reference = structure
        notes.append(Finding(
            id="GROMPP_RESTRAINT_REFERENCE_DEFAULTED",
            message=f"{mdp_name or mdp.name} switches position restraints on "
            f"({' '.join(defines)}) and no restraint_reference was given: they hold atoms at "
            f"{structure_name or structure.name}, the structure this run starts from"))
    if reference is not None:
        args += ["-r", str(reference)]
    proc = run_gmx(ctx, args, cwd=cwd)
    command = list(proc.args)
    engine_notes = grompp_warnings(proc.stderr)
    notes += engine_notes
    if proc.returncode != 0 or not tpr.exists():
        errors = [w for w in engine_notes if w.id in ("GROMPP_ERROR", "GROMPP_FATAL")]
        warned = [w.id for w in engine_notes if w.id in ("GROMPP_WARNING", "GROMPP_NET_CHARGE")]
        hint = ("neutralise first: gmx_genion builds its own run input from the structure, "
                "then this grompp runs on the structure and topology it returns"
                if "GROMPP_NET_CHARGE" in warned else
                "fix what grompp's error names" if errors else
                "read grompp's words in the log tail and fix the cause; max_warn only hides it")
        first = (errors or [w for w in engine_notes if w.id in warned])[:1]
        return Observation(
            ok=False,
            summary=f"grompp failed (exit {proc.returncode})"
            + (f": {errors[0].message}" if errors else
               f" on grompp warning(s) {warned} — read them, fix the cause; max_warn only "
               "hides them" if warned else " — see log tail"),
            command=command,
            findings=notes,
            log_tail=tail(proc.stderr, 40),
            error=ToolError(code=first[0].id if first else "GROMPP_FAILED",
                            message=first[0].message if first
                            else f"grompp exited {proc.returncode}", hint=hint),
        )
    return Observation(
        ok=True,
        summary=f"Run input written: {ctx.workspace.relpath(tpr)}"
        + (f" ({len(notes)} grompp note(s) attached)" if notes else ""),
        command=command,
        findings=notes,
        artifacts=[ArtifactRef.from_path(tpr, "tpr", root=ctx.workspace.root)],
    )


def compile_stage_run_input(ctx: ToolContext, mdp: Path, files: dict[str, Path], stage_dir: Path,
                            name: str, max_warn: int = 0) -> Observation | None:
    """A stage's .tpr, one per stage since the output stride is the stage's own. The mdp's
    sensors judge it first, and a blocking one refuses as grompp's own error does. None while
    the stage's structure or topology is not yet there."""
    from myrmidon.engines.gromacs.mdp import parse_mdp_file, validate_mdp
    from myrmidon.knowledge.catalog import for_engine

    structure, topology = files.get("structure"), files.get("topology")
    if structure is None or topology is None:
        return None
    found, _ = validate_mdp(parse_mdp_file(mdp), for_engine(ctx.engines, "gromacs"))
    stopping = [f for f in found if f.severity == "block"]
    if stopping:
        return Observation(
            ok=False, summary=f"{mdp.name} would not run as meant: "
            + " ".join(f.message for f in stopping),
            findings=found,
            error=ToolError(code=stopping[0].id, message=stopping[0].message,
                            hint=stopping[0].suggestion or ""))
    out = grompp(ctx, mdp, structure, topology, files.get("index"), None,
                 stage_dir / f"{name}.tpr", stage_dir, max_warn)
    out.findings = [*found, *out.findings]
    return out
