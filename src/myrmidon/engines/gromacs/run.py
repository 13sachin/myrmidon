"""What a run's input says about the run it will be: grompp's warnings, and whether the
booked walltime can cover the dynamics or the minimisation it carries."""

from __future__ import annotations

import re
from pathlib import Path

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation, ToolError
from myrmidon.core.tool import ToolContext
from myrmidon.core.validation import Finding
from myrmidon.engines.gromacs.common import run_gmx

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
    from myrmidon.engines.gromacs.common import tail

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


_TPR_FIELD_RE = re.compile(
    r"^\s*(nsteps|dt|init-step|integrator|natoms)\s*=\s*(\S+)", re.MULTILINE
)


#: integrators that minimise: they take steps, not time, so "the run is 0.05 ns" is not
#: a statement about them and a ns/day ceiling cannot judge their walltime
EM_INTEGRATORS = frozenset({"steep", "cg", "l-bfgs"})


def tpr_facts(ctx: ToolContext, tpr) -> dict[str, str]:
    """What the .tpr says about the run it carries, read with ``gmx dump -s``."""
    proc = run_gmx(ctx, ["dump", "-s", str(tpr)], cwd=tpr.parent)
    return dict(_TPR_FIELD_RE.findall(proc.stdout))


def tpr_run_length_ns(ctx: ToolContext, tpr) -> float | None:
    """Simulated time the .tpr carries (nsteps × dt, minus init-step), read
    with ``gmx dump -s`` — the number a walltime booking has to be able to cover."""
    fields = tpr_facts(ctx, tpr)
    try:
        nsteps, dt = int(fields["nsteps"]), float(fields["dt"])
    except (KeyError, ValueError):
        return None
    if nsteps < 0:
        return None  # -1: run until stopped
    return round(max(0, nsteps - int(fields.get("init-step", 0))) * dt / 1000.0, 6)


#: the walltime a run is booked for when nothing sizes it
FALLBACK_WALLTIME_MIN = 30


def beside(ctx: ToolContext, run_input: Path) -> Path:
    """Where a run writes: beside the file it runs on (its .tpr, the checkpoint it continues)
    when the study holds it, so the outputs and what they came from stay together for anyone
    reading the files; else the study's own run stage (a file the researcher brought)."""
    root = ctx.workspace.root.resolve()
    return (run_input.parent if run_input.resolve().is_relative_to(root)
            else ctx.workspace.stage("run"))


def mdrun_walltime(ctx: ToolContext, tpr, cores: int) -> tuple[int | None, str]:
    """(minutes, why) the run a .tpr carries needs here: a minimisation's fixed walltime
    from the stage templates, a dynamics run's length at the pace this study's runs, then
    this machine's, measured (knowledge/stage.py); (None, "") when the .tpr says neither."""
    from myrmidon.execution.pace import read_paces, study_paces
    from myrmidon.knowledge.stage import TemplateStore, booked_minutes, walltime

    fields = tpr_facts(ctx, tpr)
    if fields.get("integrator") in EM_INTEGRATORS:
        fixed = next((t.pricing["walltime_min"] for t in TemplateStore.default().all()
                      if t.engine == "gromacs" and "minimize" in t.kinds
                      and "walltime_min" in t.pricing), None)
        return (booked_minutes(fixed, 0), f"{fixed:g} min, the walltime for a minimisation"
                ) if fixed else (None, "")
    ns = tpr_run_length_ns(ctx, tpr)
    try:
        atoms = int(fields["natoms"])
    except (KeyError, ValueError):
        atoms = None
    if not ns:
        return None, ""
    minutes, why, measured = walltime(
        ns=ns, atoms=atoms, cores=cores,
        paces=[*study_paces(ctx.workspace), *read_paces(ctx.config.studies_root)],
        compute=ctx.config.compute, booking=True)
    if not measured:  # a floor is no booking: an unmeasured run gets the machine's least
        least = ctx.config.compute.unmeasured_walltime_min
        return (max(booked_minutes(minutes), least),
                why + f"; booked at least {least} min (compute.unmeasured_walltime_min) "
                "until a run here is measured")
    return booked_minutes(minutes), why


_CPT_OUTPUT_RE = re.compile(r"output filename\s*=\s*(\S+)")


def checkpoint_output_names(ctx: ToolContext, cpt) -> list[str]:
    """The output file names recorded in a checkpoint (``gmx dump -cp``) —
    what mdrun -append will compare VERBATIM against its own -deffnm expansion."""
    proc = run_gmx(ctx, ["dump", "-cp", str(cpt)], cwd=cpt.parent)
    return _CPT_OUTPUT_RE.findall(proc.stdout)


def stage_inputs(ctx: ToolContext, mandate, stage, prepared: dict, continuing: bool) -> dict:
    """What a classical stage's run reads besides what was prepared: to continue, the checkpoint
    its last run left; a frame batch, the trajectory of the run it starts from, with the
    topology and structure it builds each frame from."""
    from myrmidon.core.lineage import stage_left, stage_run

    out: dict[str, str] = {}
    if continuing:
        left = {a.role: a.path for a in stage_left(ctx.workspace, stage.name)}
        if "checkpoint" in left:
            out["checkpoint"] = left["checkpoint"]
    if frame_batch(stage) and "trajectory" not in prepared:
        begins = mandate.method.starts_from(stage)
        run = stage_run(ctx.workspace, begins.name) if begins else None
        found = next((a.path for a in (run[0] if run else []) if a.role == "trajectory"), None)
        if found:
            out["trajectory"] = found
    if frame_batch(stage):
        for kind in ("topology", "structure", "index"):
            named = next((f.path for f in mandate.system.files if f.kind == kind), None)
            if named and kind not in prepared:
                out[kind] = named
    return out


def frame_batch(stage) -> bool:
    """Whether a classical stage works over frames that exist: a single point recomputes each,
    a minimisation with a scan of frames minimises each."""
    return stage.kind == "single_point" or (
        stage.kind == "minimize" and stage.scan is not None
        and stage.scan.parameter == "frames")


def job_findings(ctx: ToolContext, stage, inputs: dict[str, Path], walltime_min: int
                 ) -> list[Finding]:
    """What stops a classical run or is worth saying of it: the input it must have, whether the
    booked walltime can cover what its .tpr carries, a checkpoint that does not name this run's
    files, a coupled .tpr rerun as if it were classical."""
    from myrmidon.core.validation import Precondition

    findings: list[Finding] = []
    if frame_batch(stage) and "trajectory" not in inputs:
        return [Precondition(
            id="STAGE_NO_FRAMES",
            message=f"stage {stage.name} works over frames, and no collected run it starts from "
            "wrote a trajectory",
            suggestion="run the stage it starts from first, or name the frames: inputs "
            "{trajectory: <path>}")]
    if stage.kind == "minimize" and frame_batch(stage):
        return findings
    tpr = inputs.get("tpr")
    if tpr is None:
        return [Precondition(
            id="STAGE_NOT_PREPARED",
            message=f"stage {stage.name} has no compiled run input",
            suggestion=f"stage_prepare stage={stage.name}: its structure and topology must be "
            "there for it to compile")]
    fields = tpr_facts(ctx, tpr)
    if stage.kind == "single_point":
        if fields.get("integrator") == "mimic":
            findings.append(Precondition(
                id="RERUN_TPR_COUPLED",
                message=f"{tpr.name} is a coupled run input: a classical rerun evaluates the MM "
                "side alone and drops the QM region's electrostatics",
                suggestion="build the rerun's run input from a classical stage (a single_point "
                "stage with regime mm, for a force comparison)"))
        return findings
    if "checkpoint" in inputs:
        names = checkpoint_output_names(ctx, inputs["checkpoint"])
        where = beside(ctx, inputs["checkpoint"])
        wrong = [n for n in names if Path(n).stem != stage.name
                 or (Path(n).is_absolute() and Path(n).parent != where)]
        if wrong:
            findings.append(Precondition(
                id="CPT_NAME_MISMATCH",
                message=f"the checkpoint records outputs {names[:4]} but this continuation would "
                f"write {stage.name}.* in {where.name}. mdrun -append compares these names "
                "verbatim and aborts; dropping -cpi would restart from the .tpr and silently "
                "not continue",
                suggestion="continue the run from the directory it wrote"))
        return findings
    if fields.get("integrator") in EM_INTEGRATORS:
        try:
            steps, atoms = int(fields["nsteps"]), int(fields["natoms"])
        except (KeyError, ValueError):
            steps = atoms = 0
        if steps > 0 and atoms > 0:
            findings.append(minimisation_warning(steps, atoms, walltime_min, ctx))
    else:
        length = tpr_run_length_ns(ctx, tpr)
        if length is not None:
            findings.append(throughput_warning(length, walltime_min, ctx))
    return findings


def size_walltime(ctx: ToolContext, stage, inputs: dict[str, Path], layout, cores: int
                  ) -> tuple[int | None, str]:
    """(minutes, why) the run the stage's .tpr carries needs here; a frame batch by its
    template's fixed walltime for each frame."""
    from myrmidon.knowledge.stage import TemplateStore, templates_for

    if frame_batch(stage):
        fixed = next((t.pricing["walltime_min"] for t in templates_for(
            stage.model_dump(mode="json"), TemplateStore.default())
            if "walltime_min" in t.pricing), 30)
        frames = len(stage.scan.values) if stage.scan is not None else 1
        return int(fixed * frames), f"{fixed:g} min, the template's for each of {frames} frame(s)"
    tpr = inputs.get("tpr")
    return mdrun_walltime(ctx, tpr, cores) if tpr is not None else (None, "no run input")


def build_job(ctx: ToolContext, stage, inputs: dict[str, Path], layout, request):
    """A classical stage as a job: an mdrun, its continuation from a checkpoint, a rerun over
    existing frames, or a batch minimising frames one by one."""
    from myrmidon.engines.gromacs.batch import minimize_frames_job
    from myrmidon.engines.gromacs.common import gmx_bin
    from myrmidon.engines.registry import StageJob
    from myrmidon.execution.backend import JobSpec

    if stage.kind == "minimize" and frame_batch(stage):
        return minimize_frames_job(ctx, stage, inputs, layout, request)
    tpr = inputs["tpr"]
    run_dir = beside(ctx, inputs.get("checkpoint", tpr))
    name = stage.name
    argv = [gmx_bin(ctx), "mdrun", "-s", str(tpr), "-deffnm", name]
    if stage.kind == "single_point":
        run_dir = ctx.workspace.run_dir(ctx.workspace.stage(name), name)
        argv = [gmx_bin(ctx), "mdrun", "-s", str(tpr), "-rerun", str(inputs["trajectory"]),
                "-deffnm", name, "-o", f"{name}.trr", "-e", f"{name}.edr", "-g", f"{name}.log"]
    artifacts = [ArtifactRef.from_path(tpr, "tpr", root=ctx.workspace.root)]
    if "checkpoint" in inputs:
        extended = run_dir / f"{name}_ext.tpr"
        args = ["convert-tpr", "-s", str(tpr), "-o", str(extended)]
        if request.extend_ps:
            args += ["-extend", str(request.extend_ps)]
        else:  # a run stopped at its walltime goes on to the end its .tpr states
            args += ["-until", str(_end_ps(ctx, tpr))]
        converted = run_gmx(ctx, args, cwd=run_dir)
        if converted.returncode != 0 or not extended.exists():
            raise ValueError(f"convert-tpr failed (exit {converted.returncode}): "
                             f"{converted.stderr[-400:]}")
        argv = [gmx_bin(ctx), "mdrun", "-s", str(extended), "-cpi", str(inputs["checkpoint"]),
                "-deffnm", name, "-append"]
        artifacts = [ArtifactRef.from_path(extended, "tpr", root=ctx.workspace.root)]
    if layout.cpus_per_task > 1:
        argv += ["-ntomp", str(layout.cpus_per_task)]
    spec = JobSpec(argv=argv, cwd=run_dir, stdout=f"{name}.stdout", stderr=f"{name}.stderr",
                   resources=request.resources())
    return StageJob(
        spec=spec, kind="gromacs", label=f"stage:{name}", artifacts=artifacts,
        summary=f"outputs {name}.* in {ctx.workspace.relpath(run_dir)}")


def _end_ps(ctx: ToolContext, tpr: Path) -> float:
    """The simulated time a .tpr's run ends at."""
    fields = tpr_facts(ctx, tpr)
    return int(fields.get("nsteps", 0)) * float(fields.get("dt", 0))
