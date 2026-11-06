"""A coupled MiMiC run as a stage's job: what stops it, what it books, how it starts, and what
it left when it ended. CPMD is the main program (its stdout is the QM log); GROMACS runs as the
coupled partner."""

from __future__ import annotations

import re
from pathlib import Path

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.gromacs.collect import collect_run, spec_deffnm
from myrmidon.engines.inputs import require_input, resolve_in_ws
from myrmidon.engines.mimic.cpmd_input import (
    cpmd_findings,
    get_keyword,
    parse_cpmd,
    render_cpmd,
)
from myrmidon.engines.mimic.diagnostics import (
    cpmd_facts,
    dipole_facts,
    parse_cpmd_output,
    read_local_errors,
)
from myrmidon.engines.mimic.launch import (
    GEOMETRY_NAME,
    align_mimic_paths,
    stage_geometry,
    stage_restart,
)
from myrmidon.engines.mimic.system import pp_findings
from myrmidon.engines.registry import StageJob
from myrmidon.execution.backend import CoupledCmd, JobSpec
from myrmidon.execution.collect import collector, progress
from myrmidon.knowledge.catalog import for_engine
from myrmidon.text import LOG_TAIL_LINES, tail


def coupled_findings(ctx: ToolContext, cpmd_input: str, tpr: str, restart_from: str | None,
                     geometry_from: str | None) -> tuple[list[Finding], list[str]]:
    """What stops a coupled run before it is booked, and what is worth saying of it."""
    warnings: list[Finding] = []
    checked = ["input_files", "cpmd_engine", "cpmd_sections", "charge_multiplicity", "qm_cell"]

    cpmd_info = ctx.engines.get("cpmd")
    if cpmd_info is None or not cpmd_info.available:
        warnings.append(
            Precondition(
                id="CPMD_UNAVAILABLE",
                message="CPMD is not available on this machine"
                + (f" — {cpmd_info.hint}" if cpmd_info else ""),
            )
        )

    inp, missing = require_input(ctx, "cpmd_input", cpmd_input, "INPUT_MISSING_CPMD",
                                 f"cpmd input not found: {cpmd_input}")
    if missing:
        warnings.append(missing)
        return warnings, checked
    _, missing = require_input(ctx, "tpr", tpr, message=f"tpr not found: {tpr}")
    if missing:
        warnings.append(missing)

    sections = parse_cpmd(inp.read_text())
    warnings.extend(cpmd_findings(sections, for_engine(ctx.engines, "cpmd"),
                                  restart_from))
    uncovered, ran = pp_findings(sections, inp.parent)
    if ran:
        checked.append("pseudopotentials")
        warnings.extend(uncovered)
    if "MIMIC" not in sections:
        warnings.append(
            Precondition(
                id="CPMD_NO_MIMIC_SECTION",
                message="input has no &MIMIC section — not a MiMiC run input",
            )
        )
    else:
        if get_keyword(sections["MIMIC"], "PATHS") is None:
            warnings.append(
                Precondition(
                    id="MIMIC_PATHS_MISSING",
                    message="&MIMIC has no PATHS — the coupled programs cannot find "
                    "each other",
                )
            )
    system = sections.get("SYSTEM", [])
    if get_keyword(system, "CHARGE") is None:
        warnings.append(
            Precondition(
                id="QM_CHARGE_MISSING",
                message="&SYSTEM has no CHARGE — run mimic_prepare (charge must be "
                "explicit, never assumed)",
            )
        )
    multiplicity = get_keyword(system, "MULTIPLICITY")
    sidecar = inp.with_name(inp.name + ".myrmidon.json")
    if multiplicity is None and not _reviewed(sidecar):
        warnings.append(
            Precondition(
                id="QM_MULTIPLICITY_UNCONFIRMED",
                message="the input states no multiplicity — run mimic_prepare (closed-shell "
                "runs carry it in the .myrmidon.json sidecar, open-shell runs in &SYSTEM)",
            )
        )
    checked.append("restart_available")
    wants_restart = any(
        line.strip().upper().startswith("RESTART") for line in sections.get("CPMD", [])
    )
    if wants_restart and not restart_from:
        warnings.append(
            Precondition(
                id="CPMD_RESTART_FILE_MISSING",
                message="the run input asks to RESTART, and no restart_from was given: "
                "CPMD reads the wavefunction from the file LATEST names in the working "
                "directory and stops in ZHRWF with FILE NOT FOUND when it is not there.",
                suggestion="pass restart_from=<the previous segment's `restart` artifact>, "
                "or render an input without a RESTART line for a run that starts fresh",
            )
        )
    if restart_from:
        restart, missing = require_input(ctx, "restart_from", restart_from,
                                         "INPUT_MISSING_RESTART",
                                         f"restart file not found: {restart_from}")
        if missing:
            warnings.append(missing)
        else:
            warnings.extend(restart_as_recorded(ctx, restart, restart_from))

    checked.append("geometry_available")
    wants_geofile = any(
        line.strip().upper().startswith("RESTART") and "GEOFILE" in line.upper()
        for line in sections.get("CPMD", [])
    )
    if geometry_from:
        _, missing = require_input(ctx, "geometry_from", geometry_from,
                                   "INPUT_MISSING_GEOMETRY",
                                   f"geometry file not found: {geometry_from}")
        if missing:
            warnings.append(missing)
    elif wants_geofile and coupled_geometry(ctx, geometry_from, restart_from) is None:
        warnings.append(
            Precondition(
                id="CPMD_GEOFILE_WITHOUT_GEOMETRY",
                message="the run input's RESTART line has GEOFILE, so CPMD reads the ions' "
                "coordinates from a GEOMETRY file in the working directory, and no "
                "geometry_from was given nor a GEOMETRY found beside restart_from: CPMD "
                "stops in GEOFILE with DO NOT READ OLD COORDINATES.",
                suggestion="pass geometry_from=<the previous segment's `geometry` artifact> "
                "(or a snapshot of this box), or drop GEOFILE from the RESTART line",
            )
        )

    return warnings, checked


def _reviewed(sidecar) -> bool:
    """Whether this input's spin was settled for a stage: a sidecar beside it that the QM
    selection did not write (it states what the selection leaves, for a render to settle)."""
    import json

    try:
        return json.loads(sidecar.read_text()).get("written_by") != "qm_region"
    except (OSError, ValueError):
        return False


def restart_as_recorded(ctx: ToolContext, restart, given: str) -> list[Finding]:
    """The blocking finding when the restart a continuation names is not the file the
    ledger recorded under that path. A continuation is the continuation of a recorded run
    only while its restart hashes to what that run's collection recorded; a file replaced
    since — by another run writing into the same directory, or by hand — makes a chain the
    ledger cannot vouch for. A path no call ever produced has nothing to compare against
    and passes: the ledger records it as this call's input."""
    from myrmidon.core.artifacts import sha256_file

    recorded = ctx.workspace.ledger.last_output(ctx.workspace.relpath(restart))
    if recorded is None:
        return []
    now = sha256_file(restart)
    if now == recorded.sha256:
        return []
    return [Precondition(
        id="RESTART_NOT_AS_RECORDED",
        message=f"{given} is not the file the ledger recorded at that path: it hashes to "
                f"{now[:12]}…, the run that produced it recorded {recorded.sha256[:12]}…. "
                "A continuation from it would not continue the run the ledger shows.",
        suggestion="restart from the `restart` artifact job_status returned for the run "
                   "you mean to continue; its directory holds only that run's files",
    )]


RESTART_NAME = "RESTART.1"


LATEST_NAME = "LATEST"


def coupled_geometry(ctx: ToolContext, geometry_from: str | None, restart_from: str | None):
    """The GEOMETRY file a ``RESTART ... GEOFILE`` run reads its coordinates from: the one
    named, else the one the previous segment wrote beside the restart it continues from;
    None when there is none. CPMD stops in GEOFILE without it (real CPMD 4.3, coupled)."""
    if geometry_from:
        return resolve_in_ws(ctx, geometry_from)
    if restart_from:
        beside = resolve_in_ws(ctx, restart_from).parent / GEOMETRY_NAME
        if beside.is_file():
            return beside
    return None


#: the walltime a coupled run is booked for when nothing sizes it
FALLBACK_WALLTIME_MIN = 60


def stage_inputs(ctx: ToolContext, mandate, stage, prepared: dict, continuing: bool) -> dict:
    """What a coupled stage's run reads besides what was prepared: the restart and the geometry
    the run it starts from left, when its input asks to restart, and the structure its layout
    divides. An input with no RESTART line starts fresh and reads neither."""
    from myrmidon.core.lineage import stage_left, stage_run
    from myrmidon.engines.mimic.cpmd_input import has_keyword

    out: dict[str, str] = {}
    named = next((f.path for f in mandate.system.files if f.kind == "structure"), None)
    if named and "structure" not in prepared:
        out["structure"] = named
    if "cpmd_input" not in prepared:
        return out
    sections = parse_cpmd(ctx.workspace.resolve(prepared["cpmd_input"]).read_text())
    if not has_keyword(sections.get("CPMD", []), "RESTART"):
        return out
    begins = stage if continuing else mandate.method.starts_from(stage)
    if begins is None:
        return out
    if begins is stage:
        outputs = stage_left(ctx.workspace, stage.name)
    else:
        run = stage_run(ctx.workspace, begins.name)
        outputs = run[0] if run else []
    left = {a.role: a.path for a in outputs}
    for kind in ("restart", "geometry"):
        if kind in left and kind not in prepared:
            out[kind] = left[kind]
    return out


def job_findings(ctx: ToolContext, stage, inputs: dict[str, Path], walltime_min: int
                 ) -> list[Finding]:
    """What stops a coupled stage's run: the gate on its input, its compiled MM half, and the
    state it restarts from."""
    root = ctx.workspace.root
    missing = [k for k in ("cpmd_input", "tpr") if k not in inputs]
    if missing:
        return [Precondition(
            id="STAGE_NOT_PREPARED",
            message=f"stage {stage.name} has no {' and no '.join(missing)}",
            suggestion=f"stage_prepare stage={stage.name}")]

    def rel(kind: str) -> str | None:
        return str(inputs[kind].relative_to(root)) if kind in inputs else None

    found, _ = coupled_findings(ctx, rel("cpmd_input"), rel("tpr"), rel("restart"),
                                rel("geometry"))
    return found


def size_walltime(ctx: ToolContext, stage, inputs: dict[str, Path], layout, cores: int
                  ) -> tuple[int | None, str]:
    """(minutes, why) a coupled stage needs, from its input's MAXSTEP at the seconds per step
    measured here on the same QM region and cutoff, else at the machine's reference step."""
    from myrmidon.engines.mimic.cpmd_input import keyword_value, qm_geometry
    from myrmidon.execution.pace import read_paces, study_paces
    from myrmidon.knowledge.stage import booked_minutes, qm_walltime

    if "cpmd_input" not in inputs:
        return None, "no run input"
    sections = parse_cpmd(inputs["cpmd_input"].read_text())
    steps = keyword_value(sections, "&CPMD/MAXSTEP")
    # a wavefunction optimisation's MAXSTEP bounds its iterations: one step's worth of work
    length = 1 if stage.kind == "single_point" else (int(float(steps)) if steps else None)
    if length is None:
        return None, "the input states no MAXSTEP"
    cutoff = keyword_value(sections, "&SYSTEM/CUTOFF")
    minutes, why = qm_walltime(
        steps=length, qm_atoms=(qm_geometry(sections) or {}).get("n_atoms"),
        cutoff_ry=float(cutoff.split()[0]) if cutoff else None, cores=cores,
        paces=[*study_paces(ctx.workspace), *read_paces(ctx.config.studies_root)],
        compute=ctx.config.compute)
    return booked_minutes(minutes) if minutes is not None else None, why


def _reads_geometry(run_input: Path, run_dir: Path) -> Path:
    """The run input with GEOFILE on its RESTART line, so CPMD takes the ions from the GEOMETRY
    a snapshot gave it: a copy in the run's directory, with its sidecar beside it."""
    import shutil

    sections = parse_cpmd(run_input.read_text())
    sections["CPMD"] = [f"{line} GEOFILE" if line.strip().upper().startswith("RESTART")
                        and "GEOFILE" not in line.upper() else line
                        for line in sections.get("CPMD", [])]
    target = run_dir / run_input.name
    target.write_text(render_cpmd(sections))
    sidecar = run_input.with_name(run_input.name + ".myrmidon.json")
    if sidecar.is_file() and sidecar.parent != run_dir:
        shutil.copy2(sidecar, target.with_name(target.name + ".myrmidon.json"))
    return target


def build_job(ctx: ToolContext, stage, inputs: dict[str, Path], layout, request) -> StageJob:
    """A coupled stage as a job: CPMD and GROMACS started together in a directory of their own,
    never reused, since CPMD's output names are fixed and a second run would erase the first."""
    from myrmidon.engines.gromacs.common import gmx_bin

    root = ctx.workspace.root
    run_dir = ctx.workspace.run_dir(ctx.workspace.stage(stage.name), stage.name)
    run_input, aligned = align_mimic_paths(inputs["cpmd_input"], run_dir)
    if request.snapshot:
        run_input = _reads_geometry(run_input, run_dir)
    staged = []
    consumed = [ArtifactRef.from_path(inputs["cpmd_input"], "inputs.cpmd_input", root=root)]
    if "restart" in inputs:
        staged.append(stage_restart(inputs["restart"], run_dir))
    sections = parse_cpmd(run_input.read_text(errors="replace"))
    if any(line.strip().upper().startswith("RESTART") and "GEOFILE" in line.upper()
           for line in sections.get("CPMD", [])):
        geometry = inputs.get("geometry") or coupled_geometry(
            ctx, None, str(inputs["restart"].relative_to(root)) if "restart" in inputs else None)
        staged.append(stage_geometry(geometry, run_dir))
    threads = {"OMP_NUM_THREADS": str(layout.cpus_per_task)}
    spec = JobSpec(
        argv=[ctx.engines["cpmd"].path, str(run_input)], cwd=run_dir, stdout="cpmd.out",
        stderr="cpmd.err", mpi_ranks=layout.ntasks, env=threads,
        # what the node reads: the run input as aligned into the run directory, the tpr, and
        # the restart and geometry staged beside them
        receipt_inputs=[str(run_input), str(inputs["tpr"]), *map(str, staged)],
        resources=request.resources(),
        coupled=[CoupledCmd(
            argv=[gmx_bin(ctx), "mdrun", "-s", str(inputs["tpr"]), "-deffnm", stage.name],
            mpi_ranks=layout.partner_ntasks, stdout=f"{stage.name}.stdout",
            stderr=f"{stage.name}.stderr")])
    return StageJob(
        spec=spec, kind="mimic", label=f"stage:{stage.name}", meta={"deffnm": stage.name},
        inputs=consumed, artifacts=[ArtifactRef.from_path(run_input, "cpmd_input", root=root)],
        summary=f"CPMD x{layout.ntasks} + GROMACS x{layout.partner_ntasks} in "
        f"{ctx.workspace.relpath(run_dir)}" + (f". {aligned}" if aligned else ""))


@progress("mimic")
def _mimic_progress(record) -> tuple[None, str]:
    """CPMD's own counter for a coupled run in flight: the GROMACS log of a coupled run
    counts nothing and names the nominal step budget."""
    from pathlib import Path

    from myrmidon.engines.mimic.diagnostics import parse_cpmd_output

    out = Path(record.spec.cwd) / "cpmd.out"
    if not out.exists():
        return None, ""
    facts = parse_cpmd_output(out.read_text(errors="replace"))
    steps, temp = facts.get("md_steps_completed"), facts.get("t_qm_last")
    parts = [f"step {steps}" if steps is not None else "",
             f"T(QM) {temp} K" if temp is not None else ""]
    return None, " · ".join(p for p in parts if p)


@collector("mimic")
def _collect_mimic_job(ctx: ToolContext, record, state, exit_code) -> Observation:
    """A MiMiC run that has ended: the MM engine's outputs by role, plus the CPMD log,
    the force-matching reference it wrote, and — for a run that failed — CPMD's own error
    message and the tail of its log, for the reader."""
    from pathlib import Path

    from myrmidon.engines.mimic.diagnostics import (
        incomplete_run_note,
        parse_cpmd_output,
        read_local_errors,
        run_completed,
    )

    obs = collect_run(ctx, record, state, exit_code)
    # the count the GROMACS half put in its own summary, before the QM half adds to it
    before = len(obs.artifacts)
    if before == 0 and state.value == "COMPLETED":
        # a coupled run that produced no MM outputs is either misconfigured or was
        # collected under the wrong prefix; neither is a success
        obs.findings.append(Finding(
            id="MM_OUTPUTS_NOT_FOUND",
            message=(f"the run completed but no MM outputs were found under "
                     f"'{spec_deffnm(record)}.*' in {record.spec.cwd} — the QM side is "
                     "collected below, the GROMACS trajectory and energies are not. "
                     "Anything fitted from them would have no recorded provenance."),
        ))
    cwd = Path(record.spec.cwd)
    root = ctx.workspace.root
    cpmd_out = cwd / "cpmd.out"
    if cpmd_out.exists():
        obs.artifacts.append(ArtifactRef.from_path(cpmd_out, "qm_log", root=root))
        # CPMD's own error file witnesses an abort whatever the exit code says, and a run
        # that ended normally without doing what it was asked — a wavefunction
        # optimisation stopped at its iteration cap, dynamics short of its step count —
        # exits 0 with a RESTART written and is not a success either. The log's tail is
        # what the cause is read from, so a failed run carries it along with the GROMACS
        # half's
        text = cpmd_out.read_text(errors="replace")
        cpmd_error = read_local_errors(cwd)
        if cpmd_error and not obs.diagnosis:
            obs.ok = False
            obs.diagnosis = f"CPMD aborted: {cpmd_error}"
            obs.summary = f"{record.job_id} {state.value.lower()}: {obs.diagnosis}"
        facts = parse_cpmd_output(text)
        if state.value == "COMPLETED":
            from myrmidon.execution.pace import cpmd_pace, cpmd_run_input, record_pace

            pace = cpmd_pace(cpmd_out, cpmd_run_input(record), record.spec.resources,
                             f"{root.name} {record.label or record.job_id}")
            if pace is not None:
                record_pace(ctx.config.studies_root, pace)
        if obs.ok and not run_completed(facts):
            obs.ok = False
            obs.diagnosis = ("CPMD ended before it did what it was asked: "
                             + incomplete_run_note(facts))
            obs.summary = f"{record.job_id} {state.value.lower()}: {obs.diagnosis}"
        if not obs.ok:
            obs.log_tail = "\n".join(part for part in (
                obs.log_tail, "--- cpmd.out ---", tail(text, LOG_TAIL_LINES)) if part)
    for fm_json in sorted(cwd.glob("FMTRAJECTORY*.json")):
        obs.artifacts.append(ArtifactRef.from_path(fm_json, "fm_reference", root=root))
    # what the next segment restarts from, and what a decision about it reads. Without
    # these a continuation has nothing to reference, and CPMD stops in ZHRWF when no
    # restart reaches its directory.
    for name, role in (
        (RESTART_NAME, "restart"), (LATEST_NAME, "restart_index"),
        ("ENERGIES", "energies"), ("GEOMETRY", "geometry"),
        ("TRAJECTORY", "qm_trajectory"),
    ):
        path = cwd / name
        if path.exists():
            obs.artifacts.append(ArtifactRef.from_path(path, role, root=root))
    if obs.ok and len(obs.artifacts) != before:
        # the GROMACS half counted its own before the QM half was collected
        obs.summary = obs.summary.replace(
            f"{before} output artifact(s)", f"{len(obs.artifacts)} output artifact(s)"
        )
    return obs


def diagnose_log(ctx: ToolContext, path: Path) -> Observation:
    """The designated reader of a CPMD log, for both run types it can be.

    `ok` is False when the run did not finish normally, when CPMD's error file or a signal
    on stderr witnesses an abort, when the wavefunction did not converge, and when nothing
    at all could be read from the log. The last two matter because a single point that
    stops at its iteration cap exits 0 and writes a RESTART and a reference file that look
    exactly like a good one's. Why a run failed is not classified here: the log's tail goes
    back with the observation for the reader.
    """
    text = path.read_text(errors="replace")
    facts = parse_cpmd_output(text)
    cpmd_error = read_local_errors(path.parent)
    # crashes (signals, MPI aborts) land on stderr, not in the CPMD log
    err_path = path.with_suffix(".err")
    err_text = err_path.read_text(errors="replace") if err_path.exists() else ""
    sig = re.search(r"Caught signal \d+ \([^)]*\)", err_text) if err_text else None
    if sig and not cpmd_error:
        frame = re.search(r"^\s*1\s+0x[0-9a-f]+\s+(\S+?)\(\)", err_text, re.MULTILINE)
        cpmd_error = sig.group(0) + (f" in {frame.group(1)}" if frame else "")
    diagnosis = f"CPMD aborted: {cpmd_error}" if cpmd_error else None
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
    dipole_path = path.parent / "DIPOLE"
    dipole = (dipole_facts(text, dipole_path.read_text(errors="replace"))
              if dipole_path.is_file() else {})
    if dipole:
        bits.append(f"QM dipole over {dipole['dipole_steps']} steps: "
                    f"{dipole['dipole_debye_mean']:.3g} ± {dipole['dipole_debye_sd']:.2g} D")
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
        head
        + (", ".join(bits) if bits else "no step data parsed")
        + (f". {diagnosis}" if diagnosis else "")
        + (". The log's last lines are in log_tail." if not ok else "")
    )
    tail_text = None
    if not ok:
        tail_text = tail(text, LOG_TAIL_LINES)
        if err_text.strip():
            tail_text += "\n--- stderr ---\n" + tail(err_text, 20)
    # the ionic temperature as the run states it: what a next stage starts from
    ionic = cpmd_facts(text)
    return Observation(
        ok=ok,
        summary=summary,
        final_temperature_k=ionic.get("final_temperature_k"),
        tail_mean_temperature_k=ionic.get("tail_mean_temperature_k"),
        mean_temperature_k=ionic.get("mean_temperature_k"),
        sd_temperature_k=ionic.get("sd_temperature_k"),
        **dipole,
        md_steps_completed=facts.get("md_steps_completed"),
        md_steps_requested=facts.get("md_steps_requested"),
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
        cpmd_error=cpmd_error,
        log_tail=tail_text,
    )
