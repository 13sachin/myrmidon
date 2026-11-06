"""QM/MM diagnostics: the CPMD output parsed into typed convergence facts, and, for a run
that did not work, the engine's own words — the log's tail, its stderr and the error file
beside it — for the model or the researcher to read."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.contracts import Contract
from myrmidon.core.observation import Observation
from myrmidon.core.tool import FileReader, ToolContext, tool
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.mimic.diagnostics import (
    cpmd_facts,
    dipole_facts,
    is_cpmd_output,
    parse_cpmd_output,
    read_local_errors,
)
from myrmidon.text import LOG_TAIL_LINES, tail


class MimicDiagnoseParams(BaseModel):
    cpmd_output: str = Field(default="", description="CPMD stdout log (default: "
                             "<stage>/cpmd.out)")
    stage: str = Field(default="qmmm")


@tool(
    "mimic_diagnose",
    description="Parse a MiMiC/CPMD log into facts: MD steps, temperatures, SCF iterations, "
                "energies, and whether a single point's wavefunction converged. ok=False when the "
                "run did not finish normally or did not converge, with the log's last lines to "
                "read the cause from.",
    tags={"mimic", "qmmm", "analysis"},
    requires={"mimicpy"},
    contract=Contract(consumes={"cpmd_output": "qm_log"}),
    reads=(FileReader("cpmd_output", is_cpmd_output, "a CPMD output"),),
)
def mimic_diagnose(ctx: ToolContext, params: MimicDiagnoseParams) -> Observation:
    """The designated reader of a CPMD log, for both run types it can be.

    `ok` is False when the run did not finish normally, when CPMD's error file or a signal
    on stderr witnesses an abort, when the wavefunction did not converge, and when nothing
    at all could be read from the log. The last two matter because a single point that
    stops at its iteration cap exits 0 and writes a RESTART and a reference file that look
    exactly like a good one's. A tool that cannot tell those apart is worse than one that
    is absent, because it is the one a chain is built to trust. Why a run failed is not
    classified here: the log's tail goes back with the observation for the reader.
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
                return Observation(
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
        located
        + head
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
