"""Sentinels: cheap predicates over a finished run's parsed values, declared on a plan
step and evaluated by the loop when the step's job has ended.

A sentinel guards a move against being big in blindness: a
successful job with a method fault — a heating that never reached its target — passes
every exit code and fails here. A violation stops the move and raises an event; nothing
here fixes anything. Every check reads the run's own output through the engine parsers
(``engines/mimic/diagnostics``, ``engines/gromacs/results``), never a number anyone typed.

A check that cannot be evaluated (no output, no series) is reported as not holding: a
guard that silently passes when blind is no guard.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from myrmidon.core.plan import Sentinel
from myrmidon.execution.jobstore import JobRecord


class SentinelResult(BaseModel):
    check: str
    holds: bool
    value: float | int | bool | str | None = None
    detail: str = ""


def _cpmd_text(record: JobRecord) -> str | None:
    out = Path(record.spec.cwd) / (record.spec.stdout or "cpmd.out")
    if not out.is_file():
        return None
    return out.read_text(errors="replace")


def _series(record: JobRecord) -> list[dict] | None:
    from myrmidon.engines.mimic.diagnostics import parse_cpmd_series

    text = _cpmd_text(record)
    return parse_cpmd_series(text) if text is not None else None


def t_qm_tail_in_range(record: JobRecord, args: dict) -> SentinelResult:
    """The mean QM-subsystem temperature over the last ``tail_fraction`` of MD steps lies
    within ``target_k`` ± ``tolerance_k``. The ionic temperature (TEMPP) stands in when
    the log prints no subsystem temperatures."""
    target, tol = float(args["target_k"]), float(args.get("tolerance_k", 10.0))
    fraction = float(args.get("tail_fraction", 0.1))
    steps = _series(record)
    if not steps:
        return SentinelResult(check="t_qm_tail_in_range", holds=False,
                              detail="no MD steps in the run's output")
    temps = [s["t_qm"] if s.get("t_qm") is not None else s["temp_k"] for s in steps]
    tail = temps[-max(1, math.ceil(len(temps) * fraction)):]
    mean = sum(tail) / len(tail)
    holds = abs(mean - target) <= tol
    return SentinelResult(
        check="t_qm_tail_in_range", holds=holds, value=round(mean, 2),
        detail=f"tail mean T {mean:.1f} K over the last {len(tail)} of {len(temps)} steps, "
               f"target {target:g} ± {tol:g} K")


def scf_iterations_under(record: JobRecord, args: dict) -> SentinelResult:
    """No MD step needed more than ``cap`` SCF iterations."""
    cap = int(args["cap"])
    steps = _series(record)
    if not steps:
        return SentinelResult(check="scf_iterations_under", holds=False,
                              detail="no MD steps in the run's output")
    worst = max((s["scf_iters"] or 0) for s in steps)
    return SentinelResult(check="scf_iterations_under", holds=worst <= cap, value=worst,
                          detail=f"most SCF iterations in one step {worst}, cap {cap}")


def no_nan(record: JobRecord, args: dict) -> SentinelResult:
    """Every parsed energy is a number."""
    steps = _series(record)
    if not steps:
        return SentinelResult(check="no_nan", holds=False,
                              detail="no MD steps in the run's output")
    bad = [s["nfi"] for s in steps
           if any(math.isnan(s[k]) or math.isinf(s[k]) for k in ("eks", "eclassic"))]
    return SentinelResult(check="no_nan", holds=not bad, value=len(bad),
                          detail=f"{len(bad)} step(s) with a non-finite energy" if bad
                          else f"{len(steps)} steps, every energy finite")


def finished_normally(record: JobRecord, args: dict) -> SentinelResult:
    """CPMD printed its end-of-run lines."""
    from myrmidon.engines.mimic.diagnostics import parse_cpmd_output

    text = _cpmd_text(record)
    if text is None:
        return SentinelResult(check="finished_normally", holds=False, detail="no output")
    facts = parse_cpmd_output(text)
    return SentinelResult(check="finished_normally", holds=bool(facts.get("finished_normally")),
                          value=bool(facts.get("finished_normally")),
                          detail=facts.get("diagnosis") or "the run ended normally")


def final_temperature_below(record: JobRecord, args: dict) -> SentinelResult:
    """The last ionic temperature is below ``limit_k`` — an annealing that cooled."""
    limit = float(args["limit_k"])
    steps = _series(record)
    if not steps:
        return SentinelResult(check="final_temperature_below", holds=False,
                              detail="no MD steps in the run's output")
    last = steps[-1]["temp_k"]
    return SentinelResult(check="final_temperature_below", holds=last < limit, value=last,
                          detail=f"final ionic temperature {last:g} K, limit {limit:g} K")


def conserved_energy_drift_under(record: JobRecord, args: dict) -> SentinelResult:
    """|drift of ECLASSIC| ≤ ``limit_ha_per_ps`` for a run whose header says the ionic
    temperature is not controlled (an NVE check)."""
    from myrmidon.engines.mimic.diagnostics import nve_drift

    limit = float(args["limit_ha_per_ps"])
    text = _cpmd_text(record)
    facts = nve_drift(text) if text is not None else {}
    drift = facts.get("eclassic_drift_ha_per_ps")
    if drift is None:
        return SentinelResult(
            check="conserved_energy_drift_under", holds=False,
            detail="no conserved-energy drift: the run's header does not say the ionic "
                   f"temperature is uncontrolled ({facts.get('ion_temperature_control')}), "
                   "or it has fewer than three steps")
    return SentinelResult(check="conserved_energy_drift_under", holds=abs(drift) <= limit,
                          value=drift, detail=f"ECLASSIC drift {drift:.3g} Ha/ps, limit {limit:g}")


CHECKS: dict[str, Callable[[JobRecord, dict], SentinelResult]] = {
    "t_qm_tail_in_range": t_qm_tail_in_range,
    "scf_iterations_under": scf_iterations_under,
    "no_nan": no_nan,
    "finished_normally": finished_normally,
    "final_temperature_below": final_temperature_below,
    "conserved_energy_drift_under": conserved_energy_drift_under,
}


def evaluate(sentinel: Sentinel, record: JobRecord) -> SentinelResult:
    check = CHECKS.get(sentinel.check)
    if check is None:
        return SentinelResult(check=sentinel.check, holds=False,
                              detail=f"no such check; known: {', '.join(sorted(CHECKS))}")
    try:
        return check(record, sentinel.args)
    except (KeyError, TypeError, ValueError) as exc:
        return SentinelResult(check=sentinel.check, holds=False,
                              detail=f"could not evaluate: {type(exc).__name__}: {exc}")
