"""Whether a time series has settled: the numbers a researcher's criterion is judged on.

Equilibration is detected as Chodera does (J. Chem. Theory Comput. 12, 1799 (2016),
doi:10.1021/acs.jctc.5b00784, the method of pymbar's ``detectEquilibration``): of every start
time t0, the one that leaves the most effectively uncorrelated samples, (T - t0 + 1) / g,
where g is the statistical inefficiency of the series from t0 (Chodera et al., J. Chem.
Theory Comput. 3, 26 (2007); ``statisticalInefficiency`` with mintime 3). After t0 the series
is described by its mean with a standard error that counts g, its drift, and the difference
between the means of its two halves in units of their combined standard error.

None of this says the run is equilibrated. That is a criterion, and the criterion is the
researcher's: given one, the sensor says whether it was met; without one it says nothing
about it.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning
from myrmidon.engines.inputs import resolve_in_ws

#: candidate start times tried at most; the series is sampled evenly between them
MAX_START_TIMES = 200


def statistical_inefficiency(a: np.ndarray, mintime: int = 3) -> float:
    """g = 1 + 2 sum_t (1 - t/N) C(t), summed until C first drops to zero after mintime."""
    n = len(a)
    da = a - a.mean()
    var = float(np.mean(da * da))
    if n < 3 or var == 0.0:
        return 1.0
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(da, size)
    acov = np.fft.irfft(f * np.conj(f), size)[:n]  # sum_i da[i] da[i+t]
    t = np.arange(1, n - 1)
    c = acov[1:n - 1] / (n - t) / var
    stop = np.nonzero((c <= 0) & (t > mintime))[0]
    end = stop[0] if stop.size else len(t)
    return max(1.0, 1.0 + 2.0 * float(np.sum(c[:end] * (1.0 - t[:end] / n))))


def equilibration_start(a: np.ndarray) -> tuple[int, float, float]:
    """(t0 index, g from t0, effective samples from t0) maximising the effective samples."""
    n = len(a)
    best = (0, 1.0, 1.0)
    for t0 in range(0, n - 1, max(1, n // MAX_START_TIMES)):
        g = statistical_inefficiency(a[t0:])
        n_eff = (n - t0 + 1) / g
        if n_eff > best[2]:
            best = (t0, g, n_eff)
    return best


def _mean_se(a: np.ndarray) -> tuple[float, float]:
    g = statistical_inefficiency(a)
    return float(a.mean()), float(a.std(ddof=1) * np.sqrt(g / len(a))) if len(a) > 1 else 0.0


class SeriesConvergence(BaseModel):
    name: str
    unit: str | None = None
    n_points: int
    span_ps: float
    equilibrated_from_ps: float  # t0: where the settled part begins
    discarded_fraction: float
    statistical_inefficiency: float  # of the settled part
    effective_samples: float
    mean: float  # of the settled part
    standard_error: float  # counts the correlation (g)
    drift_per_ns: float | None = None  # of the settled part
    drift_standard_error_per_ns: float | None = None
    halves_mean_difference: float | None = None  # second half minus first
    halves_difference_sigma: float | None = None  # in combined standard errors
    criterion_met: bool | None = None  # only when a criterion was given
    criterion_detail: list[str] = Field(default_factory=list)


def describe(name: str, t_ps: np.ndarray, y: np.ndarray, unit: str | None,
             max_drift: float | None, max_sigma: float | None,
             min_settled: float = 0.5) -> SeriesConvergence:
    t0, g, n_eff = equilibration_start(y)
    ts, ys = t_ps[t0:], y[t0:]
    mean, se = _mean_se(ys)
    drift = drift_se = None
    if len(ys) > 2 and ts[-1] > ts[0]:
        slope = float(np.polyfit(ts, ys, 1)[0])
        drift = slope * 1000.0
        if len(ys) > 3:  # the slope's standard error from the residuals, counting g
            resid = ys - np.polyval(np.polyfit(ts, ys, 1), ts)
            var = float(np.sum(resid**2)) / (len(ys) - 2) / float(np.sum((ts - ts.mean())**2))
            drift_se = float(np.sqrt(var * g)) * 1000.0
    diff = sigma = None
    half = len(ys) // 2
    if half >= 2:
        (m1, s1), (m2, s2) = _mean_se(ys[:half]), _mean_se(ys[half:])
        diff = m2 - m1
        combined = float(np.hypot(s1, s2))
        sigma = abs(diff) / combined if combined > 0 else None
    met = None
    detail: list[str] = []
    settled = 1.0 - t0 / len(y)
    if (max_drift is not None or max_sigma is not None) and settled < min_settled:
        # a series still drifting is discarded almost whole, and its last stretch looks flat
        detail.append(f"no verdict: the settled part is the last {settled:.0%} of the series, "
                      f"less than {min_settled:.0%}")
    elif max_drift is not None or max_sigma is not None:
        checks = []
        if max_drift is not None:
            ok = drift is not None and abs(drift) <= max_drift
            checks.append(ok)
            detail.append(f"|drift| {abs(drift) if drift is not None else 'unmeasured'} "
                          f"{'<=' if ok else '>'} {max_drift} {unit or ''}/ns")
        if max_sigma is not None:
            ok = sigma is not None and sigma <= max_sigma
            checks.append(ok)
            detail.append(f"halves differ by {sigma if sigma is not None else 'unmeasured'} "
                          f"sigma {'<=' if ok else '>'} {max_sigma}")
        met = all(checks)
    return SeriesConvergence(
        name=name, unit=unit, n_points=len(y), span_ps=float(t_ps[-1] - t_ps[0]),
        equilibrated_from_ps=float(t_ps[t0]), discarded_fraction=t0 / len(y),
        statistical_inefficiency=g, effective_samples=n_eff, mean=mean, standard_error=se,
        drift_per_ns=drift, drift_standard_error_per_ns=drift_se,
        halves_mean_difference=diff, halves_difference_sigma=sigma,
        criterion_met=met, criterion_detail=detail,
    )


def read_series(path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(time in ps, name -> values) from a gmx .xvg or a traj_analyze .npz over time."""
    from myrmidon.analysis.energy import parse_xvg

    if str(path).endswith(".npz"):
        with np.load(path) as z:
            if "time_ps" not in z.files:
                raise ValueError(f"{path} is a series over {z.files[0]}, not time")
            return z["time_ps"], {k: z[k] for k in z.files if k != "time_ps"}
    with open(path) as fh:
        data, legends = parse_xvg(fh.read())
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} holds no data rows")
    names = legends if len(legends) == data.shape[1] - 1 else [
        f"column {i}" for i in range(1, data.shape[1])]
    return data[:, 0], {name: data[:, i] for i, name in enumerate(names, start=1)}


def _line(s: SeriesConvergence) -> str:
    drift = "unmeasured" if s.drift_per_ns is None else f"{s.drift_per_ns:.3g}/ns"
    return (f"{s.name}: settled from {s.equilibrated_from_ps:g} ps, {s.mean:.4g} ± "
            f"{s.standard_error:.2g}, drift {drift}")


class ConvergenceObservation(Observation):
    series: list[SeriesConvergence] = Field(default_factory=list)
    available_series: list[str] = Field(default_factory=list)


class ConvergenceParams(BaseModel):
    data: str = Field(description="a time series: gmx_energy's .xvg or traj_analyze's .npz")
    series: list[str] = Field(
        default_factory=list,
        description="which series of the file (legend names, e.g. 'Density'); empty = all",
    )
    max_drift_per_ns: float | None = Field(
        default=None, gt=0,
        description="the researcher's criterion: largest |drift| of the settled part, in the "
        "series' unit per ns. Unset = no verdict",
    )
    min_settled_fraction: float = Field(
        default=0.5, gt=0, le=1,
        description="no verdict when the settled part is a smaller fraction of the series "
        "than this: a series that is still drifting has most of it discarded, and its last "
        "stretch looks flat",
    )
    max_halves_sigma: float | None = Field(
        default=None, gt=0,
        description="the researcher's criterion: largest difference between the settled "
        "part's two half-means, in combined standard errors (e.g. 2). Unset = no verdict",
    )


@tool(
    "series_convergence",
    description="Measure whether a time series has settled: where its equilibrated part "
    "begins (Chodera 2016), that part's mean with a correlation-corrected standard error, "
    "its drift, and how far its two halves' means differ. Says whether a criterion was met "
    "only when the researcher's criterion is given; whether a run is equilibrated is "
    "theirs to decide.",
    returns=ConvergenceObservation,
    tags={"analysis"},
)
def series_convergence(ctx: ToolContext, params: ConvergenceParams) -> ConvergenceObservation:
    path = resolve_in_ws(ctx, params.data)
    try:
        t_ps, columns = read_series(path)
    except (OSError, ValueError) as exc:
        return ConvergenceObservation(ok=False, summary=f"no time series read: {exc}")
    unknown = [s for s in params.series if s not in columns]
    if unknown:
        return ConvergenceObservation(
            ok=False, summary=f"{unknown} not in {params.data}",
            available_series=list(columns))
    names = params.series or list(columns)
    unit = None
    if str(path).endswith(".npz"):
        unit = "Å"  # traj_analyze's series are in MDAnalysis units
    out = [describe(name, t_ps, columns[name], unit, params.max_drift_per_ns,
                    params.max_halves_sigma, params.min_settled_fraction)
           for name in names if len(columns[name]) >= 4]
    if not out:
        return ConvergenceObservation(ok=False, summary="fewer than four points in each series",
                                      available_series=list(columns))
    warnings = []
    for s in out:
        if s.discarded_fraction > 1.0 - params.min_settled_fraction:
            warnings.append(ValidationWarning(
                id="SERIES_SETTLED_LATE", severity="info",
                message=f"{s.name}: the settled part begins at {s.equilibrated_from_ps:g} of "
                f"{s.span_ps:g} ps; most of the run is still relaxing or drifting, and the "
                "statistics describe only its end",
                suggestion="a longer run before any verdict",
            ))
        if s.effective_samples < 10:
            warnings.append(ValidationWarning(
                id="SERIES_FEW_INDEPENDENT_SAMPLES", severity="info",
                message=f"{s.name}: {s.effective_samples:.1f} effectively independent samples "
                "after equilibration; its standard error and drift are not yet informative",
                suggestion="a longer run, not a different statistic",
            ))
    verdict = [s.criterion_met for s in out if s.criterion_met is not None]
    asked = params.max_drift_per_ns is not None or params.max_halves_sigma is not None
    withheld = "; ".join(d for s in out for d in s.criterion_detail if d.startswith("no verdict"))
    return ConvergenceObservation(
        ok=True,
        summary="; ".join(_line(s) for s in out)
        + (f"; criterion {'met' if all(verdict) else 'not met'}" if verdict
           else f"; {withheld}" if asked and withheld
           else "; no criterion given, so no verdict"),
        series=out,
        advisory=warnings,
        available_series=list(columns),
    )
