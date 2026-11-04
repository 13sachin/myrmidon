"""series_convergence: whether a time series has settled, and from where."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.analysis.convergence import describe, read_series, verdict_line
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import Finding
from myrmidon.engines.inputs import resolve_in_ws


class ConvergenceParams(BaseModel):
    data: str = Field(description="a time series: gmx_energy's .xvg, traj_analyze's .npz, or "
                      "a run's .edr, whose named `series` are extracted first")
    series: list[str] = Field(
        default_factory=list,
        description="which series of the file (legend names, e.g. 'Density'); empty = all. "
        "Required for an .edr, spelled as GROMACS labels its terms",
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
    description="Measure whether a time series has settled: where its equilibrated part begins, "
                "that part's mean with a correlation-corrected error, its drift, and its halves' "
                "difference (Chodera 2016). Says whether a criterion is met only when the "
                "researcher gives one.",
    tags={"analysis"},
    contract=Contract(consumes={"data": ("data", "energy")}, produces=(Produces("data"),)),
)
def series_convergence(ctx: ToolContext, params: ConvergenceParams) -> Observation:
    from myrmidon.core.artifacts import ArtifactRef, fresh_path

    path = resolve_in_ws(ctx, params.data)
    stage = ctx.workspace.stage("analysis")
    extracted: list[ArtifactRef] = []
    if path.suffix == ".edr":
        if not params.series:
            return Observation(ok=False, summary=f"{params.data} is an energy file: "
                                          "name the terms to read in `series`")
        path, failure = _series_from_energy(ctx, path, params.series, stage)
        if failure is not None:
            return failure
        extracted.append(ArtifactRef.from_path(path, "data", root=ctx.workspace.root))
    try:
        t_ps, columns = read_series(path)
    except (OSError, ValueError) as exc:
        return Observation(ok=False, summary=f"no time series read: {exc}")
    unknown = [s for s in params.series if s not in columns]
    if unknown:
        return Observation(
            ok=False, summary=f"{unknown} not in {params.data}",
            available_series=list(columns))
    names = params.series or list(columns)

    def unit_of(name: str) -> str:
        if str(path).endswith(".npz"):
            return "Å"  # traj_analyze's series are in MDAnalysis units
        from myrmidon.analysis.energy import UNIT_BY_TERM

        return UNIT_BY_TERM.get(name.lower(), "kJ/mol")  # gmx energy's, by its term

    out = [describe(name, t_ps, columns[name], unit_of(name), params.max_drift_per_ns,
                    params.max_halves_sigma, params.min_settled_fraction)
           for name in names if len(columns[name]) >= 4]
    if not out:
        return Observation(ok=False, summary="fewer than four points in each series",
                                      available_series=list(columns))
    warnings = []
    for s in out:
        if s.discarded_fraction > 1.0 - params.min_settled_fraction:
            warnings.append(Finding(
                id="SERIES_SETTLED_LATE",
                message=f"{s.name}: the settled part begins at {s.equilibrated_from_ps:g} of "
                f"{s.span_ps:g} ps; most of the run is still relaxing or drifting, and the "
                "statistics describe only its end",
                suggestion="a longer run before any verdict",
            ))
        if s.effective_samples < 10:
            warnings.append(Finding(
                id="SERIES_FEW_INDEPENDENT_SAMPLES",
                message=f"{s.name}: {s.effective_samples:.1f} effectively independent samples "
                "after equilibration; its standard error and drift are not yet informative",
                suggestion="a longer run, not a different statistic",
            ))
    verdict = [s.criterion_met for s in out if s.criterion_met is not None]
    asked = params.max_drift_per_ns is not None or params.max_halves_sigma is not None
    withheld = "; ".join(d for s in out for d in s.criterion_detail if d.startswith("no verdict"))
    # the verdict and its statistics persist as a file the ledger names by hash, so the
    # study's report states them from it: the observation alone is not kept
    import json

    table = fresh_path(stage / f"{Path(str(path)).stem}_convergence.json")
    table.write_text(json.dumps([s.model_dump(mode="json") for s in out], indent=1))
    return Observation(
        ok=True,
        summary="; ".join(verdict_line(s) for s in out)
        + (f"; criterion {'met' if all(verdict) else 'not met'}" if verdict
           else f"; {withheld}" if asked and withheld
           else "; no criterion given, so no verdict"),
        series=out,
        artifacts=[ArtifactRef.from_path(table, "data", root=ctx.workspace.root), *extracted],
        findings=warnings,
        available_series=list(columns),
    )


def _series_from_energy(ctx: ToolContext, edr: Path, terms: list[str], stage: Path
                        ) -> tuple[Path, Observation | None]:
    """The named terms of an energy file as an .xvg beside the analysis, through the
    engine; (the file, None), or (the energy file, the failure)."""
    from myrmidon.analysis.energy import energy_menu
    from myrmidon.engines.gromacs.common import extract_energy_terms, tail

    xvg, proc = extract_energy_terms(ctx, edr, terms, stage)
    if proc.returncode != 0 or not xvg.exists():
        return edr, Observation(
            ok=False,
            summary=f"gmx energy failed (exit {proc.returncode}) reading {terms} from "
            f"{edr.name}; available_series lists this file's terms",
            command=list(proc.args), log_tail=tail(proc.stderr),
            available_series=energy_menu(proc.stderr + "\n" + proc.stdout))
    return xvg, None
