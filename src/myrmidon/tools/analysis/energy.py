"""Energy analysis: gmx energy terms parsed into typed statistics.

Means, standard deviations, and drift are computed here by harness code from
the .xvg data — the model never invents them."""

from __future__ import annotations

from pydantic import BaseModel, Field

from myrmidon.analysis.energy import UNIT_BY_TERM, energy_menu, parse_xvg, term_stats
from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import EnergyTermStats, Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import Finding
from myrmidon.engines.gromacs.common import extract_energy_terms, tail
from myrmidon.engines.inputs import resolve_in_ws


class EnergyParams(BaseModel):
    edr: str = Field(description="energy file (.edr) from a completed run")
    terms: list[str] = Field(
        description="energy terms to extract, spelled as GROMACS labels them: "
        "'Potential', 'Temperature', 'Total-Energy', 'Kinetic-En.' (not 'Total Energy')",
        min_length=1,
    )


@tool(
    "gmx_energy",
    description="Extract named energy terms from an .edr file into typed statistics "
    "(mean, std, drift per ns) computed by the harness from the data.",
    tags={"gromacs", "analysis"},
    requires={"gromacs"},
    contract=Contract(consumes={"edr": "energy"}, produces=(Produces("data"),)),
)
def gmx_energy(ctx: ToolContext, params: EnergyParams) -> Observation:
    stage = ctx.workspace.stage("analysis")
    edr = resolve_in_ws(ctx, params.edr)
    xvg, proc = extract_energy_terms(ctx, edr, params.terms, stage)
    command = list(proc.args)
    if proc.returncode != 0 or not xvg.exists():
        menu = energy_menu(proc.stderr + "\n" + proc.stdout)
        advisory = []
        if menu and "No energy terms selected" in proc.stderr + proc.stdout:
            advisory.append(Finding(
                id="ENERGY_TERM_NOT_IN_EDR",
                message=f"none of {params.terms} is a label in this .edr. GROMACS matches its "
                "own spellings (hyphens and dots), not spaced names: 'Total Energy' "
                "selects nothing, 'Total-Energy' does.",
                suggestion="retry with labels from available_terms",
            ))
        return Observation(
            ok=False,
            summary=f"gmx energy failed (exit {proc.returncode}) — "
            + ("the requested terms are not labels in this .edr; available_terms lists them"
               if advisory else "check that the requested terms exist in this .edr"),
            command=command,
            log_tail=tail(proc.stderr),
            findings=advisory,
            available_terms=menu,
        )
    data, legends = parse_xvg(xvg.read_text())
    if data.size == 0 or data.shape[1] < 2:
        return Observation(
            ok=False, summary="no data rows parsed from energy.xvg", command=command
        )
    time_ps = data[:, 0]
    terms: dict[str, EnergyTermStats] = {}
    names = legends or params.terms
    for i, name in enumerate(names, start=1):
        if i >= data.shape[1]:
            break
        unit = UNIT_BY_TERM.get(name.lower(), "kJ/mol")
        terms[name] = term_stats(time_ps, data[:, i], unit)
    span = float(time_ps[-1] - time_ps[0])
    parts = [
        f"{name}: {st.mean:.2f} ± {st.std:.2f} {st.unit}" for name, st in terms.items()
    ]
    return Observation(
        ok=True,
        summary=f"Energy terms over {span:.1f} ps — " + "; ".join(parts),
        command=command,
        artifacts=[ArtifactRef.from_path(xvg, "data", root=ctx.workspace.root)],
        terms=terms,
        time_span_ps=span,
    )
