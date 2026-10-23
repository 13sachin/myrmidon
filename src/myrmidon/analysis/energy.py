"""Energy analysis: gmx energy terms parsed into typed statistics.

Means, standard deviations, and drift are computed here by harness code from
the .xvg data — the model never invents them.
"""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import EnergyObservation, EnergyTermStats
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning
from myrmidon.engines.gromacs.common import run_gmx, tail
from myrmidon.engines.inputs import resolve_in_ws

_UNIT_BY_TERM = {
    "temperature": "K",
    "pressure": "bar",
    "density": "kg/m^3",
    "volume": "nm^3",
}


def parse_xvg(text: str) -> tuple[np.ndarray, list[str]]:
    """Return (data, legends) from xvg text; data col 0 is time."""
    legends: list[str] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            m = re.match(r'@\s+s\d+\s+legend\s+"(.+)"', line)
            if m:
                legends.append(m.group(1))
            continue
        rows.append([float(x) for x in line.split()])
    return np.array(rows), legends


def term_stats(time_ps: np.ndarray, values: np.ndarray, unit: str) -> EnergyTermStats:
    drift = None
    if len(values) > 2 and time_ps[-1] > time_ps[0]:
        slope_per_ps = float(np.polyfit(time_ps, values, 1)[0])
        drift = slope_per_ps * 1000.0  # per ns
    return EnergyTermStats(
        unit=unit,
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        drift_per_ns=drift,
        n_points=len(values),
    )


#: one label of gmx energy's selection menu: "  12  Total-Energy   "
_MENU_ENTRY = re.compile(r"(\d+)\s+([^\s\d][^\s]*)")
_MENU_LINE = re.compile(r"^\s*(?:\d+\s+[^\s\d][^\s]*\s*)+$")


def energy_menu(text: str) -> list[str]:
    """The term labels gmx energy printed, in menu order."""
    labels: list[str] = []
    for line in text.splitlines():
        if _MENU_LINE.match(line):
            labels += [label for _n, label in _MENU_ENTRY.findall(line)]
    return labels


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
    returns=EnergyObservation,
    tags={"gromacs", "analysis"},
    requires={"gromacs"},
)
def gmx_energy(ctx: ToolContext, params: EnergyParams) -> EnergyObservation:
    stage = ctx.workspace.stage("analysis")
    edr = resolve_in_ws(ctx, params.edr)
    xvg = stage / "energy.xvg"
    stdin = "\n".join(params.terms) + "\n\n"
    proc = run_gmx(
        ctx, ["energy", "-f", str(edr), "-o", str(xvg)], cwd=stage, stdin=stdin
    )
    command = list(proc.args)
    if proc.returncode != 0 or not xvg.exists():
        menu = energy_menu(proc.stderr + "\n" + proc.stdout)
        advisory = []
        if menu and "No energy terms selected" in proc.stderr + proc.stdout:
            advisory.append(ValidationWarning(
                id="ENERGY_TERM_NOT_IN_EDR",
                severity="info",
                message=f"none of {params.terms} is a label in this .edr. GROMACS matches its "
                "own spellings (hyphens and dots), not spaced names: 'Total Energy' "
                "selects nothing, 'Total-Energy' does.",
                suggestion="retry with labels from available_terms",
            ))
        return EnergyObservation(
            ok=False,
            summary=f"gmx energy failed (exit {proc.returncode}) — "
            + ("the requested terms are not labels in this .edr; available_terms lists them"
               if advisory else "check that the requested terms exist in this .edr"),
            command=command,
            log_tail=tail(proc.stderr),
            advisory=advisory,
            available_terms=menu,
        )
    data, legends = parse_xvg(xvg.read_text())
    if data.size == 0 or data.shape[1] < 2:
        return EnergyObservation(
            ok=False, summary="no data rows parsed from energy.xvg", command=command
        )
    time_ps = data[:, 0]
    terms: dict[str, EnergyTermStats] = {}
    names = legends or params.terms
    for i, name in enumerate(names, start=1):
        if i >= data.shape[1]:
            break
        unit = _UNIT_BY_TERM.get(name.lower(), "kJ/mol")
        terms[name] = term_stats(time_ps, data[:, i], unit)
    span = float(time_ps[-1] - time_ps[0])
    parts = [
        f"{name}: {st.mean:.2f} ± {st.std:.2f} {st.unit}" for name, st in terms.items()
    ]
    return EnergyObservation(
        ok=True,
        summary=f"Energy terms over {span:.1f} ps — " + "; ".join(parts),
        command=command,
        artifacts=[ArtifactRef.from_path(xvg, "data", root=ctx.workspace.root)],
        terms=terms,
        time_span_ps=span,
    )
