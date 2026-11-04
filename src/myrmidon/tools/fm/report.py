"""fm_report: the force-matching campaign report from the ledger alone.

Nothing in the report comes from a transcript or a model. The ledger says which
tools ran with which parameters and which files they produced; every number is
re-parsed from those files (fm.log, optimization_results.txt, the fitted and
original .itp files, fmcheck.log, the comparisons' own outputs). A round is one fit
stage, linked to its dataset, plan and candidate topology by the artifact paths the
ledger recorded, never by order or name; the comparison against the reference covers
every topology compared, stock and fitted."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.orchestration.fm_report import build_fm_report, fmt_num, render_fm_report


class FMReportParams(BaseModel):
    stage: str = Field(default="report", description="stage to write the report into")
    prefix: str = Field(default="fm_report", description="<prefix>.md and <prefix>.json")
    max_rows: int = Field(default=40, ge=1, description="table rows per section in the .md")


@tool(
    "fm_report",
    description="The force-matching campaign report from the ledger: per fit round the dataset, "
                "settings, outcome, charge and bonded-term changes, candidate topology, validation"
                " MAE vs the reference, compute and refusals. Writes <stage>/<prefix>.md and "
                ".json.",
    tags={"mimic", "fm", "report"},
)
def fm_report(ctx: ToolContext, params: FMReportParams) -> Observation:
    ws = ctx.workspace
    data = build_fm_report(ws, compute=ctx.config.compute, backend=ctx.backend)
    stage = ws.stage(params.stage)
    md = stage / f"{params.prefix}.md"
    js = stage / f"{params.prefix}.json"
    md.write_text(render_fm_report(data, max_rows=params.max_rows))
    js.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")
    digest = []
    for r in data["rounds"]:
        mae = {}
        for metric, m in r["validation"].get("metrics", {}).items():
            mae[metric] = m["mae"]
        digest.append({"stage": r["stage"], "clean": r["clean"],
                       "n_snapshots": r["fit"].get("n_snapshots"),
                       "validation_sdf": r["fit"].get("validation_sdf"),
                       "n_unphysical": sum(len(p.get("unphysical", []))
                                           for p in r["parameters"]),
                       "mae": mae})
    bits = []
    for d in digest:
        s = f"{d['stage']}: {d['n_snapshots']} snapshots, SDF {fmt_num(d['validation_sdf'])}"
        if d["n_unphysical"]:
            s += f", {d['n_unphysical']} UNPHYSICAL terms"
        for metric, mae in d["mae"].items():
            s += f", {metric} MAE " + ", ".join(
                f"{n}={x:.3f}" if x is not None else f"{n}=?" for n, x in mae.items())
        bits.append(s)
    return Observation(
        ok=True,
        summary=f"FM report: {len(digest)} round(s) — " + ("; ".join(bits) or "nothing fitted")
        + f". Written to {ws.relpath(md)} and {ws.relpath(js)}.",
        artifacts=[ArtifactRef.from_path(md, "report", root=ws.root),
                   ArtifactRef.from_path(js, "report", root=ws.root)],
        n_rounds=len(digest),
        report_path=ws.relpath(md),
        json_path=ws.relpath(js),
        rounds=digest,
    )
