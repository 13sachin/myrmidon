"""The journal and the study's report: why decisions were made, and the product of a study."""

from __future__ import annotations

from pydantic import BaseModel, Field

from myrmidon.core.observation import (
    Observation,
)
from myrmidon.core.tool import ToolContext, precheck, tool
from myrmidon.core.validation import Precondition

# --- the journal: why, left where a wake and the report read it ----------------------


class JournalWriteParams(BaseModel):
    kind: str = Field(
        description="decision (with its reason), ruled_out (with the evidence), suspicion, "
        "or intention")
    text: str = Field(description="one to three sentences; numbers stay in the ledger, "
                      "the journal holds why")
    cites: list[str] = Field(default_factory=list,
                             description="the ledger event ids (ev-…) the entry rests on")
    subject: str = Field(default="", description="the stage it is about, if one")
    closes: str | None = Field(default=None, description="an earlier entry (jn-…) this one "
                               "settles, which then stops being shown to a wake")


@tool(
    "journal_write",
    description="Leave a decision, a ruled-out cause, a suspicion or an intention in the "
    "study's journal, citing the events it rests on. Open entries are what the loop shows a "
    "wake-up, so nothing is re-litigated; a closing entry names the one it settles.",
    tags={"journal"},
)
def journal_write(ctx: ToolContext, params: JournalWriteParams) -> Observation:
    from myrmidon.orchestration.spool import Journal

    kinds = ("decision", "ruled_out", "suspicion", "intention")
    if params.kind not in kinds:
        return Observation(ok=False, summary=f"kind is one of {', '.join(kinds)}")
    journal = Journal(ctx.workspace.root)
    # an entry through the server or the loop is the model's; one from the shell a person's
    by = "model" if ctx.model_driven or ctx.unattended else "human"
    if params.closes:
        entry = journal.close(params.closes, by, params.text)
    else:
        entry = journal.write(params.kind, by, params.text, cites=params.cites,
                              subject=params.subject)
    return Observation(ok=True, summary=f"{entry.kind} {entry.entry_id} written",
                              entry_id=entry.entry_id, entry_kind=entry.kind, record=False)


# --- the report: the product of a study -----------------------------------------------


class StudyReportParams(BaseModel):
    narrative: dict[str, str] = Field(
        default_factory=dict,
        description="your prose for each section, keyed question, methods, ran, checks, "
        "results, deviations, record — placed after the values the harness writes "
        "there (prose under another name is a section of its own after them). Say what "
        "the values mean; a value of your own that no section states is not in the study.",
    )
    output: str = Field(default="report.md",
                        description="a file name in the study's directory")
    max_rows: int = Field(default=40, ge=1, description="table rows per section of a fit's tables")


def study_report_gate(ctx: ToolContext, params: StudyReportParams):
    from myrmidon.core.mandate import read_mandate
    from myrmidon.orchestration.fit_report import find_rounds

    checked = ["mandate_exists"]
    fitted = find_rounds(list(ctx.workspace.ledger.events()))
    if read_mandate(ctx.workspace) is None and not fitted:
        return [Precondition(
            id="REPORT_NO_MANDATE",
            message="this study has no mandate, so there is nothing to report against",
            suggestion="a report states what a mandate set out to do and what it did; "
            "write it with mandate_write")], checked
    checked.append("inside_the_study")
    from pathlib import PurePosixPath

    if PurePosixPath(params.output).name != params.output or not params.output.endswith(".md"):
        return [Precondition(
            id="REPORT_OUTPUT_NOT_IN_STUDY",
            message=f"{params.output!r}: the report is a .md file in the study's own directory",
            suggestion="give a file name such as report.md")], checked
    return [], checked


@tool(
    "study_report",
    description="Write the study's report as a file from its trace, in sections: question, "
                "methods, ran, checks, results (each deliverable), deviations, record, a fit's "
                "tables. The harness writes every value with its source; you add prose per "
                "section.",
    tags={"report"},
)
def study_report(ctx: ToolContext, params: StudyReportParams) -> Observation:
    precheck(ctx, study_report_gate(ctx, params)[0])
    import json

    from myrmidon.core.artifacts import ArtifactRef
    from myrmidon.core.mandate import read_mandate
    from myrmidon.orchestration.report import mandate_report, narrative_sections, render_sections

    mandate = read_mandate(ctx.workspace)
    sections, values, record = (mandate_report(ctx, mandate) if mandate is not None
                                else ({}, [], {}))
    path = ctx.workspace.root / params.output
    text = render_sections(sections, params.narrative) if mandate is not None else ""
    fit = _fit_tables(ctx, params)
    path.write_text(text + fit["text"])
    sidecar = ctx.workspace.root / (params.output.removesuffix(".md") + ".json")
    sidecar.write_text(json.dumps(
        {"rounds": [], **fit["data"], **record, "values": [v.model_dump() for v in values]},
        indent=1, sort_keys=True, default=str) + "\n")
    own = list(narrative_sections(params.narrative)[1])
    return Observation(
        ok=True,
        summary=f"report written to {ctx.workspace.relpath(path)}, its data to "
        f"{ctx.workspace.relpath(sidecar)}: {len(values)} parsed value(s); read it for the "
        "values before adding narrative"
        + (f"; your section(s) {', '.join(own)} follow the report's" if own else "")
        + fit["said"],
        path=ctx.workspace.relpath(path),
        artifacts=[ArtifactRef.from_path(path, "report", ctx.workspace.root),
                   ArtifactRef.from_path(sidecar, "report", ctx.workspace.root)],
        values=values,
        rounds=fit["rounds"],
        quiet={"values"},  # every value is in the report written
    )


def _fit_tables(ctx: ToolContext, params: StudyReportParams) -> dict:
    """A study with a fit's collection reports it too: per round the dataset, settings, outcome,
    charge and bonded-term changes, candidate topology and validation, every number re-parsed
    from the files the ledger names, its data for report.json beside them."""
    from myrmidon.orchestration.fit_report import (
        build_fit_report,
        find_rounds,
        fmt_num,
        render_fit_report,
    )

    ws = ctx.workspace
    if not find_rounds(list(ws.ledger.events())):
        return {"text": "", "said": "", "data": {}, "rounds": []}
    data = build_fit_report(ws, compute=ctx.config.compute, backend=ctx.backend)
    digest, bits = [], []
    for r in data["rounds"]:
        mae = {metric: m["mae"] for metric, m in r["validation"].get("metrics", {}).items()}
        terms = sum(len(p.get("unphysical", [])) for p in r["parameters"])
        digest.append({"stage": r["stage"], "clean": r["clean"],
                       "n_snapshots": r["fit"].get("n_snapshots"),
                       "validation_sdf": r["fit"].get("validation_sdf"),
                       "n_unphysical": terms, "mae": mae})
        said = (f"{r['stage']}: {r['fit'].get('n_snapshots')} snapshots, SDF "
                f"{fmt_num(r['fit'].get('validation_sdf'))}")
        bits.append(said + (f", {terms} UNPHYSICAL terms" if terms else ""))
    return {"text": "\n\n" + render_fit_report(data, max_rows=params.max_rows),
            "said": f"; {len(digest)} fit round(s) ({'; '.join(bits)})", "rounds": digest,
            "data": data}
