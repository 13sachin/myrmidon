"""A study report from the ledger alone: what ran, with what, under which
findings, producing which artifacts, at what cost. Nothing here comes
from a transcript or a model — every line is a ledger event or a job record."""

from __future__ import annotations

from myrmidon.core.lineage import stages_run
from myrmidon.core.provenance import ProvenanceEvent
from myrmidon.core.tool import REGISTRY
from myrmidon.core.workspace import Workspace
from myrmidon.engines.facts import dipole_values, engine_log, is_run_log, run_values
from myrmidon.errors import UnknownToolError
from myrmidon.knowledge.observables import PROPERTIES, compile_deliverable, measured_property
from myrmidon.orchestration.summary import summarize

#: calls the report does not count as work besides the navigation tools (which declare
#: it: `ToolSpec.navigation`): session bookkeeping, a look at a file, interactive viewers,
#: and tool names a ledger may carry that no registered tool answers to
BOOKKEEPING_TOOLS = frozenset({
    "study_create", "study_open", "file_check", "viz_open", "viz_command",
    "lesson_record", "lesson_search",
})


def bookkeeping(tool: str) -> bool:
    """A call the study report does not count as work."""
    if tool in BOOKKEEPING_TOOLS:
        return True
    try:
        return REGISTRY.get(tool).navigation
    except UnknownToolError:
        return False


_KEY_PARAMS = {
    "gmx_pdb2gmx": ("forcefield", "water", "ignore_hydrogens"),
    "gmx_editconf": ("box_type", "distance_nm"),
    "gmx_solvate": ("solvent",),
    "gmx_genion": ("neutralize", "concentration_mol_l", "positive_ion", "negative_ion"),
    "gmx_make_ndx": ("structure", "commands"),
    "gmx_genrestr": ("structure", "group", "force_constants"),
    "stage_prepare": ("stage", "structure", "topology", "max_warn"),
    "stage_run": ("stage", "cores", "walltime_min", "continue_run", "extend_ps"),
    "observable_measure": ("deliverable", "property", "stage", "selection", "pairs", "terms",
                           "topology", "max_drift_per_ns", "max_halves_sigma"),
    "run_diagnose": ("stage", "log"),
    "dataset_collect": ("stage", "fmdata", "trajectory"),
    "snapshot_select": ("stage", "target_temperature_k", "n_frames", "frames"),
    "script_run": ("script", "args", "walltime_min", "ntasks"),
    "qm_region": ("coords", "selections", "padding_nm", "template"),
    "file_write": ("path",),
    "fetch_pdb": ("pdb_id",),
    "structure_select": ("structure", "chains", "ligand", "keep", "altloc"),
    "ligand_parametrize": ("structure", "net_charge", "hydrogens", "ph", "atom_types",
                           "charge_method"),
    "complex_assemble": ("structure", "ligand_itp", "ligand_restraints", "restraints_define"),
}


def _params_line(ev: ProvenanceEvent) -> str:
    keys = _KEY_PARAMS.get(ev.tool)
    items = (
        [(k, ev.params[k]) for k in keys if k in ev.params and ev.params[k] not in (None, [], "")]
        if keys else [(k, v) for k, v in ev.params.items()
                      if v not in (None, [], "")][:4]
    )
    return ", ".join(f"{k}={v}" for k, v in items)


def render_report(ws: Workspace, compute=None, backend=None) -> str:
    events = list(ws.ledger.events())
    meta = ws.meta
    calls = [e for e in events if e.kind == "call" and not bookkeeping(e.tool)]
    ok_calls = [e for e in calls if e.ok]
    refusals = [e for e in events if e.kind == "gate_refused"]
    versions: dict[str, str] = {}
    for e in events:
        versions.update(e.engine_versions)
    lines = [
        f"# {meta.get('name', ws.root.name)} — study report from the ledger",
        "",
        f"workspace: {ws.root}",
        f"created: {meta.get('created', '?')}",
        f"question: {meta.get('description', '')}",
        f"ledger: {len(events)} events — {len(ok_calls)} successful tool calls, "
        f"{len(calls) - len(ok_calls)} failed, {len(refusals)} gate refusals",
        "engines: " + ", ".join(f"{k} {v}" for k, v in sorted(versions.items()) if v != "unknown"),
        "",
        "## Steps (successful tool calls, in order)",
        "",
    ]
    for i, e in enumerate(ok_calls, 1):
        stamp = e.timestamp.isoformat(timespec="seconds")
        acks = (
            f"  [acknowledged: {', '.join(e.warnings_acknowledged)}]"
            if e.warnings_acknowledged else ""
        )
        outs = ", ".join(f"{a.path} ({a.sha256[:8]})" for a in e.outputs[:4])
        if len(e.outputs) > 4:
            outs += f", +{len(e.outputs) - 4}"
        lines.append(f"{i:3d}. {stamp}  {e.tool}  {_params_line(e)}{acks}")
        if outs:
            lines.append(f"       -> {outs}")
        if e.knowledge_refs:
            lines.append(f"       findings: {', '.join(e.knowledge_refs)}")
    if refusals:
        lines += ["", "## Gate refusals (what the harness would not do silently)", ""]
        for e in refusals:
            att = f"  findings: {', '.join(e.knowledge_refs)}" if e.knowledge_refs else ""
            lines.append(
                f"- {e.timestamp.isoformat(timespec='seconds')}  {e.tool}  {_params_line(e)}{att}"
            )
    s = summarize(ws, "all", backend=backend, compute=compute, poll_usage=backend is not None)
    lines += ["", "## Jobs and compute", ""]
    if s.jobs:
        for j in s.jobs:
            est = (
                f"{j.core_h_estimate:g} core-h booked" if j.core_h_estimate is not None
                else "unpriced"
            )
            consumed = j.core_h_actual
            used = f", {consumed:g} used" if consumed is not None else ""
            lines.append(f"- {j.job_id}  {j.label or ''}  {j.state}  {est}{used}")
        total_used = f", {s.core_h_consumed:g} consumed" if s.core_h_consumed is not None else ""
        lines.append(f"total: {s.core_h_submitted:g} core-h booked{total_used}")
    else:
        lines.append("no jobs")
    lines += ["", f"## Artifacts: {s.n_artifacts} — "
              + ", ".join(f"{k} {v}" for k, v in sorted(s.artifacts_by_role.items()))]
    if s.knowledge_refs:
        lines += ["", "findings surfaced during the study: " + ", ".join(s.knowledge_refs)]
    return "\n".join(lines)


# --- the report: question → methods → what ran → checks → results → deviations ---------

#: the fixed sections of a study's report, in order; prose under any other key follows them
SECTIONS = ("question", "methods", "ran", "checks", "results", "deviations", "record")
_TITLES = {
    "question": "Question", "methods": "System and methods", "ran": "What ran",
    "checks": "Checks", "results": "Results", "deviations": "Deviations",
    "record": "The record",
}


#: how a deliverable's statistic is computed from the series its measurement saved
STATISTICS = {"mean": lambda y: float(y.mean()), "max": lambda y: float(y.max()),
              "min": lambda y: float(y.min()), "final": lambda y: float(y[-1]),
              "sd": lambda y: float(y.std(ddof=1)) if len(y) > 1 else None}


def _series_statistics(ws, event, wanted: list[str]) -> tuple[dict, str]:
    """(the statistics a deliverable asks for, the series file they were computed from):
    from the .npz its measuring call saved, which the ledger records; ({}, "") when it
    saved none."""
    import numpy as np

    metric = event.params.get("metric")
    for out in event.outputs:
        if not out.path.endswith(".npz"):
            continue
        try:
            with np.load(ws.root / out.path) as data:
                key = metric if metric in data else next(
                    (k for k in data.files if k not in ("time_ps", "atom_index",
                                                        "r_angstrom", "resid")), None)
                series = np.asarray(data[key], dtype=float) if key else None
        except (OSError, ValueError):
            continue
        if series is None or series.ndim != 1 or not series.size:
            continue  # a table of pairs is no one series
        values = {k: STATISTICS[k](series) for k in (wanted or list(STATISTICS))}
        return {k: v for k, v in values.items() if v is not None}, out.path
    return {}, ""


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def analysis_tools() -> set[str]:
    """The tools whose results a report collects for the run they read: those tagged
    `analysis` in the registry."""
    from myrmidon.core.tool import REGISTRY

    return {spec.name for spec in REGISTRY.specs() if "analysis" in spec.tags}


def trace_lines(events, limit: int | None = None) -> list[str]:
    """The study's calls, from its ledger events, oldest first (the last ``limit``): the
    tool, its outcome, the files it made and the job it submitted or collected."""
    calls = [e for e in events if e.kind == "call"]
    out = []
    for event in calls[-limit:] if limit else calls:
        made = ", ".join(a.path for a in event.outputs[:3])
        out.append(f"- {event.tool} {'ok' if event.ok else 'failed'}"
                   + (f" -> {made}" if made else "")
                   + (f" ({event.params['job_id']})" if "job_id" in event.params else ""))
    return out


def collection_outputs(events, job_id: str) -> list:
    """The artifacts the call that collected ``job_id`` recorded: the run's own files, as
    distinct from the other runs' in the same directory. The collecting call is the
    successful one whose parameters name the job."""
    for ev in events:
        if ev.kind == "call" and ev.ok and ev.params.get("job_id") == job_id and ev.outputs:
            return list(ev.outputs)
    return []


def analysis_values(ws, events, run_dir: str, run_files: set[str] = frozenset()
                    ) -> tuple[list[str], list]:
    """(the report's lines, its values) for every successful analysis call that read a
    file of the run: one of ``run_files`` (what its collection recorded), or, when the
    collection recorded none, any file in ``run_dir``. Each value is re-parsed from the
    analysis's own output file (the .xvg, .npz or .json it wrote), which the ledger names
    by hash — a file changed since is said to be, not read. Energy terms as mean, sd and
    drift; a trajectory metric as its mean, final and largest value; a convergence verdict
    with the settled part's mean and drift."""
    import json

    import numpy as np

    from myrmidon.core.artifacts import sha256_file
    from myrmidon.core.observation import ReportValue

    lines: list[str] = []
    values: list[ReportValue] = []
    prefix = run_dir.rstrip("/") + "/"

    def of_the_run(path: str) -> bool:
        return path in run_files if run_files else path.startswith(prefix)

    derived: set[str] = set()  # what the analyses so far wrote: a convergence check may
    # read another analysis's series rather than the run's files, and is still this run's
    analyses = analysis_tools()
    for ev in events:
        if ev.tool not in analyses or not ev.ok or ev.kind != "call":
            continue
        if not any(of_the_run(inp.path) or inp.path in derived for inp in ev.inputs):
            continue
        derived.update(a.path for a in ev.outputs)
        out = next((a for a in ev.outputs if a.path.endswith((".xvg", ".npz", ".json"))), None)
        if out is None:
            continue
        file = ws.root / out.path
        label = f"{ev.tool} (event {ev.event_id})"
        if not file.is_file() or sha256_file(file) != out.sha256:
            lines.append(f"  - {label}: its output {out.path} is not the file the ledger "
                         "recorded (changed or gone), so its values are not stated")
            continue

        def add(name: str, value, unit: str = "", event_id=ev.event_id, path=out.path) -> None:
            lines.append(f"    - {name}: {_fmt(value)}{' ' + unit if unit else ''}")
            values.append(ReportValue(step=run_dir, label=name, value=value, unit=unit,
                                      event=event_id, file=path))

        measured = measured_property(ev.params)
        extractor = PROPERTIES[measured][0] if measured in PROPERTIES else ""
        if extractor == "energy":
            from myrmidon.analysis.energy import UNIT_BY_TERM, parse_xvg, term_stats

            data, legends = parse_xvg(file.read_text())
            if data.size == 0 or data.shape[1] < 2:
                continue
            names = legends or list(ev.params.get("terms") or [])
            lines.append(f"  - {label}, over {data[-1, 0] - data[0, 0]:g} ps from {out.path}:")
            for i, name in enumerate(names, start=1):
                if i >= data.shape[1]:
                    break
                unit = UNIT_BY_TERM.get(name.lower(), "kJ/mol")
                st = term_stats(data[:, 0], data[:, i], unit)
                add(f"{name}, mean", st.mean, unit)
                add(f"{name}, sd", st.std, unit)
                if st.drift_per_ns is not None:
                    add(f"{name}, drift", st.drift_per_ns, f"{unit}/ns")
        elif measured == "pair_distance":
            with np.load(file) as z:
                labels = [str(x) for x in z["pairs"]]
                d = z["pair_distance"]
                ref = z["reference_mean"] if "reference_mean" in z.files else None
            lines.append(f"  - {label}: pair distances over {d.shape[0]} frames, from "
                         f"{out.path}:")
            for k, name in enumerate(labels):
                add(f"{name}, mean", float(d[:, k].mean()), "Å")
                if ref is not None:
                    add(f"{name}, reference mean", float(ref[k]), "Å")
            if ref is not None:
                add("mean absolute difference from the reference",
                    float(np.mean(np.abs(d.mean(axis=0) - ref))), "Å")
        elif extractor == "forces":
            data = json.loads(file.read_text())
            units = data.get("units") or ""
            lines.append(f"  - {label}: {data.get('label')} against the reference over "
                         f"{data.get('n_frames')} frames, from {out.path}:")
            add(f"{data.get('label')}, force RMSE", data["rmse"], units)
            add(f"{data.get('label')}, reference RMS", data["ref_rms"], units)
        elif extractor == "trajectory":
            metric = str(ev.params.get("metric", ""))
            with np.load(file) as z:
                if metric not in z.files:
                    continue
                y = z[metric]
                axis = next((k for k in z.files if k != metric), "")
            unit = "g(r)" if metric == "rdf" else "Å"
            over = (f"{ev.params.get('selection', '')!r}"
                    + (f" vs {ev.params.get('selection2')!r}" if ev.params.get("selection2")
                       else ""))
            lines.append(f"  - {label}: {metric} of {over}, {len(y)} points over {axis}, "
                         f"from {out.path}:")
            if metric != "rdf":
                add(f"{metric}, mean", float(np.mean(y)), unit)
            add(f"{metric}, largest", float(np.max(y)), unit)
            if axis == "time_ps":
                add(f"{metric}, final", float(y[-1]), unit)
        elif extractor == "settling":
            for s in json.loads(file.read_text()):
                unit = s.get("unit") or ""
                lines.append(f"  - {label}: {s['name']} over {_fmt(s['span_ps'])} ps, "
                             f"from {out.path}:")
                add(f"{s['name']}, settled from", s["equilibrated_from_ps"], "ps")
                add(f"{s['name']}, settled mean", s["mean"], unit)
                add(f"{s['name']}, standard error", s["standard_error"], unit)
                if s.get("drift_per_ns") is not None:
                    add(f"{s['name']}, drift", s["drift_per_ns"], f"{unit}/ns" if unit else "/ns")
                if s.get("criterion_met") is not None:
                    add(f"{s['name']}, criterion met", bool(s["criterion_met"]))
    return lines, values


def run_lines(ws, events, label: str, rec, event_id: str) -> tuple[list[str], list]:
    """(lines, values) for one run: its engine log's values and what the analyses made of
    its files, each re-read from its own output."""
    from pathlib import Path

    from myrmidon.core.observation import ReportValue

    lines: list[str] = []
    values: list[ReportValue] = []
    out = engine_log(rec)
    rel = ws.relpath(out) if out.is_file() else str(out)
    if not is_run_log(out):
        lines.append(f"- {label}: no engine log this report parses ({rel}); what the "
                     "analyses made of its files follows")
    else:
        lines.append(f"- {label}, from {rel} (event {event_id}):")
        for name, value, unit in run_values(out.read_text(errors="replace")):
            lines.append(f"  - {name}: {_fmt(value)}{' ' + unit if unit else ''}")
            values.append(ReportValue(step=label, label=name, value=value, unit=unit,
                                      event=event_id, file=rel))
        dipole = ws.relpath(out.parent / "DIPOLE")
        for name, value, unit in dipole_values(out):
            lines.append(f"  - {name}: {_fmt(value)}{' ' + unit if unit else ''} ({dipole})")
            values.append(ReportValue(step=label, label=name, value=value, unit=unit,
                                      event=event_id, file=dipole))
    found_lines, found = analysis_values(
        ws, events, ws.relpath(Path(rec.spec.cwd)),
        {a.path for a in collection_outputs(events, rec.job_id)})
    return [*lines, *found_lines], [*values, *(v.model_copy(update={"step": label})
                                               for v in found)]


def mandate_report(ctx, mandate) -> tuple[dict[str, list[str]], list]:
    """(section -> the harness's lines, the parsed values) for this study's mandate, read
    from the trace: the mandate and its approvals, the jobs booked under it, each completed
    run's values re-parsed from its own output and from the analyses of its files, the
    deliverables against what was measured, and what the gate refused or the engines
    failed. The last completed run's values are the results; the earlier runs' are the
    checks it rests on."""
    from myrmidon.core import mandate as mandates
    from myrmidon.core.observation import ReportValue
    from myrmidon.execution.accounting import describe_booking
    from myrmidon.execution.backend import JobState
    from myrmidon.execution.jobs import booked_under_mandates, existing_jobstore

    ws = ctx.workspace
    events = list(ws.ledger.events())
    signed = mandates.approvals(ws)
    digests = {a.digest for a in signed}
    since = min((a.timestamp for a in signed if a.timestamp), default=None)
    held = next((a for a in reversed(signed) if a.digest == mandate.digest), None)
    s: dict[str, list[str]] = {key: [] for key in SECTIONS}
    values: list = []

    s["question"] = [mandate.question, "",
                     f"mandate revision {mandate.revision}, digest {mandate.digest[:12]}, "
                     + (f"approved by {held.approver} (event {held.event_id})" if held
                        else "NOT approved")]
    s["methods"] = mandates.render(mandate).split("\n")[2:-2]

    store = existing_jobstore(ws)
    runs = [rec for rec in (store.list() if store else []) if rec.meta.get("mandate") in digests]
    by_id = {ev.event_id: ev for ev in events}
    for rec in runs:
        ev = by_id.get(rec.meta.get("event_id", ""))
        tool = ev.tool if ev else rec.meta.get("kind", "")
        booking = describe_booking(rec.spec.resources, ctx.config.compute)
        s["ran"].append(f"- {rec.label or rec.job_id}: {tool}, job {rec.job_id} "
                        f"({rec.state.value}), {booking}, "
                        f"{_fmt(rec.meta.get('core_h_estimate') or 0.0)} core-h booked")
    s["ran"].append(f"booked under the mandate: "
                    f"{_fmt(booked_under_mandates(ws, digests))} of "
                    f"{_fmt(mandate.cost.ceiling_core_h)} core-h")

    completed = [rec for rec in runs if rec.state is JobState.COMPLETED]
    for index, rec in enumerate(completed):
        section = "results" if index == len(completed) - 1 else "checks"
        lines, found = run_lines(ws, events, rec.label or rec.job_id, rec,
                                 rec.meta.get("event_id", ""))
        s[section].extend(lines)
        values.extend(found)

    for d in mandate.deliverables:
        tool = compile_deliverable(mandate, d).tool
        measured = [ev for ev in events if ev.ok and ev.kind == "call" and tool
                    and ev.tool == tool and (since is None or ev.timestamp >= since)
                    # a call names the deliverable it measures where its tool takes one
                    and (not d.property or ev.params.get("deliverable") == d.name
                         or "deliverable" not in REGISTRY.get(tool).params_model.model_fields)
                    and "DELIVERABLE_STAGE_MISMATCH" not in ev.knowledge_refs]
        stats, series = (_series_statistics(ws, measured[-1], d.statistics)
                         if measured and d.property else ({}, ""))
        said = ", ".join(f"{k} {_fmt(v)}" for k, v in stats.items())
        s["results"].append(f"- deliverable {d.name}: " + (
            f"measured by {tool} (event {measured[-1].event_id})" + (f": {said}" if said
                                                                       else "")
            if measured else "NOT measured"))
        values += [ReportValue(step=f"deliverable:{d.name}", label=k, value=v,
                               event=measured[-1].event_id, file=series)
                   for k, v in stats.items()]

    if held is None:
        s["deviations"].append(f"- mandate revision {mandate.revision} on disk is not "
                               "approved: nothing has run under it")
    for rec in runs:
        if rec.state.terminal and rec.state is not JobState.COMPLETED:
            s["deviations"].append(f"- job {rec.job_id} ({rec.label or 'unlabelled'}) "
                                   f"{rec.state.value.lower()}, exit code {rec.exit_code}")
    for ev in events:
        if since is None or ev.timestamp < since:
            continue
        when = ev.timestamp.date().isoformat()
        if ev.kind == "gate_refused":
            s["deviations"].append(f"- {when} {ev.tool} refused: {', '.join(ev.knowledge_refs)}")
        elif not ev.ok and ev.kind == "call" and ev.mandate in digests:
            s["deviations"].append(f"- {when} {ev.tool} failed (event {ev.event_id})")
    ran = stages_run(ws)
    for stage in mandate.method.stages:
        locked = [said for field, _, said in mandates.locks_on(mandate, stage.name)
                  if stage.name not in ran
                  and mandates.field_value(stage, field) is not None]
        if locked:
            s["deviations"].append(
                f"- stage {stage.name}, which the researcher's decisions lock ("
                + "; ".join(f'"{w}"' for w in dict.fromkeys(locked)) + "), never ran")
            values.append(ReportValue(step=f"stage:{stage.name}", label="never ran",
                                      value=True, event="", file=""))
    for a in signed[1:]:
        s["deviations"].append(f"- revision approved by {a.approver} (event {a.event_id})")
    if not s["deviations"]:
        s["deviations"] = ["none: every call under the mandate ran as approved"]
    s["record"] = [f"`{ws.root}/myrmidon.ledger.jsonl` holds every call, its command, inputs "
                   f"and outputs by hash; the jobs under the mandate carry its digest "
                   f"{mandate.digest[:12]}."]
    return s, values


def narrative_sections(narrative: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """(the prose of each fixed section, the model's own sections by title): a key read as
    a section's name or its heading, whatever its case and spacing (``Results``, ``What
    ran``); any other key a section of its own, titled by its words; no prose dropped."""
    def norm(text: str) -> str:
        return " ".join(text.replace("_", " ").split()).lower()

    known = {norm(k): k for k in SECTIONS} | {norm(t): k for k, t in _TITLES.items()}
    fixed: dict[str, list[str]] = {}
    own: dict[str, tuple[str, list[str]]] = {}  # by the title's words, its first spelling
    for key, text in narrative.items():
        if not text.strip():
            continue
        words = " ".join(key.replace("_", " ").split()).lstrip("# ").strip()
        if norm(words) in known:
            fixed.setdefault(known[norm(words)], []).append(text.strip())
        else:
            title = (words[:1].upper() + words[1:]) or "Notes"
            own.setdefault(norm(title), (title, []))[1].append(text.strip())
    return ({k: "\n\n".join(v) for k, v in fixed.items()},
            {title: "\n\n".join(v) for title, v in own.values()})


def render_sections(sections: dict[str, list[str]], narrative: dict[str, str]) -> str:
    """The report: each section's values with the prose given for it; prose under a name
    the report does not have is a section of its own after them, never dropped."""
    fixed, own = narrative_sections(narrative)
    lines = ["# Study report", ""]
    for key in SECTIONS:
        lines += [f"## {_TITLES[key]}", "", *(sections[key] or ["none"]), ""]
        if key in fixed:
            lines += [fixed[key], ""]
    for title, text in own.items():
        lines += [f"## {title}", "", text, ""]
    return "\n".join(lines)
