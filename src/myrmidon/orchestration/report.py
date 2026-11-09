"""A study report from the ledger alone: what ran, with what, under which
findings, producing which artifacts, at what cost. Nothing here comes
from a transcript or a model — every line is a ledger event or a job record."""

from __future__ import annotations

from myrmidon.core.observation import JobState
from myrmidon.core.provenance import ProvenanceEvent
from myrmidon.core.tool import REGISTRY
from myrmidon.core.workspace import Workspace
from myrmidon.engines.facts import dipole_values, engine_log, is_run_log, run_values
from myrmidon.errors import UnknownToolError
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


# --- the report: question → results → stages → methods → deviations → record -----------

#: the fixed sections of a study's report, in order; prose under any other key follows them
SECTIONS = ("question", "results", "stages", "methods", "deviations", "record")
_TITLES = {
    "question": "Question", "results": "Results", "stages": "Stages and checks",
    "methods": "System and methods", "deviations": "Deviations", "record": "Record",
}
#: other names a narrative key may give a fixed section
_ALIASES = {"ran": "stages", "what ran": "stages", "checks": "stages"}


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


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


def run_values_of(ws, label: str, rec, event_id: str) -> list:
    """What one completed run's engine log states, as report values: its facts, and the QM
    dipole beside a coupled run's log."""
    from myrmidon.core.observation import ReportValue

    out = engine_log(rec)
    if not is_run_log(out):
        return []
    rel = ws.relpath(out)
    found = [(name, value, unit, rel)
             for name, value, unit in run_values(out.read_text(errors="replace"))]
    beside, read = dipole_values(out)
    found += [(name, value, unit, ws.relpath(read)) for name, value, unit in beside]
    return [ReportValue(step=label, label=name, value=value, unit=unit, event=event_id,
                        file=file) for name, value, unit, file in found]


def mandate_report(ctx, mandate, data_file: str = "report.json"
                   ) -> tuple[dict[str, list[str]], list, dict]:
    """(section -> the harness's lines, the parsed values, the record ``data_file`` states)
    for this study's mandate, read from the trace: the deliverables as their measurements
    read them, a row per stage with how far its latest completed run got and its engine's
    headline facts, the system and method, what departed from the approved mandate, and the
    record. Every run's log facts are values in the record, not lines."""
    from myrmidon.core import mandate as mandates
    from myrmidon.core.lineage import lineage, stage_segments
    from myrmidon.core.observation import ReportValue
    from myrmidon.execution.jobs import booked_under_mandates, existing_jobstore

    ws = ctx.workspace
    events = list(ws.ledger.events())
    signed = mandates.approvals(ws)
    digests = {a.digest for a in signed}
    since = min((a.timestamp for a in signed if a.timestamp), default=None)
    held = next((a for a in reversed(signed) if a.digest == mandate.digest), None)
    s: dict[str, list[str]] = {key: [] for key in SECTIONS}

    s["question"] = [mandate.question]
    store = existing_jobstore(ws)
    runs = [rec for rec in (store.list() if store else []) if rec.meta.get("mandate") in digests]
    by_id = {ev.event_id: ev for ev in events}
    lin = lineage(ws)
    order = [stage.name for stage in mandate.method.stages]

    def submission(rec):
        return by_id.get(rec.meta.get("event_id", ""))

    def stage_of(rec) -> str:  # the first, in the method's order, its submission ran
        ran = lin.stages_of(ev) if (ev := submission(rec)) is not None else set()
        return next((name for name in order if name in ran), min(ran, default=""))

    staged = [(stage_of(rec), rec) for rec in runs]
    by_event = {rec.meta.get("event_id", ""): rec for rec in runs}
    # each stage's latest run as the ledger chains it: a fresh start and its continuations,
    # a segment stopped at its walltime included
    segments = {stage.name: [by_event[sub.event_id] for sub, _, _ in
                             stage_segments(ws, stage.name) if sub.event_id in by_event]
                for stage in mandate.method.stages}
    values = [v for _, rec in staged if rec.state is JobState.COMPLETED
              for v in run_values_of(ws, rec.label or rec.job_id, rec,
                                     rec.meta.get("event_id", ""))]

    table, found, delivered = deliverable_rows(mandate, events, digests, since)
    s["results"] = table if mandate.deliverables else ["(the mandate names no deliverables)"]
    values += found
    s["stages"], stages = stage_rows(mandate, staged, segments)
    rendered = mandates.render(mandate).split("\n")
    start = next(i for i, line in enumerate(rendered) if line.startswith("SYSTEM"))
    end = next(i for i, line in enumerate(rendered) if line.startswith("DECISIONS"))
    s["methods"] = ["```", *rendered[start:end - 1], "```"]

    deviations, raw = departures(mandate, events, staged, signed, since, held,
                                 ran={name for ev in events if ev.event_id in lin.completed
                                      for name in lin.stages_of(ev)})
    s["deviations"] = ([deviation_line(d) for d in deviations]
                       or ["none: every stage ran as approved"])
    values += [ReportValue(step=f"stage:{d['stage']}", label="never ran", value=True,
                           event="", file="") for d in deviations if d["kind"] == "never_ran"]
    booked = booked_under_mandates(ws, digests)
    s["record"] = [
        f"- revision {mandate.revision}, digest {mandate.digest[:12]}, "
        + (f"approved by {held.approver} (event {held.event_id})" if held else "NOT approved"),
        f"- core-hours booked: {_fmt(booked)} of the ceiling's "
        f"{_fmt(mandate.cost.ceiling_core_h)}",
        *(f"- {stage} wrote: " + ", ".join(f"{role} `{path}`" for role, path in files.items())
          for stage, files in key_outputs(events, staged).items()),
        f"- every call, its command, inputs and outputs by hash: "
        f"`{ws.root}/myrmidon.ledger.jsonl`",
        f"- this report's data: `{data_file}`"]
    record = {"question": mandate.question, "mandate_digest": mandate.digest,
              "revision": mandate.revision,
              "approval_event": held.event_id if held is not None else "",
              "core_h_booked": booked, "core_h_ceiling": mandate.cost.ceiling_core_h,
              "deliverables": delivered, "stages": stages,
              "refused_or_failed_calls": raw, "deviations": deviations}
    return s, values, record


def stage_rows(mandate, staged, segments: dict[str, list]) -> tuple[list[str], list[dict]]:
    """(the Stage Table, its rows as report.json states them): each mandate stage's kind,
    its target against how far its latest run got over the jobs of its ``segments``, the
    facts its engine states first of such a run, and the core-hours its jobs booked."""
    from myrmidon.engines.facts import headline_values, run_reached

    lines = ["| Stage | Kind / ensemble | Target → reached | Key check | core-h |",
             "|---|---|---|---|---|"]
    rows: list[dict] = []
    for stage in mandate.method.stages:
        jobs = [rec for name, rec in staged if name == stage.name]
        done = [rec for rec in jobs if rec.state is JobState.COMPLETED]
        kind = " ".join(x for x in (stage.kind, stage.ensemble or "",
                                    f"({stage.regime})" if stage.regime != "mm" else "") if x)
        unit = "ps" if stage.duration_ps is not None else "steps"
        target = (f"{stage.duration_ps:g} ps" if stage.duration_ps is not None
                  else f"{stage.steps} steps" if stage.steps is not None else "")
        reached, checks = (jobs[-1].state.value.lower() if jobs else "not run"), []
        if done and segments.get(stage.name):
            logs = dict.fromkeys(engine_log(rec) for rec in segments[stage.name])
            texts = [log.read_text(errors="replace") for log in logs if is_run_log(log)]
            got = run_reached(texts, unit)
            reached = f"{got:g} {unit}" if got is not None else "completed"
            checks = headline_values(texts[-1], stage.kind, stage.ensemble or "") if texts else []
        core_h = sum(rec.meta.get("core_h_estimate") or 0.0 for rec in jobs)
        said = "; ".join(f"{label} {_fmt(v)}{' ' + unit if unit else ''}"
                         for label, v, unit in checks)
        lines.append(f"| {stage.name} | {kind} | {target or '—'} → {reached} | {said or '—'} "
                     f"| {_fmt(core_h)} |")
        rows.append({"stage": stage.name, "kind": kind, "target": target, "reached": reached,
                     "checks": [{"label": label, "value": v, "unit": unit}
                                for label, v, unit in checks],
                     "jobs": len(jobs), "core_h_booked": core_h})
    return lines, rows


def _stage_ref(ev) -> str:
    """The mandate stage a call was made for: the one its refs name, else its parameter."""
    from myrmidon.core.lineage import named_stage

    return named_stage(ev) or str(ev.params.get("stage") or "")


def departures(mandate, events, staged, signed, since, held, ran: set[str]
               ) -> tuple[list[dict], int]:
    """(what departed from the approved mandate, each as a record of its kind; how many calls
    were refused or failed under it): the revision on disk not approved (``unapproved``); a
    stage the researcher's decisions lock that never ran (``never_ran``); a job that failed
    or timed out with no later completed run of its stage (``failed_job``); a refused or
    failed call no later successful call of the same tool for the same stage made good
    (``refused``, ``failed_call``); each revision approved after the first (``revision``)."""
    from myrmidon.core import mandate as mandates

    out: list[dict] = []
    if held is None:
        out.append({"kind": "unapproved", "revision": mandate.revision})
    for stage in mandate.method.stages:
        locked = [said for field, _, said in mandates.locks_on(mandate, stage.name)
                  if stage.name not in ran
                  and mandates.field_value(stage, field) is not None]
        if locked:
            out.append({"kind": "never_ran", "stage": stage.name,
                        "locked_by": list(dict.fromkeys(locked))})
    for at, (name, rec) in enumerate(staged):
        if rec.state not in (JobState.FAILED, JobState.TIMEOUT):
            continue
        key = name or rec.label
        if not any((later or other.label) == key and other.state is JobState.COMPLETED
                   for later, other in staged[at + 1:]):
            out.append({"kind": "failed_job", "stage": name, "job": rec.label or rec.job_id,
                        "state": rec.state.value, "exit_code": rec.exit_code})
    digests = {a.digest for a in signed}
    flagged = [at for at, ev in enumerate(events)
               if since is not None and ev.timestamp >= since
               and (ev.kind == "gate_refused"
                    or (ev.kind == "call" and not ev.ok and ev.mandate in digests))]
    for at in flagged:
        ev = events[at]
        if any(later.ok and later.kind == "call" and later.tool == ev.tool
               and _stage_ref(later) == _stage_ref(ev) for later in events[at + 1:]):
            continue
        refused = ev.kind == "gate_refused"
        out.append({"kind": "refused" if refused else "failed_call", "tool": ev.tool,
                    "stage": _stage_ref(ev), "event": ev.event_id,
                    # a refusal's stopping finding is its first
                    "finding": ev.knowledge_refs[0] if refused and ev.knowledge_refs else ""})
    out += [{"kind": "revision", "approver": a.approver, "event": a.event_id}
            for a in signed[1:]]
    return out, len(flagged)


def deviation_line(d: dict) -> str:
    """A departure as the report states it: in words, with no event id."""
    stage = f" for stage {d['stage']}" if d.get("stage") else ""
    return "- " + {
        "unapproved": lambda: f"mandate revision {d['revision']} on disk is not approved: "
                              "nothing has run under it",
        "never_ran": lambda: f"stage {d['stage']}, which the researcher's decisions lock ("
                             + "; ".join(f'"{w}"' for w in d["locked_by"]) + "), never ran",
        "failed_job": lambda: (f"stage {d['stage']}" if d["stage"] else d["job"])
                              + f": its job {d['state'].lower()} (exit code {d['exit_code']}) "
                              "and no later run of it completed",
        "refused": lambda: f"{d['tool']}{stage} refused"
                           + (f": {d['finding']}" if d["finding"] else "")
                           + ", and was not made good",
        "failed_call": lambda: f"{d['tool']}{stage} failed, and was not made good",
        "revision": lambda: f"revision approved by {d['approver']}",
    }[d["kind"]]()


def key_outputs(events, staged) -> dict[str, dict[str, str]]:
    """Stage -> one file of each role its latest completed run's collection recorded."""
    out: dict[str, dict[str, str]] = {}
    for name, rec in staged:
        if rec.state is JobState.COMPLETED:
            files: dict[str, str] = {}
            for art in collection_outputs(events, rec.job_id):
                files.setdefault(art.role, art.path)
            out[name or rec.label or rec.job_id] = files
    return out


def deliverable_measurement(d, events, digests, since):
    """The latest ledger event that measured deliverable ``d`` under an approved revision of
    the mandate, as it states it; for a deliverable stated as a call of its own, the latest
    successful call of its tool since the first approval. None when nothing measured it."""
    found = [ev for ev in events if ev.ok and ev.kind == "call"
             and "DELIVERABLE_STAGE_MISMATCH" not in ev.knowledge_refs
             and (ev.measured.get("deliverable") == d.name and ev.mandate in digests
                  if d.property else ev.tool == d.tool
                  and since is not None and ev.timestamp >= since)]
    return found[-1] if found else None


def parameter_source(mandate, d) -> str:
    """The parameter set the stage a deliverable is measured over runs on, as the mandate
    names it; "" for the system's."""
    stage = mandate.method.stage(d.over)
    return stage.parameters_from if stage is not None else ""


def deliverable_record(d, ev, parameters_from: str = "") -> dict:
    """A deliverable as report.json states it: its property, its stage and the parameter set
    that stage runs on, and what measured it."""
    return {"property": d.property, "stage": d.over, "parameters_from": parameters_from,
            "measured": ev is not None,
            "event": ev.event_id if ev is not None else "",
            "file": ev.measured.get("file", "") if ev is not None else "",
            "quantities": ev.measured.get("quantities", []) if ev is not None else []}


def deliverable_rows(mandate, events, digests, since) -> tuple[list[str], list, dict]:
    """(the Deliverables Table, its values, each deliverable as report.json states it): each
    deliverable's quantities with the statistics the measurement that read them stated, from
    its ledger event; NOT measured where no event under an approved revision did. A
    deliverable stated as a call of its own (a dataset, a fit) states no single number: its
    row says the call was made, and its tables follow. Nothing is recomputed here."""
    from myrmidon.core.observation import ReportValue

    lines = ["| Deliverable | Stage | Result | Unit |", "|---|---|---|---|"]
    values: list[ReportValue] = []
    record: dict = {}
    for d in mandate.deliverables:
        ev = deliverable_measurement(d, events, digests, since)
        source = parameter_source(mandate, d)
        record[d.name] = deliverable_record(d, ev, source)
        over = (d.over or "—") + (f", parameters from {source}" if source else "")
        quantities = ev.measured.get("quantities", []) if ev is not None else []
        if ev is None or not quantities:
            lines.append(f"| {d.name} | {over} | "
                         f"{'NOT measured' if ev is None else 'made by its own call'} | |")
            continue
        for q in quantities:
            name = d.name if len(quantities) == 1 else f"{d.name}: {q['name']}"
            said = ", ".join(f"{k} {_fmt(v)}" for k, v in q["statistics"].items())
            lines.append(f"| {name} | {over} | {said} | {q['unit']} |")
            values += [ReportValue(step=f"deliverable:{d.name}", label=k, value=v,
                                   unit=q["unit"], quantity=q["name"], event=ev.event_id,
                                   file=ev.measured.get("file", ""))
                       for k, v in q["statistics"].items()]
    return lines, values, record


def narrative_sections(narrative: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """(the prose of each fixed section, the model's own sections by title): a key read as
    a section's name or its heading, whatever its case and spacing (``Results``, ``What
    ran``); any other key a section of its own, titled by its words; no prose dropped."""
    def norm(text: str) -> str:
        return " ".join(text.replace("_", " ").split()).lower()

    known = ({norm(k): k for k in SECTIONS} | {norm(t): k for k, t in _TITLES.items()}
             | {norm(k): v for k, v in _ALIASES.items()})
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
