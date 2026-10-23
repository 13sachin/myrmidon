"""A study report from the ledger alone: what ran, with what, under which
acknowledgments, producing which artifacts, at what cost. Nothing here comes
from a transcript or a model — every line is a ledger event or a job record."""

from __future__ import annotations

from myrmidon.core.provenance import ProvenanceEvent
from myrmidon.core.replay import replays_nothing
from myrmidon.core.summary import summarize
from myrmidon.core.workspace import Workspace

_KEY_PARAMS = {
    "gmx_pdb2gmx": ("forcefield", "water", "ignore_hydrogens"),
    "gmx_editconf": ("box_type", "distance_nm"),
    "gmx_solvate": ("solvent",),
    "gmx_genion": ("neutralize", "concentration_mol_l", "positive_ion", "negative_ion"),
    "gmx_make_ndx": ("structure", "commands"),
    "gmx_genrestr": ("structure", "group", "force_constants"),
    "gmx_grompp": ("mdp", "structure", "topology", "max_warn"),
    "gmx_mdrun": ("tpr", "deffnm", "ntasks", "walltime_min"),
    "checkpoint_continue": ("checkpoint", "extend_ps", "walltime_min"),
    "gmx_energy": ("terms",),
    "series_convergence": ("data", "series", "max_drift_per_ns", "max_halves_sigma"),
    "traj_analyze": ("analysis", "selection"),
    "script_run": ("script", "args", "walltime_min", "ntasks"),
    "mimic_select_qm": ("coords", "selections", "padding_nm", "template"),
    "mimic_prepare": ("total_charge", "multiplicity", "maxsteps", "fm_pace"),
    "mimic_run": ("cpmd_input", "cpmd_ranks", "gromacs_ranks", "walltime_min"),
    "traj_frames": ("trajectory", "frames", "n_frames"),
    "fm_merge": ("fm_jsons",),
    "fm_collect": ("fmdata", "trajectory", "selection"),
    "fm_plan": ("stride", "qm_total_charge"),
    "fm_fit": ("fmdata", "fm_input", "mode", "n_processes"),
    "fm_validate": ("metrics", "topologies", "monitor_pairs", "n_min_frames"),
    "file_write": ("path",),
    "fetch_pdb": ("pdb_id",),
}


def _params_line(ev: ProvenanceEvent) -> str:
    keys = _KEY_PARAMS.get(ev.tool)
    items = (
        [(k, ev.params[k]) for k in keys if k in ev.params and ev.params[k] not in (None, [], "")]
        if keys else [(k, v) for k, v in ev.params.items()
                      if k != "acknowledge_warnings" and v not in (None, [], "")][:4]
    )
    return ", ".join(f"{k}={v}" for k, v in items)


def render_report(ws: Workspace, compute=None, backend=None) -> str:
    events = list(ws.ledger.events())
    meta = ws.meta
    calls = [e for e in events if e.kind == "call" and not replays_nothing(e.tool)]
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
    lines += ["", "Reproduce: `myrmidon replay " + str(ws.root) + " --into <new-dir>`"]
    return "\n".join(lines)


# --- the plan's report: question → methods → what ran → checks → results → deviations ----

#: the sections of a plan's report, in order; the keys a narrative may use
SECTIONS = ("question", "methods", "ran", "checks", "results", "deviations", "reproduce")
_TITLES = {
    "question": "Question", "methods": "System and methods", "ran": "What ran",
    "checks": "Checks", "results": "Results", "deviations": "Deviations from the plan",
    "reproduce": "How to reproduce",
}


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


#: what a coupled run's log states, by the key a plan references it with
#: (``${annealing.final_temperature_k}``), and how a report labels it
RUN_FACTS = {
    "ion_temperature_control": ("ionic temperature control", ""),
    "md_steps_completed": ("MD steps completed", ""),
    "md_time_ps": ("simulated time", "ps"),
    "final_temperature_k": ("final ionic temperature", "K"),
    "tail_mean_temperature_k": ("mean ionic temperature over the last 10 % of steps", "K"),
    "mean_temperature_k": ("mean ionic temperature", "K"),
    "sd_temperature_k": ("its standard deviation", "K"),
    "eclassic_drift_ha_per_ps": ("conserved-energy drift", "Ha/ps"),
    "eclassic_fluctuation_ha": ("fluctuation about the fit", "Ha"),
    "finished_normally": ("finished normally", ""),
    "diagnosis_id": ("failure signature", ""),
}
#: the tools whose run the harness parses into RUN_FACTS
FACT_TOOLS = ("mimic_run",)


def run_facts(text: str) -> dict:
    """What a coupled run's CPMD log states, by RUN_FACTS key: how the ionic temperature
    was controlled, how far the dynamics got, the temperature it ran at, and — only for a
    run whose header says the ions were not thermostatted — the drift of the conserved
    energy. Parsed by ``engines/mimic/diagnostics``, never typed."""
    from myrmidon.engines.mimic.diagnostics import parse_cpmd_output, parse_cpmd_series

    parsed = parse_cpmd_output(text)
    series = parse_cpmd_series(text)
    facts: dict = {}
    if series:
        facts["ion_temperature_control"] = (parsed.get("ion_temperature_control")
                                            or "none (energy conserved)")
    for key in ("md_steps_completed", "md_time_ps", "eclassic_drift_ha_per_ps",
                "eclassic_fluctuation_ha"):
        if parsed.get(key) is not None:
            facts[key] = parsed[key]
    if series:
        temps = [step["temp_k"] for step in series]
        facts["final_temperature_k"] = temps[-1]
        tail = temps[-max(1, -(-len(temps) // 10)):]  # the last 10 %, at least one step
        facts["tail_mean_temperature_k"] = sum(tail) / len(tail)
        if len(temps) > 1:
            mean = sum(temps) / len(temps)
            facts["mean_temperature_k"] = mean
            facts["sd_temperature_k"] = (sum((t - mean) ** 2 for t in temps)
                                         / (len(temps) - 1)) ** 0.5
    facts["finished_normally"] = bool(parsed.get("finished_normally"))
    if parsed.get("diagnosis_id"):
        facts["diagnosis_id"] = parsed["diagnosis_id"]
    return facts


def run_values(text: str) -> list[tuple[str, object, str]]:
    """The run's facts as a report states them: (label, value, unit), in RUN_FACTS order."""
    facts = run_facts(text)
    return [(label, facts[key], unit) for key, (label, unit) in RUN_FACTS.items()
            if key in facts]


def run_output(ctx, event_id: str):
    """(the output file, the job record) of the run a call submitted; (None, None) when it
    submitted none or the study does not hold it any more."""
    from pathlib import Path

    from myrmidon.execution.jobstore import JobStore
    from myrmidon.plan_tools import _jobs_by_event

    job_id = _jobs_by_event(ctx).get(event_id)
    db = ctx.workspace.state_dir / "jobs.sqlite"
    if job_id is None or not db.exists():
        return None, None
    rec = JobStore(db).get(job_id)
    return Path(rec.spec.cwd) / (rec.spec.stdout or ""), rec


def plan_report(ctx, plan) -> tuple[dict[str, list[str]], list]:
    """(section -> the harness's lines, the parsed values) for this study's plan.

    Everything here is read, not written by a model: the plan, the ledger events made under
    it, the job store, and the sidecars of the input sets it applied. Methods
    come from what was applied, not from what the plan said. A step reused from an earlier
    run is named with that run. Values are re-parsed from each run's own output; the
    last completed run's are the results, the earlier runs' are the checks that it rests on.
    Nothing judges a check passed or failed: no criterion is in the plan, so the number is
    stated and the reader judges.
    """
    import json
    from pathlib import Path

    from myrmidon.core.approvals import approvals
    from myrmidon.core.artifacts import sha256_file
    from myrmidon.core.observation import ReportValue
    from myrmidon.core.plan_binding import step_fit
    from myrmidon.engines.mimic.diagnostics import is_cpmd_output
    from myrmidon.execution.accounting import describe_booking
    from myrmidon.execution.jobstore import JobStore
    from myrmidon.plan_tools import (
        PLAN_FILE,
        _booked_for,
        _call_events,
        _jobs_by_event,
        _match_steps,
        describe_reuse,
    )

    ws = ctx.workspace
    compute = ctx.config.compute
    events = _call_events(ctx, plan)
    states, _, _ = _match_steps(plan, events, step_fit(ws, plan))
    by_id = {ev.event_id: ev for ev in ws.ledger.events()}
    jobs = _jobs_by_event(ctx)
    db = ws.state_dir / "jobs.sqlite"
    store = JobStore(db) if db.exists() else None
    current = sha256_file(ws.root / PLAN_FILE) if (ws.root / PLAN_FILE).is_file() else None
    approval = next((a for a in reversed(approvals(ws)) if a.plan_sha256 == current), None)
    approved_at = by_id[approval.event_id].timestamp if approval else None

    s: dict[str, list[str]] = {key: [] for key in SECTIONS}
    values: list[ReportValue] = []

    s["question"] = [plan.question, "",
                     f"plan {plan.plan_id or '(no id)'}, digest {plan.digest()[:12]}, "
                     + (f"approved {approved_at.date().isoformat()} (event "
                        f"{approval.event_id})" if approval else "NOT approved")]

    s["methods"] = [f"system: {plan.system}"] if plan.system else []
    for ev in events:
        if ev.tool != "inputset_apply":
            continue
        for art in ev.outputs:
            sidecar = ws.root / (art.path + ".myrmidon.json")
            if not sidecar.is_file():
                continue
            data = json.loads(sidecar.read_text())
            chosen = {k: data["values"].get(k) for k, o in data.get("origin", {}).items()
                      if o == "user"}
            s["methods"].append(
                f"- {art.path}: {data['inputset']} v{data.get('protocol_version')}"
                + ("; set for this study: " + ", ".join(f"{k} = {v}" for k, v in chosen.items())
                   if chosen else ", as published"))
    for state in states:
        ev = by_id.get(state["events"][0]) if state["events"] else None
        if ev is not None and ev.tool != "inputset_apply" and state["state"] == "done":
            s["methods"].append(f"- {state['name']}: {ev.tool} {_params_line(ev)}")

    finished = []  # (step, event, job record) for runs with a job, in plan order
    for step, state in zip(plan.steps, states, strict=True):
        if state["state"] == "reused":
            s["ran"].append(f"- {step.name}: REUSED, not run — {describe_reuse(ctx, step.reuse)}")
        elif state["state"] == "pending":
            s["ran"].append(f"- {step.name}: not run")
            continue
        ev = by_id.get(state["events"][0])
        if ev is None:
            continue
        rec = store.get(jobs[ev.event_id]) if store and ev.event_id in jobs else None
        if state["state"] == "done":
            line = (f"- {step.name}: {ev.tool}, {ev.timestamp.date().isoformat()}, "
                    f"event {ev.event_id}")
            if rec is not None:
                used = ctx.backend.usage(rec.handle, rec.spec) \
                    if ctx.backend is not None and rec.state.terminal else None
                line += (f"; job {rec.job_id} ({rec.state.value}), "
                         f"{describe_booking(rec.spec.resources, compute)}, "
                         f"{_fmt(rec.meta.get('core_h_estimate') or 0.0)} core-h booked"
                         + (f", {_fmt(used)} used" if used is not None else ""))
            else:
                line += ", in the session"
            s["ran"].append(line)
        if rec is not None:
            finished.append((step, ev, rec))
    s["ran"].append(f"planned {_fmt(plan.total_core_h)} core-h; booked for this plan "
                    f"{_fmt(_booked_for(ctx, plan.plan_id) if plan.plan_id else 0.0)} core-h")

    # the loop's own record: a step's further segments (an extension, a continuation) and
    # a sweep's arms are runs the plan alignment does not claim, listed by their key
    from myrmidon.core.spool import LoopState

    loop = LoopState.load(ws.state_dir) if (ws.state_dir / "loop.json").is_file() else None
    extra: list[tuple[str, object, object]] = []  # (key, event, record)
    claimed = {ev.event_id for st in states for ev_id in st["events"]
               if (ev := by_id.get(ev_id)) is not None}
    for key, event_ids in (loop.runs.items() if loop else []):
        for n, event_id in enumerate(event_ids):
            ev = by_id.get(event_id)
            if ev is None or not ev.ok or event_id in claimed or event_id not in jobs:
                continue
            rec = store.get(jobs[event_id]) if store else None
            if rec is None:
                continue
            label = key if "[" in key else f"{key}, segment {n + 1}"
            s["ran"].append(f"- {label}: {ev.tool}, {ev.timestamp.date().isoformat()}, event "
                            f"{ev.event_id}; job {rec.job_id} ({rec.state.value}), "
                            f"{describe_booking(rec.spec.resources, compute)}, "
                            f"{_fmt(rec.meta.get('core_h_estimate') or 0.0)} core-h booked")
            if rec.state.terminal:
                extra.append((label, ev, rec))

    def parsed(section: str, label: str, ev, rec) -> None:
        out = Path(rec.spec.cwd) / (rec.spec.stdout or "")
        rel = ws.relpath(out) if out.is_file() else str(out)
        if not out.is_file() or not is_cpmd_output(out):
            s[section].append(f"- {label}: no parser for its output ({rel}) in this report")
            return
        s[section].append(f"- {label}, from {rel} (event {ev.event_id}):")
        for name, value, unit in run_values(out.read_text(errors="replace")):
            s[section].append(f"  - {name}: {_fmt(value)}{' ' + unit if unit else ''}")
            values.append(ReportValue(step=label, label=name, value=value, unit=unit,
                                      event=ev.event_id, file=rel))

    for index, (step, ev, rec) in enumerate(finished):
        section = "results" if index == len(finished) - 1 else "checks"
        parsed(section, step.name, ev, rec)
    # a sweep's arms side by side are the results; a step's further segments are checks
    # on the way, and the last segment of the last step is a result
    last = finished[-1][0].name if finished else None
    for label, ev, rec in extra:
        arm = "[" in label
        section = "results" if arm or label.startswith(f"{last}, segment") else "checks"
        parsed(section, label, ev, rec)

    for ev in ws.ledger.events():
        if ev.plan_id != plan.plan_id or (approved_at and ev.timestamp < approved_at):
            continue
        when = ev.timestamp.date().isoformat()
        if ev.kind == "gate_refused":
            s["deviations"].append(f"- {when} {ev.tool} refused: {', '.join(ev.knowledge_refs)}")
        elif not ev.ok and ev.kind == "call":
            s["deviations"].append(f"- {when} {ev.tool} failed (event {ev.event_id})")
        elif ev.warnings_acknowledged and ev.tool != "plan_approve":
            who = "a person (approval " + ev.approval + ")" if ev.approval else "the agent"
            s["deviations"].append(f"- {when} {ev.tool}: {', '.join(ev.warnings_acknowledged)} "
                                   f"accepted by {who}")
    if not s["deviations"]:
        s["deviations"] = ["none: every call under the plan ran as planned"]

    s["reproduce"] = [f"`myrmidon replay {ws.root} --into <new-dir>` reproduces the study "
                      f"from its ledger; this plan's calls carry plan id {plan.plan_id}."]
    return s, values


def render_plan_report(sections: dict[str, list[str]], narrative: dict[str, str]) -> str:
    lines = ["# Study report", ""]
    for key in SECTIONS:
        lines += [f"## {_TITLES[key]}", "", *(sections[key] or ["none"]), ""]
        if narrative.get(key):
            lines += [narrative[key].strip(), ""]
    return "\n".join(lines)


def ungrounded(sections: dict[str, list[str]], narrative: dict[str, str]) -> dict[str, list[str]]:
    """section -> the numbers its prose states that the section itself does not: the
    model writes around the injected values, never a value of its own."""
    from myrmidon.agent.grounding import NumericIndex, numbers_in

    out = {}
    for key, prose in narrative.items():
        index = NumericIndex()
        for token in numbers_in("\n".join(sections.get(key, []))):
            try:
                index.add_value(float(token), "observation")
            except ValueError:
                continue
        missing = index.check(prose).ungrounded
        if missing:
            out[key] = missing
    return out
