"""fm_report: the force-matching campaign report from the ledger alone.

Nothing in the report comes from a transcript or a model. The ledger says which
tools ran with which parameters and which files they produced; every number is
re-parsed from those files (fm.log, optimization_results.txt, the fitted and
original .itp files, validate_results.json, fmcheck.log). A round is one fit
stage; rounds are linked to their dataset, plan, candidate topology and
validation by the artifact paths the ledger recorded, never by order or name.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation
from myrmidon.core.provenance import ProvenanceEvent
from myrmidon.core.summary import summarize
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.workspace import Workspace
from myrmidon.engines.gromacs.itp import BONDED_N_ATOMS, parse_itp_atoms, parse_itp_bonded
from myrmidon.engines.mimic.fm import (
    VALIDATION_RESIDUALS_FILENAME,
    parse_fm_log,
    parse_fmcheck_report,
    parse_optimization_results,
    parse_validation_force_residuals,
    unphysical_force_constants,
)
from myrmidon.engines.mimic.validate import _table, force_type_table

# -- .itp parsers (grounded: the fitted files are the engine's output) ----------

_PARAM_NAMES = {
    ("bonds", 1): ("b0", "kb"),
    ("angles", 1): ("th0", "kth"),
    ("angles", 5): ("th0", "kth", "ub0", "kub"),
    ("dihedrals", 9): ("phase", "kd", "pn"),
    ("dihedrals", 4): ("phase", "kd", "pn"),
    ("dihedrals", 1): ("phase", "kd", "pn"),
    ("dihedrals", 2): ("xi0", "kxi"),
}


def charge_diff(before: dict[str, list[dict]], after: dict[str, list[dict]],
                tol: float = 5e-5) -> dict:
    """Per-atom and per-atom-type charge changes between two parsed [ atoms ] tables."""
    rows: list[dict] = []
    by_type: dict[str, list[float]] = {}
    n_atoms = 0
    if len(before) == 1 and len(after) == 1 and set(before) != set(after):
        # a single molecule on each side (one lacks its [ moleculetype ]): compare them
        before = {next(iter(after)): next(iter(before.values()))}
    for mol, atoms_after in after.items():
        atoms_before = {a["nr"]: a for a in before.get(mol, [])}
        for a in atoms_after:
            b = atoms_before.get(a["nr"])
            if b is None:
                continue
            n_atoms += 1
            dq = a["charge"] - b["charge"]
            if abs(dq) >= tol:
                rows.append({
                    "molecule": mol, "nr": a["nr"], "atom": a["name"], "type": a["type"],
                    "q_before": round(b["charge"], 6), "q_after": round(a["charge"], 6),
                    "dq": round(dq, 6),
                })
                by_type.setdefault(a["type"], []).append(dq)
    types = [
        {"type": t, "n": len(v), "mean_dq": round(sum(v) / len(v), 6),
         "max_abs_dq": round(max(abs(x) for x in v), 6)}
        for t, v in sorted(by_type.items(), key=lambda kv: -max(abs(x) for x in kv[1]))
    ]
    abs_dq = [abs(r["dq"]) for r in rows]
    return {
        "n_atoms_compared": n_atoms, "n_changed": len(rows),
        "max_abs_dq": round(max(abs_dq), 6) if abs_dq else 0.0,
        "mean_abs_dq": round(sum(abs_dq) / n_atoms, 6) if n_atoms else 0.0,
        "net_dq": round(sum(r["dq"] for r in rows), 6),
        "rows": sorted(rows, key=lambda r: -abs(r["dq"])),
        "by_type": types,
    }


def bonded_diff(before: dict, after: dict, atoms: dict[str, list[dict]] | None = None,
                rel_tol: float = 1e-6, floor_fraction: float | None = None) -> dict:
    """Bonded terms whose parameters changed, with atom names when the [ atoms ] table
    is given. With ``floor_fraction`` (mimicpy's regularization_force_constant_min_fraction)
    force constants that landed exactly on the floor are counted: those terms wanted to go
    lower and the regularizer stopped them (the signature: every changed bond landing
    at exactly the floor fraction of its original force constant)."""
    names: dict[str, dict[int, str]] = {
        mol: {a["nr"]: a["name"] for a in lst} for mol, lst in (atoms or {}).items()
    }
    rows: list[dict] = []
    n_terms = 0
    counts: dict[str, int] = {}
    at_floor = 0
    for mol, sections in after.items():
        for section, terms in sections.items():
            n_atoms = BONDED_N_ATOMS[section]
            for key, p_after in terms.items():
                p_before = before.get(mol, {}).get(section, {}).get(key)
                if p_before is None or len(p_before) != len(p_after):
                    continue
                n_terms += 1
                funct = key[n_atoms]
                labels = _PARAM_NAMES.get((section, funct)) or tuple(
                    f"p{i + 1}" for i in range(len(p_after)))
                nm = names.get(mol, {})
                atom_label = "-".join(nm.get(i, str(i)) for i in key[:n_atoms])
                for i, (x, y) in enumerate(zip(p_before, p_after, strict=True)):
                    if abs(y - x) > rel_tol * max(abs(x), abs(y), 1e-12):
                        floor = (floor_fraction is not None and x
                                 and abs(y / x - floor_fraction) < 1e-3)
                        at_floor += bool(floor)
                        rows.append({
                            "molecule": mol, "section": section, "atoms": atom_label,
                            "indices": list(key[:n_atoms]), "funct": funct,
                            "param": labels[i] if i < len(labels) else f"p{i + 1}",
                            "before": x, "after": y,
                            "rel_change": round((y - x) / abs(x), 4) if x else None,
                            "at_floor": bool(floor),
                        })
                        counts[section] = counts.get(section, 0) + 1
    return {"n_terms_compared": n_terms, "n_changed": len(rows),
            "changed_by_section": counts, "n_at_floor": at_floor,
            "floor_fraction": floor_fraction, "rows": rows}


# -- ledger walk ----------------------------------------------------------------

@dataclass
class FMRound:
    stage: str
    results: ProvenanceEvent
    fit: ProvenanceEvent | None = None
    plan: ProvenanceEvent | None = None
    collect: ProvenanceEvent | None = None
    merge: ProvenanceEvent | None = None
    candidate: ProvenanceEvent | None = None
    validate: ProvenanceEvent | None = None
    validate_results: ProvenanceEvent | None = None
    attempts: int = 1  # fm_results calls on this stage (failed ones included)
    notes: list[str] = field(default_factory=list)


def _out_paths(ev: ProvenanceEvent) -> set[str]:
    return {a.path for a in ev.outputs}


def _last(events: list[ProvenanceEvent], tool: str, pred) -> ProvenanceEvent | None:
    for ev in reversed(events):
        if ev.tool == tool and ev.kind == "call" and ev.ok and pred(ev):
            return ev
    return None


_STAGE_NUMBER_RE = re.compile(r"^\d{2,}_")


def _stage_of_output(ev: ProvenanceEvent, role: str) -> str | None:
    """The un-numbered stage an event's output of that role landed in."""
    art = next((a for a in ev.outputs if a.role == role), None)
    if art is None:
        return None
    parts = Path(art.path).parts
    return _STAGE_NUMBER_RE.sub("", parts[0]) if len(parts) > 1 else None


def is_fit_collection(ev: ProvenanceEvent) -> bool:
    """A fit read back into facts: an ``fm_results`` call, or the ``job_status`` call that
    collected an ``fm_fit`` job (it carries the fm_log)."""
    return ev.kind == "call" and (
        ev.tool == "fm_results"
        or (ev.tool == "job_status" and any(a.role == "fm_log" for a in ev.outputs))
    )


def _fit_stage(ev: ProvenanceEvent) -> str:
    if ev.tool == "fm_results":
        return ev.params.get("stage", "fm")
    return _stage_of_output(ev, "fm_log") or "fm"


def _validation_collection(events: list[ProvenanceEvent], vstage: str):
    """The validation read back for that stage: an ``fm_validate_results`` call, or the
    ``job_status`` call that collected the ``fm_validate`` job."""
    old = _last(events, "fm_validate_results",
                lambda e: e.params.get("stage", "validate") == vstage)
    if old is not None:
        return old
    return _last(events, "job_status", lambda e: _stage_of_output(e, "validation") == vstage)


def _link_round(events: list[ProvenanceEvent], stage: str,
                evs: list[ProvenanceEvent]) -> FMRound:
    ok = [e for e in evs if e.ok]
    rnd = FMRound(stage=stage, results=(ok or evs)[-1], attempts=len(evs))
    if not ok:
        rnd.notes.append("no clean fm_results on this stage: the fit did not complete")
    rnd.fit = _last(events, "fm_fit", lambda e: e.params.get("stage", "fm") == stage)
    if rnd.fit:
        fm_input = rnd.fit.params.get("fm_input")
        fmdata = rnd.fit.params.get("fmdata")
        rnd.plan = _last(events, "fm_plan", lambda e: fm_input in _out_paths(e)) or _last(
            events, "fm_plan", lambda e: e.params.get("stage") == stage)
        rnd.collect = _last(events, "fm_collect", lambda e: fmdata in _out_paths(e))
        rnd.merge = _last(events, "fm_merge", lambda e: fmdata in _out_paths(e))
    if any(a.role == "topology" for a in rnd.results.outputs):
        rnd.candidate = rnd.results  # the collection assembled it
    else:
        fit_dir = next((str(Path(a.path).parent) for a in rnd.results.outputs
                        if a.role == "fitted_topology"), None)
        if fit_dir is not None:
            rnd.candidate = _last(
                events, "fm_candidate_topology",
                lambda e: any(str(Path(v).parent) == fit_dir
                              for v in e.params.get("replace", {}).values()))
    if rnd.candidate:
        cand_top = next((a.path for a in rnd.candidate.outputs if a.role == "topology"), None)
        rnd.validate = _last(
            events, "fm_validate",
            lambda e: cand_top in e.params.get("topologies", {}).values())
    if rnd.validate:
        vstage = rnd.validate.params.get("stage", "validate")
        rnd.validate_results = _validation_collection(events, vstage)
    return rnd


def find_rounds(events: list[ProvenanceEvent]) -> list[FMRound]:
    """One round per fit stage whose fit was read back, linked by artifact paths."""
    by_stage: dict[str, list[ProvenanceEvent]] = {}
    for ev in events:
        if is_fit_collection(ev):
            by_stage.setdefault(_fit_stage(ev), []).append(ev)
    rounds = [_link_round(events, stage, evs) for stage, evs in by_stage.items()]
    rounds.sort(key=lambda r: r.results.timestamp)
    return rounds


# -- report assembly --------------------------------------------------------------

def _read(root: Path, rel: str | None) -> str | None:
    if not rel:
        return None
    p = root / rel
    return p.read_text(errors="replace") if p.exists() else None


def _art(ev: ProvenanceEvent | None, role: str) -> ArtifactRef | None:
    if ev is None:
        return None
    return next((a for a in ev.outputs if a.role == role), None)


def _ref(a: ArtifactRef | None) -> dict | None:
    return {"path": a.path, "sha256": a.sha256} if a else None


def _dataset_block(root: Path, rnd: FMRound) -> dict:
    d: dict = {}
    if rnd.merge:
        d["sources"] = rnd.merge.params.get("fm_jsons", [])
    if rnd.collect:
        h5 = _art(rnd.collect, "fm_dataset")
        trr = _art(rnd.collect, "trajectory")
        d["fm_dataset"] = _ref(h5)
        d["trajectory"] = _ref(trr)
        check = _read(root, (_art(rnd.collect, "fmcheck_log") or ArtifactRef(
            path="", role="", sha256="", size_bytes=0)).path)
        if check:
            facts = parse_fmcheck_report(check)
            d["frames"] = facts.get("h5_frames")
            d["trr_frames"] = facts.get("trr_frames")
            d["index_aligned"] = facts.get("index_aligned")
    elif rnd.fit:
        d["fm_dataset"] = {"path": rnd.fit.params.get("fmdata"), "sha256": None}
    return d


def _plan_block(rnd: FMRound) -> dict:
    if rnd.plan is None:
        return {}
    p = rnd.plan.params
    d = {k: p.get(k) for k in ("optimization_method", "stride", "qm_total_charge",
                               "reference_charges", "eq_atoms", "dataset_frames")}
    d["weights"] = {k: p.get(k) for k in ("wv", "we", "wh", "wq") if p.get(k) is not None}
    d["extra"] = p.get("extra", {})
    d["fm_input"] = _ref(_art(rnd.plan, "fm_input"))
    d["acknowledged"] = list(rnd.plan.warnings_acknowledged)
    return d


def _fit_block(root: Path, rnd: FMRound) -> dict:
    log_ref = _art(rnd.results, "fm_log")
    facts: dict = {}
    text = _read(root, log_ref.path if log_ref else None)
    if text:
        facts = parse_fm_log(text)
        res = root / Path(log_ref.path).parent / "optimization_results.txt"
        if res.exists():
            for k, v in parse_optimization_results(res.read_text()).items():
                facts.setdefault(k, v)
    d = {k: facts.get(k) for k in (
        "n_snapshots", "n_train", "n_val", "validation_sdf", "validation_cost",
        "optimizer_success", "optimizer_message", "qm_total_charge", "wall_time_s",
        "completed")}
    d["fm_log"] = _ref(log_ref)
    d["mode"] = rnd.fit.params.get("mode") if rnd.fit else None
    d["n_processes"] = rnd.fit.params.get("n_processes") if rnd.fit else None
    d["acknowledged"] = list(rnd.fit.warnings_acknowledged) if rnd.fit else []
    d["attempts"] = rnd.attempts
    d["forces"] = {}
    if log_ref:
        csv_path = root / Path(log_ref.path).parent / VALIDATION_RESIDUALS_FILENAME
        if csv_path.exists():
            d["forces"] = parse_validation_force_residuals(csv_path.read_text(errors="replace"))
            if d["forces"]:
                d["forces"]["file"] = _ref(ArtifactRef.from_path(csv_path, "force_residuals",
                                                                 root=root))
    if text and not (facts.get("completed") and rnd.results.ok):
        ends = [ln for ln in text.splitlines() if ln.strip()][-3:]
        d["log_ends_with"] = ends
    return d


def _originals(root: Path, rnd: FMRound) -> dict[str, str]:
    """fitted itp path -> original itp path, from the candidate's replace map, else from
    fm_results.original_itp (its fitted twin is opt_<name>)."""
    pairs: dict[str, str] = {}
    if rnd.candidate and rnd.candidate.params.get("replace"):  # a separate candidate call
        top = Path(rnd.candidate.params.get("topology", ""))
        for inc, fitted in rnd.candidate.params.get("replace", {}).items():
            orig = top.parent / inc
            if (root / orig).exists():
                pairs[fitted] = str(orig)
    orig = rnd.results.params.get("original_itp") or next(
        (a.path for a in rnd.results.inputs if a.role == "original_topology"), None)
    if orig:
        for a in rnd.results.outputs:
            if a.role == "fitted_topology" and Path(a.path).name == f"opt_{Path(orig).name}":
                pairs.setdefault(a.path, orig)
    return pairs


def _floor_fraction(rnd: FMRound) -> float | None:
    extra = rnd.plan.params.get("extra", {}) if rnd.plan else {}
    if str(extra.get("regularization", "")).lower() != "true":
        return None
    try:
        return float(extra.get("regularization_force_constant_min_fraction"))
    except (TypeError, ValueError):
        return None


def _parameters_block(root: Path, rnd: FMRound) -> list[dict]:
    pairs = _originals(root, rnd)
    floor = _floor_fraction(rnd)
    out: list[dict] = []
    for a in rnd.results.outputs:
        if a.role != "fitted_topology":
            continue
        fitted_text = _read(root, a.path)
        entry: dict = {"fitted": _ref(a), "original": None}
        if fitted_text is None:
            entry["note"] = "fitted file missing from disk"
            out.append(entry)
            continue
        entry["unphysical"] = unphysical_force_constants(fitted_text)
        orig = pairs.get(a.path)
        orig_text = _read(root, orig)
        if orig_text is None:
            entry["note"] = ("not compared: no original .itp recorded for this file "
                             "(not part of the candidate topology)")
            out.append(entry)
            continue
        entry["original"] = _ref(ArtifactRef.from_path(root / orig, "topology", root=root))
        atoms_before, atoms_after = parse_itp_atoms(orig_text), parse_itp_atoms(fitted_text)
        entry["charges"] = charge_diff(atoms_before, atoms_after)
        entry["bonded"] = bonded_diff(parse_itp_bonded(orig_text), parse_itp_bonded(fitted_text),
                                      atoms_after, floor_fraction=floor)
        out.append(entry)
    return out


def _validation_block(root: Path, rnd: FMRound) -> dict:
    if rnd.validate_results is None:
        return {}
    ref = _art(rnd.validate_results, "validation")
    text = _read(root, ref.path if ref else None)
    if not text:
        return {"note": "validation results file missing", "results": _ref(ref)}
    data = json.loads(text)
    d: dict = {"results": _ref(ref), "ok": data.get("ok"),
               "reference_frames": data.get("reference_frames"),
               "reference": data.get("reference", {}), "metrics": {}}
    if rnd.validate:
        d["topologies"] = rnd.validate.params.get("topologies", {})
        d["monitor_pairs"] = rnd.validate.params.get("monitor_pairs", {})
        d["min_frames"] = data.get("min_frames", rnd.validate.params.get("min_frames"))
    for metric in ("min_geometry", "md_geometry"):
        block = data.get(metric)
        if block:
            d["metrics"][metric] = {
                "mae": {n: block[n].get("mae") for n in block},
                "pairs": {n: block[n].get("pairs", {}) for n in block},
                "n_frames": {n: block[n].get("n_frames_minimized") for n in block},
            }
    if data.get("force_rmse"):
        d["force_rmse"] = data["force_rmse"]
        d["force_frames"] = data.get("force_frames", [])
    return d


def build_fm_report(ws: Workspace, compute=None, backend=None) -> dict:
    """The campaign as data: every value re-parsed from a ledger-recorded file."""
    events = list(ws.ledger.events())
    root = ws.root
    versions: dict[str, str] = {}
    myr = ""
    for e in events:
        versions.update(e.engine_versions)
        myr = e.myrmidon_version or myr
    single_points = [e for e in events if e.tool == "mimic_run" and e.kind == "call" and e.ok]
    frames = [e for e in events if e.tool == "traj_frames" and e.kind == "call" and e.ok]
    refusals = [e for e in events if e.kind == "gate_refused"]
    rounds = []
    for rnd in find_rounds(events):
        rounds.append({
            "stage": rnd.stage, "timestamp": rnd.results.timestamp.isoformat(timespec="seconds"),
            "clean": rnd.results.ok, "notes": rnd.notes,
            "dataset": _dataset_block(root, rnd),
            "plan": _plan_block(rnd),
            "fit": _fit_block(root, rnd),
            "parameters": _parameters_block(root, rnd),
            "candidate": _ref(_art(rnd.candidate, "topology")),
            "validation": _validation_block(root, rnd),
        })
    s = summarize(ws, "all", backend=backend, compute=compute, poll_usage=backend is not None)
    meta = ws.meta
    return {
        "workspace": {"name": meta.get("name", root.name), "path": str(root),
                      "description": meta.get("description", ""),
                      "created": meta.get("created")},
        "engines": {k: v for k, v in sorted(versions.items()) if v != "unknown"},
        "myrmidon_version": myr,
        "reference": {
            "single_points": len(single_points),
            "snapshot_selections": [
                {"trajectory": e.params.get("trajectory"), "frames": e.params.get("frames"),
                 "n_frames": e.params.get("n_frames")} for e in frames],
        },
        "rounds": rounds,
        "gate_refusals": [
            {"tool": e.tool, "timestamp": e.timestamp.isoformat(timespec="seconds"),
             "findings": e.knowledge_refs} for e in refusals],
        "compute": {
            "jobs": [j.model_dump() for j in s.jobs],
            "core_h_booked": s.core_h_submitted, "core_h_consumed": s.core_h_consumed,
        },
        "knowledge_refs": s.knowledge_refs,
    }


# -- rendering ----------------------------------------------------------------------

def _fmt(v, nd=4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) < 1e4 else f"{v:.4g}"
    return str(v)


def _md_table(header: list[str], rows: list[list], max_rows: int | None = None,
              more_note: str = "") -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    shown = rows if max_rows is None else rows[:max_rows]
    lines += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in shown]
    if max_rows is not None and len(rows) > max_rows:
        lines.append(f"| … | {len(rows) - max_rows} more rows{more_note} |"
                     + " |" * (len(header) - 2))
    return lines


def render_fm_report(data: dict, max_rows: int = 40) -> str:
    w = data["workspace"]
    L = [f"# {w['name']} — force-matching campaign report", "",
         f"workspace: `{w['path']}`  ", f"created: {w.get('created') or '?'}  "
         + (f"description: {w['description']}" if w.get("description") else ""),
         "engines: " + (", ".join(f"{k} {v}" for k, v in data["engines"].items()) or "—")
         + (f"; myrmidon {data['myrmidon_version']}" if data["myrmidon_version"] else ""),
         "", "Every number below is parsed from a file the ledger recorded (hash given); "
         "nothing is transcribed from a session.", ""]
    ref = data["reference"]
    L += ["## Reference data", "",
          f"- MiMiC single points run in this workspace: {ref['single_points']}"]
    for sel in ref["snapshot_selections"]:
        L.append(f"- snapshots from `{sel['trajectory']}`: "
                 + (f"frames {sel['frames']}" if sel.get("frames") else
                    f"{sel.get('n_frames')} evenly spaced"))
    if not data["rounds"]:
        L += ["", "## Rounds", "", "no fm_results event in the ledger — nothing was fitted here."]
    for i, r in enumerate(data["rounds"], 1):
        L += ["", f"## Round {i}: stage `{r['stage']}` ({r['timestamp']})"
              + ("" if r["clean"] else " — NOT CLEAN"), ""]
        for n in r["notes"]:
            L.append(f"> {n}")
        ds = r["dataset"]
        if ds:
            L += ["### Dataset", ""]
            if ds.get("sources"):
                L.append(f"- merged from {len(ds['sources'])} source(s): "
                         + ", ".join(f"`{s}`" for s in ds["sources"]))
            for key in ("fm_dataset", "trajectory"):
                if ds.get(key):
                    L.append(f"- {key}: `{ds[key]['path']}`"
                             + (f" ({ds[key]['sha256'][:12]})" if ds[key].get("sha256") else ""))
            if ds.get("frames") is not None:
                L.append(f"- frames: {ds['frames']} (h5) / {ds.get('trr_frames')} (trr), "
                         f"index-aligned: {_fmt(ds.get('index_aligned'))}")
            L.append("")
        p = r["plan"]
        if p:
            L += ["### Fit settings", "", "| setting | value |", "|---|---|"]
            for k in ("optimization_method", "stride", "qm_total_charge", "reference_charges",
                      "eq_atoms", "dataset_frames"):
                if p.get(k) is not None:
                    L.append(f"| {k} | {p[k]} |")
            if p.get("weights"):
                L.append("| weights | " + ", ".join(f"{k}={v}" for k, v in p["weights"].items())
                         + " |")
            for k, v in sorted(p.get("extra", {}).items()):
                L.append(f"| {k} | {v} |")
            if p.get("fm_input"):
                L.append(f"| fminput.dat | `{p['fm_input']['path']}` "
                         f"({p['fm_input']['sha256'][:12]}) |")
            if p.get("acknowledged"):
                L.append(f"| acknowledged | {', '.join(p['acknowledged'])} |")
            L.append("")
        f = r["fit"]
        L += ["### Fit outcome", "", "| quantity | value |", "|---|---|"]
        for k, label in (("mode", "mode"), ("n_snapshots", "snapshots used"),
                         ("n_train", "training"), ("n_val", "validation"),
                         ("validation_sdf", "validation SDF"),
                         ("validation_cost", "validation cost"),
                         ("optimizer_success", "optimizer success"),
                         ("optimizer_message", "optimizer message"),
                         ("wall_time_s", "wall time (s)"), ("n_processes", "processes"),
                         ("attempts", "fm_results calls on this stage")):
            if f.get(k) is not None:
                L.append(f"| {label} | {_fmt(f[k])} |")
        if f.get("fm_log"):
            L.append(f"| fm.log | `{f['fm_log']['path']}` ({f['fm_log']['sha256'][:12]}) |")
        if f.get("acknowledged"):
            L.append(f"| acknowledged | {', '.join(f['acknowledged'])} |")
        if f.get("log_ends_with"):
            L += ["", "fm.log ends with:", "", "```", *f["log_ends_with"], "```"]
        L.append("")
        fo = f.get("forces") or {}
        if fo:
            L += [f"### Held-out force error ({fo['n_frames']} validation frames, "
                  f"{fo['n_atoms']} QM atoms; {fo.get('units', 'Hartree/Bohr')})", "",
                  f"overall RMSE {_fmt(fo['rmse'], 2)} vs reference bonded-force RMS "
                  f"{_fmt(fo['ref_rms'], 2)} (relative {_fmt(fo['rel_rmse'], 3)}); "
                  f"residual table `{fo['file']['path']}` ({fo['file']['sha256'][:12]})", "",
                  "by atom type:", ""]
            L += _md_table(["type", "element", "n atoms", "RMSE", "ref RMS", "relative"],
                           [[t["type"], t["element"], t["n_atoms"], t["rmse"], t["ref_rms"],
                             t["rel_rmse"]] for t in fo["by_type"]], max_rows,
                           " (all in fm_report.json)")
            if fo["outliers"]:
                L += ["", f"**{len(fo['outliers'])} outlier atom(s)** (RMSE above 3× the median "
                      f"atom's {_fmt(fo['median_atom_rmse'], 2)}):", ""]
                L += _md_table(["atom", "residue", "type", "boundary", "RMSE", "relative"],
                               [[a["name"], f"{a['resname']}{a['resid']}", a["type"],
                                 a["is_bound"], a["rmse"], a["rel_rmse"]]
                                for a in fo["outliers"]], max_rows, "")
            else:
                L += ["", f"no outlier atoms (none above 3× the median atom's "
                      f"{_fmt(fo['median_atom_rmse'], 2)}); worst: "
                      + ", ".join(f"{a['resname']}{a['resid']}:{a['name']} {_fmt(a['rmse'], 2)}"
                                  for a in fo["worst_atoms"][:3])]
            L.append("")
        for entry in r["parameters"]:
            fitted = entry["fitted"]["path"]
            L += [f"### Parameters: `{fitted}` ({entry['fitted']['sha256'][:12]})", ""]
            if entry.get("note"):
                L += [f"> {entry['note']}", ""]
            if entry.get("unphysical"):
                L += [f"> **{len(entry['unphysical'])} unphysical bonded term(s)** (k ≤ 0): "
                      + "; ".join(entry["unphysical"][:5]), ""]
            if not entry.get("original"):
                continue
            L.append(f"original: `{entry['original']['path']}` "
                     f"({entry['original']['sha256'][:12]})")
            c = entry["charges"]
            L += ["", f"**Charges** — {c['n_changed']} of {c['n_atoms_compared']} atoms changed; "
                  f"max |Δq| {_fmt(c['max_abs_dq'])}, mean |Δq| {_fmt(c['mean_abs_dq'])}, "
                  f"net Δq {_fmt(c['net_dq'])} e", ""]
            if c["rows"]:
                L += _md_table(
                    ["molecule", "nr", "atom", "type", "q before", "q after", "Δq"],
                    [[x["molecule"], x["nr"], x["atom"], x["type"], x["q_before"],
                      x["q_after"], x["dq"]] for x in c["rows"]],
                    max_rows, " (all in fm_report.json)")
                L += ["", "by atom type:", ""]
                L += _md_table(["type", "n atoms", "mean Δq", "max |Δq|"],
                               [[t["type"], t["n"], t["mean_dq"], t["max_abs_dq"]]
                                for t in c["by_type"]], max_rows, "")
            b = entry["bonded"]
            if b["n_terms_compared"] == 0:
                L += ["", "**Bonded terms** — none explicit in this .itp (parameters come from "
                      "the force-field atom types; nothing was refitted here)", ""]
            else:
                L += ["", f"**Bonded terms** — {b['n_changed']} parameter(s) changed in "
                      f"{b['n_terms_compared']} compared term(s)"
                      + (": " + ", ".join(f"{k} {v}" for k, v in b["changed_by_section"].items())
                         if b["changed_by_section"] else "")
                      + (f"; **{b['n_at_floor']} force constant(s) sit exactly on the "
                         f"regularization floor ({b['floor_fraction']:g} × original)** — the "
                         "fit wanted them lower" if b.get("n_at_floor") else ""), ""]
            if b["rows"]:
                L += _md_table(
                    ["section", "atoms", "funct", "param", "before", "after", "rel. change"],
                    [[x["section"], x["atoms"], x["funct"], x["param"], x["before"],
                      x["after"], (f"{x['rel_change']:+.4f}" + (" (floor)" if x["at_floor"]
                                                                 else "")
                                   if x["rel_change"] is not None else "—")]
                     for x in b["rows"]],
                    max_rows, " (all in fm_report.json)")
            L.append("")
        if r.get("candidate"):
            L += [f"candidate topology: `{r['candidate']['path']}` "
                  f"({r['candidate']['sha256'][:12]})", ""]
        v = r["validation"]
        if v:
            L += ["### Validation vs the QM/MM reference", ""]
            if v.get("note"):
                L += [f"> {v['note']}", ""]
            if v.get("reference_frames") is not None:
                L.append(f"reference ensemble: {v['reference_frames']} frames"
                         + (f"; minimized frames {v['min_frames']}" if v.get("min_frames")
                            else ""))
            for name, top in (v.get("topologies") or {}).items():
                L.append(f"- {name}: `{top}`")
            for metric, m in v.get("metrics", {}).items():
                L += ["", f"**{metric}** (Å; MAE over the monitored pairs)", "", "```",
                      _table(v["reference"], {n: {"pairs": m["pairs"][n], "mae": m["mae"][n]}
                                              for n in m["mae"]}), "```"]
                ranked = sorted(((n, x) for n, x in m["mae"].items() if x is not None),
                                key=lambda kv: kv[1])
                if len(ranked) >= 2:
                    best, ref_ = ranked[0], ranked[-1]
                    L.append(f"best: {best[0]} {best[1]:.3f} Å vs {ref_[0]} {ref_[1]:.3f} Å")
            fr = v.get("force_rmse") or {}
            if fr:
                good = {n: b for n, b in fr.items() if b.get("rmse") is not None}
                units = next((b.get("units") for b in good.values()), "Hartree/Bohr")
                L += ["", f"**force_rmse** — held-out total force on the QM atoms, candidate "
                      f"rerun vs QM/MM reference ({len(v.get('force_frames', []))} frames; "
                      f"{units})", ""]
                L += _md_table(["candidate", "RMSE", "ref RMS", "relative", "outlier atoms"],
                               [[n, b["rmse"], b["ref_rms"], b["rel_rmse"],
                                 len(b.get("outliers", []))]
                                for n, b in sorted(good.items(), key=lambda kv: kv[1]["rmse"])])
                for n in fr:
                    if n not in good:
                        L.append(f"- {n}: {fr[n].get('error', 'no result')}")
                table = force_type_table(good)
                if table:
                    L += ["", "per atom type:", "", "```", table, "```"]
                for n, b in good.items():
                    if b.get("outliers"):
                        L += ["", f"{n} outliers (RMSE above 3× the median atom's):", ""]
                        L += _md_table(["atom", "residue", "type", "boundary", "RMSE", "relative"],
                                       [[a["name"], f"{a['resname']}{a['resid']}", a["type"],
                                         a["is_bound"], a["rmse"], a["rel_rmse"]]
                                        for a in b["outliers"]], max_rows, "")
            if v.get("results"):
                L.append(f"results: `{v['results']['path']}` ({v['results']['sha256'][:12]})")
            L.append("")
    if data["gate_refusals"]:
        L += ["## Gate refusals", ""]
        L += [f"- {g['timestamp']} {g['tool']}"
              + (f" (findings: {', '.join(g['findings'])})" if g["findings"] else "")
              for g in data["gate_refusals"]]
        L.append("")
    c = data["compute"]
    L += ["## Compute", ""]
    if c["jobs"]:
        L += _md_table(["job", "label", "state", "core-h booked", "core-h used"],
                       [[j["job_id"], j.get("label") or "", j["state"],
                         j.get("core_h_estimate"), j.get("core_h_actual")] for j in c["jobs"]])
        L.append(f"\ntotal: {_fmt(c['core_h_booked'], 1)} core-h booked"
                 + (f", {_fmt(c['core_h_consumed'], 1)} consumed"
                    if c.get("core_h_consumed") is not None else ""))
    else:
        L.append("no jobs recorded")
    if data["knowledge_refs"]:
        L += ["", "findings surfaced: " + ", ".join(data["knowledge_refs"])]
    L += ["", f"Reproduce: `myrmidon replay {w['path']} --into <new-dir>`", ""]
    return "\n".join(L)


# -- tool ------------------------------------------------------------------------------

class FMReportObservation(Observation):
    n_rounds: int = 0
    report_path: str | None = None
    json_path: str | None = None
    rounds: list[dict] = Field(default_factory=list)  # stage, clean, validation_sdf, mae


class FMReportParams(BaseModel):
    stage: str = Field(default="report", description="stage to write the report into")
    prefix: str = Field(default="fm_report", description="<prefix>.md and <prefix>.json")
    max_rows: int = Field(default=40, ge=1, description="table rows per section in the .md")


@tool(
    "fm_report",
    description="The force-matching campaign report from the ledger alone: per fit round "
    "the dataset, fit settings, fit outcome, per-atom / per-atom-type charge changes and "
    "changed bonded terms (original vs fitted .itp), candidate topology, validation MAE vs "
    "the QM/MM reference, plus compute and gate refusals. Writes <stage>/<prefix>.md "
    "(publication tables) and <prefix>.json (every row).",
    returns=FMReportObservation,
    tags={"mimic", "fm", "report"},
)
def fm_report(ctx: ToolContext, params: FMReportParams) -> FMReportObservation:
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
        s = f"{d['stage']}: {d['n_snapshots']} snapshots, SDF {_fmt(d['validation_sdf'])}"
        if d["n_unphysical"]:
            s += f", {d['n_unphysical']} UNPHYSICAL terms"
        for metric, mae in d["mae"].items():
            s += f", {metric} MAE " + ", ".join(
                f"{n}={x:.3f}" if x is not None else f"{n}=?" for n, x in mae.items())
        bits.append(s)
    return FMReportObservation(
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
