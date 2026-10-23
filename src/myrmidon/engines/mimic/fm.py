"""MiMiCPy force-matching wrap.

The ``mimicpy fm`` workflow, step by step: ``fm_collect`` registers and
validates the reference dataset (frame counts via h5info, trr<->h5 alignment via
fmcheck, sha256 of both files), ``fm_plan`` writes an explicit ``fminput.dat``
(stride and QM total charge are REQUIRED; weight defaults surface as assumption
warnings), ``fm_fit`` submits the fit asynchronously through the execution
backend, and ``collect_fit`` (via ``job_status``) parses fm.log / optimization_results.txt / the
optimized .itp into typed facts — the numbers come from MiMiCPy's output, never
from the model.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import subprocess
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import JobObservation, Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning
from myrmidon.engines.gromacs.common import stage_output, tail
from myrmidon.engines.gromacs.common import stage_topology as _stage_topology
from myrmidon.engines.gromacs.itp import atom_charges, bond_angle_rows
from myrmidon.engines.gromacs.results import register_collector
from myrmidon.engines.gromacs.run import get_jobstore
from myrmidon.engines.inputs import require_input, resolve_in_ws
from myrmidon.engines.mimic.qm_region import mimicpy_bin
from myrmidon.execution.backend import JobSpec, JobState, Resources

# -- pure parsers (unit-tested against real MiMiCPy output fixtures) ----------


def parse_h5info(text: str) -> dict:
    facts: dict = {}
    m = re.search(r"Configurations:\s*(\d+)", text)
    if m:
        facts["frames"] = int(m.group(1))
    m = re.search(r"config_\d+:\s*\d+\s*atoms\s*\((\d+)\s*QM,\s*(\d+)\s*MM\)", text)
    if m:
        facts["qm_atoms"] = int(m.group(1))
    return facts


def parse_fmcheck_report(text: str) -> dict:
    facts: dict = {}
    for key, name in (
        ("trr frames", "trr_frames"),
        ("h5 configs", "h5_frames"),
        ("requested frames", "checked_frames"),
        ("aligned \\(index\\)", "aligned"),
        ("dropped \\(no twin\\)", "dropped"),
        ("matched total", "matched"),
    ):
        m = re.search(rf"{key}\s*:\s*(\d+)", text)
        if m:
            facts[name] = int(m.group(1))
    m = re.search(r"remapped \(coord\)\s*:\s*(\d+)", text)
    if m:
        facts["remapped"] = int(m.group(1))
    m = re.search(r"status\s*:\s*(\S+)", text)
    if m:
        facts["status"] = m.group(1)
    facts["index_aligned"] = "NOT index-aligned" not in text
    return facts


def _wall_seconds(text: str) -> float | None:
    m = re.match(r"(?:(\d+) days?, )?(\d+):(\d\d):(\d\d(?:\.\d+)?)", text.strip())
    if not m:
        return None
    days = int(m.group(1) or 0)
    return days * 86400 + int(m.group(2)) * 3600 + int(m.group(3)) * 60 + float(m.group(4))


def parse_fm_log(text: str) -> dict:
    facts: dict = {}
    pats = {
        "n_snapshots": r"Number of snapshots used for fitting:\s*(\d+)",
        "qm_total_charge": r"QM total charge:\s*([-\d.eE+]+)",
    }
    for name, pat in pats.items():
        m = re.search(pat, text)
        if m:
            v = m.group(1)
            facts[name] = int(v) if name == "n_snapshots" else float(v)
    m = re.search(r"Split (\d+) configurations into (\d+) training and (\d+) validation", text)
    if m:
        facts["n_train"], facts["n_val"] = int(m.group(2)), int(m.group(3))
    m = re.search(r"Validation SD[BF]:\s*([-\d.eE+]+)", text)
    if m:
        facts["validation_sdf"] = float(m.group(1))
    m = re.search(r"Validation cost:\s*([-\d.eE+]+)", text)
    if m:
        facts["validation_cost"] = float(m.group(1))
    m = re.search(r"success:\s*(True|False)", text)
    if m:
        facts["optimizer_success"] = m.group(1) == "True"
    m = re.search(r"message:\s*`?([^`\n]+)", text)
    if m:
        facts["optimizer_message"] = m.group(1).strip()
    # one line per pass, a comma-separated list when the QM region spans several
    # molecules — a metalloenzyme site is typically ligands, two protein chains, the
    # ion's own moleculetype and water
    itps: list[str] = []
    for line in re.findall(r"wrote \.itp files for molecules containing QM atoms:\s*(.+)", text):
        itps += [t.strip().rstrip(",") for t in line.split(",") if t.strip()]
    if itps:
        facts["itp_files"] = sorted(set(itps))
    facts["dresp_completed"] = "DRESP optimization completed" in text
    facts["completed"] = "workflow completed" in text
    m = re.search(r"wall time:\s*([\d:,. days]+)", text)
    if m:
        w = _wall_seconds(m.group(1))
        if w is not None:
            facts["wall_time_s"] = round(w, 1)
    return facts


def parse_optimization_results(text: str) -> dict:
    facts: dict = {}
    rows = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    facts["n_optimizer_rows"] = len(rows)
    m = re.search(r"#\s*(?:Final )?Validation SD[BF]:\s*([-\d.eE+]+)", text)
    if m:
        facts["validation_sdf"] = float(m.group(1))
    m = re.search(r"#\s*(?:Final )?Validation cost:\s*([-\d.eE+]+)", text)
    if m:
        facts["validation_cost"] = float(m.group(1))
    m = re.search(r"#\s*Validation configurations:\s*(\d+)", text)
    if m:
        facts["n_val"] = int(m.group(1))
    return facts


VALIDATION_RESIDUALS_FILENAME = "validation_force_residuals.csv"
FORCE_OUTLIER_FACTOR = 3.0  # an atom whose validation RMSE exceeds this × the median atom's
FORCE_UNITS = "Hartree/Bohr"  # mimicpy compares forces in atomic units (kjmolnm_to_au)


def rerun_mdp_warnings(mdp: dict) -> list[ValidationWarning]:
    """What a force-matching / force-comparison rerun mdp must not be: a MiMiC production
    mdp (FM_RERUN_MDP_IS_MIMIC below), or one that writes no forces. Shared by fm_fit and
    fm_validate."""
    warnings: list[ValidationWarning] = []
    qmmm_keys = [k for k in mdp if k.replace("-", "_") == "qmmm_grps"]
    if mdp.get("integrator", "").lower() == "mimic" or qmmm_keys:
        warnings.append(
            ValidationWarning(
                id="FM_RERUN_MDP_IS_MIMIC",
                severity="block",
                message="the rerun mdp is a MiMiC production mdp ("
                + ", ".join(
                    [f"integrator = {mdp['integrator']}"] if mdp.get("integrator", "").lower()
                    == "mimic" else []
                ) + ("".join(f", {k} = {mdp[k]}" for k in qmmm_keys) if qmmm_keys else "")
                + "): the force-matching rerun must be a plain classical mdp, or the "
                "bonded target loses the QM electrostatics — and mimicpy "
                "refuses it after the queue wait anyway",
                suggestion="integrator = md, no QMMM-grps, nstfout = 1",
            )
        )
    if int(mdp.get("nstfout", "0") or 0) < 1:
        warnings.append(
            ValidationWarning(
                id="FM_MDP_NO_FORCES",
                severity="block",
                message="the rerun mdp has nstfout < 1 — mdrun -rerun would write no "
                "forces and the parameter fit reads classical forces from that trr",
                suggestion="set nstfout = 1 in the rerun mdp",
            )
        )
    return warnings


def _rms(sq_sum: float, n: int) -> float | None:
    return math.sqrt(sq_sum / n) if n else None


def parse_validation_force_residuals(text: str, n_atoms_listed: int = 10) -> dict:
    """Per-atom-type and per-atom force RMSE from a residual table in mimicpy's
    validation_force_residuals.csv layout (one row per frame × QM atom: the residual
    and the reference force it is measured against, atomic units — Hartree/Bohr —
    as mimicpy compares them; ×49614.75 for kJ mol⁻¹ nm⁻¹, CODATA 2018).

    RMSE is over the vector residual |F_fit − F_ref| per (frame, atom) sample; ref_rms
    the same for the reference bonded force, so rel_rmse = rmse / ref_rms is the
    per-type analogue of the validation SDF. Atoms whose RMSE exceeds
    FORCE_OUTLIER_FACTOR × the median atom RMSE are the outliers (the median, not the
    overall RMSE, because one bad atom dominates the overall in a small QM region)."""
    reader = csv.DictReader(io.StringIO(text))
    atoms: dict[str, dict] = {}
    frames: set[str] = set()
    tot_sq = tot_ref = 0.0
    n = 0
    for row in reader:
        try:
            sq = sum(float(row[k]) ** 2 for k in ("res_x", "res_y", "res_z"))
            rq = sum(float(row[k]) ** 2 for k in ("ref_x", "ref_y", "ref_z"))
        except (KeyError, ValueError):
            continue
        frames.add(row.get("frame_index", ""))
        a = atoms.setdefault(row["gmx_id"], {
            "gmx_id": int(row["gmx_id"]), "name": row.get("name", ""),
            "type": row.get("type", ""), "resname": row.get("resname", ""),
            "resid": row.get("resid", ""), "element": row.get("element", ""),
            "is_bound": row.get("is_bound", "0") == "1", "sq": 0.0, "ref": 0.0, "n": 0})
        a["sq"] += sq
        a["ref"] += rq
        a["n"] += 1
        tot_sq += sq
        tot_ref += rq
        n += 1
    if not n:
        return {}
    rmse = _rms(tot_sq, n)
    ref_rms = _rms(tot_ref, n)
    types: dict[str, dict] = {}
    for a in atoms.values():
        t = types.setdefault(a["type"], {"type": a["type"], "n_atoms": 0, "n_samples": 0,
                                         "sq": 0.0, "ref": 0.0, "elements": set()})
        t["n_atoms"] += 1
        t["n_samples"] += a["n"]
        t["sq"] += a["sq"]
        t["ref"] += a["ref"]
        if a["element"]:
            t["elements"].add(a["element"])
    by_type = []
    for t in types.values():
        r, rr = _rms(t["sq"], t["n_samples"]), _rms(t["ref"], t["n_samples"])
        by_type.append({"type": t["type"], "element": "/".join(sorted(t["elements"])),
                        "n_atoms": t["n_atoms"], "n_samples": t["n_samples"],
                        "rmse": round(r, 6), "ref_rms": round(rr, 6),
                        "rel_rmse": round(r / rr, 6) if rr else None})
    by_type.sort(key=lambda d: -d["rmse"])
    per_atom = []
    for a in atoms.values():
        r, rr = _rms(a["sq"], a["n"]), _rms(a["ref"], a["n"])
        per_atom.append({k: a[k] for k in ("gmx_id", "name", "type", "resname", "resid",
                                           "element", "is_bound")}
                        | {"rmse": round(r, 6), "ref_rms": round(rr, 6),
                           "rel_rmse": round(r / rr, 6) if rr else None})
    per_atom.sort(key=lambda d: -d["rmse"])
    ranked = sorted(a["rmse"] for a in per_atom)
    median = ranked[len(ranked) // 2]
    for a in per_atom:
        a["outlier"] = a["rmse"] > FORCE_OUTLIER_FACTOR * median
    return {
        "n_frames": len(frames), "n_atoms": len(atoms), "n_samples": n,
        "rmse": round(rmse, 6), "ref_rms": round(ref_rms, 6),
        "rel_rmse": round(rmse / ref_rms, 6) if ref_rms else None,
        "median_atom_rmse": round(median, 6),
        "units": FORCE_UNITS,
        "by_type": by_type,
        "outliers": [a for a in per_atom if a["outlier"]],
        "worst_atoms": per_atom[:n_atoms_listed],
    }


def render_fminput(
    values: dict[str, object],
    origin: dict[str, str] | None = None,
    header: list[str] | None = None,
) -> str:
    """Render an ``fminput.dat``, naming the source of every value in a comment.

    The same contract as ``render_mdp``: the comment is for the reader, the sidecar
    is for tools. mimicpy reads ``key: value`` lines and ignores ``#`` comments, which
    is what makes a fit recipe something an input set can carry: a campaign's arms
    differ by one key of this file, and the comment records which.
    """
    lines = [f"# {line}" for line in (header or [])]
    if lines:
        lines.append("")
    for key, value in values.items():
        if isinstance(value, (list, tuple)):
            rendered = f"{key}: [{','.join(str(v) for v in value)}]"
        elif isinstance(value, bool):
            rendered = f"{key}: {str(value).lower()}"
        else:
            rendered = f"{key}: {value}"
        source = (origin or {}).get(key)
        lines.append(f"{rendered}  # {source}" if source else rendered)
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_mimicpy(ctx: ToolContext, args: list[str], cwd: Path, timeout: int = 1800):
    argv = [mimicpy_bin(ctx), *args]
    return argv, subprocess.run(  # noqa: S603
        argv, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def _gmx_data_flags(ctx: ToolContext) -> list[str]:
    info = ctx.engines.get("gromacs")
    if info is not None and info.data_dir:
        return ["-ff", info.data_dir]
    return []


# -- fm_collect ---------------------------------------------------------------


class FMDatasetObservation(Observation):
    frames: int | None = None
    qm_atoms: int | None = None
    trr_frames: int | None = None
    index_aligned: bool | None = None
    remapped: int | None = None
    dropped: int | None = None
    fmdata_sha256: str | None = None
    trr_sha256: str | None = None


class FMCollectParams(BaseModel):
    fmdata: str = Field(description="QM reference forces: FMTRAJECTORY .h5 (or .json, converted)")
    trajectory: str = Field(description="matching GROMACS .trr trajectory")
    topology: str = Field(description="GROMACS topology (.top)")
    coords: str = Field(description="coordinate file (.gro)")
    selection: str = Field(description="QM selection file (sele.dat)")
    check_frames: int = Field(default=20, ge=1, description="frames to spot-check for alignment")
    align_if_desynced: bool = Field(
        default=True,
        description="when the spot-check finds a trr<->h5 desync, write an index-aligned "
        "pair (fmcheck -o) and register THAT as the dataset — mimicpy fm refuses "
        "desynced inputs",
    )
    stage: str = Field(default="fm")


@tool(
    "fm_collect",
    description="Register and validate a force-matching reference dataset: convert "
    ".json to .h5 if needed, count frames (h5info), spot-check trr<->h5 frame "
    "alignment (fmcheck), and record sha256 of both files so a refit after "
    "trajectory extension is a new, provenance-linked dataset.",
    returns=FMDatasetObservation,
    tags={"mimic", "fm"},
    requires={"mimicpy"},
)
def fm_collect(ctx: ToolContext, params: FMCollectParams) -> FMDatasetObservation:
    stage = ctx.workspace.stage(params.stage)
    fmdata = resolve_in_ws(ctx, params.fmdata)
    trr = resolve_in_ws(ctx, params.trajectory)
    top = resolve_in_ws(ctx, params.topology)
    coords = resolve_in_ws(ctx, params.coords)
    sele = resolve_in_ws(ctx, params.selection)

    if fmdata.suffix == ".json":
        h5 = stage / (fmdata.stem + ".h5")
        argv, proc = _run_mimicpy(
            ctx, ["json2h5", "-i", str(fmdata), "-o", str(h5)], cwd=stage
        )
        if proc.returncode != 0 or not h5.exists():
            return FMDatasetObservation(
                ok=False,
                summary=f"json2h5 failed (exit {proc.returncode})",
                command=argv,
                log_tail=tail(proc.stderr + "\n" + proc.stdout, 30),
            )
        fmdata = h5

    argv, proc = _run_mimicpy(ctx, ["h5info", "-i", str(fmdata)], cwd=stage)
    info = parse_h5info(proc.stdout + proc.stderr)
    if proc.returncode != 0 or "frames" not in info:
        return FMDatasetObservation(
            ok=False,
            summary=f"h5info failed (exit {proc.returncode}) — is {params.fmdata} an "
            "FMTRAJECTORY dataset?",
            command=argv,
            log_tail=tail(proc.stderr + "\n" + proc.stdout, 30),
        )

    # spot-check ACROSS the whole dataset (strided), then the tail: a merged set
    # goes wrong at its seams and its end, never in its first 20 frames
    n_frames = info["frames"]
    stride = max(1, n_frames // max(1, params.check_frames))
    tail_start = max(0, n_frames - max(3, params.check_frames // 4))
    combined_out = ""
    check = None
    for label, extra in (
        ("strided", ["-b", "0", "-e", str(n_frames), "-s", str(stride)]),
        ("tail", ["-b", str(tail_start), "-e", str(n_frames)]),
    ):
        argv, proc = _run_mimicpy(
            ctx,
            ["fmcheck", "-top", str(top), "-coords", str(coords), "-sele", str(sele),
             "-fmdata", str(fmdata), "-trr", str(trr), *extra, *_gmx_data_flags(ctx)],
            cwd=stage,
        )
        combined_out += (
            f"### fmcheck {label}: {' '.join(extra)}\n" + proc.stdout + "\n" + proc.stderr
        )
        part = parse_fmcheck_report(proc.stdout + proc.stderr)
        if part and (check is None or not part.get("index_aligned", True)):
            check = part
    check_log = stage / "fmcheck.log"
    check_log.write_text(combined_out)
    warnings: list[ValidationWarning] = []
    if check and not check.get("index_aligned", True):
        warnings.append(
            ValidationWarning(
                id="FM_TRR_H5_DESYNC",
                severity="warn",
                message=f"trr and h5 are not index-aligned (status "
                f"{check.get('status')}, {check.get('remapped', 0)} remapped, "
                f"{check.get('dropped', 0)} dropped; checked every {stride}th frame and "
                f"the last {n_frames - tail_start})",
                assumption="mimicpy fm re-pairs frames by coordinates at fit time",
                suggestion="write an aligned pair with `mimicpy fmcheck -o <prefix>` "
                "and collect that instead",
            )
        )
    if check.get("dropped", 0):
        warnings.append(
            ValidationWarning(
                id="FM_FRAMES_DROPPED",
                severity="warn",
                message=f"{check['dropped']} trr frame(s) have no matching h5 "
                "configuration in the checked range",
            )
        )

    frames = info["frames"]
    if not check.get("index_aligned", True) and params.align_if_desynced:
        prefix = stage / "aligned"
        argv, proc = _run_mimicpy(
            ctx,
            ["fmcheck", "-top", str(top), "-coords", str(coords), "-sele", str(sele),
             "-fmdata", str(fmdata), "-trr", str(trr), "-b", "0", "-e", str(frames),
             "-o", str(prefix), *_gmx_data_flags(ctx)],
            cwd=stage,
            timeout=3600,
        )
        aligned_h5 = prefix.with_suffix(".h5")
        aligned_trr = prefix.with_suffix(".trr")
        if proc.returncode != 0 or not (aligned_h5.exists() and aligned_trr.exists()):
            return FMDatasetObservation(
                ok=False,
                summary=f"writing the aligned trr/h5 pair failed (exit {proc.returncode})",
                command=argv,
                warnings=warnings,
                log_tail=tail(proc.stderr + "\n" + proc.stdout, 30),
            )
        with check_log.open("a") as fh:
            fh.write("\n===== aligning run =====\n" + proc.stdout + "\n" + proc.stderr)
        full = parse_fmcheck_report(proc.stdout + proc.stderr)
        fmdata, trr = aligned_h5, aligned_trr
        frames = full.get("matched", frames)
        check["dropped"] = full.get("dropped", check.get("dropped"))
        warnings.append(
            ValidationWarning(
                id="FM_ALIGNED_PAIR_WRITTEN",
                severity="info",
                message=f"index-aligned pair written ({frames} matched frame(s), "
                f"{full.get('dropped', 0)} dropped) — this pair, not the originals, "
                "is the registered dataset",
            )
        )

    fm_hash = _sha256(fmdata)
    trr_hash = _sha256(trr)
    return FMDatasetObservation(
        ok=True,
        summary=f"FM dataset registered: {frames} configurations "
        f"({info.get('qm_atoms', '?')} QM atoms), trr {check.get('trr_frames', '?')} "
        f"frames, alignment {check.get('status', 'unchecked')} "
        f"({'index-aligned' if check.get('index_aligned') else 'REMAPPED — see warning'}); "
        f"h5 sha256 {fm_hash[:12]}…, trr sha256 {trr_hash[:12]}…",
        command=argv,
        warnings=warnings,
        artifacts=[
            ArtifactRef.from_path(fmdata, "fm_dataset", root=ctx.workspace.root),
            ArtifactRef.from_path(trr, "trajectory", root=ctx.workspace.root),
            ArtifactRef.from_path(check_log, "fmcheck_log", root=ctx.workspace.root),
        ],
        frames=frames,
        qm_atoms=info.get("qm_atoms"),
        trr_frames=check.get("trr_frames"),
        index_aligned=check.get("index_aligned"),
        remapped=check.get("remapped"),
        dropped=check.get("dropped"),
        fmdata_sha256=fm_hash,
        trr_sha256=trr_hash,
    )


# -- fm_plan ------------------------------------------------------------------

_DEFAULT_WEIGHTS = {"wv": 1.0, "we": 0.01, "wh": 0.01, "wq": 100000.0}


class FMPlanObservation(Observation):
    n_snapshots: int | None = None
    stride: list[int] | None = None
    weights: dict[str, float] | None = None
    qm_total_charge: int | None = None


class FMPlanParams(BaseModel):
    stride: list[int] = Field(
        min_length=3, max_length=3,
        description="[first, last(exclusive), step] snapshot indices to fit — REQUIRED",
    )
    qm_total_charge: int = Field(
        description="net charge of the QM region (REQUIRED — confirm against mimic_prepare)"
    )
    wv: float | None = Field(default=None, description="DRESP potential weight")
    we: float | None = Field(default=None, description="DRESP field weight")
    wh: float | None = Field(default=None, description="DRESP hirshfeld weight")
    wq: float | None = Field(default=None, description="DRESP total-charge weight")
    eq_atoms: str = Field(default="use_atomtypes")
    reference_charges: str = Field(default="ff_charges")
    optimization_method: str = Field(default="simultaneous")
    extra: dict[str, str] = Field(
        default_factory=dict, description="extra fminput keys, passed through verbatim"
    )
    dataset_frames: int = Field(
        description="total configurations in the dataset (from fm_collect) — REQUIRED "
        "so the stride can be checked against reality",
    )
    output: str = Field(default="fminput.dat")
    stage: str = Field(default="fm")


@tool(
    "fm_plan",
    description="Write an explicit force-matching input (fminput.dat). Stride and QM "
    "total charge are REQUIRED; defaulted DRESP weights are surfaced as assumption "
    "warnings; the stride is validated against the dataset size (from fm_collect).",
    returns=FMPlanObservation,
    tags={"mimic", "fm"},
    requires={"mimicpy"},
)
def fm_plan(ctx: ToolContext, params: FMPlanParams) -> FMPlanObservation:
    first, last, step = params.stride
    if not (0 <= first < last and step >= 1):
        return FMPlanObservation(
            ok=False,
            summary=f"invalid stride {params.stride}: need 0 <= first < last and step >= 1",
        )
    if last > params.dataset_frames:
        return FMPlanObservation(
            ok=False,
            summary=f"stride end {last} exceeds the dataset ({params.dataset_frames} "
            "configurations) — re-collect or shrink the stride",
        )
    n_snap = len(range(first, last, step))
    warnings: list[ValidationWarning] = []
    weights: dict[str, float] = {}
    for key in ("wv", "we", "wh", "wq"):
        value = getattr(params, key)
        if value is None:
            weights[key] = _DEFAULT_WEIGHTS[key]
            warnings.append(
                ValidationWarning(
                    id=f"FM_WEIGHT_DEFAULT_{key.upper()}",
                    severity="info",
                    message=f"{key} not given — using {weights[key]}",
                    assumption=f"{key}={weights[key]} (a weight fits of a small solute "
                    "in water use; nothing derived it for this system)",
                )
            )
        else:
            weights[key] = value
    if n_snap < 100:
        warnings.append(
            ValidationWarning(
                id="FM_FEW_SNAPSHOTS",
                severity="warn",
                message=f"only {n_snap} snapshot(s) selected — a fit with fewer than 100 "
                "overfits and validates poorly",
                suggestion="widen the stride or extend the reference trajectory",
            )
        )

    stage = ctx.workspace.stage(params.stage)
    lines = [
        "# Force matching input (written by myrmidon fm_plan)",
        f"stride: [{first},{last},{step}]",
    ]
    lines += [f"{k}: {v:g}" for k, v in weights.items()]
    lines += [
        f"eq_atoms: {params.eq_atoms}",
        f"qm_total_charge: {params.qm_total_charge}",
        f"reference_charges: {params.reference_charges}",
        f"optimization_method: {params.optimization_method}",
    ]
    lines += [f"{k}: {v}" for k, v in params.extra.items()]
    out = stage_output(stage, params.output)
    out.write_text("\n".join(lines) + "\n")
    return FMPlanObservation(
        ok=True,
        summary=f"FM plan written: {n_snap} snapshots (stride {params.stride}), "
        f"qm_total_charge {params.qm_total_charge}, method "
        f"{params.optimization_method} -> {ctx.workspace.relpath(out)}",
        warnings=warnings,
        artifacts=[ArtifactRef.from_path(out, "fm_input", root=ctx.workspace.root)],
        n_snapshots=n_snap,
        stride=list(params.stride),
        weights=weights,
        qm_total_charge=params.qm_total_charge,
    )


# -- fm_fit -------------------------------------------------------------------


class FMFitParams(BaseModel):
    #: the fields this submission's cost is computed from; a plan step must state them
    RESOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("n_processes",)

    topology: str
    coords: str
    fmdata: str = Field(description="FMTRAJECTORY .h5 (from fm_collect)")
    trajectory: str
    index: str = Field(description="index file with the QMatoms group (.ndx)")
    selection: str
    fm_input: str = Field(description="fminput.dat from fm_plan")
    mdp: str | None = Field(default=None, description="mdp for the classical rerun")
    mode: Literal["fm", "dresp", "opt_ff"] = Field(default="fm")
    n_processes: int = Field(default=12, ge=1)
    account: str | None = Field(
        default=None,
        description=
        "scheduler account to bill for THIS submission instead of the machine config's "
        "default — the user may hold several project allocations",
    )
    stage: str = Field(default="fm")
    acknowledge_warnings: list[str] = Field(default_factory=list)

    def resources(self) -> Resources:
        """What the fit books: one task with ``n_processes`` cores, for the scheduler's
        default walltime — the request it submits, so a plan prices it the same way."""
        return Resources(ntasks=1, cpus_per_task=self.n_processes, account=self.account)


_SECTION_RE = re.compile(r"^\s*\[\s*(\w+)\s*\]", re.MULTILINE)


def itp_atomtypes_warnings(topology: Path) -> list[ValidationWarning]:
    """An included .itp that defines both [ atomtypes ] and a [ moleculetype ].

    mimicpy rewrites the QM molecule's .itp for the fit (ref_*.itp, opt_*.itp) with only
    the moleculetype and its sections, so atom types defined in that same file vanish and
    the classical rerun's grompp fails with "Atomtype ... not found" — after the charge
    stage has already run. Single-file molecule topologies from small-molecule generators
    are laid out exactly this way.
    """
    found = []
    for line in topology.read_text(errors="replace").splitlines():
        m = re.match(r'\s*#include\s+"([^"]+\.itp)"', line)
        if not m:
            continue
        itp = topology.parent / m.group(1)
        if not itp.is_file():
            continue
        sections = {name.lower() for name in _SECTION_RE.findall(itp.read_text(errors="replace"))}
        if {"atomtypes", "moleculetype"} <= sections:
            found.append(itp.name)
    if not found:
        return []
    return [ValidationWarning(
        id="FM_ITP_DEFINES_ATOMTYPES",
        severity="warn",
        message=f"{', '.join(found)} define(s) [ atomtypes ] beside a [ moleculetype ]. If "
        "that molecule is in the QM region, mimicpy rewrites its .itp without the atom "
        "types and the classical rerun fails ('Atomtype ... not found') after the charge "
        "stage.",
        suggestion="move [ atomtypes ] into the .top before the #include, or into a "
        "separate include; acknowledge if the molecule is not in the QM region",
    )]


def fm_gate(ctx: ToolContext, params: FMFitParams):
    warnings: list[ValidationWarning] = []
    checked = ["input_files", "fm_input_keys"]
    inputs = [
        ("topology", params.topology), ("coords", params.coords),
        ("fmdata", params.fmdata), ("trajectory", params.trajectory),
        ("index", params.index), ("selection", params.selection),
        ("fm_input", params.fm_input),
    ]
    if params.mdp:
        inputs.append(("mdp", params.mdp))
    missing = False
    for label, path in inputs:
        _, absent = require_input(ctx, label, path)
        if absent:
            missing = True
            warnings.append(absent)
    if not missing and params.mdp:
        from myrmidon.engines.gromacs.mdp import parse_mdp_file

        mdp = parse_mdp_file(resolve_in_ws(ctx, params.mdp))
        checked.append("rerun_mdp_classical")
        warnings.extend(rerun_mdp_warnings(mdp))
    if not missing:
        checked.append("itp_atomtypes")
        warnings.extend(itp_atomtypes_warnings(resolve_in_ws(ctx, params.topology)))
        text = resolve_in_ws(ctx, params.fm_input).read_text()
        keys = {
            ln.split(":", 1)[0].strip()
            for ln in text.splitlines()
            if ":" in ln and not ln.lstrip().startswith(("#", "!"))
        }
        for required in ("stride", "qm_total_charge"):
            if required not in keys:
                warnings.append(
                    ValidationWarning(
                        id="FM_INPUT_INCOMPLETE",
                        severity="block",
                        message=f"fminput has no '{required}' — write it with fm_plan "
                        "(explicit, never assumed)",
                    )
                )
    return warnings, checked


@tool(
    "fm_fit",
    description="Run the MiMiCPy force-matching fit (DRESP charges + bonded/LJ "
    "parameter optimization against QM/MM reference forces). Validation-gated and "
    "async — poll with job_status, which returns the parsed fit once it has ended.",
    returns=Observation,
    tags={"mimic", "fm", "run"},
    requires={"mimicpy", "gromacs"},
    expensive=True,
    gate=fm_gate,
)
def fm_fit(ctx: ToolContext, params: FMFitParams) -> Observation:
    from myrmidon.engines.gromacs.common import gmx_bin

    stage = ctx.workspace.stage(params.stage)
    top = _stage_topology(stage, resolve_in_ws(ctx, params.topology))
    argv = [
        mimicpy_bin(ctx), "fm",
        "-top", str(top),
        "-coords", str(resolve_in_ws(ctx, params.coords)),
        "-fmdata", str(resolve_in_ws(ctx, params.fmdata)),
        "-trr", str(resolve_in_ws(ctx, params.trajectory)),
        "-ndx", str(resolve_in_ws(ctx, params.index)),
        "-sele", str(resolve_in_ws(ctx, params.selection)),
        "-fi", str(resolve_in_ws(ctx, params.fm_input)),
        "-gmx", gmx_bin(ctx),
        "-n_processes", str(params.n_processes),
        *_gmx_data_flags(ctx),
    ]
    if params.mdp:
        argv += ["-mdp", str(resolve_in_ws(ctx, params.mdp))]
    if params.mode != "fm":
        argv.append(f"-{params.mode}")
    spec = JobSpec(
        argv=argv,
        cwd=stage,
        stdout="fm.stdout",
        stderr="fm.stderr",
        resources=params.resources(),
    )
    handle = ctx.backend.submit(spec)
    job_id = get_jobstore(ctx).add(
        ctx.backend.name, handle, spec, label=f"fm:{params.mode}",
        meta={"kind": "fm_fit", "stage": params.stage, "topology": params.topology},
    )
    return JobObservation(
        ok=True,
        summary=f"force-matching fit submitted as {job_id} (mode {params.mode}, "
        f"{params.n_processes} processes) in {ctx.workspace.relpath(stage)}. "
        "Poll with job_status: once the fit has ended it returns the parsed fit, the "
        "fitted .itp(s) and the candidate topology assembled from them.",
        command=argv,
        job_id=job_id,
        state="QUEUED",
    )


# -- fm_results ---------------------------------------------------------------


class FMObservation(Observation):
    job_id: str = ""
    state: str = ""
    exit_code: int | None = None
    core_h_estimate: float | None = None
    n_snapshots: int | None = None
    n_train: int | None = None
    n_val: int | None = None
    validation_sdf: float | None = None
    validation_cost: float | None = None
    optimizer_success: bool | None = None
    optimizer_message: str | None = None
    qm_total_charge: float | None = None
    wall_time_s: float | None = None
    max_abs_charge_delta: float | None = None
    mean_abs_charge_delta: float | None = None
    n_charges: int | None = None
    n_unphysical_terms: int = 0  # fitted bond/angle terms with k <= 0
    # held-out force error, from mimicpy's validation_force_residuals.csv (Hartree/Bohr)
    force_rmse: float | None = None
    force_ref_rms: float | None = None
    force_rel_rmse: float | None = None
    force_rmse_by_type: list[dict] = Field(default_factory=list)
    force_outliers: list[dict] = Field(default_factory=list)
    n_force_frames: int | None = None


def unphysical_force_constants(itp_text: str) -> list[str]:
    """Bond/angle rows whose force constant is not positive — what an under-determined
    fit produces (a fit on a handful of snapshots can return k of order -1e6).
    A minimizer on such a topology walks the molecule apart."""
    return [f"{section} {' '.join(idx)}: k = {k:g}"
            for section, idx, _eq, k in bond_angle_rows(itp_text) if k <= 0]


#: A fitted parameter this far from physical sense is a fit artefact, not chemistry.
#: Equilibrium angles first: one collapsed toward zero or opened to a straight line is
#: the signature of terms the reference data never constrained.
ANGLE_MIN_DEG, ANGLE_MAX_DEG = 1.0, 179.0
BOND_MIN_NM, BOND_MAX_NM = 0.05, 0.35      # real covalent bonds, in nm
FORCE_CONSTANT_FACTOR = 5.0                 # beyond this a constant was replaced, not refined
#: beyond this a *fitted* charge is not a partial charge. A formal charge is a different
#: thing: a divalent metal sits at +2 legitimately and no fit puts it there, so a large
#: charge counts against a fit only when the fit is what moved it.
CHARGE_ABS_MAX = 2.0
CHARGE_MOVED_E = 0.05

#: what each finding means, said once so the wording is the same wherever it is raised
PARAMETER_FINDINGS = {
    "FM_UNPHYSICAL_PARAMS":
        "bond/angle term(s) with a non-positive force constant — a minimiser on this "
        "topology walks the molecule apart",
    "FM_ANGLE_COLLAPSED":
        "equilibrium angle(s) beyond the limits of geometry — an angle near 0 or 180 "
        "degrees is what an under-determined fit produces, and the validation scores do "
        "not catch it",
    "FM_BOND_LENGTH_IMPLAUSIBLE":
        "equilibrium bond length(s) outside the range of real covalent bonds",
    "FM_CHARGE_IMPLAUSIBLE":
        "partial charge(s) larger than any atom carries",
    "FM_FORCE_CONSTANT_IMPLAUSIBLE":
        "force constant(s) that moved more than fivefold — replaced rather than refined",
    "FM_NET_CHARGE_DRIFTED":
        "a net molecular charge different from what it started as, which no fit should "
        "change",
}


def implausible_parameters(itp_text: str, original_text: str | None = None) -> dict:
    """Fitted parameters that are not chemistry, grouped by the finding they raise.

    ``unphysical_force_constants`` reads the force-constant column alone, so an
    equilibrium angle collapsed to a fraction of a degree beside a perfectly positive
    constant passes it. A few snapshots a few hundredths of a picosecond apart do not
    independently constrain three angle terms around one atom, so the optimiser puts them
    where the objective is smallest — and every validation score still reads as acceptable.

    ``original_text`` is the topology the fit started from. Without it the comparisons
    that need a baseline are skipped rather than guessed.
    """
    found: dict[str, list[str]] = {}

    def note(finding: str, detail: str) -> None:
        found.setdefault(finding, []).append(detail)

    for section, idx, equilibrium, constant in bond_angle_rows(itp_text):
        where = f"{section} {' '.join(idx)}"
        if constant <= 0:
            note("FM_UNPHYSICAL_PARAMS", f"{where}: k = {constant:g}")
        if section == "angles" and not ANGLE_MIN_DEG <= equilibrium <= ANGLE_MAX_DEG:
            note("FM_ANGLE_COLLAPSED", f"{where}: theta0 = {equilibrium:g} deg")
        if section == "bonds" and not BOND_MIN_NM <= equilibrium <= BOND_MAX_NM:
            note("FM_BOND_LENGTH_IMPLAUSIBLE", f"{where}: b0 = {equilibrium:g} nm")

    was_charge = dict(atom_charges(original_text)) if original_text is not None else {}
    for name, charge in atom_charges(itp_text):
        if abs(charge) <= CHARGE_ABS_MAX:
            continue
        before = was_charge.get(name)
        if before is not None and abs(charge - before) <= CHARGE_MOVED_E:
            continue  # it started there: a formal charge, not something the fit invented
        moved = "" if before is None else f" (was {before:g})"
        note("FM_CHARGE_IMPLAUSIBLE", f"{name}: q = {charge:g} e{moved}")

    if original_text is not None:
        before = {(sec, idx): k for sec, idx, _eq, k in bond_angle_rows(original_text)}
        for section, idx, _equilibrium, constant in bond_angle_rows(itp_text):
            was = before.get((section, idx))
            if not was or was <= 0 or constant <= 0:
                continue
            factor = max(constant / was, was / constant)
            if factor > FORCE_CONSTANT_FACTOR:
                note("FM_FORCE_CONSTANT_IMPLAUSIBLE",
                     f"{section} {' '.join(idx)}: k {was:g} -> {constant:g} ({factor:.0f}x)")
        net_before = sum(q for _, q in atom_charges(original_text))
        net_after = sum(q for _, q in atom_charges(itp_text))
        if abs(net_after - net_before) > 0.01:
            note("FM_NET_CHARGE_DRIFTED",
                 f"net charge {net_before:+.3f} -> {net_after:+.3f} e")
    return found


def _includes_of(top: Path) -> list[str]:
    return re.findall(r'^\s*#include\s+"([^"]+)"', top.read_text(errors="replace"), re.MULTILINE)


def _fitted_pairs(top: Path, opt_itps: list[Path]) -> dict[str, Path]:
    """include name -> fitted itp, for every opt_<name>.itp whose original the topology
    includes (mimicpy names a fitted file after the one it read)."""
    pairs: dict[str, Path] = {}
    for fitted in opt_itps:
        name = fitted.name[len("opt_"):]
        inc = next((i for i in _includes_of(top) if Path(i).name == name), None)
        if inc:
            pairs[inc] = fitted
    return pairs


def collect_fit(ctx: ToolContext, stage_dir: Path, original_itp: str | None = None,
                topology: str | None = None, *, completed: bool = True) -> FMObservation:
    """A force-matching fit that has ended, read into typed facts: snapshot counts,
    train/validation split, validation SDF/cost, optimizer status, wall time, the fitted
    charge deltas against the original .itp, and — given the topology the fit started
    from — the candidate topology with the fitted include(s) swapped in. This is what
    ``job_status`` returns for an ``fm_fit`` job; ``original_itp`` is inferred from the
    topology's includes when not given."""
    log_path = stage_dir / "fm.log"
    if not log_path.exists():
        return FMObservation(
            ok=False,
            summary=f"no fm.log at {ctx.workspace.relpath(log_path)} — has the fit run?",
        )
    facts = parse_fm_log(log_path.read_text(errors="replace"))
    results_file = log_path.parent / "optimization_results.txt"
    if results_file.exists():
        for key, value in parse_optimization_results(results_file.read_text()).items():
            facts.setdefault(key, value)

    artifacts = [ArtifactRef.from_path(log_path, "fm_log", root=ctx.workspace.root)]
    max_dq = mean_dq = n_q = None
    opt_itps: list[Path] = []
    for name in facts.get("itp_files", []):
        candidate = log_path.parent / name
        if candidate.exists() and name.startswith("opt_"):
            opt_itps.append(candidate)
            artifacts.append(
                ArtifactRef.from_path(candidate, "fitted_topology", root=ctx.workspace.root)
            )
    top = resolve_in_ws(ctx, topology) if topology else None
    if original_itp is None and top is not None and opt_itps:
        for inc in _fitted_pairs(top, opt_itps):
            candidate_orig = top.parent / inc
            if candidate_orig.exists():
                original_itp = ctx.workspace.relpath(candidate_orig)
                break
    opt_itp = None
    if original_itp and opt_itps:
        # the fitted twin of the original: opt_<name>.itp, else the only one
        original = resolve_in_ws(ctx, original_itp)
        twin = log_path.parent / f"opt_{original.name}"
        opt_itp = twin if twin in opt_itps else (opt_itps[0] if len(opt_itps) == 1 else None)
    if original_itp and opt_itp is not None:
        before = dict(atom_charges(resolve_in_ws(ctx, original_itp).read_text()))
        after = dict(atom_charges(opt_itp.read_text()))
        common = sorted(set(before) & set(after))
        if common:
            deltas = [abs(after[a] - before[a]) for a in common]
            max_dq = round(max(deltas), 5)
            mean_dq = round(sum(deltas) / len(deltas), 5)
            n_q = len(common)

    forces: dict = {}
    residuals_csv = log_path.parent / VALIDATION_RESIDUALS_FILENAME
    if residuals_csv.exists():
        forces = parse_validation_force_residuals(residuals_csv.read_text(errors="replace"))
        if forces:
            artifacts.append(ArtifactRef.from_path(residuals_csv, "force_residuals",
                                                   root=ctx.workspace.root))

    ok = (completed and facts.get("completed", False)
          and facts.get("optimizer_success") is not False)
    warnings: list[ValidationWarning] = []
    if forces.get("outliers"):
        worst = forces["outliers"][:5]
        warnings.append(ValidationWarning(
            id="FM_FORCE_OUTLIER_ATOMS", severity="info",
            message=f"{len(forces['outliers'])} QM atom(s) have a held-out force RMSE above "
            f"{FORCE_OUTLIER_FACTOR:g}× the median atom's ({forces['median_atom_rmse']} "
            f"{FORCE_UNITS}): "
            + ", ".join(f"{a['resname']}{a['resid']}:{a['name']} ({a['type']}) {a['rmse']}"
                        for a in worst)
            + (" ..." if len(forces["outliers"]) > 5 else "")
            + ". Look at these atoms' bonded terms and the QM/MM boundary before trusting "
            "the candidate.",
        ))
    # Every candidate, read against the topology it was fitted from where that can be
    # found, so a parameter that was replaced rather than refined is visible.
    pairs = _fitted_pairs(top, opt_itps) if top is not None else {}
    by_finding: dict[str, list[str]] = {}
    for itp in opt_itps:
        source = next((top.parent / inc for inc, fitted in pairs.items()
                       if Path(fitted) == itp and (top.parent / inc).is_file()), None)
        rows_by_finding = implausible_parameters(
            itp.read_text(errors="replace"),
            source.read_text(errors="replace") if source is not None else None,
        )
        for finding, rows in rows_by_finding.items():
            by_finding.setdefault(finding, []).extend(f"{itp.name}: {r}" for r in rows)
    unphysical = by_finding.get("FM_UNPHYSICAL_PARAMS", [])
    for finding, rows in sorted(by_finding.items()):
        shown = "; ".join(rows[:3]) + (" ..." if len(rows) > 3 else "")
        warnings.append(ValidationWarning(
            id=finding, severity="warn",
            message=f"{len(rows)} fitted {PARAMETER_FINDINGS[finding]} ({shown})",
            suggestion="do not use this candidate as it stands: add reference snapshots, "
            "or restrict what the fit may change (charges only). Read the parameter "
            "tables in fm_report — validation scores can look acceptable while the "
            "parameters are not.",
        ))
    bits = []
    if unphysical:
        bits.append(f"{len(unphysical)} UNPHYSICAL bonded term(s)")
    if facts.get("n_snapshots") is not None:
        bits.append(f"{facts['n_snapshots']} snapshots")
    if facts.get("n_train") is not None:
        bits.append(f"{facts['n_train']}/{facts['n_val']} train/val")
    if facts.get("validation_sdf") is not None:
        bits.append(f"validation SDF {facts['validation_sdf']}")
    if opt_itps:
        bits.append(f"fitted itp(s): {', '.join(p.name for p in opt_itps)}")
    if max_dq is not None:
        bits.append(f"max |dq| {max_dq}")
    if forces:
        worst_type = forces["by_type"][0]
        bits.append(
            f"held-out bonded-force RMSE {forces['rmse']} {FORCE_UNITS} over "
            f"{forces['n_frames']} frames "
            f"(rel {forces['rel_rmse']}; worst type {worst_type['type']} {worst_type['rmse']})"
        )
    summary = (
        ("FM fit completed: " if ok else "FM fit NOT clean: ")
        + (", ".join(bits) if bits else "no facts parsed from fm.log")
        + (
            f". Optimizer: {facts.get('optimizer_message')}"
            if facts.get("optimizer_message")
            else ""
        )
    )
    obs = FMObservation(
        ok=ok,
        summary=summary,
        artifacts=artifacts,
        warnings=warnings,
        n_unphysical_terms=len(unphysical),
        log_tail=None if ok else tail(log_path.read_text(errors="replace"), 25),
        n_snapshots=facts.get("n_snapshots"),
        n_train=facts.get("n_train"),
        n_val=facts.get("n_val"),
        validation_sdf=facts.get("validation_sdf"),
        validation_cost=facts.get("validation_cost"),
        optimizer_success=facts.get("optimizer_success"),
        optimizer_message=facts.get("optimizer_message"),
        qm_total_charge=facts.get("qm_total_charge"),
        wall_time_s=facts.get("wall_time_s"),
        max_abs_charge_delta=max_dq,
        mean_abs_charge_delta=mean_dq,
        n_charges=n_q,
        force_rmse=forces.get("rmse"),
        force_ref_rms=forces.get("ref_rms"),
        force_rel_rmse=forces.get("rel_rmse"),
        force_rmse_by_type=forces.get("by_type", []),
        force_outliers=forces.get("outliers", []),
        n_force_frames=forces.get("n_frames"),
    )
    if original_itp:
        orig_path = resolve_in_ws(ctx, original_itp)
        if orig_path.exists():
            obs.inputs.append(ArtifactRef.from_path(orig_path, "original_topology",
                                                    root=ctx.workspace.root))
    # the candidate topology: the fit's own topology with every fitted include swapped in
    if ok and top is not None and opt_itps:
        pairs = _fitted_pairs(top, opt_itps)
        if pairs:
            cand = assemble_candidate(ctx, stage_dir, top, pairs)
            if cand.ok:
                known = {a.path for a in obs.artifacts}
                obs.artifacts += [a for a in cand.artifacts if a.path not in known]
                obs.summary += f". Candidate topology: {cand.summary}"
            else:
                obs.summary += f". No candidate topology: {cand.summary}"
        else:
            obs.summary += (". No candidate topology: none of the fitted itp(s) is named "
                            "after an include of the topology")
    return obs



# -- fm_frame_report ----------------------------------------------------------

_BOHR_TO_A = 0.529177210903


class FMFrameReport(Observation):
    n_frames: int | None = None
    monitor_stats: dict[str, dict[str, float]] | None = None  # label -> min/mean/max/std (Å)
    monitor_missing: dict[str, int] | None = None  # label -> frames where an atom left the shell
    force_outlier_frames: list[int] | None = None  # |F|max > mean + 4·std
    n_near_duplicates: int | None = None  # consecutive frames with QM RMSD < threshold
    duplicate_rmsd_threshold_a: float | None = None
    md_steps_from_logs: int | None = None
    temp_mean_k: float | None = None
    temp_final_k: float | None = None
    scf_iters_mean: float | None = None
    scf_iters_max: int | None = None
    segments: list[dict] | None = None


class FMFrameReportParams(BaseModel):
    fmdata: str = Field(description="FMTRAJECTORY .h5 (from fm_collect)")
    monitor_pairs: dict[str, list[int]] = Field(
        default_factory=dict,
        description="named atom pairs to track per frame, by GLOBAL atom id as stored "
        "in the dataset (e.g. {'ion-O1': [12813, 12850]}); distances in Å. A reprocessed "
        "or aligned dataset stores its QM atoms in mimicpy's own record order, which is "
        "neither the CPMD &ATOMS order nor gro serial order: a first-shell pair reading "
        "8-13 Å means the ids name the wrong atoms. Map ids with mimicpy's QMRegion "
        "(the mapping the fit uses) before trusting a monitored distance",
    )
    cpmd_logs: list[str] = Field(
        default_factory=list,
        description="ordered CPMD log files of the generating run(s) — adds per-step "
        "temperature/SCF quality and per-segment health to the report",
    )
    duplicate_rmsd_a: float = Field(
        default=0.05, gt=0, description="QM RMSD (Å) below which consecutive frames "
        "count as near-duplicates",
    )
    overlaps_input: str | None = Field(
        default=None,
        description="MiMiC cpmd.inp whose &MIMIC OVERLAPS maps GROMACS atom ids to "
        "CPMD QM ids — lets monitor_pairs use gro ids for QM atoms. Valid only for a "
        "dataset still in CPMD order, as the generating run wrote it; not for a "
        "reprocessed or aligned one (see monitor_pairs)",
    )
    scf_flag_threshold: int = Field(
        default=100, ge=1,
        description="flag a segment when its worst SCF step needs at least this many "
        "iterations (set to the run's MAXITER)",
    )
    stage: str = Field(default="fm")
    output: str = Field(default="frame_report.csv")


@tool(
    "fm_frame_report",
    description="Characterize a force-matching reference dataset frame by frame: "
    "monitored atom-pair distances (e.g. metal-donor), QM force magnitudes and "
    "outlier frames, consecutive-frame near-duplicates, and — given the generating "
    "CPMD logs — per-step temperature and SCF-convergence health per segment. "
    "Writes a per-frame CSV; every number is computed from the dataset/logs.",
    returns=FMFrameReport,
    tags={"mimic", "fm", "analysis"},
    requires={"mimicpy"},
)
def fm_frame_report(ctx: ToolContext, params: FMFrameReportParams) -> FMFrameReport:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        return FMFrameReport(ok=False, summary=f"h5py/numpy required: {exc}")

    from myrmidon.engines.mimic.diagnostics import parse_cpmd_output, parse_cpmd_series

    stage = ctx.workspace.stage(params.stage)
    h5_path = resolve_in_ws(ctx, params.fmdata)

    monitor_pairs = dict(params.monitor_pairs)
    if params.overlaps_input:
        text = resolve_in_ws(ctx, params.overlaps_input).read_text(errors="replace")
        gmx_to_cpmd: dict[int, int] = {}
        in_overlaps = False
        for line in text.splitlines():
            fields = line.split()
            if in_overlaps and len(fields) == 4 and all(f.isdigit() for f in fields):
                gmx_to_cpmd[int(fields[1])] = int(fields[3])
            elif in_overlaps and fields and not fields[0].isdigit():
                in_overlaps = False
            if fields and fields[0].upper() == "OVERLAPS":
                in_overlaps = True
        monitor_pairs = {
            label: [gmx_to_cpmd.get(a, a) for a in pair]
            for label, pair in monitor_pairs.items()
        }

    rows: list[dict] = []
    missing = dict.fromkeys(monitor_pairs, 0)
    prev_qm = None
    n_dup = 0
    with h5py.File(h5_path, "r") as fh:
        keys = sorted(
            (k for k in fh if k.startswith("config_")),
            key=lambda k: int(k.split("_")[1]),
        )
        for frame, key in enumerate(keys):
            atoms = fh[key]["atoms"][()]
            ids = atoms["id"]
            coord = atoms["coordinate"] * _BOHR_TO_A
            qm_mask = atoms["region"] == 1
            forces = atoms["force"][qm_mask]
            fmag = np.linalg.norm(forces, axis=1)
            row: dict = {
                "frame": frame,
                "qm_fmax_au": round(float(fmag.max()), 6),
                "qm_frms_au": round(float(np.sqrt((fmag**2).mean())), 6),
            }
            qm_xyz = coord[qm_mask]
            if prev_qm is not None and len(prev_qm) == len(qm_xyz):
                rmsd = float(np.sqrt(((qm_xyz - prev_qm) ** 2).sum(axis=1).mean()))
                row["qm_rmsd_prev_a"] = round(rmsd, 5)
                if rmsd < params.duplicate_rmsd_a:
                    n_dup += 1
            prev_qm = qm_xyz
            index = {int(a): i for i, a in enumerate(ids)}
            for label, (ia, ib) in monitor_pairs.items():
                if ia in index and ib in index:
                    d = float(np.linalg.norm(coord[index[ia]] - coord[index[ib]]))
                    row[label] = round(d, 4)
                else:
                    missing[label] += 1
            rows.append(row)

    columns = ["frame", "qm_fmax_au", "qm_frms_au", "qm_rmsd_prev_a", *monitor_pairs]
    out = stage_output(stage, params.output)
    with out.open("w") as fh:
        fh.write(",".join(columns) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(c, "")) for c in columns) + "\n")

    fmax = np.array([r["qm_fmax_au"] for r in rows])
    outliers = [
        int(r["frame"]) for r in rows if r["qm_fmax_au"] > fmax.mean() + 4 * fmax.std()
    ]
    monitor_stats: dict[str, dict[str, float]] = {}
    for label in monitor_pairs:
        vals = np.array([r[label] for r in rows if label in r])
        if len(vals):
            monitor_stats[label] = {
                "min": round(float(vals.min()), 3),
                "mean": round(float(vals.mean()), 3),
                "max": round(float(vals.max()), 3),
                "std": round(float(vals.std()), 3),
            }

    segments: list[dict] = []
    all_steps: list[dict] = []
    for log in params.cpmd_logs:
        log_path = resolve_in_ws(ctx, log)
        text = log_path.read_text(errors="replace")
        series = parse_cpmd_series(text)
        facts = parse_cpmd_output(text)
        temps = [st["temp_k"] for st in series]
        scfs = [st["scf_iters"] for st in series if st["scf_iters"]]
        segments.append(
            {
                "log": log_path.name if log_path.name != "cpmd.out" else log_path.parent.name,
                "md_steps": len(series),
                "temp_first_k": temps[0] if temps else None,
                "temp_mean_k": round(sum(temps) / len(temps), 1) if temps else None,
                "scf_mean": round(sum(scfs) / len(scfs), 1) if scfs else None,
                "scf_max": max(scfs) if scfs else None,
                "finished_normally": facts.get("finished_normally", False),
                "diagnosis": facts.get("diagnosis"),
            }
        )
        all_steps.extend(series)

    temps = [st["temp_k"] for st in all_steps]
    scfs = [st["scf_iters"] for st in all_steps if st["scf_iters"]]
    n = len(rows)
    bits = [f"{n} frames"]
    for label, st in monitor_stats.items():
        bits.append(f"{label} {st['min']}-{st['max']} Å (mean {st['mean']})")
    if outliers:
        bits.append(f"{len(outliers)} force-outlier frame(s)")
    if n_dup:
        bits.append(f"{n_dup} near-duplicate consecutive frame(s)")
    if all_steps:
        bits.append(f"logs: {len(all_steps)} MD steps, T {round(sum(temps)/len(temps))} K, "
                    f"SCF mean {round(sum(scfs)/len(scfs), 1) if scfs else '?'}")
    return FMFrameReport(
        ok=True,
        summary="Frame report: " + "; ".join(bits) + f" -> {ctx.workspace.relpath(out)}",
        artifacts=[ArtifactRef.from_path(out, "frame_report", root=ctx.workspace.root)],
        n_frames=n,
        monitor_stats=monitor_stats or None,
        monitor_missing={k: v for k, v in missing.items() if v} or None,
        force_outlier_frames=outliers[:50] or None,
        n_near_duplicates=n_dup,
        duplicate_rmsd_threshold_a=params.duplicate_rmsd_a,
        md_steps_from_logs=len(all_steps) or None,
        temp_mean_k=round(sum(temps) / len(temps), 1) if temps else None,
        temp_final_k=temps[-1] if temps else None,
        scf_iters_mean=round(sum(scfs) / len(scfs), 1) if scfs else None,
        scf_iters_max=max(scfs) if scfs else None,
        segments=segments or None,
    )


# -- fm_merge (combining several runs' reference data) ------------------------


class FMMergeObservation(Observation):
    frames: int | None = None
    n_sources: int | None = None
    merged_sha256: str | None = None


class FMMergeParams(BaseModel):
    fm_jsons: list[str] = Field(min_length=1, description="FMTRAJECTORY.json files, in order")
    trajectories: list[str] = Field(
        min_length=1, description="matching GROMACS .trr files, same order"
    )
    stage: str = Field(default="fm")
    output_prefix: str = Field(default="merged")


@tool(
    "fm_merge",
    description="Merge force-matching reference data from several runs (e.g. scan "
    "windows) into one dataset: FMTRAJECTORY jsons are concatenated (the format is a "
    "stream of config objects), trajectories via gmx trjcat -cat, then json2h5. "
    "Register the result with fm_collect.",
    returns=FMMergeObservation,
    tags={"mimic", "fm"},
    requires={"mimicpy", "gromacs"},
)
def fm_merge(ctx: ToolContext, params: FMMergeParams) -> FMMergeObservation:
    from myrmidon.engines.gromacs.common import run_gmx

    if len(params.fm_jsons) != len(params.trajectories):
        return FMMergeObservation(
            ok=False,
            summary=f"{len(params.fm_jsons)} json(s) vs {len(params.trajectories)} "
            "trajectorie(s) — must match one-to-one",
        )
    stage = ctx.workspace.stage(params.stage)
    sources = [resolve_in_ws(ctx, src) for src in params.fm_jsons]
    merged_json = None
    frames = 0
    if any(src.suffix == ".h5" for src in sources):
        # binary merge: copy config groups renumbered (json sources converted first)
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            return FMMergeObservation(ok=False, summary=f"h5py required: {exc}")
        h5_sources = []
        for i, src in enumerate(sources):
            if src.suffix == ".h5":
                h5_sources.append(src)
            else:
                # one conversion per SOURCE: every window's file is FMTRAJECTORY.json, so
                # a stem-based name would collide and the last window silently win, merging
                # N single points as N copies of the same one
                conv = stage / f"{i:03d}_{src.parent.name}_{src.stem}_conv.h5"
                argv, proc = _run_mimicpy(
                    ctx, ["json2h5", "-i", str(src), "-o", str(conv)], cwd=stage
                )
                if proc.returncode != 0 or not conv.exists():
                    return FMMergeObservation(
                        ok=False, summary=f"json2h5 failed for {src.name}", command=argv,
                        log_tail=tail(proc.stderr + "\n" + proc.stdout, 20),
                    )
                h5_sources.append(conv)
        merged_h5_pre = stage / f"{params.output_prefix}.h5"
        with h5py.File(merged_h5_pre, "w") as out_h5:
            for src in h5_sources:
                with h5py.File(src, "r") as in_h5:
                    keys = sorted(
                        (k for k in in_h5 if k.startswith("config_")),
                        key=lambda k: int(k.split("_")[1]),
                    )
                    for k in keys:
                        in_h5.copy(k, out_h5, name=f"config_{frames}")
                        frames += 1
    else:
        merged_json = stage / f"{params.output_prefix}.json"
        with merged_json.open("w") as out:
            for src in sources:
                text = src.read_text()
                frames += text.count('"time_step"')
                out.write(text)
                if not text.endswith("\n"):
                    out.write("\n")
    merged_trr = stage / f"{params.output_prefix}.trr"
    trrs = [str(resolve_in_ws(ctx, t)) for t in params.trajectories]
    proc = run_gmx(
        ctx, ["trjcat", "-f", *trrs, "-o", str(merged_trr), "-cat"], cwd=stage
    )
    if proc.returncode != 0 or not merged_trr.exists():
        return FMMergeObservation(
            ok=False,
            summary=f"gmx trjcat failed (exit {proc.returncode})",
            command=list(proc.args),
            log_tail=tail(proc.stderr, 30),
        )
    merged_h5 = stage / f"{params.output_prefix}.h5"
    if merged_json is not None:
        argv, conv = _run_mimicpy(
            ctx, ["json2h5", "-i", str(merged_json), "-o", str(merged_h5)], cwd=stage
        )
        if conv.returncode != 0 or not merged_h5.exists():
            return FMMergeObservation(
                ok=False,
                summary=f"json2h5 failed (exit {conv.returncode})",
                command=argv,
                log_tail=tail(conv.stderr + "\n" + conv.stdout, 30),
            )
    return FMMergeObservation(
        ok=True,
        summary=f"merged {len(params.fm_jsons)} dataset(s): {frames} configurations -> "
        f"{ctx.workspace.relpath(merged_h5)} + {ctx.workspace.relpath(merged_trr)}. "
        "Validate with fm_collect.",
        artifacts=[
            ArtifactRef.from_path(merged_h5, "fm_dataset", root=ctx.workspace.root),
            ArtifactRef.from_path(merged_trr, "trajectory", root=ctx.workspace.root),
        ]
        + (
            [ArtifactRef.from_path(merged_json, "fm_dataset_json", root=ctx.workspace.root)]
            if merged_json is not None
            else []
        ),
        frames=frames,
        n_sources=len(params.fm_jsons),
        merged_sha256=_sha256(merged_h5),
    )


# -- fm_displace_plan (displacement single-point reference windows) -----------


class DisplacedWindow(BaseModel):
    label: str
    target_a: float
    coords: str  # workspace-relative displaced .gro


class FMDisplacePlanObservation(Observation):
    windows: list[DisplacedWindow] | None = None
    pair: list[int] | None = None
    original_distance_a: float | None = None


class FMDisplacePlanParams(BaseModel):
    coords: str = Field(description="base structure (.gro) to displace")
    pair: list[int] = Field(
        min_length=2, max_length=2,
        description="[anchor, mobile] GRO atom serials (1-based): the mobile atom is "
        "moved along the anchor->mobile axis to each target distance",
    )
    pair_label: str = Field(description="a name for the pair, e.g. 'C1-O1' or 'ion-ligand O'")
    targets_a: list[float] = Field(min_length=1, description="target distances (Å)")
    rigid_group: list[int] = Field(
        default_factory=list,
        description="GRO serials translated RIGIDLY together with the mobile "
        "atom (e.g. a whole ligand). Internal geometry is preserved exactly, "
        "so displaced single points carry no intramolecular strain, which "
        "moving one atom of a molecule alone introduces. Must include "
        "the mobile atom; must not include the anchor.",
    )
    stage: str = Field(default="displace")


@tool(
    "fm_displace_plan",
    description="Build displaced structures for force-matching reference single "
    "points: the mobile atom of the pair is moved along the pair axis to each "
    "target distance (rest of the system untouched). Feed each window's .gro to "
    "mimic_select_qm + gmx_grompp + mimic_prepare (single-point FM template) + "
    "mimic_run. The restraint-free way to sample off-equilibrium geometries: CPMD's "
    "cannot restrain (CPMD CONSTRAINTS segfault under MiMiC).",
    returns=FMDisplacePlanObservation,
    tags={"mimic", "fm"},
    requires={"mimicpy"},
)
def fm_displace_plan(ctx: ToolContext, params: FMDisplacePlanParams) -> FMDisplacePlanObservation:
    src = resolve_in_ws(ctx, params.coords)
    lines = src.read_text().splitlines()
    n_atoms = int(lines[1].strip())
    ia, ib = params.pair
    if not (1 <= ia <= n_atoms and 1 <= ib <= n_atoms):
        return FMDisplacePlanObservation(
            ok=False, summary=f"pair {params.pair} outside 1..{n_atoms}"
        )
    group = sorted(set(params.rigid_group))
    if group:
        if ib not in group:
            return FMDisplacePlanObservation(
                ok=False, summary=f"rigid_group must include the mobile atom {ib}")
        if ia in group:
            return FMDisplacePlanObservation(
                ok=False, summary=f"rigid_group must not include the anchor {ia}")
        bad = [s for s in group if not (1 <= s <= n_atoms)]
        if bad:
            return FMDisplacePlanObservation(
                ok=False, summary=f"rigid_group serials outside 1..{n_atoms}: {bad[:5]}")

    def xyz(line: str) -> list[float]:
        return [float(line[20:28]), float(line[28:36]), float(line[36:44])]

    la, lb = lines[1 + ia], lines[1 + ib]
    a, b = xyz(la), xyz(lb)
    vec = [b[k] - a[k] for k in range(3)]
    d0_nm = sum(v * v for v in vec) ** 0.5
    if d0_nm < 1e-6:
        return FMDisplacePlanObservation(ok=False, summary="pair atoms coincide")
    unit = [v / d0_nm for v in vec]

    stage = ctx.workspace.stage(params.stage)
    windows: list[DisplacedWindow] = []
    artifacts = []
    for target in params.targets_a:
        t_nm = target / 10.0
        out_lines = list(lines)
        if group:
            # rigid translation: every group atom shifts by the same vector
            # that brings the mobile atom to the target distance
            shift = [(a[k] + unit[k] * t_nm) - b[k] for k in range(3)]
            for s in group:
                ls = out_lines[1 + s]
                c = xyz(ls)
                nc = [c[k] + shift[k] for k in range(3)]
                out_lines[1 + s] = ls[:20] + "".join(f"{v:8.3f}" for v in nc) + ls[44:]
        else:
            new_b = [a[k] + unit[k] * t_nm for k in range(3)]
            moved = lb[:20] + "".join(f"{c:8.3f}" for c in new_b) + lb[44:]
            out_lines[1 + ib] = moved
        out = stage / f"disp_{target:g}A.gro"
        out.write_text("\n".join(out_lines) + "\n")
        windows.append(
            DisplacedWindow(
                label=f"{params.pair_label}@{target:g}A",
                target_a=target,
                coords=ctx.workspace.relpath(out),
            )
        )
        artifacts.append(ArtifactRef.from_path(out, "structure", root=ctx.workspace.root))
    return FMDisplacePlanObservation(
        ok=True,
        summary=f"{len(windows)} displaced structure(s) for {params.pair_label} "
        f"(gro atoms {ia}->{ib}, original {d0_nm * 10:.3f} Å) at "
        f"{', '.join(f'{t:g}' for t in params.targets_a)} Å",
        artifacts=artifacts,
        windows=windows,
        pair=params.pair,
        original_distance_a=round(d0_nm * 10, 4),
    )


def assemble_candidate(ctx: ToolContext, stage: Path, top: Path, replace: dict[str, Path],
                       output: str = "candidate.top") -> Observation:
    """The candidate topology: ``top`` copied into ``stage`` with its same-directory
    includes, each include named in ``replace`` swapped for the fitted file. Done by the
    fit's collection (``job_status`` on an ``fm_fit`` job); the result is what
    ``fm_validate`` takes."""
    import shutil

    staged = _stage_topology(stage, top)
    text = staged.read_text(errors="replace")
    includes = _includes_of(staged)
    missing = [k for k in replace if k not in includes]
    if missing:
        return Observation(
            ok=False,
            summary=f"{missing} are not #include lines of {ctx.workspace.relpath(top)}; it "
            f"includes {includes}",
        )
    swapped = []
    for inc, fitted in replace.items():
        src = Path(fitted)
        dst = stage / src.name
        if dst.resolve() != src.resolve():
            shutil.copy(src, dst)
        text = text.replace(f'#include "{inc}"', f'#include "{src.name}"')
        swapped.append(f"{inc} -> {src.name}")
    out = stage / output
    out.write_text(text)
    if staged != out and staged.name != out.name:
        staged.unlink()
    root = ctx.workspace.root
    return Observation(
        ok=True,
        summary=f"{ctx.workspace.relpath(out)} ({'; '.join(swapped)})",
        artifacts=[ArtifactRef.from_path(out, "topology", root=root)]
        + [ArtifactRef.from_path(stage / Path(f).name, "fitted_topology", root=root)
           for f in replace.values()],
    )


@register_collector("fm_fit")
def _collect_fit_job(ctx: ToolContext, record, state, exit_code) -> FMObservation:
    meta = record.meta
    return collect_fit(
        ctx, Path(record.spec.cwd), original_itp=meta.get("original_itp"),
        topology=meta.get("topology"), completed=state == JobState.COMPLETED,
    )
