"""fm_frame_report: what the reference frames of a dataset hold."""

from __future__ import annotations

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.engines.gromacs.common import stage_output
from myrmidon.engines.inputs import resolve_in_ws

_BOHR_TO_A = 0.529177210903




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
    description="Characterise a force-matching reference dataset frame by frame: monitored pair "
                "distances, QM force magnitudes and outliers, near-duplicate frames, and, given "
                "the CPMD logs, temperature and SCF health per segment. Writes a per-frame CSV.",
    tags={"mimic", "fm", "analysis"},
    requires={"mimicpy"},
    contract=Contract(consumes={"fmdata": "fm_dataset"},
                      produces=(Produces("frame_report"),)),
)
def fm_frame_report(ctx: ToolContext, params: FMFrameReportParams) -> Observation:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        return Observation(ok=False, summary=f"h5py/numpy required: {exc}")

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
    return Observation(
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
