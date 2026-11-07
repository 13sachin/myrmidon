"""A reference dataset for a fit: FMTRAJECTORY files converted and merged, each frame checked
against its trajectory (fmcheck, an aligned pair written when they have drifted), and what the
frames hold summarised. A refit after a trajectory is extended is a new, provenance-linked
dataset, as both files are hashed."""

from __future__ import annotations

import subprocess
from pathlib import Path

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext
from myrmidon.core.validation import Finding
from myrmidon.engines.gromacs.common import join_trajectories, tail
from myrmidon.engines.inputs import resolve_in_ws, stage_output
from myrmidon.engines.mimic.fm import (
    file_sha256,
    gmx_data_flags,
    parse_fmcheck_report,
    parse_h5info,
)
from myrmidon.engines.mimic.qm_region import mimicpy_bin
from myrmidon.engines.registry import DatasetRequest

_BOHR_TO_A = 0.529177210903


def _run_mimicpy(ctx: ToolContext, args: list[str], cwd: Path, timeout: int = 1800):
    argv = [mimicpy_bin(ctx), *args]
    return argv, subprocess.run(  # noqa: S603
        argv, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def collect_dataset(ctx: ToolContext, params: DatasetRequest) -> Observation:
    """One dataset from what the request names: several runs merged, then validated, then what
    its frames hold summarised."""
    extra: list[ArtifactRef] = []
    sources = ", ".join(params.reference_files)
    if len(params.reference_files) > 1:
        merged = merge_sources(ctx, params)
        if not merged.ok:
            return merged
        extra = [a for a in merged.artifacts if a.role == "fm_dataset_json"]
        params.reference_file = next(a.path for a in merged.artifacts
                                 if a.role == "reference_dataset")
    else:
        params.reference_file = params.reference_files[0]
    trajectory = joined_trajectory(ctx, params)
    if isinstance(trajectory, Observation):
        return trajectory
    params.trajectory = [trajectory]
    collected = _validated(ctx, params)
    if not collected.ok:
        return collected
    collected.artifacts += extra
    collected.sources = sources
    dataset = next(a.path for a in collected.artifacts if a.role == "reference_dataset")
    try:
        frames = frame_report(ctx, params, dataset)
    except Exception as exc:  # noqa: BLE001 -- the dataset holds without its summary, and says so
        collected.findings = [*collected.findings, Finding(
            id="FM_FRAME_REPORT_SKIPPED",
            message=f"what the frames hold was not summarised: {exc}")]
        return collected
    collected.artifacts += frames.artifacts
    collected.summary += ". " + frames.summary
    for key in ("n_frames", "monitor_stats", "monitor_missing", "force_outlier_frames",
                "n_near_duplicates", "duplicate_rmsd_threshold_a", "md_steps_from_logs",
                "temp_mean_k", "temp_final_k", "scf_iters_mean", "scf_iters_max", "segments"):
        setattr(collected, key, getattr(frames, key, None))
    return collected


def joined_trajectory(ctx: ToolContext, params: DatasetRequest) -> str | Observation:
    """The one trajectory the dataset's frames pair with: several joined in the order given,
    every frame kept."""
    if len(params.trajectory) == 1:
        return params.trajectory[0]
    stage = ctx.workspace.stage(params.stage)
    out = stage / f"{params.output_prefix}.trr"
    proc = join_trajectories(ctx, [resolve_in_ws(ctx, t) for t in params.trajectory], out)
    if proc.returncode != 0 or not out.exists():
        return Observation(ok=False, summary=f"gmx trjcat failed (exit {proc.returncode})",
                           command=list(proc.args), log_tail=tail(proc.stderr))
    return ctx.workspace.relpath(out)


def _validated(ctx: ToolContext, params: DatasetRequest) -> Observation:
    """Convert to .h5 if needed, count frames (h5info), spot-check the trajectory against the
    dataset (fmcheck), and hash both files."""
    stage = ctx.workspace.stage(params.stage)
    fmdata = resolve_in_ws(ctx, params.reference_file)
    trr = resolve_in_ws(ctx, params.trajectory[0])
    top = resolve_in_ws(ctx, params.topology)
    coords = resolve_in_ws(ctx, params.coords)
    sele = resolve_in_ws(ctx, params.selection)

    if fmdata.suffix == ".json":
        h5 = stage / (fmdata.stem + ".h5")
        argv, proc = _run_mimicpy(
            ctx, ["json2h5", "-i", str(fmdata), "-o", str(h5)], cwd=stage
        )
        if proc.returncode != 0 or not h5.exists():
            return Observation(
                ok=False,
                summary=f"json2h5 failed (exit {proc.returncode})",
                command=argv,
                log_tail=tail(proc.stderr + "\n" + proc.stdout, 30),
            )
        fmdata = h5

    argv, proc = _run_mimicpy(ctx, ["h5info", "-i", str(fmdata)], cwd=stage)
    info = parse_h5info(proc.stdout + proc.stderr)
    if proc.returncode != 0 or "frames" not in info:
        return Observation(
            ok=False,
            summary=f"h5info failed (exit {proc.returncode}) — is {params.reference_file} an "
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
             "-fmdata", str(fmdata), "-trr", str(trr), *extra, *gmx_data_flags(ctx)],
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
    warnings: list[Finding] = []
    if check and not check.get("index_aligned", True):
        warnings.append(
            Finding(
                id="FM_TRR_H5_DESYNC",
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
            Finding(
                id="FM_FRAMES_DROPPED",
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
             "-o", str(prefix), *gmx_data_flags(ctx)],
            cwd=stage,
            timeout=3600,
        )
        aligned_h5 = prefix.with_suffix(".h5")
        aligned_trr = prefix.with_suffix(".trr")
        if proc.returncode != 0 or not (aligned_h5.exists() and aligned_trr.exists()):
            return Observation(
                ok=False,
                summary=f"writing the aligned trr/h5 pair failed (exit {proc.returncode})",
                command=argv,
                findings=warnings,
                log_tail=tail(proc.stderr + "\n" + proc.stdout, 30),
            )
        with check_log.open("a") as fh:
            fh.write("\n===== aligning run =====\n" + proc.stdout + "\n" + proc.stderr)
        full = parse_fmcheck_report(proc.stdout + proc.stderr)
        fmdata, trr = aligned_h5, aligned_trr
        frames = full.get("matched", frames)
        check["dropped"] = full.get("dropped", check.get("dropped"))
        warnings.append(
            Finding(
                id="FM_ALIGNED_PAIR_WRITTEN",
                message=f"index-aligned pair written ({frames} matched frame(s), "
                f"{full.get('dropped', 0)} dropped) — this pair, not the originals, "
                "is the registered dataset",
            )
        )

    fm_hash = file_sha256(fmdata)
    trr_hash = file_sha256(trr)
    return Observation(
        ok=True,
        summary=f"FM dataset registered: {frames} configurations "
        f"({info.get('qm_atoms', '?')} QM atoms), trr {check.get('trr_frames', '?')} "
        f"frames, alignment {check.get('status', 'unchecked')} "
        f"({'index-aligned' if check.get('index_aligned') else 'REMAPPED — see warning'}); "
        f"h5 sha256 {fm_hash[:12]}…, trr sha256 {trr_hash[:12]}…",
        command=argv,
        findings=warnings,
        artifacts=[
            ArtifactRef.from_path(fmdata, "reference_dataset", root=ctx.workspace.root),
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


def merge_sources(ctx: ToolContext, params: DatasetRequest) -> Observation:
    """Reference data from several runs (scan windows) as one dataset: FMTRAJECTORY jsons
    concatenated and converted, .h5 sources copied."""
    stage = ctx.workspace.stage(params.stage)
    sources = [resolve_in_ws(ctx, src) for src in params.reference_files]
    merged_json = None
    frames = 0
    if any(src.suffix == ".h5" for src in sources):
        # binary merge: copy config groups renumbered (json sources converted first)
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            return Observation(ok=False, summary=f"h5py required: {exc}")
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
                    return Observation(
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
    merged_h5 = stage / f"{params.output_prefix}.h5"
    if merged_json is not None:
        argv, conv = _run_mimicpy(
            ctx, ["json2h5", "-i", str(merged_json), "-o", str(merged_h5)], cwd=stage
        )
        if conv.returncode != 0 or not merged_h5.exists():
            return Observation(
                ok=False,
                summary=f"json2h5 failed (exit {conv.returncode})",
                command=argv,
                log_tail=tail(conv.stderr + "\n" + conv.stdout, 30),
            )
    return Observation(
        ok=True,
        summary=f"merged {len(params.reference_files)} dataset(s): {frames} configurations -> "
        f"{ctx.workspace.relpath(merged_h5)}",
        artifacts=[
            ArtifactRef.from_path(merged_h5, "reference_dataset", root=ctx.workspace.root),
        ]
        + (
            [ArtifactRef.from_path(merged_json, "fm_dataset_json", root=ctx.workspace.root)]
            if merged_json is not None
            else []
        ),
        frames=frames,
        n_sources=len(params.reference_files),
        merged_sha256=file_sha256(merged_h5),
    )


def frame_report(ctx: ToolContext, params: DatasetRequest, dataset: str) -> Observation:
    """A reference dataset frame by frame: monitored pair distances, QM force magnitudes and
    outliers, near-duplicate frames, and, given the CPMD logs, temperature and SCF health per
    segment. Writes a per-frame CSV."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        return Observation(ok=False, summary=f"h5py/numpy required: {exc}")

    from myrmidon.engines.mimic.diagnostics import parse_cpmd_output, parse_cpmd_series

    stage = ctx.workspace.stage(params.stage)
    h5_path = resolve_in_ws(ctx, dataset)

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
    for log in params.run_logs:
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
