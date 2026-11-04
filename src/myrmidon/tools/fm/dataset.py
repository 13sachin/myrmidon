"""Force-matching dataset tools: ``fm_collect`` registers and validates a reference dataset
(frame counts via h5info, trr<->h5 alignment via fmcheck, sha256 of both files);
``fm_merge`` merges several into one."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import Finding
from myrmidon.engines.gromacs.common import tail
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.mimic.fm import (
    file_sha256,
    gmx_data_flags,
    parse_fmcheck_report,
    parse_h5info,
)
from myrmidon.engines.mimic.qm_region import mimicpy_bin


def _run_mimicpy(ctx: ToolContext, args: list[str], cwd: Path, timeout: int = 1800):
    argv = [mimicpy_bin(ctx), *args]
    return argv, subprocess.run(  # noqa: S603
        argv, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )




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
    tags={"mimic", "fm"},
    requires={"mimicpy"},
    contract=Contract(consumes={"fmdata": ("fm_reference", "fm_dataset", "fm_dataset_json"),
                                "trajectory": "trajectory", "topology": "topology",
                                "coords": "structure"},
                      produces=(Produces("fm_dataset"), Produces("trajectory"))),
)
def fm_collect(ctx: ToolContext, params: FMCollectParams) -> Observation:
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




class FMMergeParams(BaseModel):
    fm_jsons: list[str] = Field(min_length=1, description="FMTRAJECTORY.json (or .h5) files, "
                                "in order")
    stage: str = Field(default="fm")
    output_prefix: str = Field(default="merged")


@tool(
    "fm_merge",
    description="Merge force-matching reference data from several runs (scan windows) into one "
                "dataset: FMTRAJECTORY jsons concatenated and converted, .h5 sources copied. Join "
                "the trajectories with traj_concat in the same order, then register both with "
                "fm_collect.",
    tags={"mimic", "fm"},
    requires={"mimicpy"},
    contract=Contract(consumes={"fm_jsons": ("fm_reference", "fm_dataset")},
                      produces=(Produces("fm_dataset"), Produces("fm_dataset_json"))),
)
def fm_merge(ctx: ToolContext, params: FMMergeParams) -> Observation:
    stage = ctx.workspace.stage(params.stage)
    sources = [resolve_in_ws(ctx, src) for src in params.fm_jsons]
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
        summary=f"merged {len(params.fm_jsons)} dataset(s): {frames} configurations -> "
        f"{ctx.workspace.relpath(merged_h5)}. Join the trajectories with traj_concat in "
        "the same order, then validate both with fm_collect.",
        artifacts=[
            ArtifactRef.from_path(merged_h5, "fm_dataset", root=ctx.workspace.root),
        ]
        + (
            [ArtifactRef.from_path(merged_json, "fm_dataset_json", root=ctx.workspace.root)]
            if merged_json is not None
            else []
        ),
        frames=frames,
        n_sources=len(params.fm_jsons),
        merged_sha256=file_sha256(merged_h5),
    )
