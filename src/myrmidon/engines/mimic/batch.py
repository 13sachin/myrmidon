"""Coupled single points over a scan's frames as one job: what a QM/MM single-point stage over
frames submits. ``python -m myrmidon.engines.mimic.batch PARAMS.json`` runs first in the job,
on the batch host, and prepares a directory per frame; the backend then launches CPMD and
GROMACS in each directory as it launches a single coupled run.

Per frame, taken by index from the trajectory: its coordinates as a .gro; prepqm on them with
the approved selection, so the QM atoms, their order, the units and the cell padding are
prepqm's own; the stage's method and spin applied onto that system input (``apply_method``,
``set_spin``), as a stage's render applies them; grompp for the MM half with the frame's own
index. A frame whose preparation fails is said in ``prepared.json``; its directory lacks the
files the launch needs, so its run is never started and fails alone.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PREPARED_FILENAME = "prepared.json"
#: what a scan's collection says of each frame, beside the frames' directories
FRAMES_FILENAME = "frames.json"
LOG_FILENAME = "prepare.log"
#: what each frame's directory holds for the coupled launch to read
RUN_INPUT, RUN_TPR = "frame.inp", "frame.tpr"


def point_name(frame: int) -> str:
    return f"f{int(frame):05d}"


def _run(argv: list, cwd: Path, log: Path, stdin: str | None = None) -> bool:
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(str(a) for a in argv) + "\n")
        fh.flush()
        proc = subprocess.run([str(a) for a in argv], cwd=str(cwd), input=stdin, text=True,
                              stdout=fh, stderr=fh, check=False)
    return proc.returncode == 0


def prepare(p: dict) -> list[dict]:
    """Each frame's directory, ready for the coupled launch; one entry per frame."""
    import MDAnalysis as mda

    from myrmidon.engines.mimic.cpmd_input import apply_method, parse_cpmd, render_cpmd, set_spin
    from myrmidon.engines.mimic.qm_region import prepqm_argv

    run_dir = Path(p["run_dir"])
    log = run_dir / LOG_FILENAME
    method = parse_cpmd(Path(p["method_input"]).read_text())
    stdin = "\n".join(p["selections"]) + f"\n{p['quit_cmd']}\n"
    out = []
    try:
        u = mda.Universe(p["structure"], p["trajectory"])
    except (OSError, ValueError) as exc:  # no frame can be read against the structure
        with open(log, "a") as fh:
            fh.write(f"{p['trajectory']}: {exc}\n")
        u = None
    for frame in p["frames"]:
        work = run_dir / point_name(frame)
        work.mkdir(parents=True, exist_ok=True)
        coords = work / "frame.gro"
        try:
            if u is None:
                raise ValueError("the trajectory cannot be read against the structure")
            u.trajectory[int(frame)]
            with mda.Writer(str(coords), n_atoms=len(u.atoms)) as w:
                w.write(u.atoms)
        except (IndexError, OSError, ValueError) as exc:  # this frame alone fails
            with open(log, "a") as fh:
                fh.write(f"frame {frame}: {exc}\n")
            out.append({"index": int(frame), "point": point_name(frame), "prepared": False})
            continue
        argv = prepqm_argv(p["mimicpy"], Path(p["topology"]), coords, work, p["qm_group"],
                           p["padding_nm"], p.get("data_dir"), None,
                           Path(p["pp_info"]) if p.get("pp_info") else None,
                           p.get("atomtypes_map") or {})
        ok = _run(argv, work, log, stdin) and (work / "cpmd.inp").is_file()
        if ok:
            sections = apply_method(parse_cpmd((work / "cpmd.inp").read_text()), method)
            sections = set_spin(sections, p["charge"], p["multiplicity"])
            (work / RUN_INPUT).write_text(render_cpmd(sections))
            grompp = [p["gmx"], "grompp", "-f", p["mdp"], "-c", coords, "-p", p["topology"],
                      "-n", work / "index.ndx", "-o", work / RUN_TPR,
                      "-po", work / "mdout.mdp", "-maxwarn", "0", "-quiet"]
            # grompp runs in the topology's directory so its #includes resolve
            ok = _run(grompp, Path(p["topology"]).parent, log) and (work / RUN_TPR).is_file()
        out.append({"index": int(frame), "point": point_name(frame), "prepared": bool(ok)})
    (run_dir / PREPARED_FILENAME).write_text(json.dumps({"frames": out}, indent=1) + "\n")
    return out


def is_scan(stage) -> bool:
    """Whether a coupled stage is single points over the frames its scan names."""
    return (stage.kind == "single_point" and stage.scan is not None
            and stage.scan.parameter == "frames")


def frames_done(ctx, stage) -> set[int]:
    """The frames of a scan stage a collected run has computed."""
    from myrmidon.core.lineage import stage_runs

    done: set[int] = set()
    for outputs, _ in stage_runs(ctx.workspace, stage.name):
        for a in outputs:
            if Path(a.path).name == FRAMES_FILENAME:
                try:
                    data = json.loads(ctx.workspace.resolve(a.path).read_text())
                except (OSError, ValueError):
                    continue
                done |= {f["index"] for f in data.get("frames", []) if f.get("ok")}
    return done


def frames_in(structure: Path, trajectory: Path) -> int | None:
    """How many frames a trajectory holds, read against its structure; None when it cannot be
    read here, which the job's own read will say."""
    import MDAnalysis as mda

    try:
        return len(mda.Universe(str(structure), str(trajectory)).trajectory)
    except (OSError, ValueError):
        return None


def frames_in_flight(ctx, stage) -> set[int]:
    """The frames a submitted run of a scan stage computes whose job has not been collected."""
    from myrmidon.core.lineage import lineage, named_stage

    ws = ctx.workspace
    collected = lineage(ws).submissions
    return {int(f) for ev in ws.ledger.events()
            if ev.tool == "stage_run" and ev.ok and ev.kind == "call"
            and named_stage(ev) == stage.name and ev.event_id not in collected
            for f in ev.params.get("frames") or ()}


def frames_to_run(ctx, stage, asked: list[int]) -> list[int]:
    """The frames a run of a scan stage computes: those asked, else every frame of the scan
    no collected run has computed and no running one is computing."""
    if asked:
        return [int(f) for f in asked]
    taken = frames_done(ctx, stage) | frames_in_flight(ctx, stage)
    return [int(f) for f in stage.scan.values if int(f) not in taken]


def scan_frames(ctx, stage, asked: list[int]) -> list[int] | None:
    """The frames a run of a stage computes, when it is a scan's; None for any other stage."""
    return frames_to_run(ctx, stage, asked) if is_scan(stage) else None


def scan_job(ctx, stage, inputs: dict[str, Path], layout, request):
    """The frames this run computes, as one job: the prelude prepares a directory per frame
    with the partition the stage's region was made with, and CPMD and GROMACS run in each."""
    from myrmidon.core.mandate import read_mandate
    from myrmidon.engines.gromacs.common import gmx_bin
    from myrmidon.engines.mimic.partition import PARTITION_TOOL, region_settings
    from myrmidon.engines.mimic.qm_region import mimicpy_bin
    from myrmidon.engines.registry import StageJob
    from myrmidon.execution.backend import CoupledCmd, JobSpec

    ws = ctx.workspace
    frames = frames_to_run(ctx, stage, getattr(request, "frames", []))
    made = [ev for ev in ws.ledger.events()
            if ev.tool == PARTITION_TOOL and ev.ok and ev.kind == "call"]
    selection = next((ev for ev in reversed(made) if ev.params.get("stage") == stage.name),
                     made[-1] if made else None)
    run_dir = ws.run_dir(ws.stage(stage.name), f"{stage.name}_frames")
    gmx = ctx.engines.get("gromacs")
    region = stage.qm_region
    # each frame is partitioned as the stage's region was: by the call that made it, else as
    # the stage and its method state
    padding, table = region_settings(read_mandate(ws), stage)
    pp_info = (ws.resolve(selection.params["pp_info"])
               if selection and selection.params.get("pp_info") else None)
    if selection is None and table:
        pp_info = run_dir / "pp_info.dat"
        pp_info.write_text(table)
    driver = {
        "run_dir": str(run_dir), "frames": frames, "mimicpy": mimicpy_bin(ctx),
        "gmx": gmx_bin(ctx), "data_dir": gmx.data_dir if gmx else None,
        "structure": str(inputs["structure"]), "trajectory": str(inputs["trajectory"]),
        "topology": str(inputs["topology"]), "mdp": str(inputs["mdp"]),
        "method_input": str(inputs["cpmd_input"]),
        "selections": (inputs["qm_selection"].read_text().split("\n")
                       if "qm_selection" in inputs else []),
        "padding_nm": selection.params.get("padding_nm") if selection else padding,
        "pp_info": str(pp_info) if pp_info else None,
        "atomtypes_map": selection.params.get("atomtypes_map", {}) if selection else {},
        "qm_group": selection.params.get("qm_group", "QMatoms") if selection else "QMatoms",
        "quit_cmd": ctx.config.mimicpy_quit_cmd,
        "charge": region.charge, "multiplicity": region.multiplicity or 1}
    driver["selections"] = [c for c in driver["selections"] if c.strip()]
    params_file = run_dir / "frames_params.json"
    params_file.write_text(json.dumps(driver, indent=1) + "\n")
    threads = {"OMP_NUM_THREADS": str(layout.cpus_per_task)}
    spec = JobSpec(
        argv=[ctx.engines["cpmd"].path, RUN_INPUT], cwd=run_dir, stdout="cpmd.out",
        stderr="cpmd.err", mpi_ranks=layout.ntasks, env=threads,
        prelude=[sys.executable, "-m", "myrmidon.engines.mimic.batch", str(params_file)],
        points=[point_name(f) for f in frames], point_reads=[RUN_INPUT, RUN_TPR],
        receipt_inputs=[str(params_file), *(str(inputs[k]) for k in (
            "cpmd_input", "mdp", "trajectory", "topology", "qm_selection") if k in inputs)],
        resources=request.resources(),
        coupled=[CoupledCmd(
            argv=[gmx_bin(ctx), "mdrun", "-s", RUN_TPR, "-deffnm", stage.name],
            mpi_ranks=layout.partner_ntasks, stdout=f"{stage.name}.stdout",
            stderr=f"{stage.name}.stderr")])
    return StageJob(
        spec=spec, kind="mimic_scan", label=f"stage:{stage.name}:{len(frames)} frames",
        meta={"deffnm": stage.name, "frames": frames},
        summary=f"{len(frames)} frame(s) of {inputs['trajectory'].name}, each partitioned by "
        f"prepqm and run coupled, in {ws.relpath(run_dir)}")


def collect_scan(ctx, record, state, exit_code):
    """A scan job that has ended: each frame's QM log, reference forces and MM trajectory,
    and whether it finished; ``ok`` when any frame did."""
    from myrmidon.core.artifacts import ArtifactRef
    from myrmidon.core.observation import Observation
    from myrmidon.engines.mimic.diagnostics import parse_cpmd_output
    from myrmidon.text import tail

    run_dir, root = Path(record.spec.cwd), ctx.workspace.root
    deffnm = record.meta.get("deffnm", "")
    artifacts, frames = [], []
    try:
        prepared = {f["index"]: f["prepared"] for f in json.loads(
            (run_dir / PREPARED_FILENAME).read_text())["frames"]}
    except (OSError, ValueError, KeyError):
        prepared = {}
    for frame in record.meta.get("frames", []):
        work = run_dir / point_name(frame)
        log = work / "cpmd.out"
        facts = parse_cpmd_output(log.read_text(errors="replace")) if log.is_file() else {}
        forces = sorted(work.glob("FMTRAJECTORY*.json"))
        trajectory = work / f"{deffnm}.trr"
        ok = bool(facts.get("finished_normally") and forces and trajectory.is_file()
                  and facts.get("wavefunction_converged", True))
        frames.append({"index": frame, "ok": ok, "prepared": prepared.get(frame, False),
                       "converged": facts.get("wavefunction_converged"),
                       "scf_iterations": facts.get("scf_iterations")})
        if log.is_file():
            artifacts.append(ArtifactRef.from_path(log, "qm_log", root=root))
        if ok:
            artifacts += [*(ArtifactRef.from_path(f, "fm_reference", root=root) for f in forces),
                          ArtifactRef.from_path(trajectory, "trajectory", root=root)]
    summary_file = run_dir / FRAMES_FILENAME
    summary_file.write_text(json.dumps({"frames": frames}, indent=1) + "\n")
    artifacts.append(ArtifactRef.from_path(summary_file, "data", root=root))
    done = [f["index"] for f in frames if f["ok"]]
    failed = [f["index"] for f in frames if not f["ok"]]
    prep = run_dir / LOG_FILENAME
    return Observation(
        ok=bool(done),
        summary=f"{len(done)} of {len(frames)} frame(s) computed"
        + (f"; frames {failed} did not finish or converge" if failed else ""),
        artifacts=artifacts, frames=frames, n_frames=len(frames), n_computed=len(done),
        log_tail=tail(prep.read_text(errors="replace"), 20) if failed and prep.is_file()
        else None)


def main(params_path: str) -> int:
    frames = prepare(json.loads(Path(params_path).read_text()))
    return 0 if any(f["prepared"] for f in frames) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
