"""Minimise each of a trajectory's frames under one topology: the job a minimising stage
submits, run as ``python -m myrmidon.engines.gromacs.batch PARAMS.json``.

Frames are taken by index, never by time window: a concatenated trajectory's times need
not increase, and ``trjconv -b/-e`` drops frames silently. Each is written as a .gro,
built into a run input with the minimisation mdp, and minimised; the minimised
coordinates go into one trajectory, its k-th frame the k-th one minimised, and each
frame's outcome is read from its own mdrun log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

RESULTS_FILENAME = "minimize.json"
LOG_FILENAME = "minimize.log"


def minimize_frames_job(ctx, stage, inputs: dict[str, Path], layout, request):
    """A batch minimising the frames a stage's scan names, one job and one driver process
    whose mdruns take every core as threads."""
    from myrmidon.engines.gromacs.common import gmx_bin, stage_topology
    from myrmidon.engines.registry import StageJob
    from myrmidon.execution.backend import JobSpec

    run_dir = ctx.workspace.stage(stage.name)
    (run_dir / "top").mkdir(exist_ok=True)
    top = stage_topology(run_dir / "top", inputs["topology"])
    frames = [int(f) for f in stage.scan.values]
    driver = {"stage": str(run_dir), "gmx": gmx_bin(ctx), "topology": str(top),
              "mdp": str(inputs["mdp"]), "structure": str(inputs["structure"]),
              "trajectory": str(inputs["trajectory"]), "frames": frames, "max_warn": 0,
              "index": str(inputs["index"]) if "index" in inputs else None,
              "ntomp": layout.cpus_per_task, "output": "minimized.trr"}
    params_file = run_dir / "minimize_params.json"
    params_file.write_text(json.dumps(driver, indent=1) + "\n")
    return StageJob(
        spec=JobSpec(argv=[sys.executable, "-m", "myrmidon.engines.gromacs.batch",
                           str(params_file)],
                     cwd=run_dir, stdout="minimize.stdout", stderr="minimize.stderr",
                     resources=request.resources()),
        kind="minimize_frames", label=f"stage:{stage.name}:{len(frames)} frames",
        summary=f"{len(frames)} frame(s) of {inputs['trajectory'].name} minimised under "
        f"{top.name}")


def collect_minimized(ctx, record, state, exit_code):
    """A minimisation job that has ended: the minimised frames' trajectory and what each
    frame's mdrun log says."""
    from myrmidon.core.artifacts import ArtifactRef
    from myrmidon.core.observation import Observation
    from myrmidon.text import tail

    stage = Path(record.spec.cwd)
    root = ctx.workspace.root
    results = stage / RESULTS_FILENAME
    log = stage / LOG_FILENAME
    if not results.is_file():
        text = log.read_text(errors="replace") if log.is_file() else ""
        err = stage / record.spec.stderr
        text += ("\n" + err.read_text(errors="replace")) if err.is_file() else ""
        return Observation(
            ok=False, summary="the minimisation job wrote no results", log_tail=tail(text, 20))
    data = json.loads(results.read_text())
    frames = data["frames"]
    converged = sum(1 for f in frames if f.get("converged"))
    artifacts = [ArtifactRef.from_path(results, "data", root=root)]
    if data.get("output") and (stage / data["output"]).is_file():
        artifacts.insert(0, ArtifactRef.from_path(stage / data["output"], "trajectory",
                                                  root=root))
    if log.is_file():
        artifacts.append(ArtifactRef.from_path(log, "log", root=root))
    failed = [f["index"] for f in frames if not f.get("ok")]
    return Observation(
        ok=data["n_minimised"] > 0,
        summary=f"{data['n_minimised']} of {data['n_requested']} frame(s) minimised, "
        f"{converged} converged to the mdp's force tolerance"
        + (f"; frames {failed} failed (see {LOG_FILENAME})" if failed else ""),
        artifacts=artifacts, frames=frames, n_requested=data["n_requested"],
        n_minimised=data["n_minimised"], n_converged=converged,
    )


def _run(argv: list, cwd: Path, log: Path, env: dict | None = None) -> bool:
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(str(a) for a in argv) + "\n")
        fh.flush()
        proc = subprocess.run([str(a) for a in argv], cwd=str(cwd), stdout=fh, stderr=fh,
                              env=env, check=False)
    return proc.returncode == 0


def main(params_path: str) -> int:
    import MDAnalysis as mda

    from myrmidon.engines.gromacs.parse import run_facts

    p = json.loads(Path(params_path).read_text())
    stage = Path(p["stage"])
    log = stage / LOG_FILENAME
    top = Path(p["topology"])
    # mdrun aborts when -ntomp disagrees with OMP_NUM_THREADS, which batch environments'
    # module scripts commonly set to 1
    env = {**os.environ, "OMP_NUM_THREADS": str(p["ntomp"])}
    u = mda.Universe(p["structure"], p["trajectory"])
    results, minimised = [], []
    for f in p["frames"]:
        work = stage / f"f{f:05d}"
        work.mkdir(parents=True, exist_ok=True)
        u.trajectory[f]
        frame = work / "frame.gro"
        with mda.Writer(str(frame), n_atoms=len(u.atoms)) as w:
            w.write(u.atoms)
        tpr = work / "em.tpr"
        argv = [p["gmx"], "grompp", "-f", p["mdp"], "-c", frame, "-p", top, "-o", tpr,
                "-po", work / "mdout.mdp", "-maxwarn", str(p["max_warn"]), "-quiet"]
        if p.get("index"):
            argv += ["-n", p["index"]]
        # grompp runs in the topology's directory so its #includes resolve
        ok = _run(argv, top.parent, log) and _run(
            [p["gmx"], "mdrun", "-s", tpr, "-deffnm", work / "em", "-ntomp", p["ntomp"],
             "-quiet"], work, log, env)
        em_gro, em_log = work / "em.gro", work / "em.log"
        facts = run_facts(em_log.read_text(errors="replace")) if em_log.is_file() else {}
        entry = {"index": f, "ok": bool(ok and em_gro.is_file()),
                 "converged": facts.get("minimisation_converged"),
                 "steps": facts.get("minimisation_steps"),
                 "final_potential_kj_mol": facts.get("final_potential_kj_mol"),
                 "final_max_force_kj_mol_nm": facts.get("final_max_force_kj_mol_nm")}
        results.append(entry)
        if entry["ok"]:
            minimised.append((f, em_gro))
    out = stage / p["output"]
    if minimised:
        with mda.Writer(str(out), n_atoms=len(u.atoms)) as w:
            for f, gro in minimised:
                m = mda.Universe(str(gro))
                m.trajectory.ts.time = float(f)
                w.write(m.atoms)
    (stage / RESULTS_FILENAME).write_text(json.dumps({
        "frames": results, "n_requested": len(results), "n_minimised": len(minimised),
        "output": p["output"] if minimised else None}, indent=1) + "\n")
    return 0 if minimised else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
