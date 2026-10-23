"""Compute side of ``fm_validate`` — runs inside the submitted job.

Invoked as ``python -m myrmidon.engines.mimic.validate_driver params.json``.
Everything numeric it produces is parsed from engine output or trajectory
files; the JSON it writes is what ``job_status`` collects and grounds on.

Metrics:

- ``min_geometry`` — energy-minimize reference frames under each candidate
  topology and compare monitor-pair distances against the reference-ensemble
  means (does the force field put the minima where the QM/MM ensemble is?).
  Frames are selected by INDEX, never by time window: concatenated reference
  trajectories carry non-monotonic times and ``trjconv -b/-e`` silently drops
  frames.
- ``md_geometry`` — monitor-pair distance mean/std from existing MD
  trajectories of a candidate vs the reference ensemble (does the ensemble
  hold the geometry, and does anything detach?).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def em_mdp_text(em: dict) -> str:
    """The minimization recipe, from the input set store rather than from a string
    in this file. ``em`` carries the caller's nsteps/emtol, which are
    recorded as origin 'user' exactly as they would be through inputset_apply."""
    from myrmidon.engines.gromacs.mdp import render_mdp
    from myrmidon.knowledge.inputsets import InputSetStore, resolve

    inputset = InputSetStore.builtin().get("em_solvated_biomolecule")
    resolution = resolve(inputset, overrides=dict(em))
    return render_mdp(
        resolution.values,
        resolution.origin,
        header=[
            "written by myrmidon fm_validate (min_geometry metric)",
            f"inputset: {inputset.id} v{inputset.version}",
        ],
    )


def read_gro(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates (nm, shape (n, 3)) and box diagonal (nm) from a .gro file."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[1])
    coords = np.empty((n, 3))
    for i in range(n):
        ln = lines[2 + i]
        coords[i] = (float(ln[20:28]), float(ln[28:36]), float(ln[36:44]))
    box = np.array([float(x) for x in lines[2 + n].split()[:3]])
    return coords, box


def pair_distance(coords: np.ndarray, box: np.ndarray, a: int, b: int) -> float:
    """Minimum-image distance in Angstrom between 1-based gro serials a, b.

    Orthorhombic minimum image — the box diagonal only. Good for the
    protein-in-water boxes this metric is used on; triclinic boxes need the
    full matrix and are not handled here.
    """
    d = coords[a - 1] - coords[b - 1]
    d -= box * np.round(d / box)
    return float(np.linalg.norm(d) * 10.0)


def reference_stats(structure, trajectory, pairs, stride=1):
    """Per-pair distance mean/std (Angstrom) over the reference ensemble."""
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import distance_array

    u = mda.Universe(str(structure), str(trajectory))
    series = {label: [] for label in pairs}
    for _ in u.trajectory[::stride]:
        for label, (a, b) in pairs.items():
            series[label].append(float(distance_array(
                u.atoms[a - 1].position.reshape(1, 3),
                u.atoms[b - 1].position.reshape(1, 3),
                box=u.dimensions)[0, 0]))
    n_frames = len(u.trajectory)
    return {
        label: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for label, v in series.items()
    }, n_frames


def write_frame(structure, trajectory, index, out):
    """Write one trajectory frame (by INDEX) as a .gro file."""
    import MDAnalysis as mda

    u = mda.Universe(str(structure), str(trajectory))
    u.trajectory[index]
    u.trajectory.ts.time = float(index)
    with mda.Writer(str(out), n_atoms=len(u.atoms)) as w:
        w.write(u.atoms)


def _run(argv, cwd, log, env=None):
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(str(a) for a in argv) + "\n")
        fh.flush()
        proc = subprocess.run(
            [str(a) for a in argv], cwd=str(cwd), stdout=fh, stderr=fh, env=env)
    return proc.returncode == 0


def _omp_env(ntomp):
    """gmx mdrun aborts when -ntomp disagrees with OMP_NUM_THREADS, and batch
    environments' module scripts commonly export OMP_NUM_THREADS=1 — make them agree."""
    import os

    return {**os.environ, "OMP_NUM_THREADS": str(ntomp)}


def minimize_frame(gmx, top, frame_gro, workdir, em, ntomp, log):
    """grompp + EM mdrun for one frame; returns em.gro path or None."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mdp = workdir / "em.mdp"
    mdp.write_text(em_mdp_text(em))
    tpr = workdir / "em.tpr"
    # grompp runs in the topology's directory so its #includes resolve
    ok = _run([gmx, "grompp", "-f", mdp, "-c", frame_gro, "-p", top,
               "-o", tpr, "-maxwarn", "10", "-quiet"], Path(top).parent, log)
    if not ok:
        return None
    ok = _run([gmx, "mdrun", "-s", tpr, "-deffnm", workdir / "em",
               "-ntomp", str(ntomp), "-quiet"], workdir, log, env=_omp_env(ntomp))
    em_gro = workdir / "em.gro"
    return em_gro if ok and em_gro.exists() else None


def metric_min_geometry(p, ref, stage, log):
    frames = p["min_frames"]
    out = {}
    frame_gros = {}
    for f in frames:
        fg = stage / f"frame_{f:05d}.gro"
        write_frame(p["structure"], p["trajectory"], f, fg)
        frame_gros[f] = fg
    for name, top in p["candidates"].items():
        per_pair = {label: [] for label in p["monitor_pairs"]}
        n_ok = 0
        for f in frames:
            em_gro = minimize_frame(
                p["gmx"], top, frame_gros[f], stage / "em" / name / f"f{f:05d}",
                p["em"], p["ntomp"], log)
            if em_gro is None:
                continue
            n_ok += 1
            coords, box = read_gro(em_gro)
            for label, (a, b) in p["monitor_pairs"].items():
                per_pair[label].append(pair_distance(coords, box, a, b))
        pairs = {label: {"mean": float(np.mean(v))} if v else None
                 for label, v in per_pair.items()}
        devs = [pairs[lb]["mean"] - ref[lb]["mean"]
                for lb in pairs if pairs[lb] is not None]
        out[name] = {
            "n_frames_requested": len(frames),
            "n_frames_minimized": n_ok,
            "pairs": pairs,
            "mae": float(np.mean(np.abs(devs))) if devs else None,
        }
    return out


def metric_md_geometry(p, ref, log):
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import distance_array

    out = {}
    for name, trajs in (p.get("md_trajectories") or {}).items():
        series = {label: [] for label in p["monitor_pairs"]}
        n_used = 0
        for traj in trajs:
            u = mda.Universe(str(p["structure"]), str(traj))
            for ts in u.trajectory:
                if ts.time < p["md_skip_ps"]:
                    continue
                n_used += 1
                for label, (a, b) in p["monitor_pairs"].items():
                    series[label].append(float(distance_array(
                        u.atoms[a - 1].position.reshape(1, 3),
                        u.atoms[b - 1].position.reshape(1, 3),
                        box=u.dimensions)[0, 0]))
        pairs = {label: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                 for label, v in series.items() if v}
        devs = [pairs[lb]["mean"] - ref[lb]["mean"] for lb in pairs]
        out[name] = {
            "n_frames_used": n_used,
            "pairs": pairs,
            "mae": float(np.mean(np.abs(devs))) if devs else None,
        }
    return out


def write_subtrajectory(structure, trajectory, frames, out):
    """The selected reference frames (by INDEX), positions and box only, as a trr for
    mdrun -rerun. Frame k of the output is reference frame frames[k]."""
    import MDAnalysis as mda

    u = mda.Universe(str(structure), str(trajectory))
    with mda.Writer(str(out), n_atoms=len(u.atoms)) as w:
        for k, f in enumerate(frames):
            u.trajectory[f]
            u.trajectory.ts.time = float(k)
            w.write(u.atoms)


def rerun_forces(gmx, top, structure, mdp, sub_trr, index, workdir, ntomp, log):
    """grompp + mdrun -rerun of the candidate topology over the sub-trajectory; returns
    the rerun trr (forces in GROMACS units) or None."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tpr = workdir / "rerun.tpr"
    argv = [gmx, "grompp", "-f", mdp, "-c", structure, "-p", top, "-o", tpr,
            "-maxwarn", "10", "-quiet"]
    if index:
        argv += ["-n", index]
    if not _run(argv, Path(top).parent, log):
        return None
    out = workdir / "rerun.trr"
    ok = _run([gmx, "mdrun", "-s", tpr, "-rerun", sub_trr, "-o", out,
               "-e", workdir / "rerun.edr", "-g", workdir / "rerun.log",
               "-ntomp", str(ntomp), "-quiet"], workdir, log, env=_omp_env(ntomp))
    return out if ok and out.exists() else None


def force_residual_rows(rerun_trr, structure, fmdata, qm_region, frames):
    """Per (frame, QM atom) rows of ``F_candidate(MM) - F_reference(QM/MM)`` on the QM
    atoms, atomic units, in the validation_force_residuals.csv layout mimicpy writes.

    Reference forces come from the h5 dataset (CPMD order, mapped to topology order with
    mimicpy's own build_fm_to_top_order); candidate forces from the rerun trr (row g-1 of
    GROMACS atom g, converted with mimicpy's kjmolnm_to_au — the same conversion the fit
    used). This is the total-force analogue of the fit's bonded-force residual, and it
    applies to any topology, stock or fitted."""
    import MDAnalysis as mda
    from mimicpy.force_matching.frame_consistency import build_fm_to_top_order
    from mimicpy.utils.constants import kjmolnm_to_au

    order = build_fm_to_top_order(fmdata, qm_region, frames[0])
    table = qm_region.qm_atoms
    rows_idx = np.array([int(g) - 1 for g in table.index], dtype=int)
    u = mda.Universe(str(structure), str(rerun_trr))
    if len(u.trajectory) < len(frames):
        raise ValueError(f"rerun trr has {len(u.trajectory)} frames, expected {len(frames)}")
    col = lambda name, default: (list(table[name]) if name in table.columns  # noqa: E731
                                 else [default] * len(table))
    names, types, resnames = col("name", ""), col("type", ""), col("resname", "")
    resids, elements, bound = col("resid", 0), col("element", ""), col("is_bound", 0)
    rows = []
    for k, f in enumerate(frames):
        ts = u.trajectory[k]
        if not ts.has_forces:
            raise ValueError("rerun trr carries no forces — nstfout must be 1 in the rerun mdp")
        cand = ts.forces[rows_idx] * 10.0 * kjmolnm_to_au  # kJ/mol/Å -> kJ/mol/nm -> a.u.
        ref = fmdata.get_configuration_properties(f, "force", "qm")[order]
        res = cand - ref
        for a, gmx_id in enumerate(table.index):
            rows.append(
                f"{f},{int(gmx_id)},{names[a]},{types[a]},{resnames[a]},{resids[a]},"
                f"{elements[a]},{int(bound[a])},"
                f"{res[a, 0]:.6f},{res[a, 1]:.6f},{res[a, 2]:.6f},"
                f"{ref[a, 0]:.6f},{ref[a, 1]:.6f},{ref[a, 2]:.6f}")
    header = ("frame_index,gmx_id,name,type,resname,resid,element,is_bound,"
              "res_x,res_y,res_z,ref_x,ref_y,ref_z")
    return header, rows


def metric_force_rmse(p, stage, log):
    """Held-out total-force RMSE of every candidate topology on the QM atoms: candidate
    MM forces from a classical rerun over selected reference frames vs the QM/MM forces
    of the dataset. Per atom type and per atom via the shared residual parser."""
    from mimicpy.force_matching.qm_region import QMRegion
    from mimicpy.scripts.fmdata import FMDataset

    from myrmidon.engines.mimic.fm import parse_validation_force_residuals

    frames = [int(f) for f in p["force_frames"]]
    sub_trr = stage / "force_frames.trr"
    write_subtrajectory(p["structure"], p["trajectory"], frames, sub_trr)
    fmdata = FMDataset(p["fmdata"])
    out = {}
    for name, top in p["candidates"].items():
        workdir = stage / "forces" / name
        entry = {"n_frames_requested": len(frames)}
        rerun = rerun_forces(p["gmx"], top, p["structure"], p["rerun_mdp"], sub_trr,
                             p.get("index"), workdir, p["ntomp"], log)
        if rerun is None:
            entry["error"] = "grompp/mdrun -rerun failed (see validate_driver.log)"
            out[name] = entry
            continue
        qm_region = QMRegion(top, p["structure"], gmxdata=p.get("gmx_data"))
        qm_region.setup_qm_region(Path(p["selection"]))
        header, rows = force_residual_rows(rerun, p["structure"], fmdata, qm_region, frames)
        csv_path = stage / f"force_residuals_{name}.csv"
        csv_path.write_text(header + "\n" + "\n".join(rows) + "\n")
        parsed = parse_validation_force_residuals(csv_path.read_text())
        parsed.pop("worst_atoms", None)
        entry.update(parsed)
        entry["file"] = csv_path.name
        out[name] = entry
    return out


def main(params_path: str) -> int:
    p = json.loads(Path(params_path).read_text())
    stage = Path(p["stage"])
    log = stage / "validate_driver.log"
    result: dict = {"ok": False}
    try:
        pairs = {k: tuple(v) for k, v in p["monitor_pairs"].items()}
        p["monitor_pairs"] = pairs
        ref, n_frames = reference_stats(
            p["structure"], p["trajectory"], pairs, p["reference_stride"])
        result["reference"] = ref
        result["reference_frames"] = n_frames
        if not p["min_frames"]:
            k = int(p["n_min_frames"])
            lo = n_frames // 10
            p["min_frames"] = [int(i) for i in
                               np.linspace(lo, n_frames - 1, k).astype(int)]
        result["min_frames"] = p["min_frames"]
        if "min_geometry" in p["metrics"]:
            result["min_geometry"] = metric_min_geometry(p, ref, stage, log)
        if "md_geometry" in p["metrics"]:
            result["md_geometry"] = metric_md_geometry(p, ref, log)
        if "force_rmse" in p["metrics"]:
            if not p.get("force_frames"):
                k = int(p.get("n_force_frames", 50))
                p["force_frames"] = [int(i) for i in
                                     np.linspace(0, n_frames - 1, min(k, n_frames)).astype(int)]
            result["force_frames"] = p["force_frames"]
            result["force_rmse"] = metric_force_rmse(p, stage, log)
        result["ok"] = True
    except Exception as exc:  # surfaced when job_status collects the result
        result["error"] = f"{type(exc).__name__}: {exc}"
    (stage / p["out"]).write_text(json.dumps(result, indent=1) + "\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
