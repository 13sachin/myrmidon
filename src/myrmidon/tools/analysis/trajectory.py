"""Trajectory metrics via MDAnalysis (optional dependency, imported lazily).

Full series are saved as .npz artifacts; the observation carries the summary
statistics the harness computed from them, and the choices that define the metric: what
the frames were superposed on and to, how periodic boundaries were treated, which frames
were analysed. A fluctuation is internal only after the rigid-body motion is removed, so
``rmsd`` and ``rmsf`` superpose every frame on the fit selection before measuring; a
molecule split across the box is made whole only when asked, since that needs bonds, and
the tool refuses rather than guess when the structure file carries none."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from myrmidon.analysis.trajectory import has_bonds
from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext
from myrmidon.core.validation import Finding
from myrmidon.engines.inputs import resolve_in_ws

#: what each metric is a function of
_AXIS = {"rmsd": "time_ps", "rmsf": "atom_index", "rg": "time_ps", "rdf": "r_angstrom",
         "mindist": "time_ps", "pair_distance": "time_ps", "contacts": "resid"}


class TrajParams(BaseModel):
    deliverable: str | None = Field(
        default=None, description="a deliverable the mandate names: measured as approved — "
        "its selections, superposition and statistics compiled from the mandate, over the "
        "trajectory its stage's run wrote and that run's .tpr, found in the ledger")
    structure: str | None = Field(default=None, description="topology-bearing file "
                                  "(.tpr/.gro/.pdb); a deliverable's is its run's .tpr")
    trajectory: str | None = Field(default=None, description="trajectory file (.xtc/.trr); "
                                   "a deliverable's is its stage's run's")
    metric: Literal["rmsd", "rmsf", "rg", "rdf", "mindist", "pair_distance", "contacts"] | None \
        = Field(default=None, description="what to measure; a deliverable's is its own")
    selection: str = Field(
        default="protein",
        description="MDAnalysis selection the metric is computed over (e.g. 'protein and "
        "backbone' or 'resname JZ4 and not name H*'; never use 'heavy')",
    )
    selection2: str = Field(
        default="", description="second selection (rdf, mindist and contacts)"
    )
    cutoff_angstrom: float = Field(
        default=4.0, gt=0.0,
        description="contacts: heavy-atom or atom-pair distance cutoff in Å (default 4.0 Å)",
    )
    fit_selection: str = Field(
        default="", description="rmsd/rmsf: the atoms every frame is superposed on before "
        "measuring (default: the selection itself); the fluctuation reported is internal to "
        "this superposition")
    reference_frame: int = Field(
        default=0, ge=0, description="rmsd/rmsf: the frame the others are superposed to")
    unwrap: bool = Field(
        default=False, description="make molecules whole across periodic boundaries before "
        "the metric (needs bonds: a .tpr or a .pdb with CONECT records; refused otherwise). "
        "Without it a molecule split by the box is measured as split.")
    pairs: dict[str, tuple[int, int]] = Field(
        default_factory=dict,
        description="pair_distance: label -> (atom, atom), 1-based serials as in the structure "
        "file, e.g. {'C1-O1': [1, 2]}")
    reference: str | None = Field(
        default=None,
        description="pair_distance: a second trajectory of the same atoms (a reference "
        "ensemble); each pair's mean there, the difference, and the mean absolute difference "
        "over the pairs are reported beside this trajectory's")
    start_ps: float | None = Field(default=None, description="analyse frames from this time")
    stop_ps: float | None = Field(default=None, description="analyse frames up to this time")


def _hydrogen_facts(atoms, g2) -> dict:
    """How many hydrogens each selection holds, and how many its own residues carry: a
    distance from every atom of one side to the heavy atoms of the other is neither a
    heavy-atom distance nor an all-atom one, which the agent reads from these."""
    return {"selection_hydrogens": [_hydrogens(atoms), _hydrogens(g2)],
            "residue_hydrogens": [_hydrogens(atoms.residues.atoms),
                                  _hydrogens(g2.residues.atoms)]}


def _molecules(u, group):
    """The whole molecules (bonded fragments) the atoms of ``group`` belong to."""
    return u.atoms[np.isin(u.atoms.fragindices, np.unique(group.fragindices))]


def _hydrogens(group) -> int:
    """How many of a group's atoms are hydrogens, by element where the topology has them,
    else by name."""
    try:
        return int(sum(1 for e in group.elements if str(e).strip().upper() == "H"))
    except Exception:  # noqa: BLE001 — a topology without elements
        return int(sum(1 for n in group.names if str(n).lstrip("0123456789").startswith("H")))


def _selection_error(selection: str, exc: Exception) -> str:
    hint = ""
    if "heavy" in selection.lower():
        hint = " (in MDAnalysis, select heavy atoms with 'not name H*', never 'heavy')"
    return f"invalid MDAnalysis selection {selection!r}: {exc}{hint}"


def measure_trajectory(ctx: ToolContext, params) -> Observation:
    """A trajectory metric, the series saved and its statistics computed by the harness. A
    run input the reader cannot take as a topology (one newer than it knows) is measured with
    its atoms and bonds as its engine writes them from the trajectory's first frame."""
    import MDAnalysis as mda

    structure = resolve_in_ws(ctx, params.structure)
    try:
        mda.Universe(str(structure))
    except Exception:  # noqa: BLE001 — any reader failure is a file it cannot take
        bonded = _with_bonds(ctx, structure, resolve_in_ws(ctx, params.trajectory),
                             ctx.workspace.stage("analysis"))
        if bonded is not None:
            written = ctx.workspace.relpath(bonded)
            obs = _analyze(ctx, params.model_copy(update={"structure": written}))
            obs.findings = [Finding(
                id="TRAJECTORY_STRUCTURE_WRITTEN",
                message=f"{params.structure} cannot be read as a topology here: measured with "
                f"its atoms and bonds as its engine writes them, {written}"), *obs.findings]
            return obs
    return _analyze(ctx, params)


def _with_bonds(ctx: ToolContext, run_input, trajectory, stage):
    """The run input's atoms and bonds from the trajectory's first frame, as the engine that
    reads it writes them; None where no engine can."""
    from myrmidon.core.artifacts import fresh_path
    from myrmidon.engines.registry import adapter_with

    engine = adapter_with("bonded_structure")
    if engine is None:
        return None
    return engine.bonded_structure(ctx, run_input, trajectory,
                                   fresh_path(stage / f"{run_input.stem}_bonded.pdb"))


def _analyze(ctx: ToolContext, params: TrajParams) -> Observation:
    try:
        import MDAnalysis as mda
    except ImportError:
        return Observation(
            ok=False,
            summary="MDAnalysis is not installed — install myrmidon[analysis]",
            metric=params.metric,
        )
    from MDAnalysis import transformations as trans

    stage = ctx.workspace.stage("analysis")
    structure = resolve_in_ws(ctx, params.structure)
    trajectory = resolve_in_ws(ctx, params.trajectory)
    u = mda.Universe(str(structure), str(trajectory))
    metric = params.metric

    def refuse(why: str) -> Observation:
        return Observation(ok=False, summary=why, metric=metric,
                                     axis=_AXIS[metric], selection=params.selection)

    if metric in ("rdf", "mindist", "contacts") and not params.selection2:
        return refuse(f"{metric} needs selection2")
    if metric == "pair_distance":
        return _pair_distance(ctx, params, u, structure, trajectory)
    try:
        atoms = u.select_atoms(params.selection)
    except Exception as exc:  # noqa: BLE001
        return refuse(_selection_error(params.selection, exc))
    if len(atoms) == 0:
        return refuse(f"selection matched no atoms ({params.selection!r})")
    g2 = None
    if metric in ("rdf", "mindist", "contacts"):
        try:
            g2 = u.select_atoms(params.selection2)
        except Exception as exc:  # noqa: BLE001
            return refuse(_selection_error(params.selection2, exc))
        if len(g2) == 0:
            return refuse(f"selection2 matched no atoms ({params.selection2!r})")
    # what each side of a pair holds, for reading the number
    facts = _hydrogen_facts(atoms, g2) if g2 is not None else {}
    n_total = len(u.trajectory)
    if params.reference_frame >= n_total:
        return refuse(f"reference_frame {params.reference_frame} is past the trajectory's "
                      f"{n_total} frames")

    # the frames analysed: an index window from the times asked for
    times = np.array([ts.time for ts in u.trajectory])
    inside = np.ones(n_total, dtype=bool)
    if params.start_ps is not None:
        inside &= times >= params.start_ps
    if params.stop_ps is not None:
        inside &= times <= params.stop_ps
    if not inside.any():
        return refuse(f"no frame between {params.start_ps} and {params.stop_ps} ps: the "
                      f"trajectory spans {times[0]:g} to {times[-1]:g} ps")
    start, stop = int(np.argmax(inside)), int(n_total - np.argmax(inside[::-1]))

    # every frame is read through the same preprocessing, recorded as it is applied
    workflow, transformations = [], []
    fit_selection = reference = ""
    fit = None
    if metric in ("rmsd", "rmsf"):
        fit_selection = params.fit_selection or params.selection
        try:
            fit = u.select_atoms(fit_selection)
        except Exception as exc:  # noqa: BLE001
            return refuse(_selection_error(fit_selection, exc))
        if len(fit) == 0:
            return refuse(f"fit_selection matched no atoms ({fit_selection!r})")
    # a minimum distance is taken under the minimum image: whole molecules change nothing
    unwrap = params.unwrap and metric != "mindist"
    if params.unwrap and not unwrap:
        transformations.append("no unwrap: a minimum distance is taken under the minimum "
                               "image, whatever the box split")
    if unwrap and not has_bonds(u):
        return refuse(f"unwrap needs bonds and {params.structure} carries none: give the "
                      "run's .tpr (or a .pdb with CONECT records) as structure, or "
                      "unwrap=false to measure molecules as the box split them")
    # the molecules measured, and those the fit is on: made whole, and — for a fit — the
    # fit group centred with the other molecules wrapped around it as whole molecules, so
    # a ligand is measured against the pocket it sits in, not a periodic image of it
    measured = atoms | g2 if g2 is not None else atoms
    groups = measured | fit if fit is not None else measured
    if unwrap:
        whole = _molecules(u, groups)
        workflow.append(trans.unwrap(whole))
        transformations.append(
            "unwrap: the molecules of the selection" + (" and the fit" if fit is not None
                                                        else "")
            + " made whole across periodic boundaries")
        if fit is not None:
            workflow += [trans.center_in_box(fit), trans.wrap(whole, compound="fragments")]
            transformations.append(f"centred on {fit_selection!r}, the measured molecules "
                                   "wrapped around it whole")
    if fit is not None:
        ref = mda.Universe(str(structure), str(trajectory))
        ref_fit = ref.select_atoms(fit_selection)
        if unwrap:
            ref_whole = _molecules(ref, ref.atoms[groups.indices])
            ref.trajectory.add_transformations(
                trans.unwrap(ref_whole), trans.center_in_box(ref_fit),
                trans.wrap(ref_whole, compound="fragments"))
        ref.trajectory[params.reference_frame]
        workflow.append(trans.fit_rot_trans(fit, ref_fit))
        reference = f"frame {params.reference_frame} (t = {times[params.reference_frame]:g} ps)"
        transformations.append(f"superposition: translation and rotation fitted on "
                               f"{fit_selection!r} to {reference}")
    if workflow:
        u.trajectory.add_transformations(*workflow)

    if metric == "contacts":
        obs = _contacts(ctx, params, u, atoms, g2, times, start, stop, stage, trajectory,
                        transformations)
        for key, value in facts.items():
            setattr(obs, key, value)
        return obs

    x: np.ndarray
    y: np.ndarray
    unit = "Å"
    if metric == "rmsd":
        from MDAnalysis.analysis import rms

        # the frames are already superposed by the workflow; RMSD's own superposition would
        # fit on `selection`, not on fit_selection
        ref_atoms = ref.select_atoms(params.selection)
        frames = []
        for ts in u.trajectory[start:stop]:
            frames.append((ts.time, float(rms.rmsd(atoms.positions, ref_atoms.positions,
                                                   superposition=False))))
        x, y = np.array([f[0] for f in frames]), np.array([f[1] for f in frames])
    elif metric == "rmsf":
        from MDAnalysis.analysis import rms

        result = rms.RMSF(atoms).run(start=start, stop=stop)
        x = np.array(atoms.indices, dtype=float)
        y = result.results.rmsf
    elif metric == "rg":
        frames = [(ts.time, atoms.radius_of_gyration()) for ts in u.trajectory[start:stop]]
        x, y = np.array([f[0] for f in frames]), np.array([f[1] for f in frames])
    elif metric == "rdf":
        from MDAnalysis.analysis.rdf import InterRDF

        result = InterRDF(atoms, g2).run(start=start, stop=stop)
        x, y = result.results.bins, result.results.rdf
        unit = "g(r)"
        transformations.append("minimum image: pair distances under the frame's periodic box")
    else:  # mindist
        from MDAnalysis.lib import distances as libdist

        frames = []
        for ts in u.trajectory[start:stop]:
            # Fast capped distance search first (15 A), falling back to full distance_array
            pairs, dists = libdist.capped_distance(
                atoms.positions, g2.positions, max_cutoff=15.0, box=ts.dimensions,
                return_distances=True,
            )
            if len(dists) > 0:
                dmin = float(dists.min())
            else:
                d = libdist.distance_array(atoms.positions, g2.positions, box=ts.dimensions)
                dmin = float(d.min())
            frames.append((ts.time, dmin))
        x, y = np.array([f[0] for f in frames]), np.array([f[1] for f in frames])
        transformations.append("minimum image: pair distances under the frame's periodic box")

    if y.size == 0:
        return refuse(f"no values: the trajectory is empty ({params.trajectory})")

    from myrmidon.core.artifacts import fresh_path

    axis = _AXIS[metric]
    n_frames = stop - start
    window = [float(times[start]), float(times[stop - 1])]
    out = fresh_path(stage / f"{trajectory.stem}_{metric}.npz")
    np.savez(out, **{axis: x, metric: y})
    over = (f"{len(y)} atoms" if axis == "atom_index" else f"{len(y)} bins"
            if axis == "r_angstrom" else f"{len(y)} frames")
    min_val = float(np.min(y))
    max_val = float(np.max(y))
    min_t = float(x[int(np.argmin(y))]) if axis == "time_ps" else None
    max_t = float(x[int(np.argmax(y))]) if axis == "time_ps" else None
    stats = f"mean {float(np.mean(y)):.3f}, min {min_val:.3f}, largest {max_val:.3f}"
    if axis == "time_ps":
        stats = (f"mean {float(np.mean(y)):.3f}, min {min_val:.3f} at {min_t:g} ps, "
                 f"largest {max_val:.3f} at {max_t:g} ps")
    if axis == "r_angstrom":
        stats = f"peak {max_val:.3f} at r = {float(x[int(np.argmax(y))]):.2f} Å"
    elif axis == "time_ps":
        stats += f", final {float(y[-1]):.3f}"
    return Observation(
        ok=True,
        summary=f"{metric} of {params.selection!r}"
        + (f" with {params.selection2!r}" if g2 is not None else "")
        + (f", fitted on {fit_selection!r}" if metric in ("rmsd", "rmsf") else "")
        + f" over {over} ({n_frames} frames, "
        f"{window[0]:g}–{window[1]:g} ps): {stats} {unit}"
        + ("; " + "; ".join(transformations) if transformations else "; no preprocessing"),
        artifacts=[ArtifactRef.from_path(out, "data", root=ctx.workspace.root)],
        metric=metric,
        unit=unit,
        axis=axis,
        mean=float(np.mean(y)) if axis != "r_angstrom" else None,
        sd=float(np.std(y, ddof=1)) if axis != "r_angstrom" and len(y) > 1 else None,
        minimum=min_val if axis != "r_angstrom" else None,
        min_time_ps=min_t,
        final=float(y[-1]) if axis == "time_ps" else None,
        largest=max_val,
        max_time_ps=max_t,
        n_points=int(len(y)),
        n_frames=n_frames,
        time_window_ps=window,
        selection=params.selection,
        selection2=params.selection2 or None,
        fit_selection=fit_selection,
        reference=reference,
        transformations=transformations,
        **facts,
    )


def _contacts(ctx: ToolContext, params: TrajParams, u, atoms, g2, times: np.ndarray,
              start: int, stop: int, stage, trajectory, transformations: list[str]
              ) -> Observation:
    """Per-residue contact fraction between `atoms` and `g2` within `cutoff_angstrom`."""
    from MDAnalysis.lib import distances as libdist

    from myrmidon.core.artifacts import fresh_path
    from myrmidon.engines.structure import load_residue_map

    res_map = load_residue_map(ctx.workspace.root if ctx.workspace else None)
    n_frames = stop - start
    counts: dict[int, int] = {}
    min_dists: dict[int, float] = {}
    resnames: dict[int, str] = {}
    for a in g2:
        resnames.setdefault(int(a.resid), str(a.resname))

    g2_resids = np.array([int(a.resid) for a in g2])
    for ts in u.trajectory[start:stop]:
        pairs, dists = libdist.capped_distance(
            atoms.positions, g2.positions, max_cutoff=params.cutoff_angstrom,
            box=ts.dimensions, return_distances=True,
        )
        if len(pairs) == 0:
            continue
        hit_resids = g2_resids[pairs[:, 1]]
        for r in set(hit_resids.tolist()):
            counts[r] = counts.get(r, 0) + 1
        for (_, j), d in zip(pairs, dists, strict=False):
            r = int(g2_resids[j])
            if r not in min_dists or float(d) < min_dists[r]:
                min_dists[r] = float(d)

    rows = []
    for r in sorted(counts, key=lambda k: (-counts[k], k)):
        frac = round(counts[r] / max(n_frames, 1), 4)
        entry = {
            "resid": r,
            "resname": resnames.get(r, ""),
            "deposited": res_map.get(r, f"{resnames.get(r, '')}{r}"),
            "contact_fraction": frac,
            "min_distance_angstrom": round(min_dists[r], 3),
        }
        rows.append(entry)

    out = fresh_path(stage / f"{trajectory.stem}_contacts.npz")
    np.savez(
        out,
        resid=np.array([r["resid"] for r in rows], dtype=int),
        contact_fraction=np.array([r["contact_fraction"] for r in rows], dtype=float),
        min_distance_angstrom=np.array([r["min_distance_angstrom"] for r in rows], dtype=float),
    )
    window = [float(times[start]), float(times[stop - 1])]
    transformations.append(
        f"minimum image: contacts within {params.cutoff_angstrom:g} Å under periodic box"
    )
    top_str = ", ".join(
        f"{r['deposited']} (resid {r['resid']}: {r['contact_fraction']*100:.0f}%)"
        for r in rows[:8]
    ) or "none"
    return Observation(
        ok=True,
        summary=f"contacts of {params.selection!r} with {params.selection2!r} within "
        f"{params.cutoff_angstrom:g} Å over {n_frames} frames ({window[0]:g}–{window[1]:g} ps): "
        f"{len(rows)} contacting residue(s) — {top_str}",
        artifacts=[ArtifactRef.from_path(out, "data", root=ctx.workspace.root)],
        metric="contacts",
        unit="fraction",
        axis="resid",
        n_points=len(rows),
        n_frames=n_frames,
        time_window_ps=window,
        selection=params.selection,
        transformations=transformations,
        contacts=rows,
    )


def _pair_series(u, pairs: dict[str, tuple[int, int]], start: int = 0, stop: int | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(times, distances frames × pairs) in Å, each pair under the frame's periodic box."""
    from MDAnalysis.lib.distances import calc_bonds

    a = u.atoms[[i - 1 for i, _ in pairs.values()]]
    b = u.atoms[[j - 1 for _, j in pairs.values()]]
    times, rows = [], []
    for ts in u.trajectory[start:stop]:
        times.append(ts.time)
        rows.append(calc_bonds(a.positions, b.positions, box=ts.dimensions))
    return np.array(times), np.array(rows).reshape(len(rows), len(pairs))


def _pair_distance(ctx: ToolContext, params: TrajParams, u, structure, trajectory
                   ) -> Observation:
    """Named atom pairs per frame, and against a reference ensemble when one is given."""
    import MDAnalysis as mda

    from myrmidon.core.artifacts import fresh_path

    def refuse(why: str) -> Observation:
        return Observation(ok=False, summary=why, metric="pair_distance",
                                     axis="time_ps")

    if not params.pairs:
        return refuse("pair_distance needs pairs: label -> (atom, atom)")
    n_atoms = len(u.atoms)
    bad = sorted({i for pair in params.pairs.values() for i in pair if not 1 <= i <= n_atoms})
    if bad:
        return refuse(f"atom serial(s) {bad} outside 1..{n_atoms} of {params.structure}")
    times = np.array([ts.time for ts in u.trajectory])
    inside = np.ones(len(times), dtype=bool)
    if params.start_ps is not None:
        inside &= times >= params.start_ps
    if params.stop_ps is not None:
        inside &= times <= params.stop_ps
    if not inside.any():
        return refuse(f"no frame between {params.start_ps} and {params.stop_ps} ps")
    start, stop = int(np.argmax(inside)), int(len(times) - np.argmax(inside[::-1]))
    t, d = _pair_series(u, params.pairs, start, stop)
    labels = list(params.pairs)
    stats = {label: {"mean": float(d[:, k].mean()), "std": float(d[:, k].std())}
             for k, label in enumerate(labels)}
    arrays = {"time_ps": t, "pair_distance": d, "pairs": np.array(labels)}
    mae = None
    if params.reference:
        ref_u = mda.Universe(str(structure), str(resolve_in_ws(ctx, params.reference)))
        _, r = _pair_series(ref_u, params.pairs)
        for k, label in enumerate(labels):
            stats[label]["reference_mean"] = float(r[:, k].mean())
            stats[label]["reference_std"] = float(r[:, k].std())
            stats[label]["difference"] = stats[label]["mean"] - stats[label]["reference_mean"]
        mae = float(np.mean([abs(s["difference"]) for s in stats.values()]))
        arrays.update(reference_mean=r.mean(axis=0), reference_std=r.std(axis=0))
    stage = ctx.workspace.stage("analysis")
    out = fresh_path(stage / f"{trajectory.stem}_pair_distance.npz")
    np.savez(out, **arrays)
    listed = "; ".join(
        f"{label} {s['mean']:.3f} ± {s['std']:.3f}"
        + (f" (reference {s['reference_mean']:.3f}, {s['difference']:+.3f})"
           if "difference" in s else "") for label, s in stats.items())
    return Observation(
        ok=True,
        summary=f"pair_distance over {len(t)} frames ({t[0]:g}–{t[-1]:g} ps), Å: {listed}"
        + (f"; mean absolute difference from the reference {mae:.3f} Å" if mae is not None
           else ""),
        artifacts=[ArtifactRef.from_path(out, "data", root=ctx.workspace.root)],
        metric="pair_distance", unit="Å", axis="time_ps", n_frames=len(t),
        n_points=len(t), time_window_ps=[float(t[0]), float(t[-1])],
        transformations=["minimum image: pair distances under the frame's periodic box"],
        pairs=stats, mae=mae,
    )
