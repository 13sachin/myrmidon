"""MiMiCPy force matching, read: its logs, fit results and residual tables parsed into
typed facts, and the checks a fitted topology is held to. The numbers come from
MiMiCPy's output, never from the model."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from pathlib import Path

from myrmidon.core.contracts import declare_alias, define_kind
from myrmidon.core.tool import ToolContext
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.gromacs.itp import atom_charges, bond_angle_rows

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


def rerun_mdp_warnings(mdp: dict) -> list[Finding]:
    """What a force-matching / force-comparison rerun mdp must not be: a MiMiC production
    mdp (FM_RERUN_MDP_IS_MIMIC below), or one that writes no forces. fm_fit's gate."""
    warnings: list[Finding] = []
    qmmm_keys = [k for k in mdp if k.replace("-", "_") == "qmmm_grps"]
    if mdp.get("integrator", "").lower() == "mimic" or qmmm_keys:
        warnings.append(
            Precondition(
                id="FM_RERUN_MDP_IS_MIMIC",
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
            Precondition(
                id="FM_MDP_NO_FORCES",
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


def parse_fminput(path: Path) -> dict[str, str]:
    """An ``fminput.dat`` back as key -> value as written, comments off."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if ":" in text:
            key, _, value = text.partition(":")
            out[key.strip()] = value.strip()
    return out


declare_alias("fminput", "fm_input")
#: a force-matching dataset is this engine's reference dataset
declare_alias("fm_dataset", "reference_dataset")
define_kind("fm_input", parse_fminput,
            lookup=lambda values, key: values.get(key))


def stride_count(values: dict) -> int | None:
    """How many snapshots a fit's ``stride`` selects; None when it states none readably."""
    try:
        first, last, step = (int(v) for v in str(values.get("stride")).strip("[] ").split(","))
    except ValueError:
        return None
    return len(range(first, last, step))


def dataset_facts(ctx: ToolContext, dataset: Path) -> dict:
    """What a fit reads from its reference dataset: every frame it holds, as the ``stride`` to
    fit (mimicpy h5info); none when its frames cannot be counted."""
    import subprocess

    from myrmidon.core.artifacts import sha256_file
    from myrmidon.engines.mimic.qm_region import mimicpy_bin
    from myrmidon.knowledge.catalog import SystemFact

    proc = subprocess.run(  # noqa: S603
        [mimicpy_bin(ctx), "h5info", "-i", str(dataset)], capture_output=True, text=True,
        timeout=600, cwd=dataset.parent)
    frames = parse_h5info(proc.stdout + proc.stderr).get("frames")
    if not frames:
        return {}
    return {"stride": SystemFact(key="stride", value=f"[0,{frames},1]", rule="reference_dataset",
                                 read=[str(dataset)], sha256=sha256_file(dataset))}


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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gmx_data_flags(ctx: ToolContext) -> list[str]:
    info = ctx.engines.get("gromacs")
    if info is not None and info.data_dir:
        return ["-ff", info.data_dir]
    return []


# -- dataset_collect ---------------------------------------------------------------


# -- fm_plan ------------------------------------------------------------------


# -- fm_fit -------------------------------------------------------------------


_SECTION_RE = re.compile(r"^\s*\[\s*(\w+)\s*\]", re.MULTILINE)


def itp_atomtypes_warnings(topology: Path) -> list[Finding]:
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
    return [Finding(
        id="FM_ITP_DEFINES_ATOMTYPES",
        message=f"{', '.join(found)} define(s) [ atomtypes ] beside a [ moleculetype ]. If "
        "that molecule is in the QM region, mimicpy rewrites its .itp without the atom "
        "types and the classical rerun fails ('Atomtype ... not found') after the charge "
        "stage.",
        suggestion="move [ atomtypes ] into the .top before the #include, or into a "
        "separate include, unless the molecule is not in the QM region",
    )]


# -- fm_results ---------------------------------------------------------------


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


def includes_of(top: Path) -> list[str]:
    return re.findall(r'^\s*#include\s+"([^"]+)"', top.read_text(errors="replace"), re.MULTILINE)


def fitted_pairs(top: Path, opt_itps: list[Path]) -> dict[str, Path]:
    """include name -> fitted itp, for every opt_<name>.itp whose original the topology
    includes (mimicpy names a fitted file after the one it read)."""
    pairs: dict[str, Path] = {}
    for fitted in opt_itps:
        name = fitted.name[len("opt_"):]
        inc = next((i for i in includes_of(top) if Path(i).name == name), None)
        if inc:
            pairs[inc] = fitted
    return pairs


# -- dataset_collect ----------------------------------------------------------


# -- dataset_collect (combining several runs' reference data) ------------------------


# -- snapshot_select (displacement single-point reference windows) -----------


def force_type_table(blocks: dict[str, dict]) -> str:
    """Per-atom-type force RMSE, topologies side by side (rows sorted by the first one's
    RMSE; the reference RMS is the same for all)."""
    names = [n for n, b in blocks.items() if b.get("by_type")]
    if not names:
        return ""
    per = {n: {t["type"]: t for t in blocks[n]["by_type"]} for n in names}
    types = [t["type"] for t in blocks[names[0]]["by_type"]]
    lines = [f"{'type':10s} {'n':>4s} {'ref RMS':>10s}" + "".join(f" {n[:14]:>14s}" for n in names)]
    for t in types:
        first = per[names[0]][t]
        row = f"{t:10s} {first['n_atoms']:>4d} {first['ref_rms']:>10.4f}"
        for n in names:
            v = per[n].get(t)
            row += f" {v['rmse']:>14.4f}" if v else f" {'—':>14s}"
        lines.append(row)
    row = f"{'overall':10s} {'':>4s} {blocks[names[0]].get('ref_rms', 0.0):>10.4f}"
    for n in names:
        row += f" {blocks[n]['rmse']:>14.4f}"
    lines.append(row)
    return "\n".join(lines)


def force_residual_rows(forces, structure, fmdata, qm_region, frames: list[int]):
    """Per (frame, QM atom) rows of ``F_topology(MM) - F_reference(QM/MM)`` on the QM atoms,
    atomic units, in the validation_force_residuals.csv layout mimicpy writes.

    Reference forces come from the dataset (CPMD order, mapped to topology order with
    mimicpy's own build_fm_to_top_order); the topology's forces from a rerun trajectory
    whose frame k is dataset configuration ``frames[k]`` (row g-1 of GROMACS atom g,
    converted with mimicpy's kjmolnm_to_au, the conversion the fit uses). The total-force
    analogue of the fit's bonded-force residual, for any topology, stock or fitted."""
    import MDAnalysis as mda
    import numpy as np
    from mimicpy.force_matching.frame_consistency import build_fm_to_top_order
    from mimicpy.utils.constants import kjmolnm_to_au

    order = build_fm_to_top_order(fmdata, qm_region, frames[0])
    table = qm_region.qm_atoms
    rows_idx = np.array([int(g) - 1 for g in table.index], dtype=int)
    u = mda.Universe(str(structure), str(forces))
    if len(u.trajectory) < len(frames):
        raise ValueError(f"the rerun trajectory has {len(u.trajectory)} frames, expected "
                         f"{len(frames)}")
    col = lambda name, default: (list(table[name]) if name in table.columns  # noqa: E731
                                 else [default] * len(table))
    names, types, resnames = col("name", ""), col("type", ""), col("resname", "")
    resids, elements, bound = col("resid", 0), col("element", ""), col("is_bound", 0)
    rows = []
    for k, f in enumerate(frames):
        ts = u.trajectory[k]
        if not ts.has_forces:
            raise ValueError("the rerun trajectory carries no forces: nstfout must be 1 in "
                             "the rerun mdp")
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

