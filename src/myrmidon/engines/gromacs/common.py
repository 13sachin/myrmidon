"""Shared helpers for GROMACS tools."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from myrmidon.core.artifacts import fresh_path
from myrmidon.core.tool import ToolContext
from myrmidon.errors import EngineNotFoundError
from myrmidon.text import tail  # noqa: F401  (engines import it from here)


def gmx_bin(ctx: ToolContext) -> str:
    info = ctx.engines.get("gromacs")
    if info is None or not info.available:
        hint = info.hint if info is not None else ""
        raise EngineNotFoundError("gromacs", hint)
    return info.path


def run_gmx(
    ctx: ToolContext,
    args: list[str],
    cwd: Path,
    stdin: str | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Run a quick, synchronous gmx subcommand (prep/analysis — never mdrun)."""
    argv = [gmx_bin(ctx), *args]
    return subprocess.run(  # noqa: S603
        argv,
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def read_gro_info(path: Path) -> tuple[int, tuple[float, float, float] | None]:
    """Atom count and box vector from a .gro file (harness-parsed, never guessed)."""
    lines = path.read_text().splitlines()
    n_atoms = int(lines[1].strip())
    box = None
    box_fields = lines[-1].split()
    if len(box_fields) >= 3:
        box = (float(box_fields[0]), float(box_fields[1]), float(box_fields[2]))
    return n_atoms, box


def stage_topology(stage: Path, top: Path) -> Path:
    """Copy a topology and its same-directory includes into a stage, and return the copy.

    Two callers, one reason: a topology should be edited where the study can see it
    happen, not where it was planted.

    mimicpy fm writes rerun_<top>.top next to -top but generates the non-bonded .itp
    into the working directory, so unless the topology lives in the run directory the
    grompp rerun cannot resolve the include.

    gmx_solvate and gmx_genion edit the topology in place. Doing that to a planted file
    destroys the study's own starting point, and the input hash the ledger recorded no
    longer names a file that exists. Staging first leaves the planted original untouched.

    Relative includes shipped alongside the topology are staged too, or the copy's
    #include lines would not resolve from its new directory.
    """
    import shutil

    staged = stage / top.name
    if staged.resolve() != top.resolve():
        shutil.copy(top, staged)
        for line in top.read_text(errors="replace").splitlines():
            m = re.match(r'\s*#include\s+"([^"]+)"', line)
            if not m:
                continue
            inc = m.group(1)
            if "/" in inc:
                # directory-shipped force field (e.g. amber99sb-ildn.ff/forcefield.itp)
                top_dir = inc.split("/", 1)[0]
                src_dir = (top.parent / top_dir).resolve()
                if src_dir.is_dir() and not (stage / top_dir).exists():
                    shutil.copytree(src_dir, stage / top_dir)
            elif (top.parent / inc).exists():
                shutil.copy(top.parent / inc, stage / inc)
    return staged


def extract_energy_terms(ctx: ToolContext, edr: Path, terms: list[str], stage: Path
                         ) -> tuple[Path, subprocess.CompletedProcess]:
    """``gmx energy`` for the named terms of an energy file, into an .xvg under ``stage``
    named after the energy file and never overwritten, so a report re-reads the file the
    ledger recorded; (the .xvg, the process). The file exists only when the process
    succeeded."""
    xvg = fresh_path(stage / f"{edr.stem}_energy.xvg")
    proc = run_gmx(ctx, ["energy", "-f", str(edr), "-o", str(xvg)], cwd=stage,
                   stdin="\n".join(terms) + "\n\n")
    return xvg, proc


def join_trajectories(ctx: ToolContext, paths: list[Path], out: Path, continuous: bool = False
                      ) -> subprocess.CompletedProcess:
    """``gmx trjcat``: the trajectories one after another in the order given; the joined file
    exists only when the process succeeded. ``continuous``: the segments of one run, each
    re-timed to start where the one before it ended (``-settime``, ``c``), the frame they share
    at the join kept once; else every frame each holds kept with its own time (``-cat``)."""
    files = [str(p) for p in paths]
    if continuous:
        return run_gmx(ctx, ["trjcat", "-f", *files, "-settime", "-o", str(out)],
                       cwd=out.parent, stdin="0\n" + "c\n" * (len(files) - 1))
    return run_gmx(ctx, ["trjcat", "-f", *files, "-cat", "-o", str(out)], cwd=out.parent)


def bonded_structure(ctx: ToolContext, run_input: Path, structure: Path, out: Path
                     ) -> Path | None:
    """``gmx trjconv -conect``: ``structure``'s atoms (a structure, or a trajectory's first
    frame) as a .pdb with the bonds ``run_input`` holds, as CONECT records, for a reader that
    cannot take the run input itself; None when it could not be written."""
    if _atoms_in(structure) > 99_999:  # a PDB's serials wrap, and CONECT names wrong atoms
        return None
    frame = ["-dump", "0"] if structure.suffix in (".xtc", ".trr") else []
    proc = run_gmx(ctx, ["trjconv", "-s", str(run_input), "-f", str(structure), "-o", str(out),
                         "-conect", *frame], cwd=out.parent, stdin="0\n")
    return out if proc.returncode == 0 and out.is_file() else None


def _atoms_in(path: Path) -> int:
    """How many atoms a structure or a trajectory's frames hold; 0 when it cannot be read."""
    try:
        if path.suffix == ".gro":
            return read_gro_info(path)[0]
        import MDAnalysis as mda

        reader = {".xtc": mda.coordinates.XTC.XTCReader,
                  ".trr": mda.coordinates.TRR.TRRReader}.get(path.suffix)
        return reader(str(path)).n_atoms if reader else 0
    except Exception:  # noqa: BLE001 — a file no reader takes counts as none
        return 0
