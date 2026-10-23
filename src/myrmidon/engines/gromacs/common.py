"""Shared helpers for GROMACS tools."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from myrmidon.core.tool import ToolContext
from myrmidon.errors import EngineNotFoundError, MyrmidonError
from myrmidon.text import tail  # noqa: F401  (engines import it from here)


def stage_output(stage: Path, name: str) -> Path:
    """The path of an output a tool writes into its own stage.

    The tool already scopes outputs to the stage, so ``name`` is a bare file name. A
    stage-prefixed or nested name ("01_prep/boxed.gro") points into a directory that does
    not exist, and GROMACS reports it only as "File input/output error", which does not name
    the cause. Refused here, before the engine runs.
    """
    if Path(name).name != name:
        raise MyrmidonError(
            f"output {name!r} must be a bare file name: this tool writes into its own "
            f"stage ({stage.name}); pass {Path(name).name!r}"
        )
    return stage / name


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
    destroys the study's own starting point: the replay then plants the already-edited
    file and edits it again, giving a topology with the solvent counted twice. Staging
    first leaves the planted original untouched, so it is an input the twin can be given.

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
