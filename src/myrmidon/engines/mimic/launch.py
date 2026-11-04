"""A coupled run's working directory: the CPMD paths aligned to it, and the restart and
the geometry staged into it."""

from __future__ import annotations

#: CPMD reads its restart from ``RESTART`` and writes the next one to ``RESTART.1``.
#: Two names, not one: segment N's RESTART.1 and segment N+1's RESTART are the same bytes,
#: while N+1's own RESTART.1 is different. A continuation needs the predecessor's RESTART.1
#: under the name CPMD *reads*; a copy under the name it *writes* leaves nothing to read,
#: and CPMD stops in ZHRWF with FILE NOT FOUND before the first step.
RESTART_READ_NAME = "RESTART"


def align_mimic_paths(inp, stage):
    """Make the input's ``&MIMIC PATHS`` name the directory this run will execute in.

    PATHS is not a parameter — it is how MiMiC's two programs find each other, and it
    has to be the canonical absolute path of the directory mdrun runs in (MIMIC_PATHS_NOT_CWD).
    An input carried anywhere else still names where it was made: a later segment of a
    chain in its own stage, or a copy planted in a new workspace. Both fail as an MPI
    coupling error that names the symptom, not the cause.

    Returns ``(input_to_run, note)``. The original is never edited: an aligned copy is
    written into the stage, with its ``.myrmidon.json`` sidecar beside it so what the
    ledger hashes is what ran and its confirmed values travel with it.
    """
    import shutil

    from myrmidon.engines.mimic.cpmd_input import parse_cpmd, paths_dir, render_cpmd, set_paths

    here = str(stage.resolve())
    sections = parse_cpmd(inp.read_text())
    named = paths_dir(sections)
    if named is None or named == here:
        return inp, ""

    target = stage / inp.name
    if target.resolve() == inp.resolve():
        target = stage / f"{inp.stem}.aligned{inp.suffix}"
    target.write_text(render_cpmd(set_paths(sections, here)))
    sidecar = inp.with_name(inp.name + ".myrmidon.json")
    if sidecar.is_file():
        shutil.copy2(sidecar, target.with_name(target.name + ".myrmidon.json"))
    return target, f"&MIMIC PATHS re-pointed from {named} to this run's directory"


def stage_restart(restart, stage):
    """Put the previous segment's restart where CPMD looks for it.

    The chain is otherwise invisible: nothing in a CPMD input names its predecessor,
    which is why a hand-run campaign's segment order survives only as directory
    numbering. Here the link is the ``restart_from`` parameter, so the registry records
    it as a hashed input and the ledger carries the edge.

    No ``LATEST`` is written. CPMD writes one at the end of a run naming what it wrote;
    planting one that names a file this segment has not written yet points CPMD at a
    file that does not exist. Returns the staged file, which the node reads.
    """
    import shutil

    target = stage / RESTART_READ_NAME
    if restart.resolve() != target.resolve():
        shutil.copy2(restart, target)
    return target


GEOMETRY_NAME = "GEOMETRY"


def stage_geometry(geometry, stage):
    """Put the geometry where GEOFILE looks for it, as ``stage_restart`` does the restart;
    returns the staged file."""
    import shutil

    target = stage / GEOMETRY_NAME
    if geometry.resolve() != target.resolve():
        shutil.copy2(geometry, target)
    return target
