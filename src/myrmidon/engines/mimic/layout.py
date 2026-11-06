"""How a coupled run divides the cores it is given. CPMD distributes its real-space mesh over
ranks by planes along x, so it can use no more ranks than the mesh has planes; GROMACS takes one
rank in six, a count its domain decomposition can factor, and never more than its atoms can use."""

from __future__ import annotations

import math
from collections.abc import Mapping

from myrmidon.engines.gromacs.layout import fact_number, smooth_at_most
from myrmidon.engines.registry import Layout
from myrmidon.knowledge.catalog import load


def fft_planes(cell_bohr: float, cutoff_ry: float, primes: list[int] | None = None) -> int:
    """The planes of CPMD's density mesh along a cell edge at a wavefunction cutoff: the mesh
    holds G vectors up to twice the wavefunction's, so a/pi x 2 sqrt(E) points, rounded up to
    an even count its FFT can factor (REAL SPACE MESH in tests/fixtures/cpmd)."""
    primes = primes or [2, 3, 5, 7]
    n = math.ceil(cell_bohr * 2 * math.sqrt(cutoff_ry) / math.pi)
    while True:
        if n % 2 == 0 and smooth_at_most(n, primes) == n:
            return n
        n += 1


def resolve_layout(cores: int, facts: Mapping) -> Layout:
    """``cores`` as CPMD ranks and GROMACS ranks, threads per rank the fewest that keep CPMD's
    ranks within its mesh's planes along x."""
    rules = load("cpmd").decomposition
    primes = rules.get("fft_primes", [2, 3, 5, 7])
    cell = getattr(facts.get("cell_bohr"), "value", facts.get("cell_bohr"))
    cutoff = fact_number(facts, "cutoff_ry")
    first = fact_number({"edge": cell}, "edge")
    planes = fft_planes(first, cutoff, primes) if first and cutoff else None
    atoms = fact_number(facts, "atoms")
    floor = load("gromacs").decomposition.get("atoms_per_rank", 500)
    allowed = [t for t in rules.get("thread_counts", [1]) if t <= max(1, cores // 2)] or [1]
    layout = None
    for threads in allowed:
        ranks = cores // threads
        mm = max(1, ranks // rules.get("mm_share", 6))
        if atoms:
            mm = min(mm, max(1, int(atoms // floor)))
        mm = smooth_at_most(mm, rules.get("dd_primes", primes))
        qm = max(1, ranks - mm)
        layout = Layout(qm, threads, mm, why=(
            f"CPMD {qm} + GROMACS {mm} ranks of {threads} thread(s) on {cores} cores"
            + (f"; the mesh has {planes} planes along x" if planes else "")))
        if planes is None or qm <= planes:
            return layout
    # more cores than even the most threads can use: the mesh's planes bound the ranks
    return Layout(planes, layout.cpus_per_task, layout.partner_ntasks, why=(
        f"CPMD {planes} + GROMACS {layout.partner_ntasks} ranks of {layout.cpus_per_task} "
        f"thread(s): the mesh has {planes} planes along x, and the rest of {cores} cores stay "
        "idle"))
