"""How a GROMACS run divides the cores it is given: as many MPI ranks as its atoms can use, the
rest as OpenMP threads, a rank count its domain decomposition can factor."""

from __future__ import annotations

from collections.abc import Mapping

from myrmidon.engines.registry import Layout
from myrmidon.knowledge.catalog import load


def fact_number(facts: Mapping, key: str) -> float | None:
    """A numeric system fact, by key; None when it is not there or not a number."""
    found = facts.get(key)
    value = getattr(found, "value", found)
    try:
        return float(str(value).split()[0]) if value is not None else None
    except (ValueError, IndexError):
        return None


def smooth_at_most(n: int, primes: list[int]) -> int:
    """The largest count not above ``n`` that factors entirely into ``primes``."""
    for candidate in range(max(1, n), 0, -1):
        left = candidate
        for prime in primes:
            while left % prime == 0:
                left //= prime
        if left == 1:
            return candidate
    return 1


def layout_facts(inputs: Mapping, stage=None) -> dict:
    """What a classical run's layout reads from its inputs: its atoms, and its integrator; a
    batch minimising frames is one driver process, however many atoms it holds."""
    from myrmidon.core.contracts import lookup
    from myrmidon.engines.gromacs.run import frame_batch
    from myrmidon.engines.gromacs.system import atom_count

    if stage is not None and stage.kind == "minimize" and frame_batch(stage):
        return {"integrator": "l-bfgs"}
    facts: dict = {}
    if inputs.get("structure") is not None and (found := atom_count(inputs["structure"])):
        facts["atoms"] = found
    if inputs.get("mdp") is not None:
        facts["integrator"] = lookup("mdp", inputs["mdp"], "integrator")
    return facts


def resolve_layout(cores: int, facts: Mapping) -> Layout:
    """``cores`` as ranks and threads: every core a rank up to the atoms-per-rank floor, the
    rest folded into threads (``-ntomp``); one rank for a minimiser that runs on one."""
    rules = load("gromacs").decomposition
    if str(getattr(facts.get("integrator"), "value", facts.get("integrator"))) in (
            rules.get("single_rank_integrators") or []):
        return Layout(1, cores, why=f"one rank of {cores} threads: the integrator runs on one")
    atoms = fact_number(facts, "atoms")
    floor = rules.get("atoms_per_rank", 500)
    most = max(1, int(atoms // floor)) if atoms else cores
    allowed = [t for t in rules.get("thread_counts", [1]) if t <= cores] or [1]
    threads = next((t for t in allowed if cores // t <= most), allowed[-1])
    ranks = smooth_at_most(cores // threads, rules.get("dd_primes", [2, 3, 5, 7]))
    why = (f"{ranks} ranks of {threads} thread(s) on {cores} cores"
           + (f": {int(atoms)} atoms use about one rank per {floor}" if atoms else ""))
    return Layout(ranks, threads, why=why)
