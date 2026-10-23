"""How a scheduler charges, read from the scheduler.

The price of a booking needs two facts per partition: how many physical cores a node has
(what a request is rounded up to when nodes are allocated whole) and what one node-hour
is billed. Both are the site's, and both are stated by SLURM itself:

- ``sinfo -o "%P|%X|%Y|%Z|%c|%G"``: per node shape, sockets × cores per socket (physical
  cores), threads per core, CPUs (hardware threads) and generic resources (``gpu:4``);
- ``scontrol show partition --oneliner``: ``OverSubscribe=EXCLUSIVE`` (nodes allocated
  whole), ``TRESBillingWeights`` (what each CPU and GPU is charged), ``Default=YES``.

A node-hour is billed CPUs × the CPU weight + GPUs × the GPU weight; with no weights SLURM
bills one per CPU. A site that counts hardware threads at 0.5 (2 × 24 cores, 96 CPUs,
``CPU=0.5``) bills 48 per node-hour, which is why neither the CPU count nor the core count
alone is the price. A partition mixing node shapes is priced on its fewest cores per node
and its most expensive node: an upper bound, as every estimate here is. A weight on
anything else (memory, licences) is not priced and is said so in the partition's ``note``.

Nothing reads the scheduler at pricing time: ``myrmidon init`` writes what this module
reads into the machine config, so a price is reproducible, works offline and replays.
"""

from __future__ import annotations

import re
import shutil
import subprocess

from myrmidon.config import PartitionCompute

SINFO_ARGS = ["sinfo", "-h", "-o", "%P|%X|%Y|%Z|%c|%G"]
SCONTROL_ARGS = ["scontrol", "show", "partition", "--oneliner"]

_GPUS = re.compile(r"(?:^|,)gpu(?::[^:,()]+)?:(\d+)")
_FIELD = re.compile(r"(\w+)=(\S*)")


def _weights(text: str) -> dict[str, float]:
    """``CPU=0.5,Mem=0.25G,GRES/gpu=2.0`` -> {"cpu": 0.5, "mem": 0.25, "gres/gpu": 2.0};
    SLURM reads TRES names without regard to case."""
    out = {}
    for item in filter(None, text.split(",")):
        key, _, value = item.partition("=")
        number = re.match(r"[\d.]+", value)
        if number:
            out[key.lower()] = float(number.group())
    return out


def parse_partitions(sinfo: str, scontrol: str) -> tuple[dict[str, PartitionCompute], str | None]:
    """(partition -> how it charges, the scheduler's default partition) from the output of
    :data:`SINFO_ARGS` and :data:`SCONTROL_ARGS`. A partition either command leaves out is
    left out."""
    shapes: dict[str, list[tuple[int, int, int]]] = {}  # name -> [(cores, cpus, gpus)]
    default = None
    for line in sinfo.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5:
            continue
        name, sockets, cores, _threads, cpus = parts[:5]
        if name.endswith("*"):
            name = default = name.rstrip("*")
        try:
            physical = int(sockets) * int(cores)
            cpu_count = int(cpus)
        except ValueError:
            continue
        gpus = _GPUS.search(parts[5]) if len(parts) > 5 else None
        shapes.setdefault(name, []).append((physical, cpu_count, int(gpus.group(1)) if gpus else 0))

    out: dict[str, PartitionCompute] = {}
    for line in scontrol.splitlines():
        fields = dict(_FIELD.findall(line))
        name = fields.get("PartitionName")
        if not name or name not in shapes:
            continue
        if fields.get("Default") == "YES":
            default = name
        weights = _weights(fields.get("TRESBillingWeights", ""))
        cpu_weight = weights.pop("cpu", 1.0 if not weights else 0.0)
        gpu_weight = weights.pop("gres/gpu", 0.0)
        out[name] = PartitionCompute(
            cores_per_node=min(cores for cores, _, _ in shapes[name]),
            billed_per_node_h=max(cpus * cpu_weight + gpus * gpu_weight
                                  for _, cpus, gpus in shapes[name]),
            node_exclusive=fields.get("OverSubscribe", "").upper().startswith("EXCLUSIVE"),
            note=("billing weights not priced: " + ", ".join(sorted(weights)))
            if weights else "",
        )
    return out, default


def read_partitions(timeout: float = 20.0) -> tuple[dict[str, PartitionCompute], str | None]:
    """What this machine's scheduler says, or ({}, None) when there is none to ask."""
    if not (shutil.which("sinfo") and shutil.which("scontrol")):
        return {}, None
    try:
        sinfo, scontrol = (
            subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                           check=True).stdout
            for args in (SINFO_ARGS, SCONTROL_ARGS))
    except (OSError, subprocess.SubprocessError):
        return {}, None
    return parse_partitions(sinfo, scontrol)
