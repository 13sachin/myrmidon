"""How a scheduler charges, read from the scheduler.

The price of a booking needs two facts per partition: how many physical cores a node has
(what a request is rounded up to when nodes are allocated whole) and what one node-hour
is billed. Both are the site's, and both are stated by SLURM itself:

- ``sinfo -o "%P|%X|%Y|%Z|%c|%G"``: per node shape, sockets × cores per socket (physical
  cores), threads per core, CPUs (hardware threads) and generic resources (``gpu:4``);
- ``scontrol show partition --oneliner``: ``OverSubscribe=EXCLUSIVE`` (nodes allocated
  whole), ``TRESBillingWeights`` (what each CPU and GPU is charged), ``Default=YES``,
  ``QoS`` (the partition's own quality of service);
- ``sacctmgr show qos`` and ``show assoc``: each QoS's longest job, and the QoS a job of this
  user carries on each partition. A site may state no time limit on its partitions at all
  (``sinfo %l`` says ``infinite``) and hold jobs to these instead, so a partition's limit is
  the least of its own, its QoS's and the job's.

A node-hour is billed CPUs × the CPU weight + GPUs × the GPU weight; with no weights SLURM
bills one per CPU. A site that counts hardware threads at 0.5 (2 × 24 cores, 96 CPUs,
``CPU=0.5``) bills 48 per node-hour, which is why neither the CPU count nor the core count
alone is the price. A partition mixing node shapes is priced on its fewest cores per node
and its most expensive node: an upper bound, as every estimate here is. A weight on
anything else (memory, licences) is not priced and is said so in the partition's ``note``.

Nothing reads the scheduler at pricing time: ``myrmidon init`` writes what this module
reads into the machine config, so a price is the same offline as on the cluster.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess

from myrmidon.config import PartitionCompute

SINFO_ARGS = ["sinfo", "-h", "-o", "%P|%X|%Y|%Z|%c|%G|%l"]
SCONTROL_ARGS = ["scontrol", "show", "partition", "--oneliner"]
QOS_ARGS = ["sacctmgr", "-n", "-p", "show", "qos", "format=name,maxwall"]
ASSOC_ARGS = ["sacctmgr", "-n", "-p", "show", "assoc", "user=$USER",
              "format=partition,defaultqos"]

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


def walltime_minutes(limit: str) -> int | None:
    """A scheduler's time limit in minutes: ``[days-]hours:minutes:seconds``, or ``minutes``
    alone; None for one stated as no limit (``infinite``, ``UNLIMITED``, empty)."""
    days, _, clock = limit.strip().rpartition("-")
    parts = clock.split(":")
    try:
        numbers = [int(p) for p in parts]
    except ValueError:
        return None
    if len(numbers) == 1:
        hours, minutes = 0, numbers[0]
    else:
        hours, minutes = numbers[-3] if len(numbers) == 3 else 0, numbers[-2]
    return int(days or 0) * 1440 + hours * 60 + minutes


def _rows(text: str) -> list[list[str]]:
    return [line.strip().split("|") for line in text.splitlines() if "|" in line]


def parse_partitions(sinfo: str, scontrol: str, qos: str = "", assoc: str = ""
                     ) -> tuple[dict[str, PartitionCompute], str | None]:
    """(partition -> how it charges, the scheduler's default partition) from the output of
    :data:`SINFO_ARGS` and :data:`SCONTROL_ARGS`, with its time limit from :data:`QOS_ARGS` and
    :data:`ASSOC_ARGS` where the site holds jobs to those. A partition either of the first two
    commands leaves out is left out."""
    shapes: dict[str, list[tuple[int, int, int]]] = {}  # name -> [(cores, cpus, gpus)]
    limits: dict[str, list[int]] = {}
    default = None
    qos_wall = {row[0]: walltime_minutes(row[1]) for row in _rows(qos) if len(row) >= 2}
    job_qos = {row[0]: row[1] for row in _rows(assoc) if len(row) >= 2}
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
        own = walltime_minutes(parts[6]) if len(parts) > 6 else None
        if own is not None:
            limits.setdefault(name, []).append(own)

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
        walls = [*limits.get(name, []), qos_wall.get(fields.get("QoS", "")),
                 qos_wall.get(job_qos.get(name, ""))]
        walls = [w for w in walls if w is not None]
        out[name] = PartitionCompute(
            max_walltime_min=min(walls) if walls else None,
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

    def ask(args: list[str]) -> str:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                              check=True).stdout

    try:
        sinfo, scontrol = ask(SINFO_ARGS), ask(SCONTROL_ARGS)
    except (OSError, subprocess.SubprocessError):
        return {}, None
    qos, assoc = "", ""
    if shutil.which("sacctmgr"):  # the accounting database may not be reachable from here
        user = os.environ.get("USER", "")
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            qos = ask(QOS_ARGS)
            assoc = ask([a.replace("$USER", user) for a in ASSOC_ARGS])
    return parse_partitions(sinfo, scontrol, qos, assoc)
