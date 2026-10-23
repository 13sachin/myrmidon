"""``myrmidon init``: the machine config from a commented template.

The repository holds code; each machine holds one config file. ``init`` writes it once
from what it can discover (engine binaries on PATH, the backend asked for) with every
key present and explained, so the user edits a complete file instead of guessing key
names. It never overwrites an existing config without ``--force`` — it prints the diff
of what it would write instead. A ``secrets.env`` placeholder (chmod 600) is created
next to it for the API key the standalone runner needs.
"""

from __future__ import annotations

import difflib
import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path("~/.config/myrmidon")
CONFIG_NAME = "config.toml"
SECRETS_NAME = "secrets.env"

_GROMACS_CANDIDATES = ("gmx", "gmx_mpi", "gmx_d", "gmx_mpi_d")
_CPMD_CANDIDATES = ("cpmd.x", "cpmd")
_MIMICPY_CANDIDATES = ("mimicpy",)
_CHIMERAX_CANDIDATES = ("chimerax", "ChimeraX")

SECRETS_TEMPLATE = """# myrmidon secrets — chmod 600. Source per run
# (`source ~/.config/myrmidon/secrets.env`); never export from .bashrc: sbatch --export=ALL
# would copy it into every job environment. Each line needs `export`: a bare KEY=value only
# sets a shell variable, which the `myrmidon` process never sees.
# export ANTHROPIC_API_KEY=sk-ant-...
"""


def config_path() -> Path:
    """Where the machine config lives: $MYRMIDON_CONFIG, else ~/.config/myrmidon/config.toml."""
    env = os.environ.get("MYRMIDON_CONFIG")
    return Path(env).expanduser() if env else (CONFIG_DIR / CONFIG_NAME).expanduser()


def local_cores() -> int:
    """How many cores this machine has, for a config with no scheduler behind it.

    Prefers the CPUs this process is actually allowed to use over the CPUs that exist:
    inside a container or under a taskset the second number is a promise the machine
    cannot keep.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not Linux
        return max(1, os.cpu_count() or 1)


def _first_on_path(candidates: tuple[str, ...], default: str) -> tuple[str, bool]:
    for c in candidates:
        if shutil.which(c):
            return c, True
    return default, False


@dataclass
class InitPlan:
    backend: str = "local"
    studies_root: str = ""
    env_script: str | None = None
    gromacs_bin: str = "gmx"
    cpmd_bin: str = "cpmd.x"
    mimicpy_bin: str = "mimicpy"
    chimerax_bin: str = "chimerax"
    found: dict[str, bool] = field(default_factory=dict)
    partition: str = ""
    account: str = ""
    # how each partition charges, as the scheduler states it (execution/billing.py)
    partitions: dict = field(default_factory=dict)
    default_partition: str | None = None


def discover_plan(backend: str = "local", studies_root: str | None = None,
                  env_script: str | None = None, partition: str = "",
                  account: str = "") -> InitPlan:
    plan = InitPlan(backend=backend, env_script=env_script, partition=partition,
                    account=account)
    plan.studies_root = studies_root or str(Path("~/myrmidon-workspaces").expanduser())
    for attr, cands, default in (("gromacs_bin", _GROMACS_CANDIDATES, "gmx"),
                                 ("cpmd_bin", _CPMD_CANDIDATES, "cpmd.x"),
                                 ("mimicpy_bin", _MIMICPY_CANDIDATES, "mimicpy"),
                                 ("chimerax_bin", _CHIMERAX_CANDIDATES, "chimerax")):
        value, ok = _first_on_path(cands, default)
        setattr(plan, attr, value)
        plan.found[attr] = ok
    if backend == "slurm":
        from myrmidon.execution.billing import read_partitions

        plan.partitions, plan.default_partition = read_partitions()
    return plan


def _q(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_config(plan: InitPlan) -> str:
    """The complete, commented config.toml for this plan."""
    def binline(key: str, value: str) -> str:
        note = ("on PATH" if plan.found.get(key)
                else "NOT on PATH — fix or load its module")
        return f"{key} = {_q(value)}    # {note}"

    slurm = plan.backend == "slurm"
    env_line = (f"env_script = {_q(plan.env_script)}" if plan.env_script else
                '# env_script = "~/.config/myrmidon/env_engines.sh"')
    lines = [
        "# Myrmidon per-machine config — written by `myrmidon init`. Edit freely;",
        "# `myrmidon doctor` reports what it resolves to. Keys and defaults: myrmidon/config.py.",
        "",
        "# Engine binaries (names on PATH or absolute paths).",
        binline("gromacs_bin", plan.gromacs_bin),
        binline("cpmd_bin", plan.cpmd_bin),
        binline("mimicpy_bin", plan.mimicpy_bin),
        binline("chimerax_bin", plan.chimerax_bin),
        "",
        "# Shell script that provides the engines (module loads, PATH, GMXLIB ...). Sourced",
        "# by `myrmidon serve`/`doctor` themselves and at the top of every batch script, so",
        "# the agent client need not be started from a prepared shell. `module load` one per",
        "# line, never piped.",
        env_line,
        "",
        "# Where studies live: one workspace directory per study, ledger inside. On HPC this",
        "# must be a filesystem the compute nodes see (scratch), never /tmp.",
        f"studies_root = {_q(plan.studies_root)}",
        "",
        '# "local" runs jobs as subprocesses; "slurm" submits them with sbatch.',
        f"backend = {_q(plan.backend)}",
        "",
        "# The agent client `myrmidon client` starts: claude | qwen | goose. A project overrides",
        "# this with `myrmidon init-project . --client <name>`; a missing binary is an error",
        "# naming the clients this machine has, never a silent switch to another.",
        'client = "claude"',
        "",
        "[slurm]" if slurm else "# [slurm]",
    ]
    pre = "" if slurm else "# "
    lines += [
        f"{pre}partition = {_q(plan.partition or 'batch')}",
        f"{pre}account = {_q(plan.account or 'your-project')}",
        f"{pre}walltime_min = 60",
        f'{pre}coupled_launch = "steps"   # MiMiC: "steps" (srun --exact each) | "multi-prog"',
        f'{pre}srun_args = []             # e.g. ["--distribution=cyclic:cyclic"]',
        f"{pre}extra_sbatch_lines = []    # lines inserted after the #SBATCH block",
        "",
        "[compute]",
        "# How this machine charges compute — drives the core-hour gate and the summary.",
        *_compute_shape(plan, slurm),
        "warn_core_h = 10.0            # submissions above this need an explicit acknowledgment",
        "# block_core_h = 5000.0          # submissions above this are refused",
        "",
    ]
    if plan.partitions:
        lines += [
            "# Each partition as the scheduler states it (sinfo, scontrol), read by `init`:",
            "# physical cores per node, what one node-hour is billed, whether nodes are",
            "# allocated whole. A plan and a submission are priced on the partition they go to.",
            "# Re-run `init` when the machine changes; `doctor` says when these disagree.",
            "[compute.partitions]",
            *(f"{name} = {{ cores_per_node = {pc.cores_per_node}, billed_per_node_h = "
              f"{pc.billed_per_node_h!r}, node_exclusive = "
              f"{'true' if pc.node_exclusive else 'false'}"
              + (f", note = {_q(pc.note)}" if pc.note else "") + " }"
              for name, pc in sorted(plan.partitions.items())),
        ]
    return "\n".join(lines)


def _compute_shape(plan: InitPlan, slurm: bool) -> list[str]:
    """The machine-wide node shape: the scheduler's, when it answered; this machine's own
    cores otherwise — never a number carried in from some other machine."""
    if slurm and plan.partitions:
        name = plan.partition or plan.default_partition or ""
        node = plan.partitions.get(name) or next(iter(plan.partitions.values()))
        return [
            "# the shape of the partition submissions go to by default; the table below",
            "# prices each partition on its own nodes.",
            f"cores_per_node = {node.cores_per_node}",
            f"node_exclusive = {'true' if node.node_exclusive else 'false'}",
            *([f"default_partition = {_q(plan.default_partition)}   # the scheduler's default"]
              if plan.default_partition else []),
        ]
    if slurm:
        return [
            "# The scheduler did not answer `init` (sinfo/scontrol), so this is the login",
            "# node's core count, not a compute node's: set it to what one compute node has,",
            "# or re-run `init` where the scheduler answers. Every price depends on it.",
            f"cores_per_node = {local_cores()}",
            "node_exclusive = true   # true: a 1-task job bills the node",
        ]
    return [
        "# cores_per_node is what this machine has, counted when `init` ran. A submission",
        "# asking for more cores than exist gets a finding, because there is no scheduler",
        "# to queue it: the ranks would start together and contend.",
        f"cores_per_node = {local_cores()}",
        "node_exclusive = false   # true: a 1-task job bills the node",
    ]


@dataclass
class InitResult:
    path: Path
    written: bool
    diff: str = ""
    secrets_created: bool = False
    plan: InitPlan | None = None


def init_config(plan: InitPlan, path: Path | None = None, force: bool = False) -> InitResult:
    path = path or config_path()
    text = render_config(plan)
    result = InitResult(path=path, written=False, plan=plan)
    if path.exists() and not force:
        current = path.read_text().splitlines(keepends=True)
        result.diff = "".join(difflib.unified_diff(
            current, text.splitlines(keepends=True), fromfile=str(path),
            tofile="myrmidon init (not written)"))
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    result.written = True
    secrets = path.parent / SECRETS_NAME
    if not secrets.exists():
        secrets.write_text(SECRETS_TEMPLATE)
        secrets.chmod(stat.S_IRUSR | stat.S_IWUSR)
        result.secrets_created = True
    return result
