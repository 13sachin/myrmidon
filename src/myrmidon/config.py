"""Per-machine configuration.

The repository holds code; each machine holds a config. Machine-specific facts
(engine binary names/paths, default execution backend, SLURM defaults) live in a
TOML file that is never committed:

    1. $MYRMIDON_CONFIG (explicit path), else
    2. ./.myrmidon.toml (project-local override), else
    3. ~/.config/myrmidon/config.toml, else
    4. built-in defaults.

Individual keys can also be overridden via MYRMIDON_* environment variables
(e.g. MYRMIDON_GROMACS_BIN=gmx_mpi).

``MyrmidonConfig.from_file`` (``--config FILE`` on ``serve`` and ``run``) skips all of
that: the file is the whole configuration, and nothing about the machine, the working
directory or the shell changes what a session runs with.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from myrmidon.errors import ConfigError


class SlurmDefaults(BaseModel):
    partition: str | None = None
    account: str | None = None
    walltime_min: int = 60
    ntasks: int = 1
    extra_sbatch_lines: list[str] = Field(default_factory=list)
    # How coupled (MiMiC) programs share one allocation:
    #   "steps"      — one backgrounded `srun --exact` step per program
    #   "multi-prog" — a single `srun --multi-prog` with a rank-range conf file
    #                  (the launch mode validated with ParaStationMPI)
    coupled_launch: Literal["steps", "multi-prog"] = "steps"
    srun_args: list[str] = Field(default_factory=list)  # e.g. --distribution=block:cyclic
    # A per-user submit limit is a fact about this site, not about a campaign: sbatch
    # refusals matching these patterns are retried after the waits below, so no campaign
    # carries its own regex.
    submit_retry_on: list[str] = Field(
        default_factory=lambda: [r"QOSMax\w*Limit", r"AssocMax\w*Limit", r"submit limit"]
    )
    submit_retry_backoff_s: list[int] = Field(default_factory=lambda: [30, 60, 120, 300])


class PartitionCompute(BaseModel):
    """How one scheduler partition charges, as the scheduler states it.

    Written by ``myrmidon init`` from ``sinfo`` and ``scontrol`` (``execution/billing.py``),
    never typed: a node size carried from one machine to another prices every job there
    wrong without a word. ``cores_per_node`` is physical cores, which is what a booking is
    rounded up to; ``billed_per_node_h`` is what one node-hour is charged, which on a site
    that bills hardware threads at a weight is not the same number.
    """

    cores_per_node: int = Field(ge=1)
    billed_per_node_h: float = Field(gt=0)
    node_exclusive: bool = True
    note: str = ""  # what the scheduler bills that this price leaves out


class ComputeDefaults(BaseModel):
    """How this machine charges compute — drives the core-hour gate and summary.

    Example: on a cluster that allocates nodes exclusively (48 cores each), a 1-task
    job on one node costs 48 core-h per hour: ``cores_per_node = 48``,
    ``node_exclusive = true``.
    """

    cores_per_node: int = 1
    node_exclusive: bool = False
    warn_core_h: float = 10.0  # submissions above this need an acknowledgment
    block_core_h: float | None = None  # submissions above this are refused
    # the fastest a single MD job plausibly runs here (ns/day, small system, one
    # node); a booking that needs more than this cannot finish the run it carries
    plausible_ns_per_day: float = 1000.0
    # the fastest an energy minimisation plausibly goes here, in atom-steps per second
    # for one job. Measured, not guessed: GROMACS 2021.6, 22 378 atoms, 10 995
    # steepest-descent steps in 10 minutes on one exclusive 48-core node is 4.1e5
    # atom-steps/s. Used as measured — per job, like
    # plausible_ns_per_day — because the rate per core is not the same number on a
    # different system size. A minimisation whose steps × atoms needs more than the
    # booking can deliver is stopped at the time limit, not finished.
    plausible_em_atom_steps_per_s: float = 410_000.0
    # per partition, read from the scheduler by `myrmidon init`; a partition not listed is
    # priced from the machine-wide numbers above
    partitions: dict[str, PartitionCompute] = Field(default_factory=dict)
    default_partition: str | None = None  # the scheduler's own default, read by `init`
    # where a submission that names no partition goes: [slurm] partition, else the default
    _submit_partition: str | None = PrivateAttr(default=None)

    def billing(self, partition: str | None = None) -> tuple[str, PartitionCompute]:
        """(the partition a booking is priced on, how it charges). "" and the machine-wide
        numbers when it names no partition the scheduler described."""
        name = partition or self._submit_partition or self.default_partition or ""
        if name in self.partitions:
            return name, self.partitions[name]
        return "", PartitionCompute(cores_per_node=self.cores_per_node,
                                    billed_per_node_h=float(self.cores_per_node),
                                    node_exclusive=self.node_exclusive)


def _merge(base: dict, over: dict) -> dict:
    """``over`` wins, one table deep: [slurm] in a project file adds to the machine's
    rather than replacing it."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


class MyrmidonConfig(BaseModel):
    gromacs_bin: str = "gmx"
    cpmd_bin: str = "cpmd.x"
    mimicpy_bin: str = "mimicpy"
    chimerax_bin: str = "chimerax"
    mpirun_bin: str = "mpirun"
    # mimicpy prepqm's interactive prompt: command that ends the session
    mimicpy_quit_cmd: str = "q"
    backend: str = "local"  # "local" | "slurm"
    # the agent client `myrmidon client` starts here: a key in server/client_registry.py.
    # One flat key, so a project file can state it in a line (`init-project --client`
    # records it); without this field pydantic would drop the setting silently
    client: str = "claude"
    # also read under its alias `workspaces_root`, which config files and project files
    # may carry
    studies_root: Path = Path("studies")
    # shell script that provides the engines (module loads, PATH); sourced by the
    # server/CLI process itself and by every sbatch script
    env_script: Path | None = None
    slurm: SlurmDefaults = Field(default_factory=SlurmDefaults)
    compute: ComputeDefaults = Field(default_factory=ComputeDefaults)
    # which layers of input sets and protocols a session reads (knowledge/layered.py): the
    # package's own pack ("engine"), the directory beside the machine config ("site"), the
    # project's ("project"). ["engine"] is a session on the shipped knowledge alone
    knowledge_layers: list[Literal["engine", "site", "project"]] = Field(
        default_factory=lambda: ["engine", "site", "project"])

    @model_validator(mode="after")
    def _submissions_go_to_the_slurm_partition(self) -> MyrmidonConfig:
        self.compute._submit_partition = self.slurm.partition
        return self

    @classmethod
    def candidates(cls, path: Path | None = None) -> list[Path]:
        """The config files consulted, most specific first: an explicit path or
        $MYRMIDON_CONFIG alone; else the nearest project `.myrmidon.toml` at or above the
        working directory, then the machine config."""
        candidates: list[Path] = []
        if path is not None:
            candidates.append(path)
        elif env_path := os.environ.get("MYRMIDON_CONFIG"):
            candidates.append(Path(env_path))
        else:
            # the nearest project file at or above the working directory: a run from a
            # subdirectory of a project keeps its settings, including where its studies
            # live, instead of falling through to whatever the machine-wide pointer names
            for directory in (Path.cwd(), *Path.cwd().parents):
                project = directory / ".myrmidon.toml"
                if project.is_file():
                    candidates.append(project)
                    break
            candidates.append(Path.home() / ".config" / "myrmidon" / "config.toml")
        return candidates

    @classmethod
    def load(cls, path: Path | None = None) -> MyrmidonConfig:
        candidates = cls.candidates(path)
        data: dict = {}
        for candidate in reversed(candidates):
            # layered, least specific first: a project file states what differs here and
            # inherits the rest, so moving one key does not mean restating the engines,
            # the backend and the account.
            if candidate.is_file():
                loaded = _read_file(candidate)
                data = _merge(data, loaded)
                if "studies_root" in loaded:
                    data.pop("workspaces_root", None)
        env_overrides = {
            "gromacs_bin": os.environ.get("MYRMIDON_GROMACS_BIN"),
            "cpmd_bin": os.environ.get("MYRMIDON_CPMD_BIN"),
            "mimicpy_bin": os.environ.get("MYRMIDON_MIMICPY_BIN"),
            "chimerax_bin": os.environ.get("MYRMIDON_CHIMERAX_BIN"),
            "backend": os.environ.get("MYRMIDON_BACKEND"),
            "env_script": os.environ.get("MYRMIDON_ENV_SCRIPT"),
        }
        data.update({k: v for k, v in env_overrides.items() if v})
        if data.get("env_script"):
            data["env_script"] = str(Path(str(data["env_script"])).expanduser())
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: Path) -> MyrmidonConfig:
        """This file and nothing else: no project file, no machine config, no ``MYRMIDON_*``
        variable. A key the file leaves out takes the built-in default, never this
        machine's value; a missing file is an error, not an empty config."""
        path = Path(path).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        data = _read_file(path)
        if data.get("env_script"):
            data["env_script"] = str(Path(str(data["env_script"])).expanduser())
        return cls.model_validate(data)

    def to_toml(self) -> str:
        """The config as a file :meth:`from_file` reads back to an equal config. Keys at
        their default are written too, so the file states everything a session runs with."""
        return _toml(self.model_dump(mode="json", exclude_none=True))


def _read_file(path: Path) -> dict:
    """One config file's keys, with ``studies_root`` (or its alias ``workspaces_root``)
    made absolute relative to the file that says it, not to wherever it is run from."""
    try:
        loaded = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    if "workspaces_root" in loaded and "studies_root" not in loaded:
        loaded["studies_root"] = loaded.pop("workspaces_root")
    if "studies_root" in loaded:
        root = Path(str(loaded["studies_root"])).expanduser()
        if not root.is_absolute():
            root = path.resolve().parent / root
        loaded["studies_root"] = str(root)
    return loaded


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ConfigError(f"cannot write {value!r} as a config value")


def _toml(data: dict, prefix: str = "") -> str:
    """Flat keys, then one table per nested model (``[compute]``, and below it
    ``[compute.partitions.batch]``) — the shape of every config file."""
    lines = [f"{k} = {_toml_value(v)}" for k, v in data.items()
             if not isinstance(v, dict) and (v is not None or not prefix)]
    for table, values in data.items():
        if isinstance(values, dict):
            lines += ["", f"[{prefix}{table}]", _toml(values, f"{prefix}{table}.").rstrip("\n")]
    return "\n".join(lines) + "\n"
