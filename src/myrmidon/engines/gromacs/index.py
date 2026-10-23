"""Index groups and position restraints: gmx make_ndx and gmx genrestr.

Both read their commands from a prompt and carry on past one they cannot apply: make_ndx
prints the error and writes the index from the rest, exit 0, and a group number that
shifted under an earlier failure silently names the wrong group
(tests/fixtures/gromacs/make_ndx_missing_group.*). So the commands' output is read prompt by
prompt, and the groups are reported from the written index, never from what was asked.

genrestr numbers the restrained atoms as they are numbered in the structure it is given,
while ``[ position_restraints ]`` numbers them within their molecule. Restraining a ligand
picked out of the whole complex writes the complex's numbers into the ligand's restraints:
grompp then refuses them as out of range or, for a molecule large enough, restrains other
atoms. The topology's layout says which molecule the atoms are in and where it starts.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning
from myrmidon.engines.gromacs.common import run_gmx, stage_output, tail
from myrmidon.engines.gromacs.itp import expanded_topology, molecule_layout, molecule_of
from myrmidon.engines.inputs import resolve_in_ws

_GROUP_LINE = re.compile(r"^\s*(\d+)\s+(.+?)\s*:\s+(\d+) atoms\s*$", re.MULTILINE)
_REJECTION = re.compile(r"Syntax error|Group is empty|does not exist|^Error", re.MULTILINE)
_GENRESTR_GROUP = re.compile(r"Group\s+(\d+)\s+\(\s*(.+?)\)\s+has\s+(\d+) elements")


class IndexGroup(BaseModel):
    number: int
    name: str
    atoms: int


class IndexObservation(Observation):
    groups: list[IndexGroup] = Field(default_factory=list)  # every group in the index
    created: list[IndexGroup] = Field(default_factory=list)  # the ones the commands made


def ndx_groups(text: str) -> list[tuple[str, list[int]]]:
    """(name, atom numbers) for every group of a .ndx file, in order."""
    groups: list[tuple[str, list[int]]] = []
    for line in text.splitlines():
        s = line.split(";")[0].strip()
        if s.startswith("["):
            groups.append((s.strip("[] "), []))
        elif s and groups:
            groups[-1][1].extend(int(x) for x in s.split())
    return groups


def rejected_ndx_commands(stdout: str, commands: list[str]) -> list[tuple[str, str]]:
    """(command, make_ndx's message) for each command make_ndx printed an error for.
    The listing comes first; after it, each prompt '> ' is followed by its command's output."""
    segments = re.split(r"^> ", stdout, flags=re.MULTILINE)[1:]
    out = []
    for i, segment in enumerate(segments[: len(commands)]):
        m = _REJECTION.search(segment)
        if m:
            line = segment[m.start():].splitlines()[0].strip()
            out.append((commands[i], line))
    return out


class MakeNdxParams(BaseModel):
    structure: str = Field(description="structure the groups are built over (.gro/.pdb/.tpr)")
    commands: list[str] = Field(
        min_length=1,
        description="make_ndx commands, applied in order; the tool ends the input itself. "
        "'r LIG' (residue name), 'a CA' (atom name), group numbers joined by '&' or '|' "
        "(e.g. '1 | 13'), 'r LYS & a NZ', 'name 20 Protein_LIG', 'del 20'. A new group "
        "takes the next number after the existing ones, and the default groups depend on "
        "what the system holds: read them from a first call rather than assuming them",
    )
    index: str | None = Field(default=None, description="existing .ndx to extend (-n)")
    output: str = Field(default="index.ndx", description="bare file name, written in stage")
    stage: str = Field(default="prep")

    @field_validator("commands")
    @classmethod
    def _no_quit(cls, v: list[str]) -> list[str]:
        if any(c.strip() == "q" for c in v):
            raise ValueError("leave out 'q': the tool ends make_ndx's input itself, and a "
                             "'q' before the last command drops the commands after it")
        return [c.strip() for c in v if c.strip()]


@tool(
    "gmx_make_ndx",
    description="Build an index (.ndx) of atom groups with gmx make_ndx, for tc-grps, "
    "energygrps, a QM region, restraints or analysis. Reports every group with its atom "
    "count as written. A command make_ndx cannot apply is a refusal, not a smaller index "
    "(NDX_COMMAND_REJECTED).",
    returns=IndexObservation,
    tags={"gromacs", "prep"},
    requires={"gromacs"},
)
def gmx_make_ndx(ctx: ToolContext, params: MakeNdxParams) -> IndexObservation:
    stage = ctx.workspace.stage(params.stage)
    out = stage_output(stage, params.output)
    args = ["make_ndx", "-f", str(resolve_in_ws(ctx, params.structure)), "-o", str(out)]
    if params.index:
        args += ["-n", str(resolve_in_ws(ctx, params.index))]
    proc = run_gmx(ctx, args, cwd=stage, stdin="\n".join([*params.commands, "q"]) + "\n")
    command = list(proc.args)
    if proc.returncode != 0 or not out.exists():
        return IndexObservation(
            ok=False, summary=f"make_ndx failed (exit {proc.returncode})", command=command,
            log_tail=tail(proc.stderr + "\n" + proc.stdout, 40))
    listing = proc.stdout.split("\n> ", 1)[0]
    n_before = len(_GROUP_LINE.findall(listing))
    groups = [IndexGroup(number=i, name=name, atoms=len(members))
              for i, (name, members) in enumerate(ndx_groups(out.read_text()))]
    rejected = rejected_ndx_commands(proc.stdout, params.commands)
    names = {g.name for g in groups}
    for cmd in params.commands:  # 'name N X' on a group that does not exist is silent
        verb, _, rest = cmd.partition(" ")
        new = rest.partition(" ")[2].strip()
        if verb == "name" and new and new not in names:
            rejected.append((cmd, "no group was renamed: that group number does not exist"))
    created = groups[n_before:]
    if rejected:
        return IndexObservation(
            ok=False,
            summary=f"make_ndx skipped {len(rejected)} of {len(params.commands)} command(s); "
            "the index it wrote is not the one asked for",
            command=command,
            warnings=[ValidationWarning(
                id="NDX_COMMAND_REJECTED",
                severity="block",
                message="; ".join(f"{c!r}: {m}" for c, m in rejected),
                suggestion="a group number counts every group before it, including any a "
                "failed command left behind; build the index again from the commands that "
                "were meant, reading the numbers from `groups`",
            )],
            log_tail=tail(proc.stdout, 40),
            groups=groups,
            created=created,
        )
    return IndexObservation(
        ok=True,
        summary=f"index with {len(groups)} groups; made "
        + (", ".join(f"{g.number} {g.name} ({g.atoms} atoms)" for g in created) or "none"),
        command=command,
        artifacts=[ArtifactRef.from_path(out, "index", root=ctx.workspace.root)],
        groups=groups,
        created=created,
    )


class RestraintObservation(Observation):
    restrained_atoms: int | None = None
    first_atom: int | None = None  # as numbered in the itp
    last_atom: int | None = None
    molecule: str | None = None  # the moleculetype the itp belongs inside
    force_constants: list[float] | None = None  # kJ mol^-1 nm^-2, x y z
    available_groups: list[IndexGroup] | None = None  # when the group was not found


class GenrestrParams(BaseModel):
    structure: str = Field(
        description="coordinates the atoms are numbered from (.gro/.pdb). For a molecule "
        "that is not the first in the system — a ligand in a complex — this is the "
        "molecule's own structure, so that its numbering starts at 1"
    )
    group: str = Field(description="group to restrain: its name or its number")
    force_constants: list[float] = Field(
        min_length=3, max_length=3,
        description="x, y, z force constants (kJ mol^-1 nm^-2); the researcher's choice. "
        "genrestr's own default is 1000 1000 1000",
    )
    topology: str = Field(
        description="the system's topology (.top): its [ molecules ] say which molecule "
        "the atoms belong to, so the numbering can be checked against it"
    )
    index: str | None = Field(default=None, description=".ndx holding the group (-n)")
    output: str = Field(default="posre_group.itp", description="bare file name")
    stage: str = Field(default="prep")


@tool(
    "gmx_genrestr",
    description="Write position restraints for a group with gmx genrestr. Checks the "
    "atom numbers against the topology's molecules: restraints are numbered within one "
    "molecule, so a group that is not numbered from its molecule's first atom is refused "
    "(GENRESTR_NOT_MOLECULE_NUMBERING, GENRESTR_SPANS_MOLECULES). Reports the moleculetype "
    "the .itp has to be included in.",
    returns=RestraintObservation,
    tags={"gromacs", "prep"},
    requires={"gromacs"},
)
def gmx_genrestr(ctx: ToolContext, params: GenrestrParams) -> RestraintObservation:
    stage = ctx.workspace.stage(params.stage)
    out = stage_output(stage, params.output)
    fc = [f"{v:g}" for v in params.force_constants]
    args = ["genrestr", "-f", str(resolve_in_ws(ctx, params.structure)), "-o", str(out),
            "-fc", *fc]
    if params.index:
        args += ["-n", str(resolve_in_ws(ctx, params.index))]
    proc = run_gmx(ctx, args, cwd=stage, stdin=params.group + "\n")
    command = list(proc.args)
    if proc.returncode != 0 or not out.exists():
        text = proc.stdout + "\n" + proc.stderr
        available = [IndexGroup(number=int(n), name=name, atoms=int(k))
                     for n, name, k in _GENRESTR_GROUP.findall(text)] or None
        missing = "No such group" in text
        return RestraintObservation(
            ok=False,
            summary=f"genrestr found no group {params.group!r}" if missing
            else f"genrestr failed (exit {proc.returncode})",
            command=command,
            log_tail=tail(text, 30),
            available_groups=available,
        )
    atoms = [int(f[0]) for f in (ln.split() for ln in out.read_text().splitlines())
             if len(f) >= 2 and f[0].isdigit()]
    if not atoms:
        return RestraintObservation(ok=False, summary=f"{params.output} restrains no atoms",
                                    command=command)
    top = resolve_in_ws(ctx, params.topology)
    gmx_info = ctx.engines.get("gromacs")
    search = (Path(gmx_info.data_dir),) if gmx_info is not None and gmx_info.data_dir else ()
    text, unresolved = expanded_topology(top, search)
    where = molecule_of(molecule_layout(text), atoms)
    base = dict(command=command, restrained_atoms=len(atoms), first_atom=min(atoms),
                last_atom=max(atoms), force_constants=params.force_constants)
    if where == "spans":
        return RestraintObservation(
            ok=False,
            summary=f"atoms {min(atoms)}-{max(atoms)} lie in more than one molecule",
            warnings=[ValidationWarning(
                id="GENRESTR_SPANS_MOLECULES", severity="block",
                message=f"group {params.group!r} holds atoms of more than one molecule of "
                f"{params.topology}'s [ molecules ]; one [ position_restraints ] block "
                "belongs to one moleculetype",
                suggestion="restrain each molecule from a group of its own atoms",
            )],
            **base,
        )
    if isinstance(where, tuple) and where[1] != 1:
        name, first = where
        return RestraintObservation(
            ok=False,
            summary=f"the restraints are numbered from the structure, not from {name}",
            warnings=[ValidationWarning(
                id="GENRESTR_NOT_MOLECULE_NUMBERING", severity="block",
                message=f"the atoms are {name}'s, whose first atom is atom {first} of "
                f"{params.structure}; inside [ moleculetype ] {name} they are numbered from "
                f"1, so these numbers are off by {first - 1}",
                suggestion=f"run genrestr on {name}'s own coordinates, numbered from 1 (the "
                "structure it had before it was combined into the system)",
            )],
            molecule=name,
            **base,
        )
    warnings = []
    if where is None:
        warnings.append(ValidationWarning(
            id="GENRESTR_LAYOUT_UNKNOWN", severity="warn",
            message=f"could not follow {params.topology}'s molecules as far as these atoms: "
            f"no definition read for a molecule before them (unresolved includes: "
            f"{unresolved or 'none'}); the numbering was not checked",
        ))
    molecule = where[0] if isinstance(where, tuple) else None
    return RestraintObservation(
        ok=True,
        summary=f"{len(atoms)} atoms of {params.group} restrained at {' '.join(fc)} "
        "kJ mol^-1 nm^-2"
        + (f"; include it inside [ moleculetype ] {molecule}, after its sections and "
           "before the next moleculetype, under an #ifdef the run's define names"
           if molecule else ""),
        artifacts=[ArtifactRef.from_path(out, "position_restraints", root=ctx.workspace.root)],
        warnings=warnings,
        molecule=molecule,
        **base,
    )
