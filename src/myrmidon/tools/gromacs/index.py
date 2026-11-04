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
atoms. The topology's layout says which molecule the atoms are in and where it starts."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.gromacs.common import run_gmx, stage_output, tail
from myrmidon.engines.gromacs.index import ndx_groups, rejected_ndx_commands
from myrmidon.engines.gromacs.itp import expanded_topology, molecule_layout, molecule_of
from myrmidon.engines.inputs import resolve_in_ws

_GROUP_LINE = re.compile(r"^\s*(\d+)\s+(.+?)\s*:\s+(\d+) atoms\s*$", re.MULTILINE)


_GENRESTR_GROUP = re.compile(r"Group\s+(\d+)\s+\(\s*(.+?)\)\s+has\s+(\d+) elements")


class IndexGroup(BaseModel):
    number: int
    name: str
    atoms: int




class MakeNdxParams(BaseModel):
    structure: str = Field(description="structure the groups are built over (.gro/.pdb/.tpr)")
    commands: list[str] = Field(
        min_length=1,
        description="make_ndx commands, applied in order; the tool ends the input itself. "
        "'r LIG' (residue name), 'a CA' (atom name), groups joined by '&' or '|' — by "
        "number bare ('1 | 13') or by name in double quotes ('\"Protein\" | \"Other\"'; a "
        "bare name is a syntax error), 'r LYS & a NZ', 'name 20 Protein_LIG', 'del 20'. A new "
        "group takes the next number after the existing ones, and the default groups depend on "
        "what the system holds: read them from a first call rather than assuming them",
    )
    index: str | None = Field(default=None, description="existing .ndx to extend (-n)")
    output: str = Field(default="index.ndx", description="bare file name, written in stage")
    stage: str = Field(default="prep")

    @field_validator("commands")
    @classmethod
    def _no_quit(cls, v: list[str]) -> list[str]:
        commands = []
        for given in v:
            text = given.strip()
            # the whole command quoted as a string ('r LIG', "r LIG\nname 1 X"): the quote
            # is the writing's. make_ndx never reads ', and reads "X" as copying group X,
            # whose name is one word: a " around words with a space between is the writing's
            q = text[:1]
            if len(text) > 1 and q in "'\"" and text.endswith(q) and q not in text[1:-1] \
                    and (q == "'" or any(ch.isspace() for ch in text[1:-1])):
                text = text[1:-1]
            commands += [c.strip() for c in text.splitlines() if c.strip()]
        while commands and commands[-1] == "q":  # the end the tool writes itself
            commands.pop()
        if "q" in commands:
            raise ValueError("leave out 'q': the tool ends make_ndx's input itself, and a "
                             "'q' before the last command drops the commands after it")
        return commands


@tool(
    "gmx_make_ndx",
    description="Build an index (.ndx) of atom groups with gmx make_ndx, for tc-grps, "
    "energygrps, a QM region, restraints or analysis. Reports every group with its atom "
    "count as written. A command make_ndx cannot apply is a refusal, not a smaller index "
    "(NDX_COMMAND_REJECTED).",
    tags={"gromacs", "prep"},
    requires={"gromacs"},
    contract=Contract(consumes={"structure": ("structure", "tpr"), "index": "index"},
                      produces=(Produces("index"),)),
)
def gmx_make_ndx(ctx: ToolContext, params: MakeNdxParams) -> Observation:
    stage = ctx.workspace.stage(params.stage)
    out = stage_output(stage, params.output)
    args = ["make_ndx", "-f", str(resolve_in_ws(ctx, params.structure)), "-o", str(out)]
    if params.index:
        args += ["-n", str(resolve_in_ws(ctx, params.index))]
    proc = run_gmx(ctx, args, cwd=stage, stdin="\n".join([*params.commands, "q"]) + "\n")
    command = list(proc.args)
    if proc.returncode != 0 or not out.exists():
        return Observation(
            ok=False, summary=f"make_ndx failed (exit {proc.returncode})", command=command,
            log_tail=tail(proc.stderr + "\n" + proc.stdout, 40))
    listing = proc.stdout.split("\n> ", 1)[0]
    n_before = len(_GROUP_LINE.findall(listing))
    groups = [IndexGroup(number=i, name=name, atoms=len(members))
              for i, (name, members) in enumerate(ndx_groups(out.read_text()))]
    rejected = rejected_ndx_commands(proc.stdout, params.commands)
    for cmd in params.commands:  # 'name N X' on a group that does not exist is silent
        parts = cmd.split()
        if len(parts) == 3 and parts[0] == "name" and parts[1].isdigit() and not (
                int(parts[1]) < len(groups) and groups[int(parts[1])].name == parts[2]):
            rejected.append((cmd, "no group was renamed: that group number does not exist"))
    created = groups[n_before:]
    if rejected:
        return Observation(
            ok=False,
            summary=f"make_ndx skipped {len(rejected)} of {len(params.commands)} command(s); "
            "the index it wrote is not the one asked for",
            command=command,
            findings=[Precondition(
                id="NDX_COMMAND_REJECTED",
                message="; ".join(f"{c!r}: {m}" for c, m in rejected),
                suggestion="a group number counts every group before it, including any a "
                "failed command left behind; build the index again from the commands that "
                "were meant, reading the numbers from `groups`",
            )],
            log_tail=tail(proc.stdout, 40),
            groups=groups,
            created=created,
        )
    # a rename makes no group, so it is said with what the renamed group holds
    renamed = []
    for cmd in params.commands:
        parts = cmd.split()
        if len(parts) == 3 and parts[0] == "name" and parts[1].isdigit():
            number = int(parts[1])
            if number < len(groups) and groups[number].name == parts[2]:
                renamed.append(groups[number])
    seen: dict[str, int] = {}
    for g in groups:  # GROMACS matches a group's name whatever its case
        seen[g.name.lower()] = seen.get(g.name.lower(), 0) + 1
    repeated = sorted({g.name for g in groups if seen[g.name.lower()] > 1})
    return Observation(
        ok=True,
        summary=f"index with {len(groups)} groups; made "
        + (", ".join(f"{g.number} {g.name} ({g.atoms} atoms)" for g in created) or "none")
        + ("; renamed " + ", ".join(f"{g.number} to {g.name} ({g.atoms} atoms)"
                                    for g in renamed) if renamed else ""),
        command=command,
        artifacts=[ArtifactRef.from_path(out, "index", root=ctx.workspace.root)],
        groups=groups,
        created=created,
        findings=[Finding(
            id="NDX_GROUP_NAME_REPEATED",
            message="more than one group is named " + ", ".join(repeated) + ": a run "
            "parameter naming it takes the first, whatever the others hold",
            suggestion="build the index again with one group per name")] if repeated else [],
    )




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
    description="Write position restraints for a group with gmx genrestr. Restraints are numbered "
                "within one molecule, so a group not numbered from its molecule's first atom is "
                "refused. Reports the moleculetype the .itp must be included in.",
    tags={"gromacs", "prep"},
    requires={"gromacs"},
    contract=Contract(consumes={"structure": "structure", "topology": "topology",
                                "index": "index"},
                      produces=(Produces("position_restraints"),)),
)
def gmx_genrestr(ctx: ToolContext, params: GenrestrParams) -> Observation:
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
        return Observation(
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
        return Observation(ok=False, summary=f"{params.output} restrains no atoms",
                                    command=command)
    top = resolve_in_ws(ctx, params.topology)
    gmx_info = ctx.engines.get("gromacs")
    search = (Path(gmx_info.data_dir),) if gmx_info is not None and gmx_info.data_dir else ()
    text, unresolved = expanded_topology(top, search)
    where = molecule_of(molecule_layout(text), atoms)
    base = dict(command=command, restrained_atoms=len(atoms), first_atom=min(atoms),
                last_atom=max(atoms), force_constants=params.force_constants)
    if where == "spans":
        return Observation(
            ok=False,
            summary=f"atoms {min(atoms)}-{max(atoms)} lie in more than one molecule",
            findings=[Precondition(
                id="GENRESTR_SPANS_MOLECULES",
                message=f"group {params.group!r} holds atoms of more than one molecule of "
                f"{params.topology}'s [ molecules ]; one [ position_restraints ] block "
                "belongs to one moleculetype",
                suggestion="restrain each molecule from a group of its own atoms",
            )],
            **base,
        )
    if isinstance(where, tuple) and where[1] != 1:
        name, first = where
        return Observation(
            ok=False,
            summary=f"the restraints are numbered from the structure, not from {name}",
            findings=[Precondition(
                id="GENRESTR_NOT_MOLECULE_NUMBERING",
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
        warnings.append(Finding(
            id="GENRESTR_LAYOUT_UNKNOWN",
            message=f"could not follow {params.topology}'s molecules as far as these atoms: "
            f"no definition read for a molecule before them (unresolved includes: "
            f"{unresolved or 'none'}); the numbering was not checked",
        ))
    molecule = where[0] if isinstance(where, tuple) else None
    return Observation(
        ok=True,
        summary=f"{len(atoms)} atoms of {params.group} restrained at {' '.join(fc)} "
        "kJ mol^-1 nm^-2"
        + (f"; include it inside [ moleculetype ] {molecule}, after its sections and "
           "before the next moleculetype, under an #ifdef the run's define names"
           if molecule else ""),
        artifacts=[ArtifactRef.from_path(out, "position_restraints", root=ctx.workspace.root)],
        findings=warnings,
        molecule=molecule,
        **base,
    )
