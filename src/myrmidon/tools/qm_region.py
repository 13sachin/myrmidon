"""Partition a system into its QM region and the rest, in one call.

Writes the system's CPMD input and the QM atoms' index group (prepqm), the selection commands
a parameter fit reads back, and a sidecar with what the selection decides: the charge the
selected atoms' topology charges sum to, the spin multiplicity and electron count that charge
leaves, and the elements the engine guessed for atom types the topology gives no atomic number
for. Which atoms are quantum is the researcher's; everything else is read and said."""

from __future__ import annotations

import json
import re
import subprocess

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, precheck, tool
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.gromacs.common import tail
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.mimic.cpmd_input import get_keyword, parse_cpmd, qm_geometry
from myrmidon.engines.mimic.qm_region import (
    SELECTION_FILE,
    guessed_elements,
    mimicpy_bin,
    prepqm_argv,
    region_chemistry,
    rejected_commands,
    rounded_charge,
    unknown_commands,
)
from myrmidon.engines.mimic.system import pp_findings, qm_pseudopotentials


class QMRegionParams(BaseModel):
    topology: str = Field(description="GROMACS topology (.top)")
    coords: str = Field(description="coordinate file (.gro)")
    selections: list[str] = Field(
        min_length=1,
        description="MiMiCPy selection commands, applied in order: 'add SEL', 'add-bound SEL' "
        "(adds them marked as boundary atoms, which get pp_info's boundary pseudopotential), "
        "'delete SEL', 'clear'. SEL is 'KEYWORD is VALUE' over the topology (resname, resid, name, "
        "id, ...) joined by 'and'/'or', e.g. 'add resname is LIG' or 'add resid is 42 and "
        "name is OG'; the residue and atom names are this system's",
    )
    padding_nm: float = Field(
        gt=0,
        description="vacuum between the QM atoms and the QM cell wall, per side (nm). "
        "REQUIRED: prepqm's default of 0 makes the cell the atoms' bounding box, which "
        "crashes CPMD; 0.3-0.5 nm is typical for an isolated-molecule Poisson solver",
    )
    template: str | None = Field(
        default=None, description="CPMD template input (&CPMD/&DFT/&SYSTEM settings) — prepqm -inp"
    )
    pp_info: str | None = Field(
        default=None,
        description="pseudopotential table (element, file, ..., LMAX) — prepqm -pp; the file "
        "names must exist in CPMD's PP_LIBRARY_PATH. Without it every species gets LMAX=S",
    )
    atomtypes_map: dict[str, str] = Field(
        default_factory=dict,
        description="atom type -> element, for a type the topology gives no atomic number and "
        "guessed_elements shows misread — prepqm -nsa",
    )
    qm_group: str = Field(default="QMatoms", description="index group name for the QM atoms")
    stage: str = Field(
        default="qmmm",
        description="stage the coupled runs belong to — its absolute path is written into "
        "&MIMIC PATHS, which each run re-points to its own directory under it",
    )


def _gate(ctx: ToolContext, params: QMRegionParams):
    """Refuse a selection command prepqm does not have, before it runs: it would skip the
    command and write the region the others make, exit 0."""
    unknown = unknown_commands(params.selections)
    if not unknown:
        return [], ["selection_commands"]
    return [Precondition(
        id="QM_SELECTION_COMMAND_UNKNOWN",
        message=f"prepqm has no command for {unknown}: it would skip "
        f"{'it' if len(unknown) == 1 else 'them'} and write the region the other commands "
        "make. Its commands are 'add SEL', 'add-bound SEL', 'delete SEL' and 'clear'.",
        suggestion="'delete SEL' removes atoms; each command is one verb and one selection",
    )], ["selection_commands"]


def _refused(params: QMRegionParams, argv, proc, inp, ndx) -> Observation | None:
    """Why the files prepqm left are not the region asked for; None when they are."""
    rejected = rejected_commands(proc.stdout, params.selections)
    if rejected:  # the files exist, and hold the region of the commands it did not skip
        return Observation(
            ok=False,
            summary=f"prepqm skipped {len(rejected)} of {len(params.selections)} selection "
            "command(s); the region it wrote is not the one asked for",
            command=argv,
            findings=[Precondition(
                id="QM_SELECTION_REJECTED",
                message="; ".join(f"{cmd!r}: {msg}" for cmd, msg in rejected),
                suggestion="fix those commands and select again; a selection that matched "
                "nothing names atoms this topology does not have",
            )],
            log_tail=tail(proc.stdout, 40),
            selections=params.selections,
        )
    # mimicpy can exit 0 after an internal failure, an empty region among them: judge by the
    # artifacts, not the exit code
    if proc.returncode != 0 or not inp.exists() or not ndx.exists():
        return Observation(
            ok=False,
            summary=f"mimicpy prepqm failed (exit {proc.returncode}; cpmd.inp "
            f"{'written' if inp.exists() else 'missing'}, index.ndx "
            f"{'written' if ndx.exists() else 'missing'})",
            command=argv,
            log_tail=tail(proc.stderr + "\n" + proc.stdout, 40),
            selections=params.selections,
        )
    return None


@tool(
    "qm_region",
    description="Select the QM region; writes the system's CPMD input, the QM index group and "
    "the selection file, and says the charge, multiplicity and electrons the selection leaves "
    "and the elements guessed for atom types the topology gives no atomic number for.",
    tags={"qmmm"},
    requires={"mimicpy"},
    contract=Contract(consumes={"topology": "topology", "coords": "structure",
                                "template": "cpmd_input"},
                      produces=(Produces("cpmd_input"), Produces("index"),
                                Produces("qm_selection")),
                      upstream="template"),
)
def qm_region(ctx: ToolContext, params: QMRegionParams) -> Observation:
    precheck(ctx, _gate(ctx, params)[0])
    stage = ctx.workspace.stage(params.stage)
    gmx = ctx.engines.get("gromacs")
    argv = prepqm_argv(
        mimicpy_bin(ctx), resolve_in_ws(ctx, params.topology), resolve_in_ws(ctx, params.coords),
        stage, params.qm_group, params.padding_nm, gmx.data_dir if gmx else None,
        resolve_in_ws(ctx, params.template) if params.template else None,
        resolve_in_ws(ctx, params.pp_info) if params.pp_info else None, params.atomtypes_map)
    stdin = "\n".join(params.selections) + f"\n{ctx.config.mimicpy_quit_cmd}\n"
    proc = subprocess.run(  # noqa: S603
        argv, cwd=stage, input=stdin, capture_output=True, text=True, timeout=600
    )
    ndx, inp = stage / "index.ndx", stage / "cpmd.inp"
    refused = _refused(params, argv, proc, inp, ndx)
    if refused is not None:
        return refused
    sections = parse_cpmd(inp.read_text())
    root = ctx.workspace.root
    selection = stage / SELECTION_FILE
    selection.write_text("\n".join(params.selections) + "\n")
    declared = get_keyword(sections.get("SYSTEM", []), "CHARGE")
    declared_charge = (
        int(float(declared)) if declared and re.match(r"^-?\d+(\.0*)?$", declared) else None
    )
    rounded_from = rounded_charge(proc.stderr + proc.stdout)
    guessed = guessed_elements(proc.stderr + proc.stdout)
    chemistry, findings = region_chemistry(
        sections, None if rounded_from is not None else declared_charge,
        params.selections, guessed)
    sidecar = inp.with_name(inp.name + ".myrmidon.json")
    sidecar.write_text(json.dumps({"written_by": "qm_region", **chemistry}, indent=1))
    artifacts = [ArtifactRef.from_path(inp, "cpmd_input", root=root),
                 ArtifactRef.from_path(ndx, "index", root=root),
                 ArtifactRef.from_path(selection, "qm_selection", root=root),
                 ArtifactRef.from_path(sidecar, "provenance", root=root)]
    uncovered, _ = pp_findings(sections, stage)
    geom = qm_geometry(sections) or {}
    qm_atoms = geom.get("n_atoms")
    if uncovered:
        return Observation(
            ok=False,
            summary="the QM region holds an element CPMD has no pseudopotential for here: "
            + "; ".join(w.message for w in uncovered),
            command=argv, artifacts=artifacts, findings=uncovered,
            selections=params.selections, qm_atoms=qm_atoms,
            pseudopotentials=qm_pseudopotentials(sections),
        )
    if rounded_from is not None:
        findings.append(Finding(
            id="QM_CHARGE_ROUNDED",
            message=f"the selected atoms' MM charges sum to {rounded_from:g}, which prepqm "
            f"rounded to {declared_charge}: a region that cuts a covalent bond or part of a "
            "residue's charge distribution sums to a non-integer, and its QM charge is the "
            "researcher's, not the rounding's",
            suggestion="say what the selection cuts, and put the charge to the researcher",
        ))
    multiplicity = chemistry["multiplicity"]
    return Observation(
        ok=True,
        summary=f"QM region selected: {qm_atoms if qm_atoms is not None else '?'} atoms; "
        f"declared charge {declared_charge if declared_charge is not None else 'absent'}"
        + (f" (prepqm rounded the selection's MM charges, {rounded_from:g})"
           if rounded_from is not None else "")
        + (f", multiplicity {multiplicity}" if multiplicity is not None
           else f", multiplicity open ({chemistry['spin_why']})")
        + (f"; QM cell {'x'.join(str(v) for v in geom['cell_bohr'])} bohr, "
           f"min vacuum {geom['min_vacuum_bohr']} bohr" if geom else ""),
        command=argv,
        artifacts=artifacts,
        findings=findings,
        qm_atoms=qm_atoms,
        declared_charge=declared_charge,
        rounded_from=rounded_from,
        derived_multiplicity=multiplicity,
        electrons=chemistry["electrons"],
        guessed_elements=guessed,
        selections=params.selections,
        qm_cell_bohr=geom.get("cell_bohr"),
        qm_extent_bohr=geom.get("extent_bohr"),
        min_vacuum_bohr=geom.get("min_vacuum_bohr"),
        pseudopotentials=qm_pseudopotentials(sections),
    )
