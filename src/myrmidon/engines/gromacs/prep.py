"""GROMACS system-preparation tools: pdb2gmx, editconf, solvate, genion.

Force field and water model are REQUIRED parameters — myrmidon never picks
them for you. All reported counts/charges are parsed from GROMACS output.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import PrepObservation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning
from myrmidon.engines.gromacs.common import (
    read_gro_info,
    run_gmx,
    stage_output,
    stage_topology,
    tail,
)
from myrmidon.engines.inputs import require_input, resolve_in_ws

#: GROMACS' two wordings for a structure file with no atom records
_NO_ATOMS_RE = re.compile(
    r"Read 0 atoms|Number of atoms.*\bis 0\b|deduce atomnumbers when no pdb information"
)


def _artifact(ctx: ToolContext, path: Path, role: str) -> ArtifactRef:
    return ArtifactRef.from_path(path, role, root=ctx.workspace.root)


class Pdb2gmxParams(BaseModel):
    structure: str = Field(description="input structure (.pdb/.gro), workspace-relative")
    forcefield: str = Field(
        description="force field name, e.g. 'amber99sb-ildn' (required — never defaulted)"
    )
    water: str = Field(description="water model, e.g. 'tip3p' (required — never defaulted)")
    ignore_hydrogens: bool = Field(default=False, description="pass -ignh")
    output_prefix: str = Field(default="processed")


@tool(
    "gmx_pdb2gmx",
    description="Generate a GROMACS topology from a structure. Force field and water "
    "model are required — myrmidon never chooses them silently. Reports atom count and "
    "net charge parsed from GROMACS output.",
    returns=PrepObservation,
    tags={"gromacs", "prep"},
    requires={"gromacs"},
)
def gmx_pdb2gmx(ctx: ToolContext, params: Pdb2gmxParams) -> PrepObservation:
    stage = ctx.workspace.stage("prep")
    structure = resolve_in_ws(ctx, params.structure)
    gro = stage / f"{params.output_prefix}.gro"
    top = stage / f"{params.output_prefix}.top"
    itp = stage / f"{params.output_prefix}_posre.itp"
    args = [
        "pdb2gmx", "-f", str(structure), "-o", str(gro), "-p", str(top),
        "-i", str(itp), "-ff", params.forcefield, "-water", params.water,
    ]
    if params.ignore_hydrogens:
        args.append("-ignh")
    proc = run_gmx(ctx, args, cwd=stage)
    command = list(proc.args)
    if proc.returncode != 0 or not gro.exists():
        advisory = []
        if _NO_ATOMS_RE.search(proc.stderr + proc.stdout):
            advisory.append(ValidationWarning(
                id="STRUCTURE_HAS_NO_ATOMS",
                severity="info",
                message=f"GROMACS read no atoms from {params.structure}: the file has no "
                "parsable ATOM/HETATM records — it is empty or not a structure (a saved "
                "error page, a truncated download). No force field, water model or -ignh "
                "changes that.",
                suggestion="report the input as corrupt; do not retry with other flags",
            ))
        return PrepObservation(
            ok=False,
            summary=f"pdb2gmx failed (exit {proc.returncode}) — "
            + ("the input structure contains no atoms" if advisory else "see log tail"),
            command=command,
            log_tail=tail(proc.stderr),
            advisory=advisory,
        )
    n_atoms, box = read_gro_info(gro)
    charge = None
    m = re.search(r"[Tt]otal charge(?: in system)?\s*[:=]?\s*(-?[\d.]+)", proc.stderr + proc.stdout)
    if m:
        charge = float(m.group(1))
    return PrepObservation(
        ok=True,
        summary=(
            f"Topology generated with {params.forcefield}/{params.water}: "
            f"{n_atoms} atoms, net charge "
            f"{charge if charge is not None else 'not reported by GROMACS'}"
        ),
        command=command,
        artifacts=[
            _artifact(ctx, gro, "structure"),
            _artifact(ctx, top, "topology"),
        ],
        n_atoms=n_atoms,
        box_nm=box,
        net_charge=charge,
        forcefield=params.forcefield,
        water_model=params.water,
    )


class EditconfParams(BaseModel):
    structure: str
    box_type: str = Field(description="cubic | triclinic | dodecahedron | octahedron")
    distance_nm: float = Field(
        description="minimum solute-box distance in nm (required — never defaulted)"
    )
    output: str = Field(default="boxed.gro")


@tool(
    "gmx_editconf",
    description="Define the simulation box around a structure. The solute-box distance "
    "is required. Warns when the clearance is below 1.0 nm (periodic-image contact risk).",
    returns=PrepObservation,
    tags={"gromacs", "prep"},
    requires={"gromacs"},
)
def gmx_editconf(ctx: ToolContext, params: EditconfParams) -> PrepObservation:
    stage = ctx.workspace.stage("prep")
    structure = resolve_in_ws(ctx, params.structure)
    out = stage_output(stage, params.output)
    args = [
        "editconf", "-f", str(structure), "-o", str(out),
        "-c", "-d", str(params.distance_nm), "-bt", params.box_type,
    ]
    proc = run_gmx(ctx, args, cwd=stage)
    command = list(proc.args)
    if proc.returncode != 0 or not out.exists():
        return PrepObservation(
            ok=False,
            summary=f"editconf failed (exit {proc.returncode})",
            command=command,
            log_tail=tail(proc.stderr),
        )
    n_atoms, box = read_gro_info(out)
    warnings = []
    if params.distance_nm < 1.0:
        warnings.append(
            ValidationWarning(
                id="BOX_CLEARANCE_TIGHT",
                severity="warn",
                message=f"solute-box distance {params.distance_nm} nm < 1.0 nm — with "
                "typical 1.0-1.2 nm cutoffs the solute may see its periodic image",
            )
        )
    return PrepObservation(
        ok=True,
        summary=f"Box defined: {params.box_type}, d = {params.distance_nm} nm, "
        f"box = {box} nm",
        command=command,
        artifacts=[_artifact(ctx, out, "structure")],
        warnings=warnings,
        n_atoms=n_atoms,
        box_nm=box,
    )


class SolvateParams(BaseModel):
    structure: str
    topology: str
    solvent_coords: str = Field(default="spc216.gro", description="solvent template")
    output: str = Field(default="solvated.gro")
    acknowledge_warnings: list[str] = Field(
        default_factory=list, description="ids of warn-level gate findings you accept"
    )


#: residue names that mean "this is already solvent". GROMACS' own water models plus the
#: two spellings the PDB uses.
SOLVENT_RESIDUES = frozenset({"SOL", "WAT", "HOH", "TIP3", "TIP3P", "SPC", "SPCE", "T3P"})


def count_solvent_residues(structure: Path) -> tuple[int, str]:
    """(how many solvent molecules the coordinate file already holds, their name).

    Read off the .gro's residue column — the same place GROMACS reads it.
    """
    seen: set[str] = set()
    name = ""
    for line in structure.read_text(errors="replace").splitlines()[2:]:
        if len(line) < 10:
            continue
        residue = line[5:10].strip().upper()
        if residue in SOLVENT_RESIDUES:
            name = residue
            seen.add(line[:10])
    return len(seen), name


def solvate_gate(ctx: ToolContext, params: SolvateParams):
    """Refuse to solvate a box that is already solvated unless it is said out loud.

    A second solvation shell is not an error GROMACS reports: it wraps thousands of waters
    around a box that is already solvated, and every later step runs on a system many
    times the size that was meant.
    """
    from myrmidon.core.validation import ValidationWarning

    warnings: list[ValidationWarning] = []
    checked = ["input_files"]
    structure, missing = require_input(ctx, "structure", params.structure)
    if missing:
        return [missing], checked
    checked.append("already_solvated")
    count, residue = count_solvent_residues(structure)
    if count:
        warnings.append(ValidationWarning(
            id="SOLVATE_ALREADY_SOLVATED",
            severity="warn",
            message=f"{params.structure} already holds {count} {residue} molecule(s): "
            "solvating it again wraps a second shell around a solvated box and the "
            "system is not the one you meant. The entry to a solvation is the "
            "bare solute; an equilibrated box is some earlier stage's output.",
            suggestion="start from the solute structure, or skip this step if the box "
            "is already the system you want",
        ))
    return warnings, checked


@tool(
    "gmx_solvate",
    description="Solvate the boxed structure. The topology is copied into the stage and "
    "the copy is updated, so the topology you passed is left as the study's starting "
    "point — use the topology artifact this returns for every later step. Reports the "
    "solvent molecule count parsed from GROMACS output.",
    returns=PrepObservation,
    tags={"gromacs", "prep"},
    requires={"gromacs"},
    gate=solvate_gate,
)
def gmx_solvate(ctx: ToolContext, params: SolvateParams) -> PrepObservation:
    stage = ctx.workspace.stage("prep")
    structure = resolve_in_ws(ctx, params.structure)
    given = resolve_in_ws(ctx, params.topology)
    # solvate rewrites the topology it is given. Edit a copy in this stage rather than the
    # file the study started from: mutating a planted input destroys the starting point, so
    # a replay plants the already-solvated file and solvates it again.
    consumed = [_artifact(ctx, structure, "structure"), _artifact(ctx, given, "topology")]
    topology = stage_topology(stage, given)
    out = stage_output(stage, params.output)
    args = [
        "solvate", "-cp", str(structure), "-cs", params.solvent_coords,
        "-o", str(out), "-p", str(topology),
    ]
    proc = run_gmx(ctx, args, cwd=stage)
    command = list(proc.args)
    if proc.returncode != 0 or not out.exists():
        return PrepObservation(
            ok=False,
            summary=f"solvate failed (exit {proc.returncode})",
            command=command,
            log_tail=tail(proc.stderr),
        )
    n_atoms, box = read_gro_info(out)
    n_solvent = None
    m = re.search(r"Number of solvent molecules:\s*(\d+)", proc.stderr + proc.stdout)
    if m:
        n_solvent = int(m.group(1))
    return PrepObservation(
        ok=True,
        summary=f"Solvated: {n_solvent if n_solvent is not None else '?'} solvent "
        f"molecules, {n_atoms} atoms total",
        command=command,
        inputs=consumed,
        artifacts=[
            _artifact(ctx, out, "structure"),
            _artifact(ctx, topology, "topology"),
        ],
        n_atoms=n_atoms,
        n_solvent=n_solvent,
        box_nm=box,
    )


class GenionParams(BaseModel):
    structure: str | None = Field(
        default=None, description="the solvated structure (gmx_solvate's): genion's run "
        "input is built from it with the genion_run_input input set. Give this or tpr")
    tpr: str | None = Field(default=None, description="a run input already made for genion; "
                            "leave unset to give structure instead")
    topology: str = Field(description="the solvated topology gmx_solvate returned")
    output: str = Field(default="ionized.gro")
    neutralize: bool = Field(default=True)
    concentration_mol_l: float | None = Field(
        default=None, description="add ions to this concentration (mol/l)"
    )
    positive_ion: str = Field(default="NA")
    negative_ion: str = Field(default="CL")
    replace_group: str = Field(default="SOL")

    @model_validator(mode="after")
    def _one_input(self) -> GenionParams:
        if (self.structure is None) == (self.tpr is None):
            raise ValueError("give structure (the solvated .gro; genion's run input is built "
                             "from it) or tpr (one already made), not both and not neither")
        return self


@tool(
    "gmx_genion",
    description="Replace solvent molecules with ions to neutralize and/or reach a target "
    "concentration. The topology is copied into the stage and the copy is updated — use "
    "the topology artifact this returns for every later step. Reports ion counts parsed "
    "from GROMACS output.",
    returns=PrepObservation,
    tags={"gromacs", "prep"},
    requires={"gromacs"},
)
def gmx_genion(ctx: ToolContext, params: GenionParams) -> PrepObservation:
    stage = ctx.workspace.stage("prep")
    given = resolve_in_ws(ctx, params.topology)
    built = []
    if params.tpr is not None:
        tpr = resolve_in_ws(ctx, params.tpr)
    else:
        tpr, built, failed = _genion_run_input(ctx, stage, params.structure, given)
        if failed is not None:
            return failed
    # genion rewrites the topology too — see gmx_solvate
    consumed = [_artifact(ctx, tpr, "tpr"), _artifact(ctx, given, "topology")]
    topology = stage_topology(stage, given)
    out = stage_output(stage, params.output)
    args = [
        "genion", "-s", str(tpr), "-o", str(out), "-p", str(topology),
        "-pname", params.positive_ion, "-nname", params.negative_ion,
    ]
    if params.neutralize:
        args.append("-neutral")
    if params.concentration_mol_l is not None:
        args += ["-conc", str(params.concentration_mol_l)]
    proc = run_gmx(ctx, args, cwd=stage, stdin=params.replace_group + "\n")
    command = list(proc.args)
    if proc.returncode != 0 or not out.exists():
        # genion rewrites -p before it writes -o; GROMACS keeps the prior version as
        # #name.N#. If one is there, the topology this call used is not the one it was
        # given any more, and a retry against it adds a second round of ions.
        backups = sorted(stage.glob(f"#{topology.name}.*#"))
        advisory = []
        if backups:
            advisory.append(ValidationWarning(
                id="GENION_TOPOLOGY_ALREADY_EDITED",
                severity="info",
                message=f"genion failed after rewriting {topology.name} "
                f"({len(backups)} GROMACS backup(s): {backups[-1].name}). Retrying against "
                "it would add ions on top of the ones already counted.",
                suggestion=f"retry from the original topology ({params.topology}) or the "
                f"backup {backups[-1].name}, not from the edited copy",
            ))
        return PrepObservation(
            ok=False,
            summary=f"genion failed (exit {proc.returncode})"
            + (f" after editing {topology.name}" if advisory else ""),
            command=command,
            log_tail=tail(proc.stderr),
            advisory=advisory,
        )
    n_atoms, box = read_gro_info(out)
    n_pos = n_neg = None
    m = re.search(
        r"add\s+(\d+)\s+\S+\s+ions?\s+and\s+(\d+)\s+\S+\s+ions?",
        proc.stderr + proc.stdout,
    )
    if m:
        n_pos, n_neg = int(m.group(1)), int(m.group(2))
    return PrepObservation(
        ok=True,
        summary=f"Ions added: +{n_pos if n_pos is not None else '?'} / "
        f"-{n_neg if n_neg is not None else '?'}; {n_atoms} atoms total",
        command=command,
        inputs=consumed,
        artifacts=[
            *built,
            _artifact(ctx, out, "structure"),
            _artifact(ctx, topology, "topology"),
        ],
        n_atoms=n_atoms,
        n_ions_positive=n_pos,
        n_ions_negative=n_neg,
        box_nm=box,
    )


def _genion_run_input(ctx: ToolContext, stage: Path, structure: str, topology: Path):
    """(the .tpr genion reads, its artifacts, a failed observation or None). Rendered from
    the genion_run_input input set, with its sidecar, and preprocessed by grompp, so that no
    caller has to build the scaffold by hand."""
    import json

    from myrmidon.engines.gromacs.mdp import render_mdp
    from myrmidon.engines.gromacs.run import grompp_warnings
    from myrmidon.knowledge.inputsets import InputSetStore, resolve

    store = getattr(ctx, "inputsets", None) or InputSetStore.builtin()
    inputset = store.get("genion_run_input")
    resolution = resolve(inputset)
    mdp = stage / "ions.mdp"
    mdp.write_text(render_mdp(resolution.values, resolution.origin, [
        "written by myrmidon gmx_genion: the run input genion reads",
        f"inputset: {inputset.id} v{inputset.version}"]))
    mdp.with_name(mdp.name + ".myrmidon.json").write_text(json.dumps({
        "inputset": inputset.id, "protocol_version": inputset.version, "renders": "mdp",
        "values": resolution.values, "origin": resolution.origin, "overrides": {},
        "invariants": [], "sources": [x.model_dump(exclude_none=True)
                                      for x in resolution.sources],
        "written_by": "gmx_genion"}, indent=1, sort_keys=True))
    coords = resolve_in_ws(ctx, structure)
    tpr = stage / "ions.tpr"
    proc = run_gmx(ctx, ["grompp", "-f", str(mdp), "-c", str(coords), "-p", str(topology),
                         "-o", str(tpr), "-po", str(stage / "ions_mdout.mdp")], cwd=stage)
    if proc.returncode != 0 or not tpr.exists():
        return None, [], PrepObservation(
            ok=False, summary=f"grompp could not make genion's run input from {structure} "
            f"and {ctx.workspace.relpath(topology)} (exit {proc.returncode}) — see log tail",
            command=list(proc.args), warnings=grompp_warnings(proc.stderr),
            log_tail=tail(proc.stderr, 30))
    return tpr, [_artifact(ctx, mdp, "mdp"), _artifact(ctx, tpr, "tpr")], None
