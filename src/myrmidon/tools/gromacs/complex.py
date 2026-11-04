"""Assembling a protein–ligand complex: one coordinate file and one topology.

grompp reads a topology top to bottom: every ``[ atomtypes ]`` before the first
``[ moleculetype ]``, a molecule's position restraints inside its own moleculetype, and the
``[ molecules ]`` rows in the order the coordinates lay the atoms out. A ligand's .itp as
acpype writes it opens with its own ``[ atomtypes ]``, so including it where the molecule
belongs puts atom types after the protein's moleculetype, which grompp refuses. The
assembly splits the ligand's atom types from its molecule and places each where grompp
reads it, appends the ligand's coordinates after the protein's, and reports the counts
and the net charge read back from what it wrote."""

from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import Finding
from myrmidon.engines.gromacs.common import read_gro_info, stage_topology
from myrmidon.engines.gromacs.complex import (
    gro_line,
    itp_block,
    place_ligand,
    read_gro,
    split_atomtypes,
)
from myrmidon.engines.gromacs.itp import (
    expanded_topology,
    itp_rows,
    molecule_layout,
    parse_itp_atoms,
)
from myrmidon.engines.inputs import resolve_in_ws


class ComplexParams(BaseModel):
    structure: str = Field(description="the protein's coordinates (.gro), gmx_pdb2gmx's")
    topology: str = Field(description="the protein's topology (.top), gmx_pdb2gmx's")
    ligand_structure: str = Field(description="the ligand's coordinates (.gro), "
                                  "ligand_parametrize's, in the protein's frame")
    ligand_itp: str = Field(description="the ligand's .itp, ligand_parametrize's: its "
                            "atom types and its moleculetype")
    ligand_restraints: str | None = Field(
        description="the ligand's [ position_restraints ] (.itp), numbered within the "
        "ligand: gmx_genrestr on the ligand's own coordinates. null leaves the ligand "
        "unrestrained")
    restraints_define: str = Field(
        default="POSRES_LIG", description="the define that switches the ligand's "
        "restraints on; the run's mdp names it in define (-DPOSRES_LIG)")
    output_prefix: str = Field(default="complex", description="bare name of the .gro/.top")




@tool(
    "complex_assemble",
    description="Assemble a protein-ligand complex from gmx_pdb2gmx's and ligand_parametrize's "
                "outputs: ligand atom types, its molecule and restraints, one [ molecules ] row, "
                "coordinates appended. Reports atom counts and the net charge read back; box, "
                "solvent and ions follow.",
    tags={"gromacs", "prep", "ligand"},
    contract=Contract(consumes={"structure": "structure", "topology": "topology",
                                "ligand_structure": "structure", "ligand_itp": "itp",
                                "ligand_restraints": "position_restraints"},
                      produces=(Produces("structure"), Produces("topology"))),
)
def complex_assemble(ctx: ToolContext, params: ComplexParams) -> Observation:
    protein_gro = resolve_in_ws(ctx, params.structure)
    protein_top = resolve_in_ws(ctx, params.topology)
    ligand_gro = resolve_in_ws(ctx, params.ligand_structure)
    ligand_itp = resolve_in_ws(ctx, params.ligand_itp)
    restraints = resolve_in_ws(ctx, params.ligand_restraints) \
        if params.ligand_restraints else None

    def refuse(summary: str, *findings: Finding) -> Observation:
        return Observation(ok=False, summary=summary, findings=list(findings))

    itp_text = ligand_itp.read_text(errors="replace")
    sections = {s for _, s, _ in itp_rows(itp_text)}
    if "defaults" in sections:
        return refuse("the ligand's .itp defines [ defaults ]", itp_block(
            "COMPLEX_LIGAND_DEFAULTS",
            f"{params.ligand_itp} carries a [ defaults ] section, which only the force field "
            "states once; give the molecule's .itp, not a whole topology", ["ligand_itp"]))
    molecules = parse_itp_atoms(itp_text)
    if len(molecules) != 1:
        return refuse(f"{params.ligand_itp} defines {len(molecules)} molecules", itp_block(
            "COMPLEX_LIGAND_NOT_ONE_MOLECULE",
            f"{params.ligand_itp} defines {sorted(molecules)}; the ligand is one "
            "moleculetype", ["ligand_itp"]))
    molecule, itp_atoms = next(iter(molecules.items()))

    gmx_info = ctx.engines.get("gromacs")
    search = (Path(gmx_info.data_dir),) if gmx_info is not None and gmx_info.data_dir else ()
    protein_text, _ = expanded_topology(protein_top, search)
    if molecule in parse_itp_atoms(protein_text):
        return refuse(f"{params.topology} already defines a moleculetype {molecule}", itp_block(
            "COMPLEX_MOLECULE_NAME_TAKEN",
            f"a moleculetype {molecule!r} is already in {params.topology}; the ligand needs "
            "a name of its own (ligand_parametrize's name)", ["ligand_itp"]))
    ff_types = {f[0]: f for _, s, f in itp_rows(protein_text) if s == "atomtypes"}
    clashes = sorted({f[0] for _, s, f in itp_rows(itp_text)
                      if s == "atomtypes" and f[0] in ff_types})
    if clashes:
        return refuse(f"the ligand redefines force-field atom types {clashes}", itp_block(
            "COMPLEX_ATOMTYPE_CONFLICT",
            f"atom types {clashes} are defined by both the force field and "
            f"{params.ligand_itp}; grompp keeps one definition for both molecules"))

    _, ligand_lines, _ = read_gro(ligand_gro)
    gro_names = [ln[10:15].strip() for ln in ligand_lines]
    itp_names = [a["name"] for a in itp_atoms]
    if gro_names != itp_names:
        pairs = enumerate(zip(gro_names, itp_names, strict=False))
        first = next((i for i, (a, b) in pairs if a != b),
                     min(len(gro_names), len(itp_names)))
        return refuse("the ligand's coordinates and its .itp list different atoms", itp_block(
            "COMPLEX_LIGAND_ATOM_ORDER",
            f"{params.ligand_structure} has {len(gro_names)} atoms and {params.ligand_itp} "
            f"{len(itp_names)}; they first differ at atom {first + 1}: the coordinates "
            "must be the ones the .itp was written with", ["ligand_structure"]))
    restrained: list[int] = []
    if restraints is not None:
        restrained = [int(f[0]) for _, s, f in itp_rows(restraints.read_text())
                      if s == "position_restraints" and f[0].isdigit()]
        if not restrained or max(restrained) > len(itp_atoms):
            return refuse("the ligand's restraints are not numbered within the ligand", itp_block(
                "COMPLEX_RESTRAINTS_OUT_OF_RANGE",
                f"{params.ligand_restraints} restrains atoms up to "
                f"{max(restrained, default=0)}, and {molecule} has {len(itp_atoms)}; "
                "restraints are numbered from the ligand's first atom (gmx_genrestr on its "
                "own coordinates)", ["ligand_restraints"]))

    stage = ctx.workspace.stage("complex")
    staged = stage_topology(stage, protein_top)
    types_itp, mol_itp = f"{molecule}_atomtypes.itp", f"{molecule}.itp"
    types_text, mol_text = split_atomtypes(itp_text)
    (stage / types_itp).write_text(types_text)
    (stage / mol_itp).write_text(mol_text)
    posre_itp = None
    if restraints is not None:
        posre_itp = f"posre_{molecule}.itp"
        shutil.copy(restraints, stage / posre_itp)
    text = place_ligand(staged.read_text(), molecule, types_itp, mol_itp, posre_itp,
                        params.restraints_define)
    if text is None:
        return refuse(f"{params.topology} includes no forcefield.itp", itp_block(
            "COMPLEX_NO_FORCEFIELD_INCLUDE",
            f"{params.topology} has no #include of a force field's forcefield.itp, after "
            "which the ligand's atom types go", ["topology"]))
    top = stage / f"{params.output_prefix}.top"
    top.write_text(text)
    if staged != top:
        staged.unlink()

    title, protein_lines, box = read_gro(protein_gro)
    last_res = int(protein_lines[-1][:5]) if protein_lines else 0
    n_protein = len(protein_lines)
    merged = [*protein_lines, *(gro_line(ln, last_res + 1, n_protein + i + 1)
                                for i, ln in enumerate(ligand_lines))]
    gro = stage / f"{params.output_prefix}.gro"
    gro.write_text("\n".join([f"{title.strip()} + {molecule}", f"{len(merged):5d}",
                              *merged, box]) + "\n")

    written, unresolved = expanded_topology(top, search)
    charges = {m: sum(a["charge"] for a in atoms)
               for m, atoms in parse_itp_atoms(written).items()}
    layout = molecule_layout(written)
    net = (round(sum(n * charges[m] for m, n, _ in layout), 4)
           if all(m in charges for m, _, _ in layout) else None)
    n_atoms, box_nm = read_gro_info(gro)
    charge = (f"{net:g}" if net is not None
              else f"not read (unresolved includes: {', '.join(unresolved)})")
    from myrmidon.engines.gromacs.system import coupling_groups, forcefield, restraint_defines
    from myrmidon.knowledge.rules import SystemProfile, evaluate_rules

    ff_fact = forcefield(top)
    defines_fact = restraint_defines(top, list(search))
    tc_fact = coupling_groups(top, list(search))
    defines_list = [d for d in defines_fact.value.split() if d]
    profile = SystemProfile(
        protein_atoms=n_protein,
        ligand=molecule,
        ligand_atoms=len(ligand_lines),
        forcefield=ff_fact.value if ff_fact else None,
        restraint_defines=defines_list,
        tc_grps=tc_fact.value,
    )
    profile.triggered_rules = evaluate_rules(profile)
    root = ctx.workspace.root
    return Observation(
        ok=True,
        summary=f"Complex assembled: {n_protein} protein atoms + {len(ligand_lines)} "
        f"{molecule} atoms = {n_atoms}; net charge {charge}"
        + (f"; {len(restrained)} ligand atoms restrained under -D{params.restraints_define}"
           if restrained else "; the ligand is unrestrained"),
        artifacts=[ArtifactRef.from_path(gro, "structure", root=root),
                   ArtifactRef.from_path(top, "topology", root=root)],
        inputs=[ArtifactRef.from_path(p, role, root=root) for p, role in (
            (protein_gro, "structure"), (protein_top, "topology"),
            (ligand_gro, "structure"), (ligand_itp, "itp"),
            *([(restraints, "position_restraints")] if restraints else []))],
        n_atoms=n_atoms,
        box_nm=box_nm,
        net_charge=net,
        molecule=molecule,
        protein_atoms=n_protein,
        ligand_atoms=len(ligand_lines),
        restrained_atoms=len(restrained),
        restraints_define=params.restraints_define if restrained else None,
        system_profile=profile.model_dump(exclude_defaults=True, exclude={"triggered_rules"}),
        modeling_rules=[r.model_dump(exclude={"when", "sources"}) for r in profile.triggered_rules],
    )
