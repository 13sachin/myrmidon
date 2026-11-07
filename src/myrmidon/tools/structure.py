"""Structure tools: fetch a deposited structure, and select what it contributes to a system.

A deposited PDB file carries more than the system a study simulates: crystal waters,
buffer and cryo-protectant molecules, alternate locations, other copies of the chain.
GROMACS reads what it is given: pdb2gmx keeps the first alternate location of every atom
without saying so, and solvate refuses a box that already holds waters. The selection is
made here, once, and every count it drops is reported."""

from __future__ import annotations

from collections import Counter

import httpx
from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import (
    Observation,
)
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import Finding
from myrmidon.engines.inputs import resolve_in_ws, stage_output
from myrmidon.engines.structure import (
    PdbAtom,
    build_missing_atoms,
    hetero_chain_ids,
    parse_pdb_atoms,
    pdb_records,
    remark_missing_atoms,
    resolve_altlocs,
    selection_finding,
)


class FetchPdbParams(BaseModel):
    pdb_id: str = Field(description="4-character RCSB PDB id, e.g. 1L2Y")


@tool(
    "fetch_pdb",
    description="Download a structure from RCSB PDB into the workspace; reports chains "
    "and heteroatom residues parsed from the file.",
    tags={"knowledge", "structures"},
    contract=Contract(produces=(Produces("structure"),)),
)
def fetch_pdb(ctx: ToolContext, params: FetchPdbParams) -> Observation:
    pdb_id = params.pdb_id.strip().lower()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        return Observation(
            ok=False, summary=f"'{params.pdb_id}' is not a valid 4-character PDB id",
            source=f"rcsb:{params.pdb_id}",
        )
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        resp = httpx.get(url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return Observation(
            ok=False, summary=f"RCSB fetch failed for {pdb_id}: {exc}",
            source=f"rcsb:{pdb_id}",
        )
    stage = ctx.workspace.stage("fetch")
    path = stage / f"{pdb_id}.pdb"
    path.write_bytes(resp.content)
    from myrmidon.knowledge.rules import inspect_pdb_atoms

    atoms, _, _ = parse_pdb_atoms(resp.text)
    profile, _ = inspect_pdb_atoms(atoms)
    chains: set[str] = set()
    hets: set[str] = set()
    n_atoms = 0
    for line in resp.text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            n_atoms += 1
            chains.add(line[21].strip() or "?")
        if line.startswith("HETATM"):
            res = line[17:20].strip()
            if res != "HOH":
                hets.add(res)
    return Observation(
        ok=True,
        summary=f"Fetched {pdb_id.upper()}: {n_atoms} atoms, chains "
        f"{sorted(chains)}, non-water heteroatoms {sorted(hets) or 'none'}",
        source=f"rcsb:{pdb_id}",
        artifacts=[ArtifactRef.from_path(path, "structure", root=ctx.workspace.root)],
        n_atoms=n_atoms,
        chains=sorted(chains),
        het_residues=sorted(hets),
        system_profile=profile.model_dump(exclude_defaults=True, exclude={"triggered_rules"}),
        modeling_rules=[r.model_dump(exclude={"when", "sources"}) for r in profile.triggered_rules],
    )


def _listed(atoms: dict[str, list[str]]) -> str:
    return "; ".join(f"{r} {' '.join(n)}" for r, n in atoms.items())


WATER_RESIDUES = frozenset({"HOH", "WAT", "DOD", "SOL"})


class StructureSelectParams(BaseModel):
    structure: str = Field(description="deposited structure (.pdb), workspace-relative")
    chains: list[str] = Field(
        min_length=1, description="chain ids whose polymer (ATOM records) is kept as the "
        "macromolecule, e.g. ['A']")
    ligand: str | None = Field(
        description="residue name of the ligand to split out (e.g. 'JZ4'), or null for "
        "none: an apo system")
    ligand_chain: str | None = Field(
        default=None, description="the ligand copy's chain, when the residue name occurs "
        "in more than one chain")
    ligand_resid: int | None = Field(
        default=None, description="the ligand copy's residue number, when the residue "
        "name occurs more than once")
    keep: list[str] = Field(
        default_factory=list,
        description="residue names of heteroatoms kept with the macromolecule (a "
        "structural metal such as 'ZN', crystal waters as 'HOH'), each written after TER "
        "as a chain of its own so pdb2gmx types it apart from the polymer; every other "
        "HETATM residue is discarded and counted")
    build_missing_atoms: bool = Field(
        default=True,
        description="build the heavy atoms residues of the kept chains lack (a side chain "
        "the deposit leaves out as disordered; pdb2gmx refuses a residue without them) from "
        "the residue's template with PDBFixer, each one listed; false keeps the residues "
        "as deposited. Missing residues are never built")
    altloc: str = Field(
        default="A", min_length=1, max_length=1,
        description="the alternate location kept where atoms have more than one")
    protein_output: str = Field(default="protein.pdb", description="bare file name")
    ligand_output: str = Field(default="ligand.pdb", description="bare file name")




@tool(
    "structure_select",
    description="Split a deposited structure into the macromolecule (the chains kept), the ligand "
                "(one copy of a residue name) and what is discarded (waters, buffer, other chains,"
                " alternate locations), with counts. The chains and the ligand are the "
                "researcher's.",
    tags={"prep", "structures"},
    contract=Contract(consumes={"structure": "structure"},
                      produces=(Produces("structure"),
                                Produces("structure"))),
)
def structure_select(ctx: ToolContext,
                     params: StructureSelectParams) -> Observation:
    source = resolve_in_ws(ctx, params.structure)
    text = source.read_text(errors="replace")
    raw, models, conect = parse_pdb_atoms(text)
    atoms, n_alt, lacking = resolve_altlocs(raw, params.altloc)

    def refuse(summary: str, finding: Finding) -> Observation:
        return Observation(ok=False, summary=summary, findings=[finding],
                                             models=models)

    if lacking:
        return refuse(
            f"{len(lacking)} residue(s) have alternate locations without {params.altloc!r}",
            selection_finding("STRUCTURE_ALTLOC_MISSING",
                     f"residues {', '.join(lacking[:10])} have alternate locations, none of "
                     f"them {params.altloc!r}",
                     "name the location they have in altloc", ["altloc"]))
    present = sorted({a.chain for a in atoms if a.record == "ATOM"})
    missing = [c for c in params.chains if c not in present]
    if missing:
        return refuse(
            f"no polymer chain {', '.join(missing)} in {params.structure}",
            selection_finding("STRUCTURE_CHAIN_NOT_FOUND",
                     f"chains {missing} carry no ATOM records; the file's chains are "
                     f"{present}", "name chains the file has", ["chains"]))

    ligand_atoms: list[PdbAtom] = []
    if params.ligand:
        copies = sorted({a.residue for a in atoms if a.resname == params.ligand
                         and (params.ligand_chain is None or a.chain == params.ligand_chain)
                         and (params.ligand_resid is None or a.resseq == params.ligand_resid)})
        if not copies:
            hets = sorted({a.resname for a in atoms if a.record == "HETATM"}
                          - WATER_RESIDUES)
            return refuse(
                f"no residue {params.ligand} in {params.structure}",
                selection_finding("STRUCTURE_LIGAND_NOT_FOUND",
                         f"no residue named {params.ligand!r} matches; the file's "
                         f"heteroatom residues are {hets}",
                         "name one of them in ligand", ["ligand"]))
        if len(copies) > 1:
            listed = ", ".join(f"chain {c} resid {r}{i}" for c, r, i in copies)
            return refuse(
                f"{len(copies)} copies of {params.ligand}",
                selection_finding("STRUCTURE_LIGAND_AMBIGUOUS",
                         f"{params.ligand} occurs {len(copies)} times ({listed}); one copy "
                         "is the ligand of the system", "name one with ligand_chain and "
                         "ligand_resid", ["ligand_chain", "ligand_resid"]))
        ligand_atoms = [a for a in atoms if a.residue == copies[0]
                        and a.resname == params.ligand]

    ligand_ids = {id(a) for a in ligand_atoms}
    polymer: list[PdbAtom] = []
    hetero: list[PdbAtom] = []
    kept_hetero: Counter[str] = Counter()
    discarded: Counter[str] = Counter()
    discarded_chains: set[str] = set()
    seen: set[tuple] = set()
    for a in atoms:
        if id(a) in ligand_ids:
            continue
        first = (a.residue, a.resname) not in seen
        seen.add((a.residue, a.resname))
        if a.record == "ATOM":
            if a.chain in params.chains:
                polymer.append(a)
            else:
                discarded_chains.add(a.chain)
        elif a.resname in params.keep and a.chain in params.chains:
            hetero.append(a)
            kept_hetero[a.resname] += first
        else:
            discarded[a.resname] += first

    advisory = []
    built: dict[str, list[str]] = {}
    unbuilt = "build_missing_atoms is false"
    if params.build_missing_atoms:
        try:
            polymer, built = build_missing_atoms(polymer, max(
                (a.serial for a in hetero), default=0))
            unbuilt = "PDBFixer did not build them (no template, or not in the kept polymer)"
        except ImportError:
            unbuilt = "PDBFixer is not installed (pip install 'myrmidon[structure]')"
        except Exception as exc:  # noqa: BLE001 - the residues stay as deposited, said
            unbuilt = f"PDBFixer could not build them: {exc}"
    lacking_atoms = {r: n for r, n in remark_missing_atoms(text, params.chains).items()
                     if r not in built}
    if built:
        advisory.append(Finding(
            id="STRUCTURE_ATOMS_BUILT",
            message=f"built the heavy atoms {len(built)} residue(s) lacked "
            f"({_listed(built)}) from their templates with PDBFixer: their positions are "
            "modelled, not observed",
            suggestion="say so in the report; build_missing_atoms false keeps the residues "
            "as deposited"))
    if lacking_atoms:
        advisory.append(Finding(
            id="STRUCTURE_ATOMS_MISSING",
            message=f"{len(lacking_atoms)} residue(s) lack heavy atoms in the deposit "
            f"({_listed(lacking_atoms)}) and were not built: {unbuilt}",
            means="pdb2gmx refuses a residue without its heavy atoms"))
    # what is kept beside the macromolecule is written as chains of its own, after TER:
    # pdb2gmx types a chain by its residues and refuses one that mixes them
    kept_chains = hetero_chain_ids(list(kept_hetero), {a.chain for a in atoms})
    protein = polymer + hetero
    stage = ctx.workspace.stage("select")
    protein_path = stage_output(stage, params.protein_output)
    protein_path.write_text(pdb_records(
        polymer, hetero, kept_chains,
        f"REMARK   myrmidon structure_select from {source.name}: chains "
        f"{' '.join(params.chains)}, altloc {params.altloc}"
        + (f"\nREMARK   built by PDBFixer: {_listed(built)}" if built else "")))
    artifacts = [ArtifactRef.from_path(protein_path, "structure", root=ctx.workspace.root)]
    n_residues = len({a.residue for a in polymer})
    ligand_h = None
    if ligand_atoms:
        serials = {a.serial for a in ligand_atoms}
        bonds = [ln for ln in conect if all(
            int(ln[i:i + 5]) in serials for i in range(6, len(ln.rstrip()), 5)
            if ln[i:i + 5].strip().isdigit())]
        ligand_path = stage_output(stage, params.ligand_output)
        ligand_path.write_text("\n".join(
            [*(a.line for a in ligand_atoms), *bonds, "END"]) + "\n")
        artifacts.append(ArtifactRef.from_path(ligand_path, "structure",
                                               root=ctx.workspace.root))
        ligand_h = sum(a.element in ("H", "D") for a in ligand_atoms)
        if ligand_h == 0:
            advisory.append(Finding(
                id="LIGAND_HAS_NO_HYDROGENS",
                message=f"{params.ligand} as deposited carries no hydrogens: its "
                "protonation state is not in the file",
                suggestion="state the protonation and the net charge when it is "
                "parametrised",
            ))
    if models > 1:
        advisory.append(Finding(
            id="STRUCTURE_FIRST_MODEL",
            message=f"{params.structure} holds {models} models; the first is selected",
        ))
    if n_alt:
        advisory.append(Finding(
            id="STRUCTURE_ALTLOC_CHOSEN",
            message=f"{n_alt} atoms have alternate locations; location {params.altloc!r} "
            "is kept for each", suggestion="name another in altloc to keep it instead",
        ))
    lig = ligand_atoms[0] if ligand_atoms else None
    from myrmidon.knowledge.rules import inspect_pdb_atoms, save_residue_map

    profile, res_map = inspect_pdb_atoms(
        atoms,
        selected_chains=params.chains,
        ligand_name=lig.resname if lig else None,
        ligand_atoms=ligand_atoms,
        ligand_hydrogens=ligand_h,
    )
    save_residue_map(ctx.workspace.root, res_map)
    dropped = ", ".join(f"{n} {name}" for name, n in sorted(discarded.items())) or "nothing"
    return Observation(
        ok=True,
        summary=f"Selected chains {', '.join(params.chains)} ({n_residues} residues, "
        f"{len(protein)} atoms)"
        + (f"; ligand {lig.resname} chain {lig.chain} resid {lig.resseq} "
           f"({len(ligand_atoms)} atoms, {ligand_h} hydrogens)" if lig else "; no ligand")
        + "".join(f"; kept {kept_hetero[name]} {name} as chain {cid} after TER"
                  for name, cid in kept_chains.items())
        + f"; discarded {dropped}"
        + (f" and chains {', '.join(sorted(discarded_chains))}" if discarded_chains else "")
        + (f"; altloc {params.altloc} kept for {n_alt} atoms" if n_alt else "")
        + (f"; built {sum(map(len, built.values()))} missing heavy atoms in "
           f"{', '.join(built)}" if built else ""),
        artifacts=artifacts,
        inputs=[ArtifactRef.from_path(source, "structure", root=ctx.workspace.root)],
        findings=advisory,
        chains=params.chains,
        protein_atoms=len(protein),
        protein_residues=n_residues,
        kept_hetero=dict(kept_hetero),
        kept_chains=kept_chains,
        ligand=lig.resname if lig else None,
        ligand_chain=lig.chain if lig else None,
        ligand_resid=lig.resseq if lig else None,
        ligand_atoms=len(ligand_atoms) if lig else None,
        ligand_hydrogens=ligand_h,
        discarded=dict(discarded),
        discarded_chains=sorted(discarded_chains),
        altloc=params.altloc if n_alt else None,
        altloc_atoms=n_alt,
        built_atoms=built,
        missing_atoms=lacking_atoms,
        models=models,
        residue_map=res_map,
        system_profile=profile.model_dump(exclude_defaults=True, exclude={"triggered_rules"}),
        modeling_rules=[r.model_dump(exclude={"when", "sources"}) for r in profile.triggered_rules],
    )
