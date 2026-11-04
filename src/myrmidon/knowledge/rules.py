"""SystemProfile & Condition -> Action modeling/coupling rules.

Inspects deposited PDB structures and assembled GROMACS topologies/coordinates to build a
typed `SystemProfile` (sequence gaps, disulfide candidates, coordinated metals, bridging
pocket waters, ligand atom/H counts, active #ifdef restraints, coupling groups, and the
deposited-to-topology residue numbering map) and evaluates curated modeling and engine-key
coupling rules (sensor records of `subject: system`) against it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.validation import Finding
from myrmidon.knowledge.sensors import evaluate

METAL_RESNAMES = frozenset({
    "ZN", "MG", "FE", "FE2", "CA", "MN", "CU", "CU1", "CO", "NI", "NA", "K", "CD", "HG",
})
WATER_RESNAMES = frozenset({"HOH", "WAT", "SOL", "DOD", "TIP3", "T3P"})


class SystemProfile(BaseModel):
    """Deterministic structural and topological facts extracted from the system's files."""

    chains: list[str] = Field(default_factory=list)
    protein_residues: int = 0
    protein_atoms: int = 0
    ligand: str | None = None
    ligand_atoms: int = 0
    ligand_hydrogens: int | None = None
    sequence_gaps: list[str] = Field(default_factory=list)  # e.g. "A:2-21"
    disulfide_candidates: list[str] = Field(default_factory=list)  # e.g. "A:CYS14-A:CYS38 (2.04 A)"
    histidines: list[str] = Field(default_factory=list)  # e.g. "A:HIS31"
    metals: list[str] = Field(default_factory=list)  # e.g. "A:ZN201"
    pocket_waters: int = 0  # crystal waters within 3.5 A of both polymer and hetero/ligand
    forcefield: str | None = None
    restraint_defines: list[str] = Field(default_factory=list)
    tc_grps: str | None = None
    has_velocities: bool | None = None
    # topology 1-based resid -> deposited label "A:MET1", saved when numbering shifts
    residue_offset_examples: list[str] = Field(default_factory=list)
    has_residue_offset: bool = False
    triggered_rules: list[Finding] = Field(default_factory=list)


def _coords(line: str) -> tuple[float, float, float] | None:
    try:
        return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except (ValueError, IndexError):
        return None


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def evaluate_rules(profile: SystemProfile) -> list[Finding]:
    """What the system records (knowledge/rules/, `subject: system`) make of the profile's
    flags. Every flag is always present, true or false: a record's `false` condition reads
    an absent key as not matching."""
    flags = {
        "ligand_present": bool(profile.ligand or profile.ligand_atoms > 0),
        "ligand_hydrogens_zero": (profile.ligand_hydrogens == 0)
        if profile.ligand_hydrogens is not None else False,
        "small_ligand": 0 < profile.ligand_atoms < 100,
        "has_restraints": bool(profile.restraint_defines),
        "has_sequence_gaps": bool(profile.sequence_gaps),
        "has_disulfides": bool(profile.disulfide_candidates),
        "has_metals": bool(profile.metals),
        "has_residue_offset": profile.has_residue_offset,
    }
    return evaluate("system", flags)


def build_residue_map(polymer_atoms: list) -> dict[int, str]:
    """Map 1-based topology resid to deposited PDB 'chain:resname+resseq+icode'."""
    mapping: dict[int, str] = {}
    seen: list[tuple[tuple[str, int, str], str]] = []
    seen_set: set[tuple[str, int, str]] = set()
    for a in polymer_atoms:
        key = a.residue
        if key not in seen_set:
            seen_set.add(key)
            seen.append((key, a.resname))
    for top_resid, ((chain, resseq, icode), resname) in enumerate(seen, start=1):
        mapping[top_resid] = f"{chain}:{resname}{resseq}{icode}"
    return mapping


def inspect_pdb_atoms(
    atoms: list,
    *,
    selected_chains: list[str] | None = None,
    ligand_name: str | None = None,
    ligand_atoms: list | None = None,
    ligand_hydrogens: int | None = None,
) -> tuple[SystemProfile, dict[int, str]]:
    """Inspect parsed PdbAtom records to build a SystemProfile and a 1-based topology
    residue numbering map."""
    chains = selected_chains or sorted({a.chain for a in atoms if a.record == "ATOM"})
    polymer = [a for a in atoms if a.record == "ATOM" and a.chain in chains]
    res_map = build_residue_map(polymer)

    # Sequence gaps & offset detection within each chain
    by_chain: dict[str, list[tuple[int, str, str]]] = {}
    seen_res: set[tuple[str, int, str]] = set()
    histidines: list[str] = []
    sg_atoms: list[tuple[str, tuple[float, float, float]]] = []
    for a in polymer:
        if a.residue not in seen_res:
            seen_res.add(a.residue)
            by_chain.setdefault(a.chain, []).append((a.resseq, a.icode, a.resname))
            if a.resname in ("HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"):
                histidines.append(f"{a.chain}:{a.resname}{a.resseq}{a.icode}")
        if a.resname in ("CYS", "CYX") and a.name == "SG":
            xyz = _coords(a.line)
            if xyz is not None:
                sg_atoms.append((f"{a.chain}:{a.resname}{a.resseq}{a.icode}", xyz))

    gaps: list[str] = []
    for chain, residues in by_chain.items():
        for (r1, _, _), (r2, _, _) in zip(residues, residues[1:], strict=False):
            if r2 > r1 + 1:
                gaps.append(f"{chain}:{r1}-{r2}")

    # Disulfide candidates (SG-SG < 2.5 A)
    disulfides: list[str] = []
    for i, (label1, c1) in enumerate(sg_atoms):
        for label2, c2 in sg_atoms[i + 1:]:
            d = _dist(c1, c2)
            if d <= 2.5:
                disulfides.append(f"{label1}-{label2} ({d:.2f} A)")

    # Metals & pocket waters
    metals: list[str] = []
    seen_metals: set[tuple[str, int, str]] = set()
    water_coords: list[tuple[float, float, float]] = []
    for a in atoms:
        if a.record == "HETATM":
            is_metal = a.resname in METAL_RESNAMES or a.element in METAL_RESNAMES
            if is_metal and a.residue not in seen_metals:
                seen_metals.add(a.residue)
                metals.append(f"{a.chain}:{a.resname}{a.resseq}{a.icode}")
            elif a.resname in WATER_RESNAMES:
                xyz = _coords(a.line)
                if xyz is not None:
                    water_coords.append(xyz)

    lig_list = ligand_atoms or (
        [a for a in atoms if a.record == "HETATM" and a.resname == ligand_name]
        if ligand_name else []
    )
    lig_coords = [c for a in lig_list if (c := _coords(a.line)) is not None]
    poly_coords = [c for a in polymer if (c := _coords(a.line)) is not None]
    pocket_waters = 0
    if lig_coords and poly_coords and water_coords:
        for wc in water_coords:
            near_lig = any(_dist(wc, lc) <= 3.5 for lc in lig_coords)
            if near_lig and any(_dist(wc, pc) <= 3.5 for pc in poly_coords):
                pocket_waters += 1

    offsets: list[str] = []
    for top_id, label in res_map.items():
        # label is e.g. "A:THR21"
        num_str = "".join(ch for ch in label.split(":", 1)[-1][3:] if ch.isdigit() or ch == "-")
        if num_str and int(num_str) != top_id:
            offsets.append(f"topol resid {top_id} -> PDB {label}")

    profile = SystemProfile(
        chains=chains,
        protein_residues=len(res_map),
        protein_atoms=len(polymer),
        ligand=ligand_name,
        ligand_atoms=len(lig_list),
        ligand_hydrogens=ligand_hydrogens,
        sequence_gaps=gaps,
        disulfide_candidates=disulfides,
        histidines=histidines[:20],
        metals=metals,
        pocket_waters=pocket_waters,
        residue_offset_examples=offsets[:10],
        has_residue_offset=bool(offsets),
    )
    profile.triggered_rules = evaluate_rules(profile)
    return profile, res_map


RESIDUE_MAP_FILE = ".myrmidon/residue_map.json"


def save_residue_map(workspace_root: Path, res_map: dict[int, str]) -> None:
    if not res_map:
        return
    target = workspace_root / RESIDUE_MAP_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({str(k): v for k, v in res_map.items()}, indent=1) + "\n")


def load_residue_map(workspace_root: Path | None) -> dict[int, str]:
    if workspace_root is None:
        return {}
    target = workspace_root / RESIDUE_MAP_FILE
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text())
        return {int(k): str(v) for k, v in data.items()}
    except (OSError, ValueError):
        return {}
