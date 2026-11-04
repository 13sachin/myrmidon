"""A deposited PDB file read into atoms, with one alternate location per atom."""

from __future__ import annotations

import re

from pydantic import BaseModel

from myrmidon.core.validation import Finding, Precondition
from myrmidon.errors import MyrmidonError


class PdbAtom(BaseModel):
    record: str
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    element: str
    line: str

    @property
    def residue(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode)


def parse_pdb_atoms(text: str) -> tuple[list[PdbAtom], int, list[str]]:
    """(ATOM/HETATM records of the first model, the number of models, CONECT lines)."""
    atoms: list[PdbAtom] = []
    conect: list[str] = []
    models = 0
    in_first = True
    for line in text.splitlines():
        record = line[:6].strip()
        if record == "MODEL":
            models += 1
            in_first = models == 1
        elif record == "ENDMDL":
            in_first = False
        elif record == "CONECT":
            conect.append(line)
        elif record in ("ATOM", "HETATM") and in_first:
            line = line.ljust(80)
            try:
                serial = int(line[6:11])
                resseq = int(line[22:26])
            except ValueError:
                continue
            name = line[12:16].strip()
            element = line[76:78].strip() or name.lstrip("0123456789")[:1]
            atoms.append(PdbAtom(record=record, serial=serial, name=name,
                                 altloc=line[16].strip(), resname=line[17:20].strip(),
                                 chain=line[21].strip(), resseq=resseq,
                                 icode=line[26].strip(), element=element.upper(),
                                 line=line.rstrip()))
    return atoms, max(models, 1), conect


CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def hetero_chain_ids(names: list[str], used: set[str]) -> dict[str, str]:
    """A chain id of its own for each kept heteroatom residue name, none the file uses.
    pdb2gmx types a chain by its residues and refuses one that mixes a polymer with an ion
    or a water, so what is kept beside the macromolecule is written as chains of its own."""
    free = [c for c in CHAIN_IDS if c not in used]
    if len(names) > len(free):
        raise MyrmidonError(f"the file uses {len(used)} chain ids and leaves {len(free)} for "
                            f"{len(names)} kept residue names; keep fewer")
    return dict(zip(names, free, strict=False))


def rechained(line: str, chain: str, resseq: int) -> str:
    """The record under another chain id (column 22) and residue number (columns 23-26),
    with no insertion code: numbered within its new chain."""
    return f"{line[:21]}{chain}{resseq:4d} {line[27:]}"


def pdb_records(polymer: list[PdbAtom], hetero: list[PdbAtom], chains: dict[str, str],
                header: str) -> str:
    """The macromolecule file: each polymer chain closed by TER, then each kept heteroatom
    residue name as a chain of its own, its residues numbered from 1, closed by TER. The
    numbering is the new chain's: two kept residues of one name from two selected chains
    may share a deposited number, and pdb2gmx would read them as one residue."""
    lines = [header]
    for i, a in enumerate(polymer):
        lines.append(a.line)
        if i + 1 == len(polymer) or polymer[i + 1].chain != a.chain:
            lines.append("TER")
    for name, chain in chains.items():
        numbers: dict[tuple, int] = {}
        for a in (a for a in hetero if a.resname == name):
            n = numbers.setdefault(a.residue, len(numbers) + 1)
            lines.append(rechained(a.line, chain, n))
        lines.append("TER")
    return "\n".join([*lines, "END"]) + "\n"


def pdb_atom_line(serial: int, name: str, resname: str, chain: str, resseq: int,
                  icode: str, xyz: tuple[float, float, float], element: str) -> str:
    """An ATOM record in the PDB's columns: a name shorter than four characters of a
    one-letter element starts in column 14."""
    field = f" {name:<3s}" if len(name) < 4 and len(element) == 1 else f"{name:<4s}"
    x, y, z = xyz
    return (f"ATOM  {serial % 100000:5d} {field} {resname:>3s} {chain:1s}{resseq:4d}"
            f"{icode or ' ':1s}   {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}"
            f"          {element:>2s}")


def build_missing_atoms(polymer: list[PdbAtom], serial: int = 0
                        ) -> tuple[list[PdbAtom], dict[str, list[str]]]:
    """The polymer with the heavy atoms its residues lack built in, and what was built per
    residue (``"A:LYS141"``), read from the atoms written. PDBFixer places each from the
    residue's template and relaxes the new atoms against the fixed rest; missing residues and
    hydrogens are not built. New records continue from ``serial`` and carry the residue's
    deposited name. Raises ImportError when PDBFixer is not installed."""
    import io

    from openmm import unit
    from pdbfixer import PDBFixer

    def key(res) -> tuple[str, int, str]:
        return (res.chain.id.strip(), int(res.id), res.insertionCode.strip())

    fixer = PDBFixer(pdbfile=io.StringIO(pdb_records(polymer, [], {}, "REMARK")))
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    wanted = {key(r): {a.name for a in atoms} for r, atoms in fixer.missingAtoms.items()}
    if not wanted:
        return polymer, {}
    fixer.addMissingAtoms()
    resnames = {a.residue: a.resname for a in polymer}
    serial = max([serial, *(a.serial for a in polymer)])
    built: dict[tuple, list[PdbAtom]] = {}
    for res in fixer.topology.residues():
        k = key(res)
        for atom in res.atoms():
            if atom.name not in wanted.get(k, ()) or k not in resnames:
                continue
            serial += 1
            xyz = tuple(fixer.positions[atom.index].value_in_unit(unit.angstrom))
            element = atom.element.symbol.upper()
            line = pdb_atom_line(serial, atom.name, resnames[k], *k, xyz, element)
            built.setdefault(k, []).append(PdbAtom(
                record="ATOM", serial=serial, name=atom.name, altloc="",
                resname=resnames[k], chain=k[0], resseq=k[1], icode=k[2], element=element,
                line=line))
    out: list[PdbAtom] = []
    names: dict[str, list[str]] = {}
    for i, a in enumerate(polymer):
        out.append(a)
        if (i + 1 == len(polymer) or polymer[i + 1].residue != a.residue) \
                and a.residue in built:
            new = built.pop(a.residue)
            out.extend(new)
            names[f"{a.chain}:{a.resname}{a.resseq}{a.icode}"] = [b.name for b in new]
    return out, names


#: REMARK 470 rows after the optional model number: residue, chain, number (may be
#: negative, may run into the chain from 1000 on), insertion code, then the atoms
_REMARK_470 = re.compile(r"^\s*(?:\d+\s+)?([A-Z0-9]{1,3})\s+(\S)\s*(-?\d+)([A-Z]?)\s+(.+)$")


def remark_missing_atoms(text: str, chains: list[str]) -> dict[str, list[str]]:
    """What the deposit's REMARK 470 says residues of ``chains`` lack (``"A:LYS141"``)."""
    out: dict[str, list[str]] = {}
    for ln in text.splitlines():
        m = _REMARK_470.match(ln[10:]) if ln.startswith("REMARK 470") else None
        if m and m.group(2) in chains:
            out[f"{m.group(2)}:{m.group(1)}{int(m.group(3))}{m.group(4)}"] = m.group(5).split()
    return out


def resolve_altlocs(atoms: list[PdbAtom], keep: str) -> tuple[list[PdbAtom], int, list[str]]:
    """(atoms with one location each, atoms that had alternates, residues whose alternates
    lack ``keep``). The kept record's altloc column is blanked."""
    alternates: dict[tuple, set[str]] = {}
    for a in atoms:
        if a.altloc:
            alternates.setdefault((*a.residue, a.name), set()).add(a.altloc)
    lacking = sorted({f"{k[0]}:{k[1]}{k[2]}" for k, locs in alternates.items()
                      if keep not in locs})
    out = []
    for a in atoms:
        if not a.altloc:
            out.append(a)
        elif a.altloc == keep:
            out.append(a.model_copy(update={"altloc": "",
                                            "line": a.line[:16] + " " + a.line[17:]}))
    return out, len(alternates), lacking


def selection_finding(fid: str, message: str, suggestion: str,
                      needs: list[str]) -> Finding:
    return Precondition(id=fid, message=message,
                        suggestion=suggestion, needs=needs)


def atom_count(path) -> int | None:
    """How many atoms a structure file holds: a .gro's count line, a PDB's ATOM and HETATM
    records of the first model; None for a file it cannot read so."""
    from pathlib import Path

    path = Path(path)
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    if path.suffix.lower() == ".gro":
        lines = text.splitlines()
        return int(lines[1].split()[0]) if len(lines) > 1 and lines[1].split() and \
            lines[1].split()[0].isdigit() else None
    if path.suffix.lower() in (".pdb", ".ent"):
        atoms, _, _ = parse_pdb_atoms(text)
        return len(atoms) or None
    return None
