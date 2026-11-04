"""A deposited PDB file read into atoms, with one alternate location per atom."""

from __future__ import annotations

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
