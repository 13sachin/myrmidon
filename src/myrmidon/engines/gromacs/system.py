"""Facts read from a GROMACS system's own files: the force field its topology includes,
whether its structure carries velocities, the restraints it can switch on, how its atoms
split into temperature-coupling groups, and the QM group an index holds.

Each fact names the rule that read it and the hash of what it read, so a value an engine
input takes from the system is as traceable as one an input set states.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
from pathlib import Path

from myrmidon.core.contracts import declare_identity, declare_includes
from myrmidon.engines.gromacs.index import ndx_groups
from myrmidon.engines.gromacs.itp import expanded_topology, itp_rows, molecule_layout
from myrmidon.knowledge.catalog import SystemFact

#: what a GROMACS system is: its coordinates, its topology and the molecule files it
#: includes, its atom groups; and where a run continues from, its checkpoint
declare_identity("structure", "topology", "itp", "index", "checkpoint")

_FF_INCLUDE = re.compile(r'^\s*#include\s+"([^"/]+)\.ff/forcefield\.itp"', re.M)
_DIRECTIVE = re.compile(r"^\s*#\s*(ifdef|ifndef|else|endif)\b\s*(\S*)")
_SECTION = re.compile(r"^\s*\[\s*(\w+)\s*\]")
#: moleculetype names GROMACS's water models use
WATER = {"SOL", "WAT", "HOH", "TIP3", "TIP3P", "TIP4P", "TIP4PEW", "TIP5P", "SPC", "SPCE",
         "T3P"}


def _hash(*paths: Path) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
    return h.hexdigest()


def forcefield(top: Path) -> SystemFact | None:
    """The force-field directory the topology includes (`amber99sb-ildn` for
    `amber99sb-ildn.ff/forcefield.itp`), or None when it includes none."""
    m = _FF_INCLUDE.search(top.read_text(errors="replace"))
    if not m:
        return None
    return SystemFact(key="forcefield", value=m.group(1), rule="forcefield",
                      read=[str(top)], sha256=_hash(top))


def starting_state(structure: Path) -> list[SystemFact]:
    """Whether a run starting from `structure` continues a previous one: a .gro whose atom
    lines carry velocities continues it (continuation, no new velocities); one without
    starts from rest and draws velocities at the run's temperature."""
    lines = structure.read_text(errors="replace").splitlines()
    moving = (structure.suffix.lower() == ".gro" and len(lines) > 2
              and len(lines[2].rstrip()) >= 68)
    common = {"rule": "starting_state:" + ("velocities" if moving else "at_rest"),
              "read": [str(structure)], "sha256": _hash(structure)}
    return [SystemFact(key="continuation", value="yes" if moving else "no", **common),
            SystemFact(key="gen-vel", value="no" if moving else "yes", **common)]


def _atoms_per_molecule(text: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for mol, section, _ in itp_rows(text):
        if section == "atoms" and mol is not None:
            sizes[mol] = sizes.get(mol, 0) + 1
    return sizes


_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"')


def _walk(path: Path, search_dirs: list[Path], depth: int = 0):
    """The topology's lines with its includes inlined, as grompp reads them. An include
    that cannot be found (a force field's water or ion file, read without GROMACS's data
    directory) yields a `[ moleculetype ]` named after the file in its place."""
    for line in path.read_text(errors="replace").splitlines():
        m = _INCLUDE.match(line)
        if not m:
            yield line
            continue
        inc = m.group(1)
        found = next((base / inc for base in (path.parent, *search_dirs)
                      if (base / inc).is_file()), None)
        if found is not None and depth < 20:
            yield from _walk(found, search_dirs, depth + 1)
        elif not inc.endswith(("forcefield.itp", "ffnonbonded.itp", "ffbonded.itp")):
            yield "[ moleculetype ]"
            yield Path(inc).stem


def restraint_defines(top: Path, search_dirs: list[Path]) -> SystemFact:
    """The `define` that switches on the solute's position restraints: every name an
    `#ifdef` guarding a `[ position_restraints ]` block tests, for molecules other than
    water and ions, and by molecule in ``parts``. Empty when the topology has none."""
    lines = list(_walk(top, search_dirs))
    sizes = _atoms_per_molecule("\n".join(lines))
    stack: list[str | None] = []
    mol = None
    found: list[tuple[str, str | None]] = []
    expect_mol = False
    for line in lines:
        d = _DIRECTIVE.match(line)
        if d:
            kind, name = d.groups()
            if kind == "ifdef":
                stack.append(name)
            elif kind == "ifndef":
                stack.append(None)
            elif kind == "else" and stack:
                stack[-1] = None
            elif kind == "endif" and stack:
                stack.pop()
            continue
        body = line.split(";")[0].strip()
        s = _SECTION.match(body)
        if s:
            section = s.group(1).lower()
            expect_mol = section == "moleculetype"
            if section == "position_restraints" and stack and stack[-1]:
                found.append((stack[-1], mol))
            continue
        if expect_mol and body:
            mol, expect_mol = body.split()[0], False
    names: list[str] = []
    parts: dict[str, list[str]] = {}
    for name, molecule in found:
        if _solvent(molecule, sizes):
            continue
        if name not in names:
            names.append(name)
        if molecule is not None and name not in parts.setdefault(molecule, []):
            parts[molecule].append(name)
    return SystemFact(key="define", value=" ".join(f"-D{n}" for n in names),
                      rule="restraint_defines", read=[str(top)], sha256=_hash(top),
                      parts={m: " ".join(f"-D{n}" for n in ns) for m, ns in parts.items()})


def _solvent(molecule: str | None, sizes: dict[str, int]) -> bool:
    """Water, or an ion: a one-atom molecule, or a force field's ion file."""
    name = (molecule or "").upper()
    return name in WATER or name == "IONS" or sizes.get(molecule or "", 2) <= 1


def _protein_residues(residuetypes: Path | None) -> set[str]:
    if residuetypes is None or not residuetypes.is_file():
        return set()
    return {f[0] for line in residuetypes.read_text().splitlines()
            if len(f := line.split()) >= 2 and f[1] == "Protein"}


def coupling_groups(top: Path, search_dirs: list[Path], ndx: Path | None = None,
                    residuetypes: Path | None = None) -> SystemFact:
    """The temperature-coupling groups: an index's solute group beside Water_and_ions when
    the two partition the system; Protein and Non-Protein when a protein is present (both
    are GROMACS's default groups); otherwise the whole System."""
    text, _ = expanded_topology(top, search_dirs)
    read = [top]
    layout = molecule_layout(text)
    total = sum(n * (size or 0) for _, n, size in layout)
    if ndx is not None and ndx.is_file():
        read.append(ndx)
        groups = dict(ndx_groups(ndx.read_text()))
        total = len(set(groups.get("System", []))) or total
        solvent = set(groups.get("Water_and_ions", []))
        for name, atoms in groups.items():
            if (solvent and name != "Water_and_ions" and not solvent & set(atoms)
                    and len(solvent) + len(set(atoms)) == total):
                return SystemFact(key="tc-grps", value=f"{name} Water_and_ions",
                                  rule="coupling_groups:index", read=[str(p) for p in read],
                                  sha256=_hash(*read))
    protein = _protein_residues(residuetypes)
    molecules = {name for name, _, _ in layout}
    has_protein = any(m.startswith("Protein") for m in molecules) or any(
        mol in molecules and section == "atoms" and len(f) > 3 and f[3] in protein
        for mol, section, f in itp_rows(text))
    value, rule = (("Protein Non-Protein", "coupling_groups:protein") if has_protein
                   else ("System", "coupling_groups:solute"))
    return SystemFact(key="tc-grps", value=value, rule=rule, read=[str(p) for p in read],
                      sha256=_hash(*read))


def qm_group(ndx: Path, name: str | None = None) -> SystemFact | None:
    """The index group of the QM atoms: `name` when the index holds it, else the one group
    whose name starts with QM."""
    groups = [g for g, _ in ndx_groups(ndx.read_text())]
    if name and name in groups:
        chosen = name
    else:
        qm = [g for g in groups if g.upper().startswith("QM")]
        if len(qm) != 1:
            return None
        chosen = qm[0]
    return SystemFact(key="QMMM-grps", value=chosen, rule="qm_group", read=[str(ndx)],
                      sha256=_hash(ndx))


def atom_count(structure: Path) -> SystemFact | None:
    """The atoms a .gro holds, from its second line: what a run's ranks divide."""
    try:
        with structure.open() as fh:
            fh.readline()
            count = int(fh.readline().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return SystemFact(key="atoms", value=str(count), rule="atoms", read=[str(structure)],
                      sha256=_hash(structure))


def system_facts(top: Path | None = None, ndx: Path | None = None,
                 structure: Path | None = None, data_dir: Path | None = None,
                 rules: set[str] | None = None) -> dict[str, SystemFact]:
    """Every fact the given files hold, by the engine key it decides (``forcefield`` by its
    own name); ``rules`` keeps only the ones named."""
    def wanted(rule: str) -> bool:
        return rules is None or rule in rules

    search = [data_dir] if data_dir is not None else []
    facts: list[SystemFact] = []
    if top is not None:
        if wanted("forcefield") and (ff := forcefield(top)) is not None:
            facts.append(ff)
        if wanted("coupling_groups"):
            facts.append(coupling_groups(
                top, search, ndx, data_dir / "residuetypes.dat" if data_dir else None))
        if wanted("restraint_defines"):
            facts.append(restraint_defines(top, search))
    if ndx is not None and wanted("qm_group") and (fact := qm_group(ndx)) is not None:
        facts.append(fact)
    if structure is not None and wanted("starting_state"):
        facts += starting_state(structure)
    if structure is not None and wanted("atoms") and (fact := atom_count(structure)) is not None:
        facts.append(fact)
    return {f.key: f for f in facts}


def subsystems_of(top: Path, search_dirs: list[Path],
                  residuetypes: Path | None = None) -> dict[str, dict]:
    """The system's parts as its topology lays them out, by what each molecule type is: the
    protein chains, nucleic acids, the solvent, the ions, the membrane's lipids, each
    cofactor and each other molecule — one ``ligand``, or each by its own name when there
    are several —, with the residue names its atoms carry and its net charge over its
    copies (None for a molecule whose definition was not read). A proposal the model
    confirms or corrects."""
    from myrmidon.knowledge.residues import residue_classes

    text, _ = expanded_topology(top, search_dirs)
    layout = molecule_layout(text)
    resnames: dict[str, list[str]] = {}
    charges: dict[str, float] = {}
    for mol, section, f in itp_rows(text):
        if section != "atoms" or mol is None or len(f) < 7:
            continue
        if f[3] not in resnames.setdefault(mol, []):
            resnames[mol].append(f[3])
        with contextlib.suppress(ValueError):
            charges[mol] = charges.get(mol, 0.0) + float(f[6])
    types = {}
    if residuetypes is not None and residuetypes.is_file():
        types = {f[0]: f[1] for line in residuetypes.read_text().splitlines()
                 if len(f := line.split()) >= 2}
    classes = residue_classes()
    parts: dict[str, dict] = {}
    singles: dict[str, list[tuple[str, int, int | None]]] = {"ligand": [], "cofactor": []}

    def add(name: str, kind: str, mol: str, copies: int, size: int | None,
            own_resnames: bool) -> None:
        part = parts.setdefault(name, {"kind": kind, "molecules": [], "resnames": []})
        if mol not in part["molecules"]:
            part["molecules"].append(mol)
        if own_resnames:
            names = resnames.get(mol) or [mol]
            part["resnames"] += [r for r in names if r not in part["resnames"]]
        if kind in ("ligand", "cofactor"):
            known = size is not None and part.get("charge", 0.0) is not None
            part["charge"] = (round(part.get("charge", 0.0) + charges.get(mol, 0.0) * copies,
                                    3) + 0.0) if known else None

    for mol, copies, size in layout:
        residues = {types.get(r) for r in resnames.get(mol, [])}
        named = {classes.get(r.upper()) for r in resnames.get(mol) or [mol]}
        if mol.upper().startswith("PROTEIN") or (residues and residues <= {"Protein"}):
            add("protein", "protein", mol, copies, size, False)
        elif (residues and residues <= {"DNA", "RNA"}) or mol.upper().startswith(
                ("DNA", "RNA")):
            add("nucleic", "nucleic", mol, copies, size, False)
        elif mol.upper() in WATER or residues == {"Water"}:
            add("solvent", "solvent", mol, copies, size, True)
        elif named == {"ions"} or residues == {"Ion"} or (size == 1):
            add("ions", "ions", mol, copies, size, True)
        elif named == {"membrane"}:
            add("membrane", "membrane", mol, copies, size, True)
        else:
            kind = "cofactor" if named == {"cofactor"} else "ligand"
            singles[kind].append((mol, copies, size))
    for kind, mols in singles.items():
        for mol, copies, size in mols:
            add(kind if len(mols) == 1 else mol.lower(), kind, mol, copies, size, True)
    return parts


def local_includes(top: Path, depth: int = 0) -> list[Path]:
    """The files a topology ``#include``s from beside itself, and theirs in turn: its
    molecules' .itp files and their position restraints. A force field's directory
    (``*.ff/``) is the force field's, named by the force field itself, and is left out."""
    out: list[Path] = []
    for line in top.read_text(errors="replace").splitlines():
        m = _INCLUDE.match(line)
        if not m or ".ff/" in m.group(1).replace("\\", "/"):
            continue
        found = top.parent / m.group(1)
        if found.is_file() and found not in out:
            out.append(found)
            if depth < 10:
                out += [p for p in local_includes(found, depth + 1) if p not in out]
    return out


declare_includes("topology", local_includes)
