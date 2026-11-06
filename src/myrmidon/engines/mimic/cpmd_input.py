"""CPMD input parsing and patching.

CPMD inputs are &SECTION ... &END blocks with keyword lines whose values sit on the
following line(s)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from myrmidon.core.contracts import Property, declare_alias, declare_identity, define_kind
from myrmidon.core.validation import Finding
from myrmidon.knowledge.catalog import Catalog
from myrmidon.knowledge.sensors import evaluate

_SECTION_RE = re.compile(r"^\s*&(\w+)", re.IGNORECASE)
_END_RE = re.compile(r"^\s*&END", re.IGNORECASE)


def parse_cpmd(text: str) -> dict[str, list[str]]:
    """Split a CPMD input into {SECTION: [lines]} (section names uppercased)."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if _END_RE.match(line):
            current = None
            continue
        m = _SECTION_RE.match(line)
        if m and current is None:
            current = m.group(1).upper()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line.rstrip())
    return sections


def keyword_value(sections: dict[str, list[str]], key: str) -> str | None:
    """What a CPMD input sets for ``&SECTION/KEYWORD``, as written: the value line after a
    line that is the keyword (``CUTOFF`` then ``70.``), else the words after it on a line
    no value line follows (``FUNCTIONAL BLYP``), else ``True`` for a keyword that stands
    alone; None when the section has no such keyword. A line with more words and a value
    line after it is a longer keyword (``TEMPERATURE RAMP``), not this one."""
    if not key.startswith("&") or "/" not in key:
        return None
    section, keyword = key[1:].split("/", 1)
    keyword = keyword.strip().upper()
    lines = sections.get(section.upper(), [])
    for i, line in enumerate(lines):
        text = line.strip().upper()
        if text != keyword and not text.startswith(keyword + " "):
            continue
        rest = line.strip()[len(keyword):].strip()
        valued = i + 1 < len(lines) and re.match(r"^\s*[-+\d.]", lines[i + 1])
        if rest:
            if valued:
                continue
            return rest
        return lines[i + 1].strip() if valued else "True"
    return None


def sections_from_values(values: dict[str, object]) -> dict[str, list[str]]:
    """Build CPMD sections from inputset keys of the form ``&SECTION/KEYWORD LINE``.

    CPMD keywords carry their arguments in two places: inline, as part of the keyword
    line (``FUNCTIONAL BLYP``, ``POISSON SOLVER TUCKERMAN``), and on the following line
    for numeric parameters (``CUTOFF`` then ``70.``). An input set expresses both by
    putting the whole keyword line in the key and whatever follows it in the value:
    ``True`` means the keyword stands alone, anything else becomes the next line.
    """
    sections: dict[str, list[str]] = {}
    for key, value in values.items():
        if value is None or value is False:
            continue
        if "/" not in key or not key.startswith("&"):
            raise ValueError(
                f"{key!r} is not a CPMD parameter — use '&SECTION/KEYWORD', e.g. "
                "'&SYSTEM/CUTOFF'"
            )
        section, keyword = key[1:].split("/", 1)
        lines = sections.setdefault(section.upper(), [])
        lines.append(f" {keyword.strip()}")
        if value is not True:
            lines.append(f"  {value}")
    return sections


def render_cpmd_from_values(
    values: dict[str, object],
    header: list[str] | None = None,
) -> str:
    """Render a CPMD input from inputset parameters.

    Provenance goes in an ``&INFO`` section rather than in comments: CPMD echoes
    &INFO into the run output, so the input set that produced the input is printed
    next to the results it produced. The per-key origin map lives in the sidecar.
    """
    sections: dict[str, list[str]] = {}
    if header:
        sections["INFO"] = [f" {line}" for line in header]
    for name, lines in sections_from_values(values).items():
        sections.setdefault(name, []).extend(lines)
    return render_cpmd(sections)


def render_cpmd(sections: dict[str, list[str]]) -> str:
    out: list[str] = []
    for name, lines in sections.items():
        out.append(f"&{name}")
        out.extend(lines)
        out.append("&END")
        out.append("")
    return "\n".join(out)


def get_keyword(lines: list[str], keyword: str) -> str | None:
    """Value on the line following `keyword` (CPMD convention)."""
    for i, line in enumerate(lines):
        if line.strip().upper().startswith(keyword.upper()):
            if i + 1 < len(lines):
                return lines[i + 1].strip()
            return ""
    return None


def has_keyword(lines: list[str], keyword: str) -> bool:
    return any(line.strip().upper().startswith(keyword.upper()) for line in lines)


# What a CPMD input decides about a coupled run's outputs: force-matching reference data
# is written only with FORCE_MATCHING in &CPMD. An input set's flat values are read as
# the sections they render.
declare_alias("cpmd_inp", "cpmd_input")
define_kind(
    "cpmd_input", lambda path: parse_cpmd(path.read_text()),
    Property("fm_reference", lambda s: has_keyword(s.get("CPMD", []), "FORCE_MATCHING")),
    normalize=lambda d: d if all(isinstance(v, list) for v in d.values())
    else sections_from_values(d),
    lookup=keyword_value,
)
#: the CPMD half of a coupled system: the wavefunction and the coordinates a stage
#: restarts from
declare_identity("restart", "geometry")


def drop_keyword(lines: list[str], keyword: str) -> list[str]:
    """Remove a keyword line and its value line (if the next line is a value)."""
    out: list[str] = []
    skip_value = False
    for line in lines:
        if skip_value:
            skip_value = False
            if re.match(r"^\s*[-+\d.]", line):
                continue
        if line.strip().upper().startswith(keyword.upper()):
            skip_value = True
            continue
        out.append(line)
    return out


_BOHR_PER_ANGSTROM = 1.8897259886
_FLOAT = r"(-?\d+\.\d*(?:[eE][-+]?\d+)?)"
_COORD_RE = re.compile(rf"^\s*{_FLOAT}\s+{_FLOAT}\s+{_FLOAT}\s*$")


def qm_geometry(sections: dict[str, list[str]]) -> dict | None:
    """QM cell (bohr) and QM-atom extent (bohr) parsed from &SYSTEM/&ATOMS.

    Returns {cell_bohr, extent_bohr, min_vacuum_bohr, n_atoms} or None when the
    input has no CELL or no coordinates. Handles ``CELL`` (a, b/a, c/a, cosines),
    ``CELL ABSOLUTE`` (a, b, c) and the ``ANGSTROM`` unit switch.
    """
    system = sections.get("SYSTEM", [])
    cell_line = None
    absolute = False
    for i, line in enumerate(system):
        up = line.strip().upper()
        if up.startswith("CELL"):
            absolute = "ABSOLUTE" in up
            if i + 1 < len(system):
                cell_line = system[i + 1]
            break
    coords = [
        tuple(float(v) for v in m.groups())
        for line in sections.get("ATOMS", [])
        if (m := _COORD_RE.match(line))
    ]
    if cell_line is None or not coords:
        return None
    try:
        nums = [float(v) for v in cell_line.split()[:3]]
    except ValueError:
        return None
    if len(nums) < 3:
        return None
    a, b, c = nums
    cell = [a, b, c] if absolute else [a, a * b, a * c]
    scale = _BOHR_PER_ANGSTROM if has_keyword(system, "ANGSTROM") else 1.0
    cell = [v * scale for v in cell]
    extent = [
        (max(p[k] for p in coords) - min(p[k] for p in coords)) * scale for k in range(3)
    ]
    vacuum = min((cell[k] - extent[k]) / 2.0 for k in range(3))
    return {
        "cell_bohr": [round(v, 3) for v in cell],
        "extent_bohr": [round(v, 3) for v in extent],
        "min_vacuum_bohr": round(vacuum, 3),
        "n_atoms": len(coords),
    }


class DistanceRestraint(BaseModel):
    """Harmonic distance restraint between two CPMD atoms (&ATOMS input order,
    1-based — NOT gro serials, NOT dataset record order)."""

    atom_i: int = Field(ge=1)
    atom_j: int = Field(ge=1)
    target_a: float = Field(gt=0, description="target distance (Å)")
    k_au: float = Field(default=0.5, gt=0, description="force constant (a.u.)")


def add_distance_restraints(
    sections: dict[str, list[str]], restraints: list[DistanceRestraint]
) -> dict[str, list[str]]:
    """Append a CONSTRAINTS/RESTRAINTS block to &ATOMS (CPMD coninp syntax:
    ``RESTRAINTS n`` then ``DIST i j value k`` lines, distances in bohr)."""
    if not restraints:
        return sections
    atoms = sections.get("ATOMS", [])
    if has_keyword(atoms, "CONSTRAINTS"):
        raise ValueError("&ATOMS already has a CONSTRAINTS block")
    # CPMD's coninp reads the restraint count from the line AFTER the keyword
    block = ["CONSTRAINTS", "RESTRAINTS", f"  {len(restraints)}"]
    for r in restraints:
        block.append(
            f"  DIST {r.atom_i} {r.atom_j} "
            f"{r.target_a * _BOHR_PER_ANGSTROM:.5f} {r.k_au:.4f}"
        )
    block.append("END CONSTRAINTS")
    out = dict(sections)
    out["ATOMS"] = [*atoms, *block]
    return out


def set_paths(sections: dict[str, list[str]], run_dir: str) -> dict[str, list[str]]:
    """Point &MIMIC PATHS at ``run_dir`` (single-path layout: count line + path line)."""
    mimic = list(sections.get("MIMIC", []))
    for i, line in enumerate(mimic):
        if line.strip().upper().startswith("PATHS") and i + 2 < len(mimic):
            mimic[i + 1] = "        1"
            mimic[i + 2] = run_dir
            break
    out = dict(sections)
    out["MIMIC"] = mimic
    return out


def paths_dir(sections: dict[str, list[str]]) -> str | None:
    """The single directory &MIMIC PATHS names, or None when it says something else.

    The value is not a parameter: it is where the two coupled programs find each
    other, so it has to be the directory the job actually runs in.
    """
    mimic = sections.get("MIMIC", [])
    for i, line in enumerate(mimic):
        if line.strip().upper().startswith("PATHS") and i + 2 < len(mimic):
            if mimic[i + 1].strip() != "1":
                return None  # several paths: not ours to rewrite
            return mimic[i + 2].strip()
    return None


_KEYWORD_LINE = re.compile(r"^\s*[A-Z][A-Z_-]")


def keyword_blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    """A section's lines as (first word of the keyword, the keyword line and the value lines
    under it). Lines before the first keyword come back under ""."""
    blocks: list[tuple[str, list[str]]] = []
    for line in lines:
        if _KEYWORD_LINE.match(line):
            blocks.append((line.split()[0].upper(), [line]))
        elif blocks:
            blocks[-1][1].append(line)
        else:
            blocks.append(("", [line]))
    return blocks


#: the sections whose keywords the catalog checks; &ATOMS is pseudopotentials and
#: coordinates, &INFO free text
CHECKED_SECTIONS = ("CPMD", "SYSTEM", "DFT", "MIMIC")


def keyword_status(sections: dict[str, list[str]], catalog: Catalog) -> list[dict]:
    """Each keyword line as the CPMD version's catalog sees it — unknown, decided by
    nothing the harness reasons about (out_of_scope), or known — the subject the per-key
    sensor records read."""
    out = []
    for section in CHECKED_SECTIONS:
        for line in sections.get(section, []):
            if not _KEYWORD_LINE.match(line) or "/" in line:
                continue
            key = catalog.keyword(section, line)
            out.append({"section": section, "line": line.strip(), "version": catalog.version,
                        "name": key.name if key is not None else "",
                        "status": "unknown" if key is None
                        else "known" if _decided(key) else "out_of_scope"})
    return out


def _decided(key) -> bool:
    """A keyword the harness reasons about: one a stage template sets, or one prepqm writes
    from the system."""
    from myrmidon.knowledge.stage import decided_keys

    name = f"{key.section}/{key.name}"
    return name in PREPQM_KEYWORDS or name in decided_keys("cpmd")


def cpmd_subject(sections: dict[str, list[str]], restart_from: str | None = None) -> dict:
    """The facts of a CPMD input the sensor records read: the &MIMIC box, the QM cell's
    vacuum around the QM atoms, RESTART and restraints, MAXSTEP and whether the run
    optimises a wavefunction, and the restart a call names."""
    cpmd = sections.get("CPMD", [])
    geom = qm_geometry(sections) or {}
    maxstep = get_keyword(cpmd, "MAXSTEP")
    return {
        "has_mimic_section": "MIMIC" in sections,
        "has_mimic_box": get_keyword(sections.get("MIMIC", []), "BOX") is not None,
        "min_vacuum_bohr": geom.get("min_vacuum_bohr"),
        "cell_bohr": "x".join(str(v) for v in geom.get("cell_bohr", [])) or None,
        "wants_restart": any(line.strip().upper().startswith("RESTART") for line in cpmd),
        "restart_from": restart_from,
        "has_restraints": has_keyword(sections.get("ATOMS", []), "RESTRAINTS"),
        "maxstep": maxstep.strip() if maxstep is not None else None,
        "optimising": has_keyword(cpmd, "WAVEFUNCTION OPTIMIZATION"),
    }


def cpmd_findings(sections: dict[str, list[str]], catalog: Catalog,
                  restart_from: str | None = None) -> list[Finding]:
    """What the sensor records make of a CPMD input (knowledge/rules/)."""
    return evaluate("cpmd_input", cpmd_subject(sections, restart_from),
                    version=catalog.version, keys=keyword_status(sections, catalog))


#: the keywords mimicpy's prepqm writes from the system and the researcher's charge: the QM
#: cell, the MiMiC paths, box and overlaps, the charge, multiplicity and spin. A stage input
#: takes these from the system input prepqm wrote, never from its method
PREPQM_KEYWORDS = frozenset({
    "SYSTEM/CELL", "SYSTEM/ANGSTROM", "MIMIC/OVERLAPS", "MIMIC/PATHS", "MIMIC/BOX",
    "SYSTEM/CHARGE", "SYSTEM/MULTIPLICITY", "CPMD/LSD"})


def apply_method(system: dict[str, list[str]], method: dict[str, list[str]],
                 catalog: Catalog | None = None) -> dict[str, list[str]]:
    """A stage's run input: the stage's purpose and method, on the system prepqm wrote once.

    A staged MiMiC run (anneal, NVE test, heat, produce) changes the purpose between stages
    and never the system, so the system input is made once from the QM selection and each
    stage takes from it ``&ATOMS`` and every keyword prepqm writes (``PREPQM_KEYWORDS``);
    every other section and keyword is the stage's own. Sections keep the system input's
    order.
    """
    from myrmidon.knowledge.catalog import load

    catalog = catalog or load("cpmd")

    def owned(section: str, line: str) -> bool:
        key = catalog.keyword(section, line) if _KEYWORD_LINE.match(line) else None
        return key is not None and f"{key.section}/{key.name}" in PREPQM_KEYWORDS

    out: dict[str, list[str]] = {}
    for name in [*system, *[n for n in method if n not in system]]:
        if name == "ATOMS":
            out[name] = list(system.get(name, []))
        elif name not in method:
            continue  # a section the stage's method does not have is not part of it
        else:
            lines = [line for _, block in keyword_blocks(method[name])
                     if not owned(name, block[0]) for line in block]
            lines += [line for _, block in keyword_blocks(system.get(name, []))
                      if owned(name, block[0]) for line in block]
            out[name] = lines
    return out


def set_keyword(lines: list[str], keyword: str, value: str) -> list[str]:
    """Replace (or append) a keyword + value-line pair."""
    out = list(lines)
    for i, line in enumerate(out):
        if line.strip().upper().startswith(keyword.upper()):
            if i + 1 < len(out):
                out[i + 1] = f"  {value}"
            else:
                out.append(f"  {value}")
            return out
    out.append(keyword.upper())
    out.append(f"  {value}")
    return out
