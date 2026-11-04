"""Facts read from a MiMiC system's own files: the elements of the QM region, the
pseudopotential each one is given, and whether CPMD will find that file.

prepqm names a pseudopotential for every element, guessing `<El>_MT_BLYP.psp` for one its
table lacks, and CPMD stops with ECPFILE NOT FOUND only after the job is scheduled; the
file is checked here, before anything is submitted.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

from myrmidon.core.validation import Finding, Precondition

_ELEMENT = re.compile(r"[A-Z][a-z]?")
_COORD = re.compile(r"^[-+]?\d*\.\d+([eE][-+]?\d+)?\s+[-+]?\d*\.\d+")


def qm_pseudopotentials(sections: dict[str, list[str]]) -> dict[str, str]:
    """pseudopotential file -> element, for each species line (`*FILE …`) of &ATOMS."""
    out: dict[str, str] = {}
    for line in sections.get("ATOMS", []):
        s = line.strip()
        if s.startswith("*") and len(s) > 1:
            name = s[1:].split()[0]
            m = _ELEMENT.match(name)
            out[name] = m.group(0) if m else name
    return out


#: the elements by atomic number, to 86: an electron count's parity is the atoms'
ELEMENTS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
    "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc",
    "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn",
)
#: the transition metals and lanthanides with open d or f shells (group 12, d10, is closed):
#: a region holding one has no spin state its electron count decides
OPEN_SHELL = ({*ELEMENTS[20:29], *ELEMENTS[38:47], *ELEMENTS[56:79]}
              - {"Zn", "Cd", "Hg"})


def qm_elements(sections: dict[str, list[str]]) -> dict[str, int] | None:
    """element -> number of QM atoms, read from &ATOMS: each species line (`*FILE …`, the
    element the file name's first part), its LMAX line, then the count of the coordinates
    that follow. None when any line cannot be read so — an element not named, a count not
    given, counts that do not add up to the coordinate lines: a count read wrongly is a
    spin state guessed."""
    out: dict[str, int] = {}
    element, coords, counted = None, 0, 0
    for line in sections.get("ATOMS", []):
        s = line.strip()
        if s.startswith("*") and len(s) > 1:
            if element is not None:
                return None  # a species whose count never came
            name = s[1:].split()[0].split("_")[0].split(".")[0].capitalize()
            if name not in ELEMENTS:
                return None
            element = name
        elif element is not None and not s.upper().startswith(("LMAX", "LOC", "SKIP")):
            first = s.split()[0] if s.split() else ""
            if not first.isdigit():
                return None
            out[element] = out.get(element, 0) + int(first)
            counted += int(first)
            element = None
        elif element is None and _COORD.match(s):
            coords += 1
    if element is not None or not out or counted != coords:
        return None
    return out


def electron_count(sections: dict[str, list[str]], charge: int) -> int | None:
    """The QM region's all-electron count at this charge; None when &ATOMS cannot be read
    atom by atom."""
    counts = qm_elements(sections)
    if counts is None:
        return None
    return sum((ELEMENTS.index(e) + 1) * n for e, n in counts.items()) - charge


def spin_state(sections: dict[str, list[str]], charge: int) -> tuple[int | None, str]:
    """(the multiplicity the QM region's electrons decide, why) — 1 for an even count with
    no open-shell metal — or (None, what leaves it open): an odd count, an open-shell metal,
    or atoms this input does not list readably. The count is all-electron: its parity is the
    valence electrons' for pseudopotentials with closed-shell cores."""
    counts = qm_elements(sections)
    if counts is None:
        return None, "the input's &ATOMS cannot be read atom by atom"
    metals = sorted(set(counts) & OPEN_SHELL)
    if metals:
        return None, f"the QM region holds {', '.join(metals)}, whose spin state is chemistry"
    electrons = electron_count(sections, charge)
    if electrons % 2:
        return None, (f"{electrons} electrons (all-electron count): an odd count is a "
                      "doublet or a higher state")
    return 1, (f"{electrons} electrons (all-electron count), no open-shell metal: a "
               "closed-shell singlet")


def pp_library(environ: Mapping[str, str] = os.environ) -> Path | None:
    """Where CPMD looks for pseudopotentials: CPMD_PP_LIBRARY_PATH, else PP_LIBRARY_PATH
    (the engine environment sets them); None when neither is set."""
    for var in ("CPMD_PP_LIBRARY_PATH", "PP_LIBRARY_PATH"):
        if environ.get(var):
            return Path(environ[var])
    return None


def pp_findings(sections: dict[str, list[str]], run_dir: Path | None,
                environ: Mapping[str, str] = os.environ) -> tuple[list[Finding], bool]:
    """A QM element whose pseudopotential CPMD would not find: in the library, then in the
    run directory, as CPMD looks. Returns the findings and whether the check could run
    (it cannot without a library path)."""
    library = pp_library(environ)
    if library is None:
        return [], False
    out = []
    for name, element in qm_pseudopotentials(sections).items():
        if (library / name).is_file() or (run_dir is not None and (run_dir / name).is_file()):
            continue
        out.append(Precondition(
            id="QM_ELEMENT_NOT_COVERED",
            message=f"{element} in the QM region is given the pseudopotential {name}, which "
            f"neither {library} nor the run directory holds; CPMD would stop with ECPFILE "
            "NOT FOUND after the job starts",
            suggestion=f"add a pseudopotential for {element} of the method's family to the "
            "library and name it in pp_info, or leave the element out of the QM region"))
    return out, True
