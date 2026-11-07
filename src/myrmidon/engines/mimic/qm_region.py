"""MiMiCPy's prepqm as a program: its binary and arguments, the commands it has and those it
refused, the charge it rounded, the elements it guessed, and what a selection leaves of the
region's chemistry."""

from __future__ import annotations

import re
from pathlib import Path

from myrmidon.core.tool import ToolContext
from myrmidon.core.validation import Check, Finding
from myrmidon.engines.mimic.system import (
    electron_count,
    heavy_species_all_s,
    spin_state,
    valence_electrons,
)
from myrmidon.errors import EngineNotFoundError

#: the selection commands a run and a parameter fit read back, written beside the input
SELECTION_FILE = "sele.dat"
#: the commands prepqm's prompt dispatches (mimicpy/cli/prepqm.py). 'view' and 'help' only
#: print, and 'q'/'quit' end the input, which the tool appends itself
SELECTION_COMMANDS = ("add", "add-bound", "delete", "clear")
#: prepqm sums the selected atoms' topology charges into &SYSTEM CHARGE, and says so on
#: stderr when the sum is not an integer to two decimals (mimicpy/core/prepare.py)
_ROUNDED = re.compile(r"Total charge of QM region is (-?[\d.]+(?:e-?\d+)?), Rounding")
#: a row of the table prepqm prints for atom types it guessed an element for
_GUESSED_ROW = re.compile(r"^\|\s*(\S+)\s*\|\s*([A-Z][a-z]?)\s*\|\s*$")


def mimicpy_bin(ctx: ToolContext) -> str:
    info = ctx.engines.get("mimicpy")
    if info is None or not info.available:
        raise EngineNotFoundError("mimicpy", info.hint if info else "")
    return info.path


def prepqm_argv(binary: str, topology: Path, coords: Path, stage: Path, qm_group: str,
                padding_nm: float, data_dir: str | None, template: Path | None,
                pp_info: Path | None, atomtypes_map: dict[str, str]) -> list[str]:
    """The prepqm command that writes the region into ``stage``; a map of atom types to
    elements is written beside it as prepqm's two-column non-standard atom types file."""
    argv = [binary, "prepqm", "-top", str(topology), "-coords", str(coords),
            "-out", "cpmd.inp", "-ndx", "index.ndx", "-qma", qm_group,
            "-path", str(stage.resolve()), "-pad", str(padding_nm)]
    if data_dir:
        argv += ["-ff", data_dir]  # where the topology's #include'd .ff live
    if template is not None:
        argv += ["-inp", str(template)]
    if pp_info is not None:
        argv += ["-pp", str(pp_info)]
    if atomtypes_map:
        nsa = stage / "atomtypes.dat"
        nsa.write_text("".join(f"{t} {e}\n" for t, e in atomtypes_map.items()))
        argv += ["-nsa", str(nsa)]
    return argv


def unknown_commands(selections: list[str]) -> list[str]:
    """The selection commands prepqm does not have: it reads an unknown one, prints that it
    is invalid and carries on, so a verb such as 'rm' leaves its atoms in the region with
    exit 0 (MiMiCPy 0.3.0)."""
    unknown = []
    for cmd in selections:
        verb, _, rest = cmd.strip().partition(" ")
        verb = verb.lower()
        if verb not in SELECTION_COMMANDS or (verb == "clear") == bool(rest.strip()):
            unknown.append(cmd)
    return unknown


# What prepqm prints when it skips a command: it prints the error and goes on to write the
# region from the commands it did apply, exit 0 (tests/fixtures/mimicpy/
# prepqm_rejected_commands.out). Each line follows the prompt of the command it rejects.
_REJECTION = re.compile(
    r"is an invalid command!|did not return any atoms|is not a valid (?:selection keyword"
    r"|logical operator|boolean operator)|bracket is missing in selection|selection cannot "
    r"be empty|were not found in the coordinates file"
)


def rejected_commands(stdout: str, selections: list[str]) -> list[tuple[str, str]]:
    """(command, prepqm's message) for each command prepqm printed an error for and skipped.

    After the line inviting a selection, every prompt '> ' is followed by what that command
    printed, so the n-th segment belongs to the n-th command."""
    _, _, prompts = stdout.partition("Please enter selection below")
    segments = prompts.split("\n", 1)[-1].split("> ")[1:]
    out = []
    for i, segment in enumerate(segments):
        message = segment.strip().splitlines()[0] if segment.strip() else ""
        if _REJECTION.search(message):
            out.append((selections[i] if i < len(selections) else "?", message))
    return out


def rounded_charge(output: str) -> float | None:
    """The non-integer sum prepqm says it rounded into &SYSTEM CHARGE; None when it did not."""
    found = _ROUNDED.search(output)
    return float(found.group(1)) if found else None


def guessed_elements(output: str) -> dict[str, str]:
    """Atom type -> the element prepqm guessed for it, from the table it prints for types the
    topology gives no atomic number (tests/fixtures/mimicpy/prepqm_acetone.err)."""
    _, _, table = output.partition("They were guessed as follows")
    return {m.group(1): m.group(2) for line in table.splitlines()
            if (m := _GUESSED_ROW.match(line.strip())) and m.group(1) != "Atom"}


def region_chemistry(sections: dict[str, list[str]], charge: int | None,
                     selections: list[str], guessed: dict[str, str]
                     ) -> tuple[dict, list[Finding]]:
    """What the selection decides of the region's chemistry, for its sidecar, and what CPMD
    would run silently on it. ``charge`` is the integer the selected atoms' topology charges
    sum to, None when they sum to no integer. A region with boundary atoms has an electron
    count their pseudopotentials' valence decides, so neither count nor spin is derived here."""
    bound = any(cmd.strip().lower().startswith("add-bound") for cmd in selections)
    multiplicity, why, electrons, total = None, "", None, None
    if bound:
        why = "the region has boundary atoms, whose pseudopotentials' valence decides its electrons"
    elif charge is None:
        why = "the selected atoms' topology charges sum to no integer, so the charge is open"
    else:
        multiplicity, why = spin_state(sections, charge)
        electrons = valence_electrons(sections, charge)
        total = electron_count(sections, charge)
    chemistry = {"charge": charge, "multiplicity": multiplicity, "spin_why": why,
                 "boundary_atoms": bound,
                 "electrons": electrons, "electrons_all": total, "guessed_elements": guessed}
    return chemistry, region_findings(total, multiplicity, heavy_species_all_s(sections))


# real prepqm on a topology with no -pp writes every species LMAX=S
PP_LMAX_ALL_S = Check(id="PP_LMAX_ALL_S_ON_HEAVY_ATOMS",
                      sources=("tests/fixtures/mimicpy/prepqm_acetone.inp",),
                      sha256="e5855dfe71e058a84a07e23b2bbac4415c14930a1e634392aedcb6d74dad15af")
MULTIPLICITY_PARITY = Check(id="QM_MULTIPLICITY_PARITY", sources=(
    "https://gitlab.com/mimic-project/cpmd/-/tree/72922c3/doc/manual/manual.tex"
    "#system-multiplicity",))
CHECKS = (PP_LMAX_ALL_S, MULTIPLICITY_PARITY)


def region_findings(electrons: int | None, multiplicity: int | None,
                    heavy_lmax_s: list[str] = ()) -> list[Finding]:
    """What CPMD would run silently on this region: a spin its electrons cannot have, heavy
    atoms whose pseudopotentials prepqm wrote with no p channel."""
    out: list[Finding] = []
    if heavy_lmax_s:
        out.append(PP_LMAX_ALL_S.finding(
            f"the QM region's {', '.join(heavy_lmax_s)} carry LMAX=S",
            means="prepqm gives every species LMAX=S when no pseudopotential table names its "
            "channels; a non-hydrogen atom needs its p channel (a norm-conserving "
            "Troullier-Martins C, N or O is LMAX=P), or CPMD computes its nonlocal term with too "
            "few projectors and says nothing",
            suggestion="pp_info naming each element's file and LMAX, and select the region "
            "again"))
    if (electrons is not None and multiplicity is not None
            and electrons % 2 != (multiplicity - 1) % 2):
        out.append(MULTIPLICITY_PARITY.finding(
            f"multiplicity {multiplicity} for {electrons} electrons (all-electron count)",
            means="a multiplicity of 2S+1 needs 2S unpaired electrons: an even count cannot be "
            "a doublet, an odd one cannot be a singlet; CPMD would run a state the region "
            "cannot have",
            suggestion="check the QM region's charge and the multiplicity against its "
            "electrons"))
    return out
