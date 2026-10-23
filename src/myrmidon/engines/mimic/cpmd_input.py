"""CPMD input parsing/patching and the mimic_prepare review tool.

CPMD inputs are &SECTION ... &END blocks with keyword lines whose values sit
on the following line(s). MiMiCPy's prepqm generates the initial cpmd.inp;
mimic_prepare turns it into a reviewed, explicitly-confirmed run input —
charge and multiplicity are REQUIRED parameters here, never inferred.
"""

from __future__ import annotations

import json
import re
from typing import ClassVar

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import ValidationWarning
from myrmidon.engines.inputs import resolve_in_ws

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


#: what prepqm writes for a system — the QM atoms, where they sit among the MM atoms, the QM
#: cell, the charge. Everything else in a run input is the method a stage chooses.
SYSTEM_KEYWORDS = {"SYSTEM": ("CELL", "CHARGE", "MULTIPLICITY"),
                   "MIMIC": ("OVERLAPS", "PATHS", "BOX")}
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


def apply_method(system: dict[str, list[str]], method: dict[str, list[str]]
                 ) -> dict[str, list[str]]:
    """A stage's run input: the stage's method, on the system prepqm wrote once.

    A staged MiMiC run (anneal, NVE test, heat, produce) changes the method between stages
    and never the system, so the system input is made once from the QM selection and each
    stage takes from it ``&ATOMS`` and the keywords of :data:`SYSTEM_KEYWORDS`; every other
    section and keyword is the stage's own. Sections keep the system input's order.
    """
    out: dict[str, list[str]] = {}
    for name in [*system, *[n for n in method if n not in system]]:
        if name == "ATOMS":
            out[name] = list(system.get(name, []))
        elif name not in method:
            continue  # a section the stage's method does not have is not part of it
        elif name in SYSTEM_KEYWORDS:
            kept = SYSTEM_KEYWORDS[name]
            lines = [line for key, block in keyword_blocks(method[name]) if key not in kept
                     for line in block]
            lines += [line for key, block in keyword_blocks(system.get(name, []))
                      if key in kept for line in block]
            out[name] = lines
        else:
            out[name] = list(method[name])
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


class MimicPrepareParams(BaseModel):
    #: the QM region's charge and spin are the researcher's to confirm, never inferred
    RESEARCHER_FIELDS: ClassVar[tuple[str, ...]] = ("total_charge", "multiplicity")

    cpmd_input: str = Field(description="cpmd.inp generated by mimic_select_qm")
    total_charge: int = Field(
        description="net charge of the QM region (REQUIRED — confirm it against the "
        "topology and mimic_select_qm's report; never guess)"
    )
    multiplicity: int = Field(ge=1, description="spin multiplicity (REQUIRED)")
    maxsteps: int | None = Field(
        default=None,
        description="&CPMD MAXSTEP — MD steps under MOLECULAR DYNAMICS, but SCF "
                    "iterations under WAVEFUNCTION OPTIMIZATION (every single point). "
                    "Leave unset for a single point: CPMD's own limit is the right one, "
                    "and a small value returns an unconverged wavefunction with exit 0")
    timestep_au: float | None = Field(default=None, description="&CPMD TIMESTEP (a.u.)")
    fm_pace: int | None = Field(
        default=None, ge=1,
        description="&CPMD FORCE_MATCHING pace — write FM reference data every N MD steps",
    )
    distance_restraints: list[DistanceRestraint] = Field(
        default_factory=list,
        description="harmonic distance restraints (CPMD atom numbers, &ATOMS order). "
        "mimic_run warns on them (CPMD_RESTRAINTS_UNDER_MIMIC): CPMD builds without PLUMED "
        "hooks segfault applying them in a coupled run",
    )
    method: str | None = Field(
        default=None,
        description="a stage's method: the CPMD input inputset_apply rendered from the "
        "stage's input set (annealing, NVE test, heating, production). Its &CPMD, &DFT and "
        "method keywords are used; the system — &ATOMS, &MIMIC OVERLAPS/PATHS/BOX, &SYSTEM "
        "CELL — stays the one cpmd_input holds. One QM selection, every stage.",
    )
    output: str = Field(
        default="cpmd_run.inp",
        description="a bare filename is written beside the input it is made from; a "
        "path with a directory is workspace-relative, as every other tool's output is",
    )
    acknowledge_warnings: list[str] = Field(default_factory=list)


def _output_path(ctx: ToolContext, src, output: str):
    """Where the reviewed input is written.

    A bare filename lands beside the input it was made from. A path with a directory in
    it is workspace-relative, like every other tool's ``output``; resolved against the
    input's directory instead, ``02_heating/cpmd_run.inp`` would silently become
    ``02_heating/02_heating/...``. Returns None for a path that escapes.
    """
    from pathlib import Path

    candidate = Path(output)
    if candidate.is_absolute() or len(candidate.parts) > 1:
        root = ctx.workspace.root.resolve()
        out = (root / candidate).resolve() if not candidate.is_absolute() else candidate
        return out if str(out).startswith(str(root) + "/") else None
    return src.parent / output


class QMMMPrepObservation(Observation):
    charge: int | None = None
    multiplicity: int | None = None
    maxsteps: int | None = None
    timestep_au: float | None = None
    functional: str | None = None
    paths: int | None = None  # host-program paths declared in &MIMIC
    sections: list[str] = Field(default_factory=list)


def _confirmed_by(ctx: ToolContext, params: MimicPrepareParams) -> str:
    """Who stated the charge and multiplicity: the approved plan, or the person who
    accepted QM_CHARGE_MULTIPLICITY_UNCONFIRMED (the gate lets nothing else through)."""
    return "plan" if ctx.workspace is not None and not _spin_state_findings(ctx, params) \
        else "researcher"


def prepare_gate(ctx: ToolContext, params: MimicPrepareParams):
    """Where did the QM method settings come from? The cpmd.inp reaching this tool was
    built by prepqm from a template, and the template is what carries the functional,
    the cutoff and the coupling — so the question is about the template."""
    from myrmidon.core.sourcing import parameter_source, sourced_finding, unsourced_finding

    warnings: list[ValidationWarning] = []
    checked = ["parameter_source"]
    if ctx.workspace is not None:
        sources = [parameter_source(ctx.workspace, params.cpmd_input)]
        for finding in (unsourced_finding(sources), sourced_finding(sources)):
            if finding is not None:
                warnings.append(finding)
        from myrmidon.core.file_origin import edit_findings

        warnings += edit_findings(ctx, params.cpmd_input)
        warnings += _spin_state_findings(ctx, params)
        checked.append("charge_multiplicity_confirmed")
    return warnings, checked


def _spin_state_findings(ctx: ToolContext, params: MimicPrepareParams):
    """The QM region's charge and multiplicity are the researcher's: a required parameter
    makes a model state them, not confirm them, and a model left to itself writes charge 0
    and multiplicity 1 into an input that then records them as confirmed. A plan step a
    person approved with these values confirms them; otherwise a person does."""
    from myrmidon.core.plan_binding import approved_steps

    for step in approved_steps(ctx.workspace, "mimic_prepare"):
        if (step.values.get("total_charge"), step.values.get("multiplicity")) == (
                params.total_charge, params.multiplicity):
            return []
    return [ValidationWarning(
        id="QM_CHARGE_MULTIPLICITY_UNCONFIRMED", severity="warn", needs_human=True,
        message=f"charge {params.total_charge} and multiplicity {params.multiplicity} of "
        "the QM region are written into the run input, and nothing the researcher approved "
        "states them",
        assumption=f"the QM region carries charge {params.total_charge} with "
        f"multiplicity {params.multiplicity}",
        suggestion="put the charge and multiplicity to the researcher (mimic_select_qm "
        "reports the declared charge), or record them in a plan step they approve",
    )]


@tool(
    "mimic_prepare",
    description="Turn the generated cpmd.inp into a reviewed MiMiC run input. Charge "
    "and multiplicity are REQUIRED and written into &SYSTEM explicitly — confirm them "
    "against mimic_select_qm's report; the observation is the review surface.",
    returns=QMMMPrepObservation,
    tags={"mimic", "qmmm"},
    requires={"mimicpy"},
    gate=prepare_gate,
)
def mimic_prepare(ctx: ToolContext, params: MimicPrepareParams) -> QMMMPrepObservation:
    src = resolve_in_ws(ctx, params.cpmd_input)
    sections = parse_cpmd(src.read_text())
    if params.method:
        method = parse_cpmd(resolve_in_ws(ctx, params.method).read_text())
        if "CPMD" not in method:
            return QMMMPrepObservation(
                ok=False,
                summary=f"{params.method} has no &CPMD section, so it states no method",
                sections=sorted(method),
            )
        sections = apply_method(sections, method)
    if "MIMIC" not in sections:
        return QMMMPrepObservation(
            ok=False,
            summary=f"{params.cpmd_input} has no &MIMIC section — is this really a "
            "MiMiCPy-generated input?",
            sections=sorted(sections),
        )
    system = sections.setdefault("SYSTEM", [])
    sections["SYSTEM"] = set_keyword(system, "CHARGE", str(params.total_charge))
    cpmd_sec = sections.setdefault("CPMD", [])
    # CPMD accepts MULTIPLICITY only in a spin-polarized (LSD) run; for a
    # closed-shell singlet the keyword must be absent (CPMD aborts in sysin
    # otherwise, CPMD 4.3). The confirmed value always goes to
    # the sidecar so the gate can verify it was set explicitly.
    if params.multiplicity > 1:
        sections["SYSTEM"] = set_keyword(
            sections["SYSTEM"], "MULTIPLICITY", str(params.multiplicity)
        )
        if not has_keyword(cpmd_sec, "LSD"):
            cpmd_sec = ["LSD", *cpmd_sec]
            sections["CPMD"] = cpmd_sec
        spin_note = f"multiplicity {params.multiplicity} (LSD)"
    else:
        sections["SYSTEM"] = drop_keyword(sections["SYSTEM"], "MULTIPLICITY")
        spin_note = "multiplicity 1 (closed shell: no MULTIPLICITY keyword, as CPMD requires)"
    if params.maxsteps is not None:
        sections["CPMD"] = set_keyword(cpmd_sec, "MAXSTEP", str(params.maxsteps))
    if params.timestep_au is not None:
        sections["CPMD"] = set_keyword(
            sections["CPMD"], "TIMESTEP", str(params.timestep_au)
        )
    if params.fm_pace is not None:
        sections["CPMD"] = set_keyword(
            sections["CPMD"], "FORCE_MATCHING", str(params.fm_pace)
        )
    try:
        sections = add_distance_restraints(sections, params.distance_restraints)
    except ValueError as exc:
        return QMMMPrepObservation(ok=False, summary=str(exc), sections=sorted(sections))

    out = _output_path(ctx, src, params.output)
    if out is None:
        return QMMMPrepObservation(
            ok=False,
            summary=f"output '{params.output}' escapes the workspace",
            sections=sorted(sections),
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_cpmd(sections))
    sidecar = out.with_name(out.name + ".myrmidon.json")
    sidecar.write_text(
        json.dumps(
            {
                "charge": params.total_charge,
                "multiplicity": params.multiplicity,
                "confirmed_by": _confirmed_by(ctx, params),
                "source_input": src.name,
                "method": params.method,
            },
            indent=1,
        )
    )

    paths_value = get_keyword(sections.get("MIMIC", []), "PATHS")
    functional = None
    for line in sections.get("DFT", []):
        if line.strip().upper().startswith("FUNCTIONAL"):
            parts = line.split()
            functional = parts[-1] if len(parts) > 1 else None
    maxstep_val = get_keyword(sections.get("CPMD", []), "MAXSTEP")
    timestep_val = get_keyword(sections.get("CPMD", []), "TIMESTEP")
    return QMMMPrepObservation(
        ok=True,
        summary=f"MiMiC run input written: charge {params.total_charge}, {spin_note}"
        + (f", {maxstep_val} steps" if maxstep_val else "")
        + (f", FM pace {params.fm_pace}" if params.fm_pace else "")
        + (
            f", {len(params.distance_restraints)} distance restraint(s)"
            if params.distance_restraints
            else ""
        )
        + f" -> {ctx.workspace.relpath(out)}",
        artifacts=[
            ArtifactRef.from_path(out, "cpmd_input", root=ctx.workspace.root),
            ArtifactRef.from_path(sidecar, "run_metadata", root=ctx.workspace.root),
        ],
        charge=params.total_charge,
        multiplicity=params.multiplicity,
        maxsteps=int(maxstep_val) if maxstep_val and maxstep_val.isdigit() else None,
        timestep_au=float(timestep_val) if timestep_val else None,
        functional=functional,
        paths=int(paths_value) if paths_value and paths_value.isdigit() else None,
        sections=sorted(sections),
    )
