"""QM region selection via MiMiCPy's prepqm.

Drives ``mimicpy prepqm -top ... -coords ... -ff ... -path ...`` non-interactively by feeding
selection commands (MiMiCPy syntax, e.g. ``add resname is LIG``) to its
prompt. Produces index.ndx + cpmd.inp. The reported QM charge is whatever
MiMiCPy/the input declares — it is NOT confirmed until mimic_prepare."""

from __future__ import annotations

import re
import subprocess

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, precheck, tool
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.gromacs.common import tail
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.mimic.cpmd_input import get_keyword, parse_cpmd, qm_geometry
from myrmidon.engines.mimic.qm_region import mimicpy_bin, rejected_commands
from myrmidon.engines.mimic.system import pp_findings, qm_pseudopotentials

# prepqm sums the selected atoms' topology charges into &SYSTEM CHARGE, and says so on
# stderr when the sum is not an integer to two decimals (mimicpy/core/prepare.py)
_ROUNDED_RE = re.compile(r"Total charge of QM region is (-?[\d.]+(?:e-?\d+)?), Rounding")




class SelectQMParams(BaseModel):
    topology: str = Field(description="GROMACS topology (.top)")
    coords: str = Field(description="coordinate file (.gro)")
    selections: list[str] = Field(
        min_length=1,
        description="MiMiCPy selection commands, applied in order: 'add SEL', 'add-bound SEL' "
        "(adds them marked as boundary atoms, which get pp_info's boundary pseudopotential), "
        "'delete SEL', 'clear'. SEL is 'KEYWORD is VALUE' over the topology (resname, resid, name, "
        "id, ...) joined by 'and'/'or', e.g. 'add resname is LIG' or 'add resid is 42 and "
        "name is OG'; the residue and atom names are this system's",
    )
    padding_nm: float = Field(
        gt=0,
        description="vacuum between the QM atoms and the QM cell wall, per side (nm). "
        "REQUIRED: prepqm's default of 0 makes the cell the atoms' bounding box, which "
        "crashes CPMD; 0.3-0.5 nm is typical for an isolated-molecule Poisson solver",
    )
    template: str | None = Field(
        default=None, description="CPMD template input (&CPMD/&DFT/&SYSTEM settings) — prepqm -inp"
    )
    pp_info: str | None = Field(
        default=None,
        description="pseudopotential table (element, file, ..., LMAX) — prepqm -pp; the file "
        "names must exist in CPMD's PP_LIBRARY_PATH",
    )
    qm_group: str = Field(default="QMatoms", description="index group name for the QM atoms")
    stage: str = Field(
        default="qmmm",
        description="stage the coupled runs belong to — its absolute path is written into "
        "&MIMIC PATHS, so name the same stage in mimic_run, which re-points PATHS to the "
        "run's own directory under it",
    )


# The commands prepqm's prompt dispatches (mimicpy/cli/prepqm.py). 'view' and 'help' only
# print, and 'q'/'quit' end the input, which the tool appends itself.
SELECTION_COMMANDS = ("add", "add-bound", "delete", "clear")


def select_qm_gate(ctx: ToolContext, params: SelectQMParams):
    """Refuse a selection command prepqm does not have, before it runs.

    prepqm reads an unknown command, prints that it is invalid and carries on: the region it
    writes is the one the other commands make: a verb it does not have, such as 'rm',
    leaves its atoms in the QM region with exit 0 (MiMiCPy 0.3.0).
    """
    unknown = []
    for cmd in params.selections:
        verb, _, rest = cmd.strip().partition(" ")
        verb = verb.lower()
        if verb not in SELECTION_COMMANDS or (verb == "clear") == bool(rest.strip()):
            unknown.append(cmd)
    if not unknown:
        return [], ["selection_commands"]
    return [Precondition(
        id="QM_SELECTION_COMMAND_UNKNOWN",
        message=f"prepqm has no command for {unknown}: it would skip "
        f"{'it' if len(unknown) == 1 else 'them'} and write the region the other commands "
        "make. Its commands are 'add SEL', 'add-bound SEL', 'delete SEL' and 'clear'.",
        suggestion="'delete SEL' removes atoms; each command is one verb and one selection",
    )], ["selection_commands"]


@tool(
    "mimic_select_qm",
    description="Select the QM region with MiMiCPy prepqm; writes index.ndx and "
    "cpmd.inp. The QM charge in the generated input is reported but UNCONFIRMED — "
    "it must be explicitly confirmed via mimic_prepare before any run.",
    tags={"mimic", "qmmm"},
    requires={"mimicpy"},
    contract=Contract(consumes={"topology": "topology", "coords": "structure",
                                "template": "cpmd_input"},
                      produces=(Produces("cpmd_input"), Produces("index"))),
)
def mimic_select_qm(ctx: ToolContext, params: SelectQMParams) -> Observation:
    precheck(ctx, select_qm_gate(ctx, params)[0])
    stage = ctx.workspace.stage(params.stage)
    top = resolve_in_ws(ctx, params.topology)
    coords = resolve_in_ws(ctx, params.coords)
    argv = [
        mimicpy_bin(ctx), "prepqm", "-top", str(top), "-coords", str(coords),
        "-out", "cpmd.inp", "-ndx", "index.ndx", "-qma", params.qm_group,
        "-path", str(stage.resolve()), "-pad", str(params.padding_nm),
    ]
    gmx_info = ctx.engines.get("gromacs")
    if gmx_info is not None and gmx_info.data_dir:
        argv += ["-ff", gmx_info.data_dir]  # where the topology's #include'd .ff live
    if params.template:
        argv += ["-inp", str(resolve_in_ws(ctx, params.template))]
    if params.pp_info:
        argv += ["-pp", str(resolve_in_ws(ctx, params.pp_info))]
    stdin = "\n".join(params.selections) + f"\n{ctx.config.mimicpy_quit_cmd}\n"
    proc = subprocess.run(  # noqa: S603
        argv, cwd=stage, input=stdin, capture_output=True, text=True, timeout=600
    )
    ndx = stage / "index.ndx"
    inp = stage / "cpmd.inp"
    rejected = rejected_commands(proc.stdout, params.selections)
    if rejected:
        # the files exist, but they hold the region of the commands prepqm did not skip
        return Observation(
            ok=False,
            summary=f"prepqm skipped {len(rejected)} of {len(params.selections)} selection "
            "command(s); the region it wrote is not the one asked for",
            command=argv,
            findings=[Precondition(
                id="QM_SELECTION_REJECTED",
                message="; ".join(f"{cmd!r}: {msg}" for cmd, msg in rejected),
                suggestion="fix those commands and select again; a selection that matched "
                "nothing names atoms this topology does not have",
            )],
            log_tail=tail(proc.stdout, 40),
            selections=params.selections,
        )
    # mimicpy can exit 0 after an internal failure: judge by the artifacts, not rc
    if proc.returncode != 0 or not inp.exists() or not ndx.exists():
        # an empty region is not an error to prepqm: it writes nothing and exits 0
        return Observation(
            ok=False,
            summary=f"mimicpy prepqm failed (exit {proc.returncode}; cpmd.inp "
            f"{'written' if inp.exists() else 'missing'}, index.ndx "
            f"{'written' if ndx.exists() else 'missing'})",
            command=argv,
            log_tail=tail(proc.stderr + "\n" + proc.stdout, 40),
            selections=params.selections,
        )
    sections = parse_cpmd(inp.read_text())
    qm_atoms = None
    if "ATOMS" in sections:
        qm_atoms = sum(
            1 for line in sections["ATOMS"] if re.match(r"\s*-?\d+\.?\d*\s+-?\d", line)
        ) or None
    declared = get_keyword(sections.get("SYSTEM", []), "CHARGE")
    declared_charge = (
        int(float(declared)) if declared and re.match(r"^-?\d+(\.0*)?$", declared) else None
    )
    rounded = _ROUNDED_RE.search(proc.stderr + proc.stdout)
    rounded_from = float(rounded.group(1)) if rounded else None
    geom = qm_geometry(sections) or {}
    artifacts = [ArtifactRef.from_path(inp, "cpmd_input", root=ctx.workspace.root)]
    if ndx.exists():
        artifacts.append(ArtifactRef.from_path(ndx, "index", root=ctx.workspace.root))
    uncovered, _ = pp_findings(sections, stage)
    if uncovered:
        return Observation(
            ok=False,
            summary="the QM region holds an element CPMD has no pseudopotential for here: "
            + "; ".join(w.message for w in uncovered),
            command=argv, artifacts=artifacts, findings=uncovered,
            selections=params.selections, qm_atoms=qm_atoms,
            pseudopotentials=qm_pseudopotentials(sections),
        )
    return Observation(
        ok=True,
        summary=f"QM region selected: {qm_atoms if qm_atoms is not None else '?'} atoms; "
        f"declared charge {declared_charge if declared_charge is not None else 'absent'} "
        + (f"(prepqm rounded the selection's MM charges, {rounded_from:g}) "
           if rounded_from is not None else "")
        + "(mimic_prepare takes it when it is an integer sum, and asks otherwise)"
        + (
            f"; QM cell {'x'.join(str(v) for v in geom['cell_bohr'])} bohr, "
            f"min vacuum {geom['min_vacuum_bohr']} bohr"
            if geom
            else ""
        ),
        command=argv,
        artifacts=artifacts,
        findings=[*([Finding(
            id="QM_CHARGE_ROUNDED",
            message=f"the selected atoms' MM charges sum to {rounded_from:g}, which prepqm "
            f"rounded to {declared_charge}: a region that cuts a covalent bond or part of a "
            "residue's charge distribution sums to a non-integer, and its QM charge is the "
            "researcher's, not the rounding's",
            suggestion="say what the selection cuts, and put the charge to the researcher",
        )] if rounded_from is not None else [])],
        qm_atoms=qm_atoms,
        declared_charge=declared_charge,
        rounded_from=rounded_from,
        selections=params.selections,
        qm_cell_bohr=geom.get("cell_bohr"),
        qm_extent_bohr=geom.get("extent_bohr"),
        min_vacuum_bohr=geom.get("min_vacuum_bohr"),
        pseudopotentials=qm_pseudopotentials(sections),
    )
