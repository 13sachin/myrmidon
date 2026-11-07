"""A system partitioned into its QM region and the rest, in one call: prepqm writes the
system's CPMD input and the QM atoms' index group, the selection commands a parameter fit
reads back are written beside them, and a sidecar states what the selection decides: the
charge the selected atoms' topology charges sum to, the spin multiplicity and electron count
that charge leaves, and the elements the engine guessed for atom types the topology gives no
atomic number for. Which atoms are quantum is the researcher's; everything else is read and
said."""

from __future__ import annotations

import json
import re
import subprocess

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import declare_identity
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, precheck
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.mimic.cpmd_input import get_keyword, parse_cpmd, qm_geometry
from myrmidon.engines.mimic.qm_region import (
    SELECTION_FILE,
    guessed_elements,
    mimicpy_bin,
    prepqm_argv,
    region_chemistry,
    rejected_commands,
    rounded_charge,
    unknown_commands,
)
from myrmidon.engines.mimic.system import pp_findings, qm_pseudopotentials
from myrmidon.text import tail

#: the tool whose call is the partition, as the ledger records it
PARTITION_TOOL = "qm_region"
#: which atoms are quantum is a system's identity; a partition writes the system's CPMD input
#: and index anew for the stage it was made for
declare_identity("qm_selection", replaced_by=PARTITION_TOOL)


def region_settings(mandate, stage) -> tuple[float | None, str]:
    """(the padding in nm, prepqm's pseudopotential table) a stage's region is partitioned
    with: the stage's padding, else its QM method's; the method's pseudopotentials."""
    from myrmidon.knowledge.stage import MethodStore, methods_for

    qm = {name: value for name, (value, _) in methods_for(
        mandate, MethodStore.default(), None).values.get("qm", {}).items()}
    region = stage.qm_region
    return (region.padding_nm if region and region.padding_nm else qm.get("qm_cell_padding_nm"),
            pp_table(qm))


def pp_table(qm: dict) -> str:
    """The pseudopotential table prepqm reads (``-pp``: element, file, boundary file, the
    nonlocal type, LMAX, LOC), from a QM method's pseudopotential family and each element's
    channels; empty when the method names neither."""
    family, lmax = qm.get("pseudopotentials"), qm.get("pseudopotential_lmax") or {}
    if not family or not lmax:
        return ""
    return "".join(f"{el} {el}_{family} - KLEINMAN-BYLANDER {channel} -\n"
                   for el, channel in lmax.items())


def subsystem_selections(parts: dict, names: list[str]) -> tuple[list[str], list[str]]:
    """(the selection commands that add each named subsystem's residues, why each name that
    does not resolve does not): a name is a part's own, or the kind of every part of that
    kind (``ligand``). A part named by its residue numbers or chains is some copies of its
    residues, which their names alone would overshoot; one with no residue names has none to
    select by. Each part is the mandate's subsystem or the topology's proposal, as a dict."""
    commands: list[str] = []
    unresolved: list[str] = []
    for name in names:
        matched = [parts[name]] if name in parts else [
            p for p in parts.values() if p.get("kind") == name]
        resnames = [r for p in matched for r in p.get("resnames") or []]
        if not matched:
            unresolved.append(f"{name}: no part of the system")
        elif any(p.get("resid") or p.get("chains") for p in matched):
            unresolved.append(f"{name}: named by residue number or chain, which its residue "
                              "names alone would overshoot")
        elif not resnames:
            unresolved.append(f"{name}: names no residue")
        else:
            commands += [f"add resname is {r}" for r in dict.fromkeys(resnames)]
    return commands, unresolved


def open_chemistry(ws, stages: list, decisions: list, selected_for: str | None = None
                   ) -> list[str]:
    """What the latest QM selection (for a stage, if named) left open of the region's chemistry
    and no stage or decision settles: a charge whose atoms sum to no integer is the researcher's
    unless they locked it, a multiplicity the electrons do not decide unless a stage states it."""
    from myrmidon.core.lineage import latest_outputs

    regions = [s.qm_region for s in stages if s.qm_region is not None]
    side = latest_outputs(ws, PARTITION_TOOL, selected_for).get("provenance")
    if not regions or side is None:
        return []
    try:
        left = json.loads(ws.resolve(side).read_text())
    except (OSError, ValueError):
        return []
    questions = []
    # a lock settles the charge of the stage it names, and no other stage's
    named = ({selected_for} if selected_for else {s.name for s in stages}) | {"*"}
    decided = any(path in (f"{n}.qm_region.charge", f"{n}.qm_region")
                  for d in decisions for path in d.locks for n in named)
    if left.get("charge") is None and not decided:
        questions.append("What is the QM region's net charge? (the selected atoms' topology "
                         "charges sum to no integer: say what the selection cuts)")
    if left.get("multiplicity") is None and any(r.multiplicity is None for r in regions):
        questions.append(f"What is the QM region's spin multiplicity? ({left.get('spin_why')})")
    return questions


def _gate(params) -> list[Finding]:
    """Refuse a selection command prepqm does not have, before it runs: it would skip the
    command and write the region the others make, exit 0."""
    unknown = unknown_commands(params.selections)
    if not unknown:
        return []
    return [Precondition(
        id="QM_SELECTION_COMMAND_UNKNOWN",
        message=f"prepqm has no command for {unknown}: it would skip "
        f"{'it' if len(unknown) == 1 else 'them'} and write the region the other commands "
        "make. Its commands are 'add SEL', 'add-bound SEL', 'delete SEL' and 'clear'.",
        suggestion="'delete SEL' removes atoms; each command is one verb and one selection",
    )]


def _refused(params, argv, proc, inp, ndx) -> Observation | None:
    """Why the files prepqm left are not the region asked for; None when they are."""
    rejected = rejected_commands(proc.stdout, params.selections)
    if rejected:  # the files exist, and hold the region of the commands it did not skip
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
    # mimicpy can exit 0 after an internal failure, an empty region among them: judge by the
    # artifacts, not the exit code
    if proc.returncode != 0 or not inp.exists() or not ndx.exists():
        return Observation(
            ok=False,
            summary=f"mimicpy prepqm failed (exit {proc.returncode}; cpmd.inp "
            f"{'written' if inp.exists() else 'missing'}, index.ndx "
            f"{'written' if ndx.exists() else 'missing'})",
            command=argv,
            log_tail=tail(proc.stderr + "\n" + proc.stdout, 40),
            selections=params.selections,
        )
    return None


def partition_qm(ctx: ToolContext, params) -> Observation:
    """prepqm on ``params``' selection commands, into the stage the coupled runs belong to (a
    directory of its own when the stage already holds a selection's input), and what the
    region it wrote holds."""
    precheck(ctx, _gate(params))
    stage = ctx.workspace.stage(params.stage)
    if (stage / "cpmd.inp").exists():  # a mandate may hold what an earlier selection wrote
        stage = ctx.workspace.run_dir(stage, "selection")
    gmx = ctx.engines.get("gromacs")
    argv = prepqm_argv(
        mimicpy_bin(ctx), resolve_in_ws(ctx, params.topology), resolve_in_ws(ctx, params.coords),
        stage, params.qm_group, params.padding_nm, gmx.data_dir if gmx else None,
        resolve_in_ws(ctx, params.template) if params.template else None,
        resolve_in_ws(ctx, params.pp_info) if params.pp_info else None, params.atomtypes_map)
    stdin = "\n".join(params.selections) + f"\n{ctx.config.mimicpy_quit_cmd}\n"
    proc = subprocess.run(  # noqa: S603
        argv, cwd=stage, input=stdin, capture_output=True, text=True, timeout=600
    )
    ndx, inp = stage / "index.ndx", stage / "cpmd.inp"
    refused = _refused(params, argv, proc, inp, ndx)
    if refused is not None:
        return refused
    sections = parse_cpmd(inp.read_text())
    root = ctx.workspace.root
    selection = stage / SELECTION_FILE
    selection.write_text("\n".join(params.selections) + "\n")
    declared = get_keyword(sections.get("SYSTEM", []), "CHARGE")
    declared_charge = (
        int(float(declared)) if declared and re.match(r"^-?\d+(\.0*)?$", declared) else None
    )
    rounded_from = rounded_charge(proc.stderr + proc.stdout)
    guessed = guessed_elements(proc.stderr + proc.stdout)
    chemistry, findings = region_chemistry(
        sections, None if rounded_from is not None else declared_charge,
        params.selections, guessed)
    sidecar = inp.with_name(inp.name + ".myrmidon.json")
    sidecar.write_text(json.dumps({"written_by": PARTITION_TOOL, **chemistry}, indent=1))
    artifacts = [ArtifactRef.from_path(inp, "cpmd_input", root=root),
                 ArtifactRef.from_path(ndx, "index", root=root),
                 ArtifactRef.from_path(selection, "qm_selection", root=root),
                 ArtifactRef.from_path(sidecar, "provenance", root=root)]
    uncovered, _ = pp_findings(sections, stage)
    geom = qm_geometry(sections) or {}
    qm_atoms = geom.get("n_atoms")
    if uncovered:
        return Observation(
            ok=False,
            summary="the QM region holds an element CPMD has no pseudopotential for here: "
            + "; ".join(w.message for w in uncovered),
            command=argv, artifacts=artifacts, findings=uncovered,
            selections=params.selections, qm_atoms=qm_atoms,
            pseudopotentials=qm_pseudopotentials(sections),
        )
    if rounded_from is not None:
        findings.append(Finding(
            id="QM_CHARGE_ROUNDED",
            message=f"the selected atoms' MM charges sum to {rounded_from:g}, which prepqm "
            f"rounded to {declared_charge}: a region that cuts a covalent bond or part of a "
            "residue's charge distribution sums to a non-integer, and its QM charge is the "
            "researcher's, not the rounding's",
            suggestion="say what the selection cuts, and put the charge to the researcher",
        ))
    multiplicity = chemistry["multiplicity"]
    return Observation(
        ok=True,
        summary=f"QM region selected: {qm_atoms if qm_atoms is not None else '?'} atoms; "
        f"declared charge {declared_charge if declared_charge is not None else 'absent'}"
        + (f" (prepqm rounded the selection's MM charges, {rounded_from:g})"
           if rounded_from is not None else "")
        + (f", multiplicity {multiplicity}" if multiplicity is not None
           else f", multiplicity open ({chemistry['spin_why']})")
        + (f"; QM cell {'x'.join(str(v) for v in geom['cell_bohr'])} bohr, "
           f"min vacuum {geom['min_vacuum_bohr']} bohr" if geom else ""),
        command=argv,
        artifacts=artifacts,
        findings=findings,
        qm_atoms=qm_atoms,
        declared_charge=declared_charge,
        rounded_from=rounded_from,
        derived_multiplicity=multiplicity,
        electrons=chemistry["electrons"],
        guessed_elements=guessed,
        selections=params.selections,
        qm_cell_bohr=geom.get("cell_bohr"),
        qm_extent_bohr=geom.get("extent_bohr"),
        min_vacuum_bohr=geom.get("min_vacuum_bohr"),
        pseudopotentials=qm_pseudopotentials(sections),
    )
