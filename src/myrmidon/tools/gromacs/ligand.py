"""Ligand parameters from acpype: GAFF or GAFF2 atom types and AM1-BCC or Gasteiger
charges, as a GROMACS molecule.

The force-field family, the protonation and the net charge are the researcher's. A
deposited ligand carries no hydrogens, and AM1-BCC charges computed for a molecule without
them describe a radical: hydrogens are added by Open Babel only when asked, and a ligand
said to carry its hydrogens that has none is refused before sqm runs. The charges are read
back from the written topology and must sum to the net charge stated."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, precheck, tool
from myrmidon.core.validation import Precondition
from myrmidon.engines.gromacs.itp import itp_rows, parse_itp_atoms
from myrmidon.engines.inputs import require_input, resolve_in_ws
from myrmidon.engines.ligand import hydrogen_count, nuclear_charge, residue_name, run_acpype
from myrmidon.errors import EngineNotFoundError
from myrmidon.text import tail

CHARGE_METHODS = {"bcc": "AM1-BCC (sqm)", "gas": "Gasteiger"}


#: what acpype writes when a run fails without a non-zero exit
_ACPYPE_ERROR = re.compile(r"^\s*(?:ERROR|Error)\b.*$", re.MULTILINE)


class LigandParams(BaseModel):
    structure: str = Field(description="the ligand's coordinates (.pdb or .mol2), one "
                           "molecule; structure_select's ligand output for a deposited one")
    net_charge: int = Field(description="the ligand's net charge in its protonation state")
    hydrogens: Literal["present", "add_by_valence", "add_at_ph"] = Field(
        description="'present': the structure carries every hydrogen; 'add_by_valence': "
        "Open Babel completes every valence (each group neutral); 'add_at_ph': Open Babel "
        "protonates for the pH given in ph")
    ph: float | None = Field(default=None, description="the pH, with add_at_ph")
    atom_types: Literal["gaff2", "gaff"] = Field(description="the atom-type family")
    charge_method: Literal["bcc", "gas"] = Field(
        description="'bcc': AM1-BCC, from an AM1 calculation with sqm; 'gas': Gasteiger")
    name: str | None = Field(
        default=None, max_length=5,
        description="the molecule and residue name in the topology (default: the "
        "structure's residue name)")

    @model_validator(mode="after")
    def _ph_with_ph(self) -> LigandParams:
        if (self.hydrogens == "add_at_ph") != (self.ph is not None):
            raise ValueError("give ph exactly when hydrogens is 'add_at_ph'")
        return self




def ligand_gate(ctx: ToolContext, params: LigandParams):
    checked = ["input_files"]
    structure, missing = require_input(ctx, "structure", params.structure)
    if missing:
        return [missing], checked
    checked.append("hydrogens")
    if params.hydrogens == "present" and hydrogen_count(structure) == 0:
        return [Precondition(
            id="LIGAND_HYDROGENS_ABSENT",
            message=f"{params.structure} carries no hydrogens, and charges computed for it "
            "would be a radical's",
            suggestion="state the protonation: hydrogens 'add_by_valence' or 'add_at_ph', "
            "or give a structure that carries them",
            needs=["hydrogens"],
        )], checked
    return [], checked


@tool(
    "ligand_parametrize",
    description="Parametrise a small molecule with acpype (antechamber, sqm, parmchk2): GAFF2/GAFF"
                " types, AM1-BCC/Gasteiger charges, a GROMACS .itp, .gro and topology. Atom types,"
                " charge method, protonation and net charge are the researcher's; the charges must"
                " sum to the net charge.",
    tags={"prep", "ligand"},
    requires={"acpype"},
    contract=Contract(consumes={"structure": "structure"},
                      produces=(Produces("structure"), Produces("itp"),
                                Produces("topology"))),
)
def ligand_parametrize(ctx: ToolContext, params: LigandParams) -> Observation:
    precheck(ctx, ligand_gate(ctx, params)[0])
    info = ctx.engines.get("acpype")
    if info is None or not info.available:
        raise EngineNotFoundError("acpype", info.hint if info is not None else "")
    source = resolve_in_ws(ctx, params.structure)
    name = params.name or residue_name(source) or "LIG"
    stage = ctx.workspace.stage("ligand")
    run = ctx.workspace.run_dir(stage, name)
    given = run / f"{name}_input{source.suffix.lower()}"
    shutil.copy(source, given)

    envelope = ctx.config.session
    bindir = str(Path(info.path).parent)
    env = {**os.environ, "PATH": bindir + os.pathsep + os.environ.get("PATH", ""),
           "OMP_NUM_THREADS": str(envelope.max_cores)}
    timeout = envelope.max_minutes * 60

    structure, added = given, None
    if params.hydrogens != "present":
        obabel = shutil.which("obabel", path=env["PATH"]) or "obabel"
        structure = run / f"{name}_h{source.suffix.lower()}"
        argv = [obabel, str(given), "-O", str(structure)]
        argv += ["-p", f"{params.ph:g}"] if params.ph is not None else ["-h"]
        proc = run_acpype(argv, run, env, timeout)
        if proc.returncode != 0 or not structure.is_file():
            return Observation(ok=False, molecule=name,
                                     summary=f"Open Babel could not add hydrogens to "
                                     f"{params.structure} (exit {proc.returncode})",
                                     command=argv, log_tail=tail(proc.stderr + proc.stdout))
        before, after = hydrogen_count(given), hydrogen_count(structure)
        added = after - before if before is not None and after is not None else None

    z = nuclear_charge(structure)
    if params.charge_method == "bcc" and z is not None and (z - params.net_charge) % 2:
        h = hydrogen_count(structure)
        return Observation(
            ok=False, molecule=name, hydrogens=h, hydrogens_added=added,
            electrons=z - params.net_charge,
            summary=f"{name} with {h} hydrogens and net charge {params.net_charge} has "
            f"{z - params.net_charge} electrons: an open shell, which AM1-BCC cannot charge",
            findings=[Precondition(
                id="LIGAND_ODD_ELECTRONS",
                message=f"{h} hydrogens and net charge {params.net_charge} leave "
                f"{z - params.net_charge} electrons, an odd number",
                means="the hydrogens and the charge describe a radical; with these "
                f"hydrogens a closed shell has net charge {params.net_charge - 1} or "
                f"{params.net_charge + 1}",
                suggestion="state the net charge of the protonation the hydrogens give, or "
                "another protonation", needs=["net_charge", "hydrogens"])])

    argv = [info.path, "-i", structure.name, "-b", name, "-c", params.charge_method,
            "-a", params.atom_types, "-n", str(params.net_charge), "-o", "gmx"]
    proc = run_acpype(argv, run, env, timeout)
    out_dir = run / f"{name}.acpype"
    itp, gro, top = (out_dir / f"{name}_GMX{ext}" for ext in (".itp", ".gro", ".top"))
    log = proc.stdout + proc.stderr
    errors = _ACPYPE_ERROR.findall(log)
    if proc.returncode != 0 or errors or not (itp.is_file() and gro.is_file()):
        return Observation(
            ok=False, molecule=name,
            summary=f"acpype failed for {name} (exit {proc.returncode})"
            + (f": {errors[0].strip()}" if errors else ""),
            command=argv, log_tail=tail(log, 30))

    text = itp.read_text()
    atoms = [a for mol in parse_itp_atoms(text).values() for a in mol]
    total = round(sum(a["charge"] for a in atoms), 4)
    atomtypes = [f[0] for _, section, f in itp_rows(text) if section == "atomtypes"]
    n_gro = int(gro.read_text().splitlines()[1].strip())
    sqm = out_dir / "sqm.out"
    completed = ("Calculation Completed" in sqm.read_text(errors="replace")
                 if params.charge_method == "bcc" and sqm.is_file() else None)
    method = CHARGE_METHODS[params.charge_method]
    base = dict(molecule=name, n_atoms=len(atoms), hydrogens_added=added, net_charge=total,
                charge_method=method, atom_types=params.atom_types, atomtypes=atomtypes,
                sqm_completed=completed, command=argv)
    findings = []
    if abs(total - params.net_charge) > 1e-3:
        findings.append(Precondition(
            id="LIGAND_CHARGE_SUM",
            message=f"the written charges sum to {total:g} e, not the net charge "
            f"{params.net_charge} stated", needs=["net_charge"]))
    if n_gro != len(atoms):
        findings.append(Precondition(
            id="LIGAND_COORDINATES_MISMATCH",
            message=f"{gro.name} holds {n_gro} atoms and {itp.name} defines {len(atoms)}"))
    if params.charge_method == "bcc" and not completed:
        findings.append(Precondition(
            id="LIGAND_SQM_INCOMPLETE",
            message="sqm did not report a completed AM1 calculation; the AM1-BCC charges "
            "are not from a converged one"))
    if findings:
        return Observation(ok=False, summary=f"{name}: {findings[0].message}",
                                 findings=findings, log_tail=tail(log, 30), **base)
    # acpype nests its products in <name>.acpype/; the ones the study builds on are also
    # written at the top of the ligand's own directory, where the next calls look for them
    for f in (gro, itp, top, *out_dir.glob("posre_*.itp")):  # and the restraints it includes
        shutil.copy2(f, run / f.name)
    gro, itp, top = (run / f.name for f in (gro, itp, top))
    root = ctx.workspace.root
    hydro = {"present": "hydrogens as given",
             "add_by_valence": f"{added} hydrogens added by valence with Open Babel",
             "add_at_ph": f"{added} hydrogens added for pH {params.ph:g} with Open Babel"
             if params.ph is not None else ""}[params.hydrogens]
    return Observation(
        ok=True,
        summary=f"{name} parametrised: {params.atom_types.upper()} atom types, {method} "
        f"charges summing to {total:g} e, {len(atoms)} atoms ({hydro}); "
        f"{len(atomtypes)} atom types defined in the itp",
        artifacts=[ArtifactRef.from_path(gro, "structure", root=root),
                   ArtifactRef.from_path(itp, "itp", root=root),
                   ArtifactRef.from_path(top, "topology", root=root)],
        inputs=[ArtifactRef.from_path(source, "structure", root=root)],
        **base,
    )
