"""A deliverable's physical observable compiled into the tool call that measures it.

A deliverable says what is measured in physical terms — the ligand's RMSD after the
protein backbone is superposed, the minimum distance between the ligand and the protein,
heavy atoms only, over the production stage — and the researcher approves it in those
terms. :func:`compile_deliverable` turns it into the measuring tool and the parameters that
define the number: the selections, the superposition, the reference frame. The atom filter
applies to every side alike, so two sides never differ in whether they count hydrogens.

The selection language is MDAnalysis's (the language ``observable_measure`` reads); a subsystem
is selected by what the mandate names it with: ``protein`` and ``nucleic`` by kind, any
other by its residue names, else by its molecule types (a ligand's molecule is named after
its residue). The table of properties is the one place a new observable is added.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from myrmidon.core.mandate import Deliverable, Mandate, Subsystem

#: what a selection by backbone or by alpha carbon means for a subsystem of each kind
POLYMERS = {"protein", "nucleic"}

#: the one tool every property is measured with
MEASURE_TOOL = "observable_measure"
#: property -> (the extractor that measures it, its name for it, the sides it relates, whether
#: it superposes). A new property is one row here and its extractor (tools/measure.py)
PROPERTIES: dict[str, tuple[str, str, int, bool]] = {
    "rmsd": ("trajectory", "rmsd", 1, True),
    "rmsf": ("trajectory", "rmsf", 1, True),
    "radius_of_gyration": ("trajectory", "rg", 1, False),
    "min_distance": ("trajectory", "mindist", 2, False),
    "contacts": ("trajectory", "contacts", 2, False),
    "pair_distance": ("trajectory", "pair_distance", 1, False),
    "rdf": ("trajectory", "rdf", 2, False),
    "thermodynamics": ("energy", "thermodynamics", 0, False),
    "energy_drift": ("energy", "energy_drift", 0, False),
    "energy_terms": ("energy", "energy_terms", 0, False),
    "settling": ("settling", "settling", 0, False),
    "dipole": ("run_log", "dipole", 0, False),
    "step_timing": ("run_log", "step_timing", 0, False),
    "qm_forces": ("forces", "qm_forces", 0, False),
    "force_rmse": ("forces", "force_rmse", 0, False),
    "boundary_bonds": ("region", "boundary_bonds", 0, False),
}
#: a coupled stage's thermodynamics and drift are what its QM log states
COUPLED = {"thermodynamics": "run_log", "energy_drift": "run_log"}
#: what an energy property reads of a classical run's energy file
ENERGY_TERMS = {"thermodynamics": ["Temperature", "Pressure", "Density", "Potential"],
                "energy_drift": ["Conserved-En."]}
#: the atom filter, the same on every side
ATOMS = {"heavy": "not name H*", "backbone": "backbone", "ca": "name CA", "all": ""}


class CompiledDeliverable(BaseModel):
    name: str
    tool: str
    extractor: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    over: str = ""
    statistics: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # measurable, but worth a second look

    def line(self) -> str:
        """The measurement as the researcher reads it beside the deliverable."""
        said = ", ".join(f"{k} {v!r}" for k, v in self.params.items())
        return f"{self.tool}: {said}" + (
            f"; reports {', '.join(self.statistics)}" if self.statistics else "")


def subsystem_selection(sub: Subsystem, mandate: Mandate | None = None) -> str:
    """The atoms of a subsystem, as MDAnalysis selects them: a protein or nucleic acid by
    its keyword when it is the system's only one, else by its molecule types (read from
    the run's .tpr); any other by its residue names, else its molecule types."""
    alone = mandate is None or sum(s.kind == sub.kind for s in
                                   mandate.system.subsystems.values()) == 1
    if sub.kind in POLYMERS and not sub.resnames:
        base = sub.kind if alone or not sub.molecules else "moltype " + " ".join(sub.molecules)
    else:
        names = sub.resnames or sub.molecules
        base = "resname " + " ".join(names) if names else ""
    parts = [base] if base else []
    if sub.chains:
        parts.append("chainID " + " ".join(sub.chains))
    if sub.resid:
        # a range is written "first-last"; MDAnalysis spells it first:last
        parts.append("resid " + " ".join(str(r).replace("-", ":", 1) if isinstance(r, str)
                                          else str(r) for r in sub.resid))
    return " and ".join(parts)


def _filtered(selection: str, atoms: str, kind: str) -> str:
    """``selection`` held to an atom filter; a nucleic acid's backbone is its own."""
    word = ATOMS[atoms]
    if atoms == "backbone" and kind == "nucleic":
        word = "nucleicbackbone"
    return f"{selection} and {word}" if word and selection else selection


def measured_property(params: dict) -> str:
    """The property a measuring call is of, by what it names: its property, a metric, or the
    file only one extractor reads."""
    return property_named(params.get("property") or params.get("metric") or (
        "energy_terms" if params.get("edr") or params.get("terms")
        else "settling" if params.get("data")
        else "force_rmse" if params.get("forces") else ""))


def property_named(name: str) -> str:
    """The property a name means: itself, or the property its extractor calls by that name
    ("mindist" is min_distance); the name as given when neither."""
    if name in PROPERTIES:
        return name
    return next((p for p, (_, metric, _, _) in PROPERTIES.items() if metric == name), name)


def compile_deliverable(mandate: Mandate, deliverable: Deliverable) -> CompiledDeliverable:
    """The extractor and the parameters that measure a deliverable as the mandate states it;
    ``problems`` says what keeps it from being measured so."""
    d = deliverable
    out = CompiledDeliverable(name=d.name, tool=d.tool, params=dict(d.params), over=d.over,
                              statistics=list(d.statistics))
    if not d.property:
        return out  # a deliverable stated as a tool's own call
    if d.property not in PROPERTIES:
        out.problems.append(f"{d.name}: no property {d.property!r}; known: "
                            + ", ".join(PROPERTIES))
        return out
    extractor, metric, sides, superposes = PROPERTIES[d.property]
    stage = mandate.method.stage(d.over)
    if stage is not None and stage.regime != "mm" and d.property in COUPLED:
        extractor = COUPLED[d.property]
    out.tool, out.extractor = MEASURE_TOOL, extractor
    subs = mandate.system.subsystems

    def side(name: str, role: str, atoms: str) -> tuple[str, str] | None:
        sub = subs.get(name)
        if sub is None:
            out.problems.append(f"{d.name}: its {role} {name!r} is no subsystem of the "
                                f"system ({', '.join(subs) or 'none named'})")
            return None
        selection = subsystem_selection(sub, mandate)
        if not selection:
            out.problems.append(f"{d.name}: subsystem {name} names no residues or molecules")
            return None
        if atoms in ("backbone", "ca") and sub.kind not in POLYMERS:
            out.problems.append(f"{d.name}: {name} is a {sub.kind}, which has no {atoms} "
                                "atoms; take its heavy atoms or all of them")
            return None
        return _filtered(selection, atoms, sub.kind), sub.kind

    if extractor == "trajectory":
        out.params = {"property": d.property, "metric": metric}
        if d.property == "pair_distance":
            out.problems.append(f"{d.name}: a pair distance names its atom pairs, which a "
                                "deliverable cannot: measure it with its pairs")
            return out
        if not d.target:
            out.problems.append(f"{d.name}: {d.property} is measured on a target subsystem")
            return out
        atoms = d.atoms or "heavy"
        target = side(d.target, "target", atoms)
        if target is not None:
            out.params["selection"] = target[0]
        if sides == 2:
            if not d.reference:
                out.problems.append(f"{d.name}: {d.property} relates the target to a "
                                    "reference subsystem")
            elif (ref := side(d.reference, "reference", atoms)) is not None:
                out.params["selection2"] = ref[0]
        start = d.reference_frame or "stage_start"
        if superposes:
            if d.align_to is not None and (
                    fit := side(d.align_to.subsystem, "superposition", d.align_to.atoms)):
                out.params["fit_selection"] = fit[0]
                if d.align_to.subsystem == d.target and fit[1] not in POLYMERS:
                    out.notes.append(
                        f"{d.name}: superposed on {d.target} itself, the {d.property} "
                        f"measures the change of {d.target}'s shape, not where it sits; its "
                        "pose in a binding site is measured after superposing the "
                        "macromolecule (align_to a protein or nucleic subsystem)")
            # a ligand is measured against the pocket it sits in, not a periodic image
            out.params["unwrap"] = True
            if start == "stage_start":
                out.params["reference_frame"] = 0
            else:
                out.problems.append(f"{d.name}: measured from {start!r}; the harness "
                                    "measures a superposition from the stage's first frame")
        elif start != "stage_start":
            out.problems.append(f"{d.name}: {d.property} has no reference frame")
    elif extractor == "energy":
        out.params = {"property": d.property, "terms": ENERGY_TERMS[d.property]}
    else:
        out.params = {"property": d.property}  # of the stage's run as a whole
    return out
