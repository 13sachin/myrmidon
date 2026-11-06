"""The mandate's stages in physical terms: compiled into what each engine reads, held at
submission to what the researcher locked, and measured over the stage a deliverable names."""

import json
import shutil

import pytest
from tests.conftest import ACETONE
from tests.unit.findings_catalog import blocking

from myrmidon.core.tool import REGISTRY

#: a protein and a ligand, each with the position restraints its own switch turns on
COMPLEX_TOP = """#include "amber03.ff/forcefield.itp"

[ moleculetype ]
Protein_chain_A  3
[ atoms ]
1  CT  1  ALA  CA  1  0.10  12.01
2  CT  1  ALA  CB  2 -0.10  12.01
#ifdef POSRES
[ position_restraints ]
1  1  1000  1000  1000
#endif

[ moleculetype ]
JZ4  3
[ atoms ]
1  c3  1  JZ4  C1  1  0.05  12.01
2  c3  1  JZ4  C2  2 -0.05  12.01
#ifdef POSRES_LIG
[ position_restraints ]
1  1  1000  1000  1000
#endif

[ system ]
complex in vacuum

[ molecules ]
Protein_chain_A  1
JZ4              1
"""

SUBSYSTEMS = {"protein": {"kind": "protein", "molecules": ["Protein_chain_A"]},
              "ligand": {"kind": "ligand", "molecules": ["JZ4"], "charge": 0}}
STAGES = [
    {"name": "em", "kind": "minimize"},
    {"name": "nvt", "kind": "dynamics", "ensemble": "NVT", "temperature_k": 300,
     "duration_ps": 100, "restraints": ["protein", "ligand"]},
    {"name": "npt", "kind": "dynamics", "ensemble": "NPT", "temperature_k": 300,
     "pressure_bar": 1, "duration_ps": 100, "restraints": ["protein", "ligand"]},
    {"name": "production", "kind": "dynamics", "ensemble": "NPT", "temperature_k": 300,
     "pressure_bar": 1, "duration_ps": 1000, "sampling_ps": 2},
]
LOCKS = [{"said": "over 1 ns", "locks": {"production.duration_ps": 1000}},
         {"said": "300 K", "locks": {"*.temperature_k": 300}},
         {"said": "restrained equilibration",
          "locks": {"nvt.restraints": ["protein", "ligand"],
                    "npt.restraints": ["protein", "ligand"]}}]


@pytest.fixture
def complex_system(ws_ctx):
    stage = ws_ctx.workspace.stage("input")
    (stage / "topol.top").write_text(COMPLEX_TOP)
    shutil.copytree(ACETONE / "amber03.ff", stage / "amber03.ff")
    (stage / "conf.gro").write_text(
        "complex\n    4\n"
        "    1ALA     CA    1   1.000   1.000   1.000\n"
        "    1ALA     CB    2   1.100   1.000   1.000\n"
        "    2JZ4     C1    3   1.500   1.000   1.000\n"
        "    2JZ4     C2    4   1.600   1.000   1.000\n"
        "   3.00000   3.00000   3.00000\n")
    # the same atoms as an earlier run leaves them: moving
    (stage / "moving.gro").write_text("".join(
        line + ("  0.1000  0.1000  0.1000" if line.startswith("    ") and "." in line else "")
        + "\n" for line in (stage / "conf.gro").read_text().splitlines()))
    rel = ws_ctx.workspace.relpath
    return {"topology": rel(stage / "topol.top"), "structure": rel(stage / "conf.gro")}


def _write(ws_ctx, system, **params):
    call = {"question": "Equilibrate the complex and run 1 ns of production at 300 K",
            "ceiling_core_h": 500.0,
            "system_files": [{"path": system["structure"], "kind": "structure"},
                             {"path": system["topology"], "kind": "topology"}],
            "subsystems": SUBSYSTEMS, "stages": STAGES, "decisions_yours": LOCKS, **params}
    return REGISTRY.call("mandate_write", ws_ctx, call)


def _approved(ws_ctx, system, **params):
    from myrmidon.core import mandate as mandates

    written = _write(ws_ctx, system, **params)
    assert written.ok and written.status == "ready", written.summary
    ws = ws_ctx.workspace
    mandates.record_approval(ws, mandates.read_mandate(ws), approver="researcher",
                             via="terminal")
    return written


def _run_input(ws_ctx, system, mdp_call: dict, name: str) -> tuple:
    """A rendered parameter file and the run input grompp builds on it."""
    structure = mdp_call.get("structure", system["structure"])
    rendered = REGISTRY.call("inputset_apply", ws_ctx, {
        "output": f"02_run/{name}.mdp", "structure": structure, **mdp_call})
    assert rendered.ok, rendered.summary
    tpr = REGISTRY.call("gmx_grompp", ws_ctx, {
        "mdp": f"02_run/{name}.mdp", "structure": structure,
        "topology": system["topology"], "output": f"{name}.tpr", "stage": "run"})
    assert tpr.ok, tpr.summary
    return rendered, tpr.artifacts[0].path


# (the stage rendered, what else the call says, what the file holds, whether the job is
# refused at submission). A stage's file is compiled from the mandate, whatever input set
# the call also names; one that breaks what the researcher locked on that stage is refused.
MOVING = "01_input/moving.gro"  # complex_system's structure with velocities

RENDERS = [
    # production continues the moving box under Parrinello-Rahman; from rest it relaxes first
    ("production", {"structure": MOVING},
     {"nsteps": 500000, "dt": 0.002, "pcoupl": "Parrinello-Rahman", "continuation": "yes"},
     False),
    ("production", {}, {"pcoupl": "C-rescale", "gen-vel": "yes"}, False),
    ("production", {"overrides": {"nsteps": 125000}}, {"nsteps": 125000}, True),
    ("nvt", {}, {"define": "-DPOSRES -DPOSRES_LIG", "nsteps": 50000}, False),
    ("nvt", {"inputset": "gromacs.nvt"}, {"nsteps": 50000}, False),
    ("nvt", {"overrides": {"define": "-DPOSRES"}}, {"define": "-DPOSRES"}, True),
    ("npt", {"overrides": {"ref-t": 310}}, {"ref-t": "310 310"}, True),  # per group
    ("npt", {"overrides": {"ref_t": 310}}, {"ref-t": "310 310"}, True),  # any spelling
]


@pytest.mark.parametrize("stage,call,holds,refused", RENDERS)
def test_a_stage_compiles_from_the_mandate_and_is_held_to_its_locks(
        ws_ctx, complex_system, stage, call, holds, refused):
    _approved(ws_ctx, complex_system)
    rendered, tpr = _run_input(ws_ctx, complex_system, {"stage": stage, **call}, stage)
    for key, value in holds.items():
        assert rendered.values[key] == value, rendered.values
    assert "ref_t" not in rendered.values
    assert ("MANDATE_LOCK_OVERRIDDEN" in {f.id for f in rendered.findings}) is refused
    event = [e for e in ws_ctx.workspace.ledger.events() if e.tool == "inputset_apply"][-1]
    assert f"stage:{stage}" in event.knowledge_refs
    if "structure" in call:
        return  # a run built from a structure outside the approved system is refused anyway

    obs = REGISTRY.call("gmx_mdrun", ws_ctx, {"tpr": tpr, "preview": True})
    assert [w.id for w in blocking(obs)] == (["MANDATE_DECISION"] if refused else []), obs.summary


QMMM_WRITE = {"hamiltonian": {"mm": "amber", "qm": {"method": "blyp_mt"},
                              "coupling": {"scheme": "mimic"}},
              "stages": [{"name": "hot", "kind": "dynamics", "regime": "qmmm",
                          "ensemble": "NVT", "temperature_k": 300, "steps": 100,
                          "qm_region": {"subsystems": ["ligand"], "charge": 0,
                                        "padding_nm": 0.4}}]}

# (what the call adds to an honest mandate, the code it is refused with, and how many of
# its checks say why — one unless stated). A mandate whose stages, subsystems and locks do
# not bind together is never put to the researcher; nor is a compiled engine file, which
# is the harness's, never the model's.
UNWRITTEN = [
    ({"stages": [{**STAGES[0], "compiled": {"nsteps": 100}}, *STAGES[1:]]}, "INVALID_PARAMS"),
    ({"subsystems": {**SUBSYSTEMS, "water": {"kind": "solvent", "molecules": ["SOL"]}},
      "stages": [*STAGES[:3], {**STAGES[3], "restraints": ["water"]}]},
     "MANDATE_STAGE", 2),  # SOL has no restraints, nor a moleculetype of its own here
    ({"decisions_yours": [{"said": "half a nanosecond",
                           "locks": {"production.duration_ps": 500}}]}, "MANDATE_STAGE"),
    ({"decisions_yours": [{"said": "x", "locks": {"heat.duration_ps": 500}}]}, "MANDATE_STAGE"),
    ({"decisions_yours": [{"said": "x", "locks": {"nvt.production_ns": 1}}]}, "MANDATE_STAGE"),
    ({"stages": [*STAGES[:3], {**STAGES[3], "steps": 100}]}, "INVALID_PARAMS"),
    # a lock that contradicts its stage is refused, whether or not anything compiled reads it
    ({"decisions_yours": [{"said": "the em as dynamics", "locks": {"em.kind": "dynamics"}}]},
     "MANDATE_STAGE", 2),  # it disagrees with the stage, and binds nothing compiled
    # numbers that select none of the residues a subsystem's names select name none to measure
    ({"subsystems": {**SUBSYSTEMS, "site": {"kind": "protein", "resnames": ["ALA", "JZ4"],
                                            "resid": ["500-510"]}},
      "deliverables": [{"name": "RMSD", "property": "rmsd", "target": "site",
                        "over": "production"}]},
     "MANDATE_STAGE"),
    ({"subsystems": {**SUBSYSTEMS, "protein": {"kind": "protein", "molecules": ["Protein"]}},
      "stages": [{**s, "restraints": []} for s in STAGES], "decisions_yours": [],
      "deliverables": [{"name": "RMSD", "property": "rmsd", "target": "protein",
                        "over": "production"}]},
     "MANDATE_STAGE"),
    ({"deliverables": [{"name": "pocket RMSD", "property": "rmsd", "target": "pocket",
                        "over": "production"}]}, "MANDATE_STAGE"),
    ({"deliverables": [{"name": "RMSD", "property": "rmsd", "target": "ligand",
                        "over": "prod"}]}, "MANDATE_STAGE"),
    # a ramp needs a thermostat to carry it (GROMACS has none), and an earlier stage to
    # ramp from
    ({"stages": [STAGES[0], {**STAGES[1], "temperature_k": {"to_k": 300}},
                 *STAGES[2:]], "decisions_yours": []}, "MANDATE_STAGE"),
    ({**QMMM_WRITE, "decisions_yours": [], "stages": [
        {**QMMM_WRITE["stages"][0], "temperature_k": {"to_k": 300}},
        {**QMMM_WRITE["stages"][0], "name": "cool"}]}, "MANDATE_STAGE"),
    # a stage starts from an earlier stage, or itself to extend its run
    ({**QMMM_WRITE, "decisions_yours": [], "stages": [
        {**QMMM_WRITE["stages"][0], "starts_from": "cool"},
        {**QMMM_WRITE["stages"][0], "name": "cool"}]}, "MANDATE_STAGE"),
    ({"hamiltonian": {"mm": "amber", "qm": {"method": "blyp_mt"},
                      "coupling": {"scheme": "mimic"}},
      "stages": [{"name": "hot", "kind": "dynamics", "regime": "qmmm", "ensemble": "NPT",
                  "temperature_k": 300, "pressure_bar": 1, "steps": 100,
                  "qm_region": {"subsystems": ["ligand"], "charge": 0}}],
      "decisions_yours": []}, "MANDATE_STAGE"),
    # a residue range is two numbers in a pair, never a word
    ({"subsystems": {**SUBSYSTEMS, "protein": {**SUBSYSTEMS["protein"], "resid": [[1, "x"]]}}},
     "INVALID_PARAMS"),
]


@pytest.mark.parametrize("change,code,said", [(*row, 1)[:3] for row in UNWRITTEN])
def test_a_mandate_that_cannot_bind_is_not_written(ws_ctx, complex_system, change, code,
                                                   said):
    from myrmidon.core import mandate as mandates

    obs = _write(ws_ctx, complex_system, **change)
    assert (obs.ok, obs.error.code) == (False, code), obs.summary
    assert [f.id for f in obs.findings] == ([code] * said if code != "INVALID_PARAMS"
                                            else []), obs.summary
    assert mandates.read_mandate(ws_ctx.workspace) is None


def test_a_run_of_no_named_stage_is_held_to_every_stages_locks_and_stands_in_for_none(
        ws_ctx, complex_system):
    from myrmidon.core.mandate import deliverable_findings

    ws = ws_ctx.workspace
    _approved(ws_ctx, complex_system, deliverables=[
        {"name": "potential energy", "tool": "gmx_energy", "over": "production",
         "params": {"terms": ["Potential"]}}])
    adhoc = {"inputset": "gromacs.nvt", "topology": complex_system["topology"],
             "overrides": {"nsteps": 500}}
    _, hot = _run_input(ws_ctx, complex_system,
                        {**adhoc, "overrides": {"nsteps": 500, "ref_t": 310}}, "adhoc_hot")
    refused = REGISTRY.call("gmx_mdrun", ws_ctx, {"tpr": hot, "preview": True})
    assert {w.id for w in blocking(refused)} == {"MANDATE_DECISION"}, refused.summary
    _, tpr = _run_input(ws_ctx, complex_system, {**adhoc, "overrides": {"nsteps": 500,
                                                                       "ref_t": 300}}, "adhoc")
    job = REGISTRY.call("gmx_mdrun", ws_ctx, {"tpr": tpr, "deffnm": "adhoc"})
    assert job.ok, job.summary
    done = REGISTRY.call("job_status", ws_ctx, {"job_id": job.job_id, "wait_seconds": 30})
    assert done.ok, done.summary
    edr = next(a.path for a in done.artifacts if a.path.endswith(".edr"))

    said = deliverable_findings(ws, "gmx_energy", {"edr": edr, "terms": ["Potential"]})
    assert [f.id for f in said] == ["DELIVERABLE_STAGE_MISMATCH"]
    report = REGISTRY.call("study_report", ws_ctx, {})
    # every locked stage that never ran is said; one nothing locks (em) is not
    assert {v.step for v in report.values if v.label == "never ran"} == {
        "stage:nvt", "stage:npt", "stage:production"}


def test_a_schema_1_file_reads_and_hashes_as_it_was_written():
    from myrmidon.core.mandate import Mandate

    written = {"schema": 1, "question": "q", "system": {"description": "", "files": []},
               "method": {"protocol": "", "stages": ["minimise", "NPT 200 ps"]},
               "decisions": {"yours": [{"said": "300 K", "target": "gromacs.npt",
                                        "keys": {"ref_t": 300}}], "proposed": []},
               "deliverables": [{"name": "density", "tool": "gmx_energy", "over": "npt"}],
               "cost": {"estimate_core_h": 1.0, "ceiling_core_h": 10.0},
               "recovery": {"reruns": 1, "model_fixes": 2, "deadline_h": None},
               "status": "ready", "questions": [], "revision": 1, "reason": ""}
    mandate = Mandate.model_validate({**written, "digest": "d"})
    assert mandate.method.described == ["minimise", "NPT 200 ps"]
    assert mandate.body() == written


def _compiler(mandate, stage, engine, **facts):
    from myrmidon.knowledge.catalog import SystemFact
    from myrmidon.knowledge.stage import (
        MethodStore,
        TemplateStore,
        compile_stage,
        expected_keys,
        methods_for,
        templates_for,
    )

    templates = TemplateStore.builtin()
    methods = methods_for(mandate, MethodStore.builtin())
    facts = {k: SystemFact(key=k, value=v, rule="test") for k, v in facts.items()}
    view = mandate.method.stage(stage).model_dump(mode="json")
    (template,) = templates_for(view, templates, engine)
    compiled = compile_stage(mandate, stage, template, templates=templates, methods=methods,
                             facts=facts)
    expected, _ = expected_keys(mandate, stage, engine, templates=templates, methods=methods,
                                facts=facts)
    return compiled.values, {k: c for k, (c, _, _) in expected.items()}


QMMM = {"hamiltonian": {"mm": "amber", "qm": {"method": "blyp_mt"},
                        "coupling": {"scheme": "mimic"}},
        "subsystems": {"solute": {"kind": "ligand", "resnames": ["ACT"]}}}


def _qmmm_stage(**over):
    return {"name": "prod", "kind": "dynamics", "regime": "qmmm", "ensemble": "NVT",
            "temperature_k": 300, "steps": 1000,
            "qm_region": {"subsystems": ["solute"], "charge": 0}, **over}


# (the system, the stage, its locks, the engine, what it compiles to, what a file must hold
# there, a value a file may hold, a value it may not). A lock binds the keys whose value
# follows the locked field — a length binds the step too —, on a CPMD line its first number
# only, the thermostat's frequency after it the template's.
COMPILES = [
    ({"hamiltonian": {"mm": "amber"}}, {**STAGES[3]},
     {"production.duration_ps": 1000}, "gromacs",
     {"nsteps": 500000, "dt": 0.002}, ("dt", 0.002), "0.002", "0.004"),
    (QMMM, _qmmm_stage(), {"prod.temperature_k": {"min": 290, "max": 310}}, "cpmd",
     {"&CPMD/NOSE IONS": "300 4000", "&CPMD/MAXSTEP": 1000},
     ("&CPMD/NOSE IONS", {"min": [290, None], "max": [310, None]}), "300 2500", "320 4000"),
    ({**QMMM, "subsystems": {"solute": {"kind": "ligand", "resnames": ["ACT"]}}},
     _qmmm_stage(qm_region={"subsystems": ["solute"], "charge": 0, "multiplicity": 2}),
     {"prod.qm_region": {"multiplicity": 2}}, "cpmd",
     {"&CPMD/LSD": True, "&SYSTEM/MULTIPLICITY": 2},
     ("&SYSTEM/MULTIPLICITY", 2), "2", "1"),
    ({"hamiltonian": {"mm": "charmm36"},
      "subsystems": {"bilayer": {"kind": "membrane", "resnames": ["POPC"]}}},
     {**STAGES[3], "name": "production"}, {"production.pressure_bar": 1}, "gromacs",
     {"pcoupltype": "semiisotropic", "ref-p": "1 1", "compressibility": "4.5e-05 4.5e-05",
      "rcoulomb": 1.2, "vdw-modifier": "force-switch"}, ("ref-p", 1), "1 1", "1 2"),
]


@pytest.mark.parametrize("system,stage,locks,engine,compiles,expects,meets,breaks", COMPILES)
def test_a_stage_compiles_and_its_locks_bind_what_follows_them(
        system, stage, locks, engine, compiles, expects, meets, breaks):
    from myrmidon.core.lineage import satisfies
    from myrmidon.core.mandate import Mandate

    mandate = Mandate.model_validate({
        "question": "q", "system": system, "method": {"stages": [stage]},
        "decisions": {"yours": [{"said": "theirs", "locks": locks}]}})
    values, expected = _compiler(mandate, stage["name"], engine)
    for key, value in compiles.items():
        assert values[key] == value, values
    key, constraint = expects
    assert expected[key] == constraint, expected
    assert satisfies(meets, constraint) and not satisfies(breaks, constraint)


# (an every-stage lock, the key a file of no named stage is held on, (what it must hold,
# whose words, whether leaving it unset breaks it), a value that meets it, one that breaks
# it): a range binds every group's number and a file need not thermostat; a restraint list
# binds the switch that holds those subsystems, an empty one no switch at all
ANYWHERE = [
    ({"*.temperature_k": {"min": 290, "max": 310}}, "ref-t",
     ({"min": 290, "max": 310}, "theirs", False), "300 300", "320"),
    ({"*.restraints": ["protein"]}, "define", ("-DPOSRES", "theirs", True), None, None),
    ({"*.restraints": []}, "define", ("", "theirs", True), None, None),
]


@pytest.mark.parametrize("locks,key,holds,meets,breaks", ANYWHERE)
def test_a_run_of_no_named_stage_is_held_to_every_stages_locks_as_stated(
        locks, key, holds, meets, breaks):
    from myrmidon.core.lineage import satisfies
    from myrmidon.core.mandate import Mandate
    from myrmidon.knowledge.catalog import SystemFact
    from myrmidon.knowledge.stage import MethodStore, TemplateStore, expected_keys, methods_for

    mandate = Mandate.model_validate({
        "question": "q", "system": {"subsystems": SUBSYSTEMS, "hamiltonian": {"mm": "amber"}},
        "method": {"stages": STAGES},
        "decisions": {"yours": [{"said": "theirs", "locks": locks}]}})
    facts = {"define": SystemFact(key="define", value="-DPOSRES -DPOSRES_LIG",
                                  rule="restraint_defines",
                                  parts={"Protein_chain_A": "-DPOSRES", "JZ4": "-DPOSRES_LIG"})}
    expected, unresolved = expected_keys(
        mandate, None, "gromacs", templates=TemplateStore.builtin(),
        methods=methods_for(mandate, MethodStore.builtin()), facts=facts)
    assert unresolved == [] and expected[key] == holds, expected[key]
    assert meets is None or (satisfies(meets, holds[0]) and not satisfies(breaks, holds[0]))


def test_a_stage_file_edited_by_hand_is_still_held_to_its_stage(ws_ctx, complex_system):
    import re

    _approved(ws_ctx, complex_system)
    _run_input(ws_ctx, complex_system, {"stage": "production"}, "production")
    mdp = ws_ctx.workspace.root / "02_run/production.mdp"
    short = re.sub(r"(?m)^nsteps\s*=.*$", "nsteps = 125000", mdp.read_text())
    edited = REGISTRY.call("file_write", ws_ctx, {"path": "02_run/production.mdp",
                                                  "content": short})
    assert edited.ok, edited.summary
    tpr = REGISTRY.call("gmx_grompp", ws_ctx, {
        "mdp": "02_run/production.mdp", "structure": complex_system["structure"],
        "topology": complex_system["topology"], "output": "short.tpr", "stage": "run"})
    obs = REGISTRY.call("gmx_mdrun", ws_ctx, {"tpr": tpr.artifacts[0].path, "preview": True})
    assert [w.id for w in blocking(obs)] == ["MANDATE_DECISION"], obs.summary


DELIVERABLES = [
    {"name": "ligand pocket RMSD", "property": "rmsd", "target": "ligand", "atoms": "heavy",
     "align_to": {"subsystem": "protein", "atoms": "backbone"}, "over": "production",
     "statistics": ["mean", "max"]},
    {"name": "protein-ligand minimum distance", "property": "min_distance",
     "target": "ligand", "reference": "protein", "over": "production",
     "statistics": ["mean", "min"]},
]


TWO_CHAINS = {"receptor": {"kind": "protein", "molecules": ["Protein_chain_A"]},
              "peptide": {"kind": "protein", "molecules": ["Protein_chain_B"],
                          "chains": ["B"]},
              "ligand": {"kind": "ligand", "resnames": ["JZ4"]}}

# (the deliverable, a parameter it compiles to — or None, a problem keeps it from being
# measured as stated). Two proteins are told apart by their molecule types, a superposed
# ligand by its heavy atoms on the receptor's backbone from the stage's first frame; a
# filter a subsystem has no atoms for, and a measurement the harness cannot make, are said.
OBSERVABLES = [
    ({"property": "min_distance", "target": "peptide", "reference": "receptor"},
     ("selection2", "moltype Protein_chain_A and not name H*")),
    ({"property": "rmsd", "target": "peptide", "atoms": "backbone"},
     ("selection", "moltype Protein_chain_B and chainID B and backbone")),
    ({"property": "rmsd", "target": "ligand", "atoms": "backbone"}, None),
    ({"property": "rmsd", "target": "ligand", "reference_frame": "minimised"},
     None),
    ({"property": "pair_distance", "target": "ligand"}, None),
    ({"property": "thermodynamics"}, ("terms", ["Temperature", "Pressure", "Density",
                                                "Potential"])),
    ({"property": "rmsd", "target": "ligand", "atoms": "heavy",
      "align_to": {"subsystem": "receptor", "atoms": "backbone"}},
     ("fit_selection", "moltype Protein_chain_A and backbone")),
    # superposed on itself, a ligand's RMSD is its change of shape: measured, and noted
    ({"property": "rmsd", "target": "ligand", "atoms": "heavy",
      "align_to": {"subsystem": "ligand", "atoms": "heavy"}},
     ("fit_selection", "resname JZ4 and not name H*")),
    ({"property": "rmsd", "target": "ligand", "atoms": "heavy"}, ("reference_frame", 0)),
    ({"property": "min_distance", "target": "ligand", "reference": "receptor",
      "atoms": "heavy"}, ("selection", "resname JZ4 and not name H*")),
]


@pytest.mark.parametrize("deliverable,compiles", OBSERVABLES)
def test_a_deliverable_compiles_to_its_measurement_or_says_why_it_cannot(deliverable,
                                                                         compiles):
    from myrmidon.core.mandate import Deliverable, Mandate
    from myrmidon.knowledge.observables import compile_deliverable

    mandate = Mandate.model_validate({"question": "q",
                                      "system": {"subsystems": TWO_CHAINS}})
    out = compile_deliverable(mandate, Deliverable(name="d", **deliverable))
    if compiles is None:
        assert len(out.problems) == 1, out.problems
    else:
        assert out.problems == [] and out.params[compiles[0]] == compiles[1], out
        assert bool(out.notes) is (deliverable.get("align_to", {}).get("subsystem")
                                   == deliverable.get("target") == "ligand")


def test_a_deliverable_is_measured_as_approved_over_its_stages_run(ws_ctx, complex_system,
                                                                     monkeypatch, tmp_path):
    """Named, a deliverable takes the mandate's selections and the trajectory its stage's
    collected run wrote, found in the ledger; the call reports the statistics asked for."""
    mda = pytest.importorskip("MDAnalysis")
    import numpy as np

    ws = ws_ctx.workspace
    _approved(ws_ctx, complex_system, deliverables=DELIVERABLES[1:])
    before = REGISTRY.call("traj_analyze", ws_ctx, {
        "deliverable": DELIVERABLES[1]["name"], "structure": complex_system["structure"]})
    assert not before.ok  # no run of its stage is collected yet

    frames = tmp_path / "frames.xtc"
    u = mda.Universe(str(ws.root / complex_system["structure"]))
    with mda.Writer(str(frames), n_atoms=len(u.atoms)) as w:
        for i in range(5):  # the ligand moves away along x, 1 Å a frame
            pos = u.atoms.positions.copy()
            pos[2:, 0] += float(i)
            u.atoms.positions = pos
            u.trajectory.ts.time = float(i)
            w.write(u.atoms)
    monkeypatch.setenv("MOCK_GMX_XTC", str(frames))  # the frames the run writes
    _, tpr = _run_input(ws_ctx, complex_system, {"stage": "production"}, "production")
    job = REGISTRY.call("gmx_mdrun", ws_ctx, {"tpr": tpr, "deffnm": "production"})
    assert job.ok, job.summary
    done = REGISTRY.call("job_status", ws_ctx, {"job_id": job.job_id, "wait_seconds": 30})
    xtc = next(a.path for a in done.artifacts if a.path.endswith(".xtc"))

    obs = REGISTRY.call("traj_analyze", ws_ctx, {
        "deliverable": DELIVERABLES[1]["name"], "structure": complex_system["structure"],
        "selection": "all"})
    assert obs.ok, obs.summary
    assert obs.selection == "resname JZ4 and not name H*"  # the mandate's, not the call's
    assert "DELIVERABLE_MEASURED_AS_APPROVED" in {f.id for f in obs.findings}
    assert set(obs.statistics) == {"mean", "min"}
    assert np.isclose(obs.statistics["min"], 4.0, atol=1e-3)  # C2 to C1, frame 0
    event = ws.ledger.get(obs.provenance_id)
    assert any(i.path == xtc for i in event.inputs)
    # the ledger records what was measured, not what the call left unset
    assert event.params["selection"] == "resname JZ4 and not name H*"
    assert event.params["trajectory"] == xtc

    # bytes changed since the run wrote them are not that run's
    (ws.root / xtc).write_bytes((ws.root / xtc).read_bytes() + b"\0")
    stale = REGISTRY.call("traj_analyze", ws_ctx, {
        "deliverable": DELIVERABLES[1]["name"], "structure": complex_system["structure"]})
    assert not stale.ok
    (ws.root / xtc).write_bytes(frames.read_bytes())
    report = REGISTRY.call("study_report", ws_ctx, {})
    measured = {v.label: v for v in report.values
                if v.step == f"deliverable:{DELIVERABLES[1]['name']}"}
    assert set(measured) == {"mean", "min"} and np.isclose(measured["min"].value, 4.0, atol=1e-3)
    assert measured["min"].event == obs.provenance_id


# (the stage run, the paces this machine measured, the ranks the call books, the walltime
# booked on one exclusive 48-core node). A minimisation books the template's walltime;
# dynamics its length at the measured pace scaled to the system, else at the plausible pace,
# with a margin, in the scheduler's 5-min steps. Ranks left unset fill the node.
PACE = {"engine": "gromacs", "atoms": 4, "cores": 48, "ns_per_day": 10.0}
SIZINGS = [
    ("em", [], 48, 5),
    ("nvt", [], 48, 10),  # unmeasured: the plausible pace is a floor, a margin books
    ("nvt", [PACE], 48, 25),
    ("nvt", [{**PACE, "atoms": 8}], 48, 15),  # a pace measured on twice the atoms
    ("nvt", [PACE], None, 25),
]


@pytest.mark.parametrize("stage,paces,ntasks,minutes", SIZINGS)
def test_a_run_books_the_walltime_its_length_needs_at_the_measured_pace(
        ws_ctx, complex_system, stage, paces, ntasks, minutes):
    from myrmidon.execution.pace import record_pace
    from myrmidon.knowledge.stage import Pace

    compute = ws_ctx.config.compute
    compute.cores_per_node, compute.node_exclusive = 48, True
    for pace in paces:
        record_pace(ws_ctx.config.studies_root, Pace(**pace, where="measured"))
    _approved(ws_ctx, complex_system)
    _, tpr = _run_input(ws_ctx, complex_system, {"stage": stage}, stage)
    ranks = {"ntasks": ntasks} if ntasks else {}
    obs = REGISTRY.call("gmx_mdrun", ws_ctx, {"tpr": tpr, **ranks, "preview": True})
    assert (obs.ok, obs.walltime_min, obs.ntasks) == (True, minutes, 48), obs.summary


def test_a_coupled_stage_is_priced_from_the_seconds_a_qmmm_step_took_here():
    from pathlib import Path

    from myrmidon.config import ComputeDefaults
    from myrmidon.core.mandate import Mandate
    from myrmidon.engines.mimic.diagnostics import seconds_per_step
    from myrmidon.knowledge.stage import MethodStore, Pace, TemplateStore, methods_for, price_stage

    log = Path(__file__).parents[1] / "fixtures" / "cpmd" / "nose_steps.out"
    seconds = seconds_per_step(log.read_text())  # the real step lines: 15.27 s and 14.46 s
    mandate = Mandate.model_validate({"question": "q", "system": QMMM,
                                      "method": {"stages": [_qmmm_stage(steps=1000)]}})
    priced = [price_stage(mandate, mandate.method.stages[0], templates=TemplateStore.builtin(),
                          methods=methods_for(mandate, MethodStore.builtin()), atoms=None,
                          paces=paces, compute=ComputeDefaults(cores_per_node=48,
                                                               node_exclusive=True))
              for paces in ([], [Pace(engine="cpmd", atoms=10, cores=48, s_per_step=seconds)])]
    # unmeasured, unpriced; measured, 1000 steps × 15.27 s × 1.5 in the scheduler's 5-min steps
    assert [(p.core_h is None, p.measured) for p in priced] == [(True, False), (False, True)]
    assert priced[1].walltime_min == 385


def test_a_file_the_topology_includes_is_the_system_and_an_input(ws_ctx, complex_system):
    """A molecule's .itp the topology includes is in the mandate's system files and among
    the inputs of each call that reads the topology: editing it lapses the approval."""
    from myrmidon.core import mandate as mandates

    ws = ws_ctx.workspace
    top = ws.root / complex_system["topology"]
    text = top.read_text()
    head, ligand = text.split("[ moleculetype ]\nJZ4")
    ligand, rest = ligand.split("[ system ]")
    (top.parent / "jz4.itp").write_text("[ moleculetype ]\nJZ4" + ligand)
    top.write_text(head + '#include "jz4.itp"\n\n[ system ]' + rest)

    written = _approved(ws_ctx, complex_system)
    held = mandates.holding(ws)
    assert {f.path: f.kind for f in held.system.files}["01_input/jz4.itp"] == "itp"
    _, tpr = _run_input(ws_ctx, complex_system, {"stage": "nvt"}, "nvt")
    grompp = [e for e in ws.ledger.events() if e.tool == "gmx_grompp"][-1]
    assert {i.path: i.role for i in grompp.inputs}["01_input/jz4.itp"] == "topology.include"
    obs = REGISTRY.call("gmx_mdrun", ws_ctx, {"tpr": tpr, "preview": True})
    assert obs.ok, obs.summary

    (top.parent / "jz4.itp").write_text((top.parent / "jz4.itp").read_text() + "\n")
    assert mandates.holding(ws) is None
    assert "jz4.itp" in mandates.why_not_holding(ws) and written.ok



# (what the call writes, what the written mandate holds): priced on this machine; the
# subsystems and the force field's method read from the topology when left unstated; a
# Hamiltonian in one word, a residue range however spelled, a ligand numbered as the
# deposit numbers it (the one residue its name selects, said), a lock of nothing on a list
# field, a property by its tool's name — each read as meant
def _protein_selection(m):
    from myrmidon.knowledge.observables import subsystem_selection

    return subsystem_selection(m.system.subsystems["protein"], m)


#: a coupled stage no QM step measured here prices yet
QM_STAGE = {"name": "qmmm", "kind": "dynamics", "regime": "qmmm", "ensemble": "NVT",
            "temperature_k": 300, "steps": 100,
            "qm_region": {"subsystems": ["ligand"], "charge": 0}}

WRITTEN = [
    ({}, lambda m, o: m.cost.estimate_core_h > 0 and not o.findings),
    # a lock that binds no compiled key is dropped, its words kept, and said: on a field of
    # the mandate outside the stages, on one nothing compiled reads, on a mapping's item
    *(({**over, "decisions_yours": [{"said": "as stated", "locks": locks}]},
       lambda m, o, kept=kept: m.decisions.yours[0].locks == kept
       and "MANDATE_NOTE" in [f.id for f in o.findings])
      for over, locks, kept in (
          ({}, {"ceiling_core_h": 500}, {}),
          ({}, {"em.kind": "minimize"}, {}),
          (QMMM_WRITE, {"hot.qm_region.padding_nm": 0.4}, {}),
          (QMMM_WRITE, {"*.qm_region": {"subsystems": ["ligand"], "charge": 0}},
           {"*.qm_region": {"charge": 0}}))),
    # the estimate names what it leaves out
    ({"stages": [*STAGES, QM_STAGE]},
     lambda m, o: m.cost.unpriced == ["qmmm"] and m.cost.estimate_core_h > 0),
    ({"subsystems": {}}, lambda m, o: o.facts and (
        m.system.subsystems["ligand"].resnames, m.system.subsystems["protein"].molecules,
        m.system.hamiltonian.mm) == (["JZ4"], ["Protein_chain_A"], "amber")),
    *(({"subsystems": {**SUBSYSTEMS, "protein": {**SUBSYSTEMS["protein"], "resid": given}}},
       lambda m, o, selects=selects: _protein_selection(m) == selects)
      for given, selects in ((["1-163"], "protein and resid 1:163"),
                             ([[1, 163]], "protein and resid 1:163"),
                             ("1-163", "protein and resid 1:163"),
                             ([1, 163], "protein and resid 1 163"))),
    *(({"subsystems": {**SUBSYSTEMS, "ligand": {**SUBSYSTEMS["ligand"], "resid": [167],
                                                **chains}}, **measured},
       lambda m, o: m.system.subsystems["ligand"].resid == []
       and [f.id for f in o.findings] == ["MANDATE_SUBSYSTEM_NOTE"])
      for chains in ({}, {"chains": ["A"]})
      for measured in ({}, {"deliverables": [{"name": "RMSD", "property": "rmsd",
                                              "target": "ligand", "over": "production"}]})),
    # a polymer's residue keeps its number: a build renames residues, so the name is no proof
    ({"subsystems": {**SUBSYSTEMS, "site": {"kind": "protein", "resnames": ["ALA"],
                                            "resid": [5]}}},
     lambda m, o: m.system.subsystems["site"].resid == [5]
     and [f.id for f in o.findings] == ["MANDATE_SUBSYSTEM_NOTE"]),
    *(({"hamiltonian": given}, lambda m, o: m.system.hamiltonian.mm == "amber")
      for given in ("mm", " MM ", "amber", {"mm": "amber"}, {"mm": "amber99sb-ildn"})),
    ({"decisions_yours": [*LOCKS, {"said": "unrestrained production",
                                   "locks": {"production.restraints": None}}]},
     lambda m, o: m.decisions.yours[-1].locks == {"production.restraints": []}),
    ({"deliverables": [{"name": "contact", "property": "mindist", "target": "ligand",
                        "reference": "protein", "over": "production"}]},
     lambda m, o: m.deliverables[0].property == "min_distance"),
]


@pytest.mark.parametrize("change,holds", WRITTEN)
def test_a_mandate_spelled_otherwise_is_written_as_meant(ws_ctx, complex_system, change,
                                                         holds):
    from myrmidon.core.mandate import read_mandate

    written = _write(ws_ctx, complex_system, **change)
    assert written.ok, written.summary
    assert holds(read_mandate(ws_ctx.workspace), written)


# (what the mandate names the MM Hamiltonian, the force field the topology includes, the
# method that computes it — or None, said as missing). A name must agree with the
# topology; one that resolves to nothing falls back to it; the longest prefix wins.
RESOLUTIONS = [
    ("amber", "amber99sb-ildn", "amber"),
    ("charmm36", "amber99sb-ildn", None),  # an id must agree too
    ("amber99sb-ildn", "amber99sb-ildn", "amber"),
    ("amber99sb-ildn", "charmm36-jul2022", None),
    ("ambr", "amber99sb-ildn", "amber"),
    ("", "oplsaa", None),
    ("amber14sb-foo", "", "amber14sb"),
]


@pytest.mark.parametrize("named,included,resolved", RESOLUTIONS)
def test_the_force_field_resolves_to_the_method_that_computes_it(tmp_path, named, included,
                                                                 resolved):
    import shutil

    from myrmidon.core.mandate import Mandate
    from myrmidon.knowledge.stage import MethodStore, methods_for

    site = tmp_path / "methods"
    site.mkdir()
    shutil.copy(MethodStore.builtin_dir() / "amber.yaml", site / "amber14sb.yaml")
    text = (site / "amber14sb.yaml").read_text()
    (site / "amber14sb.yaml").write_text(text.replace("id: amber", "id: amber14sb").replace(
        "forcefields: [amber]", "forcefields: [amber14sb]"))
    store = MethodStore(site_path=site)
    mandate = Mandate.model_validate({"question": "q",
                                      "system": {"hamiltonian": {"mm": named}}})
    methods = methods_for(mandate, store, included or None)
    assert (methods.specs["mm"].id if "mm" in methods.specs else None) == resolved
    assert bool(methods.missing) is (resolved is None)


# (the stage's temperature, the ramp a render tells the engine): a ramp starts where its
# stage-before ended and reaches the target its template's hold (5000 a.u.) before the end
# of 1500 steps of 10 a.u.; a stated start or rate stands
RAMPS = [({"to_k": 300}, (3.4, 300, 0.02966)),
         ({"to_k": 300, "rate_k_per_au": 0.05}, (3.4, 300, 0.05)),
         ({"to_k": 300, "start_k": 10}, (10, 300, 0.029)),
         (300, None)]


@pytest.mark.parametrize("temperature,ramp", RAMPS)
def test_a_ramp_starts_where_the_stage_before_ended(temperature, ramp):
    from myrmidon.core.mandate import Ramp, Stage
    from myrmidon.knowledge.stage import PS_PER, ramp_started

    stage = Stage.model_validate(_qmmm_stage(name="heat", temperature_k=temperature,
                                             steps=1500))
    out = ramp_started(stage, 3.4, 10 * PS_PER["au"], hold_au=5000).temperature_k
    assert ((out.start_k, out.to_k, out.rate_k_per_au) if isinstance(out, Ramp)
            else None) == ramp


def _coupled_study(ws_ctx, stages):
    """Acetone in water, its QM region selected, under an approved mandate of these stages."""
    from myrmidon.core import mandate as mandates

    ws = ws_ctx.workspace
    stage = ws.stage("input")
    for name in ("acetone_equilibrated.top", "acetone.itp", "acetone_equilibrated.gro"):
        (stage / name).write_text((ACETONE / name).read_text())
    shutil.copytree(ACETONE / "amber03.ff", stage / "amber03.ff")
    rel = ws.relpath
    top, gro = rel(stage / "acetone_equilibrated.top"), rel(stage / "acetone_equilibrated.gro")
    selected = REGISTRY.call("qm_region", ws_ctx, {
        "topology": top, "coords": gro, "selections": ["add resname is ACT"], "padding_nm": 0.4})
    assert selected.ok, selected.summary
    written = REGISTRY.call("mandate_write", ws_ctx, {
        "question": "Run acetone in water under QM/MM", "ceiling_core_h": 5000.0,
        "system_files": [{"path": gro, "kind": "structure"}, {"path": top, "kind": "topology"}],
        "subsystems": {"solute": {"kind": "ligand", "resnames": ["ACT"]},
                       "solvent": {"kind": "solvent", "molecules": ["SOL"]}},
        **{k: v for k, v in QMMM.items() if k != "subsystems"},
        "stages": stages})
    assert written.ok and written.status == "ready", written.summary
    held = mandates.read_mandate(ws)
    mandates.record_approval(ws, held, approver="researcher", via="terminal")
    return held


REGION = {"subsystems": ["solute"], "charge": 0}


def test_a_coupled_stage_is_prepared_in_one_call_from_the_mandate_alone(ws_ctx):
    """Each stage of a coupled study gets a complete run input and its compiled MM half, the
    selection's files are the mandate's system, and the charge and spin are the stage's."""
    from myrmidon.engines.mimic.cpmd_input import keyword_value, parse_cpmd

    held = _coupled_study(ws_ctx, [
        {"name": "anneal", "kind": "minimize", "regime": "qmmm", "steps": 300,
         "temperature_k": 300, "qm_region": REGION},
        {"name": "nve_check", "kind": "dynamics", "regime": "qmmm", "ensemble": "NVE",
         "steps": 1000, "starts_from": "anneal", "qm_region": REGION},
        {"name": "heat", "kind": "dynamics", "regime": "qmmm", "ensemble": "NVT", "steps": 1500,
         "starts_from": "anneal", "temperature_k": {"to_k": 300, "start_k": 3.4},
         "qm_region": REGION},
        {"name": "production", "kind": "dynamics", "regime": "qmmm", "ensemble": "NVT",
         "steps": 5000, "temperature_k": 300, "qm_region": REGION}])
    assert {f.kind for f in held.system.files} >= {"cpmd_input", "index", "qm_selection"}
    assert held.method.starts_from("heat").name == "anneal"
    assert held.method.starts_from("production").name == "heat"
    ws = ws_ctx.workspace
    for name in ("anneal", "nve_check", "heat", "production"):
        obs = REGISTRY.call("stage_prepare", ws_ctx, {"stage": name})
        assert obs.ok, obs.summary
        assert {"cpmd_input", "mdp", "tpr"} <= set(obs.files)
        sections = parse_cpmd(ws.resolve(obs.files["cpmd_input"]).read_text())
        assert any(line.startswith("*C_MT") for line in sections["ATOMS"])
        assert keyword_value(sections, "&MIMIC/BOX") is not None
        assert keyword_value(sections, "&SYSTEM/CHARGE") == "0"
        sidecar = json.loads(ws.resolve(obs.files["cpmd_input"] + ".myrmidon.json").read_text())
        assert (sidecar["charge"], sidecar["multiplicity"]) == (0, 1)
        assert sidecar["stage"] == name
    tools = {ev.tool for ev in ws.ledger.events()}
    assert "stage_prepare" in tools and not tools & {"mimic_prepare", "gmx_grompp"}
    from myrmidon.engines.file_origin import edit_findings

    assert not edit_findings(ws_ctx, obs.files["cpmd_input"])


def test_a_stage_after_a_classical_run_starts_with_a_fresh_wavefunction(ws_ctx):
    """A coupled stage continues a run's restart, unless the run it starts from left none."""
    from myrmidon.core.lineage import stage_run as collected
    from myrmidon.engines.mimic.cpmd_input import keyword_value, parse_cpmd

    _coupled_study(ws_ctx, [
        {"name": "em", "kind": "minimize", "steps": 100},
        {"name": "first", "kind": "dynamics", "regime": "qmmm", "ensemble": "NVE",
         "steps": 10, "qm_region": REGION}])
    restart = "&CPMD/RESTART COORDINATES VELOCITIES WAVEFUNCTION"
    ws = ws_ctx.workspace
    assert collected(ws, "em") is None
    before = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "first"})
    assert before.ok, before.summary
    sections = parse_cpmd(ws.resolve(before.files["cpmd_input"]).read_text())
    assert keyword_value(sections, restart) == "True"  # what the run before leaves is not known
    em = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "em"})
    assert em.ok, em.summary
    run = REGISTRY.call("gmx_mdrun", ws_ctx, {"tpr": em.files["tpr"], "walltime_min": 5})
    assert run.ok, run.summary
    assert REGISTRY.call("job_status", ws_ctx, {"job_id": run.job_id,
                                                "wait_seconds": 15}).ok
    after = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "first"})
    assert after.ok, after.summary
    sections = parse_cpmd(ws.resolve(after.files["cpmd_input"]).read_text())
    assert keyword_value(sections, restart) is None


def test_a_fit_stage_is_compiled_to_its_input_over_the_collected_dataset(ws_ctx):
    """A parameter_fit stage compiles to fminput.dat in one call: its charge is the stage's QM
    region's, its stride every frame of the dataset a collection registered."""
    from pydantic import BaseModel

    from myrmidon.core.artifacts import ArtifactRef
    from myrmidon.core.observation import Observation
    from myrmidon.core.tool import ToolRegistry, tool
    from myrmidon.engines.mimic.fm import parse_fminput

    _coupled_study(ws_ctx, [
        {"name": "reference", "kind": "single_point", "regime": "qmmm", "qm_region": REGION},
        {"name": "fit", "kind": "parameter_fit", "qm_region": {**REGION, "charge": -1}}])
    ws = ws_ctx.workspace
    no_data = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "fit"})
    assert not no_data.ok and "stride" in no_data.error.message
    given = REGISTRY.call("stage_prepare", ws_ctx, {
        "stage": "fit", "overrides": {"stride": "[0,10,1]"}})
    assert given.ok, given.summary
    assert parse_fminput(ws.resolve(given.files["fm_input"]))["stride"] == "[0,10,1]"
    # the mock's h5info counts 12 configurations; what is read is said with what it read
    stage = ws.stage("fm")
    (stage / "reference.h5").write_bytes(b"h5")
    registry = ToolRegistry()

    class _Nothing(BaseModel):
        pass

    @tool("collected_here", description="d", registry=registry)
    def collected_here(ctx, params: _Nothing) -> Observation:
        return Observation(ok=True, summary="a dataset", artifacts=[
            ArtifactRef.from_path(stage / "reference.h5", "fm_dataset", root=ws.root)])

    registry.call("collected_here", ws_ctx, {})
    fit = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "fit"})
    assert fit.ok, fit.summary
    assert fit.files["fm_input"].endswith("fminput.dat")
    written = parse_fminput(ws.resolve(fit.files["fm_input"]))
    assert (written["stride"], written["qm_total_charge"]) == ("[0,12,1]", "-1")
    assert "FM_FEW_SNAPSHOTS" in {f.id for f in fit.findings}
    sidecar = json.loads(ws.resolve(fit.files["fm_input"] + ".myrmidon.json").read_text())
    assert sidecar["stage"] == "fit"
    assert sidecar["origin"]["stride"] == "system:reference_dataset"
