"""The MiMiC stages, checked against the tutorial inputs their templates were distilled from.

A template is a transcription, and a transcription can be wrong. These tests compile each
tutorial stage the way a study does — the stage as the MiMiC protocol states it, its
template, the methods, the facts of the tutorial's system — and compare it to the reference
file in tests/fixtures/mimic_tutorial/, so a typo in the YAML fails here instead of in a
simulation, and every way the harness departs from the published inputs is stated.

Excluded from the comparison, because a stage supplies the method and never the system:
&MIMIC PATHS / BOX / OVERLAPS and the whole &ATOMS block, all written per system by
mimicpy's prepqm.
"""

import shutil
from pathlib import Path

import pytest
from tests.unit.findings_catalog import blocking

from myrmidon.core.invariants import evaluate_invariants
from myrmidon.core.mandate import Mandate
from myrmidon.core.tool import REGISTRY
from myrmidon.engines.gromacs.mdp import parse_mdp, parse_mdp_file, render_mdp, validate_mdp
from myrmidon.engines.gromacs.system import coupling_groups, restraint_defines
from myrmidon.engines.mimic.cpmd_input import (
    keyword_blocks,
    parse_cpmd,
    render_cpmd_from_values,
)
from myrmidon.knowledge.catalog import SystemFact, load
from myrmidon.knowledge.protocols import ProtocolStore
from myrmidon.knowledge.stage import (
    MethodStore,
    TemplateStore,
    compile_stage,
    methods_for,
    templates_for,
    with_inherited,
)

TUTORIAL = Path(__file__).parent.parent / "fixtures" / "mimic_tutorial"
ACETONE = Path(__file__).parent.parent / "fixtures" / "acetone"
PER_SYSTEM = {"PATHS", "BOX", "OVERLAPS"}  # prepqm writes these, a stage cannot


def _number(token: str) -> str:
    """CPMD reads 10, 10.0 and 10. as one number."""
    try:
        return f"{float(token):g}"
    except ValueError:
        return token.upper()


def _keyword(section: str, line: str) -> tuple:
    """A keyword line as CPMD reads it: the keyword it matches (OPTIMIZE WAVEFUNCTION is
    WAVEFUNCTION OPTIMIZATION) and the option words beside it."""
    key = load("cpmd").keyword(section, line)
    if key is None:
        return (line.strip().upper(),)
    own = set(" ".join([key.name, *key.aliases]).upper().split())
    return (key.name, frozenset(w for w in line.upper().split() if w not in own))


def _method_sections(text: str) -> dict[str, set[tuple]]:
    """Each section's keyword blocks, minus the per-system &MIMIC entries: a keyword line
    as CPMD reads it, its values by number; order inside a section is not CPMD's."""
    return {
        name: {(_keyword(name, block[0]),
                *(tuple(_number(t) for t in line.split()) for line in block[1:] if line.strip()))
               for key, block in keyword_blocks(lines) if key and key not in PER_SYSTEM}
        for name, lines in parse_cpmd(text).items()
        if name not in ("ATOMS", "INFO")
    }


MIMIC = (("MIMIC", frozenset()),)  # &CPMD MIMIC, the coupling's
CHARGE = (("CHARGE", frozenset()), ("0",))  # &SYSTEM CHARGE 0, the stage's QM region


#: what the tutorial's system files say: acetone in SPC water, a structure at rest after
#: minimisation, prepqm's QM group
FACTS = {f.key: f for f in (
    coupling_groups(ACETONE / "acetone_equilibrated.top", []),
    restraint_defines(ACETONE / "acetone_equilibrated.top", []),
    SystemFact(key="continuation", value="no", rule="starting_state:at_rest"),
    SystemFact(key="gen-vel", value="yes", rule="starting_state:at_rest"),
    SystemFact(key="QMMM-grps", value="qmatoms", rule="qm_group"),
)}
#: the tutorial's parts: acetone is the ligand the QM region holds, in water
SUBSYSTEMS = {"ligand": {"kind": "ligand", "molecules": ["ACT"]},
              "solvent": {"kind": "solvent", "molecules": ["SOL"]}}


def protocol_mandate(protocol: str) -> Mandate:
    """A mandate whose stages and Hamiltonian are the protocol's, on the tutorial system."""
    shipped = ProtocolStore.builtin().get(protocol)
    return Mandate.model_validate({"question": protocol, "system": {
        "subsystems": SUBSYSTEMS,
        "hamiltonian": {"mm": "amber",
                        **shipped.hamiltonian.model_dump(exclude_none=True, exclude={"mm"})}},
        "method": {"stages": [s.stage.model_dump(mode="json", exclude_none=True)
                              for s in shipped.steps if s.stage is not None]}})


def compiled(stage: str, engine: str, protocol: str = "mimic_qmmm_solvated_solute",
             **overrides):
    """One stage of the protocol told one engine, as the tutorial ran it on its system."""
    mandate = protocol_mandate(protocol)
    store = TemplateStore.builtin()
    found = mandate.method.stage(stage)
    template = with_inherited(templates_for(found.model_dump(mode="json"), store, engine)[0],
                              store)
    methods = methods_for(mandate, MethodStore.builtin(), "amber03")
    return template, compile_stage(mandate, found, template, templates=store,
                                   methods=methods, facts=FACTS, overrides=overrides)


#: each tutorial file: (the protocol stage it is, the engine told it, the tutorial's own
#: choices for its system, which the stage leaves to the researcher)
FILE_BACKED = {
    "mini.mdp": ("minimise", "gromacs", {}),
    "heat.mdp": ("equilibrate", "gromacs", {}),
    "mimic.mdp": ("anneal", "gromacs", {}),
    "template.inp": ("anneal", "cpmd", {}),
}

#: where the harness departs from the published files, key -> (tutorial, harness): the
#: method's values hold for every stage, so the tutorial's per-file choices that disagree
#: with them give way, and keys left at the engine's default are not written
DIFFERENT = {
    "mini.mdp": {
        "emtol": ("100.0", "1000"),  # the minimisation template's, as for every system
        "nstlist": ("1", None),  # the engine's default neighbour-list interval
        "ns_type": ("grid", None),  # obsolete since GROMACS 2020
        # the method's, inert under steepest descent with no constraints
        "dispcorr": (None, "EnerPres"), "constraint_algorithm": (None, "lincs"),
        "constraints": (None, "none"),  # stated: minimisation relaxes every bond
    },
    "heat.mdp": {
        # the Amber method's cutoffs and constraints, not the tutorial file's; the coupled
        # run that follows takes the same method
        "rcoulomb": ("1.2", "1"), "rvdw": ("1.2", "1"),
        "constraints": ("all-bonds", "h-bonds"),
        # a box that starts from rest relaxes under stochastic cell rescaling, which grompp
        # accepts on fresh velocities where it warns of Parrinello-Rahman
        "pcoupl": ("Berendsen", "C-rescale"),
        # compressed frames at the stage's sampling, not full-precision ones
        "nstxout": ("500", None), "nstvout": ("500", None), "nstxout_compressed": (None, "5000"),
        "nstenergy": ("500", "5000"), "nstlog": ("500", "5000"),
        # the engine's defaults
        "fourierspacing": ("0.16", None), "pme_order": ("4", None), "nstlist": ("10", None),
        "lincs_iter": ("1", None), "lincs_order": ("4", None), "ns_type": ("grid", None),
    },
    "mimic.mdp": {
        "rcoulomb": ("1.2", "1"), "rvdw": ("1.2", "1"),  # the Amber method's
        "nsteps": ("100000", "300"),  # the stage's steps; CPMD's MAXSTEP ends the run
        # both engines write at the stage's one stride
        "nstxout": ("50", "10"), "nstvout": ("50", "10"), "nstenergy": ("50", "10"),
        "nstlog": ("50", None),
        "fourierspacing": ("0.16", None), "pme_order": ("4", None), "nstlist": ("10", None),
        "ns_type": ("grid", None),
        "constraint_algorithm": (None, "lincs"),  # inert under constraints = none
    },
}


def _same_value(a, b) -> bool:
    """GROMACS reads 4.5e-5 and 4.5e-05, or 1. and 1.0, as the same number."""
    if a is None or b is None:
        return a is b
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).lower() == str(b).lower()


@pytest.mark.parametrize("reference", list(FILE_BACKED))
def test_each_file_backed_stage_reproduces_the_published_input(reference):
    """Every keyword the published tutorial sets, the harness sets, with the same value —
    except the departures DIFFERENT states. The published archive is the authority: an
    earlier transcription used Car-Parrinello dynamics, and this is what would have caught
    it."""
    stage, engine, system = FILE_BACKED[reference]
    _, resolution = compiled(stage, engine, **system)
    assert not resolution.missing
    text = (TUTORIAL / reference).read_text()
    if reference.endswith(".mdp"):
        rendered = parse_mdp(render_mdp(resolution.values, resolution.origin))
        expected = parse_mdp(text)
        departures = {k: (expected.get(k), rendered.get(k))
                      for k in set(rendered) | set(expected)
                      if not _same_value(rendered.get(k), expected.get(k))}
        assert departures == DIFFERENT.get(reference, {}), departures
    else:
        rendered = _method_sections(render_cpmd_from_values(resolution.values))
        expected = _method_sections(text)
        # &CPMD MIMIC is the method's and the charge the stage's QM region: the archive's
        # template leaves both to prepqm
        assert rendered["CPMD"] - expected["CPMD"] == {MIMIC}
        assert rendered["SYSTEM"] - expected["SYSTEM"] == {CHARGE}
        rendered["CPMD"] -= {MIMIC}
        rendered["SYSTEM"] -= {CHARGE}
        assert rendered == expected


#: what each later coupled stage changes in the annealing input, as the tutorial page
#: documents it, and the NVE check's tighter wavefunction convergence
DOCUMENTED = {
    "nve_check": {"&CPMD/RESTART COORDINATES VELOCITIES WAVEFUNCTION",
                  "&CPMD/ANNEALING IONS", "&CPMD/TEMPERATURE", "&CPMD/CONVERGENCE ORBITALS"},
    "heat": {"&CPMD/RESTART COORDINATES VELOCITIES WAVEFUNCTION", "&CPMD/ANNEALING IONS",
             "&CPMD/TEMPERATURE", "&CPMD/TEMPERATURE RAMP", "&CPMD/BERENDSEN IONS"},
    "production": {"&CPMD/RESTART COORDINATES VELOCITIES WAVEFUNCTION", "&CPMD/ANNEALING IONS",
                   "&CPMD/TEMPERATURE", "&CPMD/NOSE IONS", "&CPMD/DIPOLE DYNAMICS"},
}


@pytest.mark.parametrize("stage", list(DOCUMENTED))
def test_each_dynamics_stage_is_the_annealing_input_plus_its_documented_changes(stage):
    """The archive ships no per-stage input, so the check is structural: a coupled stage
    keeps the annealing input's method — Born-Oppenheimer at 10 a.u. — and differs only
    where the tutorial says; every stage after annealing restarts from the one before."""
    _, anneal = compiled("anneal", "cpmd")
    ramp = {"&CPMD/TEMPERATURE RAMP": "1.8 300 20", "&CPMD/BERENDSEN IONS": "1.8 5000"}
    _, later = compiled(stage, "cpmd", **(ramp if stage == "heat" else {}))
    method = {k: v for k, v in anneal.values.items()
              if k.startswith(("&SYSTEM/", "&DFT/", "&MIMIC/"))}
    assert {k: later.values.get(k) for k in method} == method
    changed = set(later.values) ^ set(anneal.values)
    assert changed <= DOCUMENTED[stage]
    assert "&CPMD/MOLECULAR DYNAMICS BO" in later.values
    assert float(later.values["&CPMD/TIMESTEP"]) == 10.0
    assert any(k.startswith("&CPMD/RESTART") for k in later.values)
    assert not any(k.startswith("&CPMD/RESTART") for k in anneal.values)


def test_the_qm_charge_is_never_a_default_of_a_stage():
    """Chemistry is never a default: a stage's QM region states its charge, which the mandate's
    approval makes the researcher's."""
    from myrmidon.core.mandate import QMRegion

    assert QMRegion.model_fields["charge"].is_required()


@pytest.mark.parametrize("reference", [r for r in FILE_BACKED if r.endswith(".mdp")])
def test_tutorial_mdp_stages_pass_the_harness_mdp_audit(reference):
    stage, engine, system = FILE_BACKED[reference]
    _, resolution = compiled(stage, engine, **system)
    warnings, _ = validate_mdp(parse_mdp(render_mdp(resolution.values, resolution.origin)))
    assert not [w.id for w in warnings if w.severity == "block"]


def test_invariants_hold_on_the_protocols_own_values():
    """Each template's own rules, and the protocol's rule about two different steps."""
    protocol = ProtocolStore.builtin().get("mimic_qmmm_solvated_solute")
    seen, rendered = [], {}
    for step in protocol.steps:
        if not step.cites:
            continue
        stage, _, renders = step.cites.removeprefix("stage:").partition("/")
        ramp = ({"&CPMD/TEMPERATURE RAMP": "1.8 300 20", "&CPMD/BERENDSEN IONS": "1.8 5000"}
                if stage == "heat" else {})
        template, resolution = compiled(stage, "gromacs" if renders == "mdp" else "cpmd",
                                        **ramp)
        results = evaluate_invariants(template.invariants, {"": resolution.values})
        assert all(r.status == "satisfied" for r in results), [(r.id, r.detail) for r in results]
        seen += results
        rendered[step.name] = resolution.values
    across = protocol.check_invariants(rendered)
    assert across and all(r.status == "satisfied" for r in across)
    assert {r.id for r in seen} >= {"MIMIC_MDP_MUST_NOT_THERMOSTAT"}


# --- the stages rendered in a study ---------------------------------------------------


def _qm_study(ws_ctx, protocol: str, stages: list[dict] | None = None) -> dict:
    """The acetone system as a study's mandate, with the protocol's stages and methods;
    the files a render reads, by kind."""
    stage = ws_ctx.workspace.stage("input")
    for name in ("acetone_equilibrated.gro", "acetone_equilibrated.top"):
        shutil.copy(ACETONE / name, stage / name)
    shutil.copytree(ACETONE / "amber03.ff", stage / "amber03.ff")
    (stage / "index.ndx").write_text("[ System ]\n1 2 3 4\n[ QMatoms ]\n1 2 3\n")
    rel = ws_ctx.workspace.relpath
    files = {"structure": rel(stage / "acetone_equilibrated.gro"),
             "topology": rel(stage / "acetone_equilibrated.top"),
             "index": rel(stage / "index.ndx")}
    mandate = protocol_mandate(protocol)
    written = REGISTRY.call("mandate_write", ws_ctx, {
        "question": protocol, "protocol": protocol, "ceiling_core_h": 500.0,
        "system_files": [{"path": p, "kind": k} for k, p in files.items()],
        "subsystems": SUBSYSTEMS,
        "hamiltonian": mandate.system.hamiltonian.model_dump(exclude_none=True),
        "stages": stages or [s.model_dump(mode="json", exclude_none=True)
                             for s in mandate.method.stages]})
    assert written.ok, written.summary
    return files


def test_a_violated_invariant_blocks_the_render(ws_ctx):
    """Turning a thermostat on in the MiMiC mdp is a physics error the template knows."""
    files = _qm_study(ws_ctx, "mimic_qmmm_solvated_solute")
    refused = REGISTRY.call("inputset_apply", ws_ctx, {
        "stage": "anneal", **files, "output": "01_input/mimic.mdp",
        "overrides": {"tcoupl": "v-rescale"}})
    assert not refused.ok
    assert [w.id for w in blocking(refused)] == ["MIMIC_MDP_MUST_NOT_THERMOSTAT"]
    assert not (ws_ctx.workspace.root / "01_input/mimic.mdp").exists()


def test_rendering_a_cpmd_stage_writes_its_input_and_sidecar(ws_ctx):
    files = _qm_study(ws_ctx, "mimic_qmmm_solvated_solute")
    obs = REGISTRY.call("inputset_apply", ws_ctx, {
        "stage": "production", **files, "output": "01_input/production.inp"})
    assert obs.ok, obs.summary
    assert obs.renders == "cpmd_inp"
    sections = parse_cpmd((ws_ctx.workspace.root / "01_input/production.inp").read_text())
    assert set(sections) >= {"INFO", "CPMD", "SYSTEM", "DFT", "MIMIC"}
    # the provenance is echoed by CPMD itself, not hidden in a comment
    assert any("cpmd.dynamics" in line for line in sections["INFO"])
    # the method's input, not a runnable one: the system is prepqm's job
    assert "ATOMS" not in sections
    assert obs.origin["&SYSTEM/CHARGE"].startswith("stage:")
    assert obs.origin["&CPMD/TIMESTEP"].startswith("method:")


FM_FIXTURES = Path(__file__).parent.parent / "fixtures" / "acetone"


def test_the_force_matching_stage_reproduces_the_harness_fixtures():
    """A QM/MM single point is the harness's acetone reference pair: the two files that pair
    was run with, up to the method's cutoffs, which the classical rerun the fit subtracts
    takes as well."""
    _, cpmd = compiled("reference", "cpmd", protocol="fm_reference_singlepoint")
    rendered = _method_sections(render_cpmd_from_values(cpmd.values))
    expected = _method_sections((FM_FIXTURES / "fm_singlepoint_template.inp").read_text())
    rendered["SYSTEM"] -= {CHARGE}  # the fixture leaves it to prepqm
    assert rendered == expected

    _, mdp = compiled("reference", "gromacs", protocol="fm_reference_singlepoint")
    got = parse_mdp(render_mdp(mdp.values, mdp.origin))
    ref = parse_mdp_file(FM_FIXTURES / "mimic_fm.mdp")
    departures = {k: (ref.get(k), got.get(k)) for k in set(got) | set(ref)
                  if not _same_value(got.get(k), ref.get(k))}
    assert departures == {
        "rcoulomb": ("1.2", "1"), "rvdw": ("1.2", "1"),  # the Amber method's
        "nstvout": ("0", "1"),  # both engines write at the stage's one stride, velocities too
        "fourierspacing": ("0.16", None), "pme_order": ("4", None), "nstlist": ("10", None),
        "nstlog": ("100", None),  # the engine's defaults
        "constraint_algorithm": (None, "lincs"),  # inert under constraints = none
    }


def test_the_force_matching_pace_invariant_catches_a_mismatch(ws_ctx):
    """The rule the fixture states in a comment, enforced across two files: it relates
    values two *different* steps render, so it is procedure and lives on the protocol the
    mandate names."""
    files = _qm_study(ws_ctx, "fm_reference_singlepoint")
    ok = REGISTRY.call("inputset_apply", ws_ctx, {
        "stage": "reference", **files, "output": "01_input/sp.inp"})
    assert ok.ok, ok.summary
    # with only one file rendered the cross-file rule cannot be decided yet
    assert [i["status"] for i in ok.invariants if i["id"] == "FM_PACE_MATCHES_NSTXOUT"] == [
        "deferred"]

    refused = REGISTRY.call("inputset_apply", ws_ctx, {
        "stage": "reference", **files, "output": "01_input/mimic_fm.mdp",
        "overrides": {"nstxout": 10, "nstfout": 10}})
    assert not refused.ok
    assert [w.id for w in blocking(refused)] == ["FM_PACE_MATCHES_NSTXOUT"]

    obs = REGISTRY.call("inputset_apply", ws_ctx, {
        "stage": "reference", **files, "output": "01_input/mimic_fm.mdp"})
    assert obs.ok, obs.summary
    assert "FM_PACE_MATCHES_NSTXOUT" in {
        i["id"] for i in obs.invariants if i["status"] == "satisfied"}


def test_a_classical_single_point_refuses_the_mimic_integrator(ws_ctx):
    """A known trap as a check rather than a paragraph."""
    files = _qm_study(ws_ctx, "mimic_qmmm_solvated_solute",
                      stages=[{"name": "rerun", "kind": "single_point"}])
    refused = REGISTRY.call("inputset_apply", ws_ctx, {
        "stage": "rerun", **files, "output": "01_input/rerun.mdp",
        "overrides": {"integrator": "mimic"}})
    assert not refused.ok
    assert [w.id for w in blocking(refused)] == ["RERUN_MUST_NOT_USE_MIMIC_INTEGRATOR"]

    obs = REGISTRY.call("inputset_apply", ws_ctx, {
        "stage": "rerun", **files, "output": "01_input/rerun.mdp"})
    assert obs.ok, obs.summary
    assert parse_mdp((ws_ctx.workspace.root / "01_input/rerun.mdp").read_text())["nstfout"] == "1"


def test_the_surfaced_traps_land_in_the_ledger(ws_ctx):
    files = _qm_study(ws_ctx, "fm_reference_singlepoint")
    obs = REGISTRY.call("inputset_apply", ws_ctx, {
        "stage": "reference", **files, "output": "01_input/sp.inp"})
    assert obs.ok, obs.summary
    # the single point's own trap and the one every coupled stage inherits
    assert set(obs.pitfalls) == {"QM_ELEMENT_NOT_COVERED", "FM_RERUN_MDP_IS_MIMIC"}
    event = list(ws_ctx.workspace.ledger.events())[-1]
    assert event.tool == "inputset_apply"
    assert set(obs.pitfalls) <= set(event.knowledge_refs)


# (an override as a model gives it, the value the file, its sidecar and the ledger hold): a
# list of numbers is one engine line, its items separated by spaces, without float noise
OVERRIDDEN = [("3.4 300 0.02", "3.4 300 0.02"), ([3.4, 300, 0.0197733], "3.4 300 0.0197733"),
              ([3.4, 300.0, 1234567.0], "3.4 300 1234567")]


@pytest.mark.parametrize("given,held", OVERRIDDEN)
def test_an_override_is_held_as_the_engine_line_it_renders(given, held):
    _, resolution = compiled("heat", "cpmd", **{"&CPMD/TEMPERATURE RAMP": given,
                                                "&CPMD/BERENDSEN IONS": "3.4 5000"})
    assert resolution.values["&CPMD/TEMPERATURE RAMP"] == held
    assert render_cpmd_from_values(resolution.values).count(f"  {held}\n") == 1
