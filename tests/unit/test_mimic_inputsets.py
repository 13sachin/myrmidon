"""The MiMiC inputsets, checked against the tutorial inputs they were distilled from.

An input set is a transcription, and a transcription can be wrong. These tests render
each stage and compare it to the reference file in tests/fixtures/mimic_tutorial/, so
a typo in the YAML fails here instead of in a simulation.

Excluded from the comparison, because an input set supplies the method and never the
system: &MIMIC PATHS / BOX / OVERLAPS and the whole &ATOMS block, all written per
system by mimicpy's prepqm.
"""

from pathlib import Path

import pytest
import yaml

from myrmidon.core.tool import REGISTRY
from myrmidon.engines.gromacs.mdp import parse_mdp, parse_mdp_file, validate_mdp
from myrmidon.engines.mimic.cpmd_input import parse_cpmd, render_cpmd_from_values
from myrmidon.knowledge.inputsets import InputSetStore, evaluate_invariants, resolve

TUTORIAL = Path(__file__).parent.parent / "fixtures" / "mimic_tutorial"
PER_SYSTEM = {"PATHS", "BOX", "OVERLAPS"}  # prepqm writes these, an inputset cannot


def _keyword_lines(section_lines: list[str]) -> list[str]:
    """Keyword lines and their value lines, minus the per-system &MIMIC entries."""
    out, skip = [], False
    for raw in section_lines:
        line = raw.strip()
        if not line:
            continue
        if line.upper() in PER_SYSTEM:
            skip = True
            continue
        if skip:
            # the value line(s) belonging to a skipped keyword
            if line.upper().isupper() and not line[0].isdigit() and line[0] != "/":
                skip = False
            else:
                continue
        out.append(line)
    return out


def _method_sections(text: str) -> dict[str, list[str]]:
    sections = parse_cpmd(text)
    return {
        name: _keyword_lines(lines)
        for name, lines in sections.items()
        if name not in ("ATOMS", "INFO")
    }


#: stages whose rendering is checked against a file the MiMiC project publishes
FILE_BACKED = {
    "minimise": "mini.mdp",
    "heat": "heat.mdp",
    "mimic_mdp": "mimic.mdp",
    "cpmd_template": "template.inp",
}

#: the four dynamics stages are not shipped as files — the archive has the template and
#: the reader edits it, so the tutorial page prints each stage's &CPMD block. They are
#: checked as the template plus exactly the differences the page documents.
DYNAMICS_STAGES = ("annealing", "nve_test", "heating", "production")


#: each tutorial stage's input set, in the ``mimic_qmmm_solvated_solute`` protocol's order
STAGE_SETS = {
    "minimise": "em_solute_in_water", "heat": "npt_equilibration_solute_in_water",
    "mimic_mdp": "mimic_mm_mdp", "cpmd_template": "mimic_cpmd_template",
    "annealing": "mimic_cpmd_annealing", "nve_test": "mimic_cpmd_nve_test",
    "heating": "mimic_cpmd_heating", "production": "mimic_cpmd_production",
}

#: what the tutorial chose for its own system, which the shipped sets leave to the
#: researcher: the tutorial is the generic method plus these
TUTORIAL_SYSTEM = {
    "heat": {"nsteps": 100000},  # 200 ps of equilibration
    "production": {"&CPMD/MAXSTEP": 1000},  # about 2.4 ps of sampling
}


def acetone(stage: str):
    """The input set holding one tutorial stage's parameter set."""
    return InputSetStore.builtin().get(STAGE_SETS[stage])


def _stage_of(cites: str) -> str | None:
    set_id = cites.split("@")[0].removeprefix("inputset:")
    return next((stage for stage, sid in STAGE_SETS.items() if sid == set_id), None)


def tutorial(stage: str, **overrides):
    """One stage resolved as the tutorial ran it on its own system."""
    return resolve(acetone(stage), overrides={**TUTORIAL_SYSTEM.get(stage, {}), **overrides})


def shipped_protocol(name: str):
    from myrmidon.knowledge.protocols import ProtocolStore

    return ProtocolStore.builtin().get(name)


def _same_value(a: str, b: str) -> bool:
    """GROMACS reads 4.5e-5 and 4.5e-05, or 1. and 1.0, as the same number."""
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


@pytest.mark.parametrize("stage", list(FILE_BACKED))
def test_each_file_backed_stage_reproduces_the_published_input(stage):
    """Every keyword the published tutorial sets, the input set sets, with the same value.
    The published archive is the authority: version 1 was distilled from a 2022 copy that
    used Car-Parrinello dynamics, and this is what would have caught it."""
    from myrmidon.engines.gromacs.mdp import render_mdp

    reference = FILE_BACKED[stage]
    resolution = tutorial(stage)
    text = (TUTORIAL / reference).read_text()
    if reference.endswith(".mdp"):
        rendered = parse_mdp(render_mdp(resolution.values, resolution.origin))
        expected = parse_mdp(text)
        assert set(rendered) == set(expected)
        assert not {k: (rendered[k], expected[k]) for k in expected
                    if not _same_value(rendered[k], expected[k])}
    else:
        rendered = _method_sections(render_cpmd_from_values(resolution.values))
        expected = _method_sections(text)
        assert rendered == expected


#: what each dynamics stage may change in the template, as the tutorial page documents it
DOCUMENTED = {
    "annealing": {"&CPMD/MIMIC"},
    "nve_test": {
        "&CPMD/RESTART COORDINATES VELOCITIES WAVEFUNCTION", "&CPMD/MIMIC",
        "&CPMD/ANNEALING IONS", "&CPMD/TEMPERATURE", "&CPMD/STORE", "&CPMD/RESTFILE",
    },
    "heating": {
        "&CPMD/RESTART COORDINATES VELOCITIES WAVEFUNCTION", "&CPMD/MIMIC",
        "&CPMD/ANNEALING IONS", "&CPMD/TEMPERATURE", "&CPMD/TEMPERATURE RAMP",
        "&CPMD/BERENDSEN IONS", "&CPMD/STORE", "&CPMD/RESTFILE",
    },
    "production": {
        "&CPMD/RESTART COORDINATES VELOCITIES WAVEFUNCTION GEOFILE", "&CPMD/MIMIC",
        "&CPMD/ANNEALING IONS", "&CPMD/TEMPERATURE", "&CPMD/NOSE IONS",
        "&CPMD/DIPOLE DYNAMICS",
    },
}


@pytest.mark.parametrize("stage", DYNAMICS_STAGES)
def test_each_dynamics_stage_is_the_template_plus_its_documented_changes(stage):
    """The archive ships no per-stage input, so the check is structural: a dynamics stage
    keeps the template's method — Born-Oppenheimer at 10 a.u., the mistake version 1 made
    — and differs only where the tutorial says. Annealing is the first coupled run and has
    nothing to restart from; every later stage restarts (CPMD fails on a RESTART with no
    file)."""
    template = tutorial("cpmd_template").values
    values = tutorial(stage, **{"&SYSTEM/CHARGE": "0.0"}).values
    method = {k: v for k, v in template.items()
              if k.startswith(("&SYSTEM/", "&DFT/", "&MIMIC/")) and k != "&SYSTEM/CHARGE"}
    assert {k: values.get(k) for k in method} == method
    added = set(values) - set(template) - {"&SYSTEM/CHARGE"}
    removed = set(template) - set(values)
    assert not (added | removed) - DOCUMENTED[stage]
    assert "&CPMD/MOLECULAR DYNAMICS BO" in values
    assert str(values["&CPMD/TIMESTEP"]) == "10.0"
    restarts = any(k.startswith("&CPMD/RESTART") for k in values)
    assert restarts is (stage != "annealing")


def test_the_tutorial_protocol_will_not_supply_the_qm_charge():
    """Chemistry is never a default, even when the tutorial's answer is zero. A stage's input
    set is its method and says nothing of the charge, which belongs to the system and is
    confirmed by the researcher in mimic_prepare (RESEARCHER_FIELDS)."""
    from myrmidon.engines.mimic.cpmd_input import MimicPrepareParams

    assert {"total_charge", "multiplicity"} <= set(MimicPrepareParams.RESEARCHER_FIELDS)
    for stage in DYNAMICS_STAGES:
        assert "&SYSTEM/CHARGE" not in acetone(stage).parameters, stage
        assert "&SYSTEM/CHARGE" not in resolve(acetone(stage)).values


def test_tutorial_mdp_stages_pass_the_harness_mdp_audit():
    from myrmidon.engines.gromacs.mdp import render_mdp

    for stage in (s for s in FILE_BACKED if s != "cpmd_template"):
        resolution = tutorial(stage)
        text = render_mdp(resolution.values, resolution.origin)
        warnings, _ = validate_mdp(parse_mdp(text))
        blocking = [w.id for w in warnings if w.severity == "block"]
        # the MiMiC mdp needs an index group that only exists once prepqm has run
        allowed = {"MDP_MIMIC_NO_QMMM_GRPS"} if stage == "mimic_mdp" else set()
        assert not (set(blocking) - allowed), f"{stage}: {blocking}"


def test_invariants_hold_on_the_protocols_own_values():
    """Each input set's own rules, and the plan's rule about two different steps."""
    seen = []
    for stage in list(FILE_BACKED) + list(DYNAMICS_STAGES):
        inputset = acetone(stage)
        if not inputset.invariants:
            continue
        results = evaluate_invariants(inputset.invariants, {"": tutorial(stage).values})
        seen += results
        assert all(r.status == "satisfied" for r in results), [
            (r.id, r.status, r.detail) for r in results
        ]

    plan = shipped_protocol("mimic_qmmm_solvated_solute")
    rendered = {
        step.name: resolve(
            acetone(stage) if (stage := _stage_of(step.cites)) else
            InputSetStore.builtin().get(step.cites.split("@")[0].removeprefix("inputset:")),
            overrides={"&SYSTEM/CHARGE": "0.0", **TUTORIAL_SYSTEM.get(stage or "", {})},
        ).values
        for step in plan.steps
        if step.cites
    }
    plan_results = plan.check_invariants(rendered)
    seen += plan_results
    assert plan_results and all(r.status == "satisfied" for r in plan_results), [
        (r.id, r.status, r.detail) for r in plan_results
    ]
    assert seen, "the acetone inputsets and their plan state no invariants at all"


def test_a_violated_invariant_blocks_the_render(ws_ctx):
    """Turning a thermostat on in the MiMiC mdp is a physics error the input set knows."""
    refused = REGISTRY.call(
        "inputset_apply",
        ws_ctx,
        {
            "inputset": "mimic_mm_mdp",
            "output": "01_input/mimic.mdp",
            "overrides": {"tcoupl": "v-rescale"},
            "acknowledge_warnings": ["INPUTSET_VALUE_OVERRIDDEN"],
        },
    )
    assert not refused.ok
    assert [w.id for w in refused.blocking] == ["MIMIC_MDP_MUST_NOT_THERMOSTAT"]
    assert not (ws_ctx.workspace.root / "01_input/mimic.mdp").exists()


def test_rendering_a_cpmd_stage_writes_a_template_and_its_sidecar(ws_ctx):
    obs = REGISTRY.call(
        "inputset_apply",
        ws_ctx,
        {
            "inputset": "mimic_cpmd_production",
            "output": "01_input/production.inp",
            "overrides": {"&SYSTEM/CHARGE": "0.0", "&CPMD/MAXSTEP": 1000},
        },
    )
    assert obs.ok, obs.summary
    assert obs.renders == "cpmd_inp"
    text = (ws_ctx.workspace.root / "01_input/production.inp").read_text()
    sections = parse_cpmd(text)
    assert set(sections) >= {"INFO", "CPMD", "SYSTEM", "DFT", "MIMIC"}
    # the provenance is echoed by CPMD itself, not hidden in a comment
    assert any("mimic_cpmd_production" in line for line in sections["INFO"])
    # a template, not a runnable input: the system is prepqm's job
    assert "ATOMS" not in sections
    assert obs.origin["&SYSTEM/CHARGE"] == "user"
    assert obs.origin["&CPMD/TIMESTEP"].startswith("inputset:")


def test_charge_must_be_supplied_before_a_cpmd_template_renders(ws_ctx):
    refused = REGISTRY.call(
        "inputset_apply",
        ws_ctx,
        {
            "inputset": "fm_singlepoint_cpmd",
            "output": "01_input/singlepoint.inp",
        },
    )
    assert not refused.ok
    finding = next(
        w for w in refused.blocking if w.id == "INPUTSET_REQUIRED_PARAM_MISSING"
    )
    assert "CHARGE" in finding.message
    assert "sum of the topology charges" in finding.message


FM_FIXTURES = Path(__file__).parent.parent / "fixtures" / "acetone"


def test_fm_protocols_match_the_harness_fixtures():
    """The fm_singlepoint inputsets are the harness's own acetone reference pair, so they must
    reproduce the two files that pair was actually run with."""
    resolution = resolve(
        InputSetStore.builtin().get("fm_singlepoint_cpmd"),
        overrides={"&SYSTEM/CHARGE": "0.0"},
    )
    rendered = _method_sections(render_cpmd_from_values(resolution.values))
    expected = _method_sections((FM_FIXTURES / "fm_singlepoint_template.inp").read_text())
    for section in expected:  # the fixture has no CHARGE; the input set demands one
        assert [ln for ln in rendered[section] if not ln.startswith(("CHARGE", "0.0"))] == (
            expected[section]
        ), section

    from myrmidon.engines.gromacs.mdp import render_mdp

    mdp = resolve(InputSetStore.builtin().get("fm_singlepoint_mm"))
    got = parse_mdp(render_mdp(mdp.values, mdp.origin))
    ref = parse_mdp_file(FM_FIXTURES / "mimic_fm.mdp")
    assert set(got) == set(ref)
    assert not {k: (got[k], ref[k]) for k in ref if not _same_value(got[k], ref[k])}


def test_the_force_matching_pace_invariant_catches_a_mismatch(ws_ctx):
    """The rule the fixture states in a comment, enforced across two files.

    The rule relates values that two *different* steps render, so it is procedure and
    lives on the plan rather than on either inputset. A study without a plan is a study
    that never said the two files belong to one run, and the rule cannot be checked —
    which is why the plan is written into the workspace first.
    """
    protocol = shipped_protocol("fm_reference_singlepoint")
    (Path(ws_ctx.workspace.root) / "plan.yaml").write_text(
        yaml.safe_dump({
            "question": "reference forces for the fit",
            "steps": [s.model_dump(exclude_none=True) for s in protocol.steps],
            "given": protocol.given,
            "invariants": [i.model_dump(exclude_none=True) for i in protocol.invariants],
        })
    )

    ok = REGISTRY.call(
        "inputset_apply",
        ws_ctx,
        {
            "inputset": "fm_singlepoint_cpmd",
            "output": "01_input/sp.inp",
            "overrides": {"&SYSTEM/CHARGE": "0.0"},
        },
    )
    assert ok.ok, ok.summary
    # with only one stage rendered the cross-file rule cannot be decided yet
    assert [i["status"] for i in ok.invariants if i["id"] == "FM_PACE_MATCHES_NSTXOUT"] == [
        "deferred"
    ]

    refused = REGISTRY.call(
        "inputset_apply",
        ws_ctx,
        {
            "inputset": "fm_singlepoint_mm",
            "output": "01_input/mimic_fm.mdp",
            "overrides": {"nstxout": 10, "nstfout": 10},
            "acknowledge_warnings": ["INPUTSET_VALUE_OVERRIDDEN"],
        },
    )
    assert not refused.ok
    assert [w.id for w in refused.blocking] == ["FM_PACE_MATCHES_NSTXOUT"]

    obs = REGISTRY.call(
        "inputset_apply",
        ws_ctx,
        {
            "inputset": "fm_singlepoint_mm",
            "output": "01_input/mimic_fm.mdp",
        },
    )
    assert obs.ok, obs.summary
    satisfied = {i["id"] for i in obs.invariants if i["status"] == "satisfied"}
    assert satisfied == {"FM_PACE_MATCHES_NSTXOUT", "FM_MDP_MUST_WRITE_FORCES"}


def test_the_rerun_protocol_refuses_the_mimic_integrator(ws_ctx):
    """A known trap as a check rather than a paragraph."""
    refused = REGISTRY.call(
        "inputset_apply",
        ws_ctx,
        {
            "inputset": "fm_classical_rerun",
            "output": "01_input/rerun.mdp",
            "overrides": {"integrator": "mimic"},
            "acknowledge_warnings": ["INPUTSET_VALUE_OVERRIDDEN"],
        },
    )
    assert not refused.ok
    finding = next(w for w in refused.blocking if w.id == "RERUN_MUST_NOT_USE_MIMIC_INTEGRATOR")
    assert "FM_RERUN_MDP_IS_MIMIC" in finding.message

    obs = REGISTRY.call(
        "inputset_apply", ws_ctx, {"inputset": "fm_classical_rerun", "output": "01_input/rerun.mdp"}
    )
    assert obs.ok, obs.summary
    assert parse_mdp((ws_ctx.workspace.root / "01_input/rerun.mdp").read_text())["nstfout"] == "1"


def test_the_surfaced_traps_land_in_the_ledger(ws_ctx):
    obs = REGISTRY.call(
        "inputset_apply",
        ws_ctx,
        {
            "inputset": "fm_singlepoint_cpmd",
            "output": "01_input/sp.inp",
            "overrides": {"&SYSTEM/CHARGE": "0.0"},
        },
    )
    assert obs.ok, obs.summary
    assert set(obs.pitfalls) == {"CPMD_MULTIPLICITY_WITHOUT_LSD", "FM_RERUN_MDP_IS_MIMIC"}

    event = list(ws_ctx.workspace.ledger.events())[-1]
    assert event.tool == "inputset_apply"
    assert {"CPMD_MULTIPLICITY_WITHOUT_LSD", "FM_RERUN_MDP_IS_MIMIC"} <= set(event.knowledge_refs)

