"""MiMiC QM/MM tools against mock mimicpy/cpmd.x/mpirun binaries."""

import time
from pathlib import Path

import pytest
from tests.unit.findings_catalog import blocking

from myrmidon.core.tool import REGISTRY
from myrmidon.engines.mimic.cpmd_input import get_keyword, parse_cpmd, render_cpmd, set_keyword

CPMD_SAMPLE = """\
&CPMD
MIMIC
MOLECULAR DYNAMICS BO
&END

&MIMIC
PATHS
1
/tmp/run
BOX
  30.0 30.0 30.0
&END

&SYSTEM
CHARGE
  0
&END
"""


ACETONE_GRO = Path(__file__).parents[1] / "fixtures" / "acetone" / "acetone_stageA_entry.gro"

def test_cpmd_parse_render_roundtrip():
    sections = parse_cpmd(CPMD_SAMPLE)
    assert set(sections) == {"CPMD", "MIMIC", "SYSTEM"}
    assert get_keyword(sections["SYSTEM"], "CHARGE") == "0"
    assert get_keyword(sections["MIMIC"], "PATHS") == "1"
    assert get_keyword(sections["CPMD"], "MAXSTEP") is None

    sections["SYSTEM"] = set_keyword(sections["SYSTEM"], "CHARGE", "-1")
    sections["SYSTEM"] = set_keyword(sections["SYSTEM"], "MULTIPLICITY", "2")
    text = render_cpmd(sections)
    again = parse_cpmd(text)
    assert get_keyword(again["SYSTEM"], "CHARGE") == "-1"
    assert get_keyword(again["SYSTEM"], "MULTIPLICITY") == "2"


def _select_qm(ws_ctx):
    ws = ws_ctx.workspace
    stage = ws.stage("prep")
    for name in ("acetone_stageA_entry.top", "acetone.itp"):  # prepqm sums its charges
        (stage / name.replace("acetone_stageA_entry", "topol")).write_text(
            (ACETONE_GRO.parent / name).read_text())
    # a real acetone: the mock prepqm selects from the coordinates it is given
    (stage / "conf.gro").write_text(ACETONE_GRO.read_text())
    return REGISTRY.call(
        "qm_region",
        ws_ctx,
        {
            "topology": ws.relpath(stage / "topol.top"),
            "coords": ws.relpath(stage / "conf.gro"),
            "selections": ["add resname is ACT"],
            "padding_nm": 0.4,
        },
    )


def test_run_gate_blocks_an_input_with_no_settled_spin(ws_ctx):
    """What the selection wrote is the system, not a stage's input: nothing settled its spin."""
    from myrmidon.engines.mimic.collect import coupled_findings

    select = _select_qm(ws_ctx)
    cpmd_inp = next(a.path for a in select.artifacts if a.role == "cpmd_input")
    stage = ws_ctx.workspace.stage("qmmm")
    (stage / "mimic.tpr").write_text("MOCKTPR\n")
    found, _ = coupled_findings(ws_ctx, cpmd_inp, ws_ctx.workspace.relpath(stage / "mimic.tpr"),
                                None, None)
    assert "QM_MULTIPLICITY_UNCONFIRMED" in {w.id for w in found if w.severity == "block"}


def _coupled_stage(ws_ctx, name="run", **over):
    """A prepared coupled stage under an approved mandate, of 5 steps."""
    from tests.conftest import REGION, coupled_study

    coupled_study(ws_ctx, [{"name": name, "kind": "dynamics", "regime": "qmmm",
                            "ensemble": "NVE", "steps": 5, "qm_region": REGION, **over}])
    prepared = REGISTRY.call("stage_prepare", ws_ctx, {"stage": name})
    assert prepared.ok, prepared.summary
    return name


def _wait(ctx, job_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = REGISTRY.call("job_status", ctx, {"job_id": job_id})
        if st.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
        time.sleep(0.1)
    raise AssertionError("mimic job never finished")


TIGHT_CELL_INPUT = """\
&MIMIC
PATHS
1
/tmp/x
BOX
 30 30 30
&END
&CPMD
MIMIC
MAXSTEP
 5
&END
&SYSTEM
CELL
 5.3 1.5 0.8 0.0 0.0 0.0
CHARGE
 0
&END
&ATOMS
*C_MT_BLYP KLEINMAN-BYLANDER
 LMAX=P
 2
 20.98 19.77 21.52
 26.32 27.53 25.79
&END
"""


# (total charge, multiplicity, MULTIPLICITY written, LSD written). CPMD accepts
# MULTIPLICITY only in a spin-polarised run, so a closed shell omits it — and the confirmed
# value still travels in the sidecar the run gate reads.
SPIN = [(-1, 1, None, False), (0, 2, "2", True)]


@pytest.mark.parametrize("charge,multiplicity,written,lsd", SPIN)
def test_a_stage_writes_the_charge_and_the_spin_cpmd_will_accept(ws_ctx, charge, multiplicity,
                                                                 written, lsd):
    import json

    from tests.conftest import coupled_study

    coupled_study(ws_ctx, [{"name": "run", "kind": "dynamics", "regime": "qmmm",
                            "ensemble": "NVE", "steps": 50, "qm_region": {
                                "subsystems": ["solute"], "charge": charge,
                                "multiplicity": multiplicity}}])
    obs = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "run"})
    assert obs.ok, obs.summary
    out = obs.files["cpmd_input"]
    sections = parse_cpmd((ws_ctx.workspace.root / out).read_text())
    assert get_keyword(sections["SYSTEM"], "CHARGE") == str(charge)
    assert get_keyword(sections["SYSTEM"], "MULTIPLICITY") == written
    assert any(line.strip() == "LSD" for line in sections["CPMD"]) is lsd
    assert get_keyword(sections["CPMD"], "MAXSTEP") == "50"
    meta = json.loads((ws_ctx.workspace.root / (out + ".myrmidon.json")).read_text())
    assert (meta["charge"], meta["multiplicity"]) == (charge, multiplicity)


def test_the_qm_cell_is_read_in_every_unit_cpmd_takes():
    from myrmidon.engines.mimic.cpmd_input import qm_geometry

    geom = qm_geometry(parse_cpmd(TIGHT_CELL_INPUT))
    assert geom["n_atoms"] == 2
    assert geom["cell_bohr"] == [5.3, 7.95, 4.24]  # relative to the first edge
    assert geom["min_vacuum_bohr"] < 0.1
    absolute = TIGHT_CELL_INPUT.replace("CELL\n 5.3 1.5 0.8", "CELL ABSOLUTE\n 20.0 20.0 20.0")
    assert qm_geometry(parse_cpmd(absolute))["cell_bohr"] == [20.0, 20.0, 20.0]
    angstrom = qm_geometry(parse_cpmd(absolute.replace("&SYSTEM\n", "&SYSTEM\nANGSTROM\n")))
    assert angstrom["cell_bohr"][0] == pytest.approx(37.79, abs=0.01)


COUPLED = ("&MIMIC\nPATHS\n1\n/tmp/x\nBOX\n 30 30 30\n&END\n"
           "&CPMD\nMIMIC\nMAXSTEP\n 5\n&END\n&SYSTEM\nCHARGE\n 0\n&END\n")

# (a CPMD input as written, the check that ran, the finding it raises, the group it lands
# in). What CPMD's own reader refuses (MULTIPLICITY without LSD) is CPMD's to say.
CPMD_GATES = [
    # QM atoms with no vacuum around them segfault in the short-range coupling, silently
    (TIGHT_CELL_INPUT, "qm_cell", "QM_CELL_NO_VACUUM", "blocking"),
    # a keyword CPMD does not read is run without; one no owner decides is outside the checks
    (COUPLED.replace("MAXSTEP", "MAXSTPE"), "keywords", "CPMD_KEYWORD_UNKNOWN", "advisory"),
    (COUPLED.replace("&SYSTEM\n", "&SYSTEM\nSTRESS TENSOR\n 1\n"), "keywords",
     "CPMD_KEYWORD_OUT_OF_SCOPE", "advisory"),
    # sulfur under a library of light elements: CPMD would stop with ECPFILE NOT FOUND
    (COUPLED + "&ATOMS\n*S_MT_BLYP.psp KLEINMAN-BYLANDER\n LMAX=P\n 1\n 0.0 0.0 0.0\n&END\n",
     "pseudopotentials", "QM_ELEMENT_NOT_COVERED", "blocking"),
]


@pytest.mark.parametrize("text,check,finding,group", CPMD_GATES)
def test_a_cpmd_input_cpmd_would_reject_is_blocked_before_it_runs(ws_ctx, monkeypatch, text,
                                                                  check, finding, group):
    monkeypatch.setenv("PP_LIBRARY_PATH", str(ACETONE_GRO.parents[1] / "cpmd" / "pp_library"))
    stage = ws_ctx.workspace.stage("qmmm")
    (stage / "run.inp").write_text(text)
    (stage / "run.inp.myrmidon.json").write_text('{"multiplicity": 1}')
    report = REGISTRY.call("file_check", ws_ctx,
                           {"path": ws_ctx.workspace.relpath(stage / "run.inp")})
    assert report.file_kind == "cpmd"
    assert finding in {w.id for w in report.findings}
    assert report.ok is (group != "blocking")
    del check  # which checks ran is not reported


# --- segment chains: a continuation restarts from its predecessor -------------------


def _select(ws_ctx, selections):
    ws = ws_ctx.workspace
    stage = ws.stage("prep")
    for name in ("acetone_stageA_entry.top", "acetone.itp"):  # prepqm sums its charges
        (stage / name.replace("acetone_stageA_entry", "topol")).write_text(
            (ACETONE_GRO.parent / name).read_text())
    (stage / "conf.gro").write_text(ACETONE_GRO.read_text())
    return REGISTRY.call("qm_region", ws_ctx, {
        "topology": ws.relpath(stage / "topol.top"), "coords": ws.relpath(stage / "conf.gro"),
        "selections": selections, "padding_nm": 0.4})


PREPQM = Path(__file__).parents[1] / "fixtures" / "mimicpy"


def test_skipped_selection_commands_are_read_from_real_prepqm():
    """prepqm prints an error for a command it cannot apply, skips it and exits 0 with the
    rest; the commands it skipped are the ones the region is missing."""
    from myrmidon.engines.mimic.qm_region import rejected_commands

    commands = (PREPQM / "prepqm_rejected_commands.in").read_text().splitlines()[:-1]
    real = rejected_commands((PREPQM / "prepqm_rejected_commands.out").read_text(), commands)
    assert [cmd for cmd, _ in real] == [
        "rm name is H1", "add resname is LIG", "add resname ACT", "add foo is 3"]
    assert rejected_commands(
        (PREPQM / "prepqm_empty_region.out").read_text(), ["add resname is LIG"]
    )[0][0] == "add resname is LIG"


# (selection commands, finding id the call is refused or failed on). The mock skips what
# real prepqm skips, with its messages, so the tool reads both the same way.
SELECTIONS = [
    (["add resname is ACT", "rm name is H1"], "QM_SELECTION_COMMAND_UNKNOWN"),
    (["add resname is ACT", "view"], "QM_SELECTION_COMMAND_UNKNOWN"),
    (["clear resname is ACT"], "QM_SELECTION_COMMAND_UNKNOWN"),
    (["add"], "QM_SELECTION_COMMAND_UNKNOWN"),
    (["add resname is ACT", "add resname is LIG"], "QM_SELECTION_REJECTED"),
    (["add resname ACT"], "QM_SELECTION_REJECTED"),
    (["add foo is 3", "add resname is ACT"], "QM_SELECTION_REJECTED"),
    (["add within 0.3 of resname ACT"], "QM_SELECTION_REJECTED"),  # the mock lacks `within`
    (["add resname is LIG"], "QM_SELECTION_REJECTED"),  # nothing matched, nothing written
]


@pytest.mark.parametrize(("selections", "finding"), SELECTIONS)
def test_a_selection_prepqm_would_skip_is_not_a_qm_region(ws_ctx, selections, finding):
    obs = _select(ws_ctx, selections)
    assert not obs.ok
    assert finding in {w.id for w in [*obs.findings, *blocking(obs)]}


# (selection, the charge prepqm writes, the MM charge sum it says it rounded): a whole
# neutral solute sums to an integer; with one atom deleted it does not
CHARGE_ROUNDING = [
    (["add resname is ACT"], 0, None),
    (["add resname is ACT", "delete name is O1"], 1, 0.5864),
]


@pytest.mark.parametrize(("selections", "charge", "rounded_from"), CHARGE_ROUNDING)
def test_the_charge_prepqm_rounded_is_reported_with_its_sum(ws_ctx, selections, charge,
                                                            rounded_from):
    obs = _select(ws_ctx, selections)
    assert obs.ok
    assert (obs.declared_charge, obs.rounded_from) == (charge, rounded_from)
    assert ("QM_CHARGE_ROUNDED" in {w.id for w in obs.findings}) == (rounded_from is not None)


# (topology, coordinates, the commands of a real prepqm capture in tests/fixtures/mimicpy):
# a solute with one atom deleted, and one residue's side chain cut from a protein. Both
# regions' charges sum to a non-integer, which prepqm rounds into &SYSTEM CHARGE.
PREPQM_CAPTURES = [
    ("acetone/acetone_equilibrated.top", "acetone/acetone_equilibrated.gro",
     "prepqm_rejected_commands"),
    ("mimicpy/trpcage_atoms.top", "gromacs/trpcage_water.gro", "prepqm_lys_side_chain"),
]


@pytest.mark.parametrize("top,coords,capture", PREPQM_CAPTURES)
def test_the_mock_prepqm_selects_and_charges_as_real_prepqm_does(tmp_path, top, coords,
                                                                   capture):
    import subprocess

    from myrmidon.engines.gromacs.index import ndx_groups
    from myrmidon.engines.mimic.qm_region import rejected_commands

    fixtures = Path(__file__).parents[1] / "fixtures"
    stdin = (PREPQM / f"{capture}.in").read_text()
    commands = stdin.splitlines()[:-1]
    mock = Path(__file__).parents[2] / "src/myrmidon/engines/mock/bin/mimicpy"
    proc = subprocess.run(
        [str(mock), "prepqm", "-top", str(fixtures / top), "-coords", str(fixtures / coords),
         "-ff", str(fixtures / "acetone"), "-out", "cpmd.inp", "-ndx", "index.ndx"],
        cwd=tmp_path, input=stdin, capture_output=True, text=True, check=True)
    if (PREPQM / f"{capture}.out").exists():
        real = (PREPQM / f"{capture}.out").read_text()
        assert rejected_commands(proc.stdout, commands) == rejected_commands(real, commands)
    said = [ln for ln in proc.stderr.splitlines() if ln.startswith("Total charge")]
    assert said == (PREPQM / f"{capture}.err").read_text().splitlines()
    system = parse_cpmd((PREPQM / f"{capture}.system").read_text())["SYSTEM"]
    written = parse_cpmd((tmp_path / "cpmd.inp").read_text())["SYSTEM"]
    assert get_keyword(written, "CHARGE") == get_keyword(system, "CHARGE")
    real_ndx = ndx_groups((PREPQM / f"{capture}.ndx").read_text())
    assert [m for _, m in ndx_groups((tmp_path / "index.ndx").read_text())] == [
        m for _, m in real_ndx]


WHOLE, CUT = ["add resname is ACT"], ["add resname is ACT", "delete name is O1"]

# (the selection, the charge the mock's topology rounds it to, the multiplicity the stage
# states, what the selection leaves decided, and what is noticed). The selected atoms' topology
# charges decide the charge when they sum to an integer; the electron count decides the
# multiplicity when it is even and no open-shell metal is in the region (acetone: 32 electrons).
# A value the researcher stated is taken as given, and one its electrons cannot have is said.
SPIN_STATE = [
    (WHOLE, 0, 1, 1, None),
    (WHOLE, 0, 2, 1, "QM_MULTIPLICITY_PARITY"),
    (CUT, 1, None, None, None),  # prepqm rounded a non-integer sum: neither is decided
]


@pytest.mark.parametrize("selections,declared,stated,derived,noticed", SPIN_STATE)
def test_a_qm_charge_and_multiplicity_are_derived_or_left_open(ws_ctx, selections, declared,
                                                               stated, derived, noticed):
    from tests.conftest import coupled_study

    obs = _select(ws_ctx, selections)
    assert obs.ok, obs.summary
    assert (obs.declared_charge, obs.derived_multiplicity) == (declared, derived)
    if noticed is None:
        return
    coupled_study(ws_ctx, [{"name": "run", "kind": "dynamics", "regime": "qmmm",
                            "ensemble": "NVE", "steps": 5, "qm_region": {
                                "subsystems": ["solute"], "charge": declared,
                                "multiplicity": stated}}])
    prepared = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "run"})
    assert prepared.ok and noticed in {f.id for f in prepared.findings}, prepared.summary


# (the &ATOMS block, the atoms read). A species whose element or count cannot be read, or
# counts that do not add up to the coordinates, leave the count unread rather than guessed.
ATOMS = [
    ("*CL_MT_BLYP.psp KLEINMAN-BYLANDER\n LMAX=P\n 1\n 0.0 0.0 0.0\n", {"Cl": 1}),
    ("*h_MT_BLYP.psp\n LMAX=S\n 2 # two\n 1.0 0.0 0.0\n 2.0 0.0 0.0\n", {"H": 2}),
    ("*Xx_MT.psp\n LMAX=S\n 1\n 0.0 0.0 0.0\n", None),
    ("*C_MT_BLYP.psp\n LMAX=P\n 2\n 0.0 0.0 0.0\n", None),
    ("*C_MT_BLYP.psp\n LMAX=P\n two\n 0.0 0.0 0.0\n", None),
]


@pytest.mark.parametrize("block,read", ATOMS)
def test_the_qm_atoms_are_read_atom_by_atom_or_not_at_all(block, read):
    from myrmidon.engines.mimic.cpmd_input import parse_cpmd
    from myrmidon.engines.mimic.system import qm_elements

    assert qm_elements(parse_cpmd(f"&ATOMS\n{block}&END\n")) == read


# A stage's run input is the stage's method on the system the QM selection made once: what a
# staged MiMiC run (anneal, NVE test, heat, produce) changes between stages is the method and
# never the system. Each row is a stage of a real staged run: its input made from the annealing
# system input and the stage's method must be the input it ran with.
STAGES = Path(__file__).parents[1] / "fixtures" / "mimic_stages"


@pytest.mark.parametrize("stage", ["nve_test", "heating", "production"])
def test_a_stage_input_is_its_method_on_the_system_made_once(stage):
    from myrmidon.engines.mimic.cpmd_input import apply_method, render_cpmd, set_spin

    made = set_spin(apply_method(parse_cpmd((STAGES / "system_from_prepqm.inp").read_text()),
                                 parse_cpmd((STAGES / f"{stage}_method.inp").read_text())), 0, 1)

    def lines(text):
        return {k: [x.strip() for x in v if x.strip()] for k, v in parse_cpmd(text).items()}

    made, real = lines(render_cpmd(made)), lines((STAGES / f"{stage}_run_real.inp").read_text())
    assert list(made) == list(real)
    assert made == real


def test_a_restarted_segment_hashes_the_restart_it_stages(ws_ctx):
    """What the node reads of a continued stage includes the restart staged into its directory,
    found in the ledger: an edit to the staged copy while the job waits is seen at collection."""
    from tests.conftest import REGION, coupled_study

    from myrmidon.execution.jobs import existing_jobstore

    region = {"regime": "qmmm", "ensemble": "NVE", "steps": 5, "qm_region": REGION}
    coupled_study(ws_ctx, [{"name": "first", "kind": "dynamics", **region},
                           {"name": "second", "kind": "dynamics", **region}])
    for name in ("first", "second"):
        assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": name}).ok
    first = REGISTRY.call("stage_run", ws_ctx, {"stage": "first"})
    assert first.ok, first.summary
    _wait(ws_ctx, first.job_id)
    second = REGISTRY.call("stage_run", ws_ctx, {"stage": "second"})
    assert second.ok, second.summary
    spec = existing_jobstore(ws_ctx.workspace).get(second.job_id).spec
    assert any(p.endswith("RESTART") for p in spec.receipt_inputs), spec.receipt_inputs


def test_a_systems_files_are_read_by_the_engine_each_belongs_to(tmp_path):
    """The facts of a system's files are the union of what each file's engine reads from it."""
    from myrmidon.engines.facts import system_facts

    top = tmp_path / "system.top"
    top.write_text('#include "amber99sb-ildn.ff/forcefield.itp"\n[ system ]\nx\n[ molecules ]\n')
    inp = tmp_path / "cpmd.inp"
    inp.write_text("&SYSTEM\n CELL ABSOLUTE\n  20.0 21.0 22.0 0 0 0\n CUTOFF\n  70.0\n"
                   " CHARGE\n  0\n&END\n\n&MIMIC\n OVERLAPS\n  1\n  2 1 1 1\n&END\n\n"
                   "&ATOMS\n*C_MT_BLYP.psp\n LMAX=P\n 1\n 1.0 2.0 3.0\n&END\n")
    facts = system_facts({"topology": top, "cpmd_input": inp})
    assert facts["forcefield"].value == "amber99sb-ildn"
    assert facts["qm_atoms"].value == "1"
    assert facts["cell_bohr"].value == "20.0 21.0 22.0"
    assert facts["cutoff_ry"].value == "70.0"
    assert facts["&SYSTEM/CHARGE"].value == "0"
    assert "1.0 2.0 3.0" in facts["&ATOMS"].value
    assert set(system_facts({"cpmd_input": inp}, rules={"qm_size"})) == {
        "qm_atoms", "cell_bohr", "cutoff_ry"}


def test_a_selection_says_what_it_leaves_of_the_regions_chemistry(ws_ctx):
    """One call writes the input, the index, the selection and a sidecar with the charge,
    multiplicity, electrons and guessed elements; a heavy atom left at LMAX=S is flagged."""
    import json

    from myrmidon.engines.mimic.qm_region import guessed_elements

    obs = _select(ws_ctx, ["add resname is ACT"])
    assert obs.ok
    assert (obs.declared_charge, obs.derived_multiplicity, obs.electrons) == (0, 1, 24)
    assert {a.role for a in obs.artifacts} >= {"cpmd_input", "index", "qm_selection"}
    root = ws_ctx.workspace.root
    selection = next(a.path for a in obs.artifacts if a.role == "qm_selection")
    assert (root / selection).read_text() == "add resname is ACT\n"
    sidecar = next(a.path for a in obs.artifacts if a.role == "provenance")
    written = json.loads((root / sidecar).read_text())
    assert (written["charge"], written["multiplicity"], written["electrons"]) == (0, 1, 24)
    assert written["guessed_elements"] == obs.guessed_elements
    assert "PP_LMAX_ALL_S_ON_HEAVY_ATOMS" in {f.id for f in obs.findings}
    # the mock's table reads as the real one does, type by type
    real = guessed_elements((PREPQM / "prepqm_acetone.err").read_text())
    assert real == {"c": "C", "c3": "C", "o": "O", "hc": "H", "OW_spc": "O", "HW_spc": "H"}
    assert {t: e for t, e in obs.guessed_elements.items() if t in real} == {
        t: real[t] for t in ("c", "c3", "o", "hc")}


# (selections, charge) -> (multiplicity, valence electrons): a charge the topology sums to,
# an odd count left open, boundary atoms whose pseudopotentials decide it
CHEMISTRY = [
    (["add resname is ACT"], 0, 1, 24),
    (["add resname is ACT"], 1, None, 23),
    (["add-bound resname is ACT"], 0, None, None),
]


@pytest.mark.parametrize("selections,charge,multiplicity,electrons", CHEMISTRY)
def test_a_selection_derives_spin_only_when_its_electrons_decide_it(selections, charge,
                                                                    multiplicity, electrons):
    from myrmidon.engines.mimic.qm_region import region_chemistry

    sections = parse_cpmd((PREPQM / "prepqm_acetone.inp").read_text())
    chemistry, findings = region_chemistry(sections, charge, selections, {})
    assert (chemistry["multiplicity"], chemistry["electrons"]) == (multiplicity, electrons)
    assert "PP_LMAX_ALL_S_ON_HEAVY_ATOMS" in {f.id for f in findings}


# -- what a stage starts from ---------------------------------------------------------------


MD = {"name": "md", "kind": "dynamics", "ensemble": "NVT", "temperature_k": 300, "steps": 100}


def _qmmm(name, **over):
    from tests.conftest import REGION

    return {"name": name, "kind": "dynamics", "regime": "qmmm", "ensemble": "NVE", "steps": 10,
            "qm_region": REGION, **over}


def _ran(ctx, stage, prepare=None, **run):
    """A stage prepared and run to its end: its submission."""
    from tests.conftest import prepare_and_run, wait_for

    _, submitted = prepare_and_run(ctx, stage, prepare, **run)
    done = wait_for(ctx, submitted.job_id)
    assert done.ok, done.summary
    return submitted


def _left(ctx, stage, role="structure"):
    from myrmidon.core.lineage import stage_run

    return next(a.path for a in stage_run(ctx.workspace, stage)[0] if a.role == role)


def _select_on(ctx, coords, selection="add resname is ACT", **over):
    selected = REGISTRY.call("qm_region", ctx, {
        "topology": "01_input/acetone_equilibrated.top", "coords": coords,
        "selections": [selection], "padding_nm": 0.4, **over})
    assert selected.ok, selected.summary
    return selected


def test_a_coupled_stage_compiles_on_the_coordinates_its_region_was_selected_on(ws_ctx):
    """After a classical run the region has to be selected again: naming the run's structure is
    no way round that, the render reads the new selection, and other atoms are not approved."""
    from tests.conftest import coupled_study

    from myrmidon.core.mandate import read_mandate
    from myrmidon.tools.inputsets import system_file

    coupled_study(ws_ctx, [MD, _qmmm("first")])
    _ran(ws_ctx, "md")
    left = _left(ws_ctx, "md")
    for named in ({}, {"structure": left}):
        refused = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "first", **named})
        assert refused.error.code == "QM_REGION_ON_OTHER_COORDINATES"
    selected = _select_on(ws_ctx, left)
    prepared = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "first"})
    assert prepared.ok, prepared.summary
    held = read_mandate(ws_ctx.workspace)
    assert system_file(ws_ctx, held, "cpmd_input") == next(
        a.path for a in selected.artifacts if a.role == "cpmd_input")
    run = REGISTRY.call("stage_run", ws_ctx, {"stage": "first", "preview": True})
    assert run.ok, run.summary
    _select_on(ws_ctx, left,
               selections=["add resname is ACT", "add resid is 2"])
    other = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "first"})
    assert other.error.code == "QM_SELECTION_NOT_APPROVED"


def test_a_stage_restarting_a_coupled_one_takes_its_coordinates_from_the_restart(ws_ctx):
    """c follows b, whose region was selected on a later frame: c reads b's restart, so what the
    latest selection read says nothing about it."""
    from tests.conftest import coupled_study

    coupled_study(ws_ctx, [_qmmm("a"), _qmmm("b"), _qmmm("c")])
    _ran(ws_ctx, "a")
    left = _left(ws_ctx, "a")
    _select_on(ws_ctx, left)
    _ran(ws_ctx, "b", {"structure": left})
    prepared = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "c"})
    assert prepared.ok, prepared.summary


def test_a_classical_stage_after_a_coupled_one_compiles_on_what_that_left(ws_ctx):
    from tests.conftest import coupled_study

    coupled_study(ws_ctx, [_qmmm("hot"), MD])
    _ran(ws_ctx, "hot")
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "md"}).ok
    event = [e for e in ws_ctx.workspace.ledger.events() if e.tool == "stage_prepare"][-1]
    assert _left(ws_ctx, "hot") in {i.path for i in event.inputs}


@pytest.mark.parametrize("own", [MD, _qmmm("md")], ids=["classical", "coupled"])
def test_a_stage_that_starts_from_itself_runs_afresh_until_it_has_a_run(ws_ctx, own):
    from tests.conftest import coupled_study

    coupled_study(ws_ctx, [{**own, "starts_from": "md"}])
    _ran(ws_ctx, "md")
    goes_on = REGISTRY.call("stage_run", ws_ctx, {"stage": "md", "extend_ps": 1.0,
                                                  "preview": True})
    assert goes_on.ok, goes_on.summary
    event = [e for e in ws_ctx.workspace.ledger.events() if e.tool == "stage_run"][-1]
    assert event.params["continue_run"] is True


def test_a_stage_reads_the_region_selected_for_it(ws_ctx):
    """Two selections a mandate holds, each read by the stage it was made for."""
    from myrmidon.core.mandate import read_mandate
    from myrmidon.tools.dataset import DatasetCollectParams
    from myrmidon.tools.inputsets import system_file

    call, mandates = _study_before_selecting(ws_ctx, [_qmmm("one"), _qmmm("two", starts_from="")])
    gro = "01_input/acetone_equilibrated.gro"
    one = _select_on(ws_ctx, gro, stage="one")
    two = _select_on(ws_ctx, gro, stage="two",
                     selections=["add resname is ACT", "add resid is 2"])
    picks = {name: {a.role: a.path for a in obs.artifacts}
             for name, obs in (("one", one), ("two", two))}
    call["system_files"] += [{"path": picks[n]["qm_selection"], "kind": "qm_selection"}
                             for n in picks]
    written = REGISTRY.call("mandate_write", ws_ctx, call)
    assert written.ok, written.summary
    held = read_mandate(ws_ctx.workspace)
    mandates.record_approval(ws_ctx.workspace, held, approver="researcher", via="terminal")
    for name in picks:
        prepared = REGISTRY.call("stage_prepare", ws_ctx, {"stage": name})
        assert prepared.ok, prepared.summary
        assert system_file(ws_ctx, held, "cpmd_input", name) == picks[name]["cpmd_input"]
        collecting = DatasetCollectParams(stage=name).sized(ws_ctx)
        assert collecting.selection == picks[name]["qm_selection"]


def test_a_mandates_own_input_is_not_replaced_by_a_selection(ws_ctx):
    """A file the researcher named stands, whatever a selection wrote; a classical stage's
    compile takes the same index; and a fit reads no coupled input, so its region is on no
    coordinates of its own."""
    from tests.conftest import REGION, coupled_study

    from myrmidon.core.mandate import read_mandate
    from myrmidon.tools.inputsets import system_file

    coupled_study(ws_ctx, [MD, {"name": "fit", "kind": "parameter_fit", "qm_region": REGION}])
    _ran(ws_ctx, "md")
    fit = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "fit",
                                                  "overrides": {"stride": "[0,10,1]"}})
    assert fit.ok, fit.summary
    held = read_mandate(ws_ctx.workspace)
    mine = ws_ctx.workspace.stage("mine") / "mine.ndx"
    mine.write_text("[ System ]\n1\n")
    named = held.model_copy(deep=True)
    named.system.files = [f for f in named.system.files if f.kind != "index"] + [
        type(named.system.files[0])(path=ws_ctx.workspace.relpath(mine), kind="index")]
    assert system_file(ws_ctx, named, "index") == ws_ctx.workspace.relpath(mine)


def test_a_continuation_asks_only_what_is_left_of_the_stage(ws_ctx, monkeypatch):
    """A restarted log counts from one, so two runs killed at 30 steps of 1000 leave 970, then
    940, not 970 twice, and each resumes from the last STORE; its walltime is booked for
    those steps."""
    from tests.conftest import coupled_study, wait_for

    from myrmidon.core.lineage import stage_left
    from myrmidon.engines.mimic.cpmd_input import keyword_value
    from myrmidon.knowledge import stage as pricing

    booked: list[int] = []
    priced = pricing.qm_walltime

    def booking(**kw):
        booked.append(kw["steps"])
        return priced(**kw)

    killed = (Path(__file__).parents[1] / "fixtures" / "cpmd" / "mimic_heating_berendsen.out"
              ).read_text()
    coupled_study(ws_ctx, [_qmmm("a", steps=1000)])
    _ran(ws_ctx, "a", {"overrides": {"&CPMD/STORE": "10"}})

    def killed_again():
        log = next(a.path for a in stage_left(ws_ctx.workspace, "a") if a.role == "qm_log")
        ws_ctx.workspace.resolve(log).write_text(killed)
        submitted = REGISTRY.call("stage_run", ws_ctx, {"stage": "a", "continue_run": True})
        assert submitted.ok, submitted.summary
        wait_for(ws_ctx, submitted.job_id)
        return int(keyword_value(parse_cpmd(Path(submitted.command[1]).read_text()),
                                 "&CPMD/MAXSTEP"))

    monkeypatch.setattr(pricing, "qm_walltime", booking)
    assert [killed_again(), killed_again()] == [970, 940] == booked


def test_a_continued_ramp_climbs_on_from_where_the_run_reached(ws_ctx):
    """Its thermostat too: the first values would pull a run at 95 K back to where it started."""
    from tests.conftest import coupled_study

    from myrmidon.core.lineage import stage_left
    from myrmidon.engines.mimic.cpmd_input import keyword_value
    from myrmidon.engines.mimic.diagnostics import cpmd_facts

    ramp = {"start_k": 50, "to_k": 300, "rate_k_per_au": 0.01}
    coupled_study(ws_ctx, [_qmmm("hot", steps=2000, ensemble="NVT", temperature_k=ramp)])
    _ran(ws_ctx, "hot")
    log = next(a.path for a in stage_left(ws_ctx.workspace, "hot") if a.role == "qm_log")
    reached = cpmd_facts(ws_ctx.workspace.resolve(log).read_text())["final_temperature_k"]
    goes_on = REGISTRY.call("stage_run", ws_ctx, {"stage": "hot", "continue_run": True,
                                                  "extend_ps": 1.0})
    assert goes_on.ok, goes_on.summary
    sections = parse_cpmd(Path(goes_on.command[1]).read_text())
    assert float(keyword_value(sections, "&CPMD/TEMPERATURE RAMP").split()[0]) == reached
    assert keyword_value(sections, "&CPMD/TEMPERATURE RAMP").split()[1] == "300"
    assert float(keyword_value(sections, "&CPMD/BERENDSEN IONS").split()[0]) == reached


def test_a_coupled_stage_continues_from_its_restart(ws_ctx):
    """A fresh stage continues from the restart it left, reads it, and runs a stated length at
    the input's time step. With no length, a stage that reached every step has nothing left."""
    from tests.conftest import coupled_study, wait_for

    from myrmidon.engines.mimic.cpmd_input import has_keyword, keyword_value
    from myrmidon.knowledge.stage import PS_PER

    coupled_study(ws_ctx, [_qmmm("a")])
    _ran(ws_ctx, "a")
    finished = REGISTRY.call("stage_run", ws_ctx, {"stage": "a", "continue_run": True})
    assert finished.error.code == "STAGE_NOTHING_TO_CONTINUE"
    submitted = REGISTRY.call("stage_run", ws_ctx, {"stage": "a", "continue_run": True,
                                                    "extend_ps": 1.0})
    assert submitted.ok, submitted.summary
    run_input = Path(submitted.command[1])
    sections = parse_cpmd(run_input.read_text())
    assert has_keyword(sections["CPMD"], "RESTART")
    assert (run_input.parent / "RESTART").is_file()
    step_au = float(keyword_value(sections, "&CPMD/TIMESTEP"))
    assert int(keyword_value(sections, "&CPMD/MAXSTEP")) == round(1.0 / (step_au * PS_PER["au"]))
    assert _harness_lines_unedited(ws_ctx, run_input)
    assert wait_for(ws_ctx, submitted.job_id).ok
    _measured_over_both_segments(ws_ctx, "a")


def _measured_over_both_segments(ctx, stage):
    """A continued stage is measured over every segment: the join is a call of its own, each
    segment's trajectory its input, and what it wrote is the stage's."""
    from myrmidon.core.lineage import lineage, stage_segments
    from myrmidon.engines.mimic.diagnostics import cpmd_facts
    from myrmidon.tools.join import segment_trajectory

    segments = [segment_trajectory(written) for _, written, _ in stage_segments(
        ctx.workspace, stage)]
    assert len(set(segments)) == 2
    measured = REGISTRY.call("observable_measure", ctx, {
        "stage": stage, "property": "rmsd", "structure": "01_input/acetone_equilibrated.gro",
        "selection": "resname ACT"})
    assert measured.ok, measured.summary
    join = [e for e in ctx.workspace.ledger.events() if e.tool == "trajectory_join"][-1]
    assert [i.path for i in join.inputs] == segments
    joined = join.outputs[0]
    measure = [e for e in ctx.workspace.ledger.events() if e.tool == "observable_measure"][-1]
    assert joined.sha256 in {i.sha256 for i in measure.inputs}
    assert lineage(ctx.workspace).stages_behind(joined.sha256) == {stage}
    logs = [next(a.path for a in written if a.role == "qm_log")
            for _, written, _ in stage_segments(ctx.workspace, stage)]
    diagnosed = REGISTRY.call("run_diagnose", ctx, {"stage": stage})
    assert (diagnosed.segments, diagnosed.stage_md_steps) == (2, sum(
        cpmd_facts(ctx.workspace.resolve(log).read_text())["md_steps_completed"]
        for log in logs))


def test_a_run_belongs_to_the_stage_it_was_submitted_for(ws_ctx):
    """Two stages may compile to the same bytes: a run of one is still no run of the other."""
    from tests.conftest import coupled_study

    from myrmidon.core.lineage import lineage, stage_files, stage_runs, stages_run

    gro = {"structure": "01_input/acetone_equilibrated.gro"}
    coupled_study(ws_ctx, [{**MD, "name": "one"}, {**MD, "name": "two", "starts_from": ""}])
    _ran(ws_ctx, "one", gro)
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "two", **gro}).ok
    workspace = ws_ctx.workspace
    assert workspace.resolve(stage_files(workspace, "one")["tpr"]).read_text() == (
        workspace.resolve(stage_files(workspace, "two")["tpr"]).read_text())
    assert stage_runs(workspace, "one") and not stage_runs(workspace, "two")
    assert stages_run(workspace) == {"one"}
    written = next(a for a in stage_runs(workspace, "one")[0][0] if a.role == "trajectory")
    assert lineage(workspace).stages_behind(written.sha256) == {"one"}


def test_a_stage_starting_from_itself_tries_again_after_a_first_run_that_left_nothing(
        ws_ctx, monkeypatch):
    from tests.conftest import coupled_study, prepare_and_run, wait_for

    coupled_study(ws_ctx, [{**MD, "starts_from": "md"}])
    monkeypatch.setenv("MOCK_GMX_FAIL", "lincs")
    _, failed = prepare_and_run(ws_ctx, "md")
    assert not wait_for(ws_ctx, failed.job_id).ok
    monkeypatch.delenv("MOCK_GMX_FAIL")
    again = REGISTRY.call("stage_run", ws_ctx, {"stage": "md"})
    assert again.ok, again.summary


# how a coupled stage is told of the snapshot it starts from
SNAPSHOT_AS = [lambda path: {"snapshot": path}, lambda path: {"inputs": {"geometry": path}}]


@pytest.mark.parametrize("named", SNAPSHOT_AS, ids=["snapshot", "inputs"])
def test_a_coupled_snapshot_is_what_the_next_stage_reads(ws_ctx, named):
    """Cut from a run of this study it passes the gate and CPMD reads it; another's is refused."""
    from tests.conftest import coupled_study

    from myrmidon.engines.mimic.cpmd_input import has_keyword

    coupled_study(ws_ctx, [_qmmm("a"), _qmmm("b")])
    _ran(ws_ctx, "a")
    cut = REGISTRY.call("snapshot_select", ws_ctx, {"stage": "a", "target_temperature_k": 0.5})
    assert cut.ok, cut.summary
    assert REGISTRY.call("stage_prepare", ws_ctx, {"stage": "b"}).ok
    stranger = ws_ctx.workspace.stage("elsewhere") / "GEOMETRY"
    stranger.write_text(" 2 2 2 0 0 0\n")
    refused = REGISTRY.call("stage_run", ws_ctx, {
        "stage": "b", **named(ws_ctx.workspace.relpath(stranger))})
    assert not refused.ok
    submitted = REGISTRY.call("stage_run", ws_ctx, {"stage": "b",
                                                    **named(cut.frames[0]["geometry"])})
    assert submitted.ok, submitted.summary
    run_input = Path(submitted.command[1])
    restart = next(ln for ln in parse_cpmd(run_input.read_text())["CPMD"]
                   if ln.strip().upper().startswith("RESTART"))
    assert has_keyword(parse_cpmd(run_input.read_text())["CPMD"], "RESTART")
    assert "GEOFILE" in restart.upper() and (run_input.parent / "GEOMETRY").is_file()
    assert _harness_lines_unedited(ws_ctx, run_input)


def _harness_lines_unedited(ctx, run_input: Path) -> bool:
    """What the harness rewrote in a run's own copy of its input (the &MIMIC PATHS of the run's
    directory, a RESTART line's GEOFILE, a continuation's steps) is no edit by hand."""
    from myrmidon.engines.file_origin import edit_findings

    return edit_findings(ctx, ctx.workspace.relpath(run_input)) == []


def test_a_classical_snapshot_starts_a_stage_of_the_same_system(ws_ctx):
    """A frame of a mock run reads as a real one does, and stands on the run it was cut from."""
    from tests.conftest import coupled_study

    coupled_study(ws_ctx, [MD, {**MD, "name": "next", "starts_from": "md"}])
    _ran(ws_ctx, "md")
    frames = REGISTRY.call("snapshot_select", ws_ctx, {"stage": "md", "n_frames": 1})
    assert frames.ok, frames.summary
    _ran(ws_ctx, "next", {"structure": frames.frames[0]["path"]})


def _study_before_selecting(ws_ctx, stages, **extra):
    """Acetone in water under an approved mandate that was written before any selection."""
    import shutil

    from tests.conftest import ACETONE, QM_HAMILTONIAN

    from myrmidon.core import mandate as mandates

    ws = ws_ctx.workspace
    stage = ws.stage("input")
    for name in ("acetone_equilibrated.top", "acetone.itp", "acetone_equilibrated.gro"):
        (stage / name).write_text((ACETONE / name).read_text())
    shutil.copytree(ACETONE / "amber03.ff", stage / "amber03.ff")
    return {"question": "q", "ceiling_core_h": 5000.0, "hamiltonian": QM_HAMILTONIAN,
            "system_files": [{"path": "01_input/acetone_equilibrated.gro", "kind": "structure"},
                             {"path": "01_input/acetone_equilibrated.top", "kind": "topology"}],
            "subsystems": {"solute": {"kind": "ligand", "resnames": ["ACT"]},
                           "solvent": {"kind": "solvent", "molecules": ["SOL"]}},
            "stages": stages, **extra}, mandates


# (the stage's stated multiplicity, what preparing it says)
LEFT_OPEN = [(None, "QM_REGION_OPEN"), (2, None)]


@pytest.mark.parametrize("multiplicity,code", LEFT_OPEN)
def test_chemistry_a_later_selection_leaves_open_stops_the_stage_until_it_is_settled(
        ws_ctx, multiplicity, code):
    """Written before the selection the mandate could not ask; preparing the stage still does."""
    from tests.conftest import REGION

    region = {**REGION, **({"multiplicity": multiplicity} if multiplicity else {})}
    call, mandates = _study_before_selecting(ws_ctx, [_qmmm("first", qm_region=region)])
    written = REGISTRY.call("mandate_write", ws_ctx, call)
    assert written.ok, written.summary
    mandates.record_approval(ws_ctx.workspace, mandates.read_mandate(ws_ctx.workspace),
                             approver="researcher", via="terminal")
    selected = _select_on(ws_ctx, "01_input/acetone_equilibrated.gro",
                          "add-bound resname is ACT")
    assert selected.derived_multiplicity is None
    prepared = REGISTRY.call("stage_prepare", ws_ctx, {"stage": "first"})
    assert prepared.ok is (code is None), prepared.summary
    assert (prepared.error.code if prepared.error else None) == code


def test_a_lock_on_the_spin_does_not_settle_an_open_charge(ws_ctx):
    import json

    from tests.conftest import REGION

    call, _ = _study_before_selecting(ws_ctx, [_qmmm("first", qm_region={
        **REGION, "multiplicity": 1})])
    selected = _select_on(ws_ctx, "01_input/acetone_equilibrated.gro",
                          "add-bound resname is ACT")
    side = ws_ctx.workspace.resolve(next(a.path for a in selected.artifacts
                                         if a.role == "provenance"))
    side.write_text(json.dumps({**json.loads(side.read_text()), "charge": None}))
    spin = {"said": "a singlet", "locks": {"first.qm_region.multiplicity": 1}}
    asked = REGISTRY.call("mandate_write", ws_ctx, {**call, "decisions_yours": [spin]})
    assert asked.status != "ready"
    charge = {"said": "neutral", "locks": {"first.qm_region.charge": 0}}
    settled = REGISTRY.call("mandate_write", ws_ctx, {**call, "decisions_yours": [charge]})
    assert settled.status == "ready", settled.summary

