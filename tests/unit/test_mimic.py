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


def test_run_gate_blocks_unconfirmed_charge(ws_ctx):
    select = _select_qm(ws_ctx)
    cpmd_inp = next(a.path for a in select.artifacts if a.role == "cpmd_input")
    stage = ws_ctx.workspace.stage("qmmm")
    (stage / "mimic.tpr").write_text("MOCKTPR\n")
    refused = REGISTRY.call(
        "mimic_run",
        ws_ctx,
        {"cpmd_input": cpmd_inp, "tpr": ws_ctx.workspace.relpath(stage / "mimic.tpr")},
    )
    assert not refused.ok
    blocking_ids = {w.id for w in blocking(refused)}
    # mock prepqm writes no CHARGE keyword into &SYSTEM? it does not — both block
    assert "QM_MULTIPLICITY_UNCONFIRMED" in blocking_ids


def _full_mimic_setup(ws_ctx):
    select = _select_qm(ws_ctx)
    cpmd_inp = next(a.path for a in select.artifacts if a.role == "cpmd_input")
    prepared = REGISTRY.call(
        "mimic_prepare",
        ws_ctx,
        {"cpmd_input": cpmd_inp, "total_charge": 0, "multiplicity": 1, "maxsteps": 5},
    )
    run_inp = next(a.path for a in prepared.artifacts if a.role == "cpmd_input")
    stage = ws_ctx.workspace.stage("qmmm")
    (stage / "mimic.tpr").write_text("MOCKTPR\n")
    return run_inp, ws_ctx.workspace.relpath(stage / "mimic.tpr")


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
def test_prepare_writes_the_charge_and_the_spin_cpmd_will_accept(ws_ctx, charge,
                                                                 multiplicity, written, lsd):
    import json

    select = _select_qm(ws_ctx)
    cpmd_inp = next(a.path for a in select.artifacts if a.role == "cpmd_input")
    obs = REGISTRY.call("mimic_prepare", ws_ctx, {
        "cpmd_input": cpmd_inp, "total_charge": charge, "multiplicity": multiplicity,
        "maxsteps": 50})
    assert obs.ok, obs.summary
    assert (obs.charge, obs.multiplicity, obs.functional, obs.paths) == (
        charge, multiplicity, "BLYP", 1)
    out = next(a.path for a in obs.artifacts if a.role == "cpmd_input")
    sections = parse_cpmd((ws_ctx.workspace.root / out).read_text())
    assert get_keyword(sections["SYSTEM"], "CHARGE") == str(charge)
    assert get_keyword(sections["SYSTEM"], "MULTIPLICITY") == written
    assert any(line.strip() == "LSD" for line in sections["CPMD"]) is lsd
    assert get_keyword(sections["CPMD"], "MAXSTEP") == "50"
    meta = next(a.path for a in obs.artifacts if a.role == "run_metadata")
    assert json.loads((ws_ctx.workspace.root / meta).read_text())["multiplicity"] == multiplicity


def test_prepare_refuses_an_input_that_is_not_a_coupled_run(ws_ctx):
    stage = ws_ctx.workspace.stage("qmmm")
    (stage / "plain.inp").write_text("&CPMD\nOPTIMIZE WAVEFUNCTION\n&END\n")
    obs = REGISTRY.call("mimic_prepare", ws_ctx, {
        "cpmd_input": ws_ctx.workspace.relpath(stage / "plain.inp"),
        "total_charge": 0, "multiplicity": 1})
    assert not obs.ok and not obs.artifacts


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

# (the selection, the charge and multiplicity the call gives, what is written — or what it
# asks for — and a finding on the result). The selected atoms' topology charges decide the
# charge when they sum to an integer; the electron count decides the multiplicity when it is
# even and no open-shell metal is in the region (acetone: 32 electrons). A value the
# researcher stated is taken as given, and one its electrons cannot have is said.
SPIN_STATE = [
    (WHOLE, (None, None), (0, 1), None),
    (WHOLE, (0, 3), (0, 3), None),
    (WHOLE, (0, 2), (0, 2), "QM_MULTIPLICITY_PARITY"),
    (WHOLE, (1, None), ["multiplicity"], None),  # 31 electrons: a doublet or higher
    (CUT, (None, None), ["total_charge"], None),  # prepqm rounded a non-integer sum
]


@pytest.mark.parametrize("selections,given,written,finding", SPIN_STATE)
def test_a_qm_charge_and_multiplicity_are_derived_or_asked(ws_ctx, selections, given,
                                                           written, finding):
    select = _select(ws_ctx, selections)
    cpmd_inp = next(a.path for a in select.artifacts if a.role == "cpmd_input")
    call = {"cpmd_input": cpmd_inp}
    for key, value in zip(("total_charge", "multiplicity"), given, strict=True):
        if value is not None:
            call[key] = value
    obs = REGISTRY.call("mimic_prepare", ws_ctx, call)
    if isinstance(written, list):
        assert not obs.ok and obs.status == "needs_input" and obs.needs == written
        assert obs.error.code == "NEEDS_INPUT"
        return
    assert obs.ok, obs.summary
    assert (obs.charge, obs.multiplicity) == written and len(obs.facts) == 2
    assert (finding in {f.id for f in obs.findings}) if finding else not {
        f.id for f in obs.findings} & {"QM_MULTIPLICITY_PARITY"}


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
# never the system. Each row is a stage of a real staged run: its input made by mimic_prepare
# from the annealing system input and the stage's method must be the input it ran with.
STAGES = Path(__file__).parents[1] / "fixtures" / "mimic_stages"


@pytest.mark.parametrize("stage", ["nve_test", "heating", "production"])
def test_a_stage_input_is_its_method_on_the_system_made_once(ws_ctx, stage):
    from myrmidon.engines.mimic.cpmd_input import parse_cpmd

    here = ws_ctx.workspace.stage(stage)
    for name in ("system_from_prepqm.inp", f"{stage}_method.inp"):
        (here / name).write_text((STAGES / name).read_text())
    rel = ws_ctx.workspace.relpath(here)
    obs = REGISTRY.call("mimic_prepare", ws_ctx, {
        "cpmd_input": f"{rel}/system_from_prepqm.inp", "method": f"{rel}/{stage}_method.inp",
        "total_charge": 0, "multiplicity": 1, "output": "run.inp"})
    assert obs.ok, obs.summary

    def lines(path):
        return {k: [x.strip() for x in v if x.strip()]
                for k, v in parse_cpmd(path.read_text()).items()}

    made, real = lines(here / "run.inp"), lines(STAGES / f"{stage}_run_real.inp")
    assert list(made) == list(real)
    assert made == real


def test_a_restarted_segment_hashes_the_restart_it_stages(ws_ctx):
    """What the node reads of a continued segment includes the restart staged into its
    directory: an edit to the staged copy while the job waits is seen at collection."""
    from myrmidon.execution.jobs import existing_jobstore

    run_inp, tpr = _full_mimic_setup(ws_ctx)
    ws_ctx.person = "researcher"
    first = REGISTRY.call("mimic_run", ws_ctx, {"cpmd_input": run_inp, "tpr": tpr})
    assert first.ok, first.summary
    done = _wait(ws_ctx, first.job_id)
    restart = next(a.path for a in done.artifacts if a.role == "restart")
    second = REGISTRY.call("mimic_run", ws_ctx, {"cpmd_input": run_inp, "tpr": tpr,
                                                 "restart_from": restart})
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
