"""Engine findings typed where the failure happens.

Each test plants a failure and checks that the tool's own observation carries the
diagnosis, under a stable finding id, with no free-text store involved.
"""

import pytest

from myrmidon.core.tool import REGISTRY


def _ids(obs):
    return {w.id for w in obs.advisory}


def _pdb(ws, text):
    stage = ws.stage("fetch")
    (stage / "input.pdb").write_text(text)
    return ws.relpath(stage / "input.pdb")


PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00\n"
    "END\n"
)


def test_a_structure_with_no_atoms_is_named_as_corrupt_input(ws_ctx):
    rel = _pdb(ws_ctx.workspace, "<html><body>404 Not Found</body></html>\n")
    obs = REGISTRY.call("gmx_pdb2gmx", ws_ctx,
                        {"structure": rel, "forcefield": "amber99sb-ildn", "water": "tip3p"})
    assert not obs.ok
    assert "STRUCTURE_HAS_NO_ATOMS" in _ids(obs)
    assert "no atoms" in obs.summary


def test_a_genion_that_failed_after_editing_the_topology_says_so(ws_ctx, monkeypatch):
    ws = ws_ctx.workspace
    stage = ws.stage("prep")
    (stage / "topol.top").write_text("; MOCK_NET_CHARGE 1\n[ molecules ]\nSOL 10\n")
    (stage / "ions.tpr").write_text("tpr\n")
    monkeypatch.setenv("MOCK_GMX_FAIL", "genion")
    obs = REGISTRY.call("gmx_genion", ws_ctx, {
        "tpr": ws.relpath(stage / "ions.tpr"), "topology": ws.relpath(stage / "topol.top"),
        "neutralize": True,
    })
    assert not obs.ok
    assert "GENION_TOPOLOGY_ALREADY_EDITED" in _ids(obs)
    finding = next(w for w in obs.advisory if w.id == "GENION_TOPOLOGY_ALREADY_EDITED")
    assert "#topol.top.1#" in finding.message


def _edr(ws):
    stage = ws.stage("run")
    (stage / "md.edr").write_text("edr\n")
    return ws.relpath(stage / "md.edr")


# (the term asked for, it is read). GROMACS names terms with hyphens; a spaced name is not
# in the edr, and the refusal hands back the labels the file does offer.
@pytest.mark.parametrize("term,read", [("Total-Energy", True), ("Total Energy", False)])
def test_an_energy_term_is_read_by_the_label_the_edr_gives_it(ws_ctx, term, read):
    obs = REGISTRY.call("gmx_energy", ws_ctx, {"edr": _edr(ws_ctx.workspace), "terms": [term]})
    assert obs.ok is read
    if read:
        assert term in obs.terms and obs.available_terms == []
    else:
        assert "ENERGY_TERM_NOT_IN_EDR" in _ids(obs)
        assert {"Total-Energy", "Kinetic-En."} <= set(obs.available_terms)


def test_cpmd_restraints_in_a_coupled_run_need_an_explicit_acknowledgement(ws_ctx):
    from tests.unit.test_mimic import _select_qm

    select = _select_qm(ws_ctx)
    cpmd_inp = next(a.path for a in select.artifacts if a.role == "cpmd_input")
    prepared = REGISTRY.call("mimic_prepare", ws_ctx, {
        "cpmd_input": cpmd_inp, "total_charge": 0, "multiplicity": 1, "maxsteps": 5,
        "distance_restraints": [{"atom_i": 1, "atom_j": 2, "target_a": 2.0}],
        "acknowledge_warnings": ["QM_CHARGE_MULTIPLICITY_UNCONFIRMED"],
    })
    assert prepared.ok, prepared.summary
    run_inp = next(a.path for a in prepared.artifacts if a.role == "cpmd_input")
    stage = ws_ctx.workspace.stage("qmmm")
    (stage / "mimic.tpr").write_text("MOCKTPR\n")
    tpr = ws_ctx.workspace.relpath(stage / "mimic.tpr")

    refused = REGISTRY.call("mimic_run", ws_ctx, {"cpmd_input": run_inp, "tpr": tpr})
    assert not refused.ok
    assert "CPMD_RESTRAINTS_UNDER_MIMIC" in {w.id for w in refused.unacknowledged}
    ran = REGISTRY.call("mimic_run", ws_ctx, {
        "cpmd_input": run_inp, "tpr": tpr,
        "acknowledge_warnings": ["CPMD_RESTRAINTS_UNDER_MIMIC"],
    })
    assert ran.ok, ran.summary


def test_an_included_itp_that_defines_its_own_atomtypes_is_flagged(tmp_path):
    from myrmidon.engines.mimic.fm import itp_atomtypes_warnings

    (tmp_path / "mol.itp").write_text(
        "[ atomtypes ]\nc3 6 12.01 0.0 A 0.34 0.46\n\n[ moleculetype ]\nMOL 3\n"
    )
    (tmp_path / "system.top").write_text('#include "mol.itp"\n[ system ]\nx\n')
    [finding] = itp_atomtypes_warnings(tmp_path / "system.top")
    assert finding.id == "FM_ITP_DEFINES_ATOMTYPES" and finding.severity == "warn"

    # the layout that works: types in the .top, before the molecule's include
    (tmp_path / "mol.itp").write_text("[ moleculetype ]\nMOL 3\n")
    (tmp_path / "system.top").write_text(
        '[ atomtypes ]\nc3 6 12.01 0.0 A 0.34 0.46\n#include "mol.itp"\n'
    )
    assert itp_atomtypes_warnings(tmp_path / "system.top") == []
