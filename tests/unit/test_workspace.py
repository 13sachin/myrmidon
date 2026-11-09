"""Workspace layout, ledger append and read-back."""

from pathlib import Path

import pytest

from myrmidon.core.provenance import ProvenanceEvent, ProvenanceLedger, shquote


def test_ledger_roundtrip(tmp_path):
    ledger = ProvenanceLedger(tmp_path / "ledger.jsonl")
    ev1 = ProvenanceEvent(tool="gmx_pdb2gmx", command=["gmx", "pdb2gmx", "-f", "in.pdb"])
    ev2 = ProvenanceEvent(tool="study_open", command=None)
    ev3 = ProvenanceEvent(tool="stage_run", command=["gmx", "mdrun", "-deffnm", "md run"])
    for ev in (ev1, ev2, ev3):
        ledger.append(ev)

    events = list(ledger.events())
    assert [e.tool for e in events] == ["gmx_pdb2gmx", "study_open", "stage_run"]
    assert ledger.get(ev1.event_id).command == ev1.command
    assert ledger.get(ev2.event_id).command is None
    assert ledger.get(ev3.event_id).command == ["gmx", "mdrun", "-deffnm", "md run"]


def test_shquote():
    assert shquote("simple-arg_1.0") == "simple-arg_1.0"
    assert shquote("has space") == "'has space'"
    assert shquote("it's") == "'it'\\''s'"


# (what a study of the project names, where it resolves: in the study, the project, or the
# study beside it of the same project — never another project's, never a bare name searched
# across studies)
RESOLVES = [
    ("mine.txt", "{study}/mine.txt"),
    ("box/heat.gro", "{project}/box/heat.gro"),
    ("equil/01_npt/confout.gro", "{studies}/equil/01_npt/confout.gro"),
    ("theirs/01_npt/confout.gro", "{study}/theirs/01_npt/confout.gro"),
    ("confout.gro", "{study}/confout.gro"),
    ("equil/../../outside.txt", "{study}/equil/../../outside.txt"),  # never out of a study
]


@pytest.mark.parametrize("named,resolves", RESOLVES)
def test_a_path_resolves_in_the_study_its_project_and_the_projects_studies(tmp_path, named,
                                                                           resolves):
    from myrmidon.core.workspace import Workspace

    project, studies = tmp_path / "project", tmp_path / "studies"
    (project / "box").mkdir(parents=True)
    (project / "box" / "heat.gro").write_text("x")
    (tmp_path / "outside.txt").write_text("x")
    ws = Workspace.create(studies / "prod", "prod", origin=project)
    (ws.root / "mine.txt").write_text("x")
    for name, origin in (("equil", project), ("theirs", tmp_path / "other")):
        made = Workspace.create(studies / name, name, origin=origin)
        (made.stage("npt") / "confout.gro").write_text("x")
    fill = {"study": ws.root, "project": project, "studies": studies.resolve()}
    assert ws.resolve(named) == Path(resolves.format(**fill))
