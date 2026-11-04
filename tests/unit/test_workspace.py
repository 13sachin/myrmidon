"""Workspace layout, ledger append and read-back."""


from myrmidon.core.provenance import ProvenanceEvent, ProvenanceLedger, shquote


def test_ledger_roundtrip(tmp_path):
    ledger = ProvenanceLedger(tmp_path / "ledger.jsonl")
    ev1 = ProvenanceEvent(tool="gmx_pdb2gmx", command=["gmx", "pdb2gmx", "-f", "in.pdb"])
    ev2 = ProvenanceEvent(tool="study_open", command=None)
    ev3 = ProvenanceEvent(tool="gmx_mdrun", command=["gmx", "mdrun", "-deffnm", "md run"])
    for ev in (ev1, ev2, ev3):
        ledger.append(ev)

    events = list(ledger.events())
    assert [e.tool for e in events] == ["gmx_pdb2gmx", "study_open", "gmx_mdrun"]
    assert ledger.get(ev1.event_id).command == ev1.command
    assert ledger.get(ev2.event_id).command is None
    assert ledger.get(ev3.event_id).command == ["gmx", "mdrun", "-deffnm", "md run"]


def test_shquote():
    assert shquote("simple-arg_1.0") == "simple-arg_1.0"
    assert shquote("has space") == "'has space'"
    assert shquote("it's") == "'it'\\''s'"

