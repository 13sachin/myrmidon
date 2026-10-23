"""An input set that renders no file is still an input set, and reading it is still a call.

Two of the values the MiMiC tutorial fixes — the cell type and the solute-to-box clearance,
and the QM region's vacuum padding — are arguments a tool takes, not lines in an engine
file. If an input set had nowhere to put a value like that, those numbers would travel in
a skill's prose with no source, no applicability and no version.

`renders: none` says the set states values, and `inputset_apply` without an `output`
resolves them: every value with its origin and its sources, nothing written, and the call in
the ledger like any other.
"""


from myrmidon.core.tool import REGISTRY


def test_the_call_is_in_the_ledger(ws_ctx):
    """What a value was read from, and when, is the point of reading it through a tool."""
    import json

    obs = REGISTRY.call("inputset_apply", ws_ctx, {"inputset": "box_qmmm_solute"})
    ledger = ws_ctx.workspace.root / "myrmidon.ledger.jsonl"
    events = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    assert any(e["event_id"] == obs.provenance_id for e in events)


def test_what_the_researcher_must_supply_is_still_refused(ws_ctx):
    """Resolve-only is not a way around `required_from_user`: which atoms are quantum is
    chemistry, and the gate blocks on it whether or not a file is being written. A value
    that is a fact about the system in hand (the padding) is refused the same way: the
    input set says how to derive it, and carries no other system's number for it."""
    obs = REGISTRY.call(
        "inputset_apply", ws_ctx, {"inputset": "mimic_qm_region"}
    )
    assert not obs.ok
    named = " ".join(w.message for w in obs.blocking)
    assert "selection" in named and "padding_nm" in named

    supplied = REGISTRY.call(
        "inputset_apply", ws_ctx,
        {"inputset": "mimic_qm_region",
         "overrides": {"selections": ["add resname is LIG"], "padding_nm": 0.4}},
    )
    assert supplied.ok, supplied.summary
    assert supplied.origin["selections"] == supplied.origin["padding_nm"] == "user"

