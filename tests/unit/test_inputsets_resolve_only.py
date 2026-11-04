"""An input set that renders no file is still an input set, and reading it is still a call.

Two of the values the MiMiC tutorial fixes — the cell type and the solute-to-box clearance,
and the QM region's vacuum padding — are arguments a tool takes, not lines in an engine
file. If an input set had nowhere to put a value like that, those numbers would travel in
a skill's prose with no source, no applicability and no version.

`renders: none` says the set states values, and `inputset_apply` without an `output`
resolves them: every value with its origin and its sources, nothing written, and the call in
the ledger like any other.
"""


from tests.unit.findings_catalog import blocking

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
    named = " ".join(w.message for w in blocking(obs))
    assert "selection" in named and "padding_nm" in named

    supplied = REGISTRY.call(
        "inputset_apply", ws_ctx,
        {"inputset": "mimic_qm_region",
         "overrides": {"selections": ["add resname is LIG"], "padding_nm": 0.4}},
    )
    assert supplied.ok, supplied.summary
    assert supplied.origin["selections"] == supplied.origin["padding_nm"] == "user"


def test_inputset_apply_accepts_versioned_id_and_refuses_wrong_version(ws_ctx):
    """A plan step cites an input set as `id@v1` or `inputset:id@v1`; `inputset_apply`
    accepts both and refuses only when the version does not match the installed set."""
    ok = REGISTRY.call("inputset_apply", ws_ctx, {"inputset": "box_qmmm_solute@v2",
                                                  "overrides": {"distance_nm": 1.2}})
    assert ok.ok, ok.summary
    assert ok.inputset == "box_qmmm_solute"

    prefixed = REGISTRY.call("inputset_apply", ws_ctx,
                             {"inputset": "inputset:box_qmmm_solute@v2",
                              "overrides": {"distance_nm": 1.2}})
    assert prefixed.ok, prefixed.summary

    wrong = REGISTRY.call("inputset_apply", ws_ctx, {"inputset": "box_qmmm_solute@v99",
                                                     "overrides": {"distance_nm": 1.2}})
    assert not wrong.ok
    assert [w.id for w in blocking(wrong)] == ["INPUTSET_OTHER_VERSION"]


def test_lookup_listings_fit_within_client_cap_and_single_id_returns_full(ws_ctx):
    import json

    from myrmidon.core.observation import model_view

    all_sets = REGISTRY.call("inputset_lookup", ws_ctx, {"query": ""})
    assert all_sets.ok
    assert len(json.dumps(model_view(all_sets))) < 10_000
    assert all(not h.parameters for h in all_sets.inputsets)
    nvt_summary = next(h for h in all_sets.inputsets if h.id == "gromacs.nvt")
    assert "nsteps" in nvt_summary.required_from_user

    one_set = REGISTRY.call("inputset_lookup", ws_ctx, {"query": "gromacs.nvt@v1"})
    assert one_set.ok and len(one_set.inputsets) == 1
    assert one_set.inputsets[0].parameters["integrator"] == "md"
    assert one_set.inputsets[0].sources
    # every set a query names by id, with its values, whatever other words it holds
    named = REGISTRY.call("inputset_lookup", ws_ctx,
                          {"query": "gromacs.nvt, (gromacs.npt) tc-grps nsteps"})
    assert [h.id for h in named.inputsets] == ["gromacs.nvt", "gromacs.npt"]
    assert all(h.parameters for h in named.inputsets)
    every = REGISTRY.call("inputset_lookup", ws_ctx,
                          {"query": " ".join(h.id for h in all_sets.inputsets)})
    assert len(every.inputsets) == len(all_sets.inputsets)
    assert len(json.dumps(model_view(every))) < 20_000

    all_protos = REGISTRY.call("protocol_lookup", ws_ctx, {"query": ""})
    assert all_protos.ok
    assert len(json.dumps(model_view(all_protos))) < 10_000
    one_proto = REGISTRY.call("protocol_lookup", ws_ctx,
                              {"query": "protein_ligand_complex_in_water"})
    assert one_proto.ok and len(one_proto.protocols) == 1
    assert one_proto.protocols[0].citations


