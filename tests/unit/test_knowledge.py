"""Knowledge tools with mocked HTTP — no network in unit tests."""

import httpx
import pytest

from myrmidon.core.tool import REGISTRY
from myrmidon.errors import MyrmidonError


class FakeResponse:
    def __init__(self, json_data=None, text=""):
        self._json = json_data
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


# (the source, the query, what the service is asked, what it answers, the fields the
# lookup returns): a compound's identity, citable, from PubChem by name or from the PDB by
# residue code, whose charge is the dictionary's reference form's
COMPOUNDS = [
    ("pubchem", "acetate", "acetate", {"PropertyTable": {"Properties": [{
        "CID": 175, "MolecularFormula": "C2H3O2-", "MolecularWeight": "59.04", "Charge": -1,
        "CanonicalSMILES": "CC(=O)[O-]", "IUPACName": "acetate"}]}},
     {"source": "pubchem", "cid": 175, "charge": -1, "formula": "C2H3O2-"}),
    ("ligand", "jz4", "/chemcomp/JZ4", {"chem_comp": {"id": "JZ4", "name": "2-propylphenol",
                                     "formula": "C9 H12 O", "formula_weight": 136.191,
                                     "pdbx_formal_charge": 0},
                       "rcsb_chem_comp_descriptor": {"SMILES": "CCCc1ccccc1O"}},
     {"source": "rcsb-chemcomp", "name": "2-propylphenol", "reference_charge": 0,
      "smiles": "CCCc1ccccc1O"}),
    ("ligand", "JZ4 ligand", None, None, {"source": None}),  # no residue code: not asked
]


@pytest.mark.parametrize("source,query,asked,answer,fields", COMPOUNDS)
def test_a_compound_is_looked_up_by_what_names_it(ctx, monkeypatch, source, query, asked,
                                                  answer, fields):
    def fake_get(url, **kw):
        assert asked is not None and asked in url
        return FakeResponse(json_data=answer)

    monkeypatch.setattr(httpx, "get", fake_get)
    obs = REGISTRY.call("lookup", ctx, {"source": source, "query": query})
    assert obs.ok is (asked is not None) and obs.failure is None  # refused, never raised
    assert {k: getattr(obs, k, None) for k in fields} == fields


ARXIV_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2502.09565v1</id>
    <title>MDCrow: Automating Molecular Dynamics Protocols</title>
    <published>2025-02-13T00:00:00Z</published>
    <summary>LLM agents for MD.</summary>
    <author><name>Q. Campbell</name></author>
  </entry>
</feed>
"""


def test_lit_search_merges_sources(ctx, monkeypatch):
    def fake_get(url, **kw):
        if "arxiv" in url:
            return FakeResponse(text=ARXIV_XML)
        return FakeResponse(
            json_data={
                "message": {
                    "items": [
                        {
                            "title": ["MiMiC: multiscale modeling framework"],
                            "author": [{"given": "V.", "family": "Bolnykh"}],
                            "DOI": "10.1021/acs.jctc.9b00093",
                            "issued": {"date-parts": [[2019]]},
                            "container-title": ["J. Chem. Theory Comput."],
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    obs = REGISTRY.call("lookup", ctx, {"source": "literature", "query": "mimic qmmm"})
    assert obs.ok and obs.source == "literature"
    titles = [p.title for p in obs.papers]
    assert any("MDCrow" in t for t in titles)
    assert any("MiMiC" in t for t in titles)
    doi_paper = next(p for p in obs.papers if p.doi)
    assert doi_paper.year == 2019
    arxiv_paper = next(p for p in obs.papers if p.arxiv_id)
    assert arxiv_paper.arxiv_id == "2502.09565v1"


def test_pdb_search(ctx, monkeypatch):
    def fake_post(url, **kw):
        assert kw["json"]["query"]["parameters"]["value"] == "trp-cage"
        return FakeResponse(
            json_data={
                "result_set": [
                    {"identifier": "1L2Y", "score": 1.0},
                    {"identifier": "2JOF", "score": 0.9},
                ]
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    obs = REGISTRY.call("lookup", ctx, {"source": "pdb", "query": "trp-cage"})
    assert obs.ok and obs.source == "rcsb-pdb"
    assert [h.pdb_id for h in obs.hits] == ["1L2Y", "2JOF"]
    assert "1L2Y" in obs.summary


def test_every_sensor_record_is_sourced_unique_and_readable():
    """A sensor is knowledge a reviewer audits: one id each, a verified one says where it
    comes from, and every condition is one the evaluator reads."""
    import pytest

    from myrmidon.knowledge.sensors import SensorRecord, evaluate, load_sensors

    records = load_sensors()  # a condition the evaluator cannot read fails to load
    ids = [r.id for r in records]
    assert records and len(ids) == len(set(ids))
    assert [r.id for r in records if r.status == "verified" and not r.sources] == []
    for kind in {r.subject for r in records}:
        evaluate(kind, {}, keys=[{}])
    for when in ({"dt": {"bogus": 1}}, {"dt": {"gt": "0.1"}}, {"x": {"differs_from": 1}}):
        with pytest.raises(ValueError):
            SensorRecord(id="X", subject="mdp", sees="x", when=when)


def test_a_tool_profile_names_registered_tools_and_narrows_what_a_client_sees(ws_ctx):
    """A profile shows the core set and its own tools; a tool outside it is still one
    tool_run away, and the call is that tool's own."""
    from myrmidon.core.tool import REGISTRY
    from myrmidon.knowledge.tool_profiles import profiles
    from myrmidon.server.mcp_server import _tool_run, exposed_tools

    registered = {s.name for s in REGISTRY.specs()}
    assert [t for tools in profiles().values() for t in tools if t not in registered] == []
    ws_ctx.config.tool_profile = "md"
    shown = {s.name for s in exposed_tools(ws_ctx)}
    assert {"mandate_write", "stage_run", "tool_lookup", "gmx_pdb2gmx"} <= shown
    assert "dataset_collect" not in shown and len(shown) < len(registered)
    result = _tool_run(ws_ctx)("study_open", {})
    assert result["ok"] and list(ws_ctx.workspace.ledger.events())[-1].tool == "study_open"
    assert "schema" in _tool_run(ws_ctx)("study_open", {"wrong": 1})
    with pytest.raises(MyrmidonError, match="the profiles are"):
        ws_ctx.config.tool_profile = "nonesuch"
        exposed_tools(ws_ctx)
