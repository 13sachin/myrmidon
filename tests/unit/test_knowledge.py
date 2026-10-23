"""Knowledge tools with mocked HTTP — no network in unit tests."""

import httpx

from myrmidon.core.tool import REGISTRY


class FakeResponse:
    def __init__(self, json_data=None, text=""):
        self._json = json_data
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def test_pubchem_lookup(ctx, monkeypatch):
    def fake_get(url, **kw):
        assert "acetate" in url
        return FakeResponse(
            json_data={
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 175,
                            "MolecularFormula": "C2H3O2-",
                            "MolecularWeight": "59.04",
                            "Charge": -1,
                            "CanonicalSMILES": "CC(=O)[O-]",
                            "IUPACName": "acetate",
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    obs = REGISTRY.call("lookup", ctx, {"source": "pubchem", "query": "acetate"})
    assert obs.ok and obs.source == "pubchem"
    assert obs.cid == 175
    assert obs.charge == -1
    assert obs.formula == "C2H3O2-"
    assert "CID 175" in obs.summary  # citable


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
