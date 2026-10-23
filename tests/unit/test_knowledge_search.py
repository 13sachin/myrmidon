"""`knowledge_search` and the documentation corpus behind it.

Why it is a tool: a model with the mdp-options page in reach cites the documentation for
every alternative it names and sets `refcoord-scaling` correctly, and without it does
neither — but the page planted as files costs dozens of reads. The tool is the same
knowledge at three passages a call.
"""


from myrmidon.core.tool import REGISTRY
from myrmidon.knowledge.docs import BUILTIN_DOCS, DocStore


def test_the_shipped_corpus_is_cited_versioned_and_hashed():
    store = DocStore()
    passages = store.all()
    assert len(passages) > 200
    for p in passages:
        assert p.locator.startswith("https://") and p.version and p.section and p.sha256
        assert p.licence  # what lets the text ship, or the reason it is a pointer
    assert (BUILTIN_DOCS / "NOTICE").is_file()
    # one page, one hash: a record cannot come from a page other records did not
    by_page = {}
    for p in passages:
        by_page.setdefault((p.source, p.version, p.page), set()).add(p.sha256)
    assert all(len(hashes) == 1 for hashes in by_page.values())


def test_an_option_asked_by_name_comes_first_and_says_what_it_means():
    store = DocStore()
    found = store.search("refcoord-scaling", installed={"gromacs": "2021.6"})
    assert found and found[0].section == "mdp option refcoord-scaling"
    assert "position restraints" in found[0].text and "com" in found[0].text
    # spelled with an underscore, as an mdp file spells it
    assert store.search("refcoord_scaling")[0].section == "mdp option refcoord-scaling"
    # and a question in words finds the option it is about
    found = store.search("which pressure coupling gives correct volume fluctuations")
    assert any(p.section == "mdp option pcoupl" for p in found)
    assert any("C-rescale" in p.text for p in found)


def test_the_tool_answers_with_citations_and_is_among_the_exposed_tools(ws_ctx):
    ctx = ws_ctx
    obs = REGISTRY.call("knowledge_search", ctx, {"query": "refcoord-scaling"})
    assert obs.ok, obs.summary
    assert obs.passages[0].section == "mdp option refcoord-scaling"
    assert obs.citations[0].startswith("gromacs 2021.6, mdp option refcoord-scaling — https://")
    assert "line in citations" in obs.summary
    assert obs.passages[0].text[:40] not in obs.summary
    assert obs.provenance_id  # a lookup is a ledger event like any other
    # the mock gmx reports 2024.2 and the corpus documents 2021.6: the nearest series is
    # read, and the answer says so rather than refusing or pretending
    assert obs.versions["gromacs"].startswith("2024")
    assert obs.passages[0].version == "2021.6"
    assert "installed 2024.2-mock, documentation read is the 2021 series" in obs.summary

    missing = REGISTRY.call("knowledge_search", ctx, {"query": "zzqx frobnicate"})
    assert not missing.ok and "rather than guessing" in missing.summary

    from myrmidon.server.mcp_server import exposed_tools
    assert "knowledge_search" in {s.name for s in exposed_tools(ctx)}

