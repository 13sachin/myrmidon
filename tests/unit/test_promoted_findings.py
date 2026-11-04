"""gmx_energy reads each term by the label the edr gives it."""

import pytest

from myrmidon.core.tool import REGISTRY


def _ids(obs):
    return {w.id for w in obs.findings}


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
        assert term in obs.terms and not obs.available_terms
    else:
        assert "ENERGY_TERM_NOT_IN_EDR" in _ids(obs)
        assert {"Total-Energy", "Kinetic-En."} <= set(obs.available_terms)


