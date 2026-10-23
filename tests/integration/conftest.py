"""Real-engine tests run only when the developer says so.

Engine discovery answers "can this run?", and on a cluster the answer is yes, which
would let a bare `pytest` queue a real force-matching job. Whether it *should* run is a
separate decision, so every test in this directory is skipped unless
`MYRMIDON_REAL_ENGINES=1` is set for the invocation. The per-test markers still decide
which engines a test needs once it is authorised.
"""

import os

import pytest

AUTHORISED = os.environ.get("MYRMIDON_REAL_ENGINES") == "1"


def pytest_collection_modifyitems(config, items):
    if AUTHORISED:
        return
    skip = pytest.mark.skip(
        reason="real-engine test: set MYRMIDON_REAL_ENGINES=1 to authorise it"
    )
    here = os.path.dirname(__file__)
    for item in items:
        if str(item.path).startswith(here):
            item.add_marker(skip)
