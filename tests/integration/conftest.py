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

if AUTHORISED:
    # the test modules look for the engines on import: without the machine's env script a
    # real-engine run would skip every test instead of running it
    from myrmidon.config import MyrmidonConfig
    from myrmidon.core.envscript import apply_env_script

    apply_env_script(MyrmidonConfig.load().env_script)


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


@pytest.fixture
def approve_mandate():
    """Write a mandate over the built system and record the researcher's yes, so the jobs
    run under the gate the way a study's jobs do."""

    def approve(ctx, question, ceiling_core_h, stages=(), hamiltonian=None, subsystems=None,
                **files):
        from myrmidon.core import mandate as mandates
        from myrmidon.core.tool import REGISTRY

        system = [{"path": path, "kind": kind} for kind, path in files.items()]
        written = REGISTRY.call("mandate_write", ctx, {
            "question": question, "ceiling_core_h": ceiling_core_h, "system_files": system,
            "stages": list(stages), **({"hamiltonian": hamiltonian} if hamiltonian else {}),
            **({"subsystems": subsystems} if subsystems else {})})
        assert written.ok, written.summary
        ws = ctx.workspace
        held = mandates.read_mandate(ws)
        mandates.record_approval(ws, held, approver="researcher", via="terminal")
        return held

    return approve
