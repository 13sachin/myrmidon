"""Numeric grounding: no number reaches the researcher that no tool, parameter or prompt
gave. One table of verdicts, one of sources, and the loop that enforces it."""

import pytest

from myrmidon.agent.grounding import NumericIndex

OBSERVATION = {
    "ok": True,
    "terms": {"Temperature": {"mean": 299.734, "std": 4.21}},
    "n_atoms": 8514,
    "summary": "Potential: -118234.56 kJ/mol over 10.0 ps",
}
IDENTIFIERS = {
    "ok": True, "provenance_id": "ev-4242", "job_id": "job-4242",
    "artifacts": [{"path": "02_run/x4242.gro", "sha256": "4242aa", "size_bytes": 4242}],
    "command": ["gmx", "mdrun", "-nsteps", "4242"],
    "summary": "Energy minimised in 4243 steps",
}

# (what the tools returned, the answer, the numbers that must come back ungrounded)
VERDICTS = [
    # exact, rounded to the written precision, a value quoted in a summary
    (OBSERVATION, "The temperature was 299.734 K.", []),
    (OBSERVATION, "The temperature was 299.7 K (rounded).", []),
    (OBSERVATION, "Roughly 300 K.", []),
    (OBSERVATION, "8514 atoms in -118234.56 kJ/mol.", []),
    (OBSERVATION, "time span 10.0 ps", []),
    # a number no tool gave
    (OBSERVATION, "The diffusion coefficient is 2.35e-5 cm^2/s.", ["2.35e-5"]),
    ({"count": 42}, "value 37.5 was measured", ["37.5"]),
    # a number written against its unit is still scanned; 5 is a small integer
    (OBSERVATION, "The system equilibrated at 299.7K.", []),
    (OBSERVATION, "equilibrated at 312K over 5ns, error 0.0027Ha/Bohr.", ["312", "0.0027"]),
    # typography: U+2212 minus, ×10⁵, x 10^5, a cube that is a unit
    ({"e": -139728.31}, "Mean potential energy: −139728.31 kJ/mol", []),
    ({"epot": -122400.0}, "Epot = −1.224×10⁵ kJ/mol", []),
    ({"epot": -122400.0}, "Epot = -1.224 x 10^5 kJ/mol", []),
    ({"epot": -122400.0}, "Epot = -1.3×10⁵ kJ/mol", ["-1.3e5"]),
    ({"box": 4.39229}, "final box 4.392³ nm³", []),
    ({"epot": -122437.0}, "Epot = −1.224×10⁵ kJ/mol", []),
    ({"epot": -122437.0}, "Epot = −1.25×10⁵ kJ/mol", ["-1.25e5"]),
    # one significant figure is a rounding only within 5 %
    ({"slope": 0.00029}, "slope 3×10⁻⁴ kJ/mol/ps", []),
    ({"a": 0.00029}, "slope 3e-4", []),
    ({"c": 98000.0}, "about 1e5 atoms", []),
    ({"b": 60000.0}, "about 1e5 atoms", ["1e5"]),
    ({"b": 60000.0}, "about 6e4 atoms", []),
    # identifiers are not claims, in the answer or in the observation
    ({}, "topology with amber99sb-ildn for 1l2y using spc216.gro", []),
    ({}, "tip3p water, amber99sb-ildn, task t906, 1l2y, spc216.gro, 01_input/x", []),
    (IDENTIFIERS, "It took 4242 steps.", ["4242"]),
    (IDENTIFIERS, "It took 4243 steps.", []),
    # small integers are counts and numbering
    ({}, "I ran 3 stages and produced 2 artifacts in 1 workspace.", []),
]


@pytest.mark.parametrize("tree,answer,ungrounded", VERDICTS)
def test_a_number_grounds_only_if_a_tool_gave_it(tree, answer, ungrounded):
    idx = NumericIndex()
    idx.add_tree(tree)
    report = idx.check(answer)
    assert [float(u) for u in report.ungrounded] == [float(u) for u in ungrounded]


def test_identifiers_in_the_answer_are_not_checked_at_all():
    report = NumericIndex().check("tip3p, amber99sb-ildn, t906, 1l2y, spc216.gro, 01_input/x")
    assert report.checked == 0


def test_a_bool_is_not_a_number():
    idx = NumericIndex()
    idx.add_tree({"ok": True})
    report = idx.check("value 1.0 was measured")
    assert report.by_source == {} and report.exempt == ["1.0"]


def _with(**sources):
    idx = NumericIndex()
    for source, tree in sources.items():
        if source == "parameter":
            idx.add_parameters(tree)
        else:
            idx.add_tree(tree, source=source)
    return idx


# (what each source gave, the answer, ungrounded, by_source, how many are measurements)
SOURCES = [
    # a free-text argument is where a computed number would be laundered
    (dict(parameter={"notes": "the difference is 12.34 kJ/mol", "walltime_min": 120}),
     "I booked 120 minutes; the difference is 12.34 kJ/mol.",
     ["12.34"], {"parameter": ["120"]}, 0),
    # a number the researcher said is grounded, but nobody measured it
    (dict(prompt="a 100 ns production run", observation={"core_h_estimate": 48.0}),
     "A 100 ns run would book 48 core-hours.",
     [], {"prompt": ["100"], "observation": ["48"]}, 1),
    # a number a model-authored script printed is traceable, not measured
    (dict(script={"summary": "script printed 1.2345"}),
     "The fitted value is 1.2345.",
     [], {"script": ["1.2345"]}, 0),
]


@pytest.mark.parametrize("sources,answer,ungrounded,by_source,measured", SOURCES)
def test_every_grounded_number_names_its_source(sources, answer, ungrounded, by_source,
                                                measured):
    report = _with(**sources).check(answer)
    assert report.ungrounded == ungrounded
    assert report.by_source == by_source
    assert report.measured == measured


def test_the_loop_grounds_against_the_prompt(ctx, tmp_path):
    from tests.unit.test_agent_loop import FakeMessage, scripted

    from myrmidon.agent.loop import AgentLoop

    loop = AgentLoop(
        model="fake", ctx=ctx, transcript_dir=tmp_path / "eps",
        completion_fn=scripted(FakeMessage(content="Ran 10 ps NVT at 300 K with dt 0.002.")),
    )
    episode = loop.run("Run a 10 ps NVT at 300 K (dt 0.002).")
    assert episode.n_turns == 1 and episode.grounding.ok
