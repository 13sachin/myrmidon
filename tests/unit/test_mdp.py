"""MDP parsing and the rules an mdp is checked against before GROMACS sees it."""

import pytest

from myrmidon.engines.gromacs.mdp import parse_mdp, validate_mdp

MD = ("integrator = md\ndt = 0.001\nnsteps = 10\ntcoupl = v-rescale\ntc-grps = System\n"
      "tau_t = 0.1\nref_t = 300\n")
MIMIC_OK = ("integrator = mimic\nnsteps = 1000\ndt = 0.001\ntcoupl = no\n"
            "constraints = none\nQMMM-grps = QMatoms\ncoulombtype = PME\n")


def test_parse_normalizes_keys_and_strips_comments():
    params = parse_mdp(
        """
        ; a comment
        dt = 0.002   ; timestep
        ref-t   = 300
        Tcoupl = v-rescale
        """
    )
    assert params == {"dt": "0.002", "ref_t": "300", "tcoupl": "v-rescale"}


# (the mdp, findings it must raise as id -> severity, ids it must not raise). What grompp
# itself refuses — an unknown key, a ref-t or tau-t count that does not match tc-grps — is
# grompp's to say: no sensor repeats it.
RULES = [
    ("integrator = md\ndt = 0.002\nnsteps = 100\ntcoupl = v-rescale\nref_t = 300",
     {"MDP_DT_LARGE_NO_CONSTRAINTS": "info"}, set()),
    ("integrator = md\ndt = 0.002\nconstraints = h-bonds\nnsteps = 100\n"
     "tcoupl = v-rescale\nref_t = 300", {}, {"MDP_DT_LARGE_NO_CONSTRAINTS"}),
    (MD + "gen_vel = yes\ncontinuation = yes", {"MDP_GEN_VEL_ON_CONTINUATION": "info"}, set()),
    # a minimiser has no timestep, length or thermostat to check
    ("integrator = steep\nemtol = 1000", {},
     {"MDP_DT_UNSET", "MDP_NSTEPS_UNSET", "MDP_NO_THERMOSTAT"}),
    # an unset seed is surfaced, because the record cannot name it
    (MD + "gen_vel = yes\ncoulombtype = PME", {"MDP_RANDOM_SEED": "info"}, set()),
    (MD + "gen_vel = yes\ngen_seed = 42\ncoulombtype = PME", {}, {"MDP_RANDOM_SEED"}),
    # MiMiC: the coupled run's own rules, and none of plain MD's thermostat rule
    (MIMIC_OK, {}, {"MDP_MIMIC_NO_QMMM_GRPS", "MDP_MIMIC_THERMOSTAT",
                    "MDP_MIMIC_CONSTRAINTS", "MDP_NO_THERMOSTAT"}),
    ("integrator = mimic\nnsteps = 10\ndt = 0.001\ntcoupl = v-rescale\nref_t = 300\n"
     "constraints = h-bonds\ncoulombtype = PME\n",
     {"MDP_MIMIC_NO_QMMM_GRPS": "info", "MDP_MIMIC_THERMOSTAT": "info",
      "MDP_MIMIC_CONSTRAINTS": "info"}, set()),
    # every key against what grompp reads: an obsolete key is ignored, a renamed one read
    # under its new name, a key no owner decides is outside what the harness checks
    (MD + "ns_type = grid", {"MDP_KEY_OBSOLETE": "info"}, set()),
    (MD + "nstxtcout = 100", {"MDP_KEY_RENAMED": "info"}, set()),
    (MD + "free-energy = yes", {"MDP_KEY_OUT_OF_SCOPE": "info"}, set()),
    (MD + "free_energy = no\nnstxout-compressed = 500", {}, {"MDP_KEY_OUT_OF_SCOPE"}),
    (MD, {}, {"MDP_KEY_OUT_OF_SCOPE"}),
]


@pytest.mark.parametrize("mdp,raised,absent", RULES)
def test_an_mdp_is_checked_against_the_rules_that_apply_to_it(mdp, raised, absent):
    warnings, _ = validate_mdp(parse_mdp(mdp))
    found = {w.id: w.severity for w in warnings}
    assert {k: found.get(k) for k in raised} == raised
    assert not (absent & set(found))


def test_unset_parameters_surface_as_assumptions():
    warnings, _ = validate_mdp(parse_mdp("integrator = md"))
    by_id = {w.id: w for w in warnings}
    for finding in ("MDP_DT_UNSET", "MDP_NSTEPS_UNSET"):
        assert by_id[finding].assumption  # what GROMACS would assume instead
        assert by_id[finding].means and by_id[finding].sources  # what it means, and whence
    assert "MDP_NO_THERMOSTAT" in by_id
