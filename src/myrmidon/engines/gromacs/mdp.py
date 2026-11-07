"""MDP parsing, and what GROMACS does silently with an mdp as it is on disk.

The checks here are what the engine gets wrong without saying so, each with where it was
learned (``Check``): a MiMiC run's GROMACS half that thermostats, constrains or names no QM
group, new velocities on a continuation, a seed nobody recorded, a dynamics run with no
length. A key the version's catalog does not read as written is said as the catalog sees it.
What grompp itself refuses is grompp's to say; general MD physics (a time step, a thermostat,
a cut-off) is the agent's.
"""

from __future__ import annotations

from pathlib import Path

from myrmidon.core.contracts import Property, define_kind
from myrmidon.core.validation import Check, Finding
from myrmidon.knowledge.catalog import Catalog, canonical, catalog_findings, load


def parse_mdp(text: str) -> dict[str, str]:
    """Parse mdp text into normalized {key: value}. GROMACS treats '-' and '_'
    in keys as equivalent; we normalize to '_' and lowercase."""
    params: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        params[canonical(key)] = value.strip()
    return params


def parse_mdp_file(path: Path) -> dict[str, str]:
    return parse_mdp(path.read_text())


def _normalized(values: dict) -> dict[str, str]:
    """Rendered values and parsed files under one key spelling."""
    return {canonical(k): str(v) for k, v in values.items()}


def _int_of(params: dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(params[key].split()[0]))
    except (KeyError, ValueError, IndexError):
        return default


# What an mdp decides about the run's outputs, as GROMACS defaults them: no coordinates
# unless an output interval is set, energies every 1000 steps unless set to 0.
def _length(p: dict) -> tuple[float | None, float | None]:
    """(ps, steps): nsteps × dt, and nsteps; a negative nsteps runs without end."""
    steps = _number_of(p.get("nsteps"))
    dt = _number_of(p.get("dt"))
    if steps is None or steps < 0:
        return None, None
    return (steps * dt if dt else None), steps


def _number_of(value) -> float | None:
    try:
        return float(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


define_kind(
    "mdp", parse_mdp_file,
    Property("frames", lambda p: any(_int_of(p, k, 0) > 0
                           for k in ("nstxout", "nstxout_compressed", "nstxtcout"))),
    Property("energies", lambda p: _int_of(p, "nstenergy", 1000) > 0),
    Property("forces", lambda p: _int_of(p, "nstfout", 0) > 0),
    normalize=_normalized,
    lookup=lambda p, key: p.get(canonical(key)),
    length=lambda p: _length(p),
)


def _same(a: str, b: str) -> bool:
    try:
        return [float(x) for x in a.split()] == [float(x) for x in b.split()]
    except ValueError:
        return a.strip().lower() == b.strip().lower()


def key_status(params: dict[str, str], catalog: Catalog) -> list[dict]:
    """Each key as the version's catalog sees it — obsolete, renamed, unknown, decided by
    nothing the harness reasons about (out_of_scope), or known — what
    ``catalog_findings`` reads."""
    from myrmidon.knowledge.stage import decided_keys

    decided = decided_keys("gromacs")
    out = []
    for key, value in params.items():
        name = canonical(key)
        entry = catalog.get(name)
        status = ("obsolete" if name in catalog.obsolete
                  else "renamed" if name in catalog.renamed
                  else "unknown" if entry is None
                  else "out_of_scope" if canonical(entry.name) not in decided
                  and not (entry.default is not None and _same(value, entry.default))
                  else "known")
        out.append({"key": key, "label": key, "name": key, "value": value, "status": status,
                    "renamed_to": catalog.renamed.get(name, ""), "version": catalog.version,
                    "default": (entry.default if entry is not None
                                and entry.default not in (None, "") else "unset")})
    return out


_MDP_OPTIONS = "https://manual.gromacs.org/2021.6/user-guide/mdp-options.html"
_MIMIC_MANUAL = "https://manual.gromacs.org/2021.6/reference-manual/special/mimic-qmmm.html"
#: a key's status in the version's catalog, as the finding it raises
KEY_CHECKS = {
    "obsolete": Check(id="MDP_KEY_OBSOLETE", sources=(_MDP_OPTIONS,)),
    "renamed": Check(id="MDP_KEY_RENAMED", sources=(_MDP_OPTIONS,)),
    "out_of_scope": Check(id="MDP_KEY_OUT_OF_SCOPE", sources=(_MDP_OPTIONS,)),
    "unknown": Check(id="MDP_KEY_UNKNOWN", sources=(_MDP_OPTIONS,)),
}
MIMIC_NO_QMMM_GRPS = Check(id="MDP_MIMIC_NO_QMMM_GRPS", sources=(_MIMIC_MANUAL,))
# a thermostat or constraints on the GROMACS half of a coupled run is physics the engine gets
# wrong without a word: the render, an override and a researcher's own mdp all stop on it
MIMIC_THERMOSTAT = Check(id="MDP_MIMIC_THERMOSTAT", sources=(_MIMIC_MANUAL,), stops=True)
MIMIC_CONSTRAINTS = Check(id="MDP_MIMIC_CONSTRAINTS", sources=(_MIMIC_MANUAL,), stops=True)
RANDOM_SEED = Check(id="MDP_RANDOM_SEED", sources=(f"{_MDP_OPTIONS}#mdp-gen-seed",))
GEN_VEL_ON_CONTINUATION = Check(id="MDP_GEN_VEL_ON_CONTINUATION",
                                sources=(f"{_MDP_OPTIONS}#mdp-continuation",))
NSTEPS_UNSET = Check(id="MDP_NSTEPS_UNSET", sources=(f"{_MDP_OPTIONS}#mdp-nsteps",))
CHECKS = (*KEY_CHECKS.values(), MIMIC_NO_QMMM_GRPS, MIMIC_THERMOSTAT, MIMIC_CONSTRAINTS,
          RANDOM_SEED, GEN_VEL_ON_CONTINUATION, NSTEPS_UNSET)
#: integrators that minimise: their nsteps bounds the search, and unset it is GROMACS's own
_MINIMISERS = ("steep", "cg", "l-bfgs")


def _says(params: dict[str, str], key: str, *words: str) -> bool:
    """Whether the mdp sets ``key`` to one of ``words`` (case aside); unset is none."""
    return str(params.get(key, "")).strip().lower() in words


def mdp_findings(params: dict[str, str]) -> list[Finding]:
    """What GROMACS does silently with this mdp as it is."""
    out: list[Finding] = []
    yes = ("yes", "true", "on")
    if _says(params, "integrator", "mimic"):
        if not params.get("qmmm_grps", "").strip():
            out.append(MIMIC_NO_QMMM_GRPS.finding(
                "integrator = mimic with no QMMM-grps",
                means="the MM side does not know which atoms are quantum",
                suggestion="QMMM-grps = the QM atoms' index group (qm_region writes it), and "
                "that index among the system files"))
        if params.get("tcoupl", "").strip() and not _says(params, "tcoupl", "no"):
            out.append(MIMIC_THERMOSTAT.finding(
                f"tcoupl = {params['tcoupl']} in a MiMiC run",
                means="CPMD integrates and thermostats the whole system; a GROMACS thermostat "
                "double-counts", suggestion="tcoupl = no, and the temperature in &CPMD"))
        if params.get("constraints", "").strip() and not _says(params, "constraints", "none"):
            out.append(MIMIC_CONSTRAINTS.finding(
                f"constraints = {params['constraints']} in a MiMiC run",
                means="constraints on the QM atoms fight the QM forces; the quantum solute has "
                "to be flexible", suggestion="constraints = none"))
    if not params.get("nsteps", "").strip() and not _says(params, "integrator", *_MINIMISERS):
        out.append(NSTEPS_UNSET.finding(
            "nsteps is not set", means="GROMACS defaults nsteps to 0: the run integrates "
            "nothing, and ends as one that finished", assumption="nsteps = 0",
            suggestion="state the stage's steps or duration_ps in the mandate, and "
            "stage_prepare it again"))
    if _says(params, "gen_vel", *yes):
        if not params.get("gen_seed", "").strip() or _says(params, "gen_seed", "-1"):
            out.append(RANDOM_SEED.finding(
                f"gen_vel = yes with gen_seed = {params.get('gen_seed') or 'unset'}",
                means="grompp draws a fresh seed: the starting velocities are not on record, "
                "and a second grompp of the same inputs starts elsewhere",
                assumption="gen_seed = -1 (random)", suggestion="gen_seed = an integer"))
        if _says(params, "continuation", *yes):
            out.append(GEN_VEL_ON_CONTINUATION.finding(
                "gen_vel = yes with continuation = yes",
                means="new velocities on a continuation discard the state the run continues "
                "from", suggestion="gen_vel = no on a continuation"))
    return out


def validate_mdp(params: dict[str, str], catalog: Catalog | None = None,
                 ) -> tuple[list[Finding], list[str]]:
    """What GROMACS does silently with an mdp, and its keys as the version's catalog (the
    newest shipped when none is given) reads them; and the checks that ran."""
    catalog = catalog or load("gromacs")
    keys = catalog_findings(
        key_status(params, catalog), catalog, KEY_CHECKS, noun="key",
        unknown_means="grompp stops on an unknown key, unless max_warn hides the warning and "
        "the value is silently dropped")
    return [*mdp_findings(params), *keys], ["engine_checks", "key_catalog"]


def format_mdp_value(value: object) -> str:
    """GROMACS spelling for a YAML scalar. Booleans become yes/no, which is what
    an input set author writing ``gen_vel: true`` means."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def render_mdp(
    values: dict[str, object],
    origin: dict[str, str] | None = None,
    header: list[str] | None = None,
) -> str:
    """Render an mdp file, naming the source of every value in a trailing comment.

    The comments are for the reader; the machine-readable origin map lives in the
    ``.myrmidon.json`` sidecar next to the file. Both exist because the question
    "where did this number come from" is asked by people and by tools.
    """
    lines = [f"; {line}" for line in (header or [])]
    if lines:
        lines.append("")
    width = max((len(k) for k in values), default=0)
    for key, value in values.items():
        rendered = f"{key.ljust(width)} = {format_mdp_value(value)}"
        source = (origin or {}).get(key)
        lines.append(f"{rendered}  ; {source}" if source else rendered)
    return "\n".join(lines) + "\n"
