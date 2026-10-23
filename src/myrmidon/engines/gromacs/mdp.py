"""MDP parameter parsing and sanity rules.

Rules emit ValidationWarnings with stable ids. Anything GROMACS would silently
default is surfaced as an explicit assumption — the model (or the user) must
either set the parameter or acknowledge the assumption by id.
"""

from __future__ import annotations

from pathlib import Path

from myrmidon.core.validation import ValidationWarning


def parse_mdp(text: str) -> dict[str, str]:
    """Parse mdp text into normalized {key: value}. GROMACS treats '-' and '_'
    in keys as equivalent; we normalize to '_' and lowercase."""
    params: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        params[key.strip().lower().replace("-", "_")] = value.strip()
    return params


def parse_mdp_file(path: Path) -> dict[str, str]:
    return parse_mdp(path.read_text())


def _get_float(params: dict[str, str], key: str) -> float | None:
    try:
        return float(params[key].split()[0])
    except (KeyError, ValueError, IndexError):
        return None


def validate_mdp(params: dict[str, str]) -> tuple[list[ValidationWarning], list[str]]:
    """Returns (warnings, names_of_checks_that_ran)."""
    warnings: list[ValidationWarning] = []
    checked: list[str] = []

    dt = _get_float(params, "dt")
    constraints = params.get("constraints", "").lower() or None
    integrator = params.get("integrator", "").lower()

    checked.append("timestep")
    if integrator not in ("steep", "cg", "l-bfgs"):  # minimizers have no dt semantics
        if dt is None:
            warnings.append(
                ValidationWarning(
                    id="MDP_DT_UNSET",
                    severity="warn",
                    message="dt is not set in the mdp file",
                    assumption="GROMACS would silently default to dt = 0.001 ps",
                    suggestion="set dt explicitly (e.g. 0.002 with constraints = h-bonds)",
                )
            )
        elif dt > 0.004:
            warnings.append(
                ValidationWarning(
                    id="MDP_DT_TOO_LARGE",
                    severity="block",
                    message=f"dt = {dt} ps exceeds 4 fs — unstable for atomistic MD",
                    suggestion="use dt <= 0.002 ps (0.004 only with virtual sites/HMR)",
                )
            )
        elif dt > 0.0015 and constraints in (None, "none"):
            warnings.append(
                ValidationWarning(
                    id="MDP_DT_LARGE_NO_CONSTRAINTS",
                    severity="warn",
                    message=(
                        f"dt = {dt} ps without bond constraints — hydrogen-bond "
                        "vibrations are unstable above ~1 fs"
                    ),
                    suggestion="set constraints = h-bonds, or reduce dt to 0.001",
                )
            )

    checked.append("steps")
    nsteps = _get_float(params, "nsteps")
    if integrator not in ("steep", "cg", "l-bfgs") and nsteps is None:
        warnings.append(
            ValidationWarning(
                id="MDP_NSTEPS_UNSET",
                severity="warn",
                message="nsteps is not set",
                assumption="GROMACS would silently default to nsteps = 0 (no dynamics)",
                suggestion="set nsteps for the intended simulation length",
            )
        )

    checked.append("thermostat")
    tcoupl = params.get("tcoupl", "").lower()
    if integrator == "mimic":
        checked.append("mimic")
        if not params.get("qmmm_grps"):
            warnings.append(
                ValidationWarning(
                    id="MDP_MIMIC_NO_QMMM_GRPS",
                    severity="block",
                    message="integrator = mimic requires QMMM-grps (the QM atoms index "
                    "group written by mimic_select_qm)",
                    suggestion="set QMMM-grps = QMatoms and pass index.ndx to gmx_grompp",
                )
            )
        if tcoupl not in ("", "no"):
            warnings.append(
                ValidationWarning(
                    id="MDP_MIMIC_THERMOSTAT",
                    severity="warn",
                    message=f"tcoupl = {tcoupl} in a MiMiC run — CPMD integrates and "
                    "thermostats the whole system; a GROMACS thermostat double-counts",
                    suggestion="set tcoupl = no and control temperature in &CPMD",
                )
            )
        if params.get("constraints", "none").lower() not in ("none",):
            warnings.append(
                ValidationWarning(
                    id="MDP_MIMIC_CONSTRAINTS",
                    severity="warn",
                    message=f"constraints = {params['constraints']} — constraints on QM "
                    "atoms fight the QM forces",
                    suggestion="constraints = none (or exclude the QM region)",
                )
            )
    if integrator == "md" and tcoupl in ("", "no"):
        warnings.append(
            ValidationWarning(
                id="MDP_NO_THERMOSTAT",
                severity="warn",
                message="integrator = md with no thermostat (tcoupl unset/no)",
                assumption="the run would be NVE — energy drift will not be corrected",
                suggestion="set tcoupl (e.g. v-rescale) unless NVE is intended",
            )
        )
    if tcoupl not in ("", "no") and "ref_t" not in params:
        warnings.append(
            ValidationWarning(
                id="MDP_REF_T_UNSET",
                severity="block",
                message=f"tcoupl = {tcoupl} but ref_t (target temperature) is not set",
            )
        )

    checked.append("electrostatics")
    coulombtype = params.get("coulombtype", "").lower()
    rcoulomb = _get_float(params, "rcoulomb")
    rvdw = _get_float(params, "rvdw")
    if coulombtype in ("cut-off", "cutoff", ""):
        warnings.append(
            ValidationWarning(
                id="MDP_NO_PME",
                severity="warn",
                message=f"coulombtype = '{coulombtype or 'unset'}' — plain cutoff "
                "electrostatics are inaccurate for charged/polar systems",
                assumption=(
                    "unset coulombtype silently defaults to cut-off"
                    if not coulombtype
                    else None
                ),
                suggestion="use coulombtype = PME",
            )
        )
    if rcoulomb is not None and rvdw is not None and rcoulomb != rvdw:
        warnings.append(
            ValidationWarning(
                id="MDP_CUTOFF_MISMATCH",
                severity="info",
                message=f"rcoulomb ({rcoulomb}) != rvdw ({rvdw}) — fine with PME, "
                "but verify this is intended",
            )
        )

    checked.append("velocities")
    if (
        params.get("gen_vel", "").lower() == "yes"
        and params.get("gen_seed", "-1").strip() == "-1"
    ):
        warnings.append(
            ValidationWarning(
                id="MDP_RANDOM_SEED",
                severity="info",
                message="gen_vel = yes with gen_seed unset (-1): grompp draws a fresh seed, so "
                "the run input is not byte-reproducible and a replay starts from different "
                "velocities. Set gen_seed explicitly for a replayable study.",
                assumption="gen_seed = -1 (random)",
                suggestion="gen_seed = <integer>",
            )
        )
    if (
        params.get("gen_vel", "").lower() == "yes"
        and params.get("continuation", "").lower() == "yes"
    ):
        warnings.append(
            ValidationWarning(
                id="MDP_GEN_VEL_ON_CONTINUATION",
                severity="warn",
                message="gen_vel = yes together with continuation = yes — regenerating "
                "velocities on a continuation discards the previous state",
            )
        )

    return warnings, checked


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
