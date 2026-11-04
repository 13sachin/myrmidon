"""MDP parsing, and the facts the sensor records read of an mdp.

The records (knowledge/rules/) make findings with stable ids of what they see; anything
GROMACS would silently default is surfaced as an explicit assumption.
"""

from __future__ import annotations

from pathlib import Path

from myrmidon.core.contracts import Property, define_kind
from myrmidon.core.validation import Finding
from myrmidon.knowledge.catalog import Catalog, canonical, load
from myrmidon.knowledge.sensors import evaluate


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
    nothing the harness reasons about (out_of_scope), or known — the subject the
    per-key sensor records read."""
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
        out.append({"key": key, "value": value, "status": status,
                    "renamed_to": catalog.renamed.get(name, ""), "version": catalog.version,
                    "default": (entry.default if entry is not None
                                and entry.default not in (None, "") else "unset")})
    return out


def validate_mdp(params: dict[str, str], catalog: Catalog | None = None,
                 ) -> tuple[list[Finding], list[str]]:
    """What the sensor records make of an mdp (knowledge/rules/), read against the GROMACS
    version's key catalog (the newest shipped when none is given), and the checks that
    ran."""
    catalog = catalog or load("gromacs")
    return (evaluate("mdp", params, version=catalog.version,
                     keys=key_status(params, catalog)), ["sensors"])


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
