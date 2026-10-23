"""Where each value in an engine file came from, and what the user changed.

The values a researcher reads and edits are the engine's own files,
which they already know. `inputset_apply` renders one and leaves a sidecar naming the
origin of every key; when the researcher opens the file and changes a number, this module
is what notices. A value the study declared as its experiment's axis becomes theirs and is
reported as such; a
value the input set states is a finding to acknowledge, with the input set's rationale
beside it; a key the input set never mentions is an addition.

Nothing in the file is myrmidon's — no marker, no lock. The record is in the sidecar and
the diff is computed when it is needed, so editing the file in any editor works.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.sourcing import ParameterSource, parameter_source
from myrmidon.core.validation import ValidationWarning

UNCHANGED, ON_AXIS, CHANGED, ADDED, REMOVED = (
    "unchanged", "on_declared_axis", "changed", "added", "removed"
)


class KeyOrigin(BaseModel):
    """One parameter of one file, and who put it there."""

    key: str
    value: str | None = None  # as the file has it now
    origin: str  # inputset:<id>@v<N> | user | unknown
    status: str = UNCHANGED
    protocol_value: str | None = None  # what the inputset said, when it differs
    on_declared_axis: bool = False  # the study said it varies this key
    rationale: str | None = None

    @property
    def line(self) -> str:
        if self.status in (UNCHANGED, ADDED):
            return f"{self.key} = {self.value} ({self.origin})"
        if self.status == REMOVED:
            return f"{self.key} removed (inputset said {self.protocol_value})"
        return f"{self.key} = {self.value} (yours; inputset said {self.protocol_value})"


class FileOrigin(BaseModel):
    """The answer to "where did this file's numbers come from"."""

    path: str
    provenance: str  # inputset | user_input | authored | derived | unknown
    sourced: bool
    detail: str
    inputset: str | None = None
    keys: list[KeyOrigin] = Field(default_factory=list)
    findings: list[ValidationWarning] = Field(default_factory=list)

    def by_status(self, status: str) -> list[KeyOrigin]:
        return [k for k in self.keys if k.status == status]

    @property
    def yours(self) -> list[KeyOrigin]:
        """What this file says is the user's: on the declared axis, changed, or added."""
        return [k for k in self.keys if k.status in (ON_AXIS, CHANGED, ADDED)]

    @property
    def edited(self) -> bool:
        return bool(self.yours or self.by_status(REMOVED))


def sidecar_path(root: Path, relpath: str) -> Path:
    return root / (relpath + ".myrmidon.json")


def read_sidecar(root: Path, relpath: str) -> dict | None:
    path = sidecar_path(root, relpath)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if data.get("written_by") == "inputset_apply" else None


def _same(a, b) -> bool:
    """Is the file's text the input set's value? Numbers compare as numbers."""
    left, right = str(a).strip(), str(b).strip()
    if left.lower() == right.lower():
        return True
    try:
        return float(left.split()[0]) == float(right.split()[0])
    except (ValueError, IndexError):
        return False


def _current_values(root: Path, relpath: str, renders: str | None) -> dict[str, str] | None:
    """The file as it is now, keyed the way the input set keys it."""
    path = root / relpath
    if not path.is_file():
        return None
    kind = renders or ("cpmd_inp" if path.suffix == ".inp" else "mdp")
    if kind == "mdp":
        from myrmidon.engines.gromacs.mdp import parse_mdp_file

        return parse_mdp_file(path)
    if kind == "cpmd_inp":
        return _cpmd_values(path.read_text())
    return None


def _cpmd_values(text: str) -> dict[str, str]:
    """A CPMD input back as ``&SECTION/KEYWORD -> value`` — the inverse of the render.

    CPMD carries a keyword's argument either on the keyword line or on the line after
    it, and only the writer knows which; an indented line following a keyword is taken
    as that keyword's value, which is how ``sections_from_values`` writes them.
    """
    from myrmidon.engines.mimic.cpmd_input import parse_cpmd

    values: dict[str, str] = {}
    for section, lines in parse_cpmd(text).items():
        if section == "INFO":  # provenance the renderer wrote, not a parameter
            continue
        pending: str | None = None
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if pending is not None and not line[0].isalpha():
                values[f"&{section}/{pending}"] = line
                pending = None
                continue
            if pending is not None:
                values[f"&{section}/{pending}"] = "True"
            pending = line
        if pending is not None:
            values[f"&{section}/{pending}"] = "True"
    return values


def _normalize(key: str, kind: str) -> str:
    """The key as the engine reads it. GROMACS treats '-' and '_' alike and ignores
    case, so `DispCorr` and `dispcorr` are one parameter, not an addition and a
    removal; CPMD keywords are case-insensitive and whitespace-separated."""
    if kind == "mdp":
        return key.strip().lower().replace("-", "_")
    return " ".join(key.upper().split())


def _spec_of(ctx, protocol_id: str, key: str):
    """The input set's own word on this parameter, if this machine has the input set."""
    store = getattr(ctx, "inputsets", None)
    if store is None:
        return None
    try:
        inputset = store.get(protocol_id)
    except (KeyError, AttributeError):
        return None
    return inputset.parameters.get(key)


def file_origin(ctx, relpath: str) -> FileOrigin:
    """Every value in this engine file, with its origin and what was changed by hand."""
    workspace = ctx.workspace
    root = Path(workspace.root)
    source: ParameterSource = parameter_source(workspace, relpath)
    data = read_sidecar(root, relpath)
    origin = FileOrigin(
        path=relpath, provenance=source.provenance, sourced=source.sourced,
        detail=source.detail, inputset=source.inputset,
    )
    if data is None:
        return origin  # no inputset rendered it: parameter_source already said what it is

    written = data.get("values") or {}
    origins = data.get("origin") or {}
    on_axis = set(data.get("on_axis") or [])
    protocol_id = data.get("inputset")
    kind = data.get("renders") or ("cpmd_inp" if Path(relpath).suffix == ".inp" else "mdp")
    current = _current_values(root, relpath, kind)
    if current is None:
        origin.detail += " — the file itself is gone"
        return origin
    by_norm = {_normalize(k, kind): v for k, v in current.items()}

    seen: set[str] = set()
    for key, value in written.items():
        seen.add(_normalize(key, kind))
        now = by_norm.get(_normalize(key, kind))
        spec = _spec_of(ctx, protocol_id, key)
        declared = key in on_axis  # the study declared this key as its experiment's axis
        rationale = getattr(spec, "rationale", None)
        if now is None:
            origin.keys.append(KeyOrigin(
                key=key, origin=origins.get(key, "unknown"), status=REMOVED,
                protocol_value=str(value), on_declared_axis=declared, rationale=rationale,
            ))
        elif _same(now, value):
            origin.keys.append(KeyOrigin(
                key=key, value=str(now), origin=origins.get(key, "unknown"),
                status=UNCHANGED, on_declared_axis=declared, rationale=rationale,
            ))
        else:
            origin.keys.append(KeyOrigin(
                key=key, value=str(now), origin="user",
                status=ON_AXIS if declared else CHANGED,
                protocol_value=str(value), on_declared_axis=declared, rationale=rationale,
            ))
    for key, value in current.items():
        if _normalize(key, kind) in seen:
            continue
        origin.keys.append(KeyOrigin(
            key=key, value=str(value), origin="user", status=ADDED
        ))
    origin.findings = findings_for(origin)
    return origin


def findings_for(origin: FileOrigin) -> list[ValidationWarning]:
    """What the diff means for a gate: a value on the study's declared axis is recorded,
    a stated value is a decision, an addition is noted, a removal is a decision."""
    out: list[ValidationWarning] = []
    tuned = origin.by_status(ON_AXIS)
    if tuned:
        out.append(ValidationWarning(
            id="INPUTSET_VALUE_ON_DECLARED_AXIS", severity="info",
            message=f"{origin.path}: " + "; ".join(k.line for k in tuned)
            + " — this study declared these as the experiment it is varying",
        ))
    changed = origin.by_status(CHANGED)
    if changed:
        out.append(ValidationWarning(
            id="INPUTSET_VALUE_FIXED", severity="warn",
            message=f"{origin.path} was edited after it was rendered: "
            + "; ".join(k.line for k in changed)
            + "".join(
                f". {k.key}: {k.rationale}" for k in changed if k.rationale
            ),
            assumption="the inputset states these values; the edit replaces one that "
            "something validated",
            suggestion="keep the edit and acknowledge INPUTSET_VALUE_FIXED, or render "
            "the file again from the input set",
        ))
    removed = origin.by_status(REMOVED)
    if removed:
        out.append(ValidationWarning(
            id="INPUTSET_VALUE_REMOVED", severity="warn",
            message=f"{origin.path} has lost " + "; ".join(k.line for k in removed),
            assumption="the engine would default what the inputset stated",
            suggestion="put the key back, or acknowledge INPUTSET_VALUE_REMOVED",
        ))
    added = origin.by_status(ADDED)
    if added:
        out.append(ValidationWarning(
            id="INPUTSET_VALUE_ADDED", severity="info",
            message=f"{origin.path} has {len(added)} key(s) the inputset never mentions: "
            + ", ".join(k.key for k in added),
        ))
    return out


def edit_findings(ctx, relpath: str) -> list[ValidationWarning]:
    """What a gate needs to know about a file that was edited after it was rendered.

    Empty when no inputset rendered the file: then there is nothing to diff against,
    and ``parameter_source`` has already said so.
    """
    if read_sidecar(Path(ctx.workspace.root), relpath) is None:
        return []
    return file_origin(ctx, relpath).findings


def render(origin: FileOrigin) -> str:
    lines = [f"{origin.path} — {origin.detail}"]
    if not origin.keys:
        return lines[0]
    counts: dict[str, int] = {}
    for key in origin.keys:
        counts[key.status] = counts.get(key.status, 0) + 1
    lines.append(", ".join(f"{n} {status}" for status, n in sorted(counts.items())))
    lines.append("")
    for key in origin.keys:
        mark = " " if key.status == UNCHANGED else "*"
        lines.append(f" {mark} {key.line}")
    if origin.findings:
        lines.append("")
        for finding in origin.findings:
            lines.append(f"  {finding.severity:<5} {finding.id}: {finding.message}")
    return "\n".join(lines)
