"""Where did this file's parameters come from?

A run-parameter file reaching an engine has one of four histories, and the ledger
already knows which:

``inputset``    rendered by ``inputset_apply``; the sidecar names the input set, its
                version, and the origin of every individual key.
``user_input``  no tool produced it, so it came from outside the harness — the user
                planted it in the workspace. Their file, their numbers.
``authored``    written in this session by ``file_write``. Somebody typed the values,
                and if that somebody was the model then nothing records where they
                came from. This is the case the store exists to remove.
``derived``     produced by a tool that took its scientific content from another file
                (mimicpy's prepqm builds ``cpmd.inp`` from a template), so the
                question is really about that file and we follow one hop.

This module answers the question; the engine gates decide what to do about the
answer, and they report it. The answer stays advisory rather than a finding that must
be acknowledged: while the input set store does not cover every recipe people run, a
warning on every call would be acknowledged reflexively, which is the failure the gate
inputset exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from myrmidon.core.validation import ValidationWarning

Provenance = Literal["inputset", "user_input", "authored", "derived", "unknown"]

#: tools that produce a file whose scientific content came from another file, and the
#: parameter naming that file. Following one hop answers the real question.
UPSTREAM_PARAM = {"mimic_select_qm": "template"}

#: the tool that means "the model typed this"
AUTHORING_TOOLS = {"file_write"}


class ParameterSource(BaseModel):
    path: str
    provenance: Provenance
    sourced: bool  # is there a record of where these values came from?
    detail: str
    inputset: str | None = None
    via: list[str] = Field(default_factory=list)  # the hops followed to get here

    @property
    def line(self) -> str:
        return f"{self.path}: {self.detail}"


def _sidecar_protocol(root: Path, relpath: str) -> dict | None:
    sidecar = root / (relpath + ".myrmidon.json")
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return None
    return data if data.get("written_by") == "inputset_apply" else None


def parameter_source(workspace, relpath: str, _depth: int = 0) -> ParameterSource:
    """How the values in ``relpath`` got there, from the ledger and the sidecar."""
    root = Path(workspace.root)

    data = _sidecar_protocol(root, relpath)
    if data is not None:
        name = f"{data['inputset']} v{data.get('protocol_version', 1)}"
        stage = data.get("stage")
        keys = data.get("origin") or {}
        mine = sum(1 for v in keys.values() if v == "user")
        return ParameterSource(
            path=relpath,
            provenance="inputset",
            sourced=True,
            inputset=data["inputset"],
            detail=f"rendered from inputset {name}"
            + (f" stage {stage}" if stage else "")
            + (f", {mine} value(s) set by the caller" if mine else ""),
        )

    producer = None
    for event in workspace.ledger.events():
        if event.ok and any(art.path == relpath for art in event.outputs):
            producer = event  # the last write wins
    if producer is None:
        return ParameterSource(
            path=relpath,
            provenance="user_input",
            sourced=True,
            detail="supplied as a workspace input — no tool produced it",
        )
    if producer.tool in AUTHORING_TOOLS:
        return ParameterSource(
            path=relpath,
            provenance="authored",
            sourced=False,
            detail=f"written by {producer.tool} in this session; nothing records where "
            "its values came from",
        )
    upstream_param = UPSTREAM_PARAM.get(producer.tool)
    upstream = (producer.params or {}).get(upstream_param) if upstream_param else None
    if upstream and _depth < 3:
        origin = parameter_source(workspace, upstream, _depth + 1)
        return ParameterSource(
            path=relpath,
            provenance="derived" if origin.sourced else origin.provenance,
            sourced=origin.sourced,
            inputset=origin.inputset,
            detail=f"built by {producer.tool} from {upstream} — {origin.detail}",
            via=[upstream, *origin.via],
        )
    return ParameterSource(
        path=relpath,
        provenance="unknown",
        sourced=False,
        detail=f"produced by {producer.tool}; its parameters have no recorded origin",
    )


def unsourced_finding(
    sources: list[ParameterSource], severity: str = "info"
) -> ValidationWarning | None:
    """One finding for the files whose values nobody can account for.

    Advisory by design: see the module docstring.
    """
    unsourced = [s for s in sources if not s.sourced]
    if not unsourced:
        return None
    return ValidationWarning(
        id="PARAM_UNSOURCED",
        severity=severity,
        message="no recorded source for the parameters in "
        + "; ".join(s.line for s in unsourced),
        assumption="every value in these files is taken on the author's word. A file "
        "rendered by inputset_apply records which inputset each value came from, and "
        "which values the caller set instead",
        suggestion="inputset_lookup for an inputset that covers this step, then "
        "inputset_apply to render it; if the values came from the user, say so in the "
        "answer so the record is not silent about it",
    )


def sourced_finding(sources: list[ParameterSource]) -> ValidationWarning | None:
    """The positive half: say what the parameters *are* backed by, so the record shows
    the input set and version a run used without anyone having to open the sidecar."""
    backed = [s for s in sources if s.provenance in ("inputset", "derived")]
    if not backed:
        return None
    return ValidationWarning(
        id="PARAM_SOURCED",
        severity="info",
        message="; ".join(s.line for s in backed),
    )
