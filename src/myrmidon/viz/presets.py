"""Named style presets: command blocks that make figures look consistent.

A preset is a function returning a list of ChimeraX commands, so presets that
need a selection (qmmm) can demand it explicitly instead of guessing one.
Settings chosen for print: soft (ambient-occlusion) lighting, silhouettes,
white background.
"""

from __future__ import annotations

from myrmidon.errors import MyrmidonError


def paper() -> list[str]:
    return [
        "set bgColor white",
        "lighting soft",
        "graphics silhouettes true width 1.5",
        "graphics quality 3",
    ]


def flat() -> list[str]:
    """Diagram-like look: flat lighting, strong outlines."""
    return [
        "set bgColor white",
        "lighting flat",
        "graphics silhouettes true width 2",
    ]


def qmmm(qm_selection: str) -> list[str]:
    """QM region as ball-and-stick over a ghosted MM cartoon (MiMiC figures)."""
    return [
        "hide atoms",
        "show cartoon",
        "color lightgray target c",  # ChimeraX rejects X11 'grayNN' names (1.12)
        "transparency 60 target c",
        f"show {qm_selection} atoms",
        f"style {qm_selection} ball",
        f"color {qm_selection} byhetero",
        f"transparency {qm_selection} 0 target a",
    ]


def resolve(name: str, qm_selection: str | None = None) -> list[str]:
    if name == "paper":
        return paper()
    if name == "flat":
        return flat()
    if name == "qmmm":
        if not qm_selection:
            raise MyrmidonError(
                "preset 'qmmm' needs qm_selection (e.g. ':AKG' or '#1:301-320') — "
                "myrmidon never guesses which atoms are the QM region"
            )
        return qmmm(qm_selection)
    raise MyrmidonError(f"unknown preset '{name}' (available: paper, flat, qmmm)")
