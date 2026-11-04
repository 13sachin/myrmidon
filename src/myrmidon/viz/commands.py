"""ChimeraX commands as the viewer tools write them: paths quoted, a draft snapshot named,
the view matrix read back, and the notes a command batch left."""

from __future__ import annotations

from myrmidon.viz.session import CommandResult

_NOTE_LIMIT = 2000  # keep observations readable; full log stays in ChimeraX


def cxc_quote(path: str) -> str:
    return f'"{path}"' if " " in path else path


def next_draft(stage) -> str:
    n = len(list(stage.glob("draft_*.png"))) + 1
    return f"draft_{n:03d}.png"


def view_matrix(session) -> str | None:
    result = session.run("view matrix")
    return result.text().strip() or None


def session_notes(results: list[CommandResult]) -> str:
    text = "".join(note for r in results for note in r.notes).strip()
    return text[:_NOTE_LIMIT]
