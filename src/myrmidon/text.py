"""Small text helpers every part of the harness uses."""

from __future__ import annotations


def tail(text: str, n_lines: int = 30) -> str:
    """The last ``n_lines`` lines of ``text``: what a failure shows of a log."""
    return "\n".join(text.splitlines()[-n_lines:])
