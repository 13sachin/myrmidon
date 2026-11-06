"""Every finding id the shipped code can raise, read from the source.

Input sets, protocols and skills name findings. A name nothing can raise is a pointer to
nowhere, so the tests check each one against this catalogue: ids given to a
Finding (directly or through a `note` helper).
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "myrmidon"


def known_finding_ids() -> set[str]:
    ids: set[str] = set()
    for path in SRC.rglob("*.py"):
        ids |= set(re.findall(r'\b(?:id=f?|note\()"([A-Z][A-Z0-9_]+)"', path.read_text()))
    return ids


def blocking(obs) -> list:
    """The findings that stop a call as made."""
    return [f for f in obs.findings if f.severity == "block"]


def unapproved(obs) -> list:
    """The findings that are a person's to accept, which stopped a model's call."""
    return [f for f in obs.findings if f.severity == "warn"]


def in_group(obs, group: str) -> list:
    """The findings of one group as a refusal tells them apart: `blocking` (the call must
    change), `unapproved` (a person's to accept) or `advisory` (recorded, the call runs)."""
    severity = {"blocking": "block", "unapproved": "warn", "advisory": "info"}[group]
    return [f for f in obs.findings if f.severity == severity]
