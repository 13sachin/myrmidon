"""Tool profiles: which tools a client is shown for a kind of question (tool_profiles.yaml)."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

from myrmidon.errors import MyrmidonError

PROFILES_FILE = Path(__file__).parent / "tool_profiles.yaml"
#: shows every tool whose engines are present
ALL = "all"


@cache
def profiles() -> dict[str, tuple[str, ...]]:
    data = yaml.safe_load(PROFILES_FILE.read_text()) or {}
    return {name: tuple(tools) for name, tools in data.items()}


def shown(profile: str) -> set[str] | None:
    """The tool names a profile shows — `core` and each `+`-joined profile named — or None
    for every tool. A profile nobody defined is an error naming those that are."""
    if profile == ALL:
        return None
    known = profiles()
    unknown = [p for p in filter(None, profile.split("+")) if p not in known]
    if unknown:
        raise MyrmidonError(f"tool_profile {profile!r}: no profile {', '.join(unknown)}; "
                            f"the profiles are {ALL}, " + ", ".join(sorted(known)))
    names = set(known["core"])
    for part in filter(None, profile.split("+")):
        names |= set(known[part])
    return names
