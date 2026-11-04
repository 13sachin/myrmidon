"""A documentation query built from an engine version and a question."""

from __future__ import annotations


def release_series(version: str) -> str:
    from myrmidon.knowledge.docs import _major

    return _major(version)


def params_query(query: str) -> str:
    return repr(query if len(query) <= 60 else query[:57] + "…")
