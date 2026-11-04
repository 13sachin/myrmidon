"""Every registered tool. A tool module under this package registers its tools on import."""

from __future__ import annotations

import importlib
import pkgutil


def register_all() -> None:
    """Import every module that defines tools, and give the registry the gate every call
    that books resources passes (idempotent per process)."""
    from myrmidon.core.tool import REGISTRY
    from myrmidon.execution.gate import submission_gate

    REGISTRY.submission_gate = submission_gate
    for info in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
        importlib.import_module(info.name)
